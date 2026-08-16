#!/usr/bin/env python3
"""Probe: does a tag's ABSTRACTION LEVEL predict its cosine to the real centroid?

Hypothesis
----------
Winning tags may sit at a particular rung of an abstraction ladder. If broad
tags ("finance") beat narrow ones ("chapter 13 bankruptcy filing") - or the
reverse - then "generate broader / narrower" is a knob we can turn in the pool
prompt.

Design
------
The confound in a naive test is topic: a broad tag and a specific tag are
usually about *different things*, so any difference could be topic choice, not
abstraction. So this probe asks the model for LADDERS: for one concept, five
tags naming that SAME concept at five levels, L1 (hyper-specific) .. L5 (very
broad). Every rung comparison is therefore paired *within a concept*, and
abstraction is the only variable that moves.

Two context conditions are run, because the miner's pool prompt sees only the
10-line window while the replica also sees the enrichment lines, and enrichment
is already known to be worth +0.257:

    ctx=window      the 10 lines only          (like-for-like with the pool prompt)
    ctx=enriched    the 10 lines + enrichment  (everything the miner holds)

Reference families measured on the same windows and the same real ground truth:
GROUND_TRUTH (the validator's own tags), gt_variants, and the CURRENT MINER's
actual candidate pool + its actual 12-tag answer.

Run:
    python bench/probes/abstraction-ladder.py --per-convo 8
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline
from sn33.tags import cosine, normalize, normalize_all
from sn33.variants import expand as expand_variants

SALT = "abstladder"        # our own cache namespace, per the probe rules
RUNGS = ["L1", "L2", "L3", "L4", "L5"]

LADDER_PROMPT = """You will be given material from a conversation.

Identify the {k} most important concepts in it. For EACH concept, write a ladder
of five tags that all name that SAME concept at five increasing levels of
abstraction:

L1 = hyper-specific: the exact named thing, as concrete as the material allows
L2 = specific: the particular topic it is an instance of
L3 = mid-level: the general subject being discussed
L4 = broad: the subfield it belongs to
L5 = very broad: the top-level domain, one or two words

Worked example of one ladder:
chapter 13 bankruptcy filing | personal bankruptcy | consumer debt relief | personal finance | finance

Rules for every tag:
- English, lowercase, letters digits and single spaces only, 1 to 4 words
- the five tags on a line must be the same concept, only wider each step
- do not repeat a tag within a line

Output exactly {k} lines. Each line is five tags separated by " | ", L1 first.
No numbering, no headings, no commentary.

Material:
{content}
"""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def window_text(window: Sequence) -> str:
    return "\n".join(str(l[1]) for l in window if len(l) >= 2)


def parse_ladders(raw: Optional[str]) -> List[List[str]]:
    """Rows of exactly 5 normalized tags. A row with any unusable rung is dropped
    so that every rung comparison stays perfectly paired."""
    out: List[List[str]] = []
    if not raw:
        return out
    for line in raw.splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 5:
            continue
        norm = [normalize(p) for p in parts]
        if any(n is None for n in norm):
            continue
        if len(set(norm)) < 2:          # a degenerate ladder that never widened
            continue
        out.append([str(n) for n in norm])
    return out


def fam_stats(tags: Sequence[str], vecs: Dict[str, Sequence[float]], target: np.ndarray,
              gt_clean: set) -> Optional[dict]:
    cs = [(t, cosine(target, np.asarray(vecs[t], dtype=np.float32))) for t in tags if t in vecs]
    if not cs:
        return None
    vals = [c for _, c in cs]
    uniq = [c for t, c in cs if t not in gt_clean]
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "top3": statistics.mean(sorted(vals)[-3:]) if len(vals) >= 3 else statistics.mean(vals + [0.0] * (3 - len(vals))),
        "top3_unique": (statistics.mean(sorted(uniq)[-3:]) if len(uniq) >= 3
                        else statistics.mean(uniq + [0.0] * (3 - len(uniq)))) if uniq else 0.0,
        "best": max(vals),
        "collide": sum(1 for t, _ in cs if t in gt_clean) / len(cs),
        "words": statistics.mean([len(t.split()) for t, _ in cs]),
        "chars": statistics.mean([len(t) for t, _ in cs]),
    }


def agg(rows: Sequence[dict], key: str) -> float:
    vals = [r[key] for r in rows if r]
    return statistics.mean(vals) if vals else float("nan")


# --------------------------------------------------------------------------

async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-convo", type=int, default=8)
    ap.add_argument("--k", type=int, default=8, help="ladders (concepts) per window")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--no-miner", action="store_true", help="skip running the real miner pipeline")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases, per_convo=args.per_convo, seed=0)
    print(f"{len(pairs)} windows from {len(cases)} conversations "
          f"({collections.Counter(id(c) for c, _ in pairs).most_common()})\n", flush=True)

    gt_cache: Dict[str, object] = {}
    gt_lock = asyncio.Lock()

    async def gt_for(case):
        async with gt_lock:
            if case.guid not in gt_cache:
                gt_cache[case.guid] = await real_ground_truth(case)
            return gt_cache[case.guid]

    sem = asyncio.Semaphore(args.concurrency)
    convo_id = {c.guid: f"C{i+1}" for i, c in enumerate(cases)}

    miner_cfg = pipeline.Config(
        deadline_s=600.0, call_timeout_s=300.0, use_cache=True,
    )

    async def one(idx, case, window):
        async with sem:
            gt = await gt_for(case)
            if not gt.ok():
                return None
            gt_clean = set(normalize_all(Utils.get_clean_tag_set(gt.tags)))
            wtext = window_text(window)
            enr = "\n\n".join(case.enrichment_lines)

            ctx = {
                "window": wtext[:6000],
                "enriched": (wtext + "\n\nBackground search results:\n" + enr)[:6000],
            }

            raws = await asyncio.gather(*[
                llm.chat(LADDER_PROMPT.format(k=args.k, content=v), model=GT_MODEL,
                         timeout=300, use_cache=True, salt=f"{SALT}-{name}")
                for name, v in ctx.items()
            ])
            ladders = {name: parse_ladders(r) for name, r in zip(ctx, raws)}

            miner_pool: List[str] = []
            miner_answer: List[str] = []
            if not args.no_miner:
                mres = await pipeline.mine(
                    "conversation_tagging", window=list(window),
                    enrichment=case.enrichment_lines, cfg=miner_cfg,
                )
                miner_pool = list(mres.candidates)
                miner_answer = list(mres.tags)

            fams: Dict[str, List[str]] = {}
            for name, rows in ladders.items():
                for r, rung in enumerate(RUNGS):
                    fams[f"{name}:{rung}"] = [row[r] for row in rows]
                fams[f"{name}:ALL"] = [t for row in rows for t in row]
            fams["GROUND_TRUTH"] = sorted(gt_clean)
            fams["gt_variants"] = expand_variants(sorted(gt_clean), per_tag=2)
            if miner_pool:
                mp = normalize_all(miner_pool)
                fams["MINER_pool"] = mp
                # size-matched control: `top3` grows with the number of draws, and
                # the miner pool is ~139 tags against the ladder's 40. Sample 40.
                import random as _r
                fams["MINER_pool_40"] = _r.Random(idx).sample(mp, min(40, len(mp)))
            if miner_answer:
                fams["MINER_answer"] = normalize_all(miner_answer)

            everything = sorted({t for v in fams.values() for t in v})
            vecs = await llm.embed(everything, use_cache=True)
            target = np.asarray(gt.target, dtype=np.float32)

            stats = {f: fam_stats(t, vecs, target, gt_clean) for f, t in fams.items()}

            # within-ladder argmax: for one concept, which rung is closest?
            winners = {name: [] for name in ctx}
            for name, rows in ladders.items():
                for row in rows:
                    cs = [cosine(target, np.asarray(vecs[t], dtype=np.float32)) if t in vecs else -1.0
                          for t in row]
                    winners[name].append(int(np.argmax(cs)))

            # Decisive test of usefulness: merge the ladder tags into the miner's
            # own pool, rank the union against the REAL centroid, and count how
            # many of the top 12 are ladder tags. If none, the family cannot
            # change the answer even under a perfect selector.
            displace = {}
            for name in ctx:
                lad = [t for row in ladders[name] for t in row]
                union = list(dict.fromkeys(normalize_all(miner_pool) + lad))
                ranked = sorted(
                    ((cosine(target, np.asarray(vecs[t], dtype=np.float32)), t)
                     for t in union if t in vecs), reverse=True)
                lset = set(lad)
                displace[name] = sum(1 for _, t in ranked[:12] if t in lset)

            print(f"  [{idx:>2}] {convo_id[case.guid]} ladders="
                  f"{ {n: len(r) for n, r in ladders.items()} } "
                  f"gt={len(gt_clean)} pool={len(miner_pool)}", flush=True)
            return {
                "convo": convo_id[case.guid],
                "stats": stats,
                "winners": winners,
                "displace": displace,
                "examples": {n: rows[:3] for n, rows in ladders.items()},
                "gt_tags": sorted(gt_clean)[:12],
                "miner_answer": miner_answer,
            }

    results = await asyncio.gather(*[one(i, c, w) for i, (c, w) in enumerate(pairs)])
    results = [r for r in results if r]
    n = len(results)
    print(f"\nusable windows: n={n}\n")
    if not n:
        return

    convos = sorted({r["convo"] for r in results})
    order = ([f"{c}:{r}" for c in ("window", "enriched") for r in RUNGS + ["ALL"]]
             + ["MINER_pool", "MINER_pool_40", "MINER_answer", "gt_variants", "GROUND_TRUTH"])

    # ---------------- pooled table ----------------
    print("=" * 96)
    print(f"POOLED cosine to the REAL centroid, n={n} windows, {len(convos)} conversations")
    print("=" * 96)
    print(f"{'family':<22}{'tags':>6}{'mean':>9}{'top3':>9}{'top3uniq':>10}{'best':>9}"
          f"{'gt-hit':>8}{'words':>7}{'chars':>7}")
    for f in order:
        rows = [r["stats"].get(f) for r in results if r["stats"].get(f)]
        if not rows:
            continue
        print(f"{f:<22}{agg(rows,'n'):>6.1f}{agg(rows,'mean'):>9.4f}{agg(rows,'top3'):>9.4f}"
              f"{agg(rows,'top3_unique'):>10.4f}{agg(rows,'best'):>9.4f}"
              f"{agg(rows,'collide')*100:>7.1f}%{agg(rows,'words'):>7.2f}{agg(rows,'chars'):>7.1f}")

    # ---------------- per conversation ----------------
    for metric in ("top3", "mean"):
        print("\n" + "=" * 96)
        print(f"PER-CONVERSATION {metric} (the corpus is 4 conversations, not {n} trials)")
        print("=" * 96)
        hdr = "".join(f"{c:>10}" for c in convos)
        print(f"{'family':<22}{hdr}{'pooled':>10}")
        for f in order:
            cells = ""
            for c in convos:
                rows = [r["stats"].get(f) for r in results if r["convo"] == c and r["stats"].get(f)]
                cells += f"{agg(rows, metric):>10.4f}" if rows else f"{'-':>10}"
            rows = [r["stats"].get(f) for r in results if r["stats"].get(f)]
            print(f"{f:<22}{cells}{agg(rows, metric):>10.4f}")

    # ---------------- paired within-ladder argmax ----------------
    print("\n" + "=" * 96)
    print("PAIRED: for one concept, which rung sits closest to the centroid?")
    print("=" * 96)
    for name in ("window", "enriched"):
        tot = collections.Counter()
        per_convo = {c: collections.Counter() for c in convos}
        for r in results:
            for w in r["winners"].get(name, []):
                tot[w] += 1
                per_convo[r["convo"]][w] += 1
        total = sum(tot.values())
        if not total:
            continue
        print(f"\nctx={name}  ladders={total}")
        print(f"{'rung':<8}{'wins':>8}{'share':>8}" + "".join(f"{c:>10}" for c in convos))
        for i, rung in enumerate(RUNGS):
            cells = "".join(
                f"{(per_convo[c][i]/max(1,sum(per_convo[c].values()))*100):>9.1f}%" for c in convos)
            print(f"{rung:<8}{tot[i]:>8}{tot[i]/total*100:>7.1f}%{cells}")

    # ---------------- paired rung deltas vs the best rung ----------------
    print("\n" + "=" * 96)
    print("PAIRED rung deltas on top3 (same window, same GT) - vs the best rung")
    print("=" * 96)
    for name in ("window", "enriched"):
        keys = [f"{name}:{r}" for r in RUNGS]
        means = {k: agg([r["stats"].get(k) for r in results if r["stats"].get(k)], "top3") for k in keys}
        best = max(means, key=lambda k: means[k])
        print(f"\nctx={name}  best rung = {best}  top3={means[best]:.4f}")
        for k in keys:
            diffs = [r["stats"][k]["top3"] - r["stats"][best]["top3"]
                     for r in results if r["stats"].get(k) and r["stats"].get(best)]
            if not diffs:
                continue
            wins = sum(1 for d in diffs if d > 0)
            print(f"  {k:<18} top3={means[k]:.4f}  delta={statistics.mean(diffs):+.4f}  "
                  f"W/L {wins}/{sum(1 for d in diffs if d < 0)} of {len(diffs)}")

    # ---------------- displacement ----------------
    print("\n" + "=" * 96)
    print("USEFULNESS: ladder tags in the top 12 of (miner pool + ladder) ranked by the REAL centroid")
    print("=" * 96)
    for name in ("window", "enriched"):
        vals = [r["displace"].get(name, 0) for r in results if r.get("displace")]
        if not vals:
            continue
        print(f"  ctx={name:<10} mean {statistics.mean(vals):.2f} of 12 slots   "
              f"windows with >=1 ladder tag: {sum(1 for v in vals if v)}/{len(vals)}   "
              f"per-convo: " + "  ".join(
                  f"{c}={statistics.mean([r['displace'][name] for r in results if r['convo']==c]):.2f}"
                  for c in convos))

    # ---------------- examples ----------------
    print("\n" + "=" * 96)
    print("EXAMPLE LADDERS (ctx=enriched) with per-rung cosine")
    print("=" * 96)
    for r in results[:3]:
        print(f"\n{r['convo']}  gt sample: {', '.join(r['gt_tags'][:8])}")
        print(f"  miner answer: {', '.join(r['miner_answer'][:12])}")
        for row in r["examples"].get("enriched", [])[:3]:
            print("   " + "  ->  ".join(row))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, default=str)
        print(f"\nwrote {args.out}")

    print("\nusage:", llm.Usage.snapshot())


if __name__ == "__main__":
    asyncio.run(main())
