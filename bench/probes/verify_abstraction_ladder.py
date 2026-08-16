#!/usr/bin/env python3
"""ADVERSARIAL VERIFICATION of the `abstraction-ladder` probe.

The original probe measured only COSINE TO THE CENTROID. That is not the score.
This script runs the same ladders through the REAL validator path
(`bench.harness.score_answer`: the 20-cull, the "good English keywords" LLM
screen, re-embedding, and GroundTruthTagSimilarityScoringMechanism), so the
question becomes "does adjusted go up" rather than "is the cosine higher".

Answers scored, all on identical windows and identical real ground truth:

  miner            the miner's real 12-tag answer                    (baseline)
  miner_plus7      miner 12 + the 7 best ladder tags, chosen with an
                   ORACLE (ranked against the REAL centroid) -> 19    (best case for the family)
  oracle_pool12    top-12 of the miner's own pool, oracle-ranked      (ceiling without ladders)
  oracle_union12   top-12 of (miner pool + ladders), oracle-ranked    (ceiling with ladders)
  ladder12         top-12 ladder tags alone, oracle-ranked           (family standalone)

Cache salt is shared with the original probe on purpose for the ladder CHAT
calls (identical prompt text -> free reuse); scoring calls use the harness's
own "validate" salt.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import statistics
import sys
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from bench.harness import score_answer
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline
from sn33.tags import cosine, normalize, normalize_all

SALT = "abstladder"
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


def window_text(window: Sequence) -> str:
    return "\n".join(str(l[1]) for l in window if len(l) >= 2)


def parse_ladders(raw):
    out = []
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
        if len(set(norm)) < 2:
            continue
        out.append([str(n) for n in norm])
    return out


def oracle_rank(tags: Sequence[str], vecs: Dict[str, Sequence[float]], target: np.ndarray) -> List[str]:
    scored = [(cosine(target, np.asarray(vecs[t], dtype=np.float32)), t) for t in tags if t in vecs]
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-convo", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases, per_convo=args.per_convo, seed=0)
    print(f"{len(pairs)} windows from {len(cases)} conversations\n", flush=True)

    gt_cache: Dict[str, object] = {}
    gt_lock = asyncio.Lock()

    async def gt_for(case):
        async with gt_lock:
            if case.guid not in gt_cache:
                gt_cache[case.guid] = await real_ground_truth(case)
            return gt_cache[case.guid]

    sem = asyncio.Semaphore(args.concurrency)
    convo_id = {c.guid: f"C{i+1}" for i, c in enumerate(cases)}
    miner_cfg = pipeline.Config(deadline_s=600.0, call_timeout_s=300.0, use_cache=True)

    async def one(idx, case, window):
        async with sem:
            gt = await gt_for(case)
            if not gt.ok():
                return None
            gt_clean = set(normalize_all(Utils.get_clean_tag_set(gt.tags)))
            wtext = window_text(window)
            enr = "\n\n".join(case.enrichment_lines)
            content = (wtext + "\n\nBackground search results:\n" + enr)[:6000]

            raw = await llm.chat(LADDER_PROMPT.format(k=args.k, content=content), model=GT_MODEL,
                                 timeout=300, use_cache=True, salt=f"{SALT}-enriched")
            ladders = parse_ladders(raw)
            lad = list(dict.fromkeys(t for row in ladders for t in row))

            mres = await pipeline.mine("conversation_tagging", window=list(window),
                                       enrichment=case.enrichment_lines, cfg=miner_cfg)
            miner_answer = normalize_all(list(mres.tags))
            miner_pool = normalize_all(list(mres.candidates))

            everything = sorted(set(lad) | set(miner_answer) | set(miner_pool))
            vecs = await llm.embed(everything, use_cache=True)
            target = np.asarray(gt.target, dtype=np.float32)

            lad_ranked = oracle_rank(lad, vecs, target)
            pool_ranked = oracle_rank(list(dict.fromkeys(miner_pool)), vecs, target)
            union_ranked = oracle_rank(list(dict.fromkeys(miner_pool + lad)), vecs, target)

            answers = {
                "miner": miner_answer,
                "miner_plus7": list(dict.fromkeys(list(miner_answer) + lad_ranked[:7]))[:19],
                "oracle_pool12": pool_ranked[:12],
                "oracle_union12": union_ranked[:12],
                "ladder12": lad_ranked[:12],
            }
            verdicts = {}
            for name, tags in answers.items():
                v = await score_answer(gt, tags, "conversation_tagging", model=GT_MODEL,
                                       seed=idx, use_cache=True)
                verdicts[name] = {
                    "adjusted": v.adjusted, "final": v.final, "pen": v.penalties,
                    "n_sub": v.n_submitted, "n_surv": v.n_survived,
                    "n_uniq": v.n_unique, "n_both": v.n_both,
                    "top3": v.detail.get("top_3_mean"), "mean": v.detail.get("mean_score"),
                    "median": v.detail.get("median_score"),
                }
            n_lad_in_union = sum(1 for t in union_ranked[:12] if t in set(lad))
            # English-screen survival for ladder tags specifically
            print(f"  [{idx:>2}] {convo_id[case.guid]} "
                  f"miner adj={verdicts['miner']['adjusted']:.4f} "
                  f"plus7 adj={verdicts['miner_plus7']['adjusted']:.4f} "
                  f"ladder12 adj={verdicts['ladder12']['adjusted']:.4f} "
                  f"lad_in_union12={n_lad_in_union}", flush=True)
            return {"convo": convo_id[case.guid], "v": verdicts, "lad_in_union": n_lad_in_union,
                    "n_lad": len(lad), "ladder12": answers["ladder12"],
                    "lad_surv": verdicts["ladder12"]["n_surv"]}

    results = [r for r in await asyncio.gather(*[one(i, c, w) for i, (c, w) in enumerate(pairs)]) if r]
    n = len(results)
    convos = sorted({r["convo"] for r in results})
    print(f"\nusable windows n={n}  {collections.Counter(r['convo'] for r in results)}\n")

    order = ["miner", "miner_plus7", "oracle_pool12", "oracle_union12", "ladder12"]
    print("=" * 100)
    print("REAL VALIDATOR SCORE (score_answer: 20-cull + English screen + penalties)")
    print("=" * 100)
    print(f"{'answer':<18}{'adjusted':>10}{'final':>9}{'top3':>9}{'mean':>9}{'median':>9}"
          f"{'sub':>5}{'surv':>6}{'uniq':>6}{'both':>6}")
    for name in order:
        rows = [r["v"][name] for r in results]
        def m(k):
            return statistics.mean([x[k] for x in rows if x[k] is not None])
        print(f"{name:<18}{m('adjusted'):>10.4f}{m('final'):>9.4f}{m('top3'):>9.4f}"
              f"{m('mean'):>9.4f}{m('median'):>9.4f}{m('n_sub'):>5.1f}{m('n_surv'):>6.1f}"
              f"{m('n_uniq'):>6.1f}{m('n_both'):>6.1f}")

    print("\n" + "=" * 100)
    print("PAIRED DELTAS vs the miner's real answer (adjusted), per conversation")
    print("=" * 100)
    for name in order[1:]:
        cells = []
        for c in convos:
            ds = [r["v"][name]["adjusted"] - r["v"]["miner"]["adjusted"] for r in results if r["convo"] == c]
            cells.append(f"{c} {statistics.mean(ds):+.4f} ({sum(1 for d in ds if d>0)}/{sum(1 for d in ds if d<0)})")
        ds = [r["v"][name]["adjusted"] - r["v"]["miner"]["adjusted"] for r in results]
        print(f"  {name:<16} POOLED {statistics.mean(ds):+.4f}  W/L "
              f"{sum(1 for d in ds if d>0)}/{sum(1 for d in ds if d<0)} of {len(ds)}   "
              + "  ".join(cells))

    print("\n" + "=" * 100)
    print("KEY CONTRAST: oracle_union12 - oracle_pool12  (does the family raise the CEILING?)")
    print("=" * 100)
    ds = [r["v"]["oracle_union12"]["adjusted"] - r["v"]["oracle_pool12"]["adjusted"] for r in results]
    print(f"  POOLED {statistics.mean(ds):+.4f}  W/L {sum(1 for d in ds if d>0)}/{sum(1 for d in ds if d<0)} of {len(ds)}")
    for c in convos:
        dd = [r["v"]["oracle_union12"]["adjusted"] - r["v"]["oracle_pool12"]["adjusted"]
              for r in results if r["convo"] == c]
        print(f"    {c} n={len(dd)} {statistics.mean(dd):+.4f}  W/L {sum(1 for d in dd if d>0)}/{sum(1 for d in dd if d<0)}")
    print(f"  ladder tags occupying an oracle top-12 slot: mean "
          f"{statistics.mean([r['lad_in_union'] for r in results]):.2f} of 12")

    print("\n" + "=" * 100)
    print("ENGLISH SCREEN + CLEANING survival of ladder-only answers")
    print("=" * 100)
    subs = [r["v"]["ladder12"]["n_sub"] for r in results]
    survs = [r["v"]["ladder12"]["n_surv"] for r in results]
    print(f"  submitted {statistics.mean(subs):.2f} -> survived {statistics.mean(survs):.2f} "
          f"({statistics.mean(survs)/statistics.mean(subs)*100:.1f}%)   "
          f"windows below min_tags=3: {sum(1 for s in survs if s < 3)}/{n}")
    msubs = [r["v"]["miner"]["n_sub"] for r in results]
    msurvs = [r["v"]["miner"]["n_surv"] for r in results]
    print(f"  miner baseline: submitted {statistics.mean(msubs):.2f} -> survived {statistics.mean(msurvs):.2f} "
          f"({statistics.mean(msurvs)/statistics.mean(msubs)*100:.1f}%)")

    print("\n" + "=" * 100)
    print("PENALTY FIRING RATES")
    print("=" * 100)
    for name in order:
        cnt = collections.Counter(p for r in results for p in r["v"][name]["pen"])
        print(f"  {name:<18}{dict(cnt) if cnt else 'none'}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, default=str)
        print(f"\nwrote {args.out}")
    print("\nusage:", llm.Usage.snapshot())


if __name__ == "__main__":
    asyncio.run(main())
