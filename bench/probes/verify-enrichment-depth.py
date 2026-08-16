#!/usr/bin/env python3
"""ADVERSARIAL VERIFICATION of the `enrichment-depth` probe.

The original probe compared

    armA = the miner's candidate pool
    armB = that pool + a NEW, deeper (30-tag) per-enrichment-line extraction

and concluded that "deeper enrichment extraction" is worth +0.0236 adjusted.

That comparison conflates TWO changes, because of what `sn33/replica.py` does:

    rep.enrichment_tags = [...]     # per-line tags, upstream 10-tag depth
    rep.tags = <LLM combine of doc_tags + enrichment_tags>
    pipeline.mine: predicted_gt = list(rep.tags)      <-- ONLY the combined set

The per-line enrichment tags are already generated, already paid for, and then
DISCARDED. `mined.candidates` never contains them. So armB changes:

    (1) per-line tags reach the candidate pool at all   [FREE - already paid]
    (2) the per-line budget goes 10 -> 30               [EXTRA CALL PER LINE]

This script separates them.

    armA   pool                                (baseline, unchanged)
    armE   pool + union(per-line @ upstream depth 10)   -> isolates (1), FREE
    armB   pool + union(per-line @ depth 30)            -> (1)+(2), original claim
    armBE  pool + both                                  -> sanity

If armE captures the gain, the finding is real but MISATTRIBUTED: the lever is
"stop discarding the per-line enrichment tags", not "extract deeper", and it
costs nothing in latency or tokens.

It also re-runs the whole thing under the PRODUCTION config (use_local=True,
spaCy on) rather than the original probe's use_local=False, in case armA was a
straw man missing its local-extraction candidates.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import math
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import GT_MODEL, load_cases, pick_window, real_ground_truth
from bench.harness import score_answer
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline, prompts
from sn33.replica import clean_like_validator
from sn33.tags import centroid, drop_near_duplicates, normalize_all, rank_by_centroid

# REUSE the original probe's salts so these are cache hits, not new spend.
SALT = "edepth"


def deep_enrichment(line: str, n: int = 30) -> str:
    base = prompts.gt_enrichment(line)
    old = "5.  **Limit to most important:** Return at most 10 of the most important and relevant tags."
    new = (
        f"5.  **Be exhaustive:** Return at most {n} tags. Start with the most "
        "important and relevant, then continue with broader themes, alternative "
        "phrasings of the same concepts, adjacent subtopics, and the field or "
        "industry the content belongs to."
    )
    assert old in base
    return base.replace(old, new)


async def _tags(prompt: str, salt: str, temperature):
    raw = await llm.chat(prompt, model=GT_MODEL, timeout=180, temperature=temperature,
                         use_cache=True, salt=f"{SALT}:{salt}")
    return normalize_all(clean_like_validator(raw))


def union(per_line):
    seen, acc = set(), []
    for tags in per_line:
        for t in tags:
            if t not in seen:
                seen.add(t)
                acc.append(t)
    return acc


def compose_from(pool, mined, vecs, cfg, profile, target_tags, insurance):
    """Identical to the original probe's selection path - copied deliberately."""
    predicted_gt = list(mined.predicted_gt)
    gt_for_target = Utils.get_clean_tag_set(predicted_gt) if predicted_gt else []
    est = centroid([vecs[t] for t in gt_for_target if t in vecs]) if gt_for_target else None
    if est is None:
        return []
    ranked = rank_by_centroid(normalize_all(pool), vecs, est)
    protected = set(normalize_all(predicted_gt))
    ranked = drop_near_duplicates(ranked, vecs, cfg.dedup_threshold, protected=protected)
    if not ranked:
        return []
    return pipeline.compose(ranked, predicted_gt, profile, target_tags, insurance)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--use-local", type=int, default=1, help="1 = production config (spaCy on)")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="shift the seed block; 6 gives an out-of-sample replication")
    args = ap.parse_args()

    seeds = [(i + args.seed_offset) * 1000 for i in range(args.seeds)]
    units = []
    for si, s in enumerate(seeds):
        for ci, case in enumerate(load_cases(kind="conversation_tagging", seed=s)):
            units.append((s, case, pick_window(case, seed=s + ci * 7 + si)))

    sem = asyncio.Semaphore(args.concurrency)
    cfg = pipeline.Config(use_cache=True, use_local=bool(args.use_local),
                          deadline_s=600, call_timeout_s=180)
    print(f"{len(units)} units, use_local={bool(args.use_local)}\n", flush=True)

    async def one(seed, case, window):
        async with sem:
            gt = await real_ground_truth(case)
            if not gt.ok() or not case.enrichment_lines:
                return None
            lines = list(case.enrichment_lines)

            up, deep = await asyncio.gather(
                asyncio.gather(*[_tags(prompts.gt_enrichment(l), "x1t0", 0.0) for l in lines]),
                asyncio.gather(*[_tags(deep_enrichment(l), "deep30", 0.0) for l in lines]),
            )
            f_up, f_deep = union(up), union(deep)

            mined = await pipeline.mine(case.kind, window=window,
                                        enrichment=case.enrichment_lines, cfg=cfg)
            pool = list(mined.candidates)
            pools = {
                "armA_pool":            pool,
                "armE_pool_plus_upline": list(dict.fromkeys(pool + f_up)),
                "armB_pool_plus_deep30": list(dict.fromkeys(pool + f_deep)),
                "armBE_pool_plus_both":  list(dict.fromkeys(pool + f_up + f_deep)),
            }
            everything = sorted({t for v in pools.values() for t in v})
            vecs = await llm.embed(everything, use_cache=True)
            vecs.update(mined.vectors or {})

            profile = pipeline.TASK_PROFILE[case.kind]
            tt, ins = profile["target_tags"], profile["insurance"]
            out = {}
            for name, p in pools.items():
                tags = compose_from(p, mined, vecs, cfg, profile, tt, ins)
                v = await score_answer(gt, tags, case.kind, model=GT_MODEL)
                out[name] = {"adjusted": v.adjusted, "final": v.final,
                             "n_unique": v.n_unique, "n_both": v.n_both,
                             "n_survived": v.n_survived, "n_submitted": len(tags),
                             "penalties": v.penalties, "tags": tags}
            v = await score_answer(gt, list(mined.tags), case.kind, model=GT_MODEL)
            out["shipped"] = {"adjusted": v.adjusted, "final": v.final,
                              "n_unique": v.n_unique, "n_both": v.n_both,
                              "n_survived": v.n_survived, "n_submitted": len(mined.tags),
                              "penalties": v.penalties, "tags": list(mined.tags)}
            return {"seed": seed, "guid": case.guid, "n_lines": len(lines),
                    "n_up": len(f_up), "n_deep": len(f_deep), "e2e": out,
                    "pool_size": len(pool)}

    R = [r for r in await asyncio.gather(*[one(s, c, w) for s, c, w in units]) if r]
    print(f"n = {len(R)} units;  pool {statistics.mean(r['pool_size'] for r in R):.0f} tags, "
          f"per-line union upstream {statistics.mean(r['n_up'] for r in R):.0f}, "
          f"deep30 {statistics.mean(r['n_deep'] for r in R):.0f}\n")

    ARMS = ["shipped", "armA_pool", "armE_pool_plus_upline",
            "armB_pool_plus_deep30", "armBE_pool_plus_both"]
    print(f"{'arm':24s} {'adjusted':>9s} {'final':>8s} {'uniq':>5s} {'both':>5s} "
          f"{'survd':>6s} {'pen%':>5s}")
    print("-" * 68)
    for a in ARMS:
        rows = [r["e2e"][a] for r in R]
        print(f"{a:24s} {statistics.mean(x['adjusted'] for x in rows):9.4f} "
              f"{statistics.mean(x['final'] for x in rows):8.4f} "
              f"{statistics.mean(x['n_unique'] for x in rows):5.1f} "
              f"{statistics.mean(x['n_both'] for x in rows):5.1f} "
              f"{statistics.mean(x['n_survived'] for x in rows):6.2f} "
              f"{sum(1 for x in rows if x['penalties'])/len(rows):5.0%}")

    by = collections.defaultdict(list)
    for r in R:
        by[r["guid"]].append(r)

    def report(base, arm):
        d = [r["e2e"][arm]["adjusted"] - r["e2e"][base]["adjusted"] for r in R]
        pc = []
        for g in sorted(by):
            pc.append(statistics.mean(x["e2e"][arm]["adjusted"] - x["e2e"][base]["adjusted"]
                                      for x in by[g]))
        m, sd, n = statistics.mean(pc), statistics.stdev(pc), len(pc)
        t = m / (sd / math.sqrt(n)) if sd else float("inf")
        half = 3.182 * sd / math.sqrt(n)
        print(f"  {arm:24s} vs {base:22s} pooled {statistics.mean(d):+.4f} "
              f"W/L {sum(1 for x in d if x>1e-9):2d}/{sum(1 for x in d if x<-1e-9):<2d}  |  "
              f"per-conv {m:+.4f} sd {sd:.4f} t {t:5.2f} ci95 [{m-half:+.4f},{m+half:+.4f}]  "
              f"[{' '.join(f'{x:+.4f}' for x in pc)}]")

    print("\nPAIRED DELTAS  (per-conv = the honest n=4 unit, t crit 3.182)")
    for a in ARMS[2:]:
        report("armA_pool", a)
    print()
    report("armE_pool_plus_upline", "armB_pool_plus_deep30")
    report("armE_pool_plus_upline", "armBE_pool_plus_both")

    print("\nHOW MUCH OF THE armB GAIN IS THE FREE armE PART?")
    dE = statistics.mean(r["e2e"]["armE_pool_plus_upline"]["adjusted"]
                         - r["e2e"]["armA_pool"]["adjusted"] for r in R)
    dB = statistics.mean(r["e2e"]["armB_pool_plus_deep30"]["adjusted"]
                         - r["e2e"]["armA_pool"]["adjusted"] for r in R)
    print(f"  armE (free)  {dE:+.4f}   armB (extra call/line) {dB:+.4f}   "
          f"free share {dE/dB:.0%}" if dB else "")

    print("\nper conversation, adjusted")
    print(f"  {'guid':12s} " + " ".join(f"{a.split('_')[0]:>9s}" for a in ARMS))
    for g in sorted(by):
        print(f"  {g:12s} " + " ".join(
            f"{statistics.mean(x['e2e'][a]['adjusted'] for x in by[g]):9.4f}" for a in ARMS))

    print("\nusage:", llm.Usage.snapshot())


if __name__ == "__main__":
    asyncio.run(main())
