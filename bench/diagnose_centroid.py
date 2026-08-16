#!/usr/bin/env python3
"""Where does the 0.175 tag-quality gap actually live?

Our miner ranks candidate tags by cosine to an ESTIMATE of the validator's
target vector, rebuilt from the 10-line window plus the enrichment lines. Every
selection decision rests on that estimate. Until now we had no way to check it,
because the real target was unobtainable - the validator builds it from the full
conversation.

The testnet API gives us the full conversation, so now we can. This splits the
gap into two numbers that imply completely different work:

    oracle_selection - current   how much a PERFECT centroid would buy us
                                 -> if large, fix the centroid estimate
    ceiling  - oracle_selection  what the candidate POOL is missing
                                 -> if large, fix tag generation

It also reports cos(estimated centroid, real centroid) directly, which is the
mechanism itself rather than its consequence.

    python bench/diagnose_centroid.py --n 30
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from bench.harness import score_answer
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline, scoring
from sn33.tags import centroid, cosine, rank_by_centroid


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--kind", default="conversation_tagging")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    cases = load_cases(kind=args.kind)
    pairs = expand_windows(cases)[: args.n]
    print(f"{len(cases)} distinct sources -> {len(pairs)} window test-cases "
          f"(paired; topics are limited to those {len(cases)} sources)\n")

    sem = asyncio.Semaphore(args.concurrency)
    cfg = pipeline.Config(use_cache=True, use_local=False, deadline_s=600, call_timeout_s=180)

    async def one(case, window):
        async with sem:
            gt = await real_ground_truth(case)
            if not gt.ok():
                return None
            res = await pipeline.mine(
                case.kind,
                window=window if case.kind == "conversation_tagging" else [(0, case.document)],
                enrichment=case.enrichment_lines,
                cfg=cfg,
            )
            if not res.candidates or not res.vectors:
                return None

            real_target = gt.target
            est_tags = Utils.get_clean_tag_set(res.predicted_gt) if res.predicted_gt else []
            est_target = centroid([res.vectors[t] for t in est_tags if t in res.vectors]) if est_tags else None

            # What we actually submitted.
            actual = await score_answer(gt, res.tags, case.kind, model=GT_MODEL)

            # Same candidate pool, ranked against the REAL target instead.
            # Not deployable - it reads ground truth. It is the diagnostic.
            ranked = rank_by_centroid(res.candidates, res.vectors, real_target)
            oracle_tags = [t for t, _ in ranked[:12]]
            oracle = await score_answer(gt, oracle_tags, case.kind, model=GT_MODEL)

            # Ceiling: score the ground-truth tags themselves, cleaned.
            gt_clean = Utils.get_clean_tag_set(gt.tags)[:12]
            ceiling = await score_answer(gt, gt_clean, case.kind, model=GT_MODEL)

            align = (cosine(np.asarray(est_target), np.asarray(real_target))
                     if est_target is not None else 0.0)
            return {
                "actual": actual.detail.get("adjusted", 0.0),
                "oracle": oracle.detail.get("adjusted", 0.0),
                "ceiling": ceiling.detail.get("adjusted", 0.0),
                "align": align,
                "n_cand": len(res.candidates),
            }

    out = [r for r in await asyncio.gather(*[one(c, w) for c, w in pairs]) if r]
    if not out:
        raise SystemExit("no usable cases")

    m = lambda k: statistics.mean(r[k] for r in out)
    print(f"n={len(out)}  candidates/case={statistics.mean(r['n_cand'] for r in out):.0f}\n")
    print(f"  cos(estimated centroid, REAL centroid) = {m('align'):.4f}")
    print(f"     min={min(r['align'] for r in out):.4f}  max={max(r['align'] for r in out):.4f}\n")
    print(f"  adjusted, what we submit          {m('actual'):.4f}")
    print(f"  adjusted, same pool + REAL target {m('oracle'):.4f}")
    print(f"  adjusted, ground-truth tags       {m('ceiling'):.4f}\n")
    print(f"  -> better centroid is worth        {m('oracle') - m('actual'):+.4f}")
    print(f"  -> better candidates are worth     {m('ceiling') - m('oracle'):+.4f}")

    align = [r["align"] for r in out]
    gain = [r["oracle"] - r["actual"] for r in out]
    if len(out) > 3:
        r = np.corrcoef(align, gain)[0, 1]
        print(f"\n  corr(centroid alignment, oracle gain) = {r:+.3f}"
              f"   (negative = worse alignment costs more, as expected)")


if __name__ == "__main__":
    asyncio.run(main())
