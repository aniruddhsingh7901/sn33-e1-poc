#!/usr/bin/env python3
"""How much of our answer comes from the window, and how much from enrichment?

Four different windows of the same 300-line conversation produced 10/12
identical tags. The enrichment lines are byte-identical across every window of a
conversation, so the suspicion is that they, not the window, decide our output.

That would matter two ways:
  * it explains why a paired A/B over 40 windows has sd 0.017 - the windows are
    not really 40 independent trials
  * if enrichment is doing all the work, window-side effort is wasted; if the
    window is doing the work, enrichment is a free win we may be underusing

So measure it: run each window twice, once with enrichment and once without,
and compare both the tag overlap and the score.

    python bench/diagnose_enrichment.py --n 24
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from bench.harness import score_answer
from sn33 import pipeline


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases)[: args.n]
    sem = asyncio.Semaphore(args.concurrency)
    cfg = pipeline.Config(use_cache=True, use_local=False, deadline_s=600, call_timeout_s=180)

    print(f"{len(pairs)} windows from {len(cases)} conversations\n")

    async def one(case, window):
        async with sem:
            gt = await real_ground_truth(case)
            if not gt.ok():
                return None
            both = await pipeline.mine(case.kind, window=window,
                                       enrichment=case.enrichment_lines, cfg=cfg)
            win_only = await pipeline.mine(case.kind, window=window,
                                           enrichment=[], cfg=cfg)
            s_both = await score_answer(gt, both.tags, case.kind, model=GT_MODEL)
            s_win = await score_answer(gt, win_only.tags, case.kind, model=GT_MODEL)
            a, b = set(both.tags), set(win_only.tags)
            return {
                "guid": case.guid,
                "both": s_both.detail.get("adjusted", 0.0),
                "win_only": s_win.detail.get("adjusted", 0.0),
                # of what we ship, how much survives when enrichment is removed?
                "kept": len(a & b) / max(1, len(a)),
                "tags_both": both.tags,
            }

    out = [r for r in await asyncio.gather(*[one(c, w) for c, w in pairs]) if r]
    if not out:
        raise SystemExit("no usable cases")

    m = lambda k: statistics.mean(r[k] for r in out)
    print(f"n={len(out)}\n")
    print(f"  adjusted, window + enrichment   {m('both'):.4f}")
    print(f"  adjusted, window only           {m('win_only'):.4f}")
    print(f"  -> enrichment is worth          {m('both') - m('win_only'):+.4f}")
    w = sum(1 for r in out if r["both"] > r["win_only"] + 1e-9)
    l = sum(1 for r in out if r["win_only"] > r["both"] + 1e-9)
    print(f"     wins/losses {w}/{l}\n")
    print(f"  tags kept when enrichment removed  {m('kept'):.1%}"
          f"   (high = enrichment decides our answer)\n")

    # How interchangeable are two windows of the SAME conversation?
    by_guid = {}
    for r in out:
        by_guid.setdefault(r["guid"], []).append(r)
    print("  within one conversation, do different windows give different answers?")
    for guid, rs in by_guid.items():
        if len(rs) < 2:
            continue
        sets = [set(r["tags_both"]) for r in rs]
        jac = [len(a & b) / max(1, len(a | b))
               for i, a in enumerate(sets) for b in sets[i + 1:]]
        sc = [r["both"] for r in rs]
        print(f"    {guid[:8]}  {len(rs):2d} windows   mean pairwise tag overlap "
              f"{statistics.mean(jac):.0%}   adjusted sd {statistics.pstdev(sc):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
