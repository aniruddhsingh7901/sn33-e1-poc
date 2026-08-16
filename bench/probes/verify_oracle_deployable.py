#!/usr/bin/env python3
"""The only deployable claim in `oracle-shape`, scored on `final`.

Section E of the original probe adds mechanical recombination/template phrases
built from the miner's OWN predicted ground truth (no oracle, no extra LLM call)
and reports +0.0394 adjusted.  But its own output says
`arm B penalties fired: {'no_both_tags': 38}` - it fires on EVERY window,
because both arms of E select unique-only.  The shipped miner deliberately buys
exact matches (insurance=6) and fires that penalty once in 38.

So E's arms are not comparable to the shipped miner.  This re-runs the
deployable idea through the miner's REAL composer (`pipeline.compose`), which
keeps the insurance rule, changing exactly ONE variable: whether mechanical
recombination phrases are in the candidate pool.

  arm A: miner pool           -> compose -> score (this IS the shipped miner)
  arm B: miner pool + recomb  -> compose -> score
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import itertools
import os
import statistics
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from bench.harness import score_answer
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline
from sn33.tags import cosine, normalize_all

STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with",
        "by", "from", "as", "at", "is", "are"}
SUF = ["strategy", "practices", "research", "trends", "concepts",
       "industry", "landscape", "insights", "analysis", "topics"]
PROFILE = pipeline.TASK_PROFILE["conversation_tagging"]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-convo", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases, per_convo=args.per_convo, seed=0)
    gts = {}
    for ci, case in enumerate(cases):
        gt = await real_ground_truth(case)
        if gt.ok():
            gts[case.guid] = (ci, gt)
    print(f"{len(gts)} conversations, {len(pairs)} windows\n", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    cfg = pipeline.Config(use_cache=True, use_local=False, deadline_s=600, call_timeout_s=180)

    async def one(case, window):
        async with sem:
            if case.guid not in gts:
                return None
            ci, gt = gts[case.guid]
            res = await pipeline.mine(case.kind, window=window,
                                      enrichment=case.enrichment_lines, cfg=cfg)
            if not res.candidates or not res.vectors or not res.predicted_gt:
                return None
            pred = normalize_all(Utils.get_clean_tag_set(res.predicted_gt))
            if not pred:
                return None

            pw = list(dict.fromkeys(w for t in pred for w in t.split()
                                    if w not in STOP and len(w) > 2))
            extra = [f"{a} {b}" for a, b in itertools.permutations(pw, 2)][:400]
            extra += [f"{t} {s}" for t in pred for s in SUF]
            extra = [t for t in normalize_all(extra) if t not in set(res.candidates)]
            ev = await llm.embed(extra, use_cache=True)
            allv = dict(res.vectors); allv.update(ev)
            est = np.mean([np.asarray(allv[t]) for t in pred if t in allv], axis=0)

            def rank(pool):
                return sorted(((t, cosine(est, np.asarray(allv[t])))
                               for t in dict.fromkeys(pool) if t in allv), key=lambda x: -x[1])

            a_tags = pipeline.compose(rank(res.candidates), pred, PROFILE,
                                      PROFILE["target_tags"], PROFILE["insurance"])
            b_tags = pipeline.compose(rank(list(res.candidates) + extra), pred, PROFILE,
                                      PROFILE["target_tags"], PROFILE["insurance"])
            va = await score_answer(gt, a_tags, case.kind, model=GT_MODEL)
            vb = await score_answer(gt, b_tags, case.kind, model=GT_MODEL)
            vs = await score_answer(gt, res.tags, case.kind, model=GT_MODEL)
            return {"conv": ci,
                    "a_adj": va.adjusted, "a_fin": va.final, "a_both": va.n_both,
                    "a_pen": va.penalties, "a_sur": va.n_survived,
                    "b_adj": vb.adjusted, "b_fin": vb.final, "b_both": vb.n_both,
                    "b_pen": vb.penalties, "b_sur": vb.n_survived,
                    "s_adj": vs.adjusted, "s_fin": vs.final,
                    "b_new": sum(1 for t in b_tags if t in set(extra)),
                    "b_tags": b_tags, "a_tags": a_tags}

    out = [r for r in await asyncio.gather(*[one(c, w) for c, w in pairs]) if r]
    byc = collections.defaultdict(list)
    for r in out:
        byc[r["conv"]].append(r)

    def m(rows, k):
        return statistics.mean(r[k] for r in rows)

    print(f"n = {len(out)} windows   (compose keeps insurance={PROFILE['insurance']}, "
          f"target_tags={PROFILE['target_tags']})\n")
    print(f"{'arm':44s} {'adjusted':>9s} {'final':>9s} {'both':>6s} {'survived':>9s}")
    print("-" * 84)
    for pre, lbl in (("s", "shipped miner (reference)"),
                     ("a", "A: miner pool -> compose"),
                     ("b", "B: miner pool + recombination -> compose")):
        both = f"{m(out, pre+'_both'):6.2f}" if pre != "s" else "     -"
        sur = f"{m(out, pre+'_sur'):9.2f}" if pre != "s" else "        -"
        print(f"{lbl:44s} {m(out,pre+'_adj'):9.4f} {m(out,pre+'_fin'):9.4f} {both} {sur}")

    print()
    for key, base, lbl in (("b_adj", "a_adj", "B - A on ADJUSTED"),
                           ("b_fin", "a_fin", "B - A on FINAL   ")):
        d = m(out, key) - m(out, base)
        w = sum(1 for r in out if r[key] > r[base])
        l = sum(1 for r in out if r[key] < r[base])
        pcs = [m(byc[i], key) - m(byc[i], base) for i in sorted(byc)]
        print(f"  {lbl}: {d:+.4f}   W/L {w}/{l}   per-conv "
              + "  ".join(f"{x:+.4f}" for x in pcs))
    d = m(out, "b_fin") - m(out, "s_fin")
    w = sum(1 for r in out if r["b_fin"] > r["s_fin"])
    print(f"  B vs SHIPPED on FINAL: {d:+.4f}   W/L {w}/{len(out)-w}")

    print(f"\n  recombination tags actually chosen per window: {m(out,'b_new'):.1f} of 12")
    for lbl, k in (("A", "a_pen"), ("B", "b_pen")):
        print(f"  penalties arm {lbl}: {dict(collections.Counter(p for r in out for p in r[k])) or 'none'}")
    print(f"\n  example arm B answer: {', '.join(out[0]['b_tags'])}")
    print(f"  example arm A answer: {', '.join(out[0]['a_tags'])}")


if __name__ == "__main__":
    asyncio.run(main())
