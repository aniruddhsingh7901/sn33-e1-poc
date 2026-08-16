#!/usr/bin/env python3
"""VERIFY probe: adversarial re-test of `centroid-blend`.

The original probe's end-to-end arm SWAPS blends in: it deletes the last 4 of
the miner's 12 tags and appends 4 blends. That confounds two things - the cost
of deleting 4 miner tags, and the value of the blends. The miner ships 12 but
the cap is 19, so blends could be ADDED for free.

This probe decomposes it, per window, all through score_answer:

    base      miner_final                                (12)
    drop4     miner_final[:-4]                           ( 8)   deletion cost alone
    swap4     miner_final[:-4] + 4 best blends           (12)   the original arm
    add4      miner_final     + 4 best blends            (16)   additive steelman
    add4pool  miner_final     + 4 best UNUSED miner pool (16)   like-for-like control
    add7      miner_final     + 7 best blends            (19)   fill the cap

Blends and pool fillers are both ranked by cosine to the ESTIMATED centroid
(the miner's own replica), never the real one - a miner has no oracle.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import random
import statistics
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from bench.harness import score_answer
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline
from sn33.tags import centroid, cosine, normalize, normalize_all

SALT = "verify-centroid-blend"


def blend(tags: Sequence[str]) -> Optional[str]:
    words: List[str] = []
    seen = set()
    for t in tags:
        for w in str(t).split():
            if w in seen:
                continue
            seen.add(w)
            words.append(w)
    if not words:
        return None
    return normalize(" ".join(words))


def blends_from(tags: Sequence[str], k: int, cap: int, rng: random.Random) -> List[str]:
    tags = [t for t in tags if t]
    if len(tags) < k:
        return []
    combos = list(itertools.combinations(tags, k))
    if len(combos) > cap:
        combos = rng.sample(combos, cap)
    out, seen = [], set()
    for c in combos:
        b = blend(c)
        if b and b not in seen:
            seen.add(b)
            out.append(b)
    return out


ARMS = ["base", "drop4", "swap4", "add4", "add4pool", "add7"]


async def one_window(case, window, args, rng_seed: int) -> Optional[dict]:
    gt = await real_ground_truth(case)
    if not gt.ok():
        return None
    rng = random.Random(rng_seed)

    res = await pipeline.mine(
        case.kind,
        window=window,
        enrichment=case.enrichment_lines,
        cfg=pipeline.Config(use_cache=True, use_local=False, deadline_s=600.0, call_timeout_s=180.0),
    )
    base = normalize_all(res.tags)
    pool = normalize_all(res.candidates)
    pred_gt = normalize_all(res.predicted_gt)
    if len(base) < 12 or not pred_gt:
        return None

    # the miner's own estimate of the target
    est_vecs = await llm.embed(pred_gt, use_cache=True, timeout=120)
    have = [t for t in pred_gt if t in est_vecs]
    if not have:
        return None
    est = centroid([est_vecs[t] for t in have])

    cand_blends = list(dict.fromkeys(blends_from(pred_gt, 2, args.cap, rng)))
    unused_pool = [t for t in pool if t not in set(base)]
    if not cand_blends or len(unused_pool) < 4:
        return None

    bv = await llm.embed(cand_blends, use_cache=True, timeout=120)
    pv = await llm.embed(unused_pool, use_cache=True, timeout=120)

    def rank(items, vecs):
        return [t for _, t in sorted(
            ((cosine(est, np.asarray(vecs[t], dtype=np.float32)), t) for t in items if t in vecs),
            reverse=True)]

    rb = rank(cand_blends, bv)
    rp = rank(unused_pool, pv)
    if len(rb) < 7 or len(rp) < 4:
        return None

    answers = {
        "base": base,
        "drop4": base[:-4],
        "swap4": base[:-4] + rb[:4],
        "add4": base + rb[:4],
        "add4pool": base + rp[:4],
        "add7": base + rb[:7],
    }
    answers = {k: normalize_all(v) for k, v in answers.items()}

    verdicts = {}
    for name, tags in answers.items():
        v = await score_answer(gt, tags, case.kind, model=GT_MODEL, use_cache=True)
        verdicts[name] = {
            "adjusted": v.adjusted, "final": v.final,
            "submitted": len(tags), "survived": v.n_survived,
            "unique": v.n_unique, "both": v.n_both,
            "penalties": v.penalties,
        }
    return {
        "guid": case.guid, "v": verdicts,
        "blends": rb[:7], "pool_fill": rp[:4], "base_tags": base,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--cap", type=int, default=90)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    per = max(1, args.n // max(1, len(cases)))
    pairs = expand_windows(cases, per_convo=per, seed=0)
    print(f"{len(pairs)} windows from {len(cases)} conversations ({per}/conversation)\n", flush=True)

    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(i, case, window):
        async with sem:
            try:
                return await one_window(case, window, args, rng_seed=i)
            except Exception as e:  # noqa: BLE001
                print(f"  window {i} failed: {type(e).__name__}: {e}", flush=True)
                return None

    rows = [r for r in await asyncio.gather(*[guarded(i, c, w) for i, (c, w) in enumerate(pairs)]) if r]
    print(f"{len(rows)} windows completed\n")
    if not rows:
        return

    by_convo: Dict[str, List[dict]] = {}
    for r in rows:
        by_convo.setdefault(r["guid"], []).append(r)
    heads = list(by_convo.keys())

    def m(rs, arm, key):
        return statistics.mean(r["v"][arm][key] for r in rs)

    print("=" * 96)
    print(f"ARMS  n={len(rows)}   (score_answer, real GT, validator English screen included)")
    print("=" * 96)
    print(f"{'arm':<10} {'submit':>7} {'survive':>8} {'unique':>7} {'both':>6} "
          f"{'adjusted':>9} {'final':>8}   {'d(adj) vs base':>15}  {'W/L':>7}")
    for arm in ARMS:
        d = [r["v"][arm]["adjusted"] - r["v"]["base"]["adjusted"] for r in rows]
        wl = f"{sum(1 for x in d if x>0)}/{sum(1 for x in d if x<0)}"
        print(f"{arm:<10} {m(rows,arm,'submitted'):>7.1f} {m(rows,arm,'survived'):>8.2f} "
              f"{m(rows,arm,'unique'):>7.2f} {m(rows,arm,'both'):>6.2f} "
              f"{m(rows,arm,'adjusted'):>9.4f} {m(rows,arm,'final'):>8.4f}   "
              f"{statistics.mean(d):>+15.4f}  {wl:>7}")

    print("\n" + "=" * 96)
    print("KEY PAIRED CONTRASTS  (adjusted)")
    print("=" * 96)
    contrasts = [
        ("swap4", "drop4", "blends added into a shortened answer (isolates blend value)"),
        ("add4", "base", "blends ADDED for free, nothing deleted (the steelman)"),
        ("add4", "add4pool", "blend vs 4 more of the miner's OWN candidates (like-for-like)"),
        ("add4pool", "base", "control: does adding ANY 4 more tags help?"),
        ("add7", "base", "fill the 19-tag cap with blends"),
        ("drop4", "base", "cost of the deletion the original probe bundled in"),
        ("swap4", "base", "the ORIGINAL probe's headline number"),
    ]
    for a, b, why in contrasts:
        d = [r["v"][a]["adjusted"] - r["v"][b]["adjusted"] for r in rows]
        per_c = " ".join(
            f"{statistics.mean([r['v'][a]['adjusted']-r['v'][b]['adjusted'] for r in by_convo[g]]):+.4f}"
            for g in heads)
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        print(f"  {a:>8} - {b:<9} {statistics.mean(d):+.4f}  sd {sd:.4f}  "
              f"W/L {sum(1 for x in d if x>0)}/{sum(1 for x in d if x<0)}   per-convo {per_c}")
        print(f"           {why}")

    print("\n" + "=" * 96)
    print("PENALTY / SHAPE DIAGNOSTICS")
    print("=" * 96)
    for arm in ARMS:
        pen: Dict[str, int] = {}
        for r in rows:
            for p in r["v"][arm]["penalties"]:
                pen[p] = pen.get(p, 0) + 1
        lost = m(rows, arm, "submitted") - m(rows, arm, "survived")
        print(f"  {arm:<10} screen-loss {lost:>5.2f} tags  penalties {pen}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, default=str, indent=1)
        print(f"\nwrote {args.out}")
    print("\nusage:", json.dumps(llm.Usage.snapshot()))


if __name__ == "__main__":
    asyncio.run(main())
