#!/usr/bin/env python3
"""VERIFY probe v2: isolate the two confounds in `centroid-blend`'s stage 2.

Confound 1 - the original swap arm draws its 4 tags from `blend_pred_2 +
blend_pred_3` combined, then ranks by estimated centroid. 3-blends win that
ranking often, and the probe's OWN cosine table scores blend_pred_3 at -0.0292
vs the miner pool. So the end-to-end arm tested a mixture containing the family
already known to be bad.

Confound 2 - the swap DELETES 4 of the miner's 12 tags. The miner ships 12 of a
19 cap, so blends can be added at no cost.

Arms (all scored through score_answer with the real English screen):
    base        miner_final                          (12)
    swap4_23    base[:-4] + top4 of 2-blends+3-blends (12)  <- the original arm
    swap4_2     base[:-4] + top4 of 2-blends only     (12)
    add4_23     base      + top4 of 2-blends+3-blends (16)
    add4_2      base      + top4 of 2-blends only     (16)
    add4pool    base      + top4 unused miner pool    (16)  <- like-for-like control
    add7_2      base      + top7 of 2-blends only     (19)

Run it TWICE. The first pass populates the cache; the English screen is an LLM
call at temperature 1.0 and is only reproducible once cached, so the second
pass is the number to quote.
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
from sn33 import llm, pipeline
from sn33.tags import centroid, cosine, normalize, normalize_all

ARMS = ["base", "swap4_23", "swap4_2", "add4_23", "add4_2", "add4pool", "add7_2"]


def blend(tags: Sequence[str]) -> Optional[str]:
    words, seen = [], set()
    for t in tags:
        for w in str(t).split():
            if w not in seen:
                seen.add(w)
                words.append(w)
    return normalize(" ".join(words)) if words else None


def blends_from(tags, k, cap, rng):
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


async def one_window(case, window, args, rng_seed: int) -> Optional[dict]:
    gt = await real_ground_truth(case)
    if not gt.ok():
        return None
    rng = random.Random(rng_seed)
    res = await pipeline.mine(
        case.kind, window=window, enrichment=case.enrichment_lines,
        cfg=pipeline.Config(use_cache=True, use_local=False, deadline_s=600.0, call_timeout_s=180.0))
    base, pool = normalize_all(res.tags), normalize_all(res.candidates)
    pred_gt = normalize_all(res.predicted_gt)
    if len(base) < 12 or not pred_gt:
        return None

    ev = await llm.embed(pred_gt, use_cache=True, timeout=120)
    have = [t for t in pred_gt if t in ev]
    if not have:
        return None
    est = centroid([ev[t] for t in have])

    b2 = blends_from(pred_gt, 2, args.cap, rng)
    b3 = blends_from(pred_gt, 3, args.cap, rng)
    b23 = list(dict.fromkeys(b2 + b3))
    unused = [t for t in pool if t not in set(base)]
    if len(b2) < 7 or not b3 or len(unused) < 4:
        return None

    allv = await llm.embed(list(dict.fromkeys(b23 + unused)), use_cache=True, timeout=120)

    def rank(items):
        return [t for _, t in sorted(
            ((cosine(est, np.asarray(allv[t], dtype=np.float32)), t) for t in items if t in allv),
            reverse=True)]

    r2, r23, rp = rank(b2), rank(b23), rank(unused)
    if len(r2) < 7 or len(r23) < 4 or len(rp) < 4:
        return None

    answers = {
        "base": base,
        "swap4_23": base[:-4] + r23[:4],
        "swap4_2": base[:-4] + r2[:4],
        "add4_23": base + r23[:4],
        "add4_2": base + r2[:4],
        "add4pool": base + rp[:4],
        "add7_2": base + r2[:7],
    }
    verdicts = {}
    for name, tags in answers.items():
        v = await score_answer(gt, tags, case.kind, model=GT_MODEL, use_cache=True)
        verdicts[name] = {"adjusted": v.adjusted, "final": v.final, "submitted": len(tags),
                          "survived": v.n_survived, "unique": v.n_unique, "both": v.n_both,
                          "penalties": v.penalties}
    # how many of the original arm's 4 picks were 3-blends?
    set2 = set(r2)
    n3 = sum(1 for t in r23[:4] if t not in set2)
    return {"guid": case.guid, "v": verdicts, "n3_in_orig_pick": n3,
            "orig_pick": r23[:4], "two_pick": r2[:4]}


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
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(i, case, window):
        async with sem:
            try:
                return await one_window(case, window, args, rng_seed=i)
            except Exception as e:  # noqa: BLE001
                print(f"  window {i} failed: {type(e).__name__}: {e}", flush=True)
                return None

    rows = [r for r in await asyncio.gather(*[guarded(i, c, w) for i, (c, w) in enumerate(pairs)]) if r]
    print(f"{len(rows)} windows from {len(cases)} conversations\n")
    if not rows:
        return
    by_convo: Dict[str, List[dict]] = {}
    for r in rows:
        by_convo.setdefault(r["guid"], []).append(r)
    heads = list(by_convo.keys())

    def m(rs, arm, key):
        return statistics.mean(r["v"][arm][key] for r in rs)

    print("=" * 100)
    print(f"ARMS  n={len(rows)}")
    print("=" * 100)
    print(f"{'arm':<10} {'submit':>7} {'survive':>8} {'unique':>7} {'both':>6} {'adjusted':>9} {'final':>8}")
    for a in ARMS:
        print(f"{a:<10} {m(rows,a,'submitted'):>7.1f} {m(rows,a,'survived'):>8.2f} "
              f"{m(rows,a,'unique'):>7.2f} {m(rows,a,'both'):>6.2f} "
              f"{m(rows,a,'adjusted'):>9.4f} {m(rows,a,'final'):>8.4f}")

    print("\n" + "=" * 100)
    print("PAIRED CONTRASTS (adjusted)")
    print("=" * 100)
    pairs_c = [("swap4_23", "base"), ("swap4_2", "base"), ("swap4_2", "swap4_23"),
               ("add4_2", "base"), ("add4_23", "base"), ("add4_2", "add4pool"),
               ("add4pool", "base"), ("add7_2", "base"), ("add4_2", "swap4_2")]
    for a, b in pairs_c:
        d = [r["v"][a]["adjusted"] - r["v"][b]["adjusted"] for r in rows]
        per_c = " ".join(
            f"{statistics.mean([r['v'][a]['adjusted']-r['v'][b]['adjusted'] for r in by_convo[g]]):+.4f}"
            for g in heads)
        allpos = all(statistics.mean([r['v'][a]['adjusted']-r['v'][b]['adjusted'] for r in by_convo[g]]) > 0
                     for g in heads)
        print(f"  {a:>9} - {b:<9} {statistics.mean(d):+.4f}  sd {statistics.stdev(d):.4f}  "
              f"W/L {sum(1 for x in d if x>0)}/{sum(1 for x in d if x<0)}  "
              f"per-convo {per_c}  {'ALL4' if allpos else ''}")

    n3 = [r["n3_in_orig_pick"] for r in rows]
    print(f"\nORIGINAL arm's 4 picks: {statistics.mean(n3):.2f} of 4 were 3-blends "
          f"(the family its own cosine table scores at -0.0292 vs pool); "
          f"windows with >=1 three-blend: {sum(1 for x in n3 if x)}/{len(n3)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, default=str, indent=1)
    print("\nusage:", json.dumps(llm.Usage.snapshot()))


if __name__ == "__main__":
    asyncio.run(main())
