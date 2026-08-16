#!/usr/bin/env python3
"""Compare our offline bench numbers against real production scores.

Two different measurements get confused easily, so this script states the
relationship explicitly:

* `bench/run.py` scores a strategy against ground truth **we** generate, one
  task type at a time. It is paired and good for ranking strategies.
* `bench/wandb_scores.py` reads what validators actually recorded, pooled across
  all task types in whatever mix was sent, with no task labels (the validator
  does not log them).

So an offline mean is NOT directly comparable to a W&B mean. What makes them
comparable is weighting our per-task results by the observed traffic mix, which
is what this does - then the residual gap is the honest measure of how much our
replica flatters itself.

    python bench/calibrate.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Traffic mix measured from 913 captured production responses (May 2026).
# skill_generation shipped 2026-07-10 and is absent from that capture, so the
# mix below is the pre-skill distribution and is a lower bound on how much the
# newest task type now displaces the others.
TRAFFIC_MIX = {
    "conversation_tagging": 0.659,
    "webpage_metadata_generation": 0.155,
    "named_entities_extraction": 0.124,
    "survey_tagging": 0.062,
}

# Offline bench means, from docs/SN33_RESULTS.md.
OFFLINE = {
    "conversation_tagging": {"prod": 0.5718, "sn33": 0.6408},
    "webpage_metadata_generation": {"prod": 0.5820, "sn33": 0.6320},
    "named_entities_extraction": {"prod": 0.5644, "sn33": 0.6172},
    "survey_tagging": {"prod": 0.0000, "sn33": 0.4804},
}


def weighted(which: str) -> float:
    return sum(TRAFFIC_MIX[k] * OFFLINE[k][which] for k in TRAFFIC_MIX)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=os.path.join(ROOT, "data", "wandb_scores.json"))
    args = ap.parse_args()

    if not os.path.exists(args.scores):
        raise SystemExit(f"{args.scores} missing - run bench/wandb_scores.py first")
    payload = json.load(open(args.scores))
    miners = payload["miners"]
    miners.sort(key=lambda r: -r["mean"])

    means = [m["mean"] for m in miners]
    nz_means = [m["mean_nonzero"] for m in miners]
    zero_rates = [m["zero_rate"] for m in miners]

    print("=" * 78)
    print("REAL PRODUCTION SCORES (validator W&B, all task types pooled)")
    print("=" * 78)
    print(f"miners: {len(miners)}   observations: {sum(m['n'] for m in miners)}")
    print(f"  best miner mean      : {means[0]:.4f}   (uid {miners[0]['uid']}, n={miners[0]['n']})")
    print(f"  top-10 mean          : {statistics.mean(means[:10]):.4f}")
    print(f"  median miner         : {statistics.median(means):.4f}")
    print(f"  worst miner          : {means[-1]:.4f}")
    print(f"  zero rate, best 10   : {statistics.mean(zero_rates[:10])*100:.1f}%")
    print(f"  zero rate, all       : {statistics.mean(zero_rates)*100:.1f}%")
    print(f"  best miner, nonzero-only mean: {miners[0]['mean_nonzero']:.4f}")
    print(f"  top-10 nonzero-only mean     : {statistics.mean(nz_means[:10]):.4f}")

    print()
    print("=" * 78)
    print("OUR OFFLINE BENCH, WEIGHTED BY OBSERVED TRAFFIC MIX")
    print("=" * 78)
    for k, share in sorted(TRAFFIC_MIX.items(), key=lambda kv: -kv[1]):
        o = OFFLINE[k]
        print(f"  {k:30s} {share*100:5.1f}%  prod {o['prod']:.4f} -> sn33 {o['sn33']:.4f}")
    wp, ws = weighted("prod"), weighted("sn33")
    print(f"\n  weighted production baseline : {wp:.4f}")
    print(f"  weighted sn33 miner          : {ws:.4f}   ({ws-wp:+.4f})")

    print()
    print("=" * 78)
    print("READING")
    print("=" * 78)
    gap_best = means[0] - ws
    gap_top10 = statistics.mean(means[:10]) - ws
    print(f"  vs best real miner   : {gap_best:+.4f}")
    print(f"  vs top-10 real mean  : {gap_top10:+.4f}")
    print(f"  vs median real miner : {statistics.median(means) - ws:+.4f}")
    print()
    print("  Caveats that bound these differences:")
    print("   - W&B means pool ALL task types with no labels; ours are weighted")
    print("     by a May traffic mix that predates skill_generation.")
    print("   - W&B means INCLUDE zeros from failed/timed-out responses; a miner")
    print("     with a 10% zero rate loses 10% of its mean to reliability, not tags.")
    print("   - Our ground truth is our own OpenAI samples, not a validator's.")
    print("   - Compare 'nonzero-only' columns for tag quality; compare full means")
    print("     for what actually gets paid.")


if __name__ == "__main__":
    main()
