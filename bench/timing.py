#!/usr/bin/env python3
"""Latency against the real synapse budget.

The validator calls ``dendrite.forward()`` with no timeout argument
(neurons/validator.py:269), so bittensor's default of **12 seconds** applies.
``--neuron.timeout`` (default 10) is defined but never passed, and Discord
guidance says 10s - so the miner targets 9s and treats anything above 10 as a
failure regardless of what the code allows.

A late answer is scored exactly like no answer, so this measures the tail, not
the average: p95 and max are the numbers that matter.

    python bench/timing.py --n 8 --repeat 2
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import corpus
from sn33 import extract, pipeline

BUDGET = 12.0        # bittensor Dendrite default
SAFE = 10.0          # Discord-stated expectation


def _pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


async def time_kind(kind: str, cases, cfg: pipeline.Config, repeat: int):
    times, sources, tag_counts, degraded = [], [], [], 0
    for case in cases:
        for _ in range(repeat):
            t0 = time.perf_counter()
            res = await pipeline.mine(
                kind,
                window=case.miner_window(),
                enrichment=case.enrichment,
                question=case.question,
                comment=case.comment,
                coding=case.coding,
                cfg=cfg,
            )
            times.append(time.perf_counter() - t0)
            sources.append(res.source)
            tag_counts.append(len(res.tags))
            degraded += int(res.degraded)
    return times, sources, tag_counts, degraded


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--deadline", type=float, default=9.0)
    ap.add_argument("--model", default=None, help="override SN33_GT_MODEL for this run")
    args = ap.parse_args()

    print("warming local extractor (miner does this at startup)...")
    t0 = time.perf_counter()
    extract.warm()
    print(f"  spacy warm: {time.perf_counter()-t0:.2f}s\n")

    cfg = pipeline.Config(deadline_s=args.deadline, use_cache=False, use_local=True)
    if args.model:
        cfg.gt_model = cfg.pool_model = args.model

    kinds = {
        "conversation_tagging": lambda: corpus.conversation_cases(n=args.n),
        "webpage_metadata_generation": lambda: corpus.captured_cases("webpage_metadata_generation", n=args.n),
        "named_entities_extraction": lambda: corpus.captured_cases("named_entities_extraction", n=args.n),
    }

    header = f"{'task':30s} {'n':>3s} {'mean':>6s} {'p50':>6s} {'p95':>6s} {'max':>6s} {'>10s':>5s} {'>12s':>5s} {'tags':>5s}"
    print(header)
    print("-" * len(header))
    worst = 0.0
    for kind, build in kinds.items():
        try:
            cases = build()
        except Exception as e:
            print(f"{kind:30s} corpus unavailable: {e}")
            continue
        times, sources, tags, degraded = await time_kind(kind, cases, cfg, args.repeat)
        worst = max(worst, max(times))
        print(
            f"{kind:30s} {len(times):3d} {statistics.mean(times):6.2f} {_pct(times,0.5):6.2f} "
            f"{_pct(times,0.95):6.2f} {max(times):6.2f} {sum(1 for t in times if t>SAFE):5d} "
            f"{sum(1 for t in times if t>BUDGET):5d} {statistics.mean(tags):5.1f}"
        )
        src = {s: sources.count(s) for s in sorted(set(sources))}
        print(f"{'':30s} sources={src} degraded={degraded}")

    print(f"\nworst observed: {worst:.2f}s against a {BUDGET}s synapse budget "
          f"({'OK' if worst < SAFE else 'TOO SLOW'})")


if __name__ == "__main__":
    asyncio.run(main())
