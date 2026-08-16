#!/usr/bin/env python3
"""Score miner configurations against REAL ground truth.

    python bench/run_faithful.py --kind conversation_tagging --n 15 \
        --strategies current,replica:pool_size=60

The difference from bench/run.py: ground truth here comes from the FULL
conversation via the testnet API, exactly as the validator builds it. bench/run.py
had to approximate it from the 10-line window, and that approximation produced a
recommendation that hurt production.

Reports `adjusted` alongside `final`, because 96% of the gap to the top miners
is adjusted score - penalties are worth at most 0.006 and are not the target.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from bench.harness import paired_delta, score_answer
from sn33 import llm, pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_strategy(spec: str) -> tuple:
    """`replica:pool_size=60:gt_model=gpt-5-mini` -> (name, cfg overrides)."""
    parts = spec.split(":")
    overrides: Dict[str, object] = {}
    for p in parts[1:]:
        if not p:
            continue
        k, v = p.split("=", 1)
        if v.lower() in ("true", "false"):
            overrides[k] = v.lower() == "true"
        elif v.replace(".", "", 1).isdigit():
            overrides[k] = float(v) if "." in v else int(v)
        else:
            overrides[k] = v
    return spec, overrides


async def run_one(case, spec, overrides, gt, sem, window=None):
    """Run the miner on one window of this case and score against real GT."""
    cfg_kwargs = dict(use_cache=True, use_local=False, deadline_s=600.0, call_timeout_s=180.0)
    cfg_kwargs.update(overrides)
    async with sem:
        res = await pipeline.mine(
            case.kind,
            window=window if case.kind == "conversation_tagging" else [(0, case.document)],
            enrichment=case.enrichment_lines,
            question=case.question,
            comment=case.comment,
            cfg=pipeline.Config(**cfg_kwargs),
        )
        return await score_answer(gt, res.tags, case.kind, model=GT_MODEL), res


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="conversation_tagging")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--strategies", default="current")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sources = load_cases(kind=args.kind, seed=args.seed)
    if not sources:
        raise SystemExit(f"no {args.kind} cases - run bench/fetch_testnet_corpus.py")
    # One test case per WINDOW. The testnet pool holds only 4 distinct
    # conversations, but they split into 81 windows - each a different miner
    # input sharing the same ground truth, which is exactly what a paired A/B
    # needs. Topic coverage is still only those 4 sources.
    pairs = expand_windows(sources, seed=args.seed)[: args.n]
    cases = [c for c, _ in pairs]
    windows = [w for _, w in pairs]

    specs = [parse_strategy(s.strip()) for s in args.strategies.split(",") if s.strip()]
    print(f"corpus: {args.kind}  {len(sources)} distinct sources -> {len(pairs)} windows "
          f"(REAL ground truth from full source)")
    if args.kind == "conversation_tagging":
        ln = [len(c.full_lines) for c in sources]
        en = [len(c.enrichment_lines) for c in sources]
        print(f"  full conversation lines: min={min(ln)} max={max(ln)} mean={statistics.mean(ln):.0f}"
              f"   enrichment lines/case: mean={statistics.mean(en):.1f}")
    print(f"strategies: {', '.join(s for s, _ in specs)}\n")

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()

    # Ground truth once per case, shared by every strategy -> paired comparison.
    async def gt_for(case):
        async with sem:
            return await real_ground_truth(case)

    gts = await asyncio.gather(*[gt_for(c) for c in cases])

    usable = [(c, g, w) for c, g, w in zip(cases, gts, windows) if g.ok()]
    print(f"ground truth built for {len(usable)}/{len(cases)} windows "
          f"(mean {statistics.mean([len(g.tags) for _, g, _ in usable]):.1f} gt tags each)\n")

    results: Dict[str, List] = {}
    for spec, overrides in specs:
        out = await asyncio.gather(*[run_one(c, spec, overrides, g, sem, w) for c, g, w in usable])
        results[spec] = [v for v, _ in out]

    print(f"scored in {time.perf_counter()-t0:.0f}s  ({llm.Usage.snapshot()})\n")
    hdr = f"{'strategy':38s} {'ADJUSTED':>9s} {'final':>8s} {'median':>8s} {'sd':>6s} {'tags':>5s} {'pen%':>5s}"
    print(hdr); print("-" * len(hdr))
    for spec, _ in specs:
        v = results[spec]
        if not v:
            continue
        adj = [x.detail.get("adjusted", 0.0) for x in v]
        fin = [x.final for x in v]
        pen = sum(1 for x in v if x.penalties)
        print(f"{spec:38s} {statistics.mean(adj):9.4f} {statistics.mean(fin):8.4f} "
              f"{statistics.median(fin):8.4f} {statistics.pstdev(fin):6.3f} "
              f"{statistics.mean([x.n_survived for x in v]):5.1f} {100*pen/len(v):4.0f}%")

    base = specs[0][0]
    if len(specs) > 1:
        print(f"\npaired vs {base} (bootstrap 95% CI on the per-case difference):")
        for spec, _ in specs[1:]:
            d = paired_delta(results[base], results[spec])
            if d:
                print(f"  {spec:36s} {d['delta']:+.4f} CI[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
                      f"W/L {d['wins']}/{d['losses']} "
                      f"{'SIGNIFICANT' if d['significant'] else 'not significant'}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({s: [v.__dict__ for v in results[s]] for s, _ in specs}, f, indent=1, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
