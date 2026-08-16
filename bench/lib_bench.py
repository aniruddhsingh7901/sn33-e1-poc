#!/usr/bin/env python3
"""Phase 1: which local extraction library earns its dependency?

The library is only ever used for two things, so those are what get measured:

1. **Latency** - it runs before any API call, so it must be negligible against
   the 12s synapse budget.
2. **Score as a standalone answer** - this is the number that matters. If every
   API call fails, the miner submits whatever the extractor produced. A
   fallback that scores 0.3 is worth far more than a timeout, because a zero
   decays the EMA and a low score does not.

Tag-likeness is not judged by eye here: each extractor's output is run through
the real validator path (validate_tag_set -> embed -> score) against ground
truth built from the full document.

    python bench/lib_bench.py --kind conversation_tagging --n 12
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
from bench.harness import build_ground_truth, score_answer, summarize
from sn33 import extract
from sn33.pipeline import _document_text

BACKENDS = ["spacy", "yake", "rake", "keybert"]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="conversation_tagging")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--limit", type=int, default=15, help="tags each extractor may emit")
    args = ap.parse_args()

    if args.kind == "conversation_tagging":
        cases = corpus.conversation_cases(n=args.n)
    else:
        cases = corpus.captured_cases(args.kind, n=args.n)

    # Cold-start cost is paid once at miner boot, so report it separately from
    # per-request latency.
    print("cold start (model load):")
    for b in BACKENDS:
        t0 = time.perf_counter()
        try:
            extract.EXTRACTORS[b]("warm up the model with a short sentence about databases.", limit=5)
            print(f"  {b:9s} {time.perf_counter()-t0:6.2f}s")
        except Exception as e:
            print(f"  {b:9s} FAILED {type(e).__name__}: {e}")

    per_backend = {b: [] for b in BACKENDS}
    times = {b: [] for b in BACKENDS}
    samples = {}

    for case in cases:
        gt = await build_ground_truth(case, model=args.model)
        if not gt.ok():
            continue
        text = _document_text(case.kind, case.miner_window())
        for b in BACKENDS:
            tags, secs = extract.timed_candidates(text, b, limit=args.limit)
            times[b].append(secs)
            tags = tags[: args.limit]
            if b not in samples and tags:
                samples[b] = tags[:10]
            per_backend[b].append(await score_answer(gt, tags, case.kind, model=args.model))

    print(f"\nscored {len(cases)} cases on {args.kind}\n")
    header = f"{'backend':10s} {'mean':>7s} {'median':>7s} {'zero':>5s} {'tags':>5s} {'ms/doc':>8s} {'p90 ms':>8s}"
    print(header)
    print("-" * len(header))
    for b in BACKENDS:
        v = [x for x in per_backend[b] if x]
        if not v:
            print(f"{b:10s} unavailable")
            continue
        s = summarize(b, v)
        ms = [t * 1000 for t in times[b]]
        p90 = sorted(ms)[int(len(ms) * 0.9)] if ms else 0
        print(
            f"{b:10s} {s['mean']:7.4f} {s['median']:7.4f} {s['zeros']:5d} {s['mean_tags']:5.1f} "
            f"{statistics.mean(ms):8.1f} {p90:8.1f}"
        )

    print("\nsample output (first case):")
    for b, tags in samples.items():
        print(f"  {b:9s} {tags}")


if __name__ == "__main__":
    asyncio.run(main())
