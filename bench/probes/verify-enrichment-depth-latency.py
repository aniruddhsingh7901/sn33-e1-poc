#!/usr/bin/env python3
"""Latency/cost check for the enrichment-depth proposal.

The gain was measured with pipeline.Config(deadline_s=600) - an unlimited
budget. Production runs an 11s deadline inside a 12s synapse. This measures, on
UNCACHED calls (fresh salt), what the deeper per-line prompt actually costs in
wall time and output tokens against the upstream prompt, at the real fan-out
width of each conversation.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import GT_MODEL, load_cases
from sn33 import llm, prompts

RUN = uuid.uuid4().hex[:8]


def deep_enrichment(line: str, n: int = 30) -> str:
    base = prompts.gt_enrichment(line)
    old = "5.  **Limit to most important:** Return at most 10 of the most important and relevant tags."
    new = (f"5.  **Be exhaustive:** Return at most {n} tags. Start with the most "
           "important and relevant, then continue with broader themes, alternative "
           "phrasings of the same concepts, adjacent subtopics, and the field or "
           "industry the content belongs to.")
    assert old in base
    return base.replace(old, new)


async def timed(prompt, tag):
    t0 = time.perf_counter()
    raw = await llm.chat(prompt, model=GT_MODEL, timeout=60, temperature=0.0,
                         use_cache=False, salt=f"lat{RUN}:{tag}")
    return time.perf_counter() - t0, len(raw or "")


async def main():
    cases = load_cases(kind="conversation_tagging", seed=0)
    for label, mk in (("upstream10", prompts.gt_enrichment), ("deep30", deep_enrichment)):
        per_convo_max, chars = [], []
        for c in cases:
            lines = list(c.enrichment_lines)
            if not lines:
                continue
            t0 = time.perf_counter()
            res = await asyncio.gather(*[timed(mk(l), f"{label}{i}") for i, l in enumerate(lines)])
            wall = time.perf_counter() - t0
            per_convo_max.append(wall)
            chars.extend(r[1] for r in res)
            print(f"  {label:11s} {c.guid:12s} lines={len(lines):2d} "
                  f"fanout wall {wall:5.2f}s  slowest call {max(r[0] for r in res):5.2f}s  "
                  f"chars {statistics.mean(r[1] for r in res):6.0f}")
        print(f"  -> {label}: worst conversation fan-out {max(per_convo_max):.2f}s, "
              f"mean chars/response {statistics.mean(chars):.0f}\n")
    print("usage:", llm.Usage.snapshot())


if __name__ == "__main__":
    asyncio.run(main())
