#!/usr/bin/env python3
"""Count DISTINCT sources per task type using load_cases' exact dedupe rule.

Read-only. Takes an optional corpus path so we can compare a snapshot against
the live file without touching either.

    python bench/probes/count_distinct_sources.py [path.jsonl]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from bench.faithful import load_cases  # noqa: E402

KINDS = ["conversation_tagging", "webpage_metadata_generation", "skill_generation",
         "named_entities_extraction", "survey_tagging"]


def raw_counts(path: str) -> Counter:
    c = Counter()
    with open(path) as f:
        for line in f:
            if line.strip():
                c[json.loads(line).get("type")] += 1
    return c


def distinct(path: str) -> dict:
    out = {}
    for k in KINDS:
        try:
            out[k] = len(load_cases(path=path, kind=k))
        except Exception as e:
            out[k] = f"ERR {type(e).__name__}: {e}"
    return out


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "testnet_corpus.jsonl")
    rc = raw_counts(path)
    dc = distinct(path)
    print(f"corpus: {path}")
    print(f"{'task type':32s} {'bundles':>8s} {'distinct(load_cases)':>22s}")
    for k in KINDS:
        print(f"{k:32s} {rc.get(k, 0):8d} {str(dc[k]):>22s}")
    other = {k: v for k, v in rc.items() if k not in KINDS}
    if other:
        print("unknown types in file:", other)
    print(f"{'TOTAL':32s} {sum(rc.values()):8d} "
          f"{sum(v for v in dc.values() if isinstance(v, int)):22d}")
    print(json.dumps({"path": path, "bundles": dict(rc), "distinct": dc}))


if __name__ == "__main__":
    main()
