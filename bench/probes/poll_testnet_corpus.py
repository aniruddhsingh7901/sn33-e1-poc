#!/usr/bin/env python3
"""Poll the testnet reserve API over several minutes looking for NEW sources.

The pool reissues the same conversations under fresh guids, so a single sweep
undercounts what is reachable if the pool rotates on a timer. This does R
rounds of N reserves with a gap between rounds, dedupes on CONTENT using the
fetcher's own content_key, and appends only genuinely new bundles.

    python bench/probes/poll_testnet_corpus.py --out X.jsonl --rounds 12 \
        --per-round 12 --gap 45
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from bench.fetch_testnet_corpus import api_key, content_key, describe, reserve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--per-round", type=int, default=12)
    ap.add_argument("--gap", type=float, default=45.0, help="seconds between rounds")
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    key = api_key()
    seen = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    seen.add(content_key(json.loads(line)))
                except Exception:
                    continue
    print(f"start: {len(seen)} distinct sources already held", flush=True)

    guids = set()
    served = Counter()
    new_by_type = Counter()
    rounds_log = []
    t0 = time.time()

    with open(args.out, "a") as out:
        for r in range(args.rounds):
            got = 0
            for _ in range(args.per_round):
                try:
                    b = reserve(key)
                except urllib.error.HTTPError as e:
                    print(f"  HTTP {e.code}: {e.read()[:120]!r}", flush=True)
                    time.sleep(3)
                    continue
                except Exception as e:
                    print(f"  {type(e).__name__}: {str(e)[:90]}", flush=True)
                    time.sleep(3)
                    continue
                d = describe(b)
                served[d["type"]] += 1
                guids.add(d["guid"])
                ck = content_key(b)
                if ck in seen:
                    time.sleep(args.sleep)
                    continue
                seen.add(ck)
                out.write(json.dumps(b) + "\n")
                out.flush()
                got += 1
                new_by_type[d["type"]] += 1
                print(f"  NEW round={r} {d['type']:28s} lines={d['lines']:4d} "
                      f"enrichment={d['enrichment_results']} guid={d['guid']}", flush=True)
                time.sleep(args.sleep)
            rounds_log.append({"round": r, "t": round(time.time() - t0, 1), "new": got})
            print(f"round {r}: served={sum(served.values())} new_this_round={got} "
                  f"distinct_total={len(seen)} elapsed={time.time()-t0:.0f}s", flush=True)
            if r < args.rounds - 1:
                time.sleep(args.gap)

    summary = {
        "served_bundles": dict(served),
        "served_total": sum(served.values()),
        "distinct_guids_served": len(guids),
        "new_distinct_by_type": dict(new_by_type),
        "new_distinct_total": sum(new_by_type.values()),
        "distinct_sources_held": len(seen),
        "rounds": rounds_log,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    if args.log:
        with open(args.log, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
