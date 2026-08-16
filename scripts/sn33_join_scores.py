#!/usr/bin/env python3
"""Join our request log to the scores the validators recorded.

Produces the dataset the whole optimization loop needs: for each task we
answered, what we replied and what it scored.

The join is by (validator hotkey, time), not by id: `mask_task_for_miner`
replaces `guid`, `bundle_guid` and `input.guid` with "HIDDEN" before the task
reaches us, so the miner never sees the task id that W&B logs. Every row
therefore carries `match_confidence` and `lag_s`, and analysis should filter on
them rather than trusting the join blindly.

    python scripts/sn33_join_scores.py --since 24h --out data/scored_tasks.jsonl
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import statistics
import sys
from collections import defaultdict


def read_jsonl(path: str):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def parse_since(s: str) -> float:
    import time

    if not s:
        return 0.0
    unit = s[-1]
    mult = {"h": 3600, "d": 86400, "m": 60}.get(unit)
    if mult is None:
        return float(s)
    return time.time() - float(s[:-1]) * mult


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.environ.get("MINER_TASK_LOG", "/var/log/sn33/tasks.jsonl"))
    ap.add_argument("--scores", default=os.environ.get("SN33_SCORE_LOG", "/var/log/sn33/scores.jsonl"))
    ap.add_argument("--out", default="data/scored_tasks.jsonl")
    ap.add_argument("--since", default="", help="e.g. 24h, 7d")
    ap.add_argument("--max-lag", type=float, default=180.0,
                    help="seconds after a request within which its score must appear")
    args = ap.parse_args()

    cutoff = parse_since(args.since)
    tasks = [t for t in read_jsonl(args.tasks) if t.get("timestamp", 0) >= cutoff]
    scores = [s for s in read_jsonl(args.scores) if s.get("wandb_timestamp", 0) >= cutoff]
    if not tasks:
        sys.exit(f"no tasks in {args.tasks}")
    if not scores:
        sys.exit(f"no scores in {args.scores} - is the poller running?")

    # Index scores by validator hotkey, sorted by time.
    by_val = defaultdict(list)
    for s in scores:
        by_val[s.get("validator_hotkey")].append(s)
    for v in by_val.values():
        v.sort(key=lambda r: r.get("wandb_timestamp") or 0)

    used = set()
    joined, unmatched = [], 0
    for t in tasks:
        vh = t.get("validator_hotkey")
        cand = by_val.get(vh) or []
        times = [c.get("wandb_timestamp") or 0 for c in cand]
        t0 = t.get("timestamp", 0)
        # The score is written after we answer, so look forward only.
        i = bisect.bisect_left(times, t0)
        best, best_lag = None, None
        for j in range(i, min(i + 6, len(cand))):
            key = cand[j].get("_key")
            if key in used:
                continue
            lag = times[j] - t0
            if 0 <= lag <= args.max_lag:
                best, best_lag = cand[j], lag
                break
        if best is None:
            unmatched += 1
            continue
        used.add(best["_key"])

        # Confidence: how many other scores from the same validator fell inside
        # the same window. One candidate is unambiguous; several is a guess.
        rivals = sum(1 for j in range(i, min(i + 6, len(cand)))
                     if 0 <= times[j] - t0 <= args.max_lag)
        joined.append(
            {
                "timestamp": t0,
                "task_type": t.get("task_type"),
                "duration_sec": t.get("duration_sec"),
                "validator_hotkey": vh,
                "miner_uid": best.get("miner_uid"),
                "tags": (t.get("result") or {}).get("tags"),
                "n_tags": len((t.get("result") or {}).get("tags") or []),
                "error": t.get("error"),
                "final_miner_score": best.get("final_miner_score"),
                "adjusted_score": best.get("adjusted_score"),
                "task_id": best.get("task_id"),
                "lag_s": round(best_lag, 1),
                "match_confidence": "high" if rivals == 1 else f"ambiguous({rivals})",
                "task_raw": t.get("task_raw"),
            }
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for row in joined:
            f.write(json.dumps(row, default=str) + "\n")

    print(f"tasks={len(tasks)} scores={len(scores)} joined={len(joined)} unmatched={unmatched}")
    print(f"wrote {args.out}")

    high = [r for r in joined if r["match_confidence"] == "high"]
    print(f"\nhigh-confidence matches: {len(high)}/{len(joined)}")
    if high:
        by_type = defaultdict(list)
        for r in high:
            by_type[r["task_type"]].append(r["final_miner_score"])
        print(f"\n{'task type':32s} {'n':>4s} {'mean':>7s} {'median':>7s} {'zero%':>6s}")
        print("-" * 62)
        for k, v in sorted(by_type.items()):
            zeros = sum(1 for x in v if not x)
            print(f"{k:32s} {len(v):4d} {statistics.mean(v):7.4f} "
                  f"{statistics.median(v):7.4f} {zeros/len(v)*100:5.1f}%")
        allv = [r["final_miner_score"] for r in high]
        print(f"\nOVERALL n={len(allv)} mean={statistics.mean(allv):.4f} median={statistics.median(allv):.4f}")
        print("Compare against docs/SN33_RESULTS.md predictions per task type.")


if __name__ == "__main__":
    main()
