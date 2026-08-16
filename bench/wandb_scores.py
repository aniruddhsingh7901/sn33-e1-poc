#!/usr/bin/env python3
"""Pull real per-miner scores from the validators' public W&B project.

Validators log every scored response to `afterparty/conversationgenome` as
sparse per-UID columns (neurons/validator.py -> WandbLib):

    hotkey.<uid>  adjusted_score.<uid>  final_miner_score.<uid>  task_id.<uid>

That is the only public view of what miners actually score in production - the
subnet exposes no per-tag detail and never the ground truth. It gives us two
things the offline bench cannot:

  * the real score distribution, so "top miner" is a number rather than a guess
  * a calibration point for our own replica scores

Caveat that bounds everything below: **the task type is not logged** (a known
gap, acknowledged in Discord 2026-07-11). So these are scores pooled across all
task types in whatever mix the validator happened to send. A miner's mean here
is not comparable to a single-task bench mean unless you weight by traffic mix.

    python bench/wandb_scores.py --runs 6 --out data/wandb_scores.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT = "afterparty/conversationgenome"
UID_RE = re.compile(r"^(hotkey|adjusted_score|final_miner_score|task_id)\.(\d+)$")


def collect(runs_wanted: int, netuid: int, max_rows: int):
    import wandb

    api = wandb.Api(timeout=120)
    runs = api.runs(PROJECT, order="-created_at", per_page=50)

    # (hotkey, uid) -> list of scores
    scores = defaultdict(list)
    adjusted = defaultdict(list)
    seen_runs = []

    for run in runs:
        if len(seen_runs) >= runs_wanted:
            break
        if run.config.get("netuid") != netuid:
            continue
        keys = [k for k in run.summary.keys() if UID_RE.match(k)]
        if not keys:
            continue
        seen_runs.append((run.id, run.name, run.created_at, run.state))
        print(f"  scanning {run.id} {run.name[:40]} ({run.state}) ...", flush=True)

        rows = 0
        try:
            # scan_history(keys=[...]) drops any row that lacks one of the keys,
            # and each logged row carries exactly ONE uid's four columns - so an
            # explicit key list matches nothing. Scan unfiltered and pick the
            # uid columns out of each row instead.
            for row in run.scan_history(page_size=2000):
                rows += 1
                if rows > max_rows:
                    break
                per_uid = defaultdict(dict)
                for k, v in row.items():
                    m = UID_RE.match(k)
                    if not m or v is None:
                        continue
                    per_uid[int(m.group(2))][m.group(1)] = v
                for uid, rec in per_uid.items():
                    hk = rec.get("hotkey")
                    fms = rec.get("final_miner_score")
                    adj = rec.get("adjusted_score")
                    if hk is None or fms is None:
                        continue
                    scores[(hk, uid)].append(float(fms))
                    if adj is not None:
                        adjusted[(hk, uid)].append(float(adj))
        except Exception as e:  # a single unreadable run must not kill the sweep
            print(f"    ! {type(e).__name__}: {e}")
        print(f"    rows={rows} miners so far={len(scores)}")

    return scores, adjusted, seen_runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--netuid", type=int, default=33)
    ap.add_argument("--max-rows", type=int, default=40000)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--out", default="data/wandb_scores.json")
    ap.add_argument("--raw", default=None, help="also dump every observation for distribution analysis")
    args = ap.parse_args()

    print(f"fetching up to {args.runs} netuid-{args.netuid} validator runs from {PROJECT}")
    scores, adjusted, runs = collect(args.runs, args.netuid, args.max_rows)
    if not scores:
        raise SystemExit("no scores found")

    table = []
    for (hk, uid), vals in scores.items():
        if len(vals) < args.min_samples:
            continue
        nonzero = [v for v in vals if v > 0]
        table.append(
            {
                "hotkey": hk,
                "uid": uid,
                "n": len(vals),
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
                "mean_nonzero": statistics.mean(nonzero) if nonzero else 0.0,
                "max": max(vals),
                "p90": sorted(vals)[int(len(vals) * 0.9)],
                "zero_rate": sum(1 for v in vals if v <= 0) / len(vals),
            }
        )
    table.sort(key=lambda r: -r["mean"])

    print(f"\nruns scanned: {len(runs)}   miners with >={args.min_samples} samples: {len(table)}")
    hdr = f"{'rank':>4s} {'uid':>4s} {'hotkey':14s} {'n':>5s} {'mean':>7s} {'median':>7s} {'nonzero':>8s} {'p90':>6s} {'max':>6s} {'zero%':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(table[:25], 1):
        print(
            f"{i:4d} {r['uid']:4d} {r['hotkey'][:14]:14s} {r['n']:5d} {r['mean']:7.4f} "
            f"{r['median']:7.4f} {r['mean_nonzero']:8.4f} {r['p90']:6.3f} {r['max']:6.3f} {r['zero_rate']*100:5.1f}%"
        )

    allv = [v for vals in scores.values() for v in vals]
    nz = [v for v in allv if v > 0]
    means = [r["mean"] for r in table]
    print(f"\nobservations: {len(allv)}  ({len(allv)-len(nz)} zeros = {(1-len(nz)/len(allv))*100:.1f}%)")
    print(f"all scored responses : mean={statistics.mean(allv):.4f} median={statistics.median(allv):.4f}")
    print(f"nonzero responses    : mean={statistics.mean(nz):.4f} median={statistics.median(nz):.4f} p90={sorted(nz)[int(len(nz)*.9)]:.4f} max={max(nz):.4f}")
    print(f"per-miner mean score : best={means[0]:.4f} p90={sorted(means)[int(len(means)*.9)]:.4f} median={statistics.median(means):.4f}")

    if args.raw:
        with open(args.raw, "w") as f:
            for (hk, uid), vals in scores.items():
                f.write(json.dumps({"hotkey": hk, "uid": uid, "scores": vals}) + "\n")
        print(f"wrote {args.raw}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"runs": runs, "miners": table}, f, indent=1, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
