#!/usr/bin/env python3
"""Collect our own validator scores into a local log.

A miner never learns what it scored: the validator computes the score after the
synapse closes and writes it only to its own W&B run. So the only way to build a
score history is to poll the validators' public project and pull the rows that
carry our UID.

Runs as a service alongside the miner and appends to a JSONL file, deduplicated,
so it can be restarted freely.

    python scripts/sn33_score_poller.py --hotkey 5F... --interval 900

Matching caveat, stated because it limits later analysis: the validator masks
`guid`/`bundle_guid` before sending a task, so the miner never sees the task id
that W&B logs. Scores therefore cannot be joined to requests by id - only by
(validator hotkey, time). `sn33_join_scores.py` does that join and reports how
confident each match is.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, Optional, Set

PROJECT = os.environ.get("SN33_WANDB_PROJECT", "afterparty/conversationgenome")
UID_RE = re.compile(r"^(hotkey|adjusted_score|final_miner_score|task_id)\.(\d+)$")


def load_seen(path: str) -> Set[str]:
    seen: Set[str] = set()
    if not os.path.exists(path):
        return seen
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            k = rec.get("_key")
            if k:
                seen.add(k)
    return seen


def poll_once(hotkeys: Set[str], out_path: str, netuid: int, lookback_h: int, max_rows: int) -> int:
    import wandb
    from datetime import datetime, timedelta, timezone

    api = wandb.Api(timeout=120)
    since = (datetime.now(timezone.utc) - timedelta(hours=lookback_h)).strftime("%Y-%m-%dT%H:%M:%S")
    runs = api.runs(
        PROJECT,
        filters={"createdAt": {"$gte": since}},
        order="-created_at",
        per_page=50,
    )

    seen = load_seen(out_path)
    written = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "a") as out:
        for run in runs:
            if run.config.get("netuid") != netuid:
                continue
            if not any(UID_RE.match(k) for k in run.summary.keys()):
                continue
            rows = 0
            try:
                # Each logged row carries exactly one uid's columns, so the
                # history cannot be filtered by key server-side.
                for row in run.scan_history(page_size=2000):
                    rows += 1
                    if rows > max_rows:
                        break
                    per: Dict[int, dict] = defaultdict(dict)
                    for k, v in row.items():
                        m = UID_RE.match(k)
                        if m and v is not None:
                            per[int(m.group(2))][m.group(1)] = v
                    for uid, rec in per.items():
                        hk = rec.get("hotkey")
                        if hk not in hotkeys:
                            continue
                        key = f"{run.id}:{row.get('_step')}:{uid}"
                        if key in seen:
                            continue
                        seen.add(key)
                        out.write(
                            json.dumps(
                                {
                                    "_key": key,
                                    "wandb_run": run.id,
                                    "validator_hotkey": run.config.get("hotkey"),
                                    "validator_uid": run.config.get("uid"),
                                    "validator_version": run.config.get("version"),
                                    "step": row.get("_step"),
                                    "wandb_timestamp": row.get("_timestamp"),
                                    "miner_uid": uid,
                                    "miner_hotkey": hk,
                                    "task_id": rec.get("task_id"),
                                    "adjusted_score": rec.get("adjusted_score"),
                                    "final_miner_score": rec.get("final_miner_score"),
                                    "collected_at": time.time(),
                                }
                            )
                            + "\n"
                        )
                        written += 1
                out.flush()
            except Exception as e:  # one bad run must not stop the poll
                print(f"  ! {run.id}: {type(e).__name__}: {e}", flush=True)
    return written


def resolve_hotkeys(args) -> Set[str]:
    hotkeys = set(h.strip() for h in (args.hotkey or "").split(",") if h.strip())
    if hotkeys:
        return hotkeys
    # Fall back to every hotkey in the local wallet directory.
    import glob

    root = os.path.expanduser("~/.bittensor/wallets")
    for pub in glob.glob(os.path.join(root, "*", "hotkeys", "*")):
        if pub.endswith("pub.txt"):
            continue
        try:
            hotkeys.add(json.load(open(pub))["ss58Address"])
        except Exception:
            continue
    return hotkeys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotkey", default=os.environ.get("SN33_MINER_HOTKEY", ""),
                    help="comma-separated ss58 addresses; defaults to every local wallet hotkey")
    ap.add_argument("--out", default=os.environ.get("SN33_SCORE_LOG", "/var/log/sn33/scores.jsonl"))
    ap.add_argument("--netuid", type=int, default=int(os.environ.get("NETUID", 33)))
    ap.add_argument("--interval", type=int, default=900, help="seconds between polls; 0 = run once")
    ap.add_argument("--lookback-hours", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=30000)
    args = ap.parse_args()

    hotkeys = resolve_hotkeys(args)
    if not hotkeys:
        sys.exit("no hotkeys to watch: pass --hotkey or create a wallet")
    print(f"watching {len(hotkeys)} hotkey(s) on netuid {args.netuid} -> {args.out}", flush=True)
    for hk in sorted(hotkeys):
        print(f"  {hk}", flush=True)

    while True:
        t0 = time.time()
        try:
            n = poll_once(hotkeys, args.out, args.netuid, args.lookback_hours, args.max_rows)
            print(f"[{time.strftime('%H:%M:%S')}] +{n} score rows ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] poll failed: {type(e).__name__}: {e}", flush=True)
        if args.interval <= 0:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
