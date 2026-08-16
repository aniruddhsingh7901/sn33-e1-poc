#!/usr/bin/env python3
"""One-line health + score summary for the running miner.

Written for unattended watching: prints a single line so it can be polled on a
timer without drowning the reader, and never touches the miner.

    python3 scripts/sn33_status.py
    SN33 21:45 tasks=37 [conversation_tagging=24 skill_generation=8 ...] \
        errors=0 over10s=1 ranked=35 local=2 | scores=19 mean=0.6104 zeros=0
"""

from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import time

TASKS = os.environ.get("MINER_TASK_LOG", "/var/log/sn33/tasks.jsonl")
SCORES = os.environ.get("SN33_SCORE_LOG", "/var/log/sn33/scores.jsonl")
MINER_LOG = "/var/log/sn33/miner.log"


def load(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def main() -> None:
    tasks = load(TASKS)
    scores = load(SCORES)

    by_type = collections.Counter(t.get("task_type") for t in tasks)
    errors = sum(1 for t in tasks if t.get("error"))
    # The validator's synapse timeout is 12s (bittensor Dendrite default) and it
    # covers network transport in both directions, not just our processing. With
    # the deadline at 11s the pipeline should never exceed ~11.1s; anything above
    # 11.5s is eating the transport margin and risks being scored as no answer.
    durations = [t.get("duration_sec") or 0 for t in tasks]
    slow = sum(1 for d in durations if d > 11.5)
    near = sum(1 for d in durations if 10.5 < d <= 11.5)
    worst = max(durations) if durations else 0.0
    empty = sum(1 for t in tasks if not ((t.get("result") or {}).get("tags")))

    # Which stage produced each answer - the signal that says whether the LLM
    # path is healthy or we are limping along on the local fallback.
    sources = collections.Counter()
    if os.path.exists(MINER_LOG):
        with open(MINER_LOG, errors="replace") as f:
            for m in re.finditer(r"source=(\w+)", f.read()):
                sources[m.group(1)] += 1

    finals = [s["final_miner_score"] for s in scores if s.get("final_miner_score") is not None]
    zeros = sum(1 for f in finals if not f)
    mean = sum(finals) / len(finals) if finals else 0.0

    mix = " ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
    src = " ".join(f"{k}={v}" for k, v in sorted(sources.items()))
    print(
        f"SN33 {time.strftime('%H:%M')} tasks={len(tasks)} [{mix}] "
        f"errors={errors} empty={empty} over11.5s={slow} near={near} max={worst:.1f}s src[{src}] | "
        f"scores={len(finals)} mean={mean:.4f} zeros={zeros}"
    )

    if subprocess.run(["pgrep", "-f", "neurons/miner.py"], capture_output=True).returncode != 0:
        print("ALERT: miner process is not running")
    if slow:
        print(f"ALERT: {slow} task(s) exceeded 11.5s - at risk of the 12s synapse timeout")
    if near:
        print(f"WARN: {near} task(s) in 10.5-11.5s - transport margin is thin")
    if sources.get("local"):
        print(f"WARN: {sources['local']} task(s) fell back to local extraction (LLM path failed)")
    # A cut-off replica measured 0.1994 against 0.5517 for a completed one, so
    # this is the most expensive degradation available short of a zero.
    if sources.get("pool"):
        rate = 100 * sources["pool"] / max(1, sum(sources.values()))
        print(f"WARN: {sources['pool']} task(s) ({rate:.0f}%) ran without the replica "
              f"(source=pool) - those score ~0.20 vs ~0.55")


if __name__ == "__main__":
    main()
