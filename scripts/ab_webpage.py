#!/usr/bin/env python
"""Offline A/B: enrichment-first-webpage vs shipped config, on ALL captured
webpage tasks (Apr + May + Aug = 3 miner eras, 137 tasks).

Arm A = production flags (enrichment_first for conversation is irrelevant here;
deep enrichment + theme as in prod), arm B = A + enrichment_first_webpage.
use_cache=True pairs the arms on identical replica draws.

Scoring = per-task mean/top3 cosine of the answer to the ENRICHMENT-line
centroid (window[1:]). NOTE (join audit 2026-08-09): the proxy's mapping to
real validator scores for webpage is NOT validated (the n=14 validation ran on
a type-contaminated join). The paired DELTA is still meaningful - webpage GT is
~79% enrichment-voted by construction (verifier-confirmed code) - but the live
per-type A/B is the final arbiter, not this number.

    venv/bin/python scripts/ab_webpage.py [--n N] [--conc 3]
"""

import argparse
import asyncio
import json
import os
import statistics as st
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = ("/tmp/claude-1000/-home-anirudh-bittensor-conversation-genome-project/"
           "90b878db-c14b-471e-8482-0dddd9d0390f/scratchpad")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from proxy_eval import _load_env_key, score_tasks  # noqa: E402

key = _load_env_key()
if not key:
    sys.exit("no OPENAI_API_KEY")
os.environ.setdefault("OPENAI_API_KEY", key)

from sn33 import pipeline  # noqa: E402
from sn33.pipeline import Config  # noqa: E402

SOURCES = [
    os.path.join(REPO, "data", "tasks.jsonl"),
    os.path.join(REPO, "data", "hetzner_logs", "tasks_v2.jsonl"),
    os.path.join(REPO, "data", "hetzner_logs", "tasks_v2.jsonl.1"),
    os.path.join(REPO, "data", "hetzner_logs", "tasks_v2.jsonl.2"),
    os.path.join(SCRATCH, "srv_tasks_fresh.txt"),
]


def load_tasks():
    out, seen = [], set()
    for path in SOURCES:
        if not os.path.exists(path):
            continue
        for l in open(path, encoding="utf-8", errors="replace"):
            l = l.strip()
            if not l or l == "===GZ===":
                continue
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("task_type") != "webpage_metadata_generation":
                continue
            data = (r.get("task_raw") or {}).get("input", {}).get("data", {})
            win = [tuple(w) for w in (data.get("window") or [])
                   if isinstance(w, (list, tuple)) and len(w) >= 2]
            if len(win) < 2:
                continue          # need doc + at least one enrichment line
            fp = hash(str(win[0][1])[:400])
            if fp in seen:
                continue          # dedupe repeated pages across eras
            seen.add(fp)
            out.append({"window": win,
                        "enrichment": [str(w[1]) for w in win[1:]],
                        "era": os.path.basename(path)})
    return out


async def run_arm(tasks, cfg, conc):
    sem = asyncio.Semaphore(conc)
    out = [None] * len(tasks)

    async def one(i, t):
        async with sem:
            try:
                res = await pipeline.mine("webpage_metadata_generation",
                                          window=t["window"], cfg=cfg)
                out[i] = {"tags": res.tags, "source": res.source,
                          "elapsed": res.elapsed}
            except Exception as e:  # noqa: BLE001
                out[i] = {"tags": [], "source": f"error:{e}", "elapsed": 0.0}

    await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--conc", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "ab_webpage.json"))
    args = ap.parse_args()

    tasks = load_tasks()
    if args.n:
        tasks = tasks[: args.n]
    print(f"webpage tasks (deduped, all eras): {len(tasks)}", flush=True)

    prod = dict(use_cache=True, use_deep_enrichment=True, use_theme_tags=True,
                enrichment_first=True, ner_combos=True)
    cfg_a = Config(**prod)
    cfg_b = Config(**prod, enrichment_first_webpage=True)

    t0 = time.perf_counter()
    arm_a = asyncio.run(run_arm(tasks, cfg_a, args.conc))
    print(f"arm A done {time.perf_counter()-t0:.0f}s "
          f"sources={ {s: sum(1 for r in arm_a if r['source']==s) for s in set(r['source'] for r in arm_a)} }",
          flush=True)
    t1 = time.perf_counter()
    arm_b = asyncio.run(run_arm(tasks, cfg_b, args.conc))
    print(f"arm B done {time.perf_counter()-t1:.0f}s "
          f"sources={ {s: sum(1 for r in arm_b if r['source']==s) for s in set(r['source'] for r in arm_b)} }",
          flush=True)

    rows_a = score_tasks([{"enrichment": t["enrichment"], "our_tags": a["tags"], "final": None}
                          for t, a in zip(tasks, arm_a)])
    rows_b = score_tasks([{"enrichment": t["enrichment"], "our_tags": b["tags"], "final": None}
                          for t, b in zip(tasks, arm_b)])

    paired = []
    for t, ra, rb, aa, bb in zip(tasks, rows_a, rows_b, arm_a, arm_b):
        if ra.get("proxy_mean") is None or rb.get("proxy_mean") is None:
            continue
        if not aa["tags"] or not bb["tags"]:
            continue
        paired.append({"era": t["era"],
                       "a_mean": ra["proxy_mean"], "b_mean": rb["proxy_mean"],
                       "d_mean": rb["proxy_mean"] - ra["proxy_mean"],
                       "a_top3": ra["proxy_top3"], "b_top3": rb["proxy_top3"],
                       "d_top3": rb["proxy_top3"] - ra["proxy_top3"],
                       "a_elapsed": aa["elapsed"], "b_elapsed": bb["elapsed"],
                       "a_tags": aa["tags"], "b_tags": bb["tags"]})

    d = [p["d_mean"] for p in paired]
    d3 = [p["d_top3"] for p in paired]
    print("\n================ RESULT (paired, join-free) ================")
    print(f"paired: {len(paired)}  tags changed on: {sum(1 for p in paired if p['a_tags']!=p['b_tags'])}")
    if d:
        print(f"proxy_mean : A {st.mean(p['a_mean'] for p in paired):.4f}"
              f" -> B {st.mean(p['b_mean'] for p in paired):.4f}"
              f"  delta {st.mean(d):+.4f}  median {st.median(d):+.4f}"
              f"  W/L {sum(1 for x in d if x>0)}/{sum(1 for x in d if x<0)}")
        print(f"proxy_top3 : delta {st.mean(d3):+.4f}  W/L {sum(1 for x in d3 if x>0)}/{sum(1 for x in d3 if x<0)}")
        by_era = {}
        for p in paired:
            by_era.setdefault(p["era"], []).append(p["d_mean"])
        for era, v in by_era.items():
            print(f"  {era:<24} n={len(v):>3}  delta {st.mean(v):+.4f}")
        print(f"elapsed    : A {st.mean(p['a_elapsed'] for p in paired):.2f}s"
              f"  B {st.mean(p['b_elapsed'] for p in paired):.2f}s (B cache-warm)")
    json.dump({"paired": paired}, open(args.out, "w"), indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
