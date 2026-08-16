#!/usr/bin/env python
"""Offline A/B: enrichment-first aiming vs shipped config, on captured tasks.

Arm A = shipped Config (enrichment_first False), arm B = enrichment_first True.
Both run with use_cache=True so every prompt they share (replica doc call,
per-line enrichment calls, combine) returns the SAME cached completion - the
comparison is paired on identical GT draws, and only the aiming changes.
(Arm B's pool call includes enrichment context, so that one call diverges by
design.)

Scoring = scripts/proxy_eval.py: cosine of each tag to the enrichment-line
centroid, validated against real validator scores at Spearman 0.64 (n=89,
2026-08-09, data/proxy_baseline.json).

    venv/bin/python scripts/ab_enrichment_first.py [--n N] [--conc 3] \
        [--out data/ab_enrichment_first.json]
"""

import argparse
import asyncio
import json
import os
import statistics as st
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from proxy_eval import _load_env_key, score_tasks  # noqa: E402

key = _load_env_key()
if not key:
    sys.exit("no OPENAI_API_KEY in env or .env")
os.environ.setdefault("OPENAI_API_KEY", key)

from sn33 import pipeline  # noqa: E402  (after key so llm client sees it)
from sn33.pipeline import Config  # noqa: E402


async def run_arm(tasks, cfg, conc):
    sem = asyncio.Semaphore(conc)
    out = [None] * len(tasks)

    async def one(i, t):
        async with sem:
            window = [(j, l) for j, l in enumerate(t["window"])]
            try:
                res = await pipeline.mine(
                    "conversation_tagging",
                    window=window,
                    enrichment=t["enrichment"],
                    cfg=cfg,
                )
                out[i] = {"tags": res.tags, "source": res.source,
                          "elapsed": res.elapsed, "n_tags": len(res.tags)}
            except Exception as e:  # noqa: BLE001 - record, don't die
                out[i] = {"tags": [], "source": f"error:{e}", "elapsed": 0.0,
                          "n_tags": 0}

    await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(REPO, "data", "conv_24h_full.json"))
    ap.add_argument("--n", type=int, default=0, help="limit task count (0 = all)")
    ap.add_argument("--conc", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "ab_enrichment_first.json"))
    ap.add_argument("--prod", action="store_true",
                    help="both arms use the PRODUCTION flag set (deep enrichment + "
                         "theme tags ON) so the only variable is enrichment_first")
    args = ap.parse_args()

    tasks = [t for t in json.load(open(args.tasks)) if t.get("enrichment")]
    if args.n:
        tasks = tasks[: args.n]
    print(f"tasks: {len(tasks)} (scored: {sum(1 for t in tasks if t.get('final') is not None)})",
          flush=True)

    extra = dict(use_deep_enrichment=True, use_theme_tags=True) if args.prod else {}
    cfg_a = Config(use_cache=True, **extra)
    cfg_b = Config(use_cache=True, enrichment_first=True, **extra)

    t0 = time.perf_counter()
    arm_a = asyncio.run(run_arm(tasks, cfg_a, args.conc))
    print(f"arm A done in {time.perf_counter()-t0:.0f}s "
          f"(sources: { {s: sum(1 for r in arm_a if r['source']==s) for s in set(r['source'] for r in arm_a)} })",
          flush=True)
    t1 = time.perf_counter()
    arm_b = asyncio.run(run_arm(tasks, cfg_b, args.conc))
    print(f"arm B done in {time.perf_counter()-t1:.0f}s "
          f"(sources: { {s: sum(1 for r in arm_b if r['source']==s) for s in set(r['source'] for r in arm_b)} })",
          flush=True)

    # proxy-score both arms on the SAME enrichment targets
    rows_a = score_tasks([{"enrichment": t["enrichment"], "our_tags": a["tags"],
                           "final": t.get("final"), "dt": t.get("dt")}
                          for t, a in zip(tasks, arm_a)])
    rows_b = score_tasks([{"enrichment": t["enrichment"], "our_tags": b["tags"],
                           "final": t.get("final"), "dt": t.get("dt")}
                          for t, b in zip(tasks, arm_b)])

    # production baseline for the same dts (sanity: arm A should sit near it)
    base_path = os.path.join(REPO, "data", "proxy_baseline.json")
    base = {}
    if os.path.exists(base_path):
        base = {r["dt"]: r for r in json.load(open(base_path)).get("tasks", [])
                if r.get("dt")}

    paired = []
    for t, ra, rb, aa, bb in zip(tasks, rows_a, rows_b, arm_a, arm_b):
        if ra.get("proxy_mean") is None or rb.get("proxy_mean") is None:
            continue
        if not aa["tags"] or not bb["tags"]:
            continue
        paired.append({
            "dt": t.get("dt"), "final": t.get("final"),
            "a_mean": ra["proxy_mean"], "b_mean": rb["proxy_mean"],
            "a_top3": ra["proxy_top3"], "b_top3": rb["proxy_top3"],
            "d_mean": rb["proxy_mean"] - ra["proxy_mean"],
            "d_top3": rb["proxy_top3"] - ra["proxy_top3"],
            "prod_mean": base.get(t.get("dt"), {}).get("proxy_mean"),
            "a_elapsed": aa["elapsed"], "b_elapsed": bb["elapsed"],
            "a_tags": aa["tags"], "b_tags": bb["tags"],
        })

    d_mean = [p["d_mean"] for p in paired]
    d_top3 = [p["d_top3"] for p in paired]
    wl_mean = (sum(1 for d in d_mean if d > 0), sum(1 for d in d_mean if d < 0))
    wl_top3 = (sum(1 for d in d_top3 if d > 0), sum(1 for d in d_top3 if d < 0))
    changed = sum(1 for p in paired if p["a_tags"] != p["b_tags"])

    print("\n================ RESULT ================")
    print(f"paired tasks: {len(paired)}   tags changed on: {changed}")
    print(f"proxy_mean : A {st.mean(p['a_mean'] for p in paired):.4f}"
          f" -> B {st.mean(p['b_mean'] for p in paired):.4f}"
          f"   delta {st.mean(d_mean):+.4f}  median {st.median(d_mean):+.4f}  W/L {wl_mean[0]}/{wl_mean[1]}")
    print(f"proxy_top3 : A {st.mean(p['a_top3'] for p in paired):.4f}"
          f" -> B {st.mean(p['b_top3'] for p in paired):.4f}"
          f"   delta {st.mean(d_top3):+.4f}  median {st.median(d_top3):+.4f}  W/L {wl_top3[0]}/{wl_top3[1]}")
    prod = [p for p in paired if p.get("prod_mean") is not None]
    if prod:
        print(f"sanity     : production proxy_mean {st.mean(p['prod_mean'] for p in prod):.4f}"
              f" vs arm A {st.mean(p['a_mean'] for p in prod):.4f} (should be close)")
    print(f"elapsed    : A mean {st.mean(p['a_elapsed'] for p in paired):.2f}s"
          f"  B mean {st.mean(p['b_elapsed'] for p in paired):.2f}s"
          f"  (B partly cache-warmed; not a latency measurement)")

    json.dump({"paired": paired,
               "summary": {"n": len(paired), "delta_mean": st.mean(d_mean) if d_mean else None,
                            "delta_top3": st.mean(d_top3) if d_top3 else None,
                            "wl_mean": wl_mean, "wl_top3": wl_top3}},
              open(args.out, "w"), indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
