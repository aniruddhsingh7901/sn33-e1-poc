#!/usr/bin/env python
"""4-arm offline A/B on captured conversation tasks:

  A base        = production (enrichment_first + deep + theme)
  B +quota      = A + enrichment_line_quota=2
  C +cond       = A + demote_conditional
  D +both       = A + quota + conditional

All arms use_cache=True over the same tasks -> identical replica/pool/deep
draws; the arms differ ONLY in selection logic (pure CPU), so deltas are
exactly attributable. Proxy = enrichment-centroid (validated for conversation
at Spearman 0.64 vs real finals).

    venv/bin/python scripts/ab_quota.py [--n N] [--conc 3]
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
    sys.exit("no OPENAI_API_KEY")
os.environ.setdefault("OPENAI_API_KEY", key)

from sn33 import pipeline  # noqa: E402
from sn33.pipeline import Config  # noqa: E402

PROD = dict(use_cache=True, use_deep_enrichment=True, use_theme_tags=True,
            enrichment_first=True, ner_combos=True)
ARMS = {
    "A_base":  Config(**PROD),
    "B_quota": Config(**PROD, enrichment_line_quota=2),
    "C_cond":  Config(**PROD, demote_conditional=True),
    "D_both":  Config(**PROD, enrichment_line_quota=2, demote_conditional=True),
}


async def run_arm(tasks, cfg, conc):
    sem = asyncio.Semaphore(conc)
    out = [None] * len(tasks)

    async def one(i, t):
        async with sem:
            window = [(j, l) for j, l in enumerate(t["window"])]
            try:
                res = await pipeline.mine("conversation_tagging", window=window,
                                          enrichment=t["enrichment"], cfg=cfg)
                out[i] = {"tags": res.tags, "source": res.source, "elapsed": res.elapsed}
            except Exception as e:  # noqa: BLE001
                out[i] = {"tags": [], "source": f"error:{e}", "elapsed": 0.0}

    await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--conc", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "ab_quota.json"))
    args = ap.parse_args()

    tasks = [t for t in json.load(open(os.path.join(REPO, "data", "conv_24h_full.json")))
             if t.get("enrichment")]
    if args.n:
        tasks = tasks[: args.n]
    print(f"conversation tasks: {len(tasks)}", flush=True)

    results = {}
    for name, cfg in ARMS.items():
        t0 = time.perf_counter()
        arm = asyncio.run(run_arm(tasks, cfg, args.conc))
        rows = score_tasks([{"enrichment": t["enrichment"], "our_tags": a["tags"], "final": None}
                            for t, a in zip(tasks, arm)])
        results[name] = (arm, rows)
        ok = sum(1 for a in arm if a["tags"])
        print(f"{name}: done {time.perf_counter()-t0:.0f}s  answered {ok}/{len(tasks)}", flush=True)

    base_arm, base_rows = results["A_base"]
    print("\n================ RESULT (paired vs A_base) ================")
    summary = {}
    for name in ["B_quota", "C_cond", "D_both"]:
        arm, rows = results[name]
        d = []
        changed = 0
        for ta, ra, tb, rb in zip(base_arm, base_rows, arm, rows):
            if not ta["tags"] or not tb["tags"]:
                continue
            if ra.get("proxy_mean") is None or rb.get("proxy_mean") is None:
                continue
            d.append(rb["proxy_mean"] - ra["proxy_mean"])
            if ta["tags"] != tb["tags"]:
                changed += 1
        wl = (sum(1 for x in d if x > 0), sum(1 for x in d if x < 0))
        summary[name] = dict(n=len(d), delta=st.mean(d) if d else None,
                             median=st.median(d) if d else None, wl=wl, changed=changed)
        print(f"{name}: n={len(d)} changed={changed}  delta {st.mean(d):+.4f}"
              f"  median {st.median(d):+.4f}  W/L {wl[0]}/{wl[1]}")

    json.dump({"summary": summary}, open(args.out, "w"), indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
