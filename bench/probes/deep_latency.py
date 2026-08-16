#!/usr/bin/env python3
"""Latency tail of the deep-enrichment feature, at the production deadline.

The gain was probed with deadline_s=600. Production runs 11s inside a 12s
synapse, and a replica that gets cut off scores ~0.1994 against ~0.5517 - so a
few percentage points of extra truncation erase a +0.024 gain.

Two modes:

  --mode ab     end-to-end pipeline.mine over N windows, feature OFF and ON,
                UNCACHED, serial, alternating which arm runs first so API drift
                cancels. Reports the full distribution and the source mix.

  --mode gate   the gate itself: deep calls made to hang / fail / return late,
                with every other call served from cache so the answer is
                deterministic and any difference is attributable.

Instrumentation notes:
  * `prompts.gt_combine` is called exactly once per replica, immediately before
    the combine await, so spying on it gives the combine START offset - which is
    what the deep grace actually delays.
  * `replica.replicate` is wrapped to capture the Replica object, so deep_tags
    can be counted without changing pipeline.Result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import expand_windows, load_cases
from sn33 import extract, llm, pipeline, prompts, replica

# ---------------------------------------------------------------- instrumentation

_SPY: Dict[str, object] = {}
_orig_gt_combine = prompts.gt_combine
_orig_replicate = replica.replicate


def _spy_gt_combine(sets):
    _SPY["combine_start"] = time.perf_counter()
    return _orig_gt_combine(sets)


_IMPL: Dict[str, object] = {"fn": None}      # None -> production replicate


async def _spy_replicate(*a, **kw):
    _SPY["replica_start"] = time.perf_counter()
    fn = _IMPL["fn"] or _orig_replicate
    rep = await fn(*a, **kw)
    _SPY["replica_end"] = time.perf_counter()
    _SPY["rep"] = rep
    return rep


prompts.gt_combine = _spy_gt_combine
replica.replicate = _spy_replicate
pipeline.prompts.gt_combine = _spy_gt_combine


# ---------------------------------------------------------------- deep-call injection

_orig_chat = llm.chat
_INJECT: Dict[str, object] = {"mode": "none"}


async def _patched_chat(prompt, model, timeout=8.0, temperature=0.0, use_cache=False,
                        max_completion_tokens=None, attempts=1, salt=""):
    """Only the deep calls carry salt='deep', so they are addressable."""
    mode = _INJECT["mode"]
    if salt == "deep" and mode != "none":
        if mode == "fail":
            return None
        if mode == "hang":
            await asyncio.sleep(3600)          # cancelled or graced out, never returns
            return None
        if mode == "slow":
            await asyncio.sleep(float(_INJECT.get("delay", 5.0)))
            return await _orig_chat(prompt, model, timeout, temperature, use_cache,
                                    max_completion_tokens, attempts, salt)
    return await _orig_chat(prompt, model, timeout, temperature, use_cache,
                            max_completion_tokens, attempts, salt)


llm.chat = _patched_chat
replica.llm.chat = _patched_chat
pipeline.llm.chat = _patched_chat


# ---------------------------------------------------------------- helpers

def pct(values: List[float], p: float) -> float:
    """Nearest-rank percentile. n=24 makes p99 the max; that is honest, not a bug."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p * len(s) + 0.5)) - 1))
    return s[k]


async def run_one(case, window, cfg) -> dict:
    _SPY.clear()
    t0 = time.perf_counter()
    res = await pipeline.mine(
        "conversation_tagging",
        window=window,
        enrichment=case.enrichment_lines,
        cfg=cfg,
    )
    wall = time.perf_counter() - t0

    rep = _SPY.get("rep")
    tm = dict(res.timings)
    enrich = [v for k, v in tm.items() if k.startswith("enrich")]
    deep = [v for k, v in tm.items() if k.startswith("deep")]
    fanout = max([tm.get("doc", 0.0)] + enrich) if tm else 0.0
    combine_off = (
        _SPY["combine_start"] - _SPY["replica_start"]
        if "combine_start" in _SPY and "replica_start" in _SPY else None
    )
    return {
        "guid": case.guid[:10],
        "enrich_lines": len(case.enrichment_lines),
        "wall": wall,
        "elapsed": res.elapsed,
        "source": res.source,
        "degraded": res.degraded,
        "n_tags": len(res.tags),
        "tags": list(res.tags),
        "n_cand": len(res.candidates),
        "n_deep": len(getattr(rep, "deep_tags", []) or []) if rep is not None else 0,
        "rep_wall": (_SPY["replica_end"] - _SPY["replica_start"]) if "replica_end" in _SPY else None,
        "fanout": fanout,
        "deep_max": max(deep) if deep else None,
        "combine": tm.get("combine"),
        "combine_start_off": combine_off,
        "n_pending_after": len([t for t in asyncio.all_tasks() if not t.done()]) - 1,
    }


def report(label: str, rows: List[dict], deadline: float) -> None:
    w = [r["wall"] for r in rows]
    print(f"\n{label}  n={len(rows)}")
    print(f"  wall   p50 {pct(w,.50):5.2f}  p90 {pct(w,.90):5.2f}  p95 {pct(w,.95):5.2f} "
          f" p99 {pct(w,.99):5.2f}  max {pct(w,1.0):5.2f}  mean {statistics.mean(w):5.2f}")
    src: Dict[str, int] = {}
    for r in rows:
        src[r["source"]] = src.get(r["source"], 0) + 1
    bad = sum(1 for r in rows if r["source"] != "ranked")
    print(f"  sources {src}   not-ranked {bad}/{len(rows)} = {100*bad/len(rows):.1f}%"
          f"   degraded {sum(1 for r in rows if r['degraded'])}")
    for key in ("fanout", "combine_start_off", "combine", "rep_wall"):
        v = [r[key] for r in rows if r.get(key) is not None]
        if v:
            print(f"  {key:17s} p50 {pct(v,.50):5.2f}  p90 {pct(v,.90):5.2f} "
                  f" p95 {pct(v,.95):5.2f}  max {pct(v,1.0):5.2f}")
    dm = [r["deep_max"] for r in rows if r.get("deep_max") is not None]
    if dm:
        print(f"  {'deep_call':17s} p50 {pct(dm,.50):5.2f}  p90 {pct(dm,.90):5.2f} "
              f" p95 {pct(dm,.95):5.2f}  max {pct(dm,1.0):5.2f}")
    print(f"  headroom to {deadline:.0f}s: min {deadline-max(w):+.2f}s"
          f"   over-deadline {sum(1 for x in w if x > deadline)}"
          f"   over-12s {sum(1 for x in w if x > 12.0)}")
    print(f"  candidates mean {statistics.mean([r['n_cand'] for r in rows]):.1f}"
          f"   deep tags mean {statistics.mean([r['n_deep'] for r in rows]):.1f}"
          f"   tags mean {statistics.mean([r['n_tags'] for r in rows]):.1f}")


# ---------------------------------------------------------------- modes

async def mode_ab(args) -> None:
    sources = load_cases(kind="conversation_tagging", seed=0)
    pairs = expand_windows(sources, seed=0)[: args.n]
    print(f"corpus: {len(sources)} distinct conversations -> {len(pairs)} windows")
    counts: Dict[str, int] = {}
    for c, _ in pairs:
        counts[c.guid[:10]] = counts.get(c.guid[:10], 0) + 1
    print(f"  windows per conversation: {counts}")

    def mkcfg(deep: bool) -> pipeline.Config:
        return pipeline.Config(
            deadline_s=args.deadline, call_timeout_s=6.5,
            use_cache=False, use_local=True,
            use_deep_enrichment=deep,
        )

    from bench.probes.deep_latency_fix import replicate_late
    ARMS = {                       # name -> (deep on?, replicate impl)
        "off": (False, None),
        "on": (True, None),
        "on_late": (True, replicate_late),   # harvest deep AFTER combine
    }
    names = [a.strip() for a in args.arms.split(",") if a.strip()]

    rows: Dict[str, List[dict]] = {n: [] for n in names}
    for i, (case, window) in enumerate(pairs):
        order = names[i % len(names):] + names[: i % len(names)]   # rotate: cancels drift
        for arm in order:
            deep, impl = ARMS[arm]
            _IMPL["fn"] = impl
            r = await run_one(case, window, mkcfg(deep))
            r["arm"], r["i"] = arm, i
            rows[arm].append(r)
        _IMPL["fn"] = None
        line = f"  [{i:2d}] {case.guid[:10]:10s} k={case.enrichment_lines and len(case.enrichment_lines)}  "
        for arm in names:
            r = rows[arm][-1]
            line += f"{arm} {r['wall']:5.2f}s {r['source']:6s} d={r['n_deep']:3d} | "
        print(line)

    for arm in names:
        report(f"=== {arm.upper()} ===", rows[arm], args.deadline)

    base = names[0]
    n = len(rows[base])
    base_bad = sum(1 for r in rows[base] if r["source"] != "ranked")
    for arm in names[1:]:
        d = [rows[arm][i]["wall"] - rows[base][i]["wall"] for i in range(n)]
        print(f"\nPAIRED wall delta ({arm} - {base}): mean {statistics.mean(d):+.2f}s "
              f"p50 {pct(d,.50):+.2f}  p90 {pct(d,.90):+.2f}  p95 {pct(d,.95):+.2f} "
              f" max {pct(d,1.0):+.2f}  slower {sum(1 for x in d if x>0)}/{n}")
        bad = sum(1 for r in rows[arm] if r["source"] != "ranked")
        print(f"INCREMENTAL truncation: {base} {base_bad}/{n} -> {arm} {bad}/{n} = "
              f"{100*(bad-base_bad)/n:+.1f} percentage points")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1, default=str)
        print(f"wrote {args.out}")
    print("usage:", llm.Usage.snapshot())


async def mode_gate(args) -> None:
    """Deterministic: every non-deep call cached, so tags are comparable byte-wise."""
    sources = load_cases(kind="conversation_tagging", seed=0)
    pairs = expand_windows(sources, seed=0)[: args.n]

    def mkcfg(deep: bool) -> pipeline.Config:
        return pipeline.Config(
            deadline_s=args.deadline, call_timeout_s=6.5,
            use_cache=True, use_local=False,
            use_deep_enrichment=deep,
        )

    # warm the cache for the OFF path first
    print("warming cache (OFF arm, cached) ...")
    base: List[dict] = []
    for case, window in pairs:
        base.append(await run_one(case, window, mkcfg(False)))
    base = []
    for case, window in pairs:
        base.append(await run_one(case, window, mkcfg(False)))

    arms = [("fail", 0.0), ("hang", 0.0), ("slow", 5.0), ("slow", 1.0)]
    print(f"\n{'inject':12s} {'same tags as OFF':>17s} {'wall p50':>9s} {'wall max':>9s} "
          f"{'not-ranked':>11s} {'deep tags':>10s} {'leaked tasks':>13s}")
    print("-" * 88)
    print(f"{'OFF (base)':12s} {'-':>17s} {pct([r['wall'] for r in base],.5):9.2f} "
          f"{pct([r['wall'] for r in base],1.0):9.2f} "
          f"{sum(1 for r in base if r['source']!='ranked'):11d} {'0':>10s} "
          f"{max(r['n_pending_after'] for r in base):13d}")

    for mode, delay in arms:
        _INJECT["mode"] = mode
        _INJECT["delay"] = delay
        rows = []
        for case, window in pairs:
            rows.append(await run_one(case, window, mkcfg(True)))
        same = sum(1 for a, b in zip(base, rows) if a["tags"] == b["tags"])
        label = f"{mode}{'' if not delay else f'({delay:g}s)'}"
        print(f"{label:12s} {f'{same}/{len(rows)}':>17s} "
              f"{pct([r['wall'] for r in rows],.5):9.2f} "
              f"{pct([r['wall'] for r in rows],1.0):9.2f} "
              f"{sum(1 for r in rows if r['source']!='ranked'):11d} "
              f"{statistics.mean([r['n_deep'] for r in rows]):10.1f} "
              f"{max(r['n_pending_after'] for r in rows):13d}")
    _INJECT["mode"] = "none"

    # control: injection off, deep really runs (still cached-for-others)
    rows = []
    for case, window in pairs:
        rows.append(await run_one(case, window, mkcfg(True)))
    same = sum(1 for a, b in zip(base, rows) if a["tags"] == b["tags"])
    print(f"{'real deep':12s} {f'{same}/{len(rows)}':>17s} "
          f"{pct([r['wall'] for r in rows],.5):9.2f} "
          f"{pct([r['wall'] for r in rows],1.0):9.2f} "
          f"{sum(1 for r in rows if r['source']!='ranked'):11d} "
          f"{statistics.mean([r['n_deep'] for r in rows]):10.1f} "
          f"{max(r['n_pending_after'] for r in rows):13d}")
    print("\n(cached arms: OFF wall is ~0.1s, so a broken gate would show as ~11s here)")
    print("usage:", llm.Usage.snapshot())


async def mode_hang_live(args) -> None:
    """The hang test at REAL latency: nothing cached, deep calls never return."""
    sources = load_cases(kind="conversation_tagging", seed=0)
    pairs = expand_windows(sources, seed=0)[: args.n]

    def mkcfg(deep: bool) -> pipeline.Config:
        return pipeline.Config(deadline_s=args.deadline, call_timeout_s=6.5,
                               use_cache=False, use_local=True, use_deep_enrichment=deep)

    out: Dict[str, List[dict]] = {}
    for label, mode, deep in (("off", "none", False), ("deep_hang", "hang", True),
                              ("deep_fail", "fail", True)):
        _INJECT["mode"] = mode
        rows = []
        for case, window in pairs:
            rows.append(await run_one(case, window, mkcfg(deep)))
        out[label] = rows
        report(f"=== {label} (uncached) ===", rows, args.deadline)
    _INJECT["mode"] = "none"
    print("usage:", llm.Usage.snapshot())


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ab", choices=("ab", "gate", "hang_live"))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--deadline", type=float, default=11.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", default="off,on", help="off,on,on_late")
    args = ap.parse_args()

    if args.mode in ("ab", "hang_live"):
        t0 = time.perf_counter()
        extract.warm()
        print(f"spacy warm {time.perf_counter()-t0:.2f}s (miner does this at startup)")

    await {"ab": mode_ab, "gate": mode_gate, "hang_live": mode_hang_live}[args.mode](args)


if __name__ == "__main__":
    asyncio.run(main())
