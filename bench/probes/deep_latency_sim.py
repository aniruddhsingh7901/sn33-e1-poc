#!/usr/bin/env python3
"""Deep-enrichment latency: gate verification and truncation risk under stress.

Why a replay and not more live runs: the OpenAI account ran out of credits
partway through this investigation (429 credit_balance_exhausted), so no further
uncached measurement is possible. What is preserved is the full per-call timing
of the live paired run in `ab24.json` (24 windows x 2 arms, uncached, 11s
deadline). This module replays those measured latencies through the REAL
`pipeline.mine` / `replica.replicate` code, with every prompt answered from the
on-disk cache so the tag output is the real one.

That buys two things the live run could not give:

1.  **Gate verification.** Deep calls can be made to fail, hang, or return late
    on demand, and the answer compared byte-for-byte against the feature being
    off.
2.  **Truncation risk.** At an 11s deadline neither arm truncated (0/24 both),
    so the live run cannot resolve the incremental risk. Scaling every measured
    call latency by a slowdown factor `s` reproduces the condition that
    historically caused truncation ("once OpenAI slowed"), and the arms are
    compared on identical latency draws.

The replay is a MODEL. It is calibrated by construction - at s=1.0 it reproduces
the measured wall distribution - but the sweep above s=1.0 is an extrapolation
of the measured per-call latencies, not an observation.
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
from sn33 import llm, pipeline, prompts, replica

SCRATCH = os.environ.get("SN33_SCRATCH", "/tmp")
AB = os.path.join(SCRATCH, "ab24.json")

_orig_chat = llm.chat
_orig_embed = llm.embed
_orig_replicate = replica.replicate

# ---------------------------------------------------------------- call classes


def classify(prompt: str, salt: str) -> str:
    if salt == "deep":
        return "deep"
    if "<set0>" in prompt:
        return "combine"
    if "<enrichment_content>" in prompt:
        return "enrich"
    if prompt.startswith("Below is an excerpt"):
        return "pool"
    return "doc"


# ---------------------------------------------------------------- replay state

STATE: Dict[str, object] = {
    "lat": {},          # class -> latency (s); enrich/deep are lists consumed round-robin
    "scale": 1.0,
    "deep_mode": "real",   # real | fail | hang | slow
    "deep_delay": 5.0,
    "deep_tags": [],
    "counters": {},
    "contend": False,
}


async def chat_stub(prompt, model, timeout=8.0, temperature=0.0, use_cache=False,
                    max_completion_tokens=None, attempts=1, salt=""):
    """Sleep the measured latency for this call class, then answer from cache.

    Reproduces `llm.chat`'s contract exactly, including "returns None on its own
    timeout", which is the behaviour the combine fallback depends on.
    """
    cls = classify(prompt, salt)
    lat = STATE["lat"]
    n = STATE["counters"].get(cls, 0)
    STATE["counters"][cls] = n + 1

    if cls == "deep":
        mode = STATE["deep_mode"]
        if mode == "fail":
            return None
        if mode == "hang":
            await asyncio.sleep(3600)
            return None
        d = float(STATE["deep_delay"]) if mode == "slow" else _pick(lat, "deep", n)
    else:
        d = _pick(lat, cls, n) * (CONTENTION if STATE["contend"] else 1.0)
    d *= float(STATE["scale"])

    try:
        await asyncio.wait_for(asyncio.sleep(d), timeout=timeout)
    except asyncio.TimeoutError:
        return None                      # llm.chat: deadline is sacred, no retry

    if cls == "deep":
        return ", ".join(STATE["deep_tags"])
    return await _orig_chat(prompt, model, timeout=60, temperature=temperature,
                            use_cache=True, max_completion_tokens=max_completion_tokens,
                            attempts=1, salt=salt)


def _pick(lat: dict, cls: str, i: int) -> float:
    v = lat.get(cls)
    if v is None:
        return 0.5
    if isinstance(v, list):
        return float(v[i % len(v)]) if v else 0.5
    return float(v)


async def embed_stub(texts, timeout=6.0, use_cache=True, batch_size=256):
    d = float(STATE["lat"].get("embed", 0.9)) * float(STATE["scale"])
    try:
        await asyncio.wait_for(asyncio.sleep(d), timeout=timeout)
    except asyncio.TimeoutError:
        return {}
    return await _orig_embed(texts, timeout=60, use_cache=True, batch_size=batch_size)


llm.chat = chat_stub
llm.embed = embed_stub

_IMPL: Dict[str, object] = {"fn": None}
_SPY: Dict[str, object] = {}


async def _spy_replicate(*a, **kw):
    _SPY["t0"] = time.perf_counter()
    rep = await (_IMPL["fn"] or _orig_replicate)(*a, **kw)
    _SPY["rep"] = rep
    _SPY["t1"] = time.perf_counter()
    return rep


replica.replicate = _spy_replicate


# ---------------------------------------------------------------- harness

def pct(v: List[float], p: float) -> float:
    if not v:
        return 0.0
    s = sorted(v)
    return s[max(0, min(len(s) - 1, int(round(p * len(s) + 0.5)) - 1))]


async def run_window(case, window, cfg, lat, deep_mode="real", deep_delay=5.0, scale=1.0):
    STATE["lat"] = lat
    STATE["contend"] = bool(cfg.use_deep_enrichment)
    STATE["scale"] = scale
    STATE["deep_mode"] = deep_mode
    STATE["deep_delay"] = deep_delay
    STATE["counters"] = {}
    _SPY.clear()
    t0 = time.perf_counter()
    res = await pipeline.mine("conversation_tagging", window=window,
                              enrichment=case.enrichment_lines, cfg=cfg)
    wall = time.perf_counter() - t0
    rep = _SPY.get("rep")
    return {
        "wall": wall, "source": res.source, "degraded": res.degraded,
        "tags": list(res.tags), "n_cand": len(res.candidates),
        "n_deep": len(getattr(rep, "deep_tags", []) or []) if rep is not None else 0,
        "rep_wall": (_SPY["t1"] - _SPY["t0"]) if "t1" in _SPY else None,
        "pending": len([t for t in asyncio.all_tasks() if not t.done()]) - 1,
    }


# Deep calls double the number of in-flight requests, and the measured live run
# shows that inflates the calls the ANSWER depends on: fan-out p50 2.22 -> 2.40s,
# combine p90 2.10 -> 2.43s. That contention is a real cost of the feature and
# the gate cannot protect against it, so the replay keeps it.
CONTENTION = 1.08


def latencies_from(ab: dict, i: int) -> dict:
    """Per-window measured latencies.

    Base calls come from the OFF arm (uncontended). The deep call latency is set
    so that the measured GRACE reappears: in the live ON run the grace is
    `combine_start_off - fanout`, i.e. how long past the fan-out the deep calls
    held the replica open. Reconstructing it that way reproduces the observed
    +1.0s critical-path cost instead of over- or under-stating it.
    """
    off, on = ab["off"][i], ab["on"][i]
    base = float(off["fanout"])
    grace = max(0.0, float(on["combine_start_off"]) - float(on["fanout"]))
    return {
        "doc": base,          # doc and enrichment fan out together; use their max
        "enrich": base,
        "pool": base,
        "combine": float(off["combine"] or 1.9),
        # +0.05 so the deep task lands strictly after the fan-out, as measured
        "deep": base + grace + 0.05,
        "embed": max(0.1, float(off["wall"]) - float(off["rep_wall"])),
    }


def cfgs(deadline: float):
    from bench.probes.deep_latency_fix import replicate_late
    return {
        "off": (pipeline.Config(deadline_s=deadline, call_timeout_s=6.5, use_cache=True,
                                use_local=False, use_deep_enrichment=False), None),
        "on": (pipeline.Config(deadline_s=deadline, call_timeout_s=6.5, use_cache=True,
                               use_local=False, use_deep_enrichment=True), None),
        "on_late": (pipeline.Config(deadline_s=deadline, call_timeout_s=6.5, use_cache=True,
                                    use_local=False, use_deep_enrichment=True), replicate_late),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sweep", choices=("validate", "sweep", "gate"))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--deadline", type=float, default=11.0)
    ap.add_argument("--arms", default="off,on,on_late")
    args = ap.parse_args()

    ab = json.load(open(AB))
    sources = load_cases(kind="conversation_tagging", seed=0)
    pairs = expand_windows(sources, seed=0)[: args.n]
    lats = [latencies_from(ab, i) for i in range(len(pairs))]
    ARMS = cfgs(args.deadline)
    names = [a.strip() for a in args.arms.split(",")]

    # Deep tags must be strings whose embeddings are already cached, otherwise
    # the ON arm silently drops them and the A/B is vacuous. Candidates from
    # OTHER windows satisfy that and are realistic pool material.
    deep_pool: List[str] = []
    for j in range(len(pairs)):
        deep_pool += (ab["on"][j].get("tags") or [])
    STATE["deep_tags"] = list(dict.fromkeys(deep_pool))[:40] or ["podcast", "interview"]

    if args.mode == "validate":
        print("REPLAY vs LIVE (s=1.0): does the model reproduce the measured run?\n")
        for arm in ("off", "on"):
            cfg, impl = ARMS[arm]
            _IMPL["fn"] = impl
            rows = [await run_window(c, w, cfg, lats[i], scale=1.0)
                    for i, (c, w) in enumerate(pairs)]
            live = [float(r["wall"]) for r in ab[arm]]
            sim = [r["wall"] for r in rows]
            print(f"  {arm:8s} live  p50 {pct(live,.5):5.2f} p90 {pct(live,.9):5.2f} "
                  f"max {pct(live,1.):5.2f}   not-ranked {sum(1 for r in ab[arm] if r['source']!='ranked')}")
            print(f"  {'':8s} sim   p50 {pct(sim,.5):5.2f} p90 {pct(sim,.9):5.2f} "
                  f"max {pct(sim,1.):5.2f}   not-ranked {sum(1 for r in rows if r['source']!='ranked')}")
        _IMPL["fn"] = None
        return

    if args.mode == "gate":
        cfg_off, _ = ARMS["off"]
        base = [await run_window(c, w, cfg_off, lats[i]) for i, (c, w) in enumerate(pairs)]
        print(f"baseline (feature OFF): wall p50 {pct([r['wall'] for r in base],.5):.2f}s  "
              f"not-ranked {sum(1 for r in base if r['source']!='ranked')}/{len(base)}\n")
        hdr = (f"{'arm':22s} {'==OFF tags':>10s} {'wall p50':>9s} {'wall max':>9s} "
               f"{'not-ranked':>11s} {'deep':>5s} {'leaked':>7s}")
        print(hdr); print("-" * len(hdr))
        trials = [
            ("on / deep real", "on", "real", 0.0),
            ("on / deep fail", "on", "fail", 0.0),
            ("on / deep hang", "on", "hang", 0.0),
            ("on / deep 5s", "on", "slow", 5.0),
            ("on / deep 30s", "on", "slow", 30.0),
            ("on_late / deep real", "on_late", "real", 0.0),
            ("on_late / deep hang", "on_late", "hang", 0.0),
            ("on_late / deep 5s", "on_late", "slow", 5.0),
        ]
        for label, arm, mode, delay in trials:
            cfg, impl = ARMS[arm]
            _IMPL["fn"] = impl
            rows = [await run_window(c, w, cfg, lats[i], deep_mode=mode, deep_delay=delay)
                    for i, (c, w) in enumerate(pairs)]
            same = sum(1 for a, b in zip(base, rows) if a["tags"] == b["tags"])
            print(f"{label:22s} {f'{same}/{len(rows)}':>10s} "
                  f"{pct([r['wall'] for r in rows],.5):9.2f} {pct([r['wall'] for r in rows],1.):9.2f} "
                  f"{sum(1 for r in rows if r['source']!='ranked'):11d} "
                  f"{statistics.mean([r['n_deep'] for r in rows]):5.1f} "
                  f"{max(r['pending'] for r in rows):7d}")
        _IMPL["fn"] = None
        return

    # ---- sweep ----
    print(f"latency replay, deadline {args.deadline}s, {len(pairs)} windows, "
          f"identical latency draws per arm\n")
    hdr = f"{'slowdown':9s} " + " ".join(f"{n:>26s}" for n in names)
    print(hdr); print("-" * len(hdr))
    per_arm_rows: Dict[str, Dict[float, list]] = {n: {} for n in names}
    for s in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2):
        cells = []
        for arm in names:
            cfg, impl = ARMS[arm]
            _IMPL["fn"] = impl
            rows = [await run_window(c, w, cfg, lats[i], scale=s)
                    for i, (c, w) in enumerate(pairs)]
            per_arm_rows[arm][s] = rows
            bad = sum(1 for r in rows if r["source"] != "ranked")
            deg = sum(1 for r in rows if r["degraded"])
            cells.append(f"p50 {pct([r['wall'] for r in rows],.5):5.2f} cut {bad:2d} deg {deg:2d}")
        print(f"x{s:<8.1f} " + " ".join(f"{c:>26s}" for c in cells))
    _IMPL["fn"] = None

    print("\nincremental truncation (source != ranked), percentage points vs OFF:")
    for s in sorted(per_arm_rows[names[0]]):
        base_bad = sum(1 for r in per_arm_rows[names[0]][s] if r["source"] != "ranked")
        line = f"  x{s:<5.1f} off {100*base_bad/args.n:5.1f}%"
        for arm in names[1:]:
            bad = sum(1 for r in per_arm_rows[arm][s] if r["source"] != "ranked")
            line += f"   {arm} {100*bad/args.n:5.1f}% ({100*(bad-base_bad)/args.n:+.1f}pp)"
        print(line)

    print("\ndegraded replica (combine lost -> local_combine, worth -0.025 per CLAUDE.md):")
    for s in sorted(per_arm_rows[names[0]]):
        line = f"  x{s:<5.1f}"
        for arm in names:
            deg = sum(1 for r in per_arm_rows[arm][s] if r["degraded"])
            line += f"   {arm} {100*deg/args.n:5.1f}%"
        print(line)


if __name__ == "__main__":
    asyncio.run(main())
