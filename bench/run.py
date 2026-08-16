#!/usr/bin/env python3
"""Run strategies over a corpus and print a comparison table.

    python bench/run.py --kind conversation_tagging --n 20 --strategies stock,prod,replica

Ground truth is built once per case and shared by every strategy, so the
comparison is paired: the bootstrap CI on the difference is what decides
whether a change is real, not the raw means.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import corpus, strategies as S
from bench.harness import Verdict, build_ground_truth, paired_delta, score_answer, summarize
from sn33 import llm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_strategies(names: List[str], model: str) -> List[S.Strategy]:
    out = []
    for n in names:
        n = n.strip()
        if n == "stock":
            out.append(S.stock(model))
        elif n == "prod":
            out.append(S.prod(model))
        elif n == "prod_capped":
            out.append(S.prod_capped(model))
        elif n == "replica":
            out.append(S.replica("replica", gt_model=model, pool_model=model))
        elif n == "oracle":
            out.append(S.oracle(gt_model=model, pool_model=model))
        elif n.startswith("replica:") or n.startswith("oracle:"):
            # replica:target_tags=8:insurance=1:combine=local
            # (colon-separated, because commas already separate strategies)
            overrides = {}
            for part in n.split(":")[1:]:
                if not part:
                    continue
                k, v = part.split("=", 1)
                overrides[k] = int(v) if v.isdigit() else (float(v) if v.replace(".", "", 1).isdigit() else v)
            overrides.setdefault("gt_model", model)
            overrides.setdefault("pool_model", model)
            out.append(S.oracle(n, **overrides) if n.startswith("oracle:") else S.replica(n, **overrides))
        else:
            raise SystemExit(f"unknown strategy {n!r}")
    return out


async def run_case(case, strats, model, sem) -> Dict[str, Verdict]:
    async with sem:
        gt = await build_ground_truth(case, model=model)
    if not gt.ok():
        return {}
    out = {}
    for st in strats:
        async with sem:
            try:
                tags = await st.run(case, gt)
            except Exception as e:  # a strategy that crashes scores zero, like production
                print(f"    ! {st.name} raised on {case.guid}: {type(e).__name__}: {e}")
                tags = []
            out[st.name] = await score_answer(gt, tags, case.kind, model=model)
    out["_gt"] = gt  # type: ignore
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="conversation_tagging")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--strategies", default="stock,prod,replica")
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--windows-per-convo", type=int, default=1)
    ap.add_argument("--enrichment", type=int, default=0,
                    help="synthetic enrichment lines per conversation case (models post-2026-06-12 traffic)")
    ap.add_argument("--out", default=None, help="write per-case JSON here")
    args = ap.parse_args()

    if args.kind == "conversation_tagging":
        cases = corpus.conversation_cases(n=args.n, seed=args.seed, windows_per_convo=args.windows_per_convo)
        if args.enrichment:
            # Model post-2026-06-12 traffic, where conversation tasks carry
            # enrichment lines shared by ground truth and miner alike.
            await corpus.add_synthetic_enrichment(cases, per_case=args.enrichment, model=args.model)
            print(f"attached {args.enrichment} synthetic enrichment line(s) per case\n")
    elif args.kind in ("webpage_metadata_generation", "named_entities_extraction"):
        cases = corpus.captured_cases(args.kind, n=args.n, seed=args.seed)
    elif args.kind == "survey_tagging":
        cases = corpus.survey_cases(
            n=args.n, seed=args.seed, choices_path=os.path.join(ROOT, "data", "survey_choices.json")
        )
        cases = [c for c in cases if c.reference_tags]
        if not cases:
            raise SystemExit("no survey ground truth - run bench/make_survey_choices.py first")
    elif args.kind == "skill_generation":
        cases = corpus.skill_cases(n=args.n)
    else:
        raise SystemExit(f"unknown kind {args.kind}")

    strats = build_strategies(args.strategies.split(","), args.model)
    print(f"corpus: {args.kind}  cases={len(cases)}  model={args.model}")
    print(f"strategies: {', '.join(s.name for s in strats)}\n")

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()
    results = await asyncio.gather(*[run_case(c, strats, args.model, sem) for c in cases])
    elapsed = time.perf_counter() - t0

    per_strategy: Dict[str, List[Verdict]] = {s.name: [] for s in strats}
    kept = 0
    for r in results:
        if not r:
            continue
        kept += 1
        for s in strats:
            per_strategy[s.name].append(r[s.name])

    print(f"scored {kept}/{len(cases)} cases in {elapsed:.0f}s  ({llm.Usage.snapshot()})\n")
    header = f"{'strategy':34s} {'mean':>7s} {'median':>7s} {'sd':>6s} {'min':>6s} {'max':>6s} {'tags':>5s} {'zero':>5s} {'pen%':>5s}"
    print(header)
    print("-" * len(header))
    summaries = {}
    for s in strats:
        v = per_strategy[s.name]
        summ = summarize(s.name, v)
        summaries[s.name] = summ
        if not v:
            continue
        print(
            f"{s.name:34s} {summ['mean']:7.4f} {summ['median']:7.4f} {summ['stdev']:6.3f} "
            f"{summ['min']:6.3f} {summ['max']:6.3f} {summ['mean_tags']:5.1f} {summ['zeros']:5d} {summ['penalty_rate']*100:4.0f}%"
        )

    base = strats[0].name
    print(f"\npaired vs {base} (bootstrap 95% CI on the per-case difference):")
    for s in strats[1:]:
        d = paired_delta(per_strategy[base], per_strategy[s.name])
        if not d:
            continue
        mark = "SIGNIFICANT" if d["significant"] else "not significant"
        print(
            f"  {s.name:32s} {d['delta']:+.4f}  CI[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]  "
            f"W/L {d['wins']}/{d['losses']}  {mark}"
        )

    print("\npenalties fired:")
    for s in strats:
        summ = summaries[s.name]
        if summ.get("penalties"):
            print(f"  {s.name:32s} {summ['penalties']}")

    if args.out:
        payload = []
        for case, r in zip(cases, results):
            if not r:
                continue
            payload.append(
                {
                    "guid": case.guid,
                    "kind": case.kind,
                    "gt_tags": r["_gt"].tags,
                    "scores": {s.name: r[s.name].__dict__ for s in strats},
                }
            )
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
