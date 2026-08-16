#!/usr/bin/env python3
"""Score the exact tags our miners submitted in production, offline.

This is the calibration that matters. Everywhere else the bench compares
strategies against each other on ground truth we generate; that is fine for
ranking, but it says nothing about whether our absolute numbers sit on the same
scale as the scores validators actually record.

Here we take responses this repo's miners really sent in May 2026 (captured in
data/all_uids_consolidated.jsonl by the miner's replay log), score those exact
tag lists with our harness, and compare the result with what W&B says those same
miners really scored in the same window.

If the two agree, our absolute numbers are trustworthy and the gap to the top
miners is real. If our harness reads systematically lower, then the gap is
partly a measurement artefact and our deployed miner is better placed than the
offline table suggests.

    python bench/replay_logged.py --kind webpage_metadata_generation --n 25
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.harness import Case, build_ground_truth, score_answer, summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURED = os.path.join(ROOT, "data", "by_task_type")


def load(kind: str, n: int):
    """Captured tasks paired with the tags our miner actually returned."""
    path = os.path.join(CAPTURED, f"{kind}.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out, seen = [], set()
    for r in rows:
        tags = (r.get("result") or {}).get("tags") or []
        window = (r.get("task_raw", {}).get("input", {}).get("data", {}) or {}).get("window") or []
        if not tags or not window:
            continue
        doc = window[0][1] if len(window[0]) >= 2 else ""
        if not doc.strip() or doc[:400] in seen:
            continue
        seen.add(doc[:400])
        cats = r.get("task_raw", {}).get("input", {}).get("input_categories") or []
        out.append(
            (
                Case(
                    kind=kind,
                    guid=str(len(out)),
                    document=doc,
                    enrichment=[w[1] for w in window[1:] if len(w) >= 2 and str(w[1]).strip()],
                    coding=bool(cats and "coding" in cats),
                ),
                tags,
            )
        )
        if len(out) >= n:
            break
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="webpage_metadata_generation")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    pairs = load(args.kind, args.n)
    print(f"replaying {len(pairs)} real {args.kind} responses through the harness")
    sem = asyncio.Semaphore(args.concurrency)

    async def one(case, tags):
        async with sem:
            gt = await build_ground_truth(case, model=args.model)
            if not gt.ok():
                return None
            return await score_answer(gt, tags, case.kind, model=args.model)

    verdicts = [v for v in await asyncio.gather(*[one(c, t) for c, t in pairs]) if v]
    s = summarize("logged_production_tags", verdicts)
    print(
        f"\noffline score of the tags we really submitted:\n"
        f"  n={s['n']} mean={s['mean']:.4f} median={s['median']:.4f} "
        f"sd={s['stdev']:.3f} min={s['min']:.3f} max={s['max']:.3f} zeros={s['zeros']}"
    )
    print(f"  penalties: {s['penalties']}")
    finals = sorted(v.final for v in verdicts)
    print(f"  quartiles: p25={finals[len(finals)//4]:.4f} p50={finals[len(finals)//2]:.4f} p75={finals[3*len(finals)//4]:.4f}")
    print("\nCompare against data/may_real_scores.json for the same miners in the same window.")


if __name__ == "__main__":
    asyncio.run(main())
