#!/usr/bin/env python3
"""PROBE gt-subphrase, part 4: the null control that decides it.

Part 3 left one explanation alive. The sub-phrase family adds ~65 candidates to
a pool of ~128 and the score rose +0.015. That could be a property of
SUB-PHRASES, or merely of HAVING 65 MORE ON-TOPIC CANDIDATES to rank. The
intended null control (more number-inflection variants) turned out to be empty -
variants.expand is exhausted at per_tag=2, it yields 0 extra tags at per_tag=6 -
so it tested nothing.

Two real controls:

  E  bigger LLM pool. cfg.pool_size 40 -> 120, giving a comparable number of
     extra ON-TOPIC candidates from a different generator. If E ~= B, the gain
     is candidate COUNT and the sub-phrase family is not special.

  F  off-topic sub-phrases. The same transforms applied to ANOTHER window's
     predicted ground truth. Same count, same lexical shape, wrong topic. This
     should be ~0 and confirms selection is not simply rewarding pool size.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util as _ilu
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from bench.harness import score_answer
from sn33 import pipeline, variants, vocab
from sn33.tags import normalize_all

_d = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    s = _ilu.spec_from_file_location(name, os.path.join(_d, fname))
    m = _ilu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_p1 = _load("gt_subphrase", "gt-subphrase.py")
select = _load("gt_subphrase_ab", "gt-subphrase-ab.py").select


def subphrases(pred_gt, gt_raw, base):
    union, seen = [], set()
    for fn in _p1.TRANSFORMS.values():
        for t in _p1.apply_transform(fn, pred_gt, gt_raw, set(pred_gt)):
            if t not in seen and t not in base:
                seen.add(t)
                union.append(t)
    return union


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-convo", type=int, default=8)
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases, per_convo=args.per_convo, seed=0)
    gts = {c.guid: await real_ground_truth(c, model=GT_MODEL, use_cache=True) for c in cases}

    cfg = pipeline.Config(use_cache=True, deadline_s=600.0, call_timeout_s=300.0)
    cfg_big = pipeline.Config(use_cache=True, deadline_s=600.0, call_timeout_s=300.0, pool_size=120)
    profile = pipeline.TASK_PROFILE["conversation_tagging"]

    # pass 1: mine every window once so arm F can borrow another window's pred_gt
    mined = []
    for wi, (case, window) in enumerate(pairs):
        gt = gts[case.guid]
        if not gt.ok():
            continue
        res = await pipeline.mine("conversation_tagging", window=window,
                                  enrichment=case.enrichment_lines, cfg=cfg)
        pg = normalize_all(res.predicted_gt)
        if pg:
            mined.append((case, window, res, pg))
    print(f"[mine] {len(mined)} windows", flush=True)

    arms = {"A_baseline": [], "B_subphrase": [], "E_bigger_pool": [], "F_offtopic_sub": []}
    guids, counts = [], defaultdict(list)

    for i, (case, window, res, pred_gt) in enumerate(mined):
        gt = gts[case.guid]
        gt_raw = set(gt.tags)
        anchors = vocab.anchors_for("conversation_tagging", limit=cfg.anchor_pool)
        v2 = variants.expand(pred_gt, per_tag=2)
        base = list(res.candidates)

        union = subphrases(pred_gt, gt_raw, base)

        # E: same window, a wider LLM pool
        text = pipeline._document_text("conversation_tagging", window)
        big = await pipeline._generate_pool("conversation_tagging", text, cfg_big)
        extra_pool = [t for t in normalize_all(big) if t not in base and t not in gt_raw]

        # F: sub-phrases of a DIFFERENT conversation's predicted GT
        other = mined[(i + len(mined) // 2) % len(mined)]
        off = [t for t in subphrases(other[3], gt_raw, base)][: len(union)]

        counts["sub"].append(len(union))
        counts["pool"].append(len(extra_pool))
        counts["off"].append(len(off))

        tags = {
            "A_baseline": await select(base, pred_gt, v2, anchors, cfg, profile),
            "B_subphrase": await select(base + union, pred_gt, v2, anchors, cfg, profile),
            "E_bigger_pool": await select(base + extra_pool, pred_gt, v2, anchors, cfg, profile),
            "F_offtopic_sub": await select(base + off, pred_gt, v2, anchors, cfg, profile),
        }
        vs = {}
        for k, t in tags.items():
            vs[k] = await score_answer(gt, t, "conversation_tagging", model=GT_MODEL,
                                       seed=i, use_cache=True)
            arms[k].append(vs[k])
        guids.append(case.guid)
        print(f"[{i+1}/{len(mined)}] {case.guid[:10]} "
              + "  ".join(f"{k[0]}={vs[k].adjusted:.4f}" for k in arms)
              + f"   nsub={len(union)} npool={len(extra_pool)} noff={len(off)}", flush=True)

    print()
    print("=" * 88)
    print(f"NULL CONTROLS   n = {len(guids)} windows, {len(set(guids))} conversations")
    print("=" * 88)
    print(f"extra candidates / window: sub-phrase {statistics.mean(counts['sub']):.1f}   "
          f"bigger-pool {statistics.mean(counts['pool']):.1f}   off-topic {statistics.mean(counts['off']):.1f}")
    print()
    base_adj = [v.adjusted for v in arms["A_baseline"]]
    print(f"{'arm':18s} {'adjusted':>9s} {'final':>9s} {'d_adj':>9s} {'W/L':>7s} {'holds':>7s}   per-conversation d_adj")
    for k, arr in arms.items():
        adj = [v.adjusted for v in arr]
        d = [x - y for x, y in zip(adj, base_adj)]
        per = {}
        for g in sorted(set(guids)):
            gd = [x for x, gg in zip(d, guids) if gg == g]
            per[g] = statistics.mean(gd)
        holds = sum(1 for v in per.values() if v > 0)
        cells = " ".join(f"{v:+.4f}" for v in per.values())
        print(f"{k:18s} {statistics.mean(adj):9.4f} {statistics.mean([v.final for v in arr]):9.4f} "
              f"{statistics.mean(d):+9.4f} "
              f"{str(sum(1 for x in d if x>0))+'/'+str(sum(1 for x in d if x<0)):>7s} "
              f"{str(holds)+'/'+str(len(set(guids))):>7s}   {cells}")


if __name__ == "__main__":
    asyncio.run(main())
