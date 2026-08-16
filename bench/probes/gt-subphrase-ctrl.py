#!/usr/bin/env python3
"""PROBE gt-subphrase, part 3: controls.

Part 2 showed +0.0196 adjusted for "miner + sub-phrase family". Before that can
be called a win for SUB-PHRASES specifically, three alternative explanations
have to be killed:

  1. ZERO-PAD RESCUE. Two of 26 windows jumped +0.094 / +0.111 because arm A
     fired `less_than_3_unique` - its top_3_mean was zero-padded. Any family
     that adds unique candidates fixes that. Report the delta with and without
     those windows.

  2. DEDUP PROTECTION. Arm B put the new tags in `protected`, exempting them
     from drop_near_duplicates. That is a second changed variable. Arm C adds
     the same tags UNPROTECTED.

  3. THE FAMILY IS NOT SPECIAL. If the mechanism is only "more unique
     candidates", then more of the EXISTING number-inflection variants should
     buy the same thing. Arm D adds variants at per_tag=6 instead of 2 - same
     mechanism, different family, comparable tag count.

Arms are paired on identical windows and identical real ground truth.
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

_spec = _ilu.spec_from_file_location(
    "gt_subphrase", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt-subphrase.py"))
_p1 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_p1)

_spec2 = _ilu.spec_from_file_location(
    "gt_subphrase_ab", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt-subphrase-ab.py"))
_p2 = _ilu.module_from_spec(_spec2)
_spec2.loader.exec_module(_p2)
select = _p2.select


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-convo", type=int, default=8)
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases, per_convo=args.per_convo, seed=0)
    gts = {c.guid: await real_ground_truth(c, model=GT_MODEL, use_cache=True) for c in cases}

    cfg = pipeline.Config(use_cache=True, deadline_s=600.0, call_timeout_s=300.0)
    profile = pipeline.TASK_PROFILE["conversation_tagging"]

    arms = {"A_baseline": [], "B_sub_protected": [], "C_sub_unprotected": [], "D_more_variants": []}
    rescue = []          # windows where A zero-padded top_3_mean
    guids = []
    counts = defaultdict(list)

    for wi, (case, window) in enumerate(pairs):
        gt = gts[case.guid]
        if not gt.ok():
            continue
        res = await pipeline.mine("conversation_tagging", window=window,
                                  enrichment=case.enrichment_lines, cfg=cfg)
        pred_gt = normalize_all(res.predicted_gt)
        if not pred_gt:
            continue
        gt_raw = set(gt.tags)
        anchors = vocab.anchors_for("conversation_tagging", limit=cfg.anchor_pool)
        v2 = variants.expand(pred_gt, per_tag=2)
        base = list(res.candidates)

        union, useen = [], set()
        for fn in _p1.TRANSFORMS.values():
            for t in _p1.apply_transform(fn, pred_gt, gt_raw, set(pred_gt)):
                if t not in useen and t not in base:
                    useen.add(t)
                    union.append(t)

        # arm D: same mechanism (more unique candidates from the SAME source
        # tags), different family - deeper number inflection.
        v6 = variants.expand(pred_gt, per_tag=6, limit=200)
        extra_v = [t for t in v6 if t not in base and t not in v2]

        counts["sub"].append(len(union))
        counts["extra_var"].append(len(extra_v))

        tags = {}
        tags["A_baseline"] = await select(base, pred_gt, v2, anchors, cfg, profile)
        tags["B_sub_protected"] = await select(base + union, pred_gt, v2 + union, anchors, cfg, profile)
        tags["C_sub_unprotected"] = await select(base + union, pred_gt, v2, anchors, cfg, profile)
        tags["D_more_variants"] = await select(base + extra_v, pred_gt, v6, anchors, cfg, profile)

        vs = {}
        for k, t in tags.items():
            vs[k] = await score_answer(gt, t, "conversation_tagging", model=GT_MODEL,
                                       seed=wi, use_cache=True)
            arms[k].append(vs[k])
        guids.append(case.guid)
        padded = vs["A_baseline"].n_unique < 3
        rescue.append(padded)
        print(f"[{wi+1}/{len(pairs)}] {case.guid[:10]} "
              + "  ".join(f"{k.split('_')[0]}={vs[k].adjusted:.4f}" for k in arms)
              + f"  uniqA={vs['A_baseline'].n_unique} nsub={len(union)} nvar={len(extra_v)}"
              + ("  <-- A ZERO-PADDED" if padded else ""), flush=True)

    n = len(guids)
    print()
    print("=" * 90)
    print(f"CONTROLS   n = {n} windows, {len(set(guids))} conversations")
    print("=" * 90)
    print(f"mean sub-phrase tags added / window   : {statistics.mean(counts['sub']):.1f}")
    print(f"mean extra variant tags added / window: {statistics.mean(counts['extra_var']):.1f}")
    print()
    base_adj = [v.adjusted for v in arms["A_baseline"]]
    print(f"{'arm':22s} {'adjusted':>9s} {'final':>9s} {'d_adj':>9s} {'W/L':>7s} {'holds':>6s}")
    for k, arr in arms.items():
        adj = [v.adjusted for v in arr]
        d = [x - y for x, y in zip(adj, base_adj)]
        w = sum(1 for x in d if x > 0)
        l = sum(1 for x in d if x < 0)
        holds = 0
        for g in set(guids):
            ga = [x - y for x, y, gg in zip(adj, base_adj, guids) if gg == g]
            if ga and statistics.mean(ga) > 0:
                holds += 1
        print(f"{k:22s} {statistics.mean(adj):9.4f} {statistics.mean([v.final for v in arr]):9.4f} "
              f"{statistics.mean(d):+9.4f} {str(w)+'/'+str(l):>7s} {str(holds)+'/'+str(len(set(guids))):>6s}")

    print()
    print("SPLIT: windows where arm A zero-padded top_3_mean (<3 unique) vs the rest")
    for label, mask in (("A zero-padded", rescue), ("A had >=3 unique", [not r for r in rescue])):
        idx = [i for i, m in enumerate(mask) if m]
        if not idx:
            print(f"  {label:20s} n=0")
            continue
        print(f"  {label:20s} n={len(idx):2d}   " + "   ".join(
            f"{k.split('_')[0]}={statistics.mean([arms[k][i].adjusted for i in idx]):.4f}"
            for k in arms))
        for k in arms:
            if k == "A_baseline":
                continue
            d = [arms[k][i].adjusted - base_adj[i] for i in idx]
            print(f"      d({k}) = {statistics.mean(d):+.4f}")


if __name__ == "__main__":
    asyncio.run(main())
