#!/usr/bin/env python3
"""PROBE gt-subphrase, part 5: which transform actually pays.

Parts 2-4 established that the sub-phrase family as a whole is worth ~+0.020
adjusted and that neither candidate count (arm E) nor lexical shape alone
(arm F) explains it. This part adds each transform to the pool ON ITS OWN, so
the credit can be assigned, and records which transform the selected tags came
from plus how each fares against the validator's English screen.
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
from bench.harness import score_answer, validate_tag_set
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
TRANSFORMS = _p1.TRANSFORMS


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-convo", type=int, default=8)
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases, per_convo=args.per_convo, seed=0)
    gts = {c.guid: await real_ground_truth(c, model=GT_MODEL, use_cache=True) for c in cases}

    cfg = pipeline.Config(use_cache=True, deadline_s=600.0, call_timeout_s=300.0)
    profile = pipeline.TASK_PROFILE["conversation_tagging"]

    arm_names = ["A_baseline"] + [f"only_{k}" for k in TRANSFORMS] + ["ALL", "SAFE_subset"]
    arms = {k: [] for k in arm_names}
    guids = []
    picked_by_tf = defaultdict(int)
    surv_by_tf = defaultdict(int)
    examples_kept = defaultdict(list)

    # "SAFE" = the transforms whose output still reads as an English keyword.
    # reorder ("investing estate real") and compound ("realestateinvesting")
    # are exactly what validate_tags.j2 calls a malformed / non-dictionary
    # compound; head_only was worst on cosine by a wide margin.
    SAFE = ["drop_head", "drop_modifier"]

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

        fam = {}
        for name, fn in TRANSFORMS.items():
            fam[name] = [t for t in _p1.apply_transform(fn, pred_gt, gt_raw, set(pred_gt))
                         if t not in base]
        allu, seen = [], set()
        for name in TRANSFORMS:
            for t in fam[name]:
                if t not in seen:
                    seen.add(t)
                    allu.append(t)
        safe = [t for name in SAFE for t in fam[name]]
        safe = list(dict.fromkeys(safe))

        pools = {"A_baseline": []}
        for name in TRANSFORMS:
            pools[f"only_{name}"] = fam[name]
        pools["ALL"] = allu
        pools["SAFE_subset"] = safe

        vs = {}
        for k, extra in pools.items():
            tags = await select(base + extra, pred_gt, v2, anchors, cfg, profile)
            vs[k] = await score_answer(gt, tags, "conversation_tagging", model=GT_MODEL,
                                       seed=wi, use_cache=True)
            arms[k].append(vs[k])
            if k == "ALL":
                surv = set(await validate_tag_set(tags, model=GT_MODEL, seed=wi, use_cache=True))
                for t in tags:
                    for name in TRANSFORMS:
                        if t in fam[name]:
                            picked_by_tf[name] += 1
                            if t in surv:
                                surv_by_tf[name] += 1
                                if len(examples_kept[name]) < 8:
                                    examples_kept[name].append(t)
                            break
        guids.append(case.guid)
        print(f"[{wi+1}/{len(pairs)}] {case.guid[:10]} " +
              " ".join(f"{k.replace('only_','')[:9]}={vs[k].adjusted:.4f}" for k in arm_names),
              flush=True)

    base_adj = [v.adjusted for v in arms["A_baseline"]]
    print()
    print("=" * 96)
    print(f"PER-TRANSFORM MARGINAL EFFECT   n = {len(guids)} windows, {len(set(guids))} conversations")
    print("=" * 96)
    print(f"{'arm':20s} {'adjusted':>9s} {'final':>9s} {'d_adj':>9s} {'W/L':>7s} {'holds':>7s}   per-conversation d_adj")
    for k in arm_names:
        adj = [v.adjusted for v in arms[k]]
        d = [x - y for x, y in zip(adj, base_adj)]
        per = {}
        for g in sorted(set(guids)):
            per[g] = statistics.mean([x for x, gg in zip(d, guids) if gg == g])
        holds = sum(1 for v in per.values() if v > 0)
        print(f"{k:20s} {statistics.mean(adj):9.4f} "
              f"{statistics.mean([v.final for v in arms[k]]):9.4f} {statistics.mean(d):+9.4f} "
              f"{str(sum(1 for x in d if x>0))+'/'+str(sum(1 for x in d if x<0)):>7s} "
              f"{str(holds)+'/'+str(len(set(guids))):>7s}   "
              + " ".join(f"{v:+.4f}" for v in per.values()))

    print()
    print("SELECTED TAGS BY TRANSFORM (arm ALL) - submitted vs surviving the English screen")
    for name in TRANSFORMS:
        s, k = picked_by_tf[name], surv_by_tf[name]
        print(f"  {name:15s} submitted {s:4d}   survived {k:4d}   ({100*k/max(1,s):3.0f}%)   "
              f"{examples_kept[name][:6]}")


if __name__ == "__main__":
    asyncio.run(main())
