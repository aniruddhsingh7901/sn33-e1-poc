#!/usr/bin/env python3
"""PROBE gt-subphrase, part 2: end-to-end paired A/B.

Part 1 measured candidate-pool cosine. This measures the thing that actually
pays: the validator's `adjusted` score, on the SAME windows and the SAME real
ground truth, with and without the sub-phrase family in the candidate pool.

The arms differ in exactly one variable - whether the sub-phrase transforms are
appended to the candidate list before ranking. Everything downstream (estimated
centroid, ranking, near-duplicate drop, compose, the validator's `validate_tags`
English screen, scoring) is identical and is the miner's own code.

Running the selection stage here rather than inside sn33/pipeline.py keeps that
file untouched while reproducing it line for line (pipeline.py:503-539).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from bench.harness import paired_delta, score_answer
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline, variants, vocab
from sn33.tags import centroid, drop_near_duplicates, normalize_all, rank_by_centroid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_mod = __import__("importlib").import_module("importlib").import_module  # noqa
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "gt_subphrase", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt-subphrase.py")
)
_p1 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_p1)
TRANSFORMS = _p1.TRANSFORMS
apply_transform = _p1.apply_transform


async def select(candidates: Sequence[str], predicted_gt: Sequence[str],
                 variant_tags: Sequence[str], anchors: Sequence[str],
                 cfg: pipeline.Config, profile: dict) -> List[str]:
    """pipeline.py:503-539, verbatim in behaviour."""
    cands = normalize_all(list(candidates))
    gt_for_target = Utils.get_clean_tag_set(list(predicted_gt)) if predicted_gt else []
    to_embed = list(dict.fromkeys(cands + gt_for_target))
    vectors = await llm.embed(to_embed, timeout=180, use_cache=True)

    target = centroid([vectors[t] for t in gt_for_target if t in vectors]) if gt_for_target else None
    if target is None:
        return []
    ranked = rank_by_centroid(cands, vectors, target)
    anchor_set = set(anchors)
    ranked = [(t, s) for t, s in ranked if t not in anchor_set or s >= cfg.anchor_min_cos]
    protected = set(variant_tags) | set(normalize_all(predicted_gt))
    ranked = drop_near_duplicates(ranked, vectors, cfg.dedup_threshold, protected=protected)
    if not ranked:
        return []
    return pipeline.compose(ranked, predicted_gt, profile,
                            profile["target_tags"], profile["insurance"], anchors=anchor_set)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-convo", type=int, default=8)
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases, per_convo=args.per_convo, seed=0)
    gts = {c.guid: await real_ground_truth(c, model=GT_MODEL, use_cache=True) for c in cases}
    print(f"[load] {len(pairs)} windows / {len(cases)} conversations", flush=True)

    cfg = pipeline.Config(use_cache=True, deadline_s=600.0, call_timeout_s=300.0)
    profile = pipeline.TASK_PROFILE["conversation_tagging"]

    A, B = [], []            # A = baseline miner, B = + sub-phrase family
    by_convo = defaultdict(lambda: {"a": [], "b": []})
    dropped = {"submitted": 0, "survived": 0}
    sub_dropped = {"submitted": 0, "survived": 0}

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
        variant_tags = variants.expand(pred_gt, per_tag=cfg.variants_per_tag)

        base_cands = list(res.candidates)
        union, useen = [], set()
        for fn in TRANSFORMS.values():
            for t in apply_transform(fn, pred_gt, gt_raw, set(pred_gt)):
                if t not in useen and t not in base_cands:
                    useen.add(t)
                    union.append(t)

        tags_a = await select(base_cands, pred_gt, variant_tags, anchors, cfg, profile)
        tags_b = await select(base_cands + union, pred_gt, variant_tags + union,
                              anchors, cfg, profile)

        va = await score_answer(gt, tags_a, "conversation_tagging", model=GT_MODEL,
                                seed=wi, use_cache=True)
        vb = await score_answer(gt, tags_b, "conversation_tagging", model=GT_MODEL,
                                seed=wi, use_cache=True)
        A.append(va)
        B.append(vb)
        by_convo[case.guid]["a"].append(va.adjusted)
        by_convo[case.guid]["b"].append(vb.adjusted)

        # how many sub-phrase tags did arm B actually submit, and how many
        # survived the validator's "good English keywords" screen?
        picked = [t for t in tags_b if t in useen]
        sub_dropped["submitted"] += len(picked)
        surv = set(await __import__("bench.harness", fromlist=["x"]).validate_tag_set(
            tags_b, model=GT_MODEL, seed=wi, use_cache=True))
        sub_dropped["survived"] += len([t for t in picked if t in surv])
        dropped["submitted"] += len(tags_b)
        dropped["survived"] += len(surv)

        print(f"[{wi+1}/{len(pairs)}] {case.guid[:10]} A={va.adjusted:.4f} B={vb.adjusted:.4f} "
              f"d={vb.adjusted-va.adjusted:+.4f}  sub_picked={len(picked)} "
              f"pen_a={va.penalties} pen_b={vb.penalties}", flush=True)

    n = len(A)
    print()
    print("=" * 84)
    print(f"END-TO-END PAIRED A/B   n = {n} windows, {len(by_convo)} conversations")
    print("=" * 84)
    for label, arr in (("A baseline miner", A), ("B + sub-phrase family", B)):
        print(f"{label:24s} adjusted {statistics.mean([v.adjusted for v in arr]):.4f}   "
              f"final {statistics.mean([v.final for v in arr]):.4f}   "
              f"unique/win {statistics.mean([v.n_unique for v in arr]):.1f}   "
              f"survived/win {statistics.mean([v.n_survived for v in arr]):.1f}")
    d = [b.adjusted - a.adjusted for a, b in zip(A, B)]
    print(f"\ndelta(adjusted) B-A = {statistics.mean(d):+.4f}   "
          f"wins/losses {sum(1 for x in d if x>0)}/{sum(1 for x in d if x<0)}  "
          f"ties {sum(1 for x in d if x==0)}")
    print(f"bootstrap on FINAL: {paired_delta(A, B)}")

    print("\nPER CONVERSATION (adjusted)")
    for g, v in sorted(by_convo.items()):
        da = statistics.mean(v["a"])
        db = statistics.mean(v["b"])
        print(f"  {g[:10]:12s} n={len(v['a']):2d}  A={da:.4f}  B={db:.4f}  d={db-da:+.4f}")

    print(f"\nVALIDATOR ENGLISH SCREEN (arm B)")
    print(f"  all tags        : {dropped['survived']}/{dropped['submitted']} survived "
          f"({100*dropped['survived']/max(1,dropped['submitted']):.0f}%)")
    print(f"  sub-phrase tags : {sub_dropped['survived']}/{sub_dropped['submitted']} survived "
          f"({100*sub_dropped['survived']/max(1,sub_dropped['submitted']):.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
