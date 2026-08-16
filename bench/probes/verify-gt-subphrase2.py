#!/usr/bin/env python3
"""ADVERSARIAL VERIFICATION part 2 of the `gt-subphrase` finding.

Three checks their probes did not run:

1. STRAW MAN. Their arm A is a re-implementation of pipeline.py:503-539, not
   pipeline.mine itself. If the re-implementation is weaker than the shipped
   miner, every delta is inflated. Score the miner's OWN answer (res.tags) on
   the same windows and compare.

2. IS THE GAIN THE TAGS, OR THE SCREEN? Arm B submits 12 tags of which ~9.3
   survive the English screen, vs 10.0 for arm A - i.e. the screen deletes more
   of arm B's tags. If B only wins because the screen happens to delete its
   weak tags, the win is a property of one LLM draw, not of the tag set. Score
   both arms with the screen BYPASSED and see if the gain survives.

3. COST. Candidate-pool size and embed-call count per arm.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util as _ilu
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from bench.harness import score_answer
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline, scoring, variants, vocab
from sn33.tags import normalize_all

_d = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    s = _ilu.spec_from_file_location(name, os.path.join(_d, fname))
    m = _ilu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_p1 = _load("gt_subphrase", "gt-subphrase.py")
select = _load("gt_subphrase_ab", "gt-subphrase-ab.py").select


async def raw_score(gt, tags, seed):
    """scoring.score_tags with the English screen BYPASSED."""
    if not tags:
        return 0.0, 0.0
    vecs = await llm.embed(Utils.get_clean_tag_set(list(tags)), use_cache=True)
    r = scoring.score_tags(gt.tags, gt.target, list(tags), vecs,
                           penalties=True, min_tags=3, seed=seed)
    return r["adjusted"], r["final"]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-convo", type=int, default=8)
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases, per_convo=args.per_convo, seed=0)
    gts = {c.guid: await real_ground_truth(c, model=GT_MODEL, use_cache=True) for c in cases}

    cfg = pipeline.Config(use_cache=True, deadline_s=600.0, call_timeout_s=300.0)
    profile = pipeline.TASK_PROFILE["conversation_tagging"]

    rows = []
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
        v2 = variants.expand(pred_gt, per_tag=cfg.variants_per_tag)
        base = list(res.candidates)

        union, seen = [], set()
        for fn in _p1.TRANSFORMS.values():
            for t in _p1.apply_transform(fn, pred_gt, gt_raw, set(pred_gt)):
                if t not in seen and t not in base:
                    seen.add(t)
                    union.append(t)

        tags_a = await select(base, pred_gt, v2, anchors, cfg, profile)
        tags_b = await select(base + union, pred_gt, v2, anchors, cfg, profile)

        # 1. straw man: the shipped miner's own answer
        v_mine = await score_answer(gt, res.tags, "conversation_tagging", model=GT_MODEL,
                                    seed=wi, use_cache=True)
        v_a = await score_answer(gt, tags_a, "conversation_tagging", model=GT_MODEL,
                                 seed=wi, use_cache=True)
        v_b = await score_answer(gt, tags_b, "conversation_tagging", model=GT_MODEL,
                                 seed=wi, use_cache=True)

        # 2. screen bypassed
        ra_adj, ra_fin = await raw_score(gt, tags_a, wi)
        rb_adj, rb_fin = await raw_score(gt, tags_b, wi)

        rows.append({
            "guid": case.guid,
            "same_as_miner": list(tags_a) == list(res.tags),
            "mine": v_mine.adjusted, "a": v_a.adjusted, "b": v_b.adjusted,
            "raw_a": ra_adj, "raw_b": rb_adj,
            "raw_a_fin": ra_fin, "raw_b_fin": rb_fin,
            "n_base": len(base), "n_union": len(union),
            "uniq_a": v_a.n_unique, "uniq_b": v_b.n_unique,
            "both_a": v_a.n_both, "both_b": v_b.n_both,
            "surv_a": v_a.n_survived, "surv_b": v_b.n_survived,
        })
        print(f"[{wi+1}/{len(pairs)}] {case.guid[:10]} mine={v_mine.adjusted:.4f} "
              f"A={v_a.adjusted:.4f} B={v_b.adjusted:.4f} | screen-off A={ra_adj:.4f} "
              f"B={rb_adj:.4f} | same={rows[-1]['same_as_miner']}", flush=True)

    n = len(rows)
    print()
    print("=" * 90)
    print(f"n = {n} windows, {len(set(r['guid'] for r in rows))} conversations")
    print("=" * 90)
    diff = [r for r in rows if not r["same_as_miner"]]
    print(f"1. STRAW MAN CHECK")
    print(f"   arm A identical to pipeline.mine output : {n-len(diff)}/{n}")
    print(f"   mean adjusted  shipped miner (res.tags) : {statistics.mean([r['mine'] for r in rows]):.4f}")
    print(f"   mean adjusted  their arm A              : {statistics.mean([r['a'] for r in rows]):.4f}")
    print(f"   d(armA - miner)                         : "
          f"{statistics.mean([r['a']-r['mine'] for r in rows]):+.4f}")
    if diff:
        print(f"   on the {len(diff)} differing windows: miner "
              f"{statistics.mean([r['mine'] for r in diff]):.4f} vs armA "
              f"{statistics.mean([r['a'] for r in diff]):.4f}")
    print(f"   d(B - shipped miner)                    : "
          f"{statistics.mean([r['b']-r['mine'] for r in rows]):+.4f}  "
          f"W/L {sum(1 for r in rows if r['b']>r['mine'])}/{sum(1 for r in rows if r['b']<r['mine'])}")

    print()
    print(f"2. IS THE GAIN THE TAGS OR THE ENGLISH SCREEN?")
    print(f"   with screen    A {statistics.mean([r['a'] for r in rows]):.4f}  "
          f"B {statistics.mean([r['b'] for r in rows]):.4f}  "
          f"d {statistics.mean([r['b']-r['a'] for r in rows]):+.4f}  "
          f"W/L {sum(1 for r in rows if r['b']>r['a'])}/{sum(1 for r in rows if r['b']<r['a'])}")
    print(f"   screen OFF     A {statistics.mean([r['raw_a'] for r in rows]):.4f}  "
          f"B {statistics.mean([r['raw_b'] for r in rows]):.4f}  "
          f"d {statistics.mean([r['raw_b']-r['raw_a'] for r in rows]):+.4f}  "
          f"W/L {sum(1 for r in rows if r['raw_b']>r['raw_a'])}/{sum(1 for r in rows if r['raw_b']<r['raw_a'])}")
    guids = sorted(set(r["guid"] for r in rows))
    for g in guids:
        sub = [r for r in rows if r["guid"] == g]
        print(f"     {g[:10]:12s} n={len(sub):2d}  screen-on d={statistics.mean([r['b']-r['a'] for r in sub]):+.4f}"
              f"   screen-off d={statistics.mean([r['raw_b']-r['raw_a'] for r in sub]):+.4f}")

    print()
    print(f"3. COMPOSITION / COST")
    print(f"   base candidates/window {statistics.mean([r['n_base'] for r in rows]):.1f}"
          f"  -> + {statistics.mean([r['n_union'] for r in rows]):.1f} sub-phrases"
          f"  = {statistics.mean([r['n_base']+r['n_union'] for r in rows]):.1f}")
    print(f"   unique/win  A {statistics.mean([r['uniq_a'] for r in rows]):.2f}  "
          f"B {statistics.mean([r['uniq_b'] for r in rows]):.2f}")
    print(f"   both/win    A {statistics.mean([r['both_a'] for r in rows]):.2f}  "
          f"B {statistics.mean([r['both_b'] for r in rows]):.2f}")
    print(f"   survived/win A {statistics.mean([r['surv_a'] for r in rows]):.2f}  "
          f"B {statistics.mean([r['surv_b'] for r in rows]):.2f}")
    print(f"   windows with <3 unique: A {sum(1 for r in rows if r['uniq_a']<3)}  "
          f"B {sum(1 for r in rows if r['uniq_b']<3)}")


if __name__ == "__main__":
    asyncio.run(main())
