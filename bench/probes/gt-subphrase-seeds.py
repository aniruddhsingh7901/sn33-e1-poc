#!/usr/bin/env python3
"""PROBE gt-subphrase, part 6: resolve the effect size across hash seeds.

Parts 2-5 each ran a valid PAIRED A/B but produced different effect sizes
(+0.0078 to +0.0200) and different baselines (0.6383 to 0.6465). The cause is
not the bench:

    Utils.get_clean_tag_set   ->  return list(cleanTags)        # a set

`validate_tag_set` builds the validator's English-screen prompt by joining that
list. Python randomizes string hashing per process, so the same 12 tags produce
a different prompt ORDER in every process, the screen deletes a different
subset, and the score moves. Production sees this too - it is validator noise,
not a harness artifact - but it means one run cannot resolve a small effect.

This runs the decisive arms under several fixed PYTHONHASHSEED values and pools
them. Fewer arms, more replication.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util as _ilu
import json
import os
import statistics
import sys

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
TRANSFORMS = _p1.TRANSFORMS
SAFE = ["drop_head", "drop_modifier"]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-convo", type=int, default=8)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases, per_convo=args.per_convo, seed=0)
    gts = {c.guid: await real_ground_truth(c, model=GT_MODEL, use_cache=True) for c in cases}

    cfg = pipeline.Config(use_cache=True, deadline_s=600.0, call_timeout_s=300.0)
    profile = pipeline.TASK_PROFILE["conversation_tagging"]

    arm_names = ["A_baseline", "only_drop_head", "ALL", "SAFE_subset"]
    out = []

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

        fam = {n: [t for t in _p1.apply_transform(f, pred_gt, gt_raw, set(pred_gt)) if t not in base]
               for n, f in TRANSFORMS.items()}
        allu = list(dict.fromkeys([t for n in TRANSFORMS for t in fam[n]]))
        safe = list(dict.fromkeys([t for n in SAFE for t in fam[n]]))

        pools = {"A_baseline": [], "only_drop_head": fam["drop_head"],
                 "ALL": allu, "SAFE_subset": safe}
        row = {"guid": case.guid, "wi": wi}
        for k, extra in pools.items():
            tags = await select(base + extra, pred_gt, v2, anchors, cfg, profile)
            v = await score_answer(gt, tags, "conversation_tagging", model=GT_MODEL,
                                   seed=wi, use_cache=True)
            row[k] = {"adjusted": v.adjusted, "final": v.final, "pen": v.penalties}
        out.append(row)
        print(f"[{wi+1}] {case.guid[:10]} " +
              " ".join(f"{k[:12]}={row[k]['adjusted']:.4f}" for k in arm_names), flush=True)

    path = f"/tmp/claude-1000/-home-anirudh-bittensor-conversation-genome-project/90b878db-c14b-471e-8482-0dddd9d0390f/scratchpad/seeds_{args.tag}.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[out] {path}")


if __name__ == "__main__":
    asyncio.run(main())
