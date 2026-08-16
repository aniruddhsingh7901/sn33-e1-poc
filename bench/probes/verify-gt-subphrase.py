#!/usr/bin/env python3
"""ADVERSARIAL VERIFICATION of the `gt-subphrase` finding.

Their arm B builds the sub-phrase family with

    apply_transform(fn, pred_gt, gt_raw, set(pred_gt))
                              ^^^^^^ the REAL ground truth

i.e. any transform output that string-matches a tag in the *unseen* ground
truth is discarded before the pool is built.  A production miner cannot do
that.  This script adds the deployable arm (no oracle filter) alongside theirs
so the size of the leak is measured rather than asserted.

Arms (all identical except the one variable):
  A_baseline        the miner's own candidate pool
  B_theirs          + sub-phrase family, oracle-filtered against real GT
  C_nooracle        + sub-phrase family, NO oracle filter  <- deployable
  D_safe_nooracle   + drop_head/drop_modifier only, no oracle filter

Also records, per window: penalties, unique/both counts, how many sub-phrase
tags were submitted and how many survived the English screen, and a fidelity
check that arm A reproduces pipeline.mine's own answer (so arm A is not a
straw man).

Run once per PYTHONHASHSEED; Utils.get_clean_tag_set returns list(set) so the
English-screen prompt order - and therefore the score - is per-process random.
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
from bench.harness import score_answer, validate_tag_set
from sn33 import pipeline, variants, vocab
from sn33.tags import normalize, normalize_all

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


def transform_nooracle(fn, source, exclude):
    """Their apply_transform with the real-GT filter REMOVED."""
    out, seen = [], set()
    for tag in source:
        for cand in fn(tag):
            n = normalize(cand)
            if not n or n in seen or n in exclude:
                continue
            seen.add(n)
            out.append(n)
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-convo", type=int, default=8)
    ap.add_argument("--tag", default="h0")
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases, per_convo=args.per_convo, seed=0)
    gts = {c.guid: await real_ground_truth(c, model=GT_MODEL, use_cache=True) for c in cases}

    cfg = pipeline.Config(use_cache=True, deadline_s=600.0, call_timeout_s=300.0)
    profile = pipeline.TASK_PROFILE["conversation_tagging"]

    ARMS = ["A_baseline", "B_theirs", "C_nooracle", "D_safe_nooracle"]
    out = []
    fidelity = {"same": 0, "diff": 0}
    leak = {"removed_by_oracle": 0, "transform_outputs": 0}

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
        excl = set(pred_gt)

        fam_o = {n: [t for t in _p1.apply_transform(f, pred_gt, gt_raw, excl) if t not in base]
                 for n, f in TRANSFORMS.items()}
        fam_n = {n: [t for t in transform_nooracle(f, pred_gt, excl) if t not in base]
                 for n, f in TRANSFORMS.items()}

        allo = list(dict.fromkeys([t for n in TRANSFORMS for t in fam_o[n]]))
        alln = list(dict.fromkeys([t for n in TRANSFORMS for t in fam_n[n]]))
        safen = list(dict.fromkeys([t for n in SAFE for t in fam_n[n]]))
        leak["transform_outputs"] += len(alln)
        leak["removed_by_oracle"] += len(alln) - len(allo)

        pools = {"A_baseline": [], "B_theirs": allo,
                 "C_nooracle": alln, "D_safe_nooracle": safen}
        extra_sets = {k: set(v) for k, v in pools.items()}

        row = {"guid": case.guid, "wi": wi, "n_base": len(base)}
        for k, extra in pools.items():
            tags = await select(base + extra, pred_gt, v2, anchors, cfg, profile)
            if k == "A_baseline":
                # fidelity: does the re-implemented selection equal the miner's?
                if list(tags) == list(res.tags):
                    fidelity["same"] += 1
                else:
                    fidelity["diff"] += 1
            v = await score_answer(gt, tags, "conversation_tagging", model=GT_MODEL,
                                   seed=wi, use_cache=True)
            surv = await validate_tag_set(tags, model=GT_MODEL, seed=wi, use_cache=True)
            picked = [t for t in tags if t in extra_sets[k]]
            row[k] = {"adjusted": v.adjusted, "final": v.final, "pen": v.penalties,
                      "n_unique": v.n_unique, "n_both": v.n_both,
                      "n_sub_picked": len(picked),
                      "n_sub_survived": len([t for t in picked if t in surv]),
                      "n_survived": v.n_survived, "n_submitted": v.n_submitted}
        out.append(row)
        print(f"[{wi+1}/{len(pairs)}] {case.guid[:10]} " +
              " ".join(f"{k[:11]}={row[k]['adjusted']:.4f}" for k in ARMS), flush=True)

    path = f"/tmp/claude-1000/-home-anirudh-bittensor-conversation-genome-project/90b878db-c14b-471e-8482-0dddd9d0390f/scratchpad/verify_{args.tag}.json"
    with open(path, "w") as f:
        json.dump({"rows": out, "fidelity": fidelity, "leak": leak}, f)

    print()
    print(f"arm-A fidelity vs pipeline.mine: identical {fidelity['same']}/{fidelity['same']+fidelity['diff']}")
    print(f"oracle filter removed {leak['removed_by_oracle']}/{leak['transform_outputs']} transform outputs")
    print(f"{'arm':18s} {'adjusted':>9s} {'final':>9s} {'d_adj':>9s} {'W/L':>8s} {'pen%':>6s} {'uniq':>5s} {'surv':>5s} {'sub_surv':>10s}")
    for k in ARMS:
        adj = statistics.mean([r[k]["adjusted"] for r in out])
        fin = statistics.mean([r[k]["final"] for r in out])
        d = [r[k]["adjusted"] - r["A_baseline"]["adjusted"] for r in out]
        w = sum(1 for x in d if x > 1e-9)
        l = sum(1 for x in d if x < -1e-9)
        pen = 100 * sum(1 for r in out if r[k]["pen"]) / len(out)
        uq = statistics.mean([r[k]["n_unique"] for r in out])
        sv = statistics.mean([r[k]["n_survived"] for r in out])
        sp = sum(r[k]["n_sub_picked"] for r in out)
        ss = sum(r[k]["n_sub_survived"] for r in out)
        print(f"{k:18s} {adj:9.4f} {fin:9.4f} {statistics.mean(d):+9.4f} "
              f"{str(w)+'/'+str(l):>8s} {pen:6.1f} {uq:5.1f} {sv:5.1f} "
              f"{str(ss)+'/'+str(sp):>10s}")

    print("\nPER CONVERSATION d_adjusted vs A_baseline")
    guids = sorted(set(r["guid"] for r in out))
    print(f"{'arm':18s}" + "".join(f"{g[:10]:>12s}" for g in guids) + "   holds")
    for g in guids:
        pass
    for k in ARMS[1:]:
        cells, holds = "", 0
        for g in guids:
            sub = [r[k]["adjusted"] - r["A_baseline"]["adjusted"] for r in out if r["guid"] == g]
            m = statistics.mean(sub)
            holds += 1 if m > 0 else 0
            cells += f"{m:+12.4f}"
        print(f"{k:18s}{cells}   {holds}/{len(guids)}")
    print("n per conversation: " + ", ".join(
        f"{g[:10]}={sum(1 for r in out if r['guid']==g)}" for g in guids))


if __name__ == "__main__":
    asyncio.run(main())
