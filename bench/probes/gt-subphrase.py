#!/usr/bin/env python3
"""PROBE: gt-subphrase.

Hypothesis
----------
Our best-scoring candidate family is lexical variants of predicted ground-truth
tags (top-3 cosine 0.6628). Variants today only inflect NUMBER on the head noun
(sn33/variants.py). Other cheap lexical transforms of the same source tags
should also stay semantically identical while being lexically distinct from the
ground truth (hence UNIQUE, hence feeding the 0.55 top-3 term):

    drop_head      "machine learning models" -> "machine learning"
    drop_modifier  "machine learning models" -> "learning models"
    head_only      "machine learning models" -> "models"
    reorder        "machine learning"        -> "learning machine"
    compound       "machine learning"        -> "machinelearning"

Each transform is measured SEPARATELY so we learn which one pays.

Method
------
* real ground truth per conversation via bench.faithful.real_ground_truth
  (the validator path: FULL conversation + enrichment lines)
* the miner's own pipeline is run per window to obtain (a) its predicted GT
  tags, which are the source the transforms operate on in production, and
  (b) its full candidate pool, which is the like-for-like reference
* every family's tags are embedded and scored as cosine to gt.target
* families are compared on top-3 mean cosine (0.55 of the score) and mean

Two source lists are used for the transforms:
  pred_*   from the miner's predicted GT   -> deployable today
  oracle_* from the REAL GT tags           -> separates "transform is bad"
                                              from "replica is bad"

Nothing under sn33/ or bench/ (existing files) is modified.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from sn33 import pipeline, variants
from sn33 import llm
from sn33.tags import cosine, normalize, normalize_all

SALT = "probe-gt-subphrase"


# --------------------------------------------------------------------------
# the transforms under test
# --------------------------------------------------------------------------

def t_drop_head(tag: str) -> List[str]:
    """Drop the trailing head noun."""
    w = tag.split()
    return [" ".join(w[:-1])] if len(w) >= 2 else []


def t_drop_modifier(tag: str) -> List[str]:
    """Drop the leading modifier."""
    w = tag.split()
    return [" ".join(w[1:])] if len(w) >= 2 else []


def t_head_only(tag: str) -> List[str]:
    """Keep the head noun alone."""
    w = tag.split()
    return [w[-1]] if len(w) >= 2 else []


def t_reorder(tag: str) -> List[str]:
    """Reverse the word order."""
    w = tag.split()
    return [" ".join(reversed(w))] if len(w) >= 2 else []


def t_compound(tag: str) -> List[str]:
    """Glue the words into one token (the hyphen/compound form; get_safe_tag
    turns a real hyphen into a space, so the only compound form that survives
    normalization is the concatenation)."""
    w = tag.split()
    return ["".join(w)] if len(w) >= 2 else []


TRANSFORMS = {
    "drop_head": t_drop_head,
    "drop_modifier": t_drop_modifier,
    "head_only": t_head_only,
    "reorder": t_reorder,
    "compound": t_compound,
}


def apply_transform(fn, source: Sequence[str], gt_raw: set, exclude: set) -> List[str]:
    """Transform every source tag; keep only tags that would score as UNIQUE.

    * must survive sn33.tags.normalize (else the validator silently deletes it)
    * must not string-match a ground-truth tag (else it is `both`, not `unique`)
    * must not already be in the source list (not a new candidate)
    """
    out: List[str] = []
    seen = set()
    for tag in source:
        for cand in fn(tag):
            n = normalize(cand)
            if not n or n in seen or n in exclude or n in gt_raw:
                continue
            seen.add(n)
            out.append(n)
    return out


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def fam_stats(tags: Sequence[str], vecs: Dict[str, Sequence[float]], target: np.ndarray) -> dict:
    scored = []
    for t in tags:
        v = vecs.get(t)
        if v is None or not len(v):
            continue
        scored.append(cosine(target, np.asarray(v, dtype=np.float32)))
    if not scored:
        return {"n": 0, "mean": 0.0, "top3": 0.0, "top3_pad": 0.0, "best": 0.0}
    scored.sort(reverse=True)
    top = scored[:3]
    return {
        "n": len(scored),
        "mean": statistics.mean(scored),
        "top3": statistics.mean(top),                      # no zero padding
        "top3_pad": sum(top) / 3.0,                        # the formula's own padding
        "best": scored[0],
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-convo", type=int, default=8, help="windows sampled per conversation")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    print(f"[load] {len(cases)} distinct conversations", flush=True)

    pairs = expand_windows(cases, per_convo=args.per_convo, seed=0)
    print(f"[load] {len(pairs)} (case, window) pairs across "
          f"{len(set(id(c) for c, _ in pairs))} conversations", flush=True)

    # ---- real ground truth, once per conversation (it is built from the FULL
    # conversation, so every window of a conversation shares one target) ----
    gts = {}
    for c in cases:
        gt = await real_ground_truth(c, model=GT_MODEL, use_cache=True)
        gts[c.guid] = gt
        print(f"[gt] {c.guid[:12]} tags={len(gt.tags)} ok={gt.ok()} :: {gt.tags[:8]}", flush=True)

    cfg = pipeline.Config(use_cache=True, deadline_s=600.0, call_timeout_s=300.0)

    rows = []          # one dict per window
    examples = defaultdict(list)

    for wi, (case, window) in enumerate(pairs):
        gt = gts[case.guid]
        if not gt.ok():
            continue
        res = await pipeline.mine(
            "conversation_tagging",
            window=window,
            enrichment=case.enrichment_lines,
            cfg=cfg,
        )
        pred_gt = normalize_all(res.predicted_gt)
        gt_raw = set(gt.tags)                       # the validator's both/unique key
        gt_norm = normalize_all(gt.tags)

        families: Dict[str, List[str]] = {}

        # ---- references ----
        families["ref_real_gt"] = gt_norm                                  # ceiling (scores as `both`)
        families["ref_pred_gt_unique"] = [t for t in pred_gt if t not in gt_raw]
        families["ref_variants"] = [
            t for t in variants.expand(pred_gt, per_tag=2) if t not in gt_raw
        ]
        families["ref_miner_pool"] = [t for t in res.candidates if t not in gt_raw]
        families["ref_miner_final"] = list(res.tags)

        # ---- families under test, from the miner's predicted GT ----
        excl_pred = set(pred_gt)
        for name, fn in TRANSFORMS.items():
            families[f"pred_{name}"] = apply_transform(fn, pred_gt, gt_raw, excl_pred)

        # ---- same transforms on the REAL GT (upper bound for the transform) ----
        excl_gt = set(gt_norm)
        for name, fn in TRANSFORMS.items():
            families[f"oracle_{name}"] = apply_transform(fn, gt_norm, gt_raw, excl_gt)

        # union of all five sub-phrase transforms, as a deployable pool
        union = []
        useen = set()
        for name in TRANSFORMS:
            for t in families[f"pred_{name}"]:
                if t not in useen:
                    useen.add(t)
                    union.append(t)
        families["pred_ALL_subphrase"] = union
        families["pred_variants_plus_ALL"] = list(dict.fromkeys(families["ref_variants"] + union))

        # SIZE-MATCHED CONTROL. The union carries ~5x the tags of ref_variants,
        # and top-3-of-a-bigger-bag rises on pool size alone. Truncate the union
        # to the same count so the comparison is about tag QUALITY, not count.
        k = len(families["ref_variants"])
        import random as _random
        _rng = _random.Random(1234 + wi)
        matched = union if len(union) <= k else _rng.sample(union, k)
        families["pred_ALL_sub_sizematched"] = matched

        # THE MARGINAL TEST - the actual deployment question. The miner's pool
        # ALREADY contains the number-inflection variants (pipeline.py:488).
        # Does bolting the sub-phrase family on top of it raise the top-3 that
        # selection can reach?
        families["ref_pool_PLUS_subphrase"] = list(
            dict.fromkeys(families["ref_miner_pool"] + union)
        )

        # ---- embed everything for this window in one batch ----
        allt = list(dict.fromkeys([t for fam in families.values() for t in fam]))
        vecs = await llm.embed(allt, timeout=180, use_cache=True)
        tvec = gt.target
        if tvec is None:
            continue

        row = {"guid": case.guid, "wi": wi, "n_pred_gt": len(pred_gt),
               "source": res.source, "fams": {}}
        for name, tags in families.items():
            row["fams"][name] = fam_stats(tags, vecs, tvec)
            if len(examples[name]) < 12:
                examples[name].extend(tags[:4])
        rows.append(row)

        top3s = {k: row["fams"][k]["top3"] for k in
                 ("ref_variants", "pred_ALL_subphrase", "ref_miner_pool")}
        print(f"[{wi+1}/{len(pairs)}] {case.guid[:10]} pred_gt={len(pred_gt)} "
              f"src={res.source} variants={top3s['ref_variants']:.4f} "
              f"subphrase={top3s['pred_ALL_subphrase']:.4f} "
              f"pool={top3s['ref_miner_pool']:.4f}", flush=True)

    # ----------------------------------------------------------------------
    # report
    # ----------------------------------------------------------------------
    print()
    print("=" * 92)
    print(f"POOLED  n_windows = {len(rows)}   (from {len(set(r['guid'] for r in rows))} distinct conversations)")
    print("=" * 92)
    names = list(rows[0]["fams"].keys()) if rows else []
    print(f"{'family':26s} {'tags/win':>8s} {'top3':>8s} {'top3pad':>8s} {'mean':>8s} {'best':>8s} {'wins/L vs variants':>20s}")
    base = "ref_variants"
    summary = {}
    for name in names:
        vals3 = [r["fams"][name]["top3"] for r in rows]
        vals3p = [r["fams"][name]["top3_pad"] for r in rows]
        valsm = [r["fams"][name]["mean"] for r in rows]
        valsb = [r["fams"][name]["best"] for r in rows]
        nts = [r["fams"][name]["n"] for r in rows]
        w = sum(1 for r in rows if r["fams"][name]["top3"] > r["fams"][base]["top3"])
        l = sum(1 for r in rows if r["fams"][name]["top3"] < r["fams"][base]["top3"])
        summary[name] = {
            "tags_per_window": statistics.mean(nts),
            "top3": statistics.mean(vals3),
            "top3_pad": statistics.mean(vals3p),
            "mean": statistics.mean(valsm),
            "best": statistics.mean(valsb),
            "delta_top3_vs_variants": statistics.mean(vals3) - statistics.mean([r["fams"][base]["top3"] for r in rows]),
            "wins": w, "losses": l,
        }
        print(f"{name:26s} {statistics.mean(nts):8.1f} {statistics.mean(vals3):8.4f} "
              f"{statistics.mean(vals3p):8.4f} {statistics.mean(valsm):8.4f} "
              f"{statistics.mean(valsb):8.4f} {str(w)+'/'+str(l):>20s}")

    # ---- per conversation ----
    print()
    print("=" * 92)
    print("PER CONVERSATION  (top-3 mean cosine, no padding)")
    print("=" * 92)
    guids = sorted(set(r["guid"] for r in rows))
    hdr = f"{'family':26s}" + "".join(f"{g[:8]:>11s}" for g in guids)
    print(hdr)
    for name in names:
        cells = ""
        for g in guids:
            sub = [r["fams"][name]["top3"] for r in rows if r["guid"] == g]
            cells += f"{statistics.mean(sub):11.4f}" if sub else f"{'-':>11s}"
        print(f"{name:26s}{cells}")

    print()
    print("=" * 92)
    print("PER-CONVERSATION DELTA vs ref_variants  (positive = family beats variants)")
    print("=" * 92)
    print(f"{'family':26s}" + "".join(f"{g[:8]:>11s}" for g in guids) + "   holds_in")
    for name in names:
        cells = ""
        holds = 0
        for g in guids:
            a = [r["fams"][name]["top3"] for r in rows if r["guid"] == g]
            b = [r["fams"][base]["top3"] for r in rows if r["guid"] == g]
            d = statistics.mean(a) - statistics.mean(b)
            holds += 1 if d > 0 else 0
            cells += f"{d:+11.4f}"
        print(f"{name:26s}{cells}   {holds}/{len(guids)}")

    print()
    print("=" * 92)
    print("EXAMPLE TAGS PRODUCED")
    print("=" * 92)
    for name in names:
        ex = list(dict.fromkeys(examples[name]))[:10]
        print(f"{name:26s} {ex}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "rows": rows,
                       "examples": {k: list(dict.fromkeys(v))[:20] for k, v in examples.items()}},
                      f, indent=2, default=str)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
