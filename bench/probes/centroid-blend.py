#!/usr/bin/env python3
"""PROBE: centroid-blend.

Hypothesis
----------
The validator scores every tag as cosine to ONE vector: the MEAN of the
ground-truth tag embeddings. A single ground-truth tag names a single theme, so
it pulls away from that mean - measured, GT tags average only ~0.645 cosine to
their own centroid. A phrase that BLENDS two or three dominant themes
("housing policy zoning reform") should sit nearer the mean than any single
theme does, and - being a different string from every GT tag - it counts as
`unique`, which is the 0.55 term.

Method
------
For N windows drawn evenly from all 4 conversations:
  * build the REAL ground truth (bench.faithful.real_ground_truth)
  * run the CURRENT miner on the window, keep its candidate pool + final tags
    (this is the like-for-like reference, not a number in isolation)
  * build blend families and embed everything
  * report mean / top-3 mean / best cosine to the real centroid, per family,
    pooled and per conversation

Stage 2 (--score) only matters if a blend family wins on cosine: it swaps the
best ESTIMATED-centroid blends into the miner's answer and scores both answers
end to end through the validator's own path, including the English screen.

    python bench/probes/centroid-blend.py --n 32
    python bench/probes/centroid-blend.py --n 32 --score
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import random
import statistics
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from bench.harness import score_answer, validate_tag_set
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline
from sn33.tags import centroid, cosine, normalize, normalize_all, parse_tag_list
from sn33.variants import expand as expand_variants

SALT = "probe-centroid-blend"


# --------------------------------------------------------------------------
# Blend construction
# --------------------------------------------------------------------------

def blend(tags: Sequence[str]) -> Optional[str]:
    """Concatenate tags into one phrase, deduping repeated words.

    Returns None when the result cannot survive sn33.tags.normalize (>50 chars,
    two or more orphaned single letters, non-ascii letters, ...).
    """
    words: List[str] = []
    seen = set()
    for t in tags:
        for w in str(t).split():
            if w in seen:
                continue
            seen.add(w)
            words.append(w)
    if not words:
        return None
    return normalize(" ".join(words))


def blends_from(tags: Sequence[str], k: int, cap: int, rng: random.Random) -> List[str]:
    """All (or a sample of `cap`) k-combinations of `tags`, blended.

    Combinations are enumerated from the WHOLE list, not from a top-k chosen by
    the real centroid. That keeps the family symmetric with the reference
    families, whose top-3 is likewise the best 3 of everything available.
    """
    tags = [t for t in tags if t]
    if len(tags) < k:
        return []
    combos = list(itertools.combinations(tags, k))
    if len(combos) > cap:
        combos = rng.sample(combos, cap)
    out: List[str] = []
    seen = set()
    for c in combos:
        b = blend(c)
        if b and b not in seen:
            seen.add(b)
            out.append(b)
    return out


BLEND_LLM_PROMPT = (
    "Below is an excerpt from a longer conversation.\n\n"
    "Write {n} short keyword phrases that each FUSE two or three of the main "
    "themes of this material into a single natural phrase - the kind of phrase "
    "a librarian would use to file the whole episode under, e.g. "
    "'urban housing policy reform', 'ai regulation and privacy law'. "
    "Each phrase must read as normal English, 3-6 words, lowercase, no "
    "punctuation. Comma separated, no commentary.\n\nExcerpt:\n{content}"
)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def fam_stats(tags: Sequence[str], vecs: Dict[str, List[float]], target: np.ndarray,
              est_target: Optional[np.ndarray] = None) -> Optional[dict]:
    """Cosines to the REAL centroid, plus the deployable top-3.

    `top3` is the best 3 of the whole family by real cosine - an oracle, and it
    flatters big families simply because a bigger pool has a bigger maximum.
    `top3_est` removes that confound: pick the 3 the miner would actually pick
    (highest cosine to the ESTIMATED centroid it builds from its own replica),
    then report what those three are really worth.
    """
    usable = [t for t in tags if t in vecs]
    cs = [cosine(target, np.asarray(vecs[t], dtype=np.float32)) for t in usable]
    if not cs:
        return None
    out = {
        "mean": statistics.mean(cs),
        "top3": statistics.mean(sorted(cs)[-3:]),
        "best": max(cs),
        "n": len(cs),
    }
    if est_target is not None:
        ranked = sorted(
            usable, key=lambda t: -cosine(est_target, np.asarray(vecs[t], dtype=np.float32))
        )[:3]
        out["top3_est"] = statistics.mean(
            cosine(target, np.asarray(vecs[t], dtype=np.float32)) for t in ranked
        )
        out["pick_est"] = ranked
    return out


async def one_window(case, window, args, rng_seed: int) -> Optional[dict]:
    gt = await real_ground_truth(case)
    if not gt.ok():
        return None
    target = np.asarray(gt.target, dtype=np.float32)
    rng = random.Random(rng_seed)

    # --- the current miner, exactly as bench/run_faithful.py runs it ---
    res = await pipeline.mine(
        case.kind,
        window=window,
        enrichment=case.enrichment_lines,
        cfg=pipeline.Config(
            use_cache=True, use_local=False, deadline_s=600.0, call_timeout_s=180.0
        ),
    )

    gt_tags = normalize_all(Utils.get_clean_tag_set(gt.tags))
    pred_gt = normalize_all(res.predicted_gt)

    text = "\n".join(str(l[1]) for l in window if len(l) >= 2)
    raw = await llm.chat(
        BLEND_LLM_PROMPT.format(n=args.per_family, content=text[:6000]),
        model=GT_MODEL, timeout=180, use_cache=True, salt=SALT,
    )
    llm_blends = parse_tag_list(raw)

    fams: Dict[str, List[str]] = {
        # references
        "GROUND_TRUTH": gt_tags,
        "gt_variants": expand_variants(gt_tags, per_tag=2),
        "miner_pool": normalize_all(res.candidates),
        "miner_final": normalize_all(res.tags),
        # oracle-side blends: what the ceiling looks like if our GT guess were perfect
        "blend_gt_2": blends_from(gt_tags, 2, args.cap, rng),
        "blend_gt_3": blends_from(gt_tags, 3, args.cap, rng),
        # deployable blends: built only from what the miner actually has
        "blend_pred_2": blends_from(pred_gt, 2, args.cap, rng),
        "blend_pred_3": blends_from(pred_gt, 3, args.cap, rng),
        "blend_llm": llm_blends,
    }
    # control: is any gain about BLENDING, or just about long phrases?
    fams["pred_gt"] = pred_gt

    everything = sorted({t for v in fams.values() for t in v})
    vecs = await llm.embed(everything, use_cache=True, timeout=120)

    # The miner's own estimate of the target: centroid of its predicted GT tags.
    est_vecs = await llm.embed(pred_gt, use_cache=True, timeout=120)
    est_target = centroid([est_vecs[t] for t in pred_gt if t in est_vecs]) if pred_gt else None

    stats = {}
    for fam, tags in fams.items():
        s = fam_stats(tags, vecs, target, est_target)
        if s:
            stats[fam] = s

    # best few blends by REAL cosine, for eyeballing plausibility
    best_blends = sorted(
        ((cosine(target, np.asarray(vecs[t], dtype=np.float32)), t)
         for t in set(fams["blend_pred_2"]) | set(fams["blend_pred_3"]) if t in vecs),
        reverse=True,
    )[:5]

    out = {
        "guid": case.guid,
        "stats": stats,
        "best_blends": best_blends,
        "n_gt": len(gt_tags),
        "n_pred_gt": len(pred_gt),
        "gt_tags": gt_tags,
        "miner_final": fams["miner_final"],
    }

    # --- stage 2: does a blend-augmented answer actually score better? ---
    if args.score:
        cand_blends = list(dict.fromkeys(fams["blend_pred_2"] + fams["blend_pred_3"]))
        base = list(fams["miner_final"])
        if est_target is not None and cand_blends and base:
            bl_vecs = await llm.embed(cand_blends, use_cache=True, timeout=120)
            ranked = sorted(
                ((cosine(est_target, np.asarray(bl_vecs[t], dtype=np.float32)), t)
                 for t in cand_blends if t in bl_vecs),
                reverse=True,
            )
            swapped = base[: max(0, len(base) - args.swap)] + [t for _, t in ranked[: args.swap]]
            swapped = normalize_all(swapped)
            v_base = await score_answer(gt, base, case.kind, model=GT_MODEL, use_cache=True)
            v_swap = await score_answer(gt, swapped, case.kind, model=GT_MODEL, use_cache=True)
            out["score"] = {
                "base_adjusted": v_base.adjusted, "base_final": v_base.final,
                "swap_adjusted": v_swap.adjusted, "swap_final": v_swap.final,
                "base_survived": v_base.n_survived, "swap_survived": v_swap.n_survived,
                "base_unique": v_base.n_unique, "swap_unique": v_swap.n_unique,
                "swap_tags": swapped,
                "blends_added": [t for _, t in ranked[: args.swap]],
            }
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="total windows")
    ap.add_argument("--per-family", type=int, default=15)
    ap.add_argument("--cap", type=int, default=90, help="max combinations per blend family")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--score", action="store_true", help="stage 2: end-to-end scoring")
    ap.add_argument("--swap", type=int, default=4, help="blends swapped into the answer")
    ap.add_argument("--screen", action="store_true", help="run the validator English screen on blends")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    per = max(1, args.n // max(1, len(cases)))
    pairs = expand_windows(cases, per_convo=per, seed=0)
    print(f"{len(pairs)} windows from {len(cases)} conversations ({per}/conversation)\n", flush=True)

    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(i, case, window):
        async with sem:
            try:
                return await one_window(case, window, args, rng_seed=i)
            except Exception as e:  # noqa: BLE001
                print(f"  window {i} failed: {type(e).__name__}: {e}", flush=True)
                return None

    results = await asyncio.gather(*[guarded(i, c, w) for i, (c, w) in enumerate(pairs)])
    rows = [r for r in results if r]
    print(f"{len(rows)} windows completed\n")
    if not rows:
        return

    fam_names = list(rows[0]["stats"].keys())
    by_convo: Dict[str, List[dict]] = {}
    for r in rows:
        by_convo.setdefault(r["guid"], []).append(r)

    def agg(rs, fam, key):
        vals = [r["stats"][fam][key] for r in rs if fam in r["stats"]]
        return statistics.mean(vals) if vals else float("nan")

    print("=" * 88)
    print("POOLED  (cosine to the REAL centroid)   n =", len(rows))
    print("=" * 88)
    print(f"{'family':<16} {'top3':>8} {'mean':>8} {'best':>8} {'tags/win':>9}")
    order = sorted(fam_names, key=lambda f: -agg(rows, f, "top3"))
    for fam in order:
        print(f"{fam:<16} {agg(rows, fam, 'top3'):>8.4f} {agg(rows, fam, 'mean'):>8.4f} "
              f"{agg(rows, fam, 'best'):>8.4f} {agg(rows, fam, 'n'):>9.1f}")

    print("\n" + "=" * 88)
    print("DEPLOYABLE SELECTION: top-3 picked by the ESTIMATED centroid, valued")
    print("against the REAL one. Removes the 'bigger family has a bigger max' confound.")
    print("=" * 88)
    print(f"{'family':<16} {'top3_est':>9} {'vs pool':>9}   per-conversation")
    order_e = sorted(fam_names, key=lambda f: -agg(rows, f, "top3_est"))
    for fam in order_e:
        d = [r["stats"][fam]["top3_est"] - r["stats"]["miner_pool"]["top3_est"]
             for r in rows if fam in r["stats"] and "miner_pool" in r["stats"]]
        percv = " ".join(
            f"{statistics.mean([r['stats'][fam]['top3_est'] - r['stats']['miner_pool']['top3_est'] for r in by_convo[g] if fam in r['stats']]):+.4f}"
            for g in by_convo
        )
        wl = f"W/L {sum(1 for x in d if x>0)}/{sum(1 for x in d if x<0)}"
        print(f"{fam:<16} {agg(rows, fam, 'top3_est'):>9.4f} {statistics.mean(d):>+9.4f}   {percv}  {wl}")

    print("\n" + "=" * 88)
    print("PER CONVERSATION  (top-3 mean cosine)")
    print("=" * 88)
    heads = list(by_convo.keys())
    print(f"{'family':<16} " + " ".join(f"{g[:8]:>9}" for g in heads) + f" {'n/convo':>9}")
    for fam in order:
        cells = " ".join(f"{agg(by_convo[g], fam, 'top3'):>9.4f}" for g in heads)
        print(f"{fam:<16} {cells}")
    print(f"{'windows':<16} " + " ".join(f"{len(by_convo[g]):>9d}" for g in heads))

    # paired per-window deltas against the two references
    print("\n" + "=" * 88)
    print("PAIRED per-window delta in top-3 cosine  (family - reference)")
    print("=" * 88)
    for ref in ("gt_variants", "miner_pool", "GROUND_TRUTH"):
        print(f"\n  vs {ref}:")
        for fam in order:
            if fam == ref:
                continue
            d = [r["stats"][fam]["top3"] - r["stats"][ref]["top3"]
                 for r in rows if fam in r["stats"] and ref in r["stats"]]
            if not d:
                continue
            wins = sum(1 for x in d if x > 0)
            per_c = []
            for g in heads:
                dg = [r["stats"][fam]["top3"] - r["stats"][ref]["top3"]
                      for r in by_convo[g] if fam in r["stats"] and ref in r["stats"]]
                per_c.append(statistics.mean(dg) if dg else float("nan"))
            print(f"    {fam:<16} {statistics.mean(d):+.4f}  W/L {wins}/{len(d)-wins}   "
                  f"per-convo " + " ".join(f"{x:+.4f}" for x in per_c))

    print("\n" + "=" * 88)
    print("EXAMPLE BLENDS (best by real cosine, from predicted-GT blends)")
    print("=" * 88)
    for r in rows[:6]:
        print(f"  [{r['guid'][:8]}] gt: {', '.join(r['gt_tags'][:6])}")
        for c, t in r["best_blends"]:
            print(f"      {c:.4f}  {t}")

    # hygiene: do blends string-match ground truth (-> `both`, not `unique`)?
    exact = 0
    total = 0
    for r in rows:
        gtset = set(r["gt_tags"])
        for _, t in r["best_blends"]:
            total += 1
            exact += t in gtset
    print(f"\nhygiene: {total} top blends inspected, {exact} string-match a GT tag "
          f"(the rest count as UNIQUE and feed the 0.55 term)")

    if args.screen:
        print("\n" + "=" * 88)
        print("VALIDATOR ENGLISH SCREEN on the best blends (LlmLib.validate_tag_set)")
        print("=" * 88)
        kept_n = sub_n = 0
        for r in rows[: min(12, len(rows))]:
            cand = [t for _, t in r["best_blends"]]
            if not cand:
                continue
            kept = await validate_tag_set(cand, model=GT_MODEL, use_cache=True)
            kept_n += len(kept)
            sub_n += len(cand)
            dropped = [t for t in cand if t not in kept]
            print(f"  [{r['guid'][:8]}] kept {len(kept)}/{len(cand)}"
                  + (f"   DROPPED: {dropped}" if dropped else ""))
        if sub_n:
            print(f"  survival {kept_n}/{sub_n} = {kept_n/sub_n:.1%}")

    if args.score:
        sc = [r["score"] for r in rows if "score" in r]
        if sc:
            print("\n" + "=" * 88)
            print(f"END-TO-END SCORE  n={len(sc)}  (swap {args.swap} blends into the answer)")
            print("=" * 88)
            ba = statistics.mean(s["base_adjusted"] for s in sc)
            sa = statistics.mean(s["swap_adjusted"] for s in sc)
            bf = statistics.mean(s["base_final"] for s in sc)
            sf = statistics.mean(s["swap_final"] for s in sc)
            d = [s["swap_adjusted"] - s["base_adjusted"] for s in sc]
            print(f"  baseline miner   adjusted {ba:.4f}  final {bf:.4f}")
            print(f"  +blends          adjusted {sa:.4f}  final {sf:.4f}")
            print(f"  delta adjusted   {statistics.mean(d):+.4f}  "
                  f"W/L {sum(1 for x in d if x>0)}/{sum(1 for x in d if x<0)}")
            print(f"  survived tags    base {statistics.mean(s['base_survived'] for s in sc):.2f}"
                  f"  swap {statistics.mean(s['swap_survived'] for s in sc):.2f}")
            print(f"  unique tags      base {statistics.mean(s['base_unique'] for s in sc):.2f}"
                  f"  swap {statistics.mean(s['swap_unique'] for s in sc):.2f}")
            for g in heads:
                dg = [r["score"]["swap_adjusted"] - r["score"]["base_adjusted"]
                      for r in by_convo[g] if "score" in r]
                if dg:
                    print(f"    {g[:8]}  n={len(dg):>2}  delta {statistics.mean(dg):+.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, default=str, indent=1)
        print(f"\nwrote {args.out}")

    print("\nusage:", json.dumps(llm.Usage.snapshot()))


if __name__ == "__main__":
    asyncio.run(main())
