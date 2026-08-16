#!/usr/bin/env python3
"""PROBE: weighted-centroid.

HYPOTHESIS
  The miner estimates the scoring target as the UNWEIGHTED mean of the
  embeddings of its predicted ground-truth tags (pipeline.py:516). But the
  validator's real ground truth is combine(full_conversation_tags,
  enrichment_tags x N). If enrichment contributes proportionally more tags,
  enrichment-derived concepts should carry more weight in the real centroid, so
  a centroid estimate that UP-WEIGHTS enrichment-derived predicted tags ought to
  align better with the real target than the unweighted one.

WHAT THIS MEASURES
  1. cos(estimated centroid, REAL centroid) for a family of weighting schemes,
     per window and per conversation. Current unweighted baseline: 0.9399.
  2. The thing that actually matters: rank the miner's OWN candidate pool by
     each estimate and report the TRUE cosine (to the real centroid) of the
     top-3 it selects. A better centroid is only worth anything if it changes
     which tags get picked.
  3. With --score, the full end-to-end adjusted score: same candidate pool,
     compose() under each target estimate, scored by the real validator path.

Run:
  cd <repo> && source venv/bin/activate && set -a && source .env && set +a
  python -m bench.probes.weighted_centroid            # cosine only
  python -m bench.probes.weighted_centroid --score    # + end-to-end scoring
(this file is hyphenated, so it is run via runpy - see __main__ block usage in
 the report; use  python bench/probes/weighted-centroid.py )
"""

from __future__ import annotations

import asyncio
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from bench.harness import score_answer
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline, replica
from sn33.pipeline import Config, TASK_PROFILE, _convo_xml, compose
from sn33.tags import cosine, drop_near_duplicates, normalize_all, rank_by_centroid

KIND = "conversation_tagging"
PER_CONVO = 12         # one conversation only has 2 windows -> 2+12+12+12 = 38
# Scoring costs a validate_tags LLM call per distinct tag list, so only the
# schemes that matter go through the full validator path.
SCORE_SCHEMES = ("baseline_unweighted", "enrich_w2", "enrich_w3", "ORACLE_real_centroid")
SALT = "wcentroid"     # own cache namespace for any call this probe makes itself

# Weighting schemes. Every scheme returns (tags, weights) over CLEANED tags.
ENRICH_WEIGHTS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


# --------------------------------------------------------------------------
# centroid estimators
# --------------------------------------------------------------------------

def _wmean(tags: Sequence[str], weights: Sequence[float],
           vecs: Dict[str, Sequence[float]]) -> Optional[np.ndarray]:
    acc, tot = None, 0.0
    for t, w in zip(tags, weights):
        v = vecs.get(t)
        if v is None or not len(v) or w <= 0:
            continue
        a = np.asarray(v, dtype=np.float64) * w
        acc = a if acc is None else acc + a
        tot += w
    if acc is None or tot == 0:
        return None
    return acc / tot


def build_estimates(
    combined: List[str],
    doc_set: List[str],
    enrich_sets: List[List[str]],
    vecs: Dict[str, Sequence[float]],
) -> Dict[str, Optional[np.ndarray]]:
    """All candidate centroid estimates for one window, from cleaned tag sets."""
    doc = set(doc_set)
    enr = set(t for s in enrich_sets for t in s)

    out: Dict[str, Optional[np.ndarray]] = {}

    # -- family 1: re-weight the COMBINED list by provenance ----------------
    for w in ENRICH_WEIGHTS:
        weights = [w if t in enr else 1.0 for t in combined]
        name = "baseline_unweighted" if w == 1.0 else f"enrich_w{w:g}"
        out[name] = _wmean(combined, weights, vecs)
    # doc-only side of the same knob (drop everything enrichment touched)
    out["doc_only"] = _wmean(combined, [0.0 if t in enr else 1.0 for t in combined], vecs)
    # enrichment-only (w=inf)
    out["enrich_only"] = _wmean(combined, [1.0 if t in enr else 0.0 for t in combined], vecs)

    # -- family 2: skip the combine step, use the raw source sets ----------
    # multiplicity weighting: a tag in k source sets counts k times. With N
    # enrichment lines this is exactly "enrichment gets N times the vote".
    union_mult: List[str] = []
    for s in [doc_set] + enrich_sets:
        union_mult.extend(s)
    uniq = list(dict.fromkeys(union_mult))
    out["union_multiplicity"] = _wmean(uniq, [union_mult.count(t) for t in uniq], vecs)
    out["union_flat"] = _wmean(uniq, [1.0] * len(uniq), vecs)

    # per-SET centroids averaged equally: doc counts as much as all enrichment
    # lines put together only if there is 1 of them; with N lines enrichment
    # gets N/(N+1) of the mass.
    set_centroids = []
    for s in [doc_set] + enrich_sets:
        c = _wmean(s, [1.0] * len(s), vecs)
        if c is not None:
            set_centroids.append(c)
    out["mean_of_set_centroids"] = (
        np.mean(np.stack(set_centroids), axis=0) if set_centroids else None
    )
    # doc set weighted equal to the WHOLE enrichment block
    if set_centroids:
        doc_c = set_centroids[0]
        enr_cs = set_centroids[1:]
        if enr_cs:
            out["doc_half_enrich_half"] = 0.5 * doc_c + 0.5 * np.mean(np.stack(enr_cs), axis=0)
        else:
            out["doc_half_enrich_half"] = doc_c
    else:
        out["doc_half_enrich_half"] = None

    return out


# --------------------------------------------------------------------------
# per-window work
# --------------------------------------------------------------------------

class Row:
    __slots__ = ("guid", "widx", "cos", "top3", "mean_cos", "best", "picked",
                 "n_enrich", "n_combined", "n_enrich_derived", "score", "tags")

    def __init__(self):
        self.cos, self.top3, self.mean_cos, self.best = {}, {}, {}, {}
        self.picked, self.score, self.tags = {}, {}, {}


async def one_window(case, window, widx: int, gt, cfg: Config, do_score: bool) -> Optional[Row]:
    # 1. the miner's real answer + its real candidate pool (cache-shared with
    #    the running bench; identical inputs -> free).
    res = await pipeline.mine(
        KIND, window=window, enrichment=case.enrichment_lines, cfg=cfg
    )
    if not res.candidates or not res.vectors or not res.predicted_gt:
        return None

    # 2. the same replica again, this time keeping provenance. Same prompts,
    #    same salt, so every call is a cache hit on the calls mine() just made.
    rep = await replica.replicate(
        KIND,
        convo_xml=_convo_xml(window),
        enrichment=case.enrichment_lines,
        model=cfg.gt_model,
        timeout=300,
        combine=cfg.combine,
        use_cache=True,
        samples=cfg.gt_samples,
        deadline=0,
    )
    if not rep.tags:
        return None

    combined = Utils.get_clean_tag_set(rep.tags)
    doc_set = Utils.get_clean_tag_set(rep.doc_tags)
    enrich_sets = [Utils.get_clean_tag_set(s) for s in rep.enrichment_tags]

    # 3. vectors for everything (candidates already embedded by mine()).
    need = list(dict.fromkeys(
        combined + doc_set + [t for s in enrich_sets for t in s] + list(res.candidates)
    ))
    vecs = dict(res.vectors)
    missing = [t for t in need if t not in vecs]
    if missing:
        vecs.update(await llm.embed(missing, timeout=120, use_cache=True))

    row = Row()
    row.guid, row.widx = case.guid, widx
    row.n_enrich = len(enrich_sets)
    row.n_combined = len(combined)
    enr = set(t for s in enrich_sets for t in s)
    row.n_enrich_derived = sum(1 for t in combined if t in enr)

    ests = build_estimates(combined, doc_set, enrich_sets, vecs)
    ests["ORACLE_real_centroid"] = np.asarray(gt.target, dtype=np.float64)

    profile = TASK_PROFILE[KIND]
    target_tags = cfg.target_tags or profile["target_tags"]
    insurance = cfg.insurance if cfg.insurance is not None else profile["insurance"]
    gt_norm = set(normalize_all(res.predicted_gt))

    real = np.asarray(gt.target, dtype=np.float64)
    for name, est in ests.items():
        if est is None:
            continue
        row.cos[name] = cosine(est, real)

        # rank the miner's OWN pool by this estimate, then measure the TRUE
        # cosine of what it picked.
        ranked = rank_by_centroid(res.candidates, vecs, est)
        ranked = drop_near_duplicates(ranked, vecs, cfg.dedup_threshold, protected=gt_norm)
        picked = compose(ranked, res.predicted_gt, profile, target_tags, insurance)
        row.picked[name] = picked
        true_cos = [cosine(np.asarray(vecs[t], dtype=np.float64), real)
                    for t in picked if t in vecs]
        if true_cos:
            row.mean_cos[name] = statistics.mean(true_cos)
            row.best[name] = max(true_cos)
            # top-3 as the SELECTOR ordered them (that is what the miner ships)
            row.top3[name] = statistics.mean(sorted(true_cos, reverse=True)[:3])
        row.tags[name] = picked

    if do_score:
        for name in [n for n in row.picked if n in SCORE_SCHEMES]:
            v = await score_answer(gt, row.picked[name], KIND, model=GT_MODEL, use_cache=True)
            row.score[name] = (v.adjusted, v.final, v.n_unique, v.n_both, tuple(v.penalties))
        v = await score_answer(gt, res.tags, KIND, model=GT_MODEL, use_cache=True)
        row.score["SHIPPED_miner"] = (v.adjusted, v.final, v.n_unique, v.n_both, tuple(v.penalties))
    return row


# --------------------------------------------------------------------------

def report_table(rows: List[Row], field: str, title: str, ref: Optional[float] = None):
    names = list(dict.fromkeys(k for r in rows for k in getattr(r, field)))
    base = "baseline_unweighted"
    print(f"\n### {title}   (n={len(rows)} windows, 4 conversations)")
    hdr = f"{'scheme':<24}{'pooled':>9}{'delta':>9}{'w/l':>8}   per-conversation"
    print(hdr)
    print("-" * len(hdr))
    by_guid = defaultdict(list)
    for r in rows:
        by_guid[r.guid].append(r)
    guids = sorted(by_guid)
    for name in names:
        vals = [getattr(r, field)[name] for r in rows if name in getattr(r, field)]
        if not vals:
            continue
        pooled = statistics.mean(vals)
        pairs = [(getattr(r, field)[name] - getattr(r, field)[base])
                 for r in rows if name in getattr(r, field) and base in getattr(r, field)]
        d = statistics.mean(pairs) if pairs else 0.0
        w = sum(1 for x in pairs if x > 1e-9)
        l = sum(1 for x in pairs if x < -1e-9)
        per = "  ".join(
            f"{statistics.mean([getattr(r, field)[name] for r in by_guid[g] if name in getattr(r, field)]):.4f}"
            for g in guids
            if any(name in getattr(r, field) for r in by_guid[g])
        )
        print(f"{name:<24}{pooled:>9.4f}{d:>+9.4f}{f'{w}/{l}':>8}   {per}")
    if ref is not None:
        print(f"(reference from earlier work: {ref})")


async def main() -> None:
    do_score = "--score" in sys.argv
    cases = load_cases(kind=KIND)
    print(f"loaded {len(cases)} distinct conversations: {[c.guid[:8] for c in cases]}")
    print(f"enrichment lines per conversation: {[len(c.enrichment_lines) for c in cases]}")

    gts = {}
    for c in cases:
        gt = await real_ground_truth(c, model=GT_MODEL, use_cache=True)
        gts[c.guid] = gt
        print(f"  {c.guid[:8]}  gt tags n={len(gt.tags)}  gt sets={len(gt.sets)}")

    pairs = expand_windows(cases, per_convo=PER_CONVO, seed=0)
    print(f"\n{len(pairs)} (conversation, window) pairs")

    cfg = Config(use_cache=True, deadline_s=600.0, call_timeout_s=300.0)

    sem = asyncio.Semaphore(6)

    async def guarded(i, case, window):
        async with sem:
            try:
                return await one_window(case, window, i, gts[case.guid], cfg, do_score)
            except Exception as e:  # noqa: BLE001
                print(f"  window {i} failed: {type(e).__name__}: {e}")
                return None

    rows = [r for r in await asyncio.gather(
        *[guarded(i, c, w) for i, (c, w) in enumerate(pairs)]
    ) if r is not None]
    print(f"{len(rows)} windows produced a usable row")

    n_e = statistics.mean([r.n_enrich for r in rows])
    frac = statistics.mean([r.n_enrich_derived / max(1, r.n_combined) for r in rows])
    print(f"mean enrichment sets/window = {n_e:.2f};  "
          f"mean fraction of combined predicted-GT tags that are enrichment-derived = {frac:.3f}")

    report_table(rows, "cos", "cos(estimated centroid, REAL centroid)", ref=0.9399)
    report_table(rows, "top3", "TRUE cosine of the top-3 tags this estimate selects", ref=0.6628)
    report_table(rows, "mean_cos", "TRUE mean cosine of all selected tags")
    report_table(rows, "best", "TRUE best-tag cosine")

    if do_score:
        print("\n### end-to-end adjusted score (real validator path)")
        names = list(dict.fromkeys(k for r in rows for k in r.score))
        base = "baseline_unweighted"
        by_guid = defaultdict(list)
        for r in rows:
            by_guid[r.guid].append(r)
        guids = sorted(by_guid)
        print(f"{'scheme':<24}{'adjusted':>10}{'final':>9}{'delta_adj':>11}{'w/l':>8}   per-conversation adj")
        for name in names:
            vals = [r.score[name] for r in rows if name in r.score]
            if not vals:
                continue
            adj = statistics.mean([v[0] for v in vals])
            fin = statistics.mean([v[1] for v in vals])
            pairs_d = [r.score[name][0] - r.score[base][0]
                       for r in rows if name in r.score and base in r.score]
            d = statistics.mean(pairs_d) if pairs_d else 0.0
            w = sum(1 for x in pairs_d if x > 1e-9)
            l = sum(1 for x in pairs_d if x < -1e-9)
            per = "  ".join(
                f"{statistics.mean([r.score[name][0] for r in by_guid[g] if name in r.score]):.4f}"
                for g in guids if any(name in r.score for r in by_guid[g])
            )
            print(f"{name:<24}{adj:>10.4f}{fin:>9.4f}{d:>+11.4f}{f'{w}/{l}':>8}   {per}")

    # how often does a better centroid actually change the answer?
    for n in ("enrich_w2", "enrich_w3", "ORACLE_real_centroid"):
        diff_list = sum(1 for r in rows if r.tags.get(n) != r.tags.get("baseline_unweighted"))
        diff_top3 = sum(1 for r in rows
                        if (r.tags.get(n) or [])[:3] != (r.tags.get("baseline_unweighted") or [])[:3])
        same_set = sum(1 for r in rows
                       if set(r.tags.get(n) or []) == set(r.tags.get("baseline_unweighted") or []))
        print(f"churn vs baseline  {n:<22} list differs {diff_list}/{len(rows)}, "
              f"first-3 differ {diff_top3}/{len(rows)}, identical set {same_set}/{len(rows)}")

    # example tags: where does the best weighted scheme differ from baseline?
    print("\n### example: tags selected by baseline vs the strongest weighted scheme")
    names = [n for n in rows[0].cos if n not in ("baseline_unweighted", "ORACLE_real_centroid")]
    best_name = max(names, key=lambda n: statistics.mean(
        [r.cos[n] for r in rows if n in r.cos]))
    shown = 0
    for r in rows:
        b = r.tags.get("baseline_unweighted", [])
        x = r.tags.get(best_name, [])
        if b != x and shown < 3:
            print(f"  {r.guid[:8]} window {r.widx}")
            print(f"    baseline  : {b}")
            print(f"    {best_name:<10}: {x}")
            print(f"    only in weighted : {[t for t in x if t not in b]}")
            print(f"    only in baseline : {[t for t in b if t not in x]}")
            shown += 1
    if shown == 0:
        print(f"  no window changed its answer between baseline and {best_name}")

    print("\nusage:", llm.Usage.snapshot())


if __name__ == "__main__":
    asyncio.run(main())
