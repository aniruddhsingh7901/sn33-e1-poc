#!/usr/bin/env python3
"""ADVERSARIAL VERIFICATION of the `weighted-centroid` probe.

Three things the original probe did NOT do, each of which could hide a real
effect:

 1. Its recompose path is NOT mine()'s path. mine() protects lexical variants
    from drop_near_duplicates (`protected = variant_tags | gt_norm`), filters
    anchors by anchor_min_cos, and passes anchors= into compose(). The probe
    passed `protected=gt_norm` only, no anchors at all. Variants are documented
    as the single biggest lever (+0.0893), so the probe may have re-ranked a
    variant-stripped pool. Here every scheme goes through a path that is
    verified byte-identical to mine() when handed the unweighted centroid.

 2. It never printed n_unique / n_both, so the "tags that string-match GT stop
    feeding the 0.55 top-3-unique term" trap was never checked.

 3. Its ORACLE row is a ranking oracle inside the SAME quota composer. If
    compose() is the bottleneck, the ORACLE understates the ceiling and the
    "selection is saturated" conclusion does not follow. POOL_ORACLE here is a
    greedy composer run against the REAL target with the REAL ground-truth tags
    as the both/unique oracle: the true ceiling of this candidate pool.

Also reports conversation-level (cluster) statistics, since 38 windows from 4
conversations are ~4 independent trials, not 38.
"""

from __future__ import annotations

import asyncio
import random
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from bench.harness import score_answer
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline, replica, variants, vocab
from sn33.pipeline import Config, TASK_PROFILE, _convo_xml, compose, compose_greedy
from sn33.tags import cosine, drop_near_duplicates, normalize_all, rank_by_centroid

KIND = "conversation_tagging"
PER_CONVO = 12

SCHEMES = ("baseline_matched", "enrich_w3_matched", "ORACLE_matched", "POOL_ORACLE")


def _wmean(tags, weights, vecs):
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


class Row:
    def __init__(self):
        self.tags, self.score, self.cos = {}, {}, {}
        self.guid = ""
        self.widx = 0
        self.shipped = []
        self.n_variants_in_pool = 0
        self.n_variants_surviving = {}


async def one_window(case, window, widx, gt, cfg, do_score):
    res = await pipeline.mine(KIND, window=window, enrichment=case.enrichment_lines, cfg=cfg)
    if not res.candidates or not res.vectors or not res.predicted_gt:
        return None
    rep = await replica.replicate(
        KIND, convo_xml=_convo_xml(window), enrichment=case.enrichment_lines,
        model=cfg.gt_model, timeout=300, combine=cfg.combine, use_cache=True,
        samples=cfg.gt_samples, deadline=0,
    )
    if not rep.tags:
        return None

    combined = Utils.get_clean_tag_set(rep.tags)
    enrich_sets = [Utils.get_clean_tag_set(s) for s in rep.enrichment_tags]
    enr = set(t for s in enrich_sets for t in s)

    need = list(dict.fromkeys(combined + list(res.candidates)))
    vecs = dict(res.vectors)
    missing = [t for t in need if t not in vecs]
    if missing:
        vecs.update(await llm.embed(missing, timeout=120, use_cache=True))

    row = Row()
    row.guid, row.widx = case.guid, widx
    row.shipped = list(res.tags)

    profile = TASK_PROFILE[KIND]
    target_tags = cfg.target_tags or profile["target_tags"]
    insurance = cfg.insurance if cfg.insurance is not None else profile["insurance"]

    # ---- reconstruct mine()'s EXACT stage-2, including variants + anchors ----
    gt_norm = set(normalize_all(res.predicted_gt))
    variant_tags = (
        variants.expand(normalize_all(res.predicted_gt), per_tag=cfg.variants_per_tag)
        if cfg.use_variants else []
    )
    anchors = vocab.anchors_for(KIND, limit=cfg.anchor_pool) if cfg.use_anchors else []
    anchor_set = set(anchors)
    protected_matched = set(variant_tags) | gt_norm
    row.n_variants_in_pool = sum(1 for t in res.candidates if t in set(variant_tags))

    real = np.asarray(gt.target, dtype=np.float64)
    ests = {}
    ests["baseline_matched"] = _wmean(combined, [1.0] * len(combined), vecs)
    ests["enrich_w3_matched"] = _wmean(combined, [3.0 if t in enr else 1.0 for t in combined], vecs)
    ests["ORACLE_matched"] = real

    def matched_recompose(est, protected, use_anchors=True):
        ranked = rank_by_centroid(res.candidates, vecs, est)
        if use_anchors:
            ranked = [(t, s) for t, s in ranked
                      if t not in anchor_set or s >= cfg.anchor_min_cos]
        ranked = drop_near_duplicates(ranked, vecs, cfg.dedup_threshold, protected=protected)
        return compose(ranked, res.predicted_gt, profile, target_tags, insurance,
                       anchors=anchor_set if use_anchors else None), ranked

    for name, est in ests.items():
        if est is None:
            continue
        row.cos[name] = cosine(est, real)
        picked, _ = matched_recompose(est, protected_matched)
        row.tags[name] = picked
        row.n_variants_surviving[name] = sum(1 for t in picked if t in set(variant_tags))

    # ---- the probe's OWN (unprotected, anchorless) path, for the delta ----
    picked_probe, _ = matched_recompose(ests["baseline_matched"], gt_norm, use_anchors=False)
    row.tags["baseline_probePATH"] = picked_probe
    row.n_variants_surviving["baseline_probePATH"] = sum(
        1 for t in picked_probe if t in set(variant_tags))

    # ---- true ceiling of this pool: greedy composer, REAL target, REAL gt ----
    ranked_o = rank_by_centroid(res.candidates, vecs, real)
    ranked_o = drop_near_duplicates(ranked_o, vecs, cfg.dedup_threshold,
                                    protected=protected_matched)
    real_gt_clean = Utils.get_clean_tag_set(gt.tags)
    best = compose_greedy(ranked_o, real_gt_clean, vecs, real, profile)
    if len(best) < profile["min_tags"]:
        best = [t for t, _ in ranked_o[:target_tags]]
    row.tags["POOL_ORACLE"] = best

    if do_score:
        for name in SCHEMES:
            if name not in row.tags:
                continue
            v = await score_answer(gt, row.tags[name], KIND, model=GT_MODEL, use_cache=True)
            row.score[name] = (v.adjusted, v.final, v.n_unique, v.n_both,
                               v.n_survived, tuple(v.penalties))
        v = await score_answer(gt, row.tags["baseline_probePATH"], KIND, model=GT_MODEL, use_cache=True)
        row.score["baseline_probePATH"] = (v.adjusted, v.final, v.n_unique, v.n_both,
                                           v.n_survived, tuple(v.penalties))
        v = await score_answer(gt, res.tags, KIND, model=GT_MODEL, use_cache=True)
        row.score["SHIPPED_miner"] = (v.adjusted, v.final, v.n_unique, v.n_both,
                                      v.n_survived, tuple(v.penalties))
    return row


def cluster_boot(per_convo_deltas, iters=20000, seed=0):
    """Bootstrap over CONVERSATIONS (n=4), not windows. Honest, and wide."""
    rng = random.Random(seed)
    n = len(per_convo_deltas)
    means = sorted(
        sum(per_convo_deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)
    )
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


async def main():
    do_score = "--score" in sys.argv
    cases = load_cases(kind=KIND)
    gts = {c.guid: await real_ground_truth(c, model=GT_MODEL, use_cache=True) for c in cases}
    pairs = expand_windows(cases, per_convo=PER_CONVO, seed=0)
    print(f"{len(pairs)} (conversation, window) pairs from {len(cases)} conversations")

    cfg = Config(use_cache=True, deadline_s=600.0, call_timeout_s=300.0)
    sem = asyncio.Semaphore(6)

    async def guarded(i, case, window):
        async with sem:
            try:
                return await one_window(case, window, i, gts[case.guid], cfg, do_score)
            except Exception as e:  # noqa: BLE001
                print(f"  window {i} FAILED: {type(e).__name__}: {e}")
                return None

    rows = [r for r in await asyncio.gather(*[guarded(i, c, w) for i, (c, w) in enumerate(pairs)])
            if r is not None]
    print(f"{len(rows)} usable rows\n")

    # ---- FIDELITY CHECK: does baseline_matched reproduce mine()'s own answer? --
    exact = sum(1 for r in rows if r.tags["baseline_matched"] == r.shipped)
    same_set = sum(1 for r in rows if set(r.tags["baseline_matched"]) == set(r.shipped))
    probe_exact = sum(1 for r in rows if r.tags["baseline_probePATH"] == r.shipped)
    probe_same = sum(1 for r in rows if set(r.tags["baseline_probePATH"]) == set(r.shipped))
    print("### fidelity of each control to the SHIPPED miner answer")
    print(f"  baseline_matched   == shipped list : {exact}/{len(rows)}   same set: {same_set}/{len(rows)}")
    print(f"  baseline_probePATH == shipped list : {probe_exact}/{len(rows)}   same set: {probe_same}/{len(rows)}")
    mv = statistics.mean([r.n_variants_in_pool for r in rows])
    print(f"  mean variant tags in candidate pool: {mv:.1f}")
    for k in ("baseline_matched", "baseline_probePATH", "enrich_w3_matched", "ORACLE_matched"):
        if any(k in r.n_variants_surviving for r in rows):
            print(f"  mean variants surviving into the answer, {k:<20}: "
                  f"{statistics.mean([r.n_variants_surviving[k] for r in rows]):.2f}")

    by_guid = defaultdict(list)
    for r in rows:
        by_guid[r.guid].append(r)
    guids = sorted(by_guid)

    print("\n### cos(estimate, REAL centroid)  [matched path]")
    for name in ("baseline_matched", "enrich_w3_matched"):
        vals = [r.cos[name] for r in rows]
        d = [r.cos[name] - r.cos["baseline_matched"] for r in rows]
        print(f"  {name:<20} {statistics.mean(vals):.4f}  delta {statistics.mean(d):+.4f}")

    if not do_score:
        print("\n(no --score: end-to-end omitted)")
        print("usage:", llm.Usage.snapshot())
        return

    print("\n### END-TO-END, mine()-identical recompose path (n=%d)" % len(rows))
    hdr = (f"{'scheme':<22}{'adjusted':>9}{'final':>8}{'d_adj':>9}{'w/l':>8}"
           f"{'uniq':>6}{'both':>6}{'kept':>6}   per-conversation adjusted")
    print(hdr)
    print("-" * len(hdr))
    base = "baseline_matched"
    order = ["baseline_matched", "enrich_w3_matched", "ORACLE_matched", "POOL_ORACLE",
             "baseline_probePATH", "SHIPPED_miner"]
    per_convo_delta = {}
    for name in order:
        vals = [r.score[name] for r in rows if name in r.score]
        if not vals:
            continue
        adj = statistics.mean([v[0] for v in vals])
        fin = statistics.mean([v[1] for v in vals])
        uq = statistics.mean([v[2] for v in vals])
        bo = statistics.mean([v[3] for v in vals])
        kp = statistics.mean([v[4] for v in vals])
        ds = [r.score[name][0] - r.score[base][0] for r in rows if name in r.score]
        d = statistics.mean(ds)
        w = sum(1 for x in ds if x > 1e-9)
        l = sum(1 for x in ds if x < -1e-9)
        pc = [statistics.mean([r.score[name][0] for r in by_guid[g]]) for g in guids]
        pcd = [statistics.mean([r.score[name][0] - r.score[base][0] for r in by_guid[g]])
               for g in guids]
        per_convo_delta[name] = pcd
        per = "  ".join(f"{x:.4f}" for x in pc)
        print(f"{name:<22}{adj:>9.4f}{fin:>8.4f}{d:>+9.4f}{f'{w}/{l}':>8}"
              f"{uq:>6.1f}{bo:>6.1f}{kp:>6.1f}   {per}")

    print("\n### per-conversation delta vs baseline_matched, + CLUSTER bootstrap (n=4 convos)")
    for name, pcd in per_convo_delta.items():
        if name == base:
            continue
        lo, hi = cluster_boot(pcd)
        signs = sum(1 for x in pcd if x > 0)
        print(f"  {name:<22} per-convo {['%+.4f' % x for x in pcd]}  "
              f"pooled {statistics.mean(pcd):+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
              f"positive in {signs}/4")

    # penalties
    print("\n### penalties fired")
    for name in order:
        vals = [r.score[name] for r in rows if name in r.score]
        if not vals:
            continue
        allp = [p for v in vals for p in v[5]]
        rate = sum(1 for v in vals if v[5]) / len(vals)
        print(f"  {name:<22} rate {rate:.0%}  {({p: allp.count(p) for p in sorted(set(allp))})}")

    print("\nusage:", llm.Usage.snapshot())


if __name__ == "__main__":
    asyncio.run(main())
