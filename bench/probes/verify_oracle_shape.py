#!/usr/bin/env python3
"""Adversarial verification of the `oracle-shape` probe.

Four specific attacks:

 1. FAMILY RANKING IS AN ORDER STATISTIC.  Section A ranks families by the mean
    of their TOP-3 candidates, but families have wildly different sizes
    (recombination 400, template 304, gt 20).  Top-3-of-400 beats top-3-of-20
    even when the two are drawn from the SAME distribution.  Re-rank every
    family at a matched candidate budget.

 2. `hub` IS THE SCORE.  OpenAI embeddings are unit-norm, so
        cos(t, mean(g)) == mean_g cos(t, g) / ||mean(g)||
    and ||mean(g)|| is constant inside a conversation.  So corr(cos, hub) is
    exactly 1.0 within a conversation and "hub explains r=0.965" is a
    restatement of the scoring rule, not a finding.  Measure it per conversation.

 3. THEY REPORTED `adjusted`, NOT `final`.  The oracle arm selects unique-only
    (`t not in gt_set`), so it can NEVER earn an exact match and eats the flat
    x0.9 `no_both_tags` penalty on every window.  The shipped miner deliberately
    buys exact matches.  Re-score both arms on `final`.

 4. IS THE ORACLE EVEN THE RIGHT COMPARISON?  It reads ground truth.  The only
    deployable claim is the mechanical recombination arm.  Reproduce it.

Uses the SAME cache salt as the original on purpose - the point is to reproduce
their exact universe, and every LLM call is then a cache hit (no new spend).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import importlib.util
import itertools
import os
import random
import statistics
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from bench.harness import score_answer
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline
from sn33.tags import cosine, normalize_all
from sn33 import scoring as sn_scoring

# import the probe under test by path (filename has a dash)
_spec = importlib.util.spec_from_file_location(
    "oracle_shape", os.path.join(ROOT, "bench", "probes", "oracle-shape.py"))
oracle_shape = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle_shape)

STOP = oracle_shape.STOP
TEMPLATE_SUFFIX = oracle_shape.TEMPLATE_SUFFIX


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-convo", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases, per_convo=args.per_convo, seed=0)
    print(f"{len(cases)} conversations -> {len(pairs)} windows\n", flush=True)

    per_case = {}
    for ci, case in enumerate(cases):
        gt = await real_ground_truth(case)
        if not gt.ok():
            continue
        gt_clean = normalize_all(Utils.get_clean_tag_set(gt.tags))
        uni = await oracle_shape.build_universe(gt_clean)
        everything = sorted({t for v in uni.values() for t in v} | set(gt_clean))
        vecs = await llm.embed(everything, use_cache=True)
        per_case[case.guid] = {
            "case": case, "gt": gt, "gt_clean": gt_clean, "uni": uni,
            "vecs": vecs, "gt_vecs": [vecs[t] for t in gt_clean if t in vecs],
            "target": np.asarray(gt.target), "ci": ci,
        }
        print(f"  conv {ci} ({case.guid[:8]}): {len(gt_clean)} gt, {len(everything)} candidates",
              flush=True)

    # ---------------------------------------------------------------- attack 1
    print("\n" + "=" * 78)
    print("ATTACK 1. FAMILY RANKING AT A MATCHED CANDIDATE BUDGET")
    print("=" * 78)
    K = 20          # gt has 20 tags; give every family the same budget
    REPS = 400
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in per_case.values():
        target, vecs, gt_set = d["target"], d["vecs"], set(d["gt_clean"])
        for fam, tags in d["uni"].items():
            cs = [cosine(target, np.asarray(vecs[t])) for t in tags if t in vecs]
            if len(cs) < 3:
                continue
            rows[fam]["n"].append(len(cs))
            rows[fam]["mean"].append(statistics.mean(cs))
            rows[fam]["top3_all"].append(statistics.mean(sorted(cs)[-3:]))
            rng = random.Random(0)
            sub = []
            for _ in range(REPS):
                s = cs if len(cs) <= K else rng.sample(cs, K)
                sub.append(statistics.mean(sorted(s)[-3:]))
            rows[fam]["top3_k20"].append(statistics.mean(sub))
    print(f"{'family':16s} {'n cand':>7s} {'mean':>8s} {'top-3 ALL':>10s} "
          f"{'top-3 @n=20':>12s} {'inflation':>10s}")
    print("-" * 70)
    for fam in sorted(rows, key=lambda f: -statistics.mean(rows[f]["top3_k20"])):
        r = rows[fam]
        a = statistics.mean(r["top3_all"])
        b = statistics.mean(r["top3_k20"])
        print(f"{fam:16s} {statistics.mean(r['n']):7.0f} {statistics.mean(r['mean']):8.4f} "
              f"{a:10.4f} {b:12.4f} {a-b:+10.4f}")

    # ---------------------------------------------------------------- attack 2
    print("\n" + "=" * 78)
    print("ATTACK 2. IS `hub` A FINDING, OR THE SCORING RULE RESTATED?")
    print("=" * 78)
    print("  OpenAI embeddings are unit-norm =>  cos(t, mean(g)) = hub(t) / ||mean(g)||")
    print("  ||mean(g)|| is a per-conversation constant, so within a conversation")
    print("  the two are exactly proportional.\n")
    print(f"  {'conv':6s} {'||centroid||':>13s} {'corr(cos,hub)':>15s} "
          f"{'corr(cos,n_words)':>18s} {'partial|hub':>12s}")
    pooled_c, pooled_h, pooled_w = [], [], []
    for d in per_case.values():
        target, vecs = d["target"], d["vecs"]
        gvs = [np.asarray(g) for g in d["gt_vecs"]]
        cs, hs, ws = [], [], []
        for t, v in vecs.items():
            v = np.asarray(v)
            cs.append(cosine(target, v))
            hs.append(statistics.mean(float(np.dot(v, g) / (np.linalg.norm(v) * np.linalg.norm(g)))
                                      for g in gvs))
            ws.append(len(t.split()))
        r_ch = float(np.corrcoef(cs, hs)[0, 1])
        r_cw = float(np.corrcoef(cs, ws)[0, 1])
        h = np.array(hs); c = np.array(cs); w = np.array(ws, dtype=float)
        cr = c - np.polyval(np.polyfit(h, c, 1), h)
        wr = w - np.polyval(np.polyfit(h, w, 1), h)
        r_p = float(np.corrcoef(cr, wr)[0, 1])
        print(f"  {d['ci']:<6d} {float(np.linalg.norm(target)):13.4f} {r_ch:15.6f} "
              f"{r_cw:18.3f} {r_p:12.3f}")
        pooled_c += cs; pooled_h += hs; pooled_w += ws
    print(f"  pooled corr(cos,hub) = {float(np.corrcoef(pooled_c, pooled_h)[0,1]):.4f}   "
          f"pooled corr(cos,n_words) = {float(np.corrcoef(pooled_c, pooled_w)[0,1]):+.3f}")

    # ---------------------------------------------------------------- attack 3+4
    print("\n" + "=" * 78)
    print("ATTACK 3. adjusted IS NOT THE SCORE. RE-RUN SECTION C ON `final`.")
    print("=" * 78)
    sem = asyncio.Semaphore(args.concurrency)
    cfg = pipeline.Config(use_cache=True, use_local=False, deadline_s=600, call_timeout_s=180)

    async def one(case, window):
        async with sem:
            d = per_case.get(case.guid)
            if d is None:
                return None
            gt, target = d["gt"], d["target"]
            gt_set = set(d["gt_clean"])
            res = await pipeline.mine(case.kind, window=window,
                                      enrichment=case.enrichment_lines, cfg=cfg)
            if not res.candidates or not res.vectors:
                return None

            allt = sorted(((t, cosine(target, np.asarray(v))) for t, v in d["vecs"].items()),
                          key=lambda x: -x[1])
            oracle_tags = [t for t, _ in allt if t not in gt_set][: args.top]
            # same oracle, but spend 2 slots on verbatim ground truth to clear
            # the flat no_both_tags penalty (what the shipped miner does)
            best_gt = [t for t, _ in allt if t in gt_set][:2]
            oracle_both = oracle_tags[: args.top - len(best_gt)] + best_gt

            v_miner = await score_answer(gt, res.tags, case.kind, model=GT_MODEL)
            v_orc = await score_answer(gt, oracle_tags, case.kind, model=GT_MODEL)
            v_orcb = await score_answer(gt, oracle_both, case.kind, model=GT_MODEL)
            return {
                "conv": d["ci"],
                "m_adj": v_miner.adjusted, "m_fin": v_miner.final,
                "m_both": v_miner.n_both, "m_pen": v_miner.penalties,
                "o_adj": v_orc.adjusted, "o_fin": v_orc.final,
                "o_both": v_orc.n_both, "o_pen": v_orc.penalties,
                "ob_adj": v_orcb.adjusted, "ob_fin": v_orcb.final,
                "ob_both": v_orcb.n_both, "ob_pen": v_orcb.penalties,
            }

    out = [r for r in await asyncio.gather(*[one(c, w) for c, w in pairs]) if r]
    byc = collections.defaultdict(list)
    for r in out:
        byc[r["conv"]].append(r)

    def m(rows, k):
        return statistics.mean(r[k] for r in rows)

    print(f"n = {len(out)} windows\n")
    print(f"{'arm':38s} {'adjusted':>9s} {'final':>9s} {'both/win':>9s}  per-conv final")
    print("-" * 100)
    for k, label in ((("m_adj", "m_fin", "m_both"), "miner as shipped"),
                     (("o_adj", "o_fin", "o_both"), "ORACLE, unique-only (their arm)"),
                     (("ob_adj", "ob_fin", "ob_both"), "ORACLE + 2 verbatim gt tags")):
        a, f, b = k
        pc = "  ".join(f"{m(v,f):.4f}" for v in [byc[i] for i in sorted(byc)])
        print(f"{label:38s} {m(out,a):9.4f} {m(out,f):9.4f} {m(out,b):9.2f}  {pc}")

    print()
    for tag, key in (("their arm", "o_fin"), ("oracle + gt matches", "ob_fin")):
        d_f = m(out, key) - m(out, "m_fin")
        w = sum(1 for r in out if r[key] > r["m_fin"])
        print(f"  delta FINAL, {tag:22s} vs miner: {d_f:+.4f}   W/L {w}/{len(out)-w}")
        pcs = [m(byc[i], key) - m(byc[i], "m_fin") for i in sorted(byc)]
        print(f"      per-conversation: " + "  ".join(f"{x:+.4f}" for x in pcs))
    d_a = m(out, "o_adj") - m(out, "m_adj")
    w = sum(1 for r in out if r["o_adj"] > r["m_adj"])
    print(f"\n  (their reported metric) delta ADJUSTED, their arm: {d_a:+.4f}  W/L {w}/{len(out)-w}")

    print("\n  penalties fired:")
    for lbl, k in (("miner", "m_pen"), ("oracle unique-only", "o_pen"), ("oracle+gt", "ob_pen")):
        c = collections.Counter(p for r in out for p in r[k])
        print(f"    {lbl:20s} {dict(c) or 'none'}")

    # ---------------------------------------------------------------- attack 4
    print("\n" + "=" * 78)
    print("ATTACK 4. HOW MUCH OF THE ORACLE IS REACHABLE WITHOUT GROUND TRUTH?")
    print("=" * 78)
    print("  The oracle picks from a universe BUILT FROM gt and ranks it BY the real")
    print("  centroid. Neither is available at mine time. The only mechanical, gt-free")
    print("  part is recombination/template over the miner's PREDICTED gt - section E")
    print("  of the original probe. Its numbers were omitted from the report.")


if __name__ == "__main__":
    asyncio.run(main())
