#!/usr/bin/env python3
"""oracle-shape: work backwards from the answer.

Every other probe guesses a generation recipe and measures what it gets. This
one does the opposite: it builds a large universe of candidate phrases AROUND
the real ground truth, embeds all of them, ranks them against the REAL centroid,
and then *characterises the winners*. The output is meant to be a description of
the shape of a high-cosine tag, expressed in numbers, that generation can then
be aimed at.

It reads ground truth, so nothing here is deployable. It is a diagnostic.

Universe of candidates, per conversation (all derived from gt.tags):
  gt              the ground-truth tags themselves          (score as `both`)
  variants        sn33.variants morphological expansion     (`unique`)
  paraphrase      LLM synonym/near-synonym of each gt tag
  hypernym        LLM broader category for each gt tag
  hyponym         LLM narrower instance for each gt tag
  fusion          LLM phrases combining 2-3 gt concepts
  umbrella        LLM: name the single overall topic of the whole gt list, N ways
  template        mechanical "<gt tag> <suffix>" / "<prefix> <gt tag>"
  recombination   mechanical 2-word pairs from gt vocabulary

Reference points already measured on this corpus:
  gt_variants top-3 cosine 0.6628, GROUND_TRUTH 0.6642, current miner adjusted 0.6357.
The miner's own per-window candidate pool is regenerated here so the delta is
like-for-like rather than a number in isolation.

  python bench/probes/oracle-shape.py --per-convo 7
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import itertools
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import GT_MODEL, expand_windows, load_cases, real_ground_truth
from bench.harness import score_answer
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline
from sn33.tags import cosine, normalize, normalize_all, parse_tag_list
from sn33.variants import expand as expand_variants

SALT = "oracle-shape-v1"          # our own cache namespace, per the brief

TEMPLATE_SUFFIX = ["strategy", "practices", "research", "trends", "concepts",
                   "industry", "landscape", "insights", "analysis", "topics"]
TEMPLATE_PREFIX = ["modern", "applied", "global", "digital", "advanced"]

STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with",
        "by", "from", "as", "at", "is", "are"}


# --------------------------------------------------------------------------
# candidate universe
# --------------------------------------------------------------------------

async def _ask(prompt: str) -> list:
    raw = await llm.chat(prompt, model=GT_MODEL, timeout=180, use_cache=True, salt=SALT)
    return parse_tag_list(raw)


async def build_universe(gt_tags: list) -> dict:
    """Hundreds of phrases arranged around the ground-truth tags."""
    listing = ", ".join(gt_tags)
    fams: dict = {}

    fams["gt"] = list(gt_tags)
    fams["variants"] = [t for t in expand_variants(gt_tags, per_tag=3, limit=400)
                        if t not in set(gt_tags)]

    jobs = {
        "paraphrase": (
            "For EACH of these topic tags, give 3 near-synonyms or paraphrases that mean "
            "almost the same thing but use different words.\n"
            f"Tags: {listing}\n"
            "Answer as one comma separated list, lowercase, 1-4 words each, no commentary."
        ),
        "hypernym": (
            "For EACH of these topic tags, give 2 BROADER categories it belongs to.\n"
            f"Tags: {listing}\n"
            "Answer as one comma separated list, lowercase, 1-4 words each, no commentary."
        ),
        "hyponym": (
            "For EACH of these topic tags, give 2 NARROWER, more specific sub-topics.\n"
            f"Tags: {listing}\n"
            "Answer as one comma separated list, lowercase, 1-4 words each, no commentary."
        ),
        "fusion": (
            "Here is a set of topic tags describing one conversation.\n"
            f"Tags: {listing}\n"
            "Write 40 NEW short phrases, each of which blends TWO OR THREE of these "
            "concepts into a single idea. lowercase, 2-5 words, comma separated, no commentary."
        ),
        "umbrella": (
            "Here is a set of topic tags describing one conversation.\n"
            f"Tags: {listing}\n"
            "Name the single overall subject that ALL of these tags share, in 40 different "
            "wordings - from narrow to very broad. Each is one short phrase. lowercase, "
            "1-4 words, comma separated, no commentary."
        ),
    }
    got = await asyncio.gather(*[_ask(p) for p in jobs.values()])
    for name, tags in zip(jobs.keys(), got):
        fams[name] = tags

    # mechanical families - no LLM call, so they cost the miner nothing at runtime
    tmpl = []
    for t in gt_tags:
        for s in TEMPLATE_SUFFIX:
            tmpl.append(f"{t} {s}")
        for p in TEMPLATE_PREFIX:
            tmpl.append(f"{p} {t}")
    fams["template"] = tmpl

    words = [w for t in gt_tags for w in t.split() if w not in STOP and len(w) > 2]
    words = list(dict.fromkeys(words))
    recomb = [f"{a} {b}" for a, b in itertools.permutations(words, 2)]
    fams["recombination"] = recomb[:400]

    # normalize everything, drop empties, keep family membership
    out = {}
    for k, v in fams.items():
        out[k] = normalize_all(v)
    return out


# --------------------------------------------------------------------------
# characterisation features
# --------------------------------------------------------------------------

def features(tag: str, gt_tags: list, gt_words: set, vec, gt_vecs: list) -> dict:
    """Numeric description of one candidate tag."""
    ws = tag.split()
    cos_each = [cosine(np.asarray(vec), np.asarray(g)) for g in gt_vecs]
    return {
        "n_words": len(ws),
        "n_chars": len(tag),
        # what fraction of this tag's words are ground-truth vocabulary
        "gt_word_frac": sum(1 for w in ws if w in gt_words) / max(1, len(ws)),
        "is_both": tag in set(gt_tags),
        # "hubness": how close this tag sits to ALL gt tags on average.
        # The scoring target is the MEAN of the gt embeddings, so a tag that is
        # mildly close to every gt tag beats one that is very close to just one.
        "hub": statistics.mean(cos_each),
        "spread": statistics.pstdev(cos_each) if len(cos_each) > 1 else 0.0,
        "peak": max(cos_each),
    }


# --------------------------------------------------------------------------

async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-convo", type=int, default=7, help="windows per conversation")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--top", type=int, default=12, help="tags in a submitted answer")
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases, per_convo=args.per_convo, seed=0)
    print(f"{len(cases)} distinct conversations -> {len(pairs)} windows "
          f"({args.per_convo} per conversation)\n", flush=True)

    # ---- per-conversation work: ground truth + the oracle universe ----
    per_case = {}
    for ci, case in enumerate(cases):
        gt = await real_ground_truth(case)
        if not gt.ok():
            print(f"  conversation {ci}: no ground truth, skipped")
            continue
        gt_clean = normalize_all(Utils.get_clean_tag_set(gt.tags))
        uni = await build_universe(gt_clean)
        everything = sorted({t for v in uni.values() for t in v} | set(gt_clean))
        vecs = await llm.embed(everything, use_cache=True)
        gt_vecs = [vecs[t] for t in gt_clean if t in vecs]
        per_case[case.guid] = {
            "case": case, "gt": gt, "gt_clean": gt_clean,
            "uni": uni, "vecs": vecs, "gt_vecs": gt_vecs,
            "target": np.asarray(gt.target),
        }
        print(f"  conversation {ci} ({case.guid[:8]}): {len(gt_clean)} gt tags, "
              f"{len(everything)} candidates in universe", flush=True)
    print(flush=True)

    # ---- family ranking, per conversation ----
    print("=" * 78)
    print("A. CANDIDATE FAMILIES vs the REAL centroid  (per conversation, n=4)")
    print("=" * 78)
    fam_rows = collections.defaultdict(list)
    fam_rows_u = collections.defaultdict(list)
    for ci, (guid, d) in enumerate(per_case.items()):
        target, vecs = d["target"], d["vecs"]
        gt_set = set(d["gt_clean"])
        for fam, tags in d["uni"].items():
            cs = [(t, cosine(target, np.asarray(vecs[t]))) for t in tags if t in vecs]
            if not cs:
                continue
            vals = sorted(c for _, c in cs)
            uvals = sorted(c for t, c in cs if t not in gt_set)
            fam_rows[fam].append({
                "conv": ci, "n": len(cs), "mean": statistics.mean(vals),
                "top3": statistics.mean(vals[-3:]), "best": vals[-1],
            })
            if len(uvals) >= 3:
                fam_rows_u[fam].append(statistics.mean(uvals[-3:]))

    print(f"{'family':16s} {'n cand':>7s} {'mean':>8s} {'top-3':>8s} {'best':>8s} "
          f"{'top-3 UNIQUE':>13s}   per-conversation top-3")
    print("-" * 110)
    order = sorted(fam_rows, key=lambda f: -statistics.mean(r["top3"] for r in fam_rows[f]))
    for fam in order:
        rs = fam_rows[fam]
        pc = " ".join(f"{r['top3']:.3f}" for r in rs)
        u = statistics.mean(fam_rows_u[fam]) if fam_rows_u.get(fam) else float("nan")
        print(f"{fam:16s} {statistics.mean(r['n'] for r in rs):7.0f} "
              f"{statistics.mean(r['mean'] for r in rs):8.4f} "
              f"{statistics.mean(r['top3'] for r in rs):8.4f} "
              f"{statistics.mean(r['best'] for r in rs):8.4f} "
              f"{u:13.4f}   {pc}")

    # ---- ALL families pooled: what do the actual winners look like? ----
    print()
    print("=" * 78)
    print("B. SHAPE OF A WINNING TAG   (top 30 by cosine vs the rest, per conversation)")
    print("=" * 78)
    feat_top, feat_rest = [], []
    fam_of_top = collections.Counter()
    fam_of_all = collections.Counter()
    examples = {}
    for ci, (guid, d) in enumerate(per_case.items()):
        target, vecs = d["target"], d["vecs"]
        owner = {}
        for fam, tags in d["uni"].items():
            for t in tags:
                owner.setdefault(t, fam)
        allt = [(t, cosine(target, np.asarray(v))) for t, v in vecs.items()]
        allt.sort(key=lambda x: -x[1])
        gt_words = {w for t in d["gt_clean"] for w in t.split()}
        for rank, (t, c) in enumerate(allt):
            f = features(t, d["gt_clean"], gt_words, vecs[t], d["gt_vecs"])
            f["cos"] = c
            f["fam"] = owner.get(t, "gt")
            (feat_top if rank < 30 else feat_rest).append(f)
            fam_of_all[f["fam"]] += 1
            if rank < 30:
                fam_of_top[f["fam"]] += 1
        examples[ci] = allt[:12]

    def col(rows, k):
        return statistics.mean(r[k] for r in rows) if rows else float("nan")

    print(f"{'group':12s} {'n':>6s} {'cos':>7s} {'words':>7s} {'chars':>7s} "
          f"{'gt-word frac':>13s} {'hub':>7s} {'spread':>7s} {'peak':>7s} {'%both':>7s}")
    for name, rows in (("top 30", feat_top), ("rest", feat_rest)):
        print(f"{name:12s} {len(rows):6d} {col(rows,'cos'):7.4f} {col(rows,'n_words'):7.2f} "
              f"{col(rows,'n_chars'):7.1f} {col(rows,'gt_word_frac'):13.2f} "
              f"{col(rows,'hub'):7.4f} {col(rows,'spread'):7.4f} {col(rows,'peak'):7.4f} "
              f"{100*col(rows,'is_both'):6.1f}%")

    allrows = feat_top + feat_rest
    print("\n  correlation of cosine with each feature (pooled, n=%d):" % len(allrows))
    for k in ("n_words", "n_chars", "gt_word_frac", "hub", "spread", "peak"):
        x = np.array([r[k] for r in allrows], dtype=float)
        y = np.array([r["cos"] for r in allrows], dtype=float)
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
        print(f"    {k:14s} {r:+.3f}")

    print("\n  cosine by word count (pooled):")
    byw = collections.defaultdict(list)
    for r in allrows:
        byw[min(r["n_words"], 5)].append(r["cos"])
    for w in sorted(byw):
        print(f"    {w} word{'s' if w != 1 else ' '}{'+' if w == 5 else ' '}: "
              f"n={len(byw[w]):5d}  mean cos {statistics.mean(byw[w]):.4f}")

    print("\n  cosine by ground-truth-vocabulary overlap (pooled):")
    byg = collections.defaultdict(list)
    for r in allrows:
        byg[round(r["gt_word_frac"], 1) if r["gt_word_frac"] in (0.0, 1.0) else 0.5].append(r["cos"])
    for g in sorted(byg):
        label = {0.0: "no gt words", 0.5: "some gt words", 1.0: "all gt words"}[g]
        print(f"    {label:15s} n={len(byg[g]):5d}  mean cos {statistics.mean(byg[g]):.4f}")

    # hub is near-tautological with cosine (the target IS the mean of the gt
    # vectors), so report the other features with hub partialled out - that is
    # the part of the shape that is not just restating the scoring rule.
    print("\n  partial correlation with cosine, controlling for hub:")
    hub = np.array([r["hub"] for r in allrows], dtype=float)
    y = np.array([r["cos"] for r in allrows], dtype=float)
    y_res = y - np.polyval(np.polyfit(hub, y, 1), hub)
    for k in ("n_words", "n_chars", "gt_word_frac", "spread", "peak"):
        x = np.array([r[k] for r in allrows], dtype=float)
        if x.std() == 0:
            continue
        x_res = x - np.polyval(np.polyfit(hub, x, 1), hub)
        print(f"    {k:14s} {float(np.corrcoef(x_res, y_res)[0, 1]):+.3f}")

    print("\n  where the top-30 tags came from:")
    for f, c in fam_of_top.most_common():
        print(f"    {f:16s} {c:4d} of top-30   ({c}/{fam_of_all[f]} = "
              f"{100*c/fam_of_all[f]:.1f}% of that family)")

    print("\n  the single highest-cosine tags found, per conversation:")
    for ci, best in examples.items():
        print(f"    conv {ci}: " + ", ".join(f"{t}({c:.3f})" for t, c in best[:8]))

    # ---- C. does it beat the miner, end to end, per window? ----
    print()
    print("=" * 78)
    print("C. LIKE-FOR-LIKE vs the CURRENT MINER, per window")
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

            # miner pool, ranked against the REAL centroid (best case for the miner)
            mp = [(t, cosine(target, np.asarray(v))) for t, v in res.vectors.items()
                  if t in set(res.candidates)]
            mp.sort(key=lambda x: -x[1])
            mp_u = [c for t, c in mp if t not in gt_set]

            # oracle family: unique-only, best `top` tags from the whole universe
            allt = [(t, cosine(target, np.asarray(v))) for t, v in d["vecs"].items()]
            allt.sort(key=lambda x: -x[1])
            oracle_u = [(t, c) for t, c in allt if t not in gt_set]
            oracle_tags = [t for t, _ in oracle_u[: args.top]]

            actual = await score_answer(gt, res.tags, case.kind, model=GT_MODEL)
            oracle = await score_answer(gt, oracle_tags, case.kind, model=GT_MODEL)
            # miner pool re-ranked on the real centroid, unique-only, same size
            mp_tags = [t for t, _ in mp if t not in gt_set][: args.top]
            minerbest = await score_answer(gt, mp_tags, case.kind, model=GT_MODEL)

            # ---- E. the one part of the oracle shape that is DEPLOYABLE ----
            # `recombination` and `template` need no ground truth and no LLM
            # call: they are mechanical functions of the replica's PREDICTED
            # ground truth, which the miner already holds. Paired A/B, identical
            # selection rule, ranked by the miner's own ESTIMATED centroid.
            pred = normalize_all(Utils.get_clean_tag_set(res.predicted_gt or []))
            depl = {}
            if pred:
                pw = [w for t in pred for w in t.split() if w not in STOP and len(w) > 2]
                pw = list(dict.fromkeys(pw))
                extra = [f"{a} {b}" for a, b in itertools.permutations(pw, 2)][:400]
                extra += [f"{t} {s}" for t in pred for s in TEMPLATE_SUFFIX]
                extra = [t for t in normalize_all(extra) if t not in set(res.candidates)]
                ev = await llm.embed(extra, use_cache=True)
                allv = dict(res.vectors)
                allv.update(ev)
                est = np.mean([np.asarray(allv[t]) for t in pred if t in allv], axis=0) \
                    if any(t in allv for t in pred) else None
                if est is not None:
                    pset = set(pred)

                    def pick(pool):
                        rk = sorted(((t, cosine(est, np.asarray(allv[t])))
                                     for t in pool if t in allv), key=lambda x: -x[1])
                        return [t for t, _ in rk if t not in pset][: args.top]

                    a_tags = pick(res.candidates)
                    b_tags = pick(list(res.candidates) + extra)
                    va = await score_answer(gt, a_tags, case.kind, model=GT_MODEL)
                    vb = await score_answer(gt, b_tags, case.kind, model=GT_MODEL)
                    depl = {"depl_a": va.detail.get("adjusted", 0.0),
                            "depl_b": vb.detail.get("adjusted", 0.0),
                            "depl_a_sur": va.n_survived, "depl_b_sur": vb.n_survived,
                            "depl_b_unique": vb.n_unique, "depl_b_pen": vb.penalties,
                            "depl_new": sum(1 for t in b_tags if t in set(extra)),
                            "depl_b_tags": b_tags}
            return {
                **depl,
                "conv": case.guid,
                "miner_adjusted": actual.detail.get("adjusted", 0.0),
                "miner_pool_top3u": statistics.mean(sorted(mp_u)[-3:]) if len(mp_u) >= 3 else 0.0,
                "miner_oraclesel_adjusted": minerbest.detail.get("adjusted", 0.0),
                "oracle_top3u": statistics.mean([c for _, c in oracle_u[:3]]),
                "oracle_adjusted": oracle.detail.get("adjusted", 0.0),
                "oracle_survived": oracle.n_survived,
                "oracle_submitted": len(oracle_tags),
                "oracle_unique": oracle.n_unique,
                "oracle_tags": oracle_tags,
            }

    out = [r for r in await asyncio.gather(*[one(c, w) for c, w in pairs]) if r]
    if not out:
        raise SystemExit("no usable windows")

    def m(rows, k):
        return statistics.mean(r[k] for r in rows)

    print(f"n = {len(out)} windows\n")
    print(f"{'metric':38s} {'pooled':>9s}   per-conversation")
    print("-" * 90)
    byc = collections.defaultdict(list)
    for r in out:
        byc[r["conv"]].append(r)
    for k, label in (("miner_pool_top3u", "top-3 UNIQUE cos, miner pool"),
                     ("oracle_top3u", "top-3 UNIQUE cos, ORACLE family"),
                     ("miner_adjusted", "adjusted, miner as shipped"),
                     ("miner_oraclesel_adjusted", "adjusted, miner pool + real centroid"),
                     ("oracle_adjusted", "adjusted, ORACLE family")):
        pc = "  ".join(f"{m(v,k):.4f}" for v in byc.values())
        print(f"{label:38s} {m(out,k):9.4f}   {pc}")

    print()
    d_top3 = m(out, "oracle_top3u") - m(out, "miner_pool_top3u")
    d_adj = m(out, "oracle_adjusted") - m(out, "miner_adjusted")
    print(f"  delta top-3 unique cosine (oracle - miner pool) : {d_top3:+.4f}")
    print(f"  delta adjusted            (oracle - miner)      : {d_adj:+.4f}")
    wins = sum(1 for r in out if r["oracle_adjusted"] > r["miner_adjusted"])
    print(f"  W/L on adjusted: {wins}/{len(out) - wins}")
    print("  per-conversation delta on adjusted:")
    for i, (guid, v) in enumerate(byc.items()):
        print(f"    conv {i} ({guid[:8]}, n={len(v)}): "
              f"{m(v,'oracle_adjusted') - m(v,'miner_adjusted'):+.4f}")

    # ---- D. survival check ----
    print()
    print("=" * 78)
    print("D. WOULD THE ORACLE FAMILY SURVIVE CONTACT WITH REALITY?")
    print("=" * 78)
    sub = sum(r["oracle_submitted"] for r in out)
    sur = sum(r["oracle_survived"] for r in out)
    unq = sum(r["oracle_unique"] for r in out)
    print(f"  submitted {sub} tags across {len(out)} windows")
    print(f"  survived the validator's English screen + normalize: {sur} ({100*sur/sub:.1f}%)")
    print(f"  counted as UNIQUE (not string-matching ground truth): {unq}")
    allo = [t for r in out for t in r["oracle_tags"]]
    bad = [t for t in set(allo) if normalize(t) != t]
    print(f"  tags failing sn33.tags.normalize: {len(bad)}  {bad[:8]}")
    print(f"\n  example oracle answers:")
    for r in out[:3]:
        print(f"    {', '.join(r['oracle_tags'][:12])}")

    # ---- E ----
    dep = [r for r in out if "depl_a" in r]
    if dep:
        print()
        print("=" * 78)
        print("E. DEPLOYABLE SLICE: mechanical recombination of the PREDICTED gt")
        print("   (no ground truth read, no extra LLM call; ranked by our own centroid)")
        print("=" * 78)
        a = statistics.mean(r["depl_a"] for r in dep)
        b = statistics.mean(r["depl_b"] for r in dep)
        diffs = [r["depl_b"] - r["depl_a"] for r in dep]
        print(f"  n = {len(dep)} windows")
        print(f"  adjusted, miner pool only                    {a:.4f}")
        print(f"  adjusted, + mechanical recombination         {b:.4f}")
        print(f"  delta                                        {b - a:+.4f}"
              f"   W/L {sum(1 for d in diffs if d > 0)}/{sum(1 for d in diffs if d < 0)}")
        print(f"  new tags actually chosen per window: "
              f"{statistics.mean(r['depl_new'] for r in dep):.1f} of {args.top}")
        byc2 = collections.defaultdict(list)
        for r in dep:
            byc2[r["conv"]].append(r)
        print("  per-conversation delta:")
        for i, (guid, v) in enumerate(byc2.items()):
            print(f"    conv {i} ({guid[:8]}, n={len(v)}): "
                  f"{statistics.mean(x['depl_b'] - x['depl_a'] for x in v):+.4f}")
        sb = sum(r["depl_b_sur"] for r in dep)
        sa = sum(r["depl_a_sur"] for r in dep)
        print(f"  survived the English screen: arm A {sa}/{args.top*len(dep)} "
              f"({100*sa/(args.top*len(dep)):.1f}%), arm B {sb}/{args.top*len(dep)} "
              f"({100*sb/(args.top*len(dep)):.1f}%)")
        pens = collections.Counter(p for r in dep for p in r["depl_b_pen"])
        print(f"  arm B penalties fired: {dict(pens) or 'none'}   "
              f"mean unique {statistics.mean(r['depl_b_unique'] for r in dep):.1f}")
        sd = [r["depl_b"] - r["miner_adjusted"] for r in dep]
        print(f"  arm B vs the SHIPPED miner (paired, different selection rule): "
              f"{statistics.mean(sd):+.4f}  W/L {sum(1 for d in sd if d>0)}/"
              f"{sum(1 for d in sd if d<0)}")
        print("  example answer with recombination on:")
        print(f"    {', '.join(dep[0]['depl_b_tags'][:12])}")

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump({"windows": [{k: v for k, v in r.items()} for r in out]}, f, indent=1)
        print(f"\n  dumped -> {args.dump}")


if __name__ == "__main__":
    asyncio.run(main())
