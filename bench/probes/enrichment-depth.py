#!/usr/bin/env python3
"""PROBE: enrichment-depth.

HYPOTHESIS
  98% of our shipped tags derive from the enrichment lines, and enrichment is
  worth +0.2573 adjusted. The validator runs its enrichment prompt ONCE per
  enrichment line and asks for "at most 10" tags. We do exactly the same. So
  maybe we are under-extracting the one input that is actually carrying us:
  ask for more tags per line, take several independent draws per line, or
  restructure the per-line prompt to widen it.

WHAT IS MEASURED
  For a unit (= one conversation + one enrichment draw + one window), the REAL
  ground truth is generated with the vendored upstream prompts, giving the real
  centroid gt.target. Every candidate family is embedded and scored as cosine
  to that centroid.

  Two metrics per family:
    top3       mean of the 3 highest cosines in the family  (comparable with
               the already-published family table: GROUND_TRUTH 0.6642,
               gt_variants 0.6628, LLM families 0.3862-0.4735)
    top3_uniq  same, but computed ONLY over tags that do NOT string-match a
               real ground-truth tag, zero-padded below 3 - i.e. exactly the
               quantity the 0.55 term of the score uses.

  top3_uniq is the decision metric. top3 flatters any family made of literal
  ground-truth strings: those tags are `both`, never `unique`, and contribute
  zero to the 55% term.

  Also reported: scoring.oracle_best over the family treated as a candidate
  pool, i.e. the best `adjusted`/`final` the validator would give a miner who
  could pick perfectly out of that family. That is the like-for-like ceiling.

STATISTICAL DESIGN - read this before believing any number here
  The enrichment lines are IDENTICAL across every window of a conversation, so
  an enrichment-derived family is window-invariant: running it on 40 windows
  produces 4 distinct answers, not 40. Varying the window therefore adds no
  information about this hypothesis at all.

  What does vary in production is the validator's random enrichment SELECTION
  (`random.randint(1, min(3, len(results)))` then `random.sample`). So the unit
  here is a conversation x enrichment-seed pair: a different, genuinely
  independent realization of the same task, with its own regenerated ground
  truth. 4 conversations x 6 seeds = 24 units, each carrying a distinct window
  as well.

  Even so there are only FOUR distinct conversations underneath. Everything is
  reported per-conversation, and effects below ~0.03 are below the resolution
  of this corpus.

USAGE
  python bench/probes/enrichment-depth.py --seeds 6 --concurrency 6
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import (
    GT_MODEL,
    expand_windows,
    load_cases,
    pick_window,
    real_ground_truth,
)
from bench.harness import score_answer
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline, prompts, scoring
from sn33.replica import clean_like_validator
from sn33.tags import (
    centroid,
    cosine,
    drop_near_duplicates,
    normalize_all,
    parse_tag_list,
    rank_by_centroid,
)
from sn33.variants import expand as expand_variants

# Own cache namespace, so no other probe collides with these entries.
SALT = "edepth"


# --------------------------------------------------------------------------
# The depth variants. Every one of them reads the SAME enrichment lines.
# --------------------------------------------------------------------------

def upstream_enrichment(line: str) -> str:
    """Byte-identical to what the validator runs per enrichment line."""
    return prompts.gt_enrichment(line)


def deep_enrichment(line: str, n: int = 30) -> str:
    """ONE variable changed against upstream: the tag budget per line.

    Upstream instruction 5 is 'Limit to most important: Return at most 10 of
    the most important and relevant tags.' Everything else is left alone.
    """
    base = upstream_enrichment(line)
    old = "5.  **Limit to most important:** Return at most 10 of the most important and relevant tags."
    new = (
        f"5.  **Be exhaustive:** Return at most {n} tags. Start with the most "
        "important and relevant, then continue with broader themes, alternative "
        "phrasings of the same concepts, adjacent subtopics, and the field or "
        "industry the content belongs to."
    )
    assert old in base, "upstream enrichment template changed - fix deep_enrichment"
    return base.replace(old, new)


EXPAND_TPL = """\
Below is one search-result enrichment snippet (a title and a snippet) that was \
used to describe a longer document.

Produce 30 candidate topic tags for it. Breadth matters more than precision.
Cover, in this order:
1. The core subjects the snippet is about.
2. The broader field, industry or domain each subject belongs to.
3. Alternative phrasings a different writer would use for those same subjects.
4. Named entities that carry the topic (people, organizations, products, places).
5. The concepts a librarian would file the underlying document under.

Rules for every tag: lowercase, 1-4 words, plain English dictionary words, no \
punctuation, no hyphens, no abbreviations, no duplicates.
Return ONLY a comma-delimited list.

Enrichment content:
{content}
"""

JOINT_TPL = """\
Below are ALL of the search-result enrichment snippets that were gathered for \
one document. Read them together, not one at a time.

Produce 40 candidate topic tags describing the subject matter these snippets \
collectively point at. Include the themes that recur across several snippets \
first, then the broader field, then alternative phrasings, then named entities.

Rules for every tag: lowercase, 1-4 words, plain English dictionary words, no \
punctuation, no hyphens, no abbreviations, no duplicates.
Return ONLY a comma-delimited list.

Enrichment snippets:
{content}
"""


async def _tags(prompt: str, salt: str, temperature) -> list:
    raw = await llm.chat(
        prompt, model=GT_MODEL, timeout=180, temperature=temperature,
        use_cache=True, salt=f"{SALT}:{salt}",
    )
    # clean_like_validator is what the validator does to its own output; the
    # miner would then have to normalize. Do both, so every family is judged on
    # tags that could actually be submitted.
    return normalize_all(clean_like_validator(raw))


async def build_families(case) -> dict:
    """Every enrichment-depth family for one case, from its enrichment lines."""
    lines = list(case.enrichment_lines)
    if not lines:
        return {}

    # --- per-line jobs, all concurrent ---
    jobs = {
        # current depth, exactly what sn33/replica.py does today (temp 0.0)
        "x1_t0":     [_tags(upstream_enrichment(l), "x1t0", 0.0) for l in lines],
        # same prompt, one draw at the validator's own sampling temperature
        "x1_t1":     [_tags(upstream_enrichment(l), "d0", None) for l in lines],
        # two more independent draws of the same prompt at the same temperature
        "_d1":       [_tags(upstream_enrichment(l), "d1", None) for l in lines],
        "_d2":       [_tags(upstream_enrichment(l), "d2", None) for l in lines],
        # more tags per line
        "deep30":    [_tags(deep_enrichment(l), "deep30", 0.0) for l in lines],
        # restructured, deliberately widened per-line prompt
        "expand":    [_tags(EXPAND_TPL.format(content=l), "expand", 0.0) for l in lines],
    }
    flat = [c for v in jobs.values() for c in v]
    done = await asyncio.gather(*flat, return_exceptions=True)
    out, i = {}, 0
    for k, v in jobs.items():
        out[k] = [([] if isinstance(r, BaseException) else r) for r in done[i:i + len(v)]]
        i += len(v)

    joint = await _tags(JOINT_TPL.format(content="\n\n---\n\n".join(lines)), "joint", 0.0)

    def union(per_line):
        seen, acc = set(), []
        for tags in per_line:
            for t in tags:
                if t not in seen:
                    seen.add(t)
                    acc.append(t)
        return acc

    draws = [out["x1_t1"], out["_d1"], out["_d2"]]
    pooled, consensus = [], []
    seen_p, seen_c = set(), set()
    for li in range(len(lines)):
        counts = collections.Counter()
        for d in draws:
            for t in dict.fromkeys(d[li]):
                counts[t] += 1
        for t, c in counts.items():
            if t not in seen_p:
                seen_p.add(t)
                pooled.append(t)
            if c >= 2 and t not in seen_c:
                seen_c.add(t)
                consensus.append(t)

    return {
        "enrich_x1_t0":       union(out["x1_t0"]),
        "enrich_x1_t1":       union(out["x1_t1"]),
        "enrich_draws3":      pooled,
        "enrich_draws3_cons": consensus,
        "enrich_deep30":      union(out["deep30"]),
        "enrich_expand":      union(out["expand"]),
        "enrich_joint":       joint,
    }


# --------------------------------------------------------------------------

def _top3(vals) -> float:
    v = sorted(vals)[-3:]
    while len(v) < 3:
        v.append(0.0)              # the validator's zero padding
    return statistics.mean(v)


# A family that simply emits MORE tags scores a higher top-3 for free: top-3 is
# an extreme-order statistic, so 102 draws from a distribution beat 35 draws
# from the SAME distribution. Any depth effect has to survive being cut back to
# a common size, otherwise it is an artifact of tag count and not of quality.
CTRL_K = 35        # ~ the tag count the current per-line depth produces
CTRL_REPS = 200


def family_stats(tags, vecs, target, gt_raw) -> dict:
    import random as _r

    cs, is_uniq = [], []
    for t in tags:
        v = vecs.get(t)
        if v is None or not len(v):
            continue
        cs.append(cosine(target, np.asarray(v, dtype=np.float64)))
        is_uniq.append(t not in gt_raw)
    if not cs:
        return {}
    cs_uniq = [s for s, u in zip(cs, is_uniq) if u]

    # size-controlled top3_uniq: cut every family back to CTRL_K tags chosen at
    # random, keeping the unique/both split of whatever was drawn, and average.
    if len(cs) <= CTRL_K:
        ctrl = _top3(cs_uniq)
    else:
        rng = _r.Random(12345)
        idx = range(len(cs))
        acc = []
        for _ in range(CTRL_REPS):
            pick = rng.sample(idx, CTRL_K)
            acc.append(_top3([cs[i] for i in pick if is_uniq[i]]))
        ctrl = statistics.mean(acc)

    return {
        "n": len(cs),
        "n_uniq": len(cs_uniq),
        "mean": statistics.mean(cs),
        "top3": _top3(cs),
        "top3_uniq": _top3(cs_uniq),
        "top3_uniq_ctrl": ctrl,
        "best": max(cs),
    }


def compose_from(pool, mined, vecs, cfg, profile, target_tags, insurance):
    """Run the miner's real selection path over an arbitrary candidate pool.

    Selection uses the miner's OWN estimated centroid (built from its predicted
    ground truth), never the real one - otherwise the arm gets information the
    miner does not have. Both arms go through this identical function, so the
    only difference between them is what is in ``pool``.
    """
    predicted_gt = list(mined.predicted_gt)
    gt_for_target = Utils.get_clean_tag_set(predicted_gt) if predicted_gt else []
    est = centroid([vecs[t] for t in gt_for_target if t in vecs]) if gt_for_target else None
    if est is None:
        return []
    ranked = rank_by_centroid(normalize_all(pool), vecs, est)
    protected = set(normalize_all(predicted_gt))
    ranked = drop_near_duplicates(ranked, vecs, cfg.dedup_threshold, protected=protected)
    if not ranked:
        return []
    return pipeline.compose(ranked, predicted_gt, profile, target_tags, insurance)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6, help="enrichment draws per conversation")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    seeds = [i * 1000 for i in range(args.seeds)]
    units = []
    for si, s in enumerate(seeds):
        for ci, case in enumerate(load_cases(kind="conversation_tagging", seed=s)):
            units.append((s, case, pick_window(case, seed=s + ci * 7 + si)))

    sem = asyncio.Semaphore(args.concurrency)
    cfg = pipeline.Config(use_cache=True, use_local=False, deadline_s=600, call_timeout_s=180)

    print(f"{len(units)} units = {len(seeds)} enrichment seeds x "
          f"{len(units)//max(1,len(seeds))} conversations\n", flush=True)

    async def one(seed, case, window):
        async with sem:
            gt = await real_ground_truth(case)
            if not gt.ok() or not case.enrichment_lines:
                return None
            fams = await build_families(case)
            if not fams:
                return None

            # reference families
            gt_norm = normalize_all(Utils.get_clean_tag_set(gt.tags))
            fams["REF_ground_truth"] = gt_norm
            fams["REF_gt_variants"] = expand_variants(gt_norm, per_tag=2)

            mined = await pipeline.mine(case.kind, window=window,
                                        enrichment=case.enrichment_lines, cfg=cfg)
            fams["REF_miner_pool"] = list(mined.candidates)
            fams["REF_miner_shipped"] = list(mined.tags)
            # does the deepest family ADD anything on top of the pool we already have?
            fams["AUG_pool_plus_deep30"] = list(
                dict.fromkeys(list(mined.candidates) + fams["enrich_deep30"]))

            # THE CONTROL THAT DECIDES THE HYPOTHESIS.
            # deep30 emits ~107 tags where the current depth emits ~37. If the
            # gain is really about the ENRICHMENT lines, then the same number of
            # extra candidates drawn from the WINDOW instead must gain less. If
            # it gains just as much, the finding is not "enrichment depth" at
            # all - it is "wider candidate pool", and the enrichment framing is
            # wrong.
            wtext = "\n".join(str(l[1]) for l in window if len(l) >= 2)
            wide_window = parse_tag_list(await llm.chat(
                prompts.MINER_CONVERSATION_POOL.format(n=120, content=wtext[:6000]),
                model=GT_MODEL, timeout=180, use_cache=True, salt=f"{SALT}:wide120"))
            fams["CTRL_window_wide120"] = wide_window
            fams["AUG_pool_plus_window120"] = list(
                dict.fromkeys(list(mined.candidates) + wide_window))
            fams["AUG_pool_plus_both"] = list(
                dict.fromkeys(list(mined.candidates) + fams["enrich_deep30"] + wide_window))

            everything = sorted({t for v in fams.values() for t in v})
            vecs = await llm.embed(everything, use_cache=True)
            vecs.update(mined.vectors or {})
            target = np.asarray(gt.target, dtype=np.float64)
            gt_raw = set(gt.tags)          # scoring.py diffs against the RAW gt strings

            stats, oracle = {}, {}
            for fam, tags in fams.items():
                st = family_stats(tags, vecs, target, gt_raw)
                if st:
                    stats[fam] = st
                    ob = scoring.oracle_best(list(gt.tags), target, tags, vecs,
                                             penalties=True, min_tags=3)
                    oracle[fam] = {"adjusted": ob["adjusted"], "final": ob["final"]}

            # ---- end to end: does the extra depth reach the SCORE? ----
            profile = pipeline.TASK_PROFILE[case.kind]
            tt, ins = profile["target_tags"], profile["insurance"]
            e2e = {}
            for name, tags in (
                ("shipped", mined.tags),
                ("armA_pool", compose_from(mined.candidates, mined, vecs, cfg, profile, tt, ins)),
                ("armB_pool_plus_deep30",
                 compose_from(fams["AUG_pool_plus_deep30"], mined, vecs, cfg, profile, tt, ins)),
                ("armC_pool_plus_window120",
                 compose_from(fams["AUG_pool_plus_window120"], mined, vecs, cfg, profile, tt, ins)),
                ("armD_pool_plus_both",
                 compose_from(fams["AUG_pool_plus_both"], mined, vecs, cfg, profile, tt, ins)),
            ):
                v = await score_answer(gt, tags, case.kind, model=GT_MODEL)
                e2e[name] = {"adjusted": v.adjusted, "final": v.final,
                             "n_unique": v.n_unique, "n_both": v.n_both,
                             "penalties": v.penalties, "tags": list(tags)}

            return {"seed": seed, "guid": case.guid, "n_enrich": len(case.enrichment_lines),
                    "stats": stats, "oracle": oracle, "tags": fams, "e2e": e2e,
                    "gt_tags": list(gt.tags)}

    results = [r for r in await asyncio.gather(*[one(s, c, w) for s, c, w in units]) if r]
    if not results:
        raise SystemExit("no usable units")

    fam_names = sorted({f for r in results for f in r["stats"]})

    def agg(fam, key, rows=None):
        rows = rows if rows is not None else results
        v = [r["stats"][fam][key] for r in rows if fam in r["stats"]]
        return statistics.mean(v) if v else float("nan")

    def agg_o(fam, key, rows=None):
        rows = rows if rows is not None else results
        v = [r["oracle"][fam][key] for r in rows if fam in r["oracle"]]
        return statistics.mean(v) if v else float("nan")

    print(f"n = {len(results)} units\n")
    print(f"{'family':24s} {'tags':>5s} {'mean':>7s} {'top3':>7s} {'TOP3_UNIQ':>10s} "
          f"{'@k=35':>8s} {'best':>7s} {'orc.adj':>8s} {'orc.fin':>8s}")
    print("-" * 92)
    for fam in sorted(fam_names, key=lambda f: -agg(f, "top3_uniq")):
        print(f"{fam:24s} {agg(fam,'n'):5.0f} {agg(fam,'mean'):7.4f} {agg(fam,'top3'):7.4f} "
              f"{agg(fam,'top3_uniq'):10.4f} {agg(fam,'top3_uniq_ctrl'):8.4f} "
              f"{agg(fam,'best'):7.4f} "
              f"{agg_o(fam,'adjusted'):8.4f} {agg_o(fam,'final'):8.4f}")
    print("  @k=35 = top3_uniq after cutting the family back to 35 random tags,")
    print("          which is the size the current per-line depth produces.")

    # ---- paired deltas against the family that represents CURRENT depth ----
    base = "enrich_x1_t0"
    print(f"\npaired vs {base} (the depth we run today), per unit, n={len(results)}")
    print(f"{'family':22s} {'d top3_uniq':>12s} {'W/L':>8s} {'d orc.final':>12s} {'W/L':>8s}")
    print("-" * 66)
    for fam in fam_names:
        if fam == base:
            continue
        d = [r["stats"][fam]["top3_uniq"] - r["stats"][base]["top3_uniq"]
             for r in results if fam in r["stats"] and base in r["stats"]]
        do = [r["oracle"][fam]["final"] - r["oracle"][base]["final"]
              for r in results if fam in r["oracle"] and base in r["oracle"]]
        if not d:
            continue
        print(f"{fam:22s} {statistics.mean(d):+12.4f} "
              f"{sum(1 for x in d if x>1e-9):3d}/{sum(1 for x in d if x<-1e-9):<4d} "
              f"{statistics.mean(do):+12.4f} "
              f"{sum(1 for x in do if x>1e-9):3d}/{sum(1 for x in do if x<-1e-9):<4d}")

    # ---- per conversation ----
    by_guid = collections.defaultdict(list)
    for r in results:
        by_guid[r["guid"]].append(r)
    print("\nPER CONVERSATION (top3_uniq)  - the effect must hold in all 4 to be real")
    hdr = "  ".join(f"{g[:8]:>8s}" for g in sorted(by_guid))
    print(f"{'family':22s} {hdr}")
    print("-" * (22 + len(hdr)))
    for fam in sorted(fam_names, key=lambda f: -agg(f, "top3_uniq")):
        cells = "  ".join(f"{agg(fam,'top3_uniq',by_guid[g]):8.4f}" for g in sorted(by_guid))
        print(f"{fam:22s} {cells}")

    print(f"\nPER CONVERSATION, delta vs {base}")
    print(f"{'family':22s} {hdr}")
    print("-" * (22 + len(hdr)))
    for fam in fam_names:
        if fam == base:
            continue
        cells = []
        for g in sorted(by_guid):
            rows = by_guid[g]
            d = [r["stats"][fam]["top3_uniq"] - r["stats"][base]["top3_uniq"]
                 for r in rows if fam in r["stats"] and base in r["stats"]]
            cells.append(f"{statistics.mean(d):+8.4f}" if d else "     n/a")
        print(f"{fam:22s} {'  '.join(cells)}")

    # ---- reality check on the widest family ----
    print("\nREALITY CHECK")
    for fam in fam_names:
        rows = [r for r in results if fam in r["stats"]]
        if not rows:
            continue
        n_tags = statistics.mean(len(r["tags"][fam]) for r in rows)
        both = statistics.mean(
            sum(1 for t in r["tags"][fam] if t in set(r["gt_tags"])) / max(1, len(r["tags"][fam]))
            for r in rows)
        print(f"  {fam:22s} tags/unit {n_tags:5.1f}   string-matches ground truth {both:5.1%}")

    # ---- end to end ----
    def boot(diffs, iters=10000, seed=0):
        import random as _r
        if not diffs:
            return (float("nan"), float("nan"))
        rng, n, means = _r.Random(seed), len(diffs), []
        for _ in range(iters):
            means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
        means.sort()
        return means[int(0.025 * iters)], means[int(0.975 * iters)]

    print("\nEND TO END - real score_answer against the real ground truth")
    print(f"{'arm':26s} {'adjusted':>9s} {'final':>8s} {'uniq':>5s} {'both':>5s}")
    print("-" * 58)
    arms = ["shipped", "armA_pool", "armB_pool_plus_deep30",
            "armC_pool_plus_window120", "armD_pool_plus_both"]
    for a in arms:
        rows = [r["e2e"][a] for r in results if a in r.get("e2e", {})]
        if not rows:
            continue
        print(f"{a:26s} {statistics.mean(x['adjusted'] for x in rows):9.4f} "
              f"{statistics.mean(x['final'] for x in rows):8.4f} "
              f"{statistics.mean(x['n_unique'] for x in rows):5.1f} "
              f"{statistics.mean(x['n_both'] for x in rows):5.1f}")

    print("\n  paired vs armA_pool (identical code path, only the pool differs):")
    for a in arms[2:]:
        d = [r["e2e"][a]["adjusted"] - r["e2e"]["armA_pool"]["adjusted"] for r in results]
        df = [r["e2e"][a]["final"] - r["e2e"]["armA_pool"]["final"] for r in results]
        lo, hi = boot(d)
        print(f"    {a:26s} adjusted {statistics.mean(d):+.4f} ci95 [{lo:+.4f},{hi:+.4f}] "
              f"W/L {sum(1 for x in d if x>1e-9):2d}/{sum(1 for x in d if x<-1e-9):<2d}"
              f"   final {statistics.mean(df):+.4f}")
    dbc = [r["e2e"]["armB_pool_plus_deep30"]["adjusted"]
           - r["e2e"]["armC_pool_plus_window120"]["adjusted"] for r in results]
    lo, hi = boot(dbc)
    print(f"\n    armB - armC (enrichment depth vs an equally wide WINDOW pool): "
          f"{statistics.mean(dbc):+.4f} ci95 [{lo:+.4f},{hi:+.4f}] "
          f"W/L {sum(1 for x in dbc if x>1e-9)}/{sum(1 for x in dbc if x<-1e-9)}")
    print("    (the ci95 treats units as independent; they are not - 4 conversations)")

    print("\n  end to end, per conversation (adjusted):")
    print(f"    {'guid':10s} {'n':>2s} " + " ".join(f"{a.split('_')[0]:>8s}" for a in arms))
    for g in sorted(by_guid):
        rows = by_guid[g]
        cells = " ".join(f"{statistics.mean(r['e2e'][a]['adjusted'] for r in rows):8.4f}"
                         for a in arms)
        print(f"    {g[:10]:10s} {len(rows):2d} {cells}")
    print(f"\n    {'delta vs armA':13s} " + " ".join(f"{a.split('_')[0]:>8s}" for a in arms[2:]))
    for g in sorted(by_guid):
        rows = by_guid[g]
        base_g = statistics.mean(r["e2e"]["armA_pool"]["adjusted"] for r in rows)
        cells = " ".join(
            f"{statistics.mean(r['e2e'][a]['adjusted'] for r in rows) - base_g:+8.4f}"
            for a in arms[2:])
        print(f"    {g[:10]:13s} {cells}")

    print("\nEXAMPLE TAGS (unit 1)")
    r0 = results[0]
    for fam in sorted(fam_names):
        print(f"  {fam:22s} {', '.join(r0['tags'][fam][:14])}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump([{k: v for k, v in r.items()} for r in results], f, indent=1)
        print(f"\nwrote {args.out}")

    print("\nusage:", llm.Usage.snapshot())


if __name__ == "__main__":
    asyncio.run(main())
