#!/usr/bin/env python3
"""What does a maximally-central tag actually look like?

Reasoning that leads here:

  * `adjusted` is nothing but cosine-to-centroid of the tags we submit
  * our centroid estimate is 0.94 aligned with the real one
  * ranking against a PERFECT centroid gains only +0.0075

So estimation and selection are solved, and the ceiling is set by what is in the
candidate pool. To reach 0.693 adjusted the top-3 unique tags need cosine ~0.72,
yet ground-truth tags average only ~0.645 to their own centroid - so the winning
tags are ones that sit CLOSER to the centroid than the ground truth does.

Rather than guess what those look like, generate many candidate families for the
same window, rank them all against the REAL centroid, and read off which family
wins. That turns "what technique do top miners use" into a measurement.

    python bench/what_scores_high.py --n 8
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.faithful import GT_MODEL, convo_xml, expand_windows, load_cases, real_ground_truth
from conversationgenome.utils.Utils import Utils
from sn33 import llm, prompts
from sn33.tags import cosine, normalize_all, parse_tag_list
from sn33.variants import expand as expand_variants

# Each family is a hypothesis about what "close to the centroid" looks like.
FAMILIES = {
    "specific_topics": (
        "List {n} specific topic tags for this excerpt - the concrete subjects "
        "actually discussed. lowercase, 1-4 words, comma separated, no commentary."
    ),
    "broad_themes": (
        "List {n} BROAD theme tags describing the overall subject area of the "
        "conversation this excerpt came from. Think category labels a librarian "
        "would file it under. lowercase, 1-4 words, comma separated, no commentary."
    ),
    "whole_doc_summary": (
        "In {n} different ways, name the single overall topic of the conversation "
        "this excerpt came from - as if titling the whole episode. Each answer is "
        "one short phrase. lowercase, 1-4 words, comma separated, no commentary."
    ),
    "entities": (
        "List {n} named entities in this excerpt: people, organisations, products, "
        "places, technologies. lowercase, 1-4 words, comma separated, no commentary."
    ),
    "domain_field": (
        "Name {n} academic or professional FIELDS this excerpt belongs to, from "
        "narrow to broad. lowercase, 1-3 words, comma separated, no commentary."
    ),
}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="windows to test")
    ap.add_argument("--per-family", type=int, default=15)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases)[: args.n]
    sem = asyncio.Semaphore(args.concurrency)
    print(f"{len(pairs)} windows from {len(cases)} conversations\n")

    async def one(case, window):
        async with sem:
            gt = await real_ground_truth(case)
            if not gt.ok():
                return None
            text = "\n".join(str(l[1]) for l in window)

            # every family sees the same window
            out = {}
            for fam, tpl in FAMILIES.items():
                raw = await llm.chat(
                    tpl.format(n=args.per_family) + "\n\nExcerpt:\n" + text[:6000],
                    model=GT_MODEL, timeout=180, use_cache=True, salt="fam",
                )
                out[fam] = parse_tag_list(raw)

            # the ground-truth tags themselves, as a reference family
            out["GROUND_TRUTH"] = normalize_all(Utils.get_clean_tag_set(gt.tags))
            # and lexical variants of them
            out["gt_variants"] = expand_variants(out["GROUND_TRUTH"], per_tag=2)

            everything = sorted({t for v in out.values() for t in v})
            vecs = await llm.embed(everything, use_cache=True)
            target = np.asarray(gt.target)

            scored = {}
            for fam, tags in out.items():
                cs = [cosine(target, np.asarray(vecs[t])) for t in tags if t in vecs]
                if cs:
                    scored[fam] = {
                        "mean": statistics.mean(cs),
                        "top3": statistics.mean(sorted(cs)[-3:]),
                        "best": max(cs),
                        "n": len(cs),
                    }
            best_overall = sorted(
                ((cosine(target, np.asarray(v)), t) for t, v in vecs.items()),
                reverse=True,
            )[:8]
            return scored, best_overall

    results = [r for r in await asyncio.gather(*[one(c, w) for c, w in pairs]) if r]
    if not results:
        raise SystemExit("no usable cases")

    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for scored, _ in results:
        for fam, s in scored.items():
            for k, v in s.items():
                agg[fam][k].append(v)

    print(f"{'candidate family':22s} {'mean cos':>9s} {'top-3':>8s} {'best':>8s} {'tags':>5s}")
    print("-" * 58)
    for fam in sorted(agg, key=lambda f: -statistics.mean(agg[f]["top3"])):
        a = agg[fam]
        print(f"{fam:22s} {statistics.mean(a['mean']):9.4f} {statistics.mean(a['top3']):8.4f} "
              f"{statistics.mean(a['best']):8.4f} {statistics.mean(a['n']):5.0f}")

    print("\nHighest-cosine tags found, per window (this is the shape to aim for):")
    for i, (_, best) in enumerate(results[:5]):
        print(f"  window {i+1}: " + ", ".join(f"{t}({c:.3f})" for c, t in best[:6]))


if __name__ == "__main__":
    asyncio.run(main())
