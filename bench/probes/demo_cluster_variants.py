#!/usr/bin/env python3
"""Side-by-side: naive window bootstrap vs cluster bootstrap, on the variants A/B.

Consumes the ``--out`` JSON that ``bench/run_faithful.py`` writes and re-derives
which conversation each window came from. The mapping is exact, not guessed:
``run_faithful`` builds its case list with

    pairs = expand_windows(load_cases(kind=..., seed=seed), seed=seed)[:n]

and then keeps ``[(c, g, w) for ... if g.ok()]`` in that same order, so replaying
the (deterministic, no-LLM) corpus load reproduces the ordering. The script
asserts the lengths line up before it reports anything.

    python bench/probes/demo_cluster_variants.py --results <run_faithful --out json>

Corpus warning found while building this
----------------------------------------
``expand_windows`` emits all of conversation 1's windows, then all of
conversation 2's, and so on - so ``[:n]`` slices mid-conversation rather than
sampling across them. The per-conversation window counts are 36 / 21 / 16 / 2,
which means:

    --n 40  ->  36 windows of ONE podcast + 2 + 2, and the fourth conversation
                never appears at all
    --n 75  ->  all four

Any run below --n 75 is therefore close to a single-conversation experiment
wearing a 40-observation costume. Use the full 75, or shuffle the pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import expand_windows, load_cases
from bench.harness import paired_delta
from bench.probes.cluster_stats import compare_methods, format_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="JSON written by run_faithful.py --out")
    ap.add_argument("--kind", default="conversation_tagging")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metric", default="final", choices=["final", "adjusted"])
    args = ap.parse_args()

    with open(args.results) as f:
        results = json.load(f)
    specs = list(results.keys())
    if len(specs) < 2:
        raise SystemExit("need at least two strategies in the results file")
    base, variant = specs[0], specs[1]
    n = len(results[base])

    sources = load_cases(kind=args.kind, seed=args.seed)
    pairs = expand_windows(sources, seed=args.seed)[:n]
    if len(pairs) != n:
        raise SystemExit(f"corpus replay gave {len(pairs)} windows, results hold {n}")
    # Label each conversation by its first window's opening line - a guid is
    # opaque, and the same conversation is reissued under many guids.
    labels = {}
    for c in sources:
        first = (c.full_lines[0][1] if c.full_lines else c.document)[:40].replace("\n", " ")
        labels[c.guid] = f"{first}..."
    cluster_ids = [c.guid for c, _ in pairs]

    print(f"corpus replay: {len(sources)} distinct conversations -> {n} windows")
    for g in dict.fromkeys(cluster_ids):
        print(f"  {g[:12]}  {cluster_ids.count(g):3d} windows   {labels[g]}")
    print()

    a = results[base]
    b = results[variant]

    # Sanity: harness.paired_delta on the raw Verdict dicts must reproduce the
    # number run_faithful printed, so the two arms are aligned as loaded.
    print(f"[check] harness.paired_delta reproduces run_faithful: ", end="")
    import types
    verdicts_a = [types.SimpleNamespace(**d) for d in a]
    verdicts_b = [types.SimpleNamespace(**d) for d in b]
    hd = paired_delta(verdicts_a, verdicts_b)
    print(f"delta {hd['delta']:+.4f} CI[{hd['ci95'][0]:+.4f},{hd['ci95'][1]:+.4f}] "
          f"W/L {hd['wins']}/{hd['losses']}\n")

    cmp = compare_methods(a, b, cluster_ids, metric=args.metric)
    # Swap opaque guids for readable labels in the printout.
    for pc in cmp["clustered"]["per_cluster"]:
        pc["cluster"] = labels.get(pc["cluster"], pc["cluster"])
    print(format_report(cmp, label=f"{variant}  vs  {base}"))

    print("\n--- what effect size could this corpus resolve at all? ---")
    cl = cmp["clustered"]
    k = cl["n_clusters"]
    print(f"  between-conversation sd of the per-conversation delta: {cl['sd_cluster_means']:.4f}")
    print(f"  t({k-1})={cl['t_crit']:.3f} * sd / sqrt({k}) = minimum resolvable effect: "
          f"{cl['min_resolvable_effect']:.4f}")
    print(f"  (an effect below that cannot be separated from which {k} podcasts we happened to draw)")


if __name__ == "__main__":
    main()
