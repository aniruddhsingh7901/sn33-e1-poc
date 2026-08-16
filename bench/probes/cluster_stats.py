"""Cluster-aware confidence intervals for the faithful benchmark.

Why this module exists
----------------------
``bench.harness.paired_delta`` bootstraps the per-case difference by resampling
individual WINDOWS with replacement::

    diffs = [y.final - x.final for x, y in zip(a, b)]
    ...
    for _ in range(iters):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)

That is a valid bootstrap only if the 40 windows are 40 independent draws. They
are not. The faithful corpus holds **four distinct conversations**; the 75
windows are overlapping 10-line slices of those four. Windows of the same
conversation share most of their ground truth, so their scores - and their
paired differences - are strongly correlated. Resampling windows pretends to
have 40 independent observations when the design supplies roughly 4, and the
resulting interval is too narrow by the square root of the design effect.

The fix is the standard nonparametric **cluster bootstrap**: resample whole
conversations with replacement, keeping every window of a drawn conversation
together (and dropping every window of a conversation that was not drawn). The
uncertainty then reflects "what if we had sampled four *other* podcasts", which
is the question a reader of the benchmark is actually asking.

Two intervals are reported, because with K = 4 clusters neither is sufficient
alone:

``ci95_cluster_bootstrap``
    Percentile interval from the cluster bootstrap. Honest about the clustering,
    but with only 4 clusters there are just 35 distinct resamples, so its tail
    quantiles are extremely coarse and it is biased *narrow* at small K.

``ci95_cluster_t``
    Student-t interval on the K per-cluster mean differences with K-1 degrees of
    freedom (t = 3.182 at K = 4). This is the usual small-K correction and it is
    the more conservative of the two here.

``significant`` requires **both** intervals to exclude zero. At K = 4 that is
the defensible call: either interval on its own will occasionally certify an
effect that lives in a single conversation.

The output always carries ``per_cluster`` so a reader can see whether an effect
holds in all four conversations or in one. A pooled delta that is driven by one
conversation is not a result, whatever the interval says.

Nothing here edits ``bench/harness.py``; import both and print them together.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Dict, Hashable, List, Sequence

# Two-sided 95% Student-t critical values, indexed by degrees of freedom.
# K clusters -> K-1 df. Beyond 30 the normal approximation is close enough.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_crit_95(df: int) -> float:
    if df < 1:
        return float("inf")
    return _T95.get(df, 1.96)


def metric_of(v: Any, metric: str = "final") -> float:
    """Pull one number out of a Verdict, a dict, or a bare float.

    ``final`` and ``adjusted`` are both live questions on this project - 96% of
    the gap to the top miners is ``adjusted`` - so the caller picks.
    """
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        return float(v[metric])
    if hasattr(v, metric):
        return float(getattr(v, metric))
    detail = getattr(v, "detail", None)
    if isinstance(detail, dict) and metric in detail:
        return float(detail[metric])
    raise TypeError(f"cannot read metric {metric!r} from {type(v).__name__}")


def intraclass_correlation(diffs: Sequence[float], cluster_ids: Sequence[Hashable]) -> Dict[str, float]:
    """One-way random-effects ICC of the paired differences, plus design effect.

    ICC is the share of the variance in the per-window differences that is
    between conversations rather than within one. It converts directly into how
    much the naive interval lies:

        design_effect = 1 + (m0 - 1) * ICC          (m0 = mean cluster size)
        n_eff         = n_windows / design_effect
        naive CI is too narrow by a factor of sqrt(design_effect)

    Returns zeros when there is no variance at all (identical strategies).
    """
    groups: Dict[Hashable, List[float]] = {}
    for d, c in zip(diffs, cluster_ids):
        groups.setdefault(c, []).append(d)

    k = len(groups)
    n_total = len(diffs)
    if k < 2 or n_total <= k:
        return {"icc": 0.0, "design_effect": 1.0, "n_eff": float(n_total), "m0": float(n_total) / max(k, 1)}

    grand = sum(diffs) / n_total
    means = {c: statistics.mean(v) for c, v in groups.items()}
    ms_between = sum(len(v) * (means[c] - grand) ** 2 for c, v in groups.items()) / (k - 1)
    ms_within = sum((x - means[c]) ** 2 for c, v in groups.items() for x in v) / (n_total - k)
    m0 = (n_total - sum(len(v) ** 2 for v in groups.values()) / n_total) / (k - 1)

    denom = ms_between + (m0 - 1) * ms_within
    icc = 0.0 if denom <= 0 else (ms_between - ms_within) / denom
    icc = max(0.0, min(1.0, icc))
    design_effect = max(1.0, 1.0 + (m0 - 1) * icc)
    return {
        "icc": icc,
        "design_effect": design_effect,
        "n_eff": n_total / design_effect,
        "m0": m0,
        "ms_between": ms_between,
        "ms_within": ms_within,
    }


def clustered_paired_delta(
    baseline: Sequence[Any],
    variant: Sequence[Any],
    cluster_ids: Sequence[Hashable],
    iters: int = 10000,
    seed: int = 0,
    metric: str = "final",
) -> Dict[str, Any]:
    """Paired delta (variant - baseline) with the clustering taken seriously.

    ``baseline`` and ``variant`` are aligned per-window results - Verdicts,
    dicts, or floats. ``cluster_ids`` is the conversation each window came from,
    same length and order.

    The bootstrap resamples CLUSTERS with replacement. Drawing a conversation
    brings all of its windows; not drawing it removes all of them. The pooled
    mean of the resampled windows is the bootstrap statistic, so the point
    estimate stays window-weighted and directly comparable to
    ``bench.harness.paired_delta``.

    Keys of interest:
        delta                    pooled mean difference, window-weighted
        delta_cluster_mean       unweighted mean of the K per-cluster means
        ci95_cluster_bootstrap   percentile CI over resampled clusters
        ci95_cluster_t           t CI on the K cluster means, K-1 df
        significant              BOTH intervals exclude zero
        clusters_positive/negative  how many of the K clusters moved each way
        per_cluster              n, mean diff, sd, wins/losses per conversation
        min_resolvable_effect    t * sd(cluster means) / sqrt(K) - the smallest
                                 delta this corpus could have called significant
                                 at the between-conversation spread observed here
    """
    if not (len(baseline) == len(variant) == len(cluster_ids)):
        raise ValueError(
            f"length mismatch: baseline={len(baseline)} variant={len(variant)} clusters={len(cluster_ids)}"
        )
    if not baseline:
        return {}

    diffs = [metric_of(y, metric) - metric_of(x, metric) for x, y in zip(baseline, variant)]

    # Group window indices by conversation, preserving first-seen order so the
    # report reads in corpus order rather than hash order.
    order: List[Hashable] = []
    groups: Dict[Hashable, List[int]] = {}
    for i, c in enumerate(cluster_ids):
        if c not in groups:
            groups[c] = []
            order.append(c)
        groups[c].append(i)

    k = len(order)
    n_total = len(diffs)
    pooled = statistics.mean(diffs)

    per_cluster = []
    for c in order:
        idx = groups[c]
        vals = [diffs[i] for i in idx]
        per_cluster.append({
            "cluster": c,
            "n": len(vals),
            "delta": statistics.mean(vals),
            "sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "wins": sum(1 for d in vals if d > 0),
            "losses": sum(1 for d in vals if d < 0),
            "baseline_mean": statistics.mean([metric_of(baseline[i], metric) for i in idx]),
            "variant_mean": statistics.mean([metric_of(variant[i], metric) for i in idx]),
        })

    cluster_means = [pc["delta"] for pc in per_cluster]
    delta_cluster_mean = statistics.mean(cluster_means)

    # --- cluster bootstrap -------------------------------------------------
    rng = random.Random(seed)
    cluster_vals = [[diffs[i] for i in groups[c]] for c in order]
    boot: List[float] = []
    for _ in range(iters):
        total = 0.0
        count = 0
        for _ in range(k):
            vals = cluster_vals[rng.randrange(k)]
            total += sum(vals)
            count += len(vals)
        boot.append(total / count)
    boot.sort()
    lo_b = boot[int(0.025 * iters)]
    hi_b = boot[min(int(0.975 * iters), iters - 1)]

    # --- t interval on the K cluster means ---------------------------------
    if k >= 2:
        sd_clusters = statistics.stdev(cluster_means)
        se = sd_clusters / math.sqrt(k)
        t = t_crit_95(k - 1)
        lo_t, hi_t = delta_cluster_mean - t * se, delta_cluster_mean + t * se
        mde = t * sd_clusters / math.sqrt(k)
    else:
        sd_clusters, se, t = 0.0, float("inf"), float("inf")
        lo_t, hi_t = float("-inf"), float("inf")
        mde = float("inf")

    sig_boot = lo_b > 0 or hi_b < 0
    sig_t = lo_t > 0 or hi_t < 0

    icc = intraclass_correlation(diffs, cluster_ids)

    return {
        "metric": metric,
        "n_windows": n_total,
        "n_clusters": k,
        "delta": pooled,
        "delta_cluster_mean": delta_cluster_mean,
        "ci95_cluster_bootstrap": (lo_b, hi_b),
        "ci95_cluster_t": (lo_t, hi_t),
        "significant": bool(sig_boot and sig_t),
        "significant_bootstrap": bool(sig_boot),
        "significant_t": bool(sig_t),
        "wins": sum(1 for d in diffs if d > 0),
        "losses": sum(1 for d in diffs if d < 0),
        "clusters_positive": sum(1 for m in cluster_means if m > 0),
        "clusters_negative": sum(1 for m in cluster_means if m < 0),
        "consistent": all(m > 0 for m in cluster_means) or all(m < 0 for m in cluster_means),
        "sd_cluster_means": sd_clusters,
        "se_cluster_means": se,
        "t_crit": t,
        "min_resolvable_effect": mde,
        "per_cluster": per_cluster,
        **icc,
    }


def compare_methods(
    baseline: Sequence[Any],
    variant: Sequence[Any],
    cluster_ids: Sequence[Hashable],
    iters: int = 10000,
    seed: int = 0,
    metric: str = "final",
) -> Dict[str, Any]:
    """Run the naive window bootstrap and the clustered one on the same data.

    The naive arm is computed here rather than imported so this works on floats
    and on any metric; it is a line-for-line copy of ``harness.paired_delta``'s
    resampling, which draws individual windows with replacement.
    """
    diffs = [metric_of(y, metric) - metric_of(x, metric) for x, y in zip(baseline, variant)]
    rng = random.Random(seed)
    n = len(diffs)
    means = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)]
    means.sort()
    naive = {
        "delta": statistics.mean(diffs),
        "ci95": (means[int(0.025 * iters)], means[min(int(0.975 * iters), iters - 1)]),
    }
    naive["significant"] = naive["ci95"][0] > 0 or naive["ci95"][1] < 0
    naive["halfwidth"] = (naive["ci95"][1] - naive["ci95"][0]) / 2

    clustered = clustered_paired_delta(baseline, variant, cluster_ids, iters=iters, seed=seed, metric=metric)
    cl_half = (clustered["ci95_cluster_t"][1] - clustered["ci95_cluster_t"][0]) / 2
    clustered["halfwidth_t"] = cl_half
    clustered["halfwidth_bootstrap"] = (
        clustered["ci95_cluster_bootstrap"][1] - clustered["ci95_cluster_bootstrap"][0]
    ) / 2
    return {
        "naive": naive,
        "clustered": clustered,
        "overconfidence": (cl_half / naive["halfwidth"]) if naive["halfwidth"] > 0 else float("nan"),
    }


def format_report(cmp: Dict[str, Any], label: str = "variant vs base") -> str:
    """Human-readable side-by-side. Returns a string; caller prints it."""
    nv, cl = cmp["naive"], cmp["clustered"]
    lines = [
        f"{label}   metric={cl['metric']}  n_windows={cl['n_windows']}  n_conversations={cl['n_clusters']}",
        "",
        f"  naive   (window bootstrap)  delta {nv['delta']:+.4f}  "
        f"CI95 [{nv['ci95'][0]:+.4f}, {nv['ci95'][1]:+.4f}]  halfwidth {nv['halfwidth']:.4f}  "
        f"{'SIGNIFICANT' if nv['significant'] else 'not significant'}",
        f"  clustered (bootstrap)       delta {cl['delta']:+.4f}  "
        f"CI95 [{cl['ci95_cluster_bootstrap'][0]:+.4f}, {cl['ci95_cluster_bootstrap'][1]:+.4f}]  "
        f"halfwidth {cl['halfwidth_bootstrap']:.4f}  "
        f"{'SIGNIFICANT' if cl['significant_bootstrap'] else 'not significant'}",
        f"  clustered (t, {cl['n_clusters']-1} df)        delta {cl['delta_cluster_mean']:+.4f}  "
        f"CI95 [{cl['ci95_cluster_t'][0]:+.4f}, {cl['ci95_cluster_t'][1]:+.4f}]  "
        f"halfwidth {cl['halfwidth_t']:.4f}  "
        f"{'SIGNIFICANT' if cl['significant_t'] else 'not significant'}",
        "",
        f"  VERDICT: {'SIGNIFICANT' if cl['significant'] else 'NOT significant'} "
        f"(both clustered intervals must exclude zero)",
        f"  naive interval is {cmp['overconfidence']:.1f}x too narrow",
        f"  ICC {cl['icc']:.3f}  design effect {cl['design_effect']:.1f}  "
        f"effective n {cl['n_eff']:.1f} of {cl['n_windows']} windows",
        f"  minimum resolvable effect on this corpus: {cl['min_resolvable_effect']:.4f}",
        "",
        f"  per conversation ({cl['clusters_positive']} positive / {cl['clusters_negative']} negative, "
        f"{'consistent' if cl['consistent'] else 'NOT consistent'}):",
    ]
    for pc in cl["per_cluster"]:
        lines.append(
            f"    {str(pc['cluster'])[:24]:24s} n={pc['n']:3d}  base {pc['baseline_mean']:.4f} -> "
            f"var {pc['variant_mean']:.4f}   delta {pc['delta']:+.4f}  W/L {pc['wins']}/{pc['losses']}"
        )
    return "\n".join(lines)
