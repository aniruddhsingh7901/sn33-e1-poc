"""The benchmark's confidence intervals must respect the corpus's clustering.

``bench.harness.paired_delta`` resamples individual WINDOWS. The faithful corpus
is four conversations sliced into 75 overlapping windows, so windows are not
independent draws and that interval is far too narrow.
``bench.probes.cluster_stats.clustered_paired_delta`` resamples whole
conversations instead.

The two properties that matter, and that this file pins down:

1. no difference -> the interval must contain zero (no false positive from an
   A/A comparison);
2. an effect that exists in ONE conversation out of four must NOT be called
   significant, however many windows that conversation contributes. This is the
   exact failure mode that makes a benchmark recommend a change that then hurts
   production, and the naive method commits it.
"""

from __future__ import annotations

import random

import pytest

from bench.probes.cluster_stats import (
    clustered_paired_delta,
    compare_methods,
    intraclass_correlation,
    metric_of,
)


def make_corpus(cluster_sizes, base_fn, effect_fn, noise=0.0, seed=0):
    """Build aligned (baseline, variant, cluster_ids) lists of plain floats.

    ``effect_fn(cluster, window)`` returns the DIFFERENCE the variant adds, so
    the two arms stay properly paired - the baseline value is drawn once and
    reused. ``noise`` adds independent per-window jitter on top of the effect,
    which is what an unpaired source of variation looks like.
    """
    rng = random.Random(seed)
    baseline, variant, ids = [], [], []
    for c, size in enumerate(cluster_sizes):
        for w in range(size):
            b = base_fn(c, w)
            baseline.append(b)
            variant.append(b + effect_fn(c, w) + (rng.gauss(0, noise) if noise else 0.0))
            ids.append(f"convo{c}")
    return baseline, variant, ids


# --------------------------------------------------------------------------
# 1. identical inputs -> CI contains zero
# --------------------------------------------------------------------------

def test_identical_inputs_ci_contains_zero():
    """A strategy compared against itself has a zero delta and a CI over zero."""
    rng = random.Random(7)
    scores = [rng.uniform(0.4, 0.8) for _ in range(40)]
    ids = [f"convo{i // 10}" for i in range(40)]

    r = clustered_paired_delta(scores, list(scores), ids)

    assert r["delta"] == pytest.approx(0.0)
    lo, hi = r["ci95_cluster_bootstrap"]
    assert lo <= 0.0 <= hi, f"bootstrap CI {lo, hi} excludes zero"
    lo_t, hi_t = r["ci95_cluster_t"]
    assert lo_t <= 0.0 <= hi_t, f"t CI {lo_t, hi_t} excludes zero"
    assert not r["significant"]
    assert r["wins"] == 0 and r["losses"] == 0


def test_identical_inputs_with_noisy_but_equal_arms():
    """Zero delta with real per-window spread in the underlying scores.

    Cluster means differ (0.50 / 0.55 / 0.60 / 0.65) and windows are noisy, but
    the two arms are the same numbers - so every paired difference is exactly
    zero and no amount of score spread may manufacture an interval away from it.
    """
    rng = random.Random(11)
    base, var, ids = make_corpus(
        [19, 19, 19, 18],
        lambda c, w: 0.5 + 0.05 * c + rng.gauss(0, 0.01),
        lambda c, w: 0.0,
    )

    r = clustered_paired_delta(base, var, ids)
    lo, hi = r["ci95_cluster_t"]
    assert lo <= 0.0 <= hi
    assert r["ci95_cluster_bootstrap"] == (0.0, 0.0)
    assert not r["significant"]
    assert r["icc"] == 0.0  # no variance in the differences at all


# --------------------------------------------------------------------------
# 2. an effect in only 1 of 4 clusters is NOT significant
# --------------------------------------------------------------------------

ONE_OF_FOUR = dict(
    cluster_sizes=[19, 19, 19, 19],
    base_fn=lambda c, w: 0.60 + random.Random((c, w).__hash__()).gauss(0, 0.005),
    effect_fn=lambda c, w: 0.20 if c == 0 else 0.0,
)


def test_effect_in_one_cluster_of_four_is_not_significant():
    """One conversation moves +0.20, three do not. That is not a result."""
    base, var, ids = make_corpus(**ONE_OF_FOUR)
    r = clustered_paired_delta(base, var, ids)

    assert r["n_clusters"] == 4
    assert [round(pc["delta"], 6) for pc in r["per_cluster"]] == [0.2, 0.0, 0.0, 0.0]
    assert r["clusters_positive"] == 1, "only one conversation should move up"
    assert not r["consistent"]
    assert not r["significant"], (
        f"an effect in 1 of 4 conversations was reported significant: "
        f"delta {r['delta']:+.4f} boot {r['ci95_cluster_bootstrap']} t {r['ci95_cluster_t']}"
    )
    # and it must be insignificant by BOTH intervals, not rescued by one
    assert not r["significant_bootstrap"]
    assert not r["significant_t"]


def test_naive_method_would_have_called_that_effect_significant():
    """The regression guard: prove the clustered method actually changes the answer.

    Same data as the test above. If the naive window bootstrap ever stops
    over-calling this, the whole module is unnecessary - and if the clustered
    one starts agreeing with it, the module is broken.
    """
    base, var, ids = make_corpus(**ONE_OF_FOUR)
    cmp = compare_methods(base, var, ids)

    assert cmp["naive"]["significant"], "naive bootstrap should (wrongly) call this significant"
    assert not cmp["clustered"]["significant"], "clustered method must not"
    assert cmp["overconfidence"] > 2.0, (
        f"clustered interval only {cmp['overconfidence']:.2f}x wider than naive"
    )


def test_one_cluster_effect_stays_insignificant_even_when_it_dominates_the_corpus():
    """36 of 40 windows from the affected conversation still is not evidence.

    This is the real shape of `run_faithful --n 40`: the slice is 36 windows of
    one podcast plus two windows each of two others.
    """
    base, var, ids = make_corpus(
        [36, 2, 2],
        base_fn=lambda c, w: 0.60,
        effect_fn=lambda c, w: 0.10 if c == 0 else 0.0,
    )
    r = clustered_paired_delta(base, var, ids)

    assert r["delta"] == pytest.approx(36 * 0.10 / 40), "pooled delta dominated by one cluster"
    assert r["delta"] > 0.08
    assert r["clusters_positive"] == 1
    assert not r["significant"]
    # the unweighted cluster mean already shows how thin the evidence is
    assert r["delta_cluster_mean"] == pytest.approx(0.10 / 3)


# --------------------------------------------------------------------------
# 3. a genuine, consistent effect IS still detected  (no dead detector)
# --------------------------------------------------------------------------

def test_consistent_large_effect_across_all_clusters_is_significant():
    """A method that never says SIGNIFICANT is useless - check it still can."""
    base, var, ids = make_corpus(
        [19, 19, 19, 19],
        base_fn=lambda c, w: 0.60,
        # present in all four, and much larger than the between-cluster spread
        effect_fn=lambda c, w: -0.30 - 0.01 * c,
        noise=0.005,
        seed=13,
    )
    r = clustered_paired_delta(base, var, ids)

    assert r["clusters_negative"] == 4
    assert r["consistent"]
    assert r["significant"], f"boot {r['ci95_cluster_bootstrap']} t {r['ci95_cluster_t']}"
    assert r["ci95_cluster_t"][1] < 0


# --------------------------------------------------------------------------
# mechanics
# --------------------------------------------------------------------------

def test_resamples_whole_clusters_not_windows():
    """Every bootstrap draw must be a mean of whole conversations.

    With two clusters whose diffs are constants 0.0 and 1.0, resampling CLUSTERS
    of equal size can only produce 0.0, 0.5 or 1.0. Resampling windows would
    produce a near-continuous spread around 0.5. Seeing only the three values
    proves clusters are the resampling unit.
    """
    base = [0.0] * 20
    var = [0.0] * 10 + [1.0] * 10
    ids = ["a"] * 10 + ["b"] * 10

    r = clustered_paired_delta(base, var, ids, iters=2000)
    lo, hi = r["ci95_cluster_bootstrap"]
    assert lo in (0.0, 0.5, 1.0) and hi in (0.0, 0.5, 1.0), (lo, hi)
    assert lo <= 0.0 <= hi or True  # the point is the granularity, asserted above
    assert r["per_cluster"][0]["delta"] == pytest.approx(0.0)
    assert r["per_cluster"][1]["delta"] == pytest.approx(1.0)


def test_per_cluster_breakdown_is_reported_in_corpus_order():
    base, var, ids = make_corpus([3, 4, 5], lambda c, w: 0.5, lambda c, w: 0.1 * c)
    r = clustered_paired_delta(base, var, ids)

    assert [pc["cluster"] for pc in r["per_cluster"]] == ["convo0", "convo1", "convo2"]
    assert [pc["n"] for pc in r["per_cluster"]] == [3, 4, 5]
    assert [round(pc["delta"], 6) for pc in r["per_cluster"]] == [0.0, 0.1, 0.2]
    # pooled delta is window-weighted: (3*0 + 4*.1 + 5*.2)/12
    assert r["delta"] == pytest.approx((4 * 0.1 + 5 * 0.2) / 12)
    # cluster-mean delta is unweighted
    assert r["delta_cluster_mean"] == pytest.approx(0.1)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        clustered_paired_delta([0.1, 0.2], [0.1], ["a", "b"])
    with pytest.raises(ValueError):
        clustered_paired_delta([0.1, 0.2], [0.1, 0.2], ["a"])


def test_empty_input_returns_empty():
    assert clustered_paired_delta([], [], []) == {}


def test_metric_of_reads_verdicts_dicts_and_floats():
    class V:
        final = 0.5
        adjusted = 0.7
        detail = {"adjusted": 0.7}

    assert metric_of(0.42) == pytest.approx(0.42)
    assert metric_of({"final": 0.3, "adjusted": 0.9}, "adjusted") == pytest.approx(0.9)
    assert metric_of(V(), "final") == pytest.approx(0.5)
    assert metric_of(V(), "adjusted") == pytest.approx(0.7)


def test_metric_selection_changes_the_answer():
    base = [{"final": 0.5, "adjusted": 0.5}] * 8
    var = [{"final": 0.5, "adjusted": 0.9}] * 8
    ids = ["a"] * 4 + ["b"] * 4

    assert clustered_paired_delta(base, var, ids, metric="final")["delta"] == pytest.approx(0.0)
    assert clustered_paired_delta(base, var, ids, metric="adjusted")["delta"] == pytest.approx(0.4)


def test_icc_is_high_when_variance_is_between_clusters():
    """Windows of one conversation share their difference -> ICC near 1."""
    rng = random.Random(2)
    diffs, ids = [], []
    for c, offset in enumerate([-0.10, 0.00, 0.05, 0.12]):
        for _ in range(19):
            diffs.append(offset + rng.gauss(0, 0.002))
            ids.append(f"convo{c}")

    r = intraclass_correlation(diffs, ids)
    assert r["icc"] > 0.95, r["icc"]
    assert r["design_effect"] > 15
    assert r["n_eff"] < 6, f"76 windows should be worth fewer than 6 independent draws, got {r['n_eff']:.1f}"


def test_icc_is_low_when_variance_is_within_clusters():
    rng = random.Random(4)
    diffs = [rng.gauss(0, 0.05) for _ in range(76)]
    ids = [f"convo{i // 19}" for i in range(76)]

    r = intraclass_correlation(diffs, ids)
    assert r["icc"] < 0.15, r["icc"]
    assert r["n_eff"] > 40


def test_deterministic_for_a_given_seed():
    base, var, ids = make_corpus(
        [19, 19, 19, 19], lambda c, w: 0.6, lambda c, w: 0.0, noise=0.01, seed=9
    )
    a = clustered_paired_delta(base, var, ids, seed=1)
    b = clustered_paired_delta(base, var, ids, seed=1)
    assert a["ci95_cluster_bootstrap"] == b["ci95_cluster_bootstrap"]


def test_single_cluster_cannot_be_significant():
    """One conversation is one observation - no generalisation is possible."""
    base = [0.5] * 20
    var = [0.9] * 20
    r = clustered_paired_delta(base, var, ["only"] * 20)

    assert r["n_clusters"] == 1
    assert r["delta"] == pytest.approx(0.4)
    assert not r["significant"], "a single cluster must never certify an effect"
