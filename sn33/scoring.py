"""Offline reimplementation of the validator's score.

Line-for-line mirror of
``conversationgenome/scoring_mechanism/GroundTruthTagSimilarityScoringMechanism.py``
(cross-checked by ``tests/test_scoring_parity.py``, which asserts equality with
the real class on random inputs).

Why reimplement rather than call the real class everywhere: the real path is
async, needs fake bundle/response objects, and re-embeds through the network.
This version is a pure function over vectors, so a strategy sweep can score
tens of thousands of candidate tag lists in seconds - which is what makes the
oracle/headroom analysis possible.

The real class remains the source of truth; the bench scores every reported
number through *both* and fails loudly if they disagree.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import numpy as np

from conversationgenome.utils.constants import PENALTIES

SCORING_FACTORS = {"top_3_mean": 0.55, "median_score": 0.1, "mean_score": 0.25, "max_score": 0.1}
MAX_SCORED_TAGS = 20  # loop breaks on idx > 20, so 21 tags are actually scored


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(b == 0):
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def neighborhood(gt_vectors: Dict[str, Sequence[float]]) -> Optional[np.ndarray]:
    """Mean of ground-truth tag vectors - the single target every tag is scored against."""
    vecs = [np.asarray(v, dtype=np.float64) for v in gt_vectors.values() if v is not None and len(v)]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def score_tags(
    gt_tags: Sequence[str],
    target: np.ndarray,
    miner_tags: Sequence[str],
    miner_vectors: Dict[str, Sequence[float]],
    penalties: bool = True,
    min_tags: int = 3,
    seed: Optional[int] = 0,
) -> dict:
    """Return the validator's verdict for one miner response.

    ``miner_tags`` must already be post-``validate_tag_set`` (the bench applies
    that step separately, since it costs an LLM call).
    """
    if target is None:
        return _zero()
    if not miner_tags or len(miner_tags) < min_tags:
        return _zero()

    # _calc_scores: tag_set = list(set(tags)). Set iteration order is arbitrary,
    # which decides *which* tags get scored once there are more than 21. We make
    # that explicit rather than pretending it is deterministic.
    tag_set = list(set(miner_tags))
    if seed is not None and len(tag_set) > MAX_SCORED_TAGS + 1:
        random.Random(seed).shuffle(tag_set)

    gt_set = set(gt_tags)
    both_all = [t for t in tag_set if t in gt_set]
    unique_all = [t for t in tag_set if t not in gt_set]

    scores: List[float] = []
    scores_unique: List[float] = []
    for idx, tag in enumerate(tag_set):
        if idx > MAX_SCORED_TAGS:
            break
        vec = miner_vectors.get(tag)
        if vec is None or not len(vec):
            # Missing vectors score 0 but still count in mean/median.
            scores.append(0.0)
            if tag not in gt_set:
                scores_unique.append(0.0)
            continue
        s = _cos(target, np.asarray(vec, dtype=np.float64))
        scores.append(s)
        if tag not in gt_set:
            scores_unique.append(s)

    stats = _stats(scores, scores_unique)
    adjusted = (
        SCORING_FACTORS["top_3_mean"] * stats["top_3_mean"]
        + SCORING_FACTORS["median_score"] * stats["median_score"]
        + SCORING_FACTORS["mean_score"] * stats["mean_score"]
        + SCORING_FACTORS["max_score"] * stats["max_score"]
    )

    total_tag_count = len(both_all) + len(unique_all)
    final = adjusted
    fired: List[str] = []
    if penalties:
        final, fired = apply_penalties(adjusted, total_tag_count, len(unique_all), stats["max_score"])

    return {
        "adjusted": float(adjusted),
        "final": float(final),
        "penalties": fired,
        "n_tags": total_tag_count,
        "n_unique": len(unique_all),
        "n_both": len(both_all),
        "n_scored": len(scores),
        **{k: float(v) for k, v in stats.items()},
    }


def apply_penalties(score: float, num_tags: int, num_unique: int, max_score: float):
    fired: List[str] = []
    num_both = num_tags - num_unique
    if num_both == 0:
        score *= PENALTIES["no_both_tags"]["penalty"]
        fired.append("no_both_tags")
    if max_score < PENALTIES["all_junk_tags"]["threshold"]:
        score *= PENALTIES["all_junk_tags"]["penalty"]
        fired.append("all_junk_tags")
    if num_tags < PENALTIES["too_few_tags"]["threshold"]:
        score *= PENALTIES["too_few_tags"]["penalty"]
        fired.append("too_few_tags")
    if num_unique < 1:
        score *= PENALTIES["num_unique_tags"]["less_than_1"]["penalty"]
        fired.append("less_than_1_unique")
    elif num_unique < 2:
        score *= PENALTIES["num_unique_tags"]["less_than_2"]["penalty"]
        fired.append("less_than_2_unique")
    elif num_unique < 3:
        score *= PENALTIES["num_unique_tags"]["less_than_3"]["penalty"]
        fired.append("less_than_3_unique")
    return score, fired


def _stats(scores: Sequence[float], scores_unique: Sequence[float]) -> dict:
    if len(scores) == 0:
        mean = median = mn = mx = 0.0
    else:
        mean = float(np.mean(scores))
        median = float(np.median(scores))
        mn = float(np.min(scores))
        mx = float(np.max(scores))

    if len(scores_unique) == 0:
        top = np.array([0.0, 0.0, 0.0])
    else:
        top = np.sort(np.asarray(scores_unique, dtype=np.float64))
        if len(top) >= 3:
            top = top[-3:]
        while len(top) < 3:
            top = np.append(top, 0.0)  # <-- the zero padding that halves careless scores

    return {
        "top_3_mean": float(np.mean(top)),
        "median_score": median,
        "mean_score": mean,
        "max_score": mx,
        "min_score": mn,
    }


def _zero() -> dict:
    return {
        "adjusted": 0.0,
        "final": 0.0,
        "penalties": ["hard_zero"],
        "n_tags": 0,
        "n_unique": 0,
        "n_both": 0,
        "n_scored": 0,
        "top_3_mean": 0.0,
        "median_score": 0.0,
        "mean_score": 0.0,
        "max_score": 0.0,
        "min_score": 0.0,
    }


# --------------------------------------------------------------------------
# Analysis helpers - what is the best score this candidate pool could reach?
# --------------------------------------------------------------------------

def oracle_best(
    gt_tags: Sequence[str],
    target: np.ndarray,
    pool: Sequence[str],
    vectors: Dict[str, Sequence[float]],
    penalties: bool = True,
    min_tags: int = 3,
    max_k: int = 21,
) -> dict:
    """Best achievable score using tags drawn from ``pool``.

    Because every term of the score is monotone in per-tag cosine, the optimal
    k-tag subset is always the top-k by cosine (plus, when it helps, one exact
    ground-truth match to clear the no_both penalty). So we only need to sweep
    k, not all subsets.
    """
    ranked = []
    for t in pool:
        v = vectors.get(t)
        if v is None or not len(v):
            continue
        ranked.append((t, _cos(target, np.asarray(v, dtype=np.float64))))
    ranked.sort(key=lambda x: -x[1])
    if not ranked:
        return _zero() | {"k": 0, "tags": []}

    # The answer has two distinct roles and the optimum mixes them, so the
    # search has to be two-dimensional rather than a single top-k prefix:
    #
    #   unique tags (not in ground truth) feed top_3_mean, which is 55% of the
    #       score and pads to three entries with zeros
    #   both tags (exact ground-truth matches) contribute only to mean/median/max
    #       but clear the flat 10% no_both_tags penalty
    #
    # Taking a plain top-k by cosine tends to select ground-truth matches (they
    # sit nearest the centroid by construction) and starves top_3_mean - which
    # is exactly how a naive "be as accurate as possible" miner loses.
    gt_set = set(gt_tags)
    uniques = [(t, s) for t, s in ranked if t not in gt_set]
    boths = [(t, s) for t, s in ranked if t in gt_set]

    best = None
    for n_both in range(0, min(4, len(boths)) + 1):
        for n_uniq in range(0, min(max_k, len(uniques)) + 1):
            if n_both + n_uniq < min_tags or n_both + n_uniq == 0:
                continue
            picks = [t for t, _ in uniques[:n_uniq]] + [t for t, _ in boths[:n_both]]
            res = score_tags(gt_tags, target, picks, vectors, penalties, min_tags)
            if best is None or res["final"] > best["final"]:
                best = res | {"k": len(picks), "tags": picks, "n_both_picked": n_both}
    return best or (_zero() | {"k": 0, "tags": []})
