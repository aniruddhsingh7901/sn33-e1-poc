"""The fast offline scorer must agree with the validator's own class.

``sn33.scoring`` exists so a strategy sweep can score thousands of tag lists
without network or async overhead. That is only legitimate while it produces
exactly the number the real
``GroundTruthTagSimilarityScoringMechanism`` would produce, so this test drives
both with the same random inputs and asserts equality.

If this test fails, every benchmark number in the repo is suspect.
"""

from __future__ import annotations

import asyncio
import random

import numpy as np
import pytest

from conversationgenome.api.models.conversation_metadata import ConversationMetadata
from conversationgenome.scoring_mechanism.GroundTruthTagSimilarityScoringMechanism import (
    GroundTruthTagSimilarityScoringMechanism,
)
from conversationgenome.scoring_mechanism.NoPenaltyGroundTruthTagSimilarityScoringMechanism import (
    NoPenaltyGroundTruthTagSimilarityScoringMechanism,
)
from sn33 import scoring

DIMS = 32  # small vectors keep the test fast; the maths is dimension-agnostic


class _Axon:
    hotkey = "hk"
    uuid = "uuid"


class _Dendrite:
    status_code = 200


class _Response:
    def __init__(self, output):
        self.axon = _Axon()
        self.dendrite = _Dendrite()
        self.cgp_output = output


class _Bundle:
    def __init__(self, metadata):
        class _Input:
            pass

        self.input = _Input()
        self.input.metadata = metadata


def _vec(rng: random.Random) -> list:
    return [rng.uniform(-1, 1) for _ in range(DIMS)]


def _make_case(rng: random.Random):
    """Random ground truth and miner answer with a controlled overlap."""
    n_gt = rng.randint(3, 20)
    gt_tags = [f"gt tag {i}" for i in range(n_gt)]
    n_miner = rng.randint(1, 25)
    n_overlap = rng.randint(0, min(n_miner, n_gt))

    miner_tags = rng.sample(gt_tags, n_overlap) + [f"miner tag {i}" for i in range(n_miner - n_overlap)]
    rng.shuffle(miner_tags)

    vectors = {t: {"vectors": _vec(rng)} for t in gt_tags}
    miner_vectors = {t: {"vectors": _vec(rng)} for t in miner_tags}
    for t in miner_tags:
        if t in vectors:
            miner_vectors[t] = vectors[t]
    # Occasionally drop a vector: the validator scores those tags as 0 but still
    # counts them in mean/median, which is easy to get wrong.
    if miner_tags and rng.random() < 0.3:
        del miner_vectors[rng.choice(miner_tags)]
    return gt_tags, vectors, miner_tags, miner_vectors


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("penalties", [True, False])
def test_matches_validator(seed: int, penalties: bool) -> None:
    rng = random.Random(seed)
    gt_tags, gt_vectors, miner_tags, miner_vectors = _make_case(rng)
    min_tags = rng.choice([1, 3])

    mech = (
        GroundTruthTagSimilarityScoringMechanism()
        if penalties
        else NoPenaltyGroundTruthTagSimilarityScoringMechanism()
    )
    mech.min_tags = min_tags
    metadata = ConversationMetadata(tags=gt_tags, vectors=gt_vectors)
    response = _Response([{"tags": list(miner_tags), "vectors": miner_vectors}])

    real_scores, _ = asyncio.get_event_loop().run_until_complete(
        mech.evaluate(_Bundle(metadata), [response])
    ) if False else asyncio.run(mech.evaluate(_Bundle(metadata), [response]))
    real = real_scores[0]["final_miner_score"]

    # The real path iterates list(set(tags)); when more than 21 tags are present
    # the subset scored depends on set ordering, which we cannot pin down. Those
    # cases are excluded from the equality assertion (and the miner never
    # submits that many - see MAX_TAGS).
    if len(set(miner_tags)) > scoring.MAX_SCORED_TAGS + 1:
        pytest.skip("set-ordering dependent above 21 tags")

    target = scoring.neighborhood({k: v["vectors"] for k, v in gt_vectors.items()})
    ours = scoring.score_tags(
        gt_tags,
        target,
        miner_tags,
        {k: v["vectors"] for k, v in miner_vectors.items()},
        penalties=penalties,
        min_tags=min_tags,
    )

    assert ours["final"] == pytest.approx(float(real), abs=1e-9), (
        f"seed={seed} penalties={penalties} ours={ours['final']} real={real}"
    )


def test_zero_padding_costs_what_we_think() -> None:
    """Two strong unique tags must score materially worse than three.

    This is the single largest scoring trap: top_3_mean pads to three entries
    with zeros, so the third missing tag silently removes a third of 55% of the
    score.
    """
    gt = ["alpha", "beta"]
    v = [1.0] + [0.0] * (DIMS - 1)
    gt_vectors = {t: v for t in gt}
    target = scoring.neighborhood(gt_vectors)

    near = [0.9] + [0.436] + [0.0] * (DIMS - 2)  # cosine ~0.9 with target
    two = scoring.score_tags(gt, target, ["one", "two"], {"one": near, "two": near}, min_tags=1)
    three = scoring.score_tags(
        gt, target, ["one", "two", "three"], {"one": near, "two": near, "three": near}, min_tags=1
    )
    assert three["top_3_mean"] > two["top_3_mean"]
    assert two["top_3_mean"] == pytest.approx(two["max_score"] * 2 / 3, abs=1e-6)
