"""The miner must always answer, in time, whatever the LLM does.

A zero costs more than a mediocre answer: ``get_raw_weights`` drops zero-scored
UIDs from weight distribution entirely, and ``update_scores`` decays the EMA by
5% on every zero. So each failure mode below must still produce a valid,
normalized, submittable tag list inside the deadline.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from sn33 import llm, pipeline
from sn33.tags import survives_validation

WINDOW = [
    (0, "So the mortgage rates moved again this quarter and buyers are hesitating."),
    (1, "Right, and the rental market in the metro area is absorbing that demand."),
    (0, "We are seeing more multifamily construction starts than last year."),
]

DOC_WINDOW = [
    (0, "# Housing Market Report\n\nMortgage rates, rental yields and multifamily "
        "construction across major metros. Includes transaction records and lending data."),
    (1, "Housing starts rise\nMultifamily construction increased 12% year over year."),
]


def _cfg(**kw):
    base = dict(use_local=True, use_pool=True, use_cache=False, deadline_s=9.0, call_timeout_s=7.0)
    base.update(kw)
    return pipeline.Config(**base)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def kill_llm(monkeypatch):
    """Every LLM call raises, as during an OpenAI outage."""

    async def boom(*a, **k):
        raise RuntimeError("simulated provider outage")

    monkeypatch.setattr(llm, "chat", boom)
    monkeypatch.setattr(llm, "embed", boom)


@pytest.fixture
def hang_llm(monkeypatch):
    """Every LLM call hangs well past the deadline."""

    async def hang(*a, **k):
        await asyncio.sleep(60)

    monkeypatch.setattr(llm, "chat", hang)
    monkeypatch.setattr(llm, "embed", hang)


@pytest.fixture
def garbage_llm(monkeypatch):
    """The model answers, but with prose instead of tags."""

    async def junk(*a, **k):
        return "I'm sorry, I can't help with that request. 🎉🎉🎉"

    async def embed(texts, **k):
        return {}

    monkeypatch.setattr(llm, "chat", junk)
    monkeypatch.setattr(llm, "embed", embed)


def _assert_valid(res, min_tags: int) -> None:
    assert len(res.tags) >= min_tags, f"only {len(res.tags)} tags: {res.tags}"
    assert len(res.tags) <= pipeline.MAX_TAGS, "20+ tags triggers the validator's random cull"
    assert len(res.tags) == len(set(res.tags)), "duplicates are collapsed before scoring"
    for t in res.tags:
        assert survives_validation(t), f"{t!r} would be deleted by validate_tag_set"


def test_llm_dead_still_answers(kill_llm):
    res = _run(pipeline.mine("conversation_tagging", window=WINDOW, cfg=_cfg()))
    _assert_valid(res, min_tags=3)
    assert res.source == "local"


def test_llm_hang_returns_before_deadline(hang_llm):
    t0 = time.perf_counter()
    res = _run(pipeline.mine("conversation_tagging", window=WINDOW, cfg=_cfg(deadline_s=3.0)))
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"blew the deadline: {elapsed:.1f}s"
    _assert_valid(res, min_tags=3)


def test_garbage_output_still_answers(garbage_llm):
    res = _run(pipeline.mine("webpage_metadata_generation", window=DOC_WINDOW, cfg=_cfg()))
    _assert_valid(res, min_tags=3)


def test_empty_input_does_not_raise():
    for kind in pipeline.TASK_PROFILE:
        res = _run(pipeline.mine(kind, window=[], cfg=_cfg(use_pool=False, deadline_s=2.0)))
        assert isinstance(res.tags, list)  # may be empty; must never raise


def test_never_exceeds_19_tags(monkeypatch):
    """Even when the model floods us with candidates."""

    async def flood(*a, **k):
        return ", ".join(f"topic number {i}" for i in range(80))

    async def embed(texts, **k):
        return {t: [1.0, 0.0, 0.0] for t in texts}

    monkeypatch.setattr(llm, "chat", flood)
    monkeypatch.setattr(llm, "embed", embed)
    res = _run(pipeline.mine("conversation_tagging", window=WINDOW, cfg=_cfg()))
    assert len(res.tags) <= pipeline.MAX_TAGS


def test_deadline_is_respected_under_slow_calls(monkeypatch):
    """Calls that are slow but not infinite still cannot overrun the budget."""

    async def slow(*a, **k):
        await asyncio.sleep(2.5)
        return "real estate, banking, finance, property data, market trends"

    async def slow_embed(texts, **k):
        await asyncio.sleep(2.5)
        return {t: [1.0, 0.0, 0.0] for t in texts}

    monkeypatch.setattr(llm, "chat", slow)
    monkeypatch.setattr(llm, "embed", slow_embed)

    t0 = time.perf_counter()
    res = _run(pipeline.mine("webpage_metadata_generation", window=DOC_WINDOW, cfg=_cfg(deadline_s=4.0)))
    elapsed = time.perf_counter() - t0
    assert elapsed < 6.0, f"took {elapsed:.1f}s against a 4s deadline"
    _assert_valid(res, min_tags=3)


def test_greedy_compose_returns_a_usable_set():
    """Regression: greedy must not return an empty list.

    Growing from an empty list cannot work - every set below `min_tags` scores
    0, so no first tag ever looks like an improvement and the search never
    starts. That bug returned [] on every call and was invisible in the bench,
    because the caller silently fell back to the quota composer and produced
    byte-identical results.
    """
    import numpy as np

    from sn33 import pipeline as P

    profile = {"min_tags": 3, "target_tags": 12, "insurance": 6, "penalties": True}
    target = np.array([1.0, 0.0, 0.0])

    def vec(c):
        return [c, (1 - c * c) ** 0.5, 0.0]

    cands = [f"tag {i}" for i in range(20)]
    cos = [0.80 - 0.03 * i for i in range(20)]
    vectors = {t: vec(c) for t, c in zip(cands, cos)}
    ranked = list(zip(cands, cos))
    predicted_gt = cands[2:7]

    picked = P.compose_greedy(ranked, predicted_gt, vectors, target, profile)
    assert len(picked) >= profile["min_tags"], "greedy returned an unusable set"
    assert len(picked) <= P.MAX_TAGS
    assert len(picked) == len(set(picked))

    # It must be at least as good as the quota composer on the estimate it
    # optimises - otherwise there is no reason to run it.
    from sn33 import scoring

    quota = P.compose(ranked, predicted_gt, profile, 12, 6, anchors=set())
    est = lambda tags: scoring.score_tags(
        predicted_gt, target, tags, vectors, penalties=True, min_tags=3
    )["final"]
    assert est(picked) >= est(quota)


def test_greedy_handles_a_pool_below_the_floor():
    import numpy as np

    from sn33 import pipeline as P

    profile = {"min_tags": 3, "target_tags": 12, "insurance": 2, "penalties": True}
    ranked = [("only tag", 0.7)]
    vectors = {"only tag": [1.0, 0.0, 0.0]}
    assert P.compose_greedy(ranked, [], vectors, np.array([1.0, 0.0, 0.0]), profile) == []


def test_compose_keeps_unique_margin_when_predictions_overlap():
    """Regression for the production failure at 8h.

    `compose` labels a tag "unique" when it is absent from the ground truth WE
    predicted. The validator diffs against the ground truth IT generated, and
    those sets overlap more than the offline bench assumed. 6 of 36 scored
    tasks tripped the unique floor - 4 of them with a single unique tag, which
    zero-pads top_3_mean and costs over half the score.

    So compose must leave margin: aim above the floor of 3, not at it.
    """
    from sn33 import pipeline as P

    profile = {"min_tags": 3, "target_tags": 12, "insurance": 3, "penalties": True}
    # 20 candidates; the top 8 are all predicted ground truth, so a naive
    # top-N selection would produce almost no unique tags.
    ranked = [(f"tag {i}", 0.80 - 0.02 * i) for i in range(20)]
    predicted_gt = [f"tag {i}" for i in range(8)]

    chosen = P.compose(ranked, predicted_gt, profile, 12, 3, anchors=set())
    gt = set(predicted_gt)
    uniques = [t for t in chosen if t not in gt]

    assert len(uniques) >= P.MIN_UNIQUE_TARGET, (
        f"only {len(uniques)} expected-unique tags: {chosen}"
    )
    assert P.MIN_UNIQUE_TARGET > 3, "the whole point is margin above the hard floor"
    # insurance must still be honoured so no_both_tags (flat 10%) cannot fire
    assert sum(1 for t in chosen if t in gt) >= 1


def test_variants_never_inflect_proper_nouns_or_acronyms():
    """Variants are back on, but only behind the spaCy part-of-speech guard.

    The original implementation pluralised anything: "dallas texas" ->
    "dallas texases"/"dallas texa", "iran" -> "irans", "nfl" -> "nfls". The
    validator's English screen deleted every one, costing tag slots and
    starving the unique count. spaCy tags those heads as PROPN, so they are now
    skipped, while real common nouns still inflect.
    """
    from sn33.pipeline import Config
    from sn33.variants import variants_of

    assert Config().use_variants is True

    for tag in ["dallas texas", "iran", "nfl", "ie7", "israel", "frisco texas",
                "commit splitting", "browser compatibility"]:
        assert variants_of(tag) == [], f"{tag!r} must not be inflected"

    # a genuine common noun still produces its plural
    assert "transcripts" in variants_of("transcript")


def test_15_slots_hold_both_insurance_and_unique_margin():
    """The two penalties pull in opposite directions; the list must satisfy both.

    Production measured the trade-off directly:
      insurance 6 / 12 tags -> unique starvation (x0.95 fired)
      insurance 3 / 12 tags -> no_both_tags roughly doubled (x0.90 fired)

    Twelve slots cannot hold 6 verbatim ground-truth predictions AND 5+ distinct
    tags. Fifteen can, and 12-vs-16 measured within noise offline, so the extra
    slots cost nothing.
    """
    from sn33 import pipeline as P

    profile = P.TASK_PROFILE["conversation_tagging"]
    assert profile["target_tags"] >= profile["insurance"] + P.MIN_UNIQUE_TARGET, (
        "not enough slots to satisfy both the exact-match quota and the unique margin"
    )

    ranked = [(f"tag {i}", 0.80 - 0.01 * i) for i in range(30)]
    predicted_gt = [f"tag {i}" for i in range(10)]
    chosen = P.compose(ranked, predicted_gt, profile,
                       profile["target_tags"], profile["insurance"], anchors=set())
    gt = set(predicted_gt)

    assert sum(1 for t in chosen if t in gt) >= profile["insurance"], "too few exact-match bets"
    assert sum(1 for t in chosen if t not in gt) >= P.MIN_UNIQUE_TARGET, "unique margin lost"
    assert len(chosen) <= P.MAX_TAGS
