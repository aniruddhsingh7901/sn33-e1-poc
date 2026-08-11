"""Deep enrichment extraction: pool-only, and free to fail.

Enrichment supplies 87.7% of the validator's ground-truth tags (17.8 of 20.2,
measured 2026-08-08), so re-extracting each enrichment line at 30 tags instead
of the upstream 10 is the cheapest way to see more of the target. Probed at
+0.024 adjusted on 4/4 conversations.

The two ways it can go wrong are both cheaper to catch here than in production:

1. **Leakage.** Deep tags string-match real ground truth 10.3% of the time
   against 26.2% for upstream-depth tags. If they reach ``predicted_gt`` or the
   anchor set, the verbatim insurance dilutes and the flat x0.9 ``no_both_tags``
   penalty starts firing - a sibling probe turned +0.043 adjusted into -0.024
   final that way, on 38 of 38 windows.
2. **Latency.** At an 8s deadline 15% of tasks were truncated and scored 0.1994
   against 0.5517. A missing deep result must degrade to today's answer, never
   to a truncated one.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from sn33 import llm, pipeline, prompts, replica

WINDOW = [
    (0, "So the mortgage rates moved again this quarter and buyers are hesitating."),
    (1, "Right, and the rental market in the metro area is absorbing that demand."),
    (0, "We are seeing more multifamily construction starts than last year."),
]

ENRICHMENT = [
    "Mortgage rates hit a two-year high - lenders report a sharp fall in applications.",
    "Multifamily construction starts climb as metro rental demand outpaces supply.",
]


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# 1. the prompt: exact upstream string, or a loud failure
# --------------------------------------------------------------------------

def test_deep_prompt_replaces_the_upstream_tag_cap():
    base = prompts.gt_enrichment("some enrichment line")
    deep = prompts.gt_enrichment_deep("some enrichment line", n=30)

    assert "Return at most 10 of the most important and relevant tags." in base
    assert "Return at most 10 of the most important and relevant tags." not in deep
    assert "Return at most 30 tags." in deep
    # Only that one line may differ - everything else is the validator's wording.
    assert len(base.splitlines()) == len(deep.splitlines())
    diff = [(a, b) for a, b in zip(base.splitlines(), deep.splitlines()) if a != b]
    assert len(diff) == 1, diff
    assert diff[0][0].startswith("5.  **Limit to most important:**")


def test_deep_prompt_keeps_the_instruction_number_in_the_coding_variant():
    deep = prompts.gt_enrichment_deep("line", n=25, coding=True)
    assert "6.  **Be exhaustive:** Return at most 25 tags." in deep


def test_deep_prompt_raises_if_upstream_moves_the_tag_cap(monkeypatch):
    """An upstream re-vendor must break the build, not silently return the old prompt."""
    monkeypatch.setitem(prompts._ENRICHMENT_LIMIT_LINE, False, "5. this line does not exist")
    with pytest.raises(ValueError, match="no longer contains the tag-cap line"):
        prompts.gt_enrichment_deep("line")


# --------------------------------------------------------------------------
# 2. containment: pool yes, predicted_gt / anchors never
# --------------------------------------------------------------------------

# A real English word so it survives the deep-tag language guard added
# 2026-08-08 (sn33.tags.is_probably_english) - which real deep tags do too.
# In the dictionary, absent from the shallow pool below, so it stays a unique
# provenance marker. A nonsense marker would be (correctly) dropped as non-English.
DEEP_MARKER = "escrow"


@pytest.fixture
def marked_llm(monkeypatch):
    """Deep calls return uniquely-identifiable tags; everything else does not."""

    async def chat(prompt, *a, **k):
        if "**Be exhaustive:**" in prompt:
            return ", ".join(f"{DEEP_MARKER} {i}" for i in range(30))
        if "<set0>" in prompt:  # the combine step
            return "mortgage rates, rental market, multifamily construction"
        return "mortgage rates, rental market, housing demand, construction starts"

    async def embed(texts, **k):
        # Deep tags are made slightly WEAKER than the rest so they can only be
        # selected on merit, never because nothing else was available.
        return {t: ([0.6, 0.8, 0.0] if DEEP_MARKER in t else [1.0, 0.0, 0.0]) for t in texts}

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "embed", embed)


def _mine(**cfg_kw):
    base = dict(use_local=False, use_pool=True, use_cache=False,
                deadline_s=9.0, call_timeout_s=7.0)
    base.update(cfg_kw)
    cfg = pipeline.Config(**base)
    return _run(pipeline.mine("conversation_tagging", window=WINDOW,
                              enrichment=ENRICHMENT, cfg=cfg))


def test_deep_tags_reach_the_candidate_pool(marked_llm):
    res = _mine(use_deep_enrichment=True)
    assert any(DEEP_MARKER in t for t in res.candidates), \
        "deep tags never made it into the pool - the feature is inert"


def test_deep_tags_never_reach_predicted_gt(marked_llm):
    """predicted_gt is the exact-match insurance AND the target estimate."""
    res = _mine(use_deep_enrichment=True)
    assert not any(DEEP_MARKER in t for t in res.predicted_gt), \
        f"deep tags leaked into predicted_gt: {res.predicted_gt}"


def test_deep_tags_never_reach_the_replica_combine_input(marked_llm):
    """`all_sets` is what feeds combine, `tags` and therefore predicted_gt."""
    rep = _run(replica.replicate(
        "conversation_tagging",
        convo_xml="<conversation id='1'><p0>rates</p0></conversation>",
        enrichment=ENRICHMENT,
        timeout=5.0,
        deep_enrichment=30,
    ))
    assert rep.deep_tags, "deep extraction produced nothing - test is not exercising the path"
    flat = [t for s in rep.all_sets for t in s]
    assert not any(DEEP_MARKER in t for t in flat), f"deep tags leaked into all_sets: {flat}"
    assert not any(DEEP_MARKER in t for t in rep.tags), f"deep tags leaked into rep.tags: {rep.tags}"


def test_deep_tags_are_not_treated_as_anchors(marked_llm, monkeypatch):
    """Anchors are the other route to the `verbatim` list in compose()."""
    seen = {}
    real_compose = pipeline.compose

    def spy(ranked, predicted_gt, profile, target_tags, insurance, anchors=None,
            min_unique=None, uniques_first=False):
        seen["anchors"] = set(anchors or ())
        seen["gt"] = list(predicted_gt)
        return real_compose(ranked, predicted_gt, profile, target_tags, insurance,
                            anchors=anchors, min_unique=min_unique, uniques_first=uniques_first)

    monkeypatch.setattr(pipeline, "compose", spy)
    _mine(use_deep_enrichment=True, use_anchors=True)
    assert seen, "compose was never called"
    assert not any(DEEP_MARKER in t for t in seen["anchors"]), seen["anchors"]
    assert not any(DEEP_MARKER in t for t in seen["gt"]), seen["gt"]


# --------------------------------------------------------------------------
# 3. degradation: a missing deep result == the feature being off
# --------------------------------------------------------------------------

@pytest.fixture
def deep_hangs(monkeypatch):
    """Deep calls never return; every other call is normal and fast."""

    async def chat(prompt, *a, **k):
        if "**Be exhaustive:**" in prompt:
            await asyncio.sleep(60)
        if "<set0>" in prompt:
            return "mortgage rates, rental market, multifamily construction"
        return "mortgage rates, rental market, housing demand, construction starts"

    async def embed(texts, **k):
        return {t: [1.0, 0.0, 0.0] for t in texts}

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "embed", embed)


@pytest.fixture
def deep_returns_none(monkeypatch):
    async def chat(prompt, *a, **k):
        if "**Be exhaustive:**" in prompt:
            return None
        if "<set0>" in prompt:
            return "mortgage rates, rental market, multifamily construction"
        return "mortgage rates, rental market, housing demand, construction starts"

    async def embed(texts, **k):
        return {t: [1.0, 0.0, 0.0] for t in texts}

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "embed", embed)


def test_deep_returning_none_is_identical_to_the_feature_being_off(deep_returns_none):
    off = _mine(use_deep_enrichment=False)
    on = _mine(use_deep_enrichment=True)
    assert on.tags == off.tags, f"{on.tags} != {off.tags}"


def test_deep_hanging_is_identical_to_the_feature_being_off(deep_hangs):
    off = _mine(use_deep_enrichment=False)
    on = _mine(use_deep_enrichment=True)
    assert on.tags == off.tags, f"{on.tags} != {off.tags}"


def test_hanging_deep_calls_cannot_truncate_the_answer(deep_hangs):
    """The gate, stated as the number that matters.

    A truncated replica scores 0.1994 against 0.5517, so the deep calls may
    never push the miner past its deadline.
    """
    t0 = time.perf_counter()
    res = _mine(use_deep_enrichment=True, deadline_s=6.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 6.5, f"deep calls overran the deadline: {elapsed:.1f}s"
    assert res.source == "ranked", f"answer degraded to {res.source!r}"
    assert len(res.tags) >= 3


def test_deep_grace_never_eats_the_combine_reserve(deep_hangs):
    """The grace is clamped by the remaining fan-out budget, not just by DEEP_GRACE_S."""
    rep = _run(replica.replicate(
        "conversation_tagging",
        convo_xml="<conversation id='1'><p0>rates</p0></conversation>",
        enrichment=ENRICHMENT,
        timeout=5.0,
        deadline=4.0,
        deep_enrichment=30,
    ))
    assert rep.deep_tags == []
    assert rep.tags, "losing the deep calls must not lose the replica"
    assert not rep.degraded


def test_feature_off_issues_no_deep_calls(monkeypatch):
    """The A/B must be a single variable - and off must cost nothing."""
    prompts_seen = []

    async def chat(prompt, *a, **k):
        prompts_seen.append(prompt)
        return "mortgage rates, rental market, housing demand"

    async def embed(texts, **k):
        return {t: [1.0, 0.0, 0.0] for t in texts}

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "embed", embed)
    _mine(use_deep_enrichment=False)
    assert not any("**Be exhaustive:**" in p for p in prompts_seen)


def test_default_config_leaves_deep_enrichment_off():
    assert pipeline.Config().use_deep_enrichment is False


def test_theme_tags_reach_pool_and_failure_is_safe(monkeypatch):
    """Theme tags feed the candidate pool; a theme-call failure degrades to the
    same answer as the feature being off (never sinks the response)."""
    # A real dictionary word (real theme tags are broad English), unique to the
    # theme call, so it survives the is_probably_english pool filter and stays a
    # findable provenance marker.
    THEME = "escrow"

    async def chat(prompt, *a, **k):
        if "BROAD theme tags" in prompt:
            return f"{THEME} markets, {THEME} finance, housing policy"
        if "<set0>" in prompt:
            return "mortgage rates, rental market, multifamily construction"
        return "mortgage rates, rental market, housing demand, construction starts"

    async def embed(texts, **k):
        return {t: [1.0, 0.0, 0.0] for t in texts}

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "embed", embed)
    res = _mine(use_theme_tags=True)
    assert any(THEME in t for t in res.candidates), "theme tags never reached the pool"
    assert not any(THEME in t for t in res.predicted_gt), "theme leaked into predicted_gt"

    # a raising theme call must not sink the answer
    async def chat_boom(prompt, *a, **k):
        if "BROAD theme tags" in prompt:
            raise RuntimeError("theme down")
        if "<set0>" in prompt:
            return "mortgage rates, rental market, multifamily construction"
        return "mortgage rates, rental market, housing demand, construction starts"
    monkeypatch.setattr(llm, "chat", chat_boom)
    res2 = _mine(use_theme_tags=True)
    assert res2.tags and len(res2.tags) >= 3, "theme failure sank the answer"


def test_translation_replaces_spanish_candidates_and_keeps_centroid(monkeypatch):
    """Spanish candidates get translated to English; a failed translation degrades
    to the original candidates (screen-safe floor still prevents a zero)."""
    SPAN = ["partido republicano", "residencia permanente", "ajuste de estatus"]

    async def chat(prompt, *a, **k):
        if "Translate each tag" in prompt:
            return "republican party, permanent residency, adjustment of status"
        if "<set0>" in prompt:
            return ", ".join(SPAN)
        if "candidate topic tags" in prompt:   # miner pool prompt (English)
            return "us politics, immigration policy, voter turnout"
        return ", ".join(SPAN)      # replica GT is Spanish

    async def embed(texts, **k):
        return {t: [1.0, 0.0, 0.0] for t in texts}

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "embed", embed)
    res = _mine(translate_non_english=True)
    # the Spanish GT tags must not be in the final answer; English translations are
    assert not any(t in SPAN for t in res.tags), f"Spanish shipped: {res.tags}"

    # a failing translation must not sink the answer
    async def chat_boom(prompt, *a, **k):
        if "Translate each tag" in prompt:
            raise RuntimeError("translate down")
        if "<set0>" in prompt:
            return ", ".join(SPAN)
        if "candidate topic tags" in prompt:
            return "us politics, immigration policy, voter turnout"
        return ", ".join(SPAN)
    monkeypatch.setattr(llm, "chat", chat_boom)
    res2 = _mine(translate_non_english=True)
    assert res2.tags and len(res2.tags) >= 3, "translation failure sank the answer"


# ---------------------------------------------------------------------------
# 2026-08-10: the combine call runs CONCURRENTLY with the deep grace (a pure
# reordering - combine never reads deep_tags). These pin the overlap and the
# unchanged outputs.
# ---------------------------------------------------------------------------

def test_combine_starts_before_slow_deep_finishes(monkeypatch):
    """The truncation mechanism was grace-then-combine SERIALIZED; assert the
    combine now begins while a slow deep call is still in flight."""
    import time as _t
    marks = {}

    async def chat(prompt, *a, **k):
        if "**Be exhaustive:**" in prompt:      # deep call: slow straggler
            await asyncio.sleep(1.0)
            marks["deep_end"] = _t.perf_counter()
            return "deep straggler tag"
        if "<set0>" in prompt:                  # combine
            marks.setdefault("combine_start", _t.perf_counter())
            await asyncio.sleep(0.2)
            return "mortgage rates, rental market, multifamily construction"
        return "mortgage rates, rental market, housing demand, construction starts"

    monkeypatch.setattr(llm, "chat", chat)
    rep = _run(replica.replicate(
        "conversation_tagging", convo_xml="<c/>", document="doc",
        enrichment=["line one", "line two"], deadline=8.0, timeout=6.0,
        deep_enrichment=30,
    ))
    assert "combine_start" in marks and "deep_end" in marks
    # the whole point: combine did NOT wait for the straggler
    assert marks["combine_start"] < marks["deep_end"]
    # and neither result was sacrificed
    assert rep.tags == ["mortgage rates", "rental market", "multifamily construction"]
    assert "deep straggler tag" in rep.deep_tags


def test_combine_output_identical_with_and_without_deep(marked_llm):
    """Overlap must not change WHAT combine produces - only when it starts."""
    with_deep = _run(replica.replicate(
        "conversation_tagging", convo_xml="<c/>", document="doc",
        enrichment=["line one", "line two"], deadline=8.0, timeout=6.0,
        deep_enrichment=30,
    ))
    without_deep = _run(replica.replicate(
        "conversation_tagging", convo_xml="<c/>", document="doc",
        enrichment=["line one", "line two"], deadline=8.0, timeout=6.0,
        deep_enrichment=0,
    ))
    assert with_deep.tags == without_deep.tags


def test_tight_budget_still_degrades_to_local_combine(monkeypatch):
    """The pre-launch guard mirrors the sequential path: deadline nearly spent
    -> local_combine fallback, degraded=True, no LLM combine attempted."""
    async def chat(prompt, *a, **k):
        assert "<set0>" not in prompt, "combine must not be attempted under 1.2s left"
        return "mortgage rates, rental market"

    monkeypatch.setattr(llm, "chat", chat)
    rep = _run(replica.replicate(
        "conversation_tagging", convo_xml="<c/>", document="doc",
        enrichment=["line one"], deadline=1.0, timeout=6.0,
    ))
    assert rep.degraded is True
    assert rep.tags  # local_combine union, not empty
