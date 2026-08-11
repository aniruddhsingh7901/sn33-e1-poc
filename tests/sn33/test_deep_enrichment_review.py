from __future__ import annotations
import pytest

# The deep-enrichment feature is IMPLEMENTED BUT NOT SHIPPABLE, and these tests
# are the adversarial review that established that. They are the specification
# for the fix, not dead weight - each one documents a defect reproduced against
# the real code on 2026-08-08:
#
#   1. ASCII Spanish deep tags reach the submitted answer
#      (['prestamos bancarios','tasas hipotecarias']). sn33.tags rejects
#      non-ASCII LETTERS, so unaccented Spanish walks straight through, and
#      get_safe_tag then shreds nothing - it ships and scores near zero.
#   2. A bullet-list reply fuses into one tag
#      ('mortgage rates rental market housing demand').
#   3. A numbered "1. tag\n2. tag" reply yields ZERO usable tags.
#      2 and 3 share a root cause: the parser handles comma-delimited output
#      only, but a 30-tag request usually comes back as a list.
#   4. Template drift raises out of gt_enrichment_deep and empties
#      predicted_gt - killing the whole replica, not just the deep path.
#   5. DEEP_GRACE_S = 1.6 against a measured 3.5s deep latency: the grace is
#      always spent and never returns anything. Worse than no gate.
#   6. Deep tasks outlive a cancelled replica (5 leaked).
#
# All six defects were fixed 2026-08-08 (sn33/replica.py, sn33/pipeline.py,
# sn33/tags.py). These now assert the FIXED behaviour and must pass. The score
# gain (+0.024 claimed) is still unverified against fresh ground truth - the
# feature therefore ships OFF (Config.use_deep_enrichment = False) until a
# mainnet A/B confirms it, but the correctness guards below are permanent.

"""Adversarial review of the deep-enrichment change.

Each test names the defect it originally demonstrated and now guards against
regression.
"""


import asyncio
import time

import pytest

from sn33 import llm, pipeline, prompts, replica

WINDOW = [
    (0, "So the mortgage rates moved again this quarter and buyers are hesitating."),
    (1, "Right, and the rental market in the metro area is absorbing that demand."),
]

# The real fan-out width the latency probe measured on: 4-6 enrichment lines.
ENRICHMENT = [f"enrichment line number {i} about housing and mortgage markets." for i in range(5)]


def _run(coro):
    return asyncio.run(coro)


def _cfg(**kw):
    base = dict(use_local=False, use_pool=True, use_cache=False,
                deadline_s=11.0, call_timeout_s=8.0)
    base.update(kw)
    return pipeline.Config(**base)


# ---------------------------------------------------------------------------
# DEFECT 1 - DEEP_GRACE_S is set to the measured overrun, so at the measured
# latencies the deep calls are cancelled AND the 1.6s is still spent.
# replica.py:45, replica.py:252
# ---------------------------------------------------------------------------

@pytest.fixture
def measured_latencies(monkeypatch):
    """The numbers straight out of bench/probes/verify-enrichment-depth-latency.py.

    upstream10 fan-out 1.66-2.20s, deep30 fan-out 2.88-3.80s. Real scale, real
    DEEP_GRACE_S, real 11s deadline - no scaling, so nothing here is an artifact.
    """

    async def chat(prompt, *a, **k):
        if "**Be exhaustive:**" in prompt:
            await asyncio.sleep(3.80)                 # deep30, worst conversation
            return ", ".join(f"deep tag number {i}" for i in range(30))
        if "<set0>" in prompt:
            await asyncio.sleep(0.5)
            return "mortgage rates, rental market, multifamily construction"
        await asyncio.sleep(1.90)                     # upstream10, mid-range
        return "mortgage rates, rental market, housing demand, construction starts"

    monkeypatch.setattr(llm, "chat", chat)


def test_deep_grace_is_large_enough_for_the_measured_latencies(measured_latencies):
    """FIXED: at the measured fan-out times the deep calls now land.

    Shallow fan-out 1.90s, deep fan-out 3.80s -> the deep calls need ~1.90s more
    than the shallow ones. DEEP_GRACE_S was 1.60s (smaller than that gap, so the
    grace was always spent AND the deep tags always cancelled - the worst of
    both). It is now 4.0s, clamped by the remaining fan-out budget, so the deep
    tags land and the whole reply still finishes well inside the 11s deadline.
    """
    t0 = time.perf_counter()
    rep = _run(replica.replicate(
        "conversation_tagging",
        convo_xml="<conversation id='1'><p0>rates</p0></conversation>",
        enrichment=ENRICHMENT,
        timeout=8.0,
        deadline=11.0,         # production SN33_DEADLINE_S
        deep_enrichment=30,
    ))
    elapsed = time.perf_counter() - t0
    assert rep.deep_tags, "deep tags still cancelled at the measured latencies"
    assert elapsed < 11.0, f"reply overran the deadline: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# DEFECT 2 - the "fail loudly" guard fails SILENTLY in production and takes the
# whole replica with it. prompts.py:86 raises inside replica.replicate
# (replica.py:208), _gather_within swallows it (pipeline.py:616), so the miner
# loses predicted_gt, the centroid and the ranked answer on EVERY task.
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_llm(monkeypatch):
    async def chat(prompt, *a, **k):
        if "**Be exhaustive:**" in prompt:
            return ", ".join(f"deep tag number {i}" for i in range(30))
        if "<set0>" in prompt:
            return "mortgage rates, rental market, multifamily construction"
        return "mortgage rates, rental market, housing demand, construction starts"

    async def embed(texts, **k):
        return {t: [1.0, 0.0, 0.0] for t in texts}

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "embed", embed)


def test_a_revendored_template_must_not_destroy_the_whole_replica(normal_llm, monkeypatch):
    """Simulate upstream re-wording the cap line, then check the miner still answers.

    The guard is correct that it fires - but it fires *inside* replicate, and
    pipeline._gather_within turns any replica exception into None. The result is
    not a loud failure, it is a miner that silently stops replicating ground
    truth: source drops from "ranked" to "pool", which production measured at
    0.1994 against 0.5517.
    """
    monkeypatch.setitem(
        prompts._ENRICHMENT_LIMIT_LINE, False,
        "5.  **Limit to most important:** Return at most 11 of the most important and relevant tags.",
    )
    res = _run(pipeline.mine("conversation_tagging", window=WINDOW,
                             enrichment=ENRICHMENT, cfg=_cfg(use_deep_enrichment=True)))
    assert res.predicted_gt, "template drift silently destroyed the whole replica"
    assert res.source == "ranked", f"answer degraded to {res.source!r}"


def test_the_prompt_guard_fires_on_a_mutated_template(monkeypatch):
    """Control for the test above: the guard itself does detect drift."""
    real = prompts.gt_enrichment

    def drifted(content, coding=False):
        return real(content, coding=coding).replace("at most 10", "at most 11")

    monkeypatch.setattr(prompts, "gt_enrichment", drifted)
    with pytest.raises(ValueError, match="no longer contains the tag-cap line"):
        prompts.gt_enrichment_deep("line")


# ---------------------------------------------------------------------------
# DEFECT 3 - gt_enrichment_deep is the ONLY pool-feeding prompt with no
# English-only instruction, and its output is submittable. prompts.py:67,
# pipeline.py:533.
# ---------------------------------------------------------------------------

SPANISH = ["tasas hipotecarias", "mercado inmobiliario", "demanda de alquiler",
           "construccion multifamiliar", "prestamos bancarios"]


@pytest.fixture
def spanish_deep(monkeypatch):
    """Spanish source -> the upstream enrichment prompt answers in Spanish.

    The accents are deliberately omitted: sn33.tags.normalize only drops tags
    containing non-ASCII LETTERS, so unaccented Spanish sails straight through
    it and is deleted later by the validator's English screen instead.
    """

    async def chat(prompt, *a, **k):
        if "**Be exhaustive:**" in prompt:
            return ", ".join(SPANISH)
        if "<set0>" in prompt:
            return "mortgage rates, rental market, multifamily construction"
        return "mortgage rates, rental market, housing demand, construction starts"

    async def embed(texts, **k):
        # Spanish tags sit closest to the target - which is what actually
        # happens: they are extracted from the same enrichment lines the ground
        # truth came from, so they are on-topic, just in the wrong language.
        import math
        out = {}
        far = 0
        for t in texts:
            if t in SPANISH:
                th = 0.45 * SPANISH.index(t)
            else:
                th = 1.2 + 0.45 * far
                far += 1
            out[t] = [math.cos(th), math.sin(th), 0.0]
        return out

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "embed", embed)


def test_deep_tags_cannot_ship_non_english(spanish_deep):
    """A Portuguese conversation once scored a hard 0.0000 this way."""
    res = _run(pipeline.mine("conversation_tagging", window=WINDOW,
                             enrichment=ENRICHMENT, cfg=_cfg(use_deep_enrichment=True)))
    shipped = [t for t in res.tags if t in SPANISH]
    assert not shipped, f"non-English deep tags reached the submitted answer: {shipped}"


# ---------------------------------------------------------------------------
# DEFECT 4 - deep_tags are parsed with replica.clean_like_validator (comma-only,
# replica.py:51) instead of tags.parse_tag_list. That is right for predicted_gt,
# which must reproduce the validator's own strings - but deep_tags are POOL
# material and are submitted. A 30-tag request is far more likely to come back
# as a bullet or numbered list than a 10-tag one.
# ---------------------------------------------------------------------------

FUSED = "mortgage rates rental market housing demand"


@pytest.fixture
def deep_answers_as_a_bullet_list(monkeypatch):
    async def chat(prompt, *a, **k):
        if "**Be exhaustive:**" in prompt:
            return "- mortgage rates\n- rental market\n- housing demand"
        if "<set0>" in prompt:
            return "interest rates, rental market, multifamily construction"
        return "interest rates, housing demand, construction starts"

    async def embed(texts, **k):
        import math
        out, far = {}, 0
        for t in texts:
            th = 0.0 if t == FUSED else 1.2 + 0.45 * far
            if t != FUSED:
                far += 1
            out[t] = [math.cos(th), math.sin(th), 0.0]
        return out

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "embed", embed)


def test_a_bullet_list_deep_response_cannot_ship_a_fused_tag(deep_answers_as_a_bullet_list):
    """clean_like_validator only splits on commas, so a bullet list becomes ONE tag.

    "- a\\n- b\\n- c" -> "a b c", which is 43 chars, lowercase and alphanumeric,
    so sn33.tags.normalize passes it and the validator will happily embed it.
    parse_tag_list (what every other pool source uses) splits it correctly.
    """
    res = _run(pipeline.mine("conversation_tagging", window=WINDOW,
                             enrichment=ENRICHMENT, cfg=_cfg(use_deep_enrichment=True)))
    assert FUSED not in res.tags, f"fused junk tag shipped: {res.tags}"


def test_a_numbered_deep_response_is_not_silently_discarded(monkeypatch):
    """The other half of the same bug: a numbered list yields ZERO deep tags."""

    async def chat(prompt, *a, **k):
        if "**Be exhaustive:**" in prompt:
            return "\n".join(f"{i + 1}. deep tag number {i}" for i in range(30))
        return "mortgage rates, rental market, housing demand"

    monkeypatch.setattr(llm, "chat", chat)
    rep = _run(replica.replicate(
        "conversation_tagging",
        convo_xml="<conversation id='1'><p0>rates</p0></conversation>",
        enrichment=ENRICHMENT,
        timeout=5.0,
        deep_enrichment=30,
    ))
    from sn33.tags import normalize_all
    assert normalize_all(rep.deep_tags), (
        f"a numbered 30-tag response produced no usable deep tags: {rep.deep_tags!r}"
    )


# ---------------------------------------------------------------------------
# DEFECT 5 - deep tasks are created with ensure_future inside replicate but are
# owned by nothing. When pipeline._gather_within cancels the replica task
# (pipeline.py:611) the deep tasks are orphaned and keep running.
# ---------------------------------------------------------------------------

def test_cancelling_the_replica_cancels_its_deep_tasks(monkeypatch):
    async def chat(prompt, *a, **k):
        if "**Be exhaustive:**" in prompt:
            await asyncio.sleep(30)
        await asyncio.sleep(0.05)
        return "mortgage rates, rental market, housing demand"

    monkeypatch.setattr(llm, "chat", chat)

    async def body():
        task = asyncio.ensure_future(replica.replicate(
            "conversation_tagging",
            convo_xml="<conversation id='1'><p0>rates</p0></conversation>",
            enrichment=ENRICHMENT,
            timeout=5.0,
            deadline=11.0,
            deep_enrichment=30,
        ))
        await asyncio.sleep(0.15)      # past the fan-out, inside the grace wait
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)
        return [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]

    leaked = _run(body())
    assert not leaked, f"{len(leaked)} deep tasks outlived the cancelled replica: {leaked}"


# ---------------------------------------------------------------------------
# REVIEW FINDING #1 - a slow enrichment call used to cancel the whole fan-out
# and discard the doc tags that had already landed, dropping the answer from
# "ranked" to "pool". The fan-out now salvages every completed job.
# ---------------------------------------------------------------------------

def test_a_slow_enrichment_call_does_not_discard_the_completed_doc_tags(monkeypatch):
    async def chat(prompt, *a, **k):
        if "<conversation" in prompt or "<p0>" in prompt:   # the doc call
            return "mortgage rates, rental market, housing demand"
        await asyncio.sleep(30)                             # every enrichment call hangs
        return "should never arrive"

    monkeypatch.setattr(llm, "chat", chat)
    rep = _run(replica.replicate(
        "conversation_tagging",
        convo_xml="<conversation id='1'><p0>rates and the market</p0></conversation>",
        enrichment=ENRICHMENT,
        timeout=8.0,
        deadline=3.0,          # fan-out budget = 1.0s; the doc lands, enrichment does not
    ))
    assert rep.doc_tags, "completed doc tags were thrown away when enrichment was slow"
    assert not rep.degraded, "replica degraded despite having the doc answer"
