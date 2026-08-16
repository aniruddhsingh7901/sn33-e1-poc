"""Every non-English shape the validator can send us.

The validator has NO language handling anywhere (verified by grep across
task_bundle/, LlmLib, scoring_mechanism/, Utils). It strips everything outside
[a-zA-Z0-9\\s] via get_safe_tag, then keeps only "good English keywords". So:

    construção              -> 'constru o'      (shredded)
    regularização fundiária -> 'regulariza o fundi ria'
    日本語のタグ              -> ''               (erased entirely)

Measured against the real screen with gpt-5.2:
    mangled portuguese      0 of 6 survived
    accent-free portuguese  0 of 6 survived   <- correctly spelled, still rejected
    english                 6 of 6 survived

A Portuguese conversation cost us a hard 0.0000 in production. These tests pin
down every variant so it cannot recur through a different alphabet.
"""

from __future__ import annotations

import asyncio

import pytest

from sn33 import llm, pipeline
from sn33.tags import has_non_ascii_letters, normalize, normalize_all, survives_validation

# One sample per writing system the corpus could plausibly contain.
NON_ASCII_TAGS = {
    "portuguese": ["construção", "regularização fundiária", "orçamento de obras"],
    "spanish": ["atención al cliente", "información", "años de servicio"],
    "french": ["développement durable", "sécurité", "à la carte"],
    "german": ["straße", "über uns", "grün"],
    "japanese": ["日本語のタグ", "東京", "ソフトウェア"],
    "chinese": ["软件开发", "北京"],
    "korean": ["소프트웨어", "서울"],
    "russian": ["программирование", "москва"],
    "arabic": ["برمجة", "القاهرة"],
    "hebrew": ["תוכנה"],
    "greek": ["λογισμικό"],
    "accented_caps": ["Ação Social", "ÉDUCATION"],
}


@pytest.mark.parametrize("lang", sorted(NON_ASCII_TAGS))
def test_non_ascii_tags_are_never_submitted(lang):
    """Anything the validator would mangle or erase must be dropped here."""
    for tag in NON_ASCII_TAGS[lang]:
        assert has_non_ascii_letters(tag), f"{lang}: {tag!r} not detected"
        assert normalize(tag) is None, f"{lang}: {tag!r} leaked through normalize"
    assert normalize_all(NON_ASCII_TAGS[lang]) == []


def test_ascii_tags_still_pass_untouched():
    """The guard must not damage ordinary English tags."""
    good = ["real estate", "mortgage rates", "construction budget", "podcast",
            "web3", "gpt 4", "covid 19", "b2b sales", "3d printing"]
    assert normalize_all(good) == good
    for t in good:
        assert survives_validation(t)


def test_mixed_language_keeps_only_the_ascii_half():
    mixed = ["construção", "construction", "orçamento", "budget planning",
             "日本語", "japanese language", "atención", "customer service"]
    assert normalize_all(mixed) == ["construction", "budget planning",
                                    "japanese language", "customer service"]


def test_every_surviving_tag_is_a_fixed_point_of_the_validator():
    """Whatever we emit must be unchanged by clean -> truncate -> membership."""
    messy = ["  Real   Estate  ", "MORTGAGE-RATES", "co-operative", "e.g. testing",
             "quotes\"here", "under_score", "trailing.", "(parens)", "a", "ab",
             "x" * 80, "construção", "🌍", "", "   ", "123", "tab\tsep"]
    for tag in normalize_all(messy):
        assert survives_validation(tag), f"{tag!r} would be altered by the validator"
        assert 3 <= len(tag) <= 50
        assert tag == tag.lower()


# --------------------------------------------------------------------------
# End-to-end: the pipeline must not emit non-English even when every LLM
# response comes back in another language.
# --------------------------------------------------------------------------

PT_WINDOW = [
    (0, "Precisamos discutir a regularização fundiária do loteamento."),
    (1, "O financiamento imobiliário depende do orçamento de obras."),
    (0, "O custo de construção civil subiu muito neste trimestre."),
]


def _cfg(**kw):
    base = dict(use_local=False, use_pool=True, use_cache=False,
                deadline_s=9.0, call_timeout_s=7.0)
    base.update(kw)
    return pipeline.Config(**base)


@pytest.fixture
def portuguese_llm(monkeypatch):
    """Every model response comes back in Portuguese, as it would in reality."""

    async def chat(prompt, model, **kw):
        return ("regularização fundiária, financiamento imobiliário, construção civil, "
                "orçamento de obras, loteamento, custo de construção")

    async def embed(texts, **kw):
        return {t: [1.0, 0.1, 0.0] for t in texts}

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "embed", embed)


def test_portuguese_everywhere_never_emits_mangled_tags(portuguese_llm):
    """Worst case: model ignores the English instruction on every call.

    We would rather submit nothing than submit tags the validator will shred -
    but nothing must also not crash, and must not be a malformed payload.
    """
    res = asyncio.run(pipeline.mine("conversation_tagging", window=PT_WINDOW, cfg=_cfg()))
    for tag in res.tags:
        assert not has_non_ascii_letters(tag)
        assert survives_validation(tag)
    assert len(res.tags) == len(set(res.tags))
    assert len(res.tags) <= pipeline.MAX_TAGS


@pytest.fixture
def english_llm(monkeypatch):
    """The realistic case: the prompt works and the model answers in English."""

    async def chat(prompt, model, **kw):
        return ("land regularization, real estate financing, civil construction, "
                "construction budget, land subdivision, construction costs, "
                "property development, urban planning")

    async def embed(texts, **kw):
        return {t: [1.0, 0.05 * (i % 4), 0.0] for i, t in enumerate(texts)}

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "embed", embed)


def test_portuguese_input_english_output_is_submittable(english_llm):
    """Portuguese source text, English tags -> a valid, scoreable answer."""
    res = asyncio.run(pipeline.mine("conversation_tagging", window=PT_WINDOW, cfg=_cfg()))
    assert len(res.tags) >= pipeline.TASK_PROFILE["conversation_tagging"]["min_tags"]
    for tag in res.tags:
        assert survives_validation(tag)
        assert not has_non_ascii_letters(tag)


@pytest.mark.parametrize("kind", sorted(pipeline.TASK_PROFILE))
def test_non_english_input_never_raises_for_any_task_type(kind, portuguese_llm):
    window = PT_WINDOW if kind == "conversation_tagging" else [(0, "Regularização fundiária e construção civil.")]
    res = asyncio.run(pipeline.mine(
        kind, window=window,
        question="¿Por qué razones prefiere ese banco?",
        comment="informa bien dan buena información",
        cfg=_cfg(deadline_s=6.0),
    ))
    assert isinstance(res.tags, list)
    for tag in res.tags:
        assert survives_validation(tag)
        assert not has_non_ascii_letters(tag)


def test_emoji_and_symbols_are_stripped_cleanly_not_rejected():
    """Non-ASCII SYMBOLS are harmless; non-ASCII LETTERS are not.

    get_safe_tag replaces every out-of-range character with a space. For an
    emoji that just removes a standalone token and leaves the words intact; for
    an accented letter it tears the word in half:

        "climate 🌍 change" -> "climate change"   (still a good tag - keep it)
        "construção"        -> "constru o"        (fragment - drop it)

    So the guard keys on isalpha(), not on "is non-ASCII". This test pins that
    distinction down, because widening the guard would silently throw away
    perfectly good tags.
    """
    assert normalize("climate 🌍 change") == "climate change"
    assert normalize("🚀 startup") == "startup"
    assert normalize("machine learning ✅") == "machine learning"
    # ...while letters are still rejected
    assert normalize("construção") is None
    assert normalize("Ação Social") is None
    assert normalize("straße") is None


def test_ascii_punctuation_that_shreds_acronyms_is_rejected():
    """ASCII punctuation tears words apart too, not just accents.

        "r&d strategy" -> "r d strategy"     (acronym destroyed - drop)
        "at&t wireless" -> "at t wireless"   (destroyed - drop)
        "front-end development" -> "front end development"  (fine - keep)
        "c++ programming" -> "c programming"                (fine - keep)

    The rule is >=2 orphaned single letters. One is common and usually correct,
    so a stricter rule would discard good tags.
    """
    assert normalize("r&d strategy") is None
    assert normalize("a b c") is None

    # KNOWN LIMITATION, accepted deliberately: a single orphaned letter is not
    # rejected, so "at&t wireless" -> "at t wireless" still gets through.
    # Tightening the rule to >=1 orphan would also discard "c programming",
    # "e commerce" and "vitamin c", which are correct. The cost of the leak is
    # one wasted tag slot in a rare case; the cost of over-filtering is losing
    # good tags on every task. Documented rather than hidden.
    assert normalize("at&t wireless") == "at t wireless"

    for tag, expected in [
        ("front-end development", "front end development"),
        ("covid-19 policy", "covid 19 policy"),
        ("c++ programming", "c programming"),
        ("50% growth", "50 growth"),
        ("e-commerce", "e commerce"),
    ]:
        assert normalize(tag) == expected, f"{tag!r} should clean to {expected!r}"
        assert survives_validation(expected)
