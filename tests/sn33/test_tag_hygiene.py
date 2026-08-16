"""Tags we emit must survive the validator untouched.

``LlmLib.validate_tag_set`` ends with

    return [element for element in valid_tags if element in tags]

where ``valid_tags`` have been normalized and truncated but ``tags`` is our raw
list. Any tag that is not already a fixed point of that normalization is
deleted before scoring, silently. Measured on this repo's own production logs,
that deleted 38.5% of submitted survey tags and left 5 of 53 survey responses
with nothing to score at all.
"""

from __future__ import annotations

import json
import os

import pytest

from conversationgenome.utils.Utils import Utils
from sn33.tags import MAX_LEN, normalize, normalize_all, parse_tag_list, safe_tag, survives_validation

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validator_round_trip(tags):
    """What validate_tag_set keeps, minus the LLM screen (which only removes more).

    Returns a set: ``get_clean_tag_set`` builds a Python set internally, so the
    surviving order is arbitrary and only membership is meaningful.
    """
    clean = Utils.get_clean_tag_set(list(tags))
    clean = [t[:50] for t in clean]
    return {t for t in clean if t in tags}


def test_safe_tag_matches_validator():
    samples = [
        "Real Estate", "mortgage-backed securities", "atención al cliente",
        "u.s. congress", "COVID-19", "machine   learning", " spaced ",
        "emoji 🎉 tag", "under_score", "quote's", "a" * 80,
    ]
    for s in samples:
        assert safe_tag(s) == Utils.get_safe_tag(s), s


@pytest.mark.parametrize(
    "raw",
    [
        "Real Estate", "mortgage-backed securities", "atención al cliente",
        "COVID-19", "u.s. congress", "transit-oriented development",
        "machine   learning", "Ai", "x", "a" * 200, "", "   ",
        "emoji 🎉 tag", "50/50 split", "back-end", "años",
    ],
)
def test_normalized_tags_always_survive(raw):
    """Whatever we do to a tag, the result either survives or is dropped by us."""
    n = normalize(raw)
    if n is None:
        return  # we refused to emit it - that is the correct outcome
    assert survives_validation(n), f"{raw!r} -> {n!r} would be deleted by the validator"
    assert validator_round_trip([n]) == {n}, f"{n!r} did not round-trip"
    assert len(n) <= MAX_LEN


def test_accented_tags_are_transliterated_or_dropped():
    """The exact failure that cost this miner 38.5% of its survey tags."""
    bad = ["atención al cliente", "buena información", "muchos años con el banco"]
    # Submitted raw, the validator deletes every one of them.
    assert validator_round_trip(bad) == set()
    # After normalization, whatever we keep survives.
    kept = normalize_all(bad)
    assert validator_round_trip(kept) == set(kept)


def test_full_answer_survives_round_trip():
    answer = normalize_all(
        [
            "Real Estate", "mortgage-backed securities", "property data",
            "atención al cliente", "market trends", "market trends",
            "transit-oriented development", "u.s. housing policy",
        ]
    )
    assert len(answer) == len(set(answer)), "duplicates would be collapsed by the validator"
    assert validator_round_trip(answer) == set(answer)


def test_parse_tag_list_handles_model_noise():
    assert parse_tag_list("real estate, banking, finance") == ["real estate", "banking", "finance"]
    assert parse_tag_list("Tags: Real Estate, Banking") == ["real estate", "banking"]
    assert parse_tag_list("1. real estate\n2. banking\n") == ["real estate", "banking"]
    assert parse_tag_list("- real estate\n- banking") == ["real estate", "banking"]
    assert parse_tag_list("```\nreal estate, banking\n```") == ["real estate", "banking"]
    assert parse_tag_list("") == []
    assert parse_tag_list(None) == []


def test_production_logs_regression():
    """Replay real submitted tags: the normalizer must fix the observed losses."""
    path = os.path.join(ROOT, "data", "all_uids_consolidated.jsonl")
    if not os.path.exists(path):
        pytest.skip("production capture not present")

    before_lost = after_lost = total = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            tags = (rec.get("result") or {}).get("tags") or []
            if not tags:
                continue
            total += len(tags)
            before_lost += len(set(tags)) - len(validator_round_trip(tags))
            fixed = normalize_all(tags)
            after_lost += len(set(fixed)) - len(validator_round_trip(fixed))

    assert total > 0
    assert after_lost == 0, f"{after_lost} normalized tags still deleted by the validator"
    assert before_lost > 0, "expected the historical capture to show real losses"


def test_variants_never_produce_fragments():
    """A variant must read as a tag, not as a sentence fragment.

    Truncating "city of boston" to "of boston" was observed in a live smoke
    test on the server: it survives normalization but reads as noise and risks
    the validator's malformed-keyword screen.
    """
    from sn33.variants import variants_of, _FRAGMENT_STARTERS

    for tag in ["city of boston", "department of transportation", "board of directors",
                "secretary of state", "university of texas"]:
        for v in variants_of(tag):
            assert v.split()[0] not in _FRAGMENT_STARTERS, f"{tag!r} -> fragment {v!r}"


def test_variants_are_valid_tags_and_differ_from_source():
    from sn33.variants import variants_of
    from sn33.tags import survives_validation

    for tag in ["podcast", "real estate", "personal growth", "machine learning",
                "boston redevelopment authority", "communication"]:
        for v in variants_of(tag):
            assert v != tag
            assert survives_validation(v), f"{v!r} would be deleted by the validator"


def test_variants_avoid_gerund_and_abstract_plurals():
    """Do not build plurals the validator's English screen will delete.

    Measured live: submitting "commit splittings" (from "commit splitting") and
    "commit history cleanups" cost 2 of 12 tag slots on a real skill task,
    because validate_tags.j2 discards words that are not in the dictionary.
    """
    from sn33.variants import variants_of

    for tag in ["commit splitting", "machine learning", "brand awareness",
                "software quality", "product ownership", "data availability"]:
        for v in variants_of(tag):
            assert not v.endswith(("ings", "nesses", "ities", "ships", "hoods")), (
                f"{tag!r} -> implausible plural {v!r}"
            )

    # Ordinary nouns must still pluralize - the feature has to keep working.
    assert "preserve merges" in variants_of("preserve merge")


def test_accented_tags_are_rejected_not_mangled():
    """Regression: a Portuguese conversation scored a hard zero in production.

    get_safe_tag replaces every non-ASCII character with a space, so accented
    words are torn into fragments:
        "construção"              -> "constru o"
        "regularização fundiária" -> "regulariza o fundi ria"
    All 12 tags were destroyed and the response scored 0.0000. Reject them at
    source so English candidates take the slots instead.
    """
    from sn33.tags import normalize, normalize_all, has_non_ascii_letters

    mangled_in_production = [
        "regularização fundiária", "financiamento imobiliário", "construção",
        "orçamento de obras", "construção civil", "atención al cliente",
    ]
    for tag in mangled_in_production:
        assert has_non_ascii_letters(tag)
        assert normalize(tag) is None, f"{tag!r} would be mangled into fragments"

    # plain ASCII tags must be unaffected
    for tag in ["real estate", "mortgage rates", "construction costs", "podcast"]:
        assert normalize(tag) == tag

    mixed = ["construção", "construction", "orçamento", "budget planning"]
    assert normalize_all(mixed) == ["construction", "budget planning"]


def test_pool_prompts_demand_english():
    from sn33 import prompts

    for tpl in (prompts.MINER_CONVERSATION_POOL, prompts.MINER_DOCUMENT_POOL,
                prompts.MINER_SURVEY_POOL):
        assert "ENGLISH" in tpl


def test_variants_never_double_pluralise():
    """Regression: "multifamily properties" -> "multifamily propertieses".

    _pluralize saw a trailing "s" and appended "es". Submitted in production on
    the testnet harness. spaCy's morphology knows the word is already plural, so
    we only ever inflect towards the form the word is not already in.
    """
    from sn33.variants import variants_of, _head_number

    for plural in ["multifamily properties", "zoning reforms", "building codes",
                   "federal housing policies", "market risks"]:
        assert _head_number(plural) == "Plur", f"{plural!r} not detected as plural"
        for v in variants_of(plural):
            assert not v.endswith("eses"), f"{plural!r} -> {v!r}"
            assert not v.endswith("ss"), f"{plural!r} -> {v!r}"

    # singulars still gain their plural
    assert "market risks" in variants_of("market risk")
    assert "multifamily properties" in variants_of("multifamily property")


def test_compose_never_ships_below_min_tags():
    """Below min_tags the validator DISCARDS the response (evaluator.py:157).

    That is a total loss, not a penalty multiplier - strictly worse than any
    penalised answer. Measured 2.1 tags/case at target_tags=6 before the floor
    guard, with a median score of exactly 0.0000.
    """
    from sn33.pipeline import TASK_PROFILE, compose

    ranked = [(f"tag number {i}", 0.9 - i * 0.01) for i in range(30)]
    profile = TASK_PROFILE["conversation_tagging"]
    for target in (3, 4, 5, 6, 8, 12):
        for insurance in (0, 1, 3, 6):
            out = compose(ranked, ["tag number 0"], profile,
                          target_tags=target, insurance=insurance)
            assert len(out) >= profile["min_tags"], (
                f"target={target} insurance={insurance} -> {len(out)} tags")
            assert len(out) == len(set(out)), "duplicate tags"


def test_tag_containing_malformed_is_rejected():
    """The validator cuts its screening reply at the first "malformed".

    LlmLib.validate_tag_set does
        malformed_pos = content_str.find("malformed")
        good = content_str[0:malformed_pos]
    so a tag of ours containing that word truncates the good-keyword list.
    Driving the real validate_tag_set with a stubbed reply:
        submitted ["alpha tag","malformed data","beta tag","gamma tag"]
        survived  ["alpha tag"]
    Three tags lost, and under 3 survivors the response is discarded entirely.
    """
    from sn33.tags import normalize, normalize_all

    assert normalize("malformed data") is None
    assert normalize("a malformed tag") is None
    # the ordinary word is untouched
    assert normalize("housing policy") == "housing policy"
    assert "malformed data" not in normalize_all(["housing policy", "malformed data"])


def test_is_probably_english_rejects_unaccented_spanish():
    """Deep-enrichment guard. Unaccented Spanish is pure ASCII, so it passes
    `normalize`, but the validator's English screen deletes it - a Portuguese
    conversation once scored a hard 0.0000. The bundled word set is the
    lowercase-only dictionary slice, which excludes loanwords like 'mercado'.
    """
    from sn33.tags import is_probably_english

    english_pool = {"mortgage", "rates", "rental", "market", "housing",
                    "demand", "construction", "starts"}
    for spanish in ["tasas hipotecarias", "mercado inmobiliario",
                    "demanda de alquiler", "construccion multifamiliar",
                    "prestamos bancarios",
                    # short all-ASCII Spanish - the <=4 escape must not save these
                    "casa roja", "pago mora", "ruta sur"]:
        assert not is_probably_english(spanish, english_pool), spanish

    # real English tags survive, via dictionary OR pool overlap
    for eng in ["mortgage rates", "rental market", "housing demand",
                "interest rates", "cap rates"]:
        assert is_probably_english(eng, english_pool), eng
    # a domain term absent from any general dictionary is saved by pool overlap
    assert is_probably_english("multifamily construction", english_pool)
    # short acronyms and years pass on their own
    assert is_probably_english("reit", set())
    assert is_probably_english("2026 outlook", set())


def test_uniques_first_seeds_the_top_three_slots():
    """The flag must put expected-unique tags in the first three slots."""
    from sn33.pipeline import TASK_PROFILE, compose

    # ranked best-first; even indices are predicted ground truth (expected both)
    ranked = [(f"tag number {i}", 0.9 - i * 0.01) for i in range(30)]
    gt = [f"tag number {i}" for i in range(0, 30, 2)]     # the evens are `both`
    profile = TASK_PROFILE["conversation_tagging"]

    out = compose(ranked, gt, profile, target_tags=12, insurance=6, uniques_first=True)
    gtset = set(gt)
    first_three_unique = sum(1 for t in out[:3] if t not in gtset)
    assert first_three_unique == 3, f"top 3 not all unique: {out[:3]}"


def test_deep_enrichment_off_is_the_default():
    """Production must be unaffected until the A/B confirms the gain."""
    from sn33.pipeline import Config

    assert Config().use_deep_enrichment is False
    assert Config().uniques_first is False
    assert Config().target_tags is None      # falls back to the profile default (18 for conversation)


def test_screen_safe_classifies_dictionary_vs_acronyms():
    """The screen-safe check must certify plain dictionary phrases and reject
    the acronyms/compounds the validator's LLM screen deletes."""
    from sn33.tags import screen_safe
    # Certified safe: every word is a plain dictionary word (or a number).
    for safe in ["housing policy", "electric vehicles", "rent control",
                 "interest rates", "2026 outlook", "supply and demand"]:
        assert screen_safe(safe), safe
    # NOT certified: acronyms, and any tag with a non-dictionary word. The check
    # is deliberately conservative - "customer onboarding" (onboarding is a
    # modern word absent from the system dictionary) simply won't COUNT toward
    # the floor; it is not rejected from the answer.
    for unsafe in ["nacs", "ccs", "reit", "saas", "propertieses",
                   "multifamily properties", "customer onboarding"]:
        assert not screen_safe(unsafe), unsafe


def test_compose_guarantees_screen_safe_floor_on_acronym_heavy_pool():
    """The zero we hit: an acronym-heavy pool where the LLM screen deleted every
    tag. compose must inject >= SCREEN_SAFE_FLOOR dictionary-word tags so the
    screen can never delete us below min_tags.
    """
    from sn33.pipeline import TASK_PROFILE, SCREEN_SAFE_FLOOR, compose

    # a pool dominated by acronyms, with some dictionary phrases lower down
    acro = [(f"acro{i}", 0.9 - i*0.01) for i in range(15)]   # non-dictionary tokens
    safe = [("charging standard", 0.60), ("electric vehicles", 0.59),
            ("charging network", 0.58), ("power delivery", 0.57),
            ("vehicle charging", 0.56), ("charging speed", 0.55),
            ("connector design", 0.54), ("home charging", 0.53)]
    ranked = acro + safe
    profile = TASK_PROFILE["conversation_tagging"]
    out = compose(ranked, ["acro0"], profile, target_tags=20, insurance=6)
    from sn33.tags import screen_safe
    n_safe = sum(1 for t in out if screen_safe(t))
    assert n_safe >= SCREEN_SAFE_FLOOR, f"only {n_safe} screen-safe tags: {out}"


def test_ner_is_exempt_from_screen_safe_floor():
    """NER uses validate_named_entities_tag_set (no LLM screen), so the floor
    must not force dictionary tags there and displace strong named entities."""
    from sn33.pipeline import TASK_PROFILE, compose
    ranked = [(f"acro{i}", 0.9 - i*0.01) for i in range(12)]
    profile = TASK_PROFILE["named_entities_extraction"]
    out = compose(ranked, ["acro0"], profile, target_tags=10, insurance=0)
    # floor did not fire (penalties False) -> acronym entities survive
    assert any(t.startswith("acro") for t in out)
