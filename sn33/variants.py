"""Lexical variants of a tag - the highest-value `unique` candidates available.

The 55% term, `top_3_mean`, is computed over tags that do NOT string-match
ground truth. So the ideal tag is one that is *semantically* as close to the
ground-truth centroid as a ground-truth tag, while being a *different string*.

Semantic paraphrases satisfy the second condition but pay for it in the first.
Measured against a real centroid built from eight ground-truth tags:

    ground-truth tags (score as `both`)  mean cosine 0.6453
    morphological variants (`unique`)    mean cosine 0.5895   top-3 0.6418
    semantic paraphrases  (`unique`)     mean cosine 0.5095   top-3 0.5773

`podcast` -> `podcasts` keeps the cosine identical to three decimals while
becoming a different string. That is +0.064 on a term worth 55% of the score.

These are generated blind and are not all good ("networking" -> "networks"
drifts, 0.692 -> 0.541). That is fine: every variant is embedded and ranked
against the estimated centroid like any other candidate, so weak ones simply
lose. This module's job is to make the option available, not to be right.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from sn33.tags import normalize

# Words that carry no topic weight; dropping them leaves the concept intact.
_LEADING_MODIFIERS = {
    "personal", "general", "basic", "modern", "current", "overall", "core",
    "key", "main", "important", "various", "different", "several", "specific",
}

_ES_ENDINGS = ("s", "x", "z", "ch", "sh")

# A phrase beginning with one of these is a fragment, not a tag.
_FRAGMENT_STARTERS = {
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "and", "or",
    "the", "a", "an", "as", "into", "over", "under", "about",
}


def _pluralize(word: str) -> str:
    # Gerunds and abstract nouns pluralize into non-words: "splitting" ->
    # "splittings", "awareness" -> "awarenesses". The validator screens tags
    # through validate_tags.j2, which discards "combined/compound words that
    # are not in the English Dictionary" - measured on a live skill task,
    # "commit splittings" was dropped while "preserve merges" survived. A tag
    # that gets deleted is a wasted slot, so refuse to build these.
    if word.endswith(("ing", "ness", "ity", "ism", "ance", "ence", "ship", "hood")):
        return word
    if word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(_ES_ENDINGS):
        return word + "es"
    return word + "s"


def _singularize(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3 and word[:-2].endswith(_ES_ENDINGS):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _head_is_safe_to_inflect(tag: str) -> bool:
    """True only when the last word is a common noun spaCy can inflect.

    Hand-rolled string rules cannot tell a proper noun from a common one, and
    production paid for that: "dallas texas" -> "dallas texases"/"dallas texa",
    "iran" -> "irans", "nfl" -> "nfls". Every one of those was deleted by the
    validator's English screen, wasting the slot.

    spaCy already runs in this process (warmed at startup) and does know: it
    tags proper nouns as PROPN and gives a real lemma, so "texas" stays "texas"
    instead of becoming "texa". If spaCy is unavailable we return False -
    generating no variant is strictly safer than generating a broken one.
    """
    from sn33 import extract

    nlp = extract._load_spacy()
    if nlp is None:
        return False
    try:
        doc = nlp(tag)
    except Exception:
        return False
    if not len(doc):
        return False
    head = doc[-1]
    # PROPN covers Dallas/Iran/Texas; X and NUM cover acronyms and figures that
    # spaCy cannot inflect meaningfully either.
    if head.pos_ in {"PROPN", "X", "NUM", "SYM"}:
        return False
    if head.pos_ not in {"NOUN"}:
        return False
    # NOT is_oov: en_core_web_sm ships without word vectors, so is_oov is True
    # for every token and using it rejects everything. The POS tag alone does
    # the job - measured: texas/iran/nfl -> PROPN, transcript/estate/rate ->
    # NOUN.
    if len(head.text) <= 2:
        return False
    return True


def _head_number(tag: str) -> str:
    """"Sing", "Plur" or "" - spaCy's reading of the last word's number.

    Needed because string rules cannot tell an existing plural from a singular
    ending in -s. "properties" -> "propertieses" was submitted in production:
    _pluralize saw a trailing "s" and appended "es".
    """
    from sn33 import extract

    nlp = extract._load_spacy()
    if nlp is None:
        return ""
    try:
        doc = nlp(tag)
    except Exception:
        return ""
    if not len(doc):
        return ""
    nums = doc[-1].morph.get("Number")
    return nums[0] if nums else ""


def variants_of(tag: str) -> List[str]:
    """Lexical variants of one tag. Never includes the tag itself."""
    words = tag.split()
    if not words:
        return []
    if not _head_is_safe_to_inflect(tag):
        return []
    number = _head_number(tag)

    out: List[str] = []

    def add(candidate: str) -> None:
        n = normalize(candidate)
        if n and n != tag and n not in out:
            out.append(n)

    # Number on the head noun: the cheapest change that preserves meaning.
    # Only inflect TOWARDS the form the word is not already in - pluralising a
    # plural produced "multifamily propertieses" in production.
    head = words[-1]
    if number != "Plur":
        add(" ".join(words[:-1] + [_pluralize(head)]))
    if number != "Sing":
        add(" ".join(words[:-1] + [_singularize(head)]))

    # Drop a semantically empty leading modifier ("personal growth" -> "growth").
    if len(words) > 1 and words[0] in _LEADING_MODIFIERS:
        add(" ".join(words[1:]))

    # For long phrases, the trailing head alone is usually the real topic -
    # but only when the shortened form still reads as a phrase. Truncating
    # "city of boston" to "of boston" produces a fragment, which the validator's
    # malformed-keyword screen may drop and which reads as noise either way.
    if len(words) >= 3 and words[-2] not in _FRAGMENT_STARTERS:
        add(" ".join(words[-2:]))

    return out


def expand(tags: Sequence[str], per_tag: int = 2, limit: int = 60) -> List[str]:
    """Variants for a list of tags, best-effort and bounded.

    Ordered by source-tag rank so that, if the caller truncates, the variants of
    the most central tags survive.
    """
    out: List[str] = []
    seen = set(tags)
    for tag in tags:
        for v in variants_of(tag)[:per_tag]:
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
            if len(out) >= limit:
                return out
    return out
