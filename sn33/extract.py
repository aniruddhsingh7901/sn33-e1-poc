"""Local (no-API) tag candidate extraction.

Two jobs:

1. **Fallback.** Runs in milliseconds with no network, so the miner always holds
   a submittable answer before it makes a single API call. A response that
   scores 0.3 beats a timeout, which scores 0 and decays the EMA.
2. **Candidate widening.** Free extra candidates for the ranking stage.

Each extractor is optional and lazily imported; a missing library disables that
extractor instead of breaking the miner.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Sequence

from sn33.tags import normalize_all

_nlp = None
_kw_model = None

# Words that are common enough to be semantically empty as tags.
_STOP_TAGS = {
    "thing", "things", "lot", "lots", "way", "ways", "time", "times", "people",
    "guy", "guys", "kind", "sort", "bit", "day", "days", "year", "years",
    "question", "questions", "answer", "answers", "example", "examples",
    "everything", "something", "anything", "nothing", "someone", "anyone",
    "today", "tomorrow", "yesterday", "week", "month", "point", "case",
}


def warm(spacy_model: str = "en_core_web_sm") -> bool:
    """Load models ahead of the first request. Call once at miner startup."""
    return _load_spacy(spacy_model) is not None


def _load_spacy(model: str = "en_core_web_sm"):
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy

        # The lemmatizer is needed by sn33.variants to decide whether a word is
        # a real plural ("transcripts" -> "transcript") rather than a proper
        # noun that string rules would mangle ("texas" -> "texa").
        _nlp = spacy.load(model)
        return _nlp
    except Exception:
        return None


def _clean_phrase(text: str) -> Optional[str]:
    """Trim a raw phrase down to something that reads like a tag."""
    text = re.sub(r"\b(a|an|the|this|that|these|those|my|your|our|their|his|her|its)\b", " ", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    words = [w for w in text.split() if w]
    if not words or len(words) > 4:
        return None
    if all(w.lower() in _STOP_TAGS for w in words):
        return None
    if len(words) == 1 and words[0].lower() in _STOP_TAGS:
        return None
    return " ".join(words).lower()


def spacy_candidates(text: str, limit: int = 40, model: str = "en_core_web_sm") -> List[str]:
    """Noun chunks + named entities, ranked by frequency.

    Noun chunks are the closest cheap approximation of "a topic tag a human
    would write"; entities cover the named-entity task's target directly.
    """
    nlp = _load_spacy(model)
    if nlp is None:
        return []
    doc = nlp(text[:20000])
    counts: Dict[str, int] = {}

    def add(phrase: str, weight: int = 1) -> None:
        cleaned = _clean_phrase(phrase)
        if cleaned and 3 <= len(cleaned) <= 50:
            counts[cleaned] = counts.get(cleaned, 0) + weight

    for ent in doc.ents:
        if ent.label_ in {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY", "TIME", "DATE"}:
            continue
        add(ent.text, weight=3)  # entities are disproportionately likely to be ground truth
    for chunk in doc.noun_chunks:
        add(chunk.text)

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return normalize_all([t for t, _ in ranked[:limit]])


def yake_candidates(text: str, limit: int = 40) -> List[str]:
    try:
        import yake
    except Exception:
        return []
    try:
        kw = yake.KeywordExtractor(lan="en", n=3, top=limit * 2)
        return normalize_all([_clean_phrase(k) or "" for k, _ in kw.extract_keywords(text[:20000])])[:limit]
    except Exception:
        return []


def rake_candidates(text: str, limit: int = 40) -> List[str]:
    try:
        from rake_nltk import Rake
    except Exception:
        return []
    try:
        r = Rake(max_length=4)
        r.extract_keywords_from_text(text[:20000])
        return normalize_all([_clean_phrase(p) or "" for p in r.get_ranked_phrases()[: limit * 2]])[:limit]
    except Exception:
        return []


def keybert_candidates(text: str, limit: int = 40) -> List[str]:
    global _kw_model
    try:
        from keybert import KeyBERT
    except Exception:
        return []
    try:
        if _kw_model is None:
            _kw_model = KeyBERT()
        pairs = _kw_model.extract_keywords(text[:20000], keyphrase_ngram_range=(1, 3), top_n=limit * 2)
        return normalize_all([_clean_phrase(k) or "" for k, _ in pairs])[:limit]
    except Exception:
        return []


EXTRACTORS = {
    "spacy": spacy_candidates,
    "yake": yake_candidates,
    "rake": rake_candidates,
    "keybert": keybert_candidates,
}


def candidates(text: str, backends: Sequence[str] = ("spacy",), limit: int = 40) -> List[str]:
    out: List[str] = []
    for name in backends:
        fn = EXTRACTORS.get(name)
        if fn is None:
            continue
        for tag in fn(text, limit=limit):
            if tag not in out:
                out.append(tag)
    return out


def timed_candidates(text: str, backend: str, limit: int = 40) -> tuple:
    """(tags, seconds) - used by the Phase 1 library benchmark."""
    fn = EXTRACTORS.get(backend)
    if fn is None:
        return [], 0.0
    t0 = time.perf_counter()
    tags = fn(text, limit=limit)
    return tags, time.perf_counter() - t0
