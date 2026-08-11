"""Tag hygiene and selection.

Everything in here exists because of one line in the validator:

    conversationgenome/llm/LlmLib.py:213
        return [element for element in valid_tags if element in tags]

``valid_tags`` are *normalized* (lowercased, punctuation stripped, truncated to
50 chars) but membership is tested against the miner's *raw* list. Any tag that
is not already in normal form is silently deleted before scoring. Enough
deletions and the response drops under ``min_tags`` and scores a hard zero.

So the miner must emit tags that are fixed points of ``Utils.get_safe_tag``.
``normalize`` produces exactly that, and ``survives_validation`` is the
assertion the tests use.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# Mirrors Utils.get_safe_tag (conversationgenome/utils/Utils.py:268-272).
_PASS1 = re.compile(r"\s{2,}|[^a-zA-Z0-9\s]")
_PASS2 = re.compile(r"[^\w\s]|(?<=\s)\s*")

MIN_LEN = 3      # Utils.get_clean_tag_set drops < 3
MAX_LEN = 50     # LlmLib.validate_tag_set truncates at 50 -> longer tags fail membership


def safe_tag(value: str, separator: str = " ") -> str:
    """Byte-for-byte reimplementation of Utils.get_safe_tag."""
    pass1 = _PASS1.sub(separator, value)
    return _PASS2.sub("", pass1).lower().strip()


def has_non_ascii_letters(tag: str) -> bool:
    """True if normalization would mangle this tag into fragments.

    ``get_safe_tag`` replaces anything outside [a-zA-Z0-9\\s] with a space, so
    accented characters do not survive - they blow the word apart:

        "construção"              -> "constru o"
        "regularização fundiária" -> "regulariza o fundi ria"
        "orçamento de obras"      -> "or amento de obras"

    Measured in production: a Portuguese conversation produced 12 such tags,
    every one was deleted by the validator's English screen, and the response
    scored a hard zero. Better to drop them here and let English candidates
    fill those slots.
    """
    return any(ord(c) > 127 and c.isalpha() for c in tag)


def normalize(tag: str) -> Optional[str]:
    """Return a tag guaranteed to survive the validator, or None if unusable."""
    if not tag or not isinstance(tag, str):
        return None
    if has_non_ascii_letters(tag):
        return None
    cleaned = safe_tag(tag)
    # get_safe_tag collapses runs of whitespace only via the 2+ rule; be strict.
    cleaned = " ".join(cleaned.split())
    if len(cleaned) < MIN_LEN or len(cleaned) > MAX_LEN:
        return None
    # Two or more orphaned single letters means an acronym was shredded by the
    # punctuation strip: "r&d strategy" -> "r d strategy", "at&t" -> "at t".
    # One is fine and often correct ("c programming", "vitamin c"), so the
    # threshold is deliberately 2 rather than 1 - a stricter rule would throw
    # away good tags like "front end development" and "covid 19 policy".
    if sum(1 for w in cleaned.split() if len(w) == 1) >= 2:
        return None
    # Fixed-point check: normalizing again must not change it, otherwise the
    # validator's `element in tags` membership test will drop it.
    if safe_tag(cleaned) != cleaned:
        return None
    # The validator parses its own screening reply with
    #     malformed_pos = content_str.find("malformed")
    #     good = content_str[0:malformed_pos]
    # so the FIRST occurrence of that word ends the good-keyword list. When the
    # word appears inside a tag we submitted, everything after it is discarded.
    # Measured against the real validate_tag_set: submitting
    #     ["alpha tag", "malformed data", "beta tag", "gamma tag"]
    # returned only ["alpha tag"] - three tags lost to one poisoned string, and
    # dropping under 3 survivors discards the whole response.
    if "malformed" in cleaned:
        return None
    return cleaned


def normalize_all(tags: Iterable[str]) -> List[str]:
    """Normalize, drop failures, dedupe, preserve order."""
    out: List[str] = []
    seen = set()
    for t in tags or []:
        n = normalize(t)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def survives_validation(tag: str) -> bool:
    """True iff `tag` is unchanged by the validator's clean+truncate+membership."""
    if not isinstance(tag, str) or not tag:
        return False
    cleaned = safe_tag(tag)
    if len(cleaned) < 3 or len(cleaned) > 64:
        return False          # get_clean_tag_set drops it
    truncated = cleaned[:50]  # validate_tag_set line 201
    return truncated == tag   # line 213 membership against the raw tag


# --------------------------------------------------------------------------
# Language guard for deep-enrichment candidates.
#
# The validator screens tags with an LLM that keeps only "good English
# keywords", and get_safe_tag shreds accented letters, so a Portuguese
# conversation once scored a hard 0.0000. `normalize` already rejects tags with
# non-ASCII LETTERS - but UNACCENTED Spanish ("prestamos bancarios") is pure
# ASCII and sails straight through it, only to be deleted by the validator.
#
# Deep-enrichment tags are the one pool source built from a prompt with no
# English instruction, so they are where this leaks. This is a filter for POOL
# material: it is safe to over-reject (a dropped candidate is just a candidate,
# never a shipped junk tag), so the design errs strict.
_ENGLISH_WORDS: Optional[frozenset] = None


def _english_words() -> frozenset:
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is None:
        import os
        path = os.path.join(os.path.dirname(__file__), "data", "english_words.txt")
        try:
            with open(path, encoding="utf-8") as f:
                _ENGLISH_WORDS = frozenset(w.strip() for w in f if w.strip())
        except OSError:
            _ENGLISH_WORDS = frozenset()   # degrade to overlap + short-token only
    return _ENGLISH_WORDS


def is_probably_english(tag: str, pool_vocab: Optional[Iterable[str]] = None) -> bool:
    """Heuristic English check for a deep-enrichment candidate.

    Two independent signals, either sufficient:

      1. **Vocabulary overlap.** If any word of the tag appears in
         ``pool_vocab`` - the words of our other, English candidates - the tag
         is English-consistent with what we already produce. This is the
         strongest signal and needs no word list: it keeps domain terms like
         "multifamily" that no general dictionary contains, and rejects Spanish
         that shares no vocabulary with an English pool.

      2. **Dictionary majority.** A word counts as English if it is in the
         bundled common-word set, is all digits (years, figures), or is <=4
         chars (acronyms: reit, nacs, api). More than half the words must
         qualify. The word set is the lowercase-only slice of the system
         dictionary, which deliberately excludes the capitalized loanwords
         (Mercado) that would otherwise pass Spanish tags.
    """
    if not tag:
        return False
    words = tag.split()
    if not words:
        return False
    if pool_vocab:
        vocab = pool_vocab if isinstance(pool_vocab, (set, frozenset)) else set(pool_vocab)
        if any(w in vocab for w in words):
            return True
    en = _english_words()
    # A standalone short token is an acronym (reit, nacs, api) - allow it. But
    # the short-token escape must NOT apply per-word inside a multi-word tag, or
    # short all-ASCII Spanish slips through: "casa roja" / "pago mora" are two
    # <=4 words each and would score 1.0. Inside a phrase a word only counts as
    # English if it is a real dictionary word or a number.
    if len(words) == 1:
        w = words[0]
        return w in en or w.isdigit() or len(w) <= 4
    hits = sum(1 for w in words if w in en or w.isdigit())
    return hits / len(words) > 0.5


def screen_safe(tag: str) -> bool:
    """True iff this tag is very likely KEPT by the validator's LLM screen.

    The validator runs an LLM that keeps only "good English keywords" and
    deletes "abbreviations, compound words not in the dictionary, and typos"
    (validate_tags.j2). On acronym-heavy tasks it deleted ALL of our 18 tags,
    dropping us below min_tags(3) -> a hard 0.0 (task cbc65e23, 2026-08-08).

    So this is deliberately STRICT: a tag is screen-safe only if EVERY word is a
    real dictionary word (or a number). "housing policy" -> safe. Acronyms
    ("nacs", "ccs"), non-dictionary compounds ("multifamily"), and typos are NOT
    safe. The composer guarantees a floor of these, so the screen can never
    delete us to zero. Strictness is the point: we want a floor of tags the LLM
    is certain to keep, not a lenient maybe.
    """
    if not tag:
        return False
    words = tag.split()
    if not words:
        return False
    en = _english_words()
    if not en:
        return False   # no word list -> cannot certify; treat as unsafe
    # Every word must be a dictionary word or a plain number, and a single bare
    # short token (an acronym) is never certified safe.
    if len(words) == 1 and words[0] not in en and not words[0].isdigit():
        return False
    return all(w in en or w.isdigit() for w in words)


def parse_tag_list(raw: Optional[str]) -> List[str]:
    """Parse an LLM comma-delimited response into normalized tags.

    Tolerates the usual model noise: numbered lists, bullets, code fences,
    trailing prose, newline-separated output.
    """
    if not raw:
        return []
    text = raw.strip()
    text = re.sub(r"```[a-zA-Z]*", " ", text)
    # Drop a leading "Tags:"-style preamble.
    text = re.sub(r"^\s*(here (are|is)[^:]*|tags|keywords|output)\s*:\s*", " ", text, flags=re.I)
    parts = re.split(r"[,\n;]+", text)
    out: List[str] = []
    for p in parts:
        p = re.sub(r"^\s*[\-\*•]\s*", "", p)     # bullets
        p = re.sub(r"^\s*\d+[\.\)]\s*", "", p)         # "1." / "1)"
        n = normalize(p)
        if n:
            out.append(n)
    # dedupe, keep order
    seen = set()
    return [t for t in out if not (t in seen or seen.add(t))]


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def centroid(vectors: Sequence[Sequence[float]]) -> Optional[np.ndarray]:
    """Mean vector, matching the validator's _calculate_semantic_neighborhood."""
    arr = [np.asarray(v, dtype=np.float32) for v in vectors if v is not None and len(v)]
    if not arr:
        return None
    return np.mean(arr, axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rank_by_centroid(
    candidates: Sequence[str],
    vectors: Dict[str, Sequence[float]],
    target: np.ndarray,
) -> List[tuple]:
    """[(tag, cosine)] sorted best first. Tags without a vector are dropped."""
    scored = []
    for tag in candidates:
        v = vectors.get(tag)
        if v is None or not len(v):
            continue
        scored.append((tag, cosine(target, np.asarray(v, dtype=np.float32))))
    scored.sort(key=lambda x: -x[1])
    return scored


def drop_near_duplicates(
    ranked: Sequence[tuple],
    vectors: Dict[str, Sequence[float]],
    threshold: float = 0.93,
    protected: Optional[set] = None,
) -> List[tuple]:
    """Remove tags that are semantic clones of a better-ranked tag.

    Clones cost more than they earn: they add nothing to the centroid match but
    they do occupy one of the ~21 scored slots and pull the mean toward whatever
    they score.
    """
    protected = protected or set()
    kept: List[tuple] = []
    kept_vecs: List[np.ndarray] = []
    for tag, score in ranked:
        v = vectors.get(tag)
        if v is None or not len(v):
            continue
        vec = np.asarray(v, dtype=np.float32)
        # Protected tags are exempt: a lexical variant is *meant* to be a near
        # duplicate of its source, since that is what makes it a high-cosine
        # `unique` tag.
        if tag not in protected and any(cosine(vec, kv) >= threshold for kv in kept_vecs):
            continue
        kept.append((tag, score))
        kept_vecs.append(vec)
    return kept
