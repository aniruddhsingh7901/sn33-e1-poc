"""A prior over the words the validator's ground truth actually uses.

`no_both_tags` costs a flat 10% unless at least one submitted tag string-matches
a ground-truth tag exactly. Measured on conversation tasks, that penalty fires
on roughly two thirds of responses - for the stock miner, for this repo's tuned
prompts, and for the replica alike. Matching an exact string is hard when the
ground truth is written from the whole conversation and we only see ten lines
of it.

This module supplies candidates with a high prior probability of appearing in
ground truth, mined from ReadyAI's own published corpus
(``conversations_to_tags.parquet``: 4,888 conversations, 4.9M tag rows). Tags
like ``podcast``, ``communication`` and ``technology`` occur in over 90% of
those conversations - they are the vocabulary this domain's taggers reach for.

They are only ever *offered* to the ranker, never forced: an anchor is
submitted solely when its cosine to the estimated target clears a threshold, so
a document that is not a podcast never gets tagged "podcast". Generic tags sit
further from a specific centroid than precise ones, so an unfiltered anchor
would trade 10% of penalty for more than 10% of similarity.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from sn33.tags import normalize_all

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gt_vocab.json")
_cache: Optional[List[str]] = None


def load(path: str = _DEFAULT_PATH, limit: int = 150) -> List[str]:
    """Top vocabulary tags by document frequency, already normalized."""
    global _cache
    if _cache is not None:
        return _cache[:limit]
    try:
        with open(path) as f:
            payload = json.load(f)
        _cache = normalize_all([row["tag"] for row in payload["tags"]])
    except Exception:
        _cache = []
    return _cache[:limit]


def anchors_for(kind: str, limit: int = 60) -> List[str]:
    """Anchor candidates for a task type.

    The corpus is podcast conversations, so the prior transfers to conversation
    and named-entity tasks (both are built from transcripts). Webpage and skill
    documents come from a different distribution and get no anchors - their
    ground truth is reproducible from the text we already hold, so they do not
    need the help.
    """
    if kind not in ("conversation_tagging", "named_entities_extraction"):
        return []
    return load(limit=limit)
