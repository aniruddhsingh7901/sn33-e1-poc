#!/usr/bin/env python3
"""The faithful harness: real ground truth, from the full conversation.

Every earlier conversation benchmark in this repo generated "ground truth" from
the 10-line window the miner receives. The validator generates it from the FULL
conversation (up to 300 lines). Those are different targets, and scoring against
the wrong one produced a recommendation that measurably hurt production.

This module removes that error. Using the testnet API we hold the complete
bundle, so we reproduce the validator's own path step for step:

    ConversationTaggingTaskBundle.setup()
      trim_input()                 lines[:MAX_CONVO_LINES]        (300)
      _split_conversation_in_windows()   size 10, overlap 2
      _generate_metadata()
          conversation_to_metadata(FULL conversation)   <- the part we could never do
          for each search query: pick 1-3 results at random,
              enrichment_to_metadata(title + newline + snippet)
          combine_metadata_tags(all sets)
          get_vector_embeddings_set(tags)   re-cleans, then embeds

    to_mining_tasks()  ->  mask_task_for_miner()  ->  the miner sees one window

The miner then receives exactly what production sends: one window plus the same
enrichment lines that fed the ground truth. Scoring uses the validator's own
GroundTruthTagSimilarityScoringMechanism.

Randomness note: the validator's enrichment selection is random
(`random.randint(1, min(3, len(results)))` then `random.sample`). We seed it per
conversation so a run is reproducible, and the miner always receives the same
lines the ground truth was built from - which is what production does.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from bench.harness import GroundTruth, Verdict, score_answer
from conversationgenome.utils.Utils import Utils
from sn33 import llm, prompts, scoring
from sn33.replica import clean_like_validator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_CONVO_LINES = 300      # env MAX_CONVO_LINES default
WINDOW_SIZE = 10           # convo_window max_lines
WINDOW_OVERLAP = 2         # convo_window overlap_lines
GT_MODEL = "gpt-5.2"       # llm_openai.py:14 committed default


# --------------------------------------------------------------------------
# Bundle -> the pieces the validator works with
# --------------------------------------------------------------------------

@dataclass
class FaithfulCase:
    guid: str
    kind: str
    full_lines: List[tuple]              # the whole conversation, trimmed like the validator
    enrichment_lines: List[str]          # exactly what BOTH gt and miner see
    windows: List[List[tuple]] = field(default_factory=list)
    document: str = ""                   # webpage / NER / skill
    question: str = ""
    comment: str = ""
    reference_tags: List[str] = field(default_factory=list)   # survey: the real selected_choices


def split_overlap(lines: Sequence, size: int = WINDOW_SIZE, overlap: int = WINDOW_OVERLAP) -> List[list]:
    """Mirror Utils.split_overlap_array."""
    out, i = [], 0
    while i < len(lines):
        out.append(list(lines[i : i + size]))
        i += size - overlap
    return out


def build_enrichment_lines(bundle: dict, seed: int) -> List[str]:
    """Reproduce _build_enrichment_lines_and_tags' selection, seeded.

    ConversationTaggingTaskBundle.py:261-275 - for each query it takes a random
    1..min(3, len(results)) sample, and joins title + "\\n" + snippet, capped at
    1000 chars.
    """
    data = (bundle.get("input") or {}).get("data") or {}
    enrichment = data.get("enrichment") or {}
    results_by_query = enrichment.get("search_results") or {}
    rng = random.Random(seed)

    lines: List[str] = []
    for _query, results in results_by_query.items():
        if not results:
            continue
        n = rng.randint(1, min(3, len(results)))
        for chosen in rng.sample(results, n):
            text = f"{chosen.get('title') or ''}\n{chosen.get('snippet') or ''}"[:1000]
            if text.strip():
                lines.append(text)
    return lines


def case_from_bundle(bundle: dict, seed: int = 0) -> Optional[FaithfulCase]:
    kind = bundle.get("type")
    inp = bundle.get("input") or {}
    data = inp.get("data") or {}
    raw_lines = data.get("lines") or []
    guid = str(inp.get("guid"))

    if kind == "conversation_tagging":
        lines = [tuple(l) for l in raw_lines][:MAX_CONVO_LINES]
        if len(lines) < 4:
            return None
        windows = split_overlap(lines)
        return FaithfulCase(
            guid=guid, kind=kind, full_lines=lines,
            enrichment_lines=build_enrichment_lines(bundle, seed),
            windows=windows,
        )

    # document-shaped tasks: the validator parses JSON out of lines[0]
    try:
        parsed = json.loads(raw_lines[0][1])
    except Exception:
        return None

    if kind == "survey_tagging":
        return FaithfulCase(
            guid=guid, kind=kind, full_lines=[], enrichment_lines=[],
            question=parsed.get("survey_question", ""),
            comment=parsed.get("comment", ""),
            # The literal ground truth - never visible from the miner side.
            reference_tags=parsed.get("selected_choices") or [],
        )

    key = {"webpage_metadata_generation": "website_markdown",
           "named_entities_extraction": "transcript_text",
           "skill_generation": "skill_markdown"}.get(kind)
    if not key or not parsed.get(key):
        return None
    return FaithfulCase(
        guid=guid, kind=kind, full_lines=[],
        enrichment_lines=build_enrichment_lines(bundle, seed),
        document=parsed[key][:1000],
    )


# --------------------------------------------------------------------------
# Ground truth - the validator's own path
# --------------------------------------------------------------------------

def convo_xml(lines: Sequence[tuple]) -> str:
    xml = "<conversation id='faithful'>"
    for line in lines:
        if len(line) != 2:
            continue
        xml += f"<p{line[0]}>{line[1]}</p{line[0]}>"
    xml += "</conversation>"
    return xml


async def real_ground_truth(case: FaithfulCase, model: str = GT_MODEL, use_cache: bool = True) -> GroundTruth:
    """Generate ground truth the way the validator does, from the FULL source."""
    if case.kind == "survey_tagging":
        gt_tags = [t.strip().lower() for t in case.reference_tags if t and t.strip()]
        cleaned = Utils.get_clean_tag_set(gt_tags)
        vecs = await llm.embed(cleaned, use_cache=use_cache)
        return GroundTruth(gt_tags, vecs, scoring.neighborhood(vecs), [gt_tags])

    async def tags_of(prompt: str) -> List[str]:
        # salt="gt" keeps these distinct from the miner's own replica calls,
        # which render the same prompts - a shared cache would hand the miner
        # the validator's exact answer and fake a perfect replica.
        return clean_like_validator(
            await llm.chat(prompt, model=model, timeout=180, temperature=None,
                           use_cache=use_cache, salt="gt")
        )

    if case.kind == "conversation_tagging":
        doc_prompt = prompts.gt_conversation(convo_xml(case.full_lines))
        enrich_prompt = prompts.gt_enrichment
    elif case.kind == "webpage_metadata_generation":
        doc_prompt = prompts.gt_website(case.document)
        enrich_prompt = prompts.gt_enrichment
    elif case.kind == "named_entities_extraction":
        doc_prompt = prompts.gt_transcript_ner(case.document)
        enrich_prompt = prompts.gt_enrichment_ner
    elif case.kind == "skill_generation":
        doc_prompt = prompts.gt_skill(case.document)
        enrich_prompt = prompts.gt_enrichment
    else:
        return GroundTruth([], {}, None, [])

    jobs = [tags_of(doc_prompt)] + [tags_of(enrich_prompt(t)) for t in case.enrichment_lines]
    results = await asyncio.gather(*jobs, return_exceptions=True)
    sets = [r for r in results if not isinstance(r, BaseException) and r]
    if not sets:
        return GroundTruth([], {}, None, [])

    # Conversation combines only when enrichment exists (ConversationTaggingTaskBundle:236);
    # webpage / NER / skill always combine.
    must_combine = case.kind != "conversation_tagging" or len(sets) > 1
    gt_tags = (await tags_of(prompts.gt_combine(sets))) or sets[0] if must_combine else sets[0]

    # get_vector_embeddings_set re-cleans before embedding, so the vector dict is
    # keyed by CLEANED tags while metadata.tags keeps the raw strings. Scoring
    # uses raw tags for the both/unique diff and cleaned tags for the vectors.
    cleaned = Utils.get_clean_tag_set(gt_tags)
    vecs = await llm.embed(cleaned, use_cache=use_cache)
    return GroundTruth(gt_tags, vecs, scoring.neighborhood(vecs), sets)


def pick_window(case: FaithfulCase, seed: int = 0) -> List[tuple]:
    """One window, chosen the way to_mining_tasks samples them."""
    body = case.windows[1:-1] or case.windows      # skip intro/outro boilerplate
    return random.Random(seed).choice(body)


def expand_windows(cases: List[FaithfulCase], per_convo: int = 0, seed: int = 0) -> List[Tuple[FaithfulCase, List[tuple]]]:
    """One test case per WINDOW, not per conversation.

    The testnet pool holds only 4 distinct conversations (115 bundles turned out
    to be 4 sources reissued under fresh guids), but those 4 split into 81
    windows. Each window is a genuinely different miner input, so a PAIRED A/B
    across all 81 has real power - config A and config B see identical windows
    and identical ground truth.

    What it cannot do is generalise across topics: every result is measured on
    4 podcasts. Treat a winner here as a hypothesis to confirm on mainnet, not
    as an established fact.
    """
    per_case: List[List[Tuple[FaithfulCase, List[tuple]]]] = []
    rng = random.Random(seed)
    for c in cases:
        body = c.windows[1:-1] or c.windows      # skip intro/outro boilerplate
        picks = body if not per_convo else rng.sample(body, min(per_convo, len(body)))
        per_case.append([(c, w) for w in picks])

    # INTERLEAVE by conversation. Emitting them back-to-back made `[:n]` slice
    # inside one conversation: window counts are 36/21/16/2, so `--n 40` was
    # 36 windows of one podcast plus 4 of another, with the fourth conversation
    # absent entirely. Every `--n 40` result in this repo was a single-podcast
    # experiment wearing a four-podcast label.
    out: List[Tuple[FaithfulCase, List[tuple]]] = []
    for i in range(max((len(p) for p in per_case), default=0)):
        for bucket in per_case:
            if i < len(bucket):
                out.append(bucket[i])
    return out


def load_cases(path: Optional[str] = None, kind: Optional[str] = None,
               limit: int = 0, seed: int = 0) -> List[FaithfulCase]:
    path = path or os.path.join(ROOT, "data", "testnet_corpus.jsonl")
    import hashlib

    cases: List[FaithfulCase] = []
    seen = set()
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            b = json.loads(line)
            if kind and b.get("type") != kind:
                continue
            c = case_from_bundle(b, seed=seed + i)
            if c is None:
                continue
            # Dedupe on CONTENT: the API serves the same conversation under many
            # guids, so guid-dedup would count 4 sources as 75.
            body = c.full_lines[:3] if c.full_lines else [c.document[:400], c.comment]
            ck = hashlib.md5(json.dumps(body, default=str).encode()).hexdigest()
            if ck in seen:
                continue
            seen.add(ck)
            cases.append(c)
            if limit and len(cases) >= limit:
                break
    return cases
