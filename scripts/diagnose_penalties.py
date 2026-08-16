#!/usr/bin/env python3
"""
For each task in tasks_v2.jsonl (or any path), reproduce the validator's
scoring pipeline locally and report which penalties (if any) would have fired.

Usage:
  python3 diagnose_penalties.py /root/miner_logs/tasks_v2.jsonl
"""
import asyncio
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from conversationgenome.api.models.conversation import Conversation
from conversationgenome.api.models.conversation_metadata import ConversationMetadata
from conversationgenome.llm.llm_factory import get_llm_backend
from conversationgenome.scoring_mechanism.GroundTruthTagSimilarityScoringMechanism import (
    GroundTruthTagSimilarityScoringMechanism,
)
from conversationgenome.scoring_mechanism.NoPenaltyGroundTruthTagSimilarityScoringMechanism import (
    NoPenaltyGroundTruthTagSimilarityScoringMechanism,
)
from conversationgenome.utils.constants import PENALTIES
from conversationgenome.utils.Utils import Utils


class _Axon:
    def __init__(self, name): self.hotkey = name; self.uuid = name + "-uuid"
class _Dendrite: status_code = 200
class _Response:
    def __init__(self, name, output):
        self.axon = _Axon(name); self.dendrite = _Dendrite(); self.cgp_output = output


def gen_local_gt(llml, task_raw):
    tt = task_raw.get("type")
    data = task_raw.get("input", {}).get("data", {})
    window = data.get("window") or []
    if tt == "conversation_tagging":
        if not window:
            return None
        convo = Conversation(guid="gt", lines=[tuple(l) for l in window])
        return llml.conversation_to_metadata(convo, generateEmbeddings=True)
    if tt == "named_entities_extraction":
        if not window:
            return None
        return llml.raw_transcript_to_named_entities(window[0][1][:1000], generateEmbeddings=True)
    if tt == "webpage_metadata_generation":
        if not window:
            return None
        return llml.website_to_metadata(window[0][1][:1000], generateEmbeddings=True)
    if tt == "survey_tagging":
        # Survey GT in tasks.jsonl is not stored; we don't replay it here
        return None
    return None


def explain_penalties(num_tags, num_unique, max_score):
    """Return list of penalty descriptions that fired."""
    fired = []
    num_both = num_tags - num_unique
    if num_both == 0:
        fired.append(f"no_both_tags (×{PENALTIES['no_both_tags']['penalty']}) — 0 of our tags exactly match a GT tag")
    if max_score < PENALTIES["all_junk_tags"]["threshold"]:
        fired.append(f"all_junk_tags (×{PENALTIES['all_junk_tags']['penalty']}) — max cosine sim {max_score:.3f} < 0.2")
    if num_tags < PENALTIES["too_few_tags"]["threshold"]:
        fired.append(f"too_few_tags (×{PENALTIES['too_few_tags']['penalty']}) — only {num_tags} clean tags")
    if num_unique < 1:
        fired.append(f"less_than_1_unique (×{PENALTIES['num_unique_tags']['less_than_1']['penalty']})")
    elif num_unique < 2:
        fired.append(f"less_than_2_unique (×{PENALTIES['num_unique_tags']['less_than_2']['penalty']})")
    elif num_unique < 3:
        fired.append(f"less_than_3_unique (×{PENALTIES['num_unique_tags']['less_than_3']['penalty']})")
    return fired


async def diagnose_one(idx, rec, llml):
    tt = rec.get("task_type")
    task_raw = rec.get("task_raw")
    result = rec.get("result") or {}
    miner_tags = result.get("tags") or []
    if not task_raw or not miner_tags:
        print(f"[{idx}] {tt}: no task_raw or no miner tags, skipping")
        return

    # Generate GT
    gt = gen_local_gt(llml, task_raw)
    if not gt or not gt.tags:
        print(f"[{idx}] {tt}: couldn't generate GT, skipping")
        return
    gt_tags = gt.tags
    gt_meta = ConversationMetadata(tags=list(gt_tags), vectors=gt.vectors or {})

    # format_results: validate_tag_set + embed
    if tt == "named_entities_extraction":
        clean_tags = llml.validate_named_entities_tag_set(miner_tags) or []
    else:
        clean_tags = llml.validate_tag_set(miner_tags) or []
    clean_vecs = llml.get_vector_embeddings_set(clean_tags) if clean_tags else {}

    # Score
    if tt == "named_entities_extraction":
        scorer = NoPenaltyGroundTruthTagSimilarityScoringMechanism()
        scorer.min_tags = 1
    elif tt == "survey_tagging":
        scorer = GroundTruthTagSimilarityScoringMechanism()
        scorer.min_tags = 1
    else:
        scorer = GroundTruthTagSimilarityScoringMechanism()

    resp = _Response("x", [{"tags": clean_tags, "vectors": clean_vecs, "original_tags": miner_tags}])

    class _FBI: metadata = gt_meta
    class _FB: input = _FBI()

    final_scores, _ = await scorer.evaluate(_FB(), [resp])
    s = final_scores[0] if final_scores else {}
    final = s.get("final_miner_score", 0.0)
    adj = s.get("adjustedScore", 0.0)

    # Now compute the unique/both breakdown ourselves to explain penalties
    tag_set = list(set(clean_tags))
    diff = Utils.compare_arrays(gt_tags, tag_set)
    both = diff["both"]
    unique = diff["unique_2"]

    # The scoring uses the FIRST 20 tags (max_scored_tags=20).
    scored_count = min(20, len(tag_set))
    scored_unique = [t for t in tag_set[:20] if t in unique]
    num_unique_scored = len(scored_unique)

    # max cosine sim
    import numpy as np
    full_neighborhood = np.mean([v["vectors"] for v in gt_meta.vectors.values()], axis=0) if gt_meta.vectors else None
    max_sim = 0.0
    if full_neighborhood is not None and clean_vecs:
        for tag, vec_obj in clean_vecs.items():
            v = vec_obj.get("vectors")
            if not v: continue
            v = np.array(v)
            sim = np.dot(full_neighborhood, v) / (np.linalg.norm(full_neighborhood) * np.linalg.norm(v) + 1e-9)
            if sim > max_sim: max_sim = sim

    penalties = explain_penalties(scored_count, num_unique_scored, max_sim)

    print(f"\n[{idx}] {tt}  final={final:.4f} adj={adj:.4f}")
    print(f"     raw miner tags ({len(miner_tags)}): {miner_tags[:8]}")
    print(f"     after validate_tag_set ({len(clean_tags)}): {clean_tags[:8]}")
    print(f"     GT tags ({len(gt_tags)}): {gt_tags[:8]}")
    print(f"     BOTH (matching): {both[:6]}  ({len(both)} total)")
    print(f"     UNIQUE (not in GT): {unique[:6]}  ({len(unique)} total)")
    print(f"     max cosine sim: {max_sim:.3f}")
    if penalties:
        print(f"     PENALTIES: {penalties}")
    else:
        print(f"     PENALTIES: none")


async def main():
    if len(sys.argv) < 2:
        print("Usage: diagnose_penalties.py <path_to_tasks.jsonl>")
        sys.exit(1)
    path = sys.argv[1]
    llml = get_llm_backend()
    print(f"Backend: {type(llml).__name__}  Model: {llml.model}")

    records = [json.loads(l) for l in open(path) if l.strip()]
    print(f"Diagnosing {len(records)} tasks from {path}\n")
    print("=" * 110)

    for i, rec in enumerate(records):
        await diagnose_one(i, rec, llml)


if __name__ == "__main__":
    asyncio.run(main())
