#!/usr/bin/env python3
"""
Test the new survey prompt against ALL 12 real survey tasks from tasks.jsonl.
Mirrors the EXACT validator pipeline:
  1. Miner produces tags via prompt
  2. Validator's format_results: tags = validate_tag_set(tags)  <-- LLM "malformed" filter
  3. Validator embeds the surviving tags
  4. Validator scores via GroundTruthTagSimilarityScoringMechanism (min_tags=1 for survey)

Critical risk for survey: validate_tag_set asks an LLM to classify "good English vs
malformed" keywords. Spanish tags risk being filtered out.
"""
import asyncio
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from conversationgenome.api.models.conversation_metadata import ConversationMetadata
from conversationgenome.llm.llm_factory import get_llm_backend
from conversationgenome.llm.prompt_manager import prompt_manager
from conversationgenome.scoring_mechanism.GroundTruthTagSimilarityScoringMechanism import (
    GroundTruthTagSimilarityScoringMechanism,
)
from conversationgenome.utils.Utils import Utils

TASKS_PATH = os.path.join(ROOT, "data", "tasks.jsonl")


class _Axon:
    def __init__(self, name):
        self.hotkey = name; self.uuid = name + "-uuid"


class _Dendrite: status_code = 200


class _Response:
    def __init__(self, name, output):
        self.axon = _Axon(name); self.dendrite = _Dendrite(); self.cgp_output = output


async def score_against_gt(llml, scorer, gt_meta, tags):
    """Mirror exactly what the validator does to a miner result."""
    resp = _Response("x", [{"tags": list(tags), "vectors": None}])
    mr = resp.cgp_output[0]
    mr["original_tags"] = mr["tags"]
    # Survey uses validate_tag_set (the LLM-based filter), NOT the lenient NER one.
    mr["tags"] = llml.validate_tag_set(mr["original_tags"]) or []
    if mr["tags"]:
        mr["vectors"] = llml.get_vector_embeddings_set(mr["tags"])
    else:
        mr["vectors"] = {}

    class _FBI:
        metadata = gt_meta

    class _FB:
        input = _FBI()

    scores, _ = await scorer.evaluate(_FB(), [resp])
    return scores[0] if scores else {"final_miner_score": 0.0, "adjustedScore": 0.0}, mr


async def main():
    llml = get_llm_backend()
    scorer = GroundTruthTagSimilarityScoringMechanism()
    scorer.min_tags = 1  # survey rule
    print(f"Backend: {type(llml).__name__}  Model: {llml.model}")
    print()

    survey_tasks = []
    with open(TASKS_PATH) as f:
        for line in f:
            t = json.loads(line)
            if t['task_type'] == 'survey_tagging':
                survey_tasks.append(t)
    print(f"Total survey tasks: {len(survey_tasks)}\n")
    print("=" * 110)

    new_zero, new_total, new_score_sum = 0, 0, 0.0
    old_zero, old_total, old_score_sum = 0, 0, 0.0
    new_post_validate_lt2, old_post_validate_lt2 = 0, 0

    for i, t in enumerate(survey_tasks):
        data = t['task_raw'].get('input', {}).get('data', {})
        q = data.get('survey_question', '')
        c = data.get('comment', '')
        old_tags = t.get('result', {}).get('tags', [])

        # 1) NEW prompt produces tags
        prompt = prompt_manager.survey_tag_prompt(q, c)
        try:
            response = llml.basic_prompt(prompt)
        except Exception as e:
            print(f"[{i:2d}] LLM error: {e}")
            new_zero += 1
            continue

        new_raw_tags = Utils.clean_tags(response.split(",")) if isinstance(response, str) else []
        new_raw_tags = [t for t in new_raw_tags if t]

        # 2) Build a SHARED proxy GT from the OLD miner tags so we can score symmetrically.
        #    (The real GT — selected_choices — isn't in tasks.jsonl. Using OLD as GT means
        #    OLD will always score 1.0 against itself; what we care about is whether NEW
        #    produces tags semantically close to OLD AND whether validate_tag_set kills
        #    Spanish tags or not.)
        if not old_tags:
            print(f"[{i:2d}] no OLD tags to use as GT proxy, skipping")
            continue
        # Generate GT vectors from OLD tags (this is what real validator embedding would do)
        gt_vecs = llml.get_vector_embeddings_set(old_tags)
        gt_meta = ConversationMetadata(tags=list(old_tags), vectors=gt_vecs)

        # 3) Run BOTH new and old through real validator pipeline
        new_score, new_mr = await score_against_gt(llml, scorer, gt_meta, new_raw_tags)
        old_score, old_mr = await score_against_gt(llml, scorer, gt_meta, old_tags)

        new_clean_n = len(new_mr['tags'])
        old_clean_n = len(old_mr['tags'])
        if new_clean_n < 2: new_post_validate_lt2 += 1
        if old_clean_n < 2: old_post_validate_lt2 += 1

        nf = new_score['final_miner_score']
        of = old_score['final_miner_score']
        new_score_sum += nf
        old_score_sum += of
        new_total += 1
        old_total += 1
        if nf < 0.01: new_zero += 1
        if of < 0.01: old_zero += 1

        marker = "✅" if nf > 0 else "❌"
        print(f"[{i:2d}] {marker} q='...{c[:40]}' | "
              f"NEW: raw={len(new_raw_tags)} clean={new_clean_n} score={nf:.4f}  | "
              f"OLD: raw={len(old_tags)} clean={old_clean_n} score={of:.4f}")
        if new_clean_n < 2 or old_clean_n < 2:
            print(f"     surviving NEW after validate_tag_set: {new_mr['tags']}")
            print(f"     surviving OLD after validate_tag_set: {old_mr['tags']}")

    print()
    print("=" * 110)
    print("SUMMARY (validator pipeline applied)")
    print(f"  NEW:  mean score = {new_score_sum/max(1,new_total):.4f}  zeros={new_zero}/{new_total}  "
          f"<2-clean-tags-after-validate={new_post_validate_lt2}/{new_total}")
    print(f"  OLD:  mean score = {old_score_sum/max(1,old_total):.4f}  zeros={old_zero}/{old_total}  "
          f"<2-clean-tags-after-validate={old_post_validate_lt2}/{old_total}")
    print()
    print("Note: GT used here is the OLD miner's own tags (no real selected_choices in tasks.jsonl).")
    print("So OLD will tend to score higher just by self-similarity. The KEY check is:")
    print("  - NEW must have ZERO post-validate-<2 cases (the bug we're fixing)")
    print("  - NEW must NOT score zero on any task")


if __name__ == "__main__":
    asyncio.run(main())
