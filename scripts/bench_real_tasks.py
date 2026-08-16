#!/usr/bin/env python3
"""
Score real mainnet miner outputs (from data/tasks.jsonl) against our
candidate prompt, using the same GroundTruthTagSimilarityScoringMechanism
the validator uses.

For each task:
  1. Build the window content exactly as the miner saw it.
  2. Generate ground-truth tags + vectors LOCALLY on that same content
     (so the baseline is shared for both the mainnet miner and our candidate).
  3. Pull the mainnet miner's actual tags from task.result.tags.
  4. Run our candidate prompt on the same input.
  5. Score BOTH against the local GT. Compare.

Limitations:
  - For conversation_tagging, the real validator generates GT from the
    FULL conversation, not the window. We only have the window, so our
    local GT will differ from the real validator's GT. Comparisons are
    still fair BETWEEN candidates since GT is identical.
  - For webpage_metadata_generation / named_entities_extraction, the
    first window line is the full doc, so local GT is essentially correct.
"""
import argparse
import asyncio
import json
import os
import random
import statistics
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
from conversationgenome.utils.Utils import Utils

TASKS_PATH = os.path.join(ROOT, "data", "tasks.jsonl")

# Our winning prompt from scripts/local_bench.py (candA_mimic).
# Already locked into conversation_to_metadata.j2 — kept here for parity
# with the non-coding webpage path too.
CANDA_PROMPT = (
    "Analyze the following content in terms of topic interests. "
    "Generate 10 to 15 short, general topic tags in lowercase that a "
    "semantic embedding model would consider close in meaning to the "
    "overall theme. Favor broad, reusable topic words over rare compound "
    "phrases. Include the obvious subject-matter tags that anyone reading "
    "the full content would write. Tags should be 1 to 3 words each, "
    "lowercase, no punctuation, no hyphens, no underscores, no duplicates, "
    "and use common English dictionary words only (no abbreviations, "
    "invented compounds, or typos). Do not include participant names, "
    "greetings, pronouns, or filler words. Return ONLY a comma-delimited "
    "list of tags with no commentary.\n\nContent:\n"
)


class _Axon:
    def __init__(self, name):
        self.hotkey = name
        self.uuid = name + "-uuid"


class _Dendrite:
    status_code = 200


class _Response:
    def __init__(self, name, output):
        self.axon = _Axon(name)
        self.dendrite = _Dendrite()
        self.cgp_output = output


def build_convo_xml(window_lines):
    xml = "<conversation id='83945'>"
    for line in window_lines:
        if len(line) != 2:
            continue
        xml += "<p%d>%s</p%d>" % (line[0], line[1], line[0])
    xml += "</conversation>"
    return xml


def build_content_for_task(task_raw):
    """Return (kind, content) where kind in {'conversation','text'}."""
    task_type = task_raw["type"]
    data = task_raw.get("input", {}).get("data", {})
    window = data.get("window") or []
    if task_type == "conversation_tagging":
        return ("conversation", window)
    # All other task types have window[0] as the main document text.
    if not window:
        return ("text", "")
    return ("text", window[0][1] if len(window[0]) >= 2 else "")


def generate_local_gt(llml, kind, content, task_type):
    """Generate ground-truth tags + vectors on the same content the miner saw,
    using the same code path the real validator would use for that task type."""
    if kind == "conversation":
        if not content:
            return None
        convo = Conversation(guid="local-gt", lines=[tuple(l) for l in content])
        return llml.conversation_to_metadata(convo, generateEmbeddings=True)
    if not content or not content.strip():
        return None
    if task_type == "named_entities_extraction":
        return llml.raw_transcript_to_named_entities(content[:1000], generateEmbeddings=True)
    # webpage_metadata_generation and friends
    return llml.website_to_metadata(content[:1000], generateEmbeddings=True)


def run_candidate(llml, kind, content, task_type):
    """Return {tags, vectors=None} by exercising the SAME production code
    path the miner runs (so we test the .j2 files, not an inline prompt)."""
    if kind == "conversation":
        if not content:
            return {"tags": [], "vectors": None}
        convo = Conversation(guid="cand", lines=[tuple(l) for l in content])
        result = llml.conversation_to_metadata(convo, generateEmbeddings=False)
    elif task_type == "named_entities_extraction":
        if not content or not content.strip():
            return {"tags": [], "vectors": None}
        result = llml.raw_transcript_to_named_entities(content[:1000], generateEmbeddings=False)
    else:
        # webpage_metadata_generation and friends
        if not content or not content.strip():
            return {"tags": [], "vectors": None}
        result = llml.website_to_metadata(content[:1000], generateEmbeddings=False)

    if not result or not result.tags:
        return {"tags": [], "vectors": None}
    return {"tags": result.tags, "vectors": None}


async def score_one(scorer, gt_metadata, result_dict, task_type):
    """Mirror the production format_results + scorer for the given task type."""
    llml = get_llm_backend()
    resp = _Response("x", [dict(result_dict)])
    mr = resp.cgp_output[0]
    mr["original_tags"] = mr["tags"]

    # NER uses a different (more lenient) tag validator on mainnet.
    if task_type == "named_entities_extraction":
        mr["tags"] = llml.validate_named_entities_tag_set(mr["original_tags"]) or []
    else:
        mr["tags"] = llml.validate_tag_set(mr["original_tags"]) or []

    if mr["tags"]:
        mr["vectors"] = llml.get_vector_embeddings_set(mr["tags"])
    else:
        mr["vectors"] = {}

    class _FakeBundleInput:
        metadata = gt_metadata

    class _FakeBundle:
        input = _FakeBundleInput()

    scores, _ = await scorer.evaluate(_FakeBundle(), [resp])
    return scores[0] if scores else {"final_miner_score": 0.0, "adjustedScore": 0.0}, mr


def get_scorer_for(task_type):
    """Use the same scorer the validator's task_bundle.evaluate() uses."""
    if task_type == "named_entities_extraction":
        scorer = NoPenaltyGroundTruthTagSimilarityScoringMechanism()
        scorer.min_tags = 1  # mainnet sets this for NER
        return scorer
    return GroundTruthTagSimilarityScoringMechanism()


async def run(args):
    llml = get_llm_backend()

    with open(TASKS_PATH) as f:
        tasks_all = [json.loads(l) for l in f]

    # Filter tasks we can score
    wanted_types = {"conversation_tagging", "webpage_metadata_generation",
                    "named_entities_extraction"}
    tasks = [t for t in tasks_all
             if t["task_type"] in wanted_types
             and t.get("result") and t["result"].get("tags")]

    # Balanced sample
    by_type = {}
    for t in tasks:
        by_type.setdefault(t["task_type"], []).append(t)

    random.seed(42)
    sampled = []
    per_type_cap = {
        "conversation_tagging": args.convo_tasks,
        "webpage_metadata_generation": args.webpage_tasks,
        "named_entities_extraction": args.ner_tasks,
    }
    for tt, items in by_type.items():
        cap = per_type_cap.get(tt, 0)
        if cap > 0:
            sampled.extend(random.sample(items, min(cap, len(items))))

    print(f"Scoring {len(sampled)} real mainnet tasks:")
    for tt in sorted(set(t['task_type'] for t in sampled)):
        print(f"  {tt}: {sum(1 for t in sampled if t['task_type']==tt)}")
    print(f"Model: {llml.model}  Embedding: {llml.embedding_model}")
    print("=" * 90)

    per_group = {"real_miner": {}, "candA_new": {}}
    # Group stats by task_type
    stats_by_type = {}

    for idx, t in enumerate(sampled):
        tt = t["task_type"]
        task_raw = t["task_raw"]
        kind, content = build_content_for_task(task_raw)

        # Skip if nothing to work with
        if kind == "conversation" and not content:
            continue
        if kind == "text" and not content:
            continue

        gt = generate_local_gt(llml, kind, content, tt)
        if not gt or not gt.tags:
            print(f"[{idx}] {tt}: SKIP (no GT)")
            continue
        gt_meta = ConversationMetadata(tags=gt.tags, vectors=gt.vectors or {})

        # Real miner tags from mainnet (already validated & embedded by val? no —
        # the jsonl stores the RAW miner output before validator's validate_tag_set.
        # So we treat it like any other candidate: run validate + embed, then score.)
        real_tags = t["result"]["tags"]
        real_result = {"tags": real_tags, "vectors": None}

        # Our new prompts (exercising production code path / updated .j2 files)
        cand_result = run_candidate(llml, kind, content, tt)

        # Use the production scorer for this task type
        scorer = get_scorer_for(tt)
        (real_score, real_mr) = await score_one(scorer, gt_meta, real_result, tt)
        (cand_score, cand_mr) = await score_one(scorer, gt_meta, cand_result, tt)

        rf = real_score.get("final_miner_score", 0.0)
        ra = real_score.get("adjustedScore", 0.0)
        cf = cand_score.get("final_miner_score", 0.0)
        ca = cand_score.get("adjustedScore", 0.0)

        stats_by_type.setdefault(tt, {"real": [], "cand": [], "real_adj": [], "cand_adj": []})
        stats_by_type[tt]["real"].append(float(rf))
        stats_by_type[tt]["cand"].append(float(cf))
        stats_by_type[tt]["real_adj"].append(float(ra))
        stats_by_type[tt]["cand_adj"].append(float(ca))

        print(f"[{idx:3d}] {tt:28s} | real: final={rf:.4f} clean={len(real_mr['tags']):2d} "
              f"raw={len(real_tags):2d}  | candA: final={cf:.4f} clean={len(cand_mr['tags']):2d}")

    print("\n" + "=" * 90)
    print("SUMMARY by task type (real mainnet miner vs our candA_mimic)")
    print("=" * 90)

    totals = {"real": [], "cand": []}
    for tt, s in stats_by_type.items():
        print(f"\n{tt} (n={len(s['real'])})")
        for label, key in [("real_miner", "real"), ("candA_new", "cand")]:
            arr = s[key]
            adj_arr = s[key + "_adj"]
            pen = sum(1 for f, a in zip(arr, adj_arr) if a - f > 1e-3) / len(arr) * 100
            print(f"  {label:12s}: mean={statistics.mean(arr):.4f} "
                  f"median={statistics.median(arr):.4f} "
                  f"max={max(arr):.4f} min={min(arr):.4f} "
                  f"penalty%={pen:.1f}")
        diff = statistics.mean(s["cand"]) - statistics.mean(s["real"])
        print(f"  --> candA lift: {diff:+.4f}  ({diff/max(1e-6, statistics.mean(s['real']))*100:+.1f}%)")
        totals["real"].extend(s["real"])
        totals["cand"].extend(s["cand"])

    print("\n" + "=" * 90)
    print(f"OVERALL n={len(totals['real'])}")
    print(f"  real_miner: mean={statistics.mean(totals['real']):.4f} "
          f"median={statistics.median(totals['real']):.4f} "
          f"max={max(totals['real']):.4f} min={min(totals['real']):.4f}")
    print(f"  candA_new : mean={statistics.mean(totals['cand']):.4f} "
          f"median={statistics.median(totals['cand']):.4f} "
          f"max={max(totals['cand']):.4f} min={min(totals['cand']):.4f}")
    lift = statistics.mean(totals['cand']) - statistics.mean(totals['real'])
    print(f"  candA lift: {lift:+.4f}  ({lift/statistics.mean(totals['real'])*100:+.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convo-tasks", type=int, default=15)
    ap.add_argument("--webpage-tasks", type=int, default=8)
    ap.add_argument("--ner-tasks", type=int, default=0, help="NER is scored differently on mainnet (NoPenalty). Default skipped.")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
