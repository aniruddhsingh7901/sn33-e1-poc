#!/usr/bin/env python3
"""
Local benchmark harness: runs the REAL validator scoring pipeline against
multiple candidate miner prompts on canned conversations. No ReadyAI/Bittensor
network dependency; only OpenAI.

Flow per conversation:
  1. Validator generates ground-truth tags + vectors on the FULL conversation
     (conversation_to_metadata, generateEmbeddings=True).
  2. Conversation is split into windows exactly like the validator does.
  3. For each window, each candidate miner calls the LLM with its own prompt
     and returns {tags, vectors=None}.
  4. Validator's format_results runs validate_tag_set and re-embeds the tags.
  5. GroundTruthTagSimilarityScoringMechanism scores the responses.

Usage:
  export OPENAI_API_KEY=sk-...
  python3 scripts/local_bench.py --convos 5 --windows-per-convo 3
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Silence bittensor's logger noise
os.environ.setdefault("BT_LOGGING_LOGGING_LEVEL", "info")

from conversationgenome.api.models.conversation import Conversation
from conversationgenome.api.models.conversation_metadata import ConversationMetadata
from conversationgenome.api.models.raw_metadata import RawMetadata
from conversationgenome.ConfigLib import c
from conversationgenome.llm.llm_factory import get_llm_backend
from conversationgenome.scoring_mechanism.GroundTruthTagSimilarityScoringMechanism import (
    GroundTruthTagSimilarityScoringMechanism,
)
from conversationgenome.task_bundle.ConversationTaggingTaskBundle import (
    ConversationInput,
    ConversationInputData,
    ConversationTaggingTaskBundle,
)
from conversationgenome.utils.Utils import Utils

DATA_PATH = os.path.join(ROOT, "data", "facebook-chat-data.json")


# --------- Candidate prompts ----------------------------------------------
# Each candidate takes a conversation_xml string and returns the LLM prompt.
# The VALIDATOR ground truth always uses the default conversation_to_metadata.j2.

# This is the STOCK prompt (v2.30.65) — what every mainnet miner runs today.
# Kept here hard-coded so we can always benchmark against the untouched baseline
# even after conversation_to_metadata.j2 is edited.
DEFAULT_MINER_PROMPT = (
    "Analyze conversation in terms of topic interests of the participants. "
    "Analyze the conversation (provided in structured XML format) where <p0> "
    "has the questions and <p1> has the answers. Return comma-delimited tags. "
    "Only return the tags without any English commentary."
)

# Candidate A: Ground-truth-mimicry. Matches the validator's own style more tightly.
CANDIDATE_A_PROMPT = (
    "Analyze the following conversation in terms of topic interests of the "
    "participants. Generate 10-15 short, general topic tags in lowercase that "
    "a semantic embedding model would consider close in meaning to the "
    "overall conversation theme. Favor broad, reusable topic words over "
    "rare compound phrases. Include the obvious subject-matter tags that "
    "anyone reading the full conversation would write. Tags should be "
    "1-3 words each, lowercase, no punctuation, no duplicates. "
    "Return ONLY a comma-delimited list of tags with no commentary.\n\n"
    "Conversation (XML format, <p0> and <p1> are participants):"
)

# Candidate B: "Hub + Spoke" — a few hub tags (to hit 'both' tags) + many
# on-topic unique tags to boost top_3_mean_unique.
CANDIDATE_B_PROMPT = (
    "You are tagging a short window of a longer conversation for a semantic "
    "search index. Produce exactly 12 tags as a comma-delimited list, "
    "lowercase, 1-2 words each. "
    "Include at least 3 broad topic tags that summarize the overall subject "
    "of the conversation (e.g. 'hobbies', 'cooking', 'travel'). "
    "Then add 8-9 specific on-topic tags that are strongly related to the "
    "main theme but describe details mentioned in this window. "
    "Do NOT include participant names, greetings, filler words, or tags "
    "with more than 2 words. "
    "Return only the tags, no commentary.\n\n"
    "Conversation XML:"
)

# Candidate C: aggressive - ask for Title-style noun tags only
CANDIDATE_C_PROMPT = (
    "Read the conversation XML below. Output 10 single-noun or two-word "
    "topic tags in lowercase, comma-delimited, that capture the main "
    "interests and subject matter. No verbs, no greetings, no adjectives "
    "alone. Common English dictionary words only (avoid abbreviations, "
    "invented compounds, or typos). Output only the comma list.\n\n"
)

# Candidate D: targeted at scoring formula.
# Score = 0.55*top_3_mean_UNIQUE + 0.25*mean + 0.10*median + 0.10*max
# Since the top_3_mean_UNIQUE term dominates, we want MANY unique-yet-close
# semantic variations of the conversation's theme — not just the obvious
# tags a ground-truth LLM would also pick. Dictionary-strict (validate_tag_set).
CANDIDATE_D_PROMPT = (
    "You are producing semantic tags for a conversation window. These tags "
    "will be embedded with text-embedding-3-small and compared by cosine "
    "similarity to tags generated for the ENTIRE conversation.\n\n"
    "Rules:\n"
    "- Output exactly 15 comma-delimited tags, all lowercase.\n"
    "- Each tag is 1-2 common English dictionary words (no hyphens, no "
    "underscores, no punctuation, no abbreviations, no typos, no invented "
    "compound words). Single nouns are preferred.\n"
    "- Produce a mix: 4-5 broad high-level topic tags (the obvious subject "
    "headings a reader would pick), PLUS 10-11 tightly related on-topic "
    "synonyms, hyponyms, and adjacent concepts that a thesaurus would "
    "group near the main topic. These synonym variants are the most "
    "important — they should describe the same semantic space as the "
    "broad topics, but use different words.\n"
    "- Do NOT include participant names, greetings, pronouns, generic "
    "words ('thing', 'stuff', 'conversation'), or anything not tied to "
    "the actual subject.\n"
    "- No duplicates. No tag repeated in singular+plural form.\n\n"
    "Example: if the conversation is about hurricanes in Florida, good tags "
    "include: hurricanes, storms, weather, florida, coast, tropical, "
    "cyclone, wind, flooding, evacuation, disasters, climate, atlantic, "
    "gulf, forecast.\n\n"
    "Return only the comma-delimited tag list with no commentary. "
    "Conversation XML:"
)

# Candidate E: leaner on tag count (10 tags) but optimized for max_score term.
CANDIDATE_E_PROMPT = (
    "Produce exactly 12 semantic tags summarizing the conversation below. "
    "Tags will be embedded and compared by cosine similarity to the "
    "conversation's overall theme vector.\n\n"
    "Strategy: First privately identify the 1-2 CORE subjects of the "
    "conversation. Then emit 12 lowercase tags that all sit tightly in "
    "that semantic region. Prefer single common nouns; allow occasional "
    "two-word noun phrases. Include the single strongest on-topic noun "
    "twice is FORBIDDEN — all 12 tags must be distinct words. No "
    "hyphens, underscores, punctuation, abbreviations, names of "
    "participants, or filler.\n\n"
    "Output only the comma-delimited list.\n"
    "Conversation XML:"
)


def build_xml(window_lines):
    xml = "<conversation id='%d'>" % 83945
    for line in window_lines:
        if len(line) != 2:
            continue
        participant = "p%d" % line[0]
        xml += "<%s>%s</%s>" % (participant, line[1], participant)
    xml += "</conversation>"
    return xml


CANDIDATES = {
    "default": DEFAULT_MINER_PROMPT,
    "candA_mimic": CANDIDATE_A_PROMPT,
    "candB_hub": CANDIDATE_B_PROMPT,
    "candC_noun": CANDIDATE_C_PROMPT,
    "candD_syn": CANDIDATE_D_PROMPT,
    "candE_core": CANDIDATE_E_PROMPT,
}


# --------- Lightweight "response" object ------------------------------------
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


# --------- Core run ---------------------------------------------------------

async def run_candidate(llml, name, prompt_text, xml):
    """Send one miner-style prompt to the LLM and return {tags, vectors}."""
    full_prompt = prompt_text + "\n\n" + xml
    resp = llml.basic_prompt(full_prompt)
    if not isinstance(resp, str):
        return {"tags": [], "vectors": None}
    tags = Utils.clean_tags(resp.split(","))
    return {"tags": tags, "vectors": None}


async def score_window(scorer, full_convo_metadata, miner_results):
    # Build fake "responses" that the scorer expects.
    responses = [_Response(name, [res]) for name, res in miner_results]
    # Stash miner_metadata (must have vectors, but scorer will regenerate via format_results).
    # The scoring mechanism expects vectors already present — so we generate them here
    # the same way format_results does (validate_tag_set + embed).
    llml = get_llm_backend()
    for resp in responses:
        mr = resp.cgp_output[0]
        mr["original_tags"] = mr["tags"]
        mr["tags"] = llml.validate_tag_set(mr["original_tags"]) or []
        if mr["tags"]:
            mr["vectors"] = llml.get_vector_embeddings_set(mr["tags"])
        else:
            mr["vectors"] = {}

    # Inject metadata into a pretend task_bundle input.
    class _FakeBundleInput:
        metadata = None

    class _FakeBundle:
        input = _FakeBundleInput()

    _FakeBundleInput.metadata = full_convo_metadata
    fake_bundle = _FakeBundle()

    final_scores, _rank_scores = await scorer.evaluate(fake_bundle, responses)
    return final_scores


async def run_bench(convos_to_use, windows_per_convo, candidate_names):
    llml = get_llm_backend()

    with open(DATA_PATH) as f:
        data = json.load(f)

    convo_ids = list(data.keys())[:convos_to_use]
    print(f"Running {len(convo_ids)} conversations × up to {windows_per_convo} windows")
    print(f"Candidates: {candidate_names}")
    print(f"Model: {llml.model}  Embedding: {llml.embedding_model}")
    print("=" * 80)

    per_candidate_scores = {name: [] for name in candidate_names}
    per_candidate_adjusted = {name: [] for name in candidate_names}

    scorer = GroundTruthTagSimilarityScoringMechanism()

    for idx, cid in enumerate(convo_ids):
        convo_raw = data[cid]
        lines = [tuple(l) for l in convo_raw["lines"]]
        print(f"\n--- Convo {cid} ({len(lines)} lines) ---")

        # 1. Validator: generate ground truth on FULL conversation
        convo = Conversation(guid=str(cid), lines=lines,
                             miner_task_prompt=DEFAULT_MINER_PROMPT)
        gt = llml.conversation_to_metadata(convo, generateEmbeddings=True)
        if not gt or not gt.success or not gt.tags:
            print("  SKIP: no ground truth")
            continue
        print(f"  GT tags: {gt.tags}")
        gt_metadata = ConversationMetadata(tags=gt.tags, vectors=gt.vectors or {})

        # 2. Split into windows (same logic as ConversationTaggingTaskBundle)
        windows = Utils.split_overlap_array(lines, size=10, overlap=2)
        if len(windows) < 2:
            windows = Utils.split_overlap_array(lines, size=2, overlap=2)
        windows = windows[:windows_per_convo]

        for w_idx, window in enumerate(windows):
            if not window:
                continue
            xml = build_xml(window)

            # 3. Each candidate miner generates tags
            miner_results = []
            for name in candidate_names:
                prompt = CANDIDATES[name]
                res = await run_candidate(llml, name, prompt, xml)
                miner_results.append((name, res))

            # 4. Score them all with the real validator scoring mechanism
            scores = await score_window(scorer, gt_metadata, miner_results)

            print(f"  Window {w_idx}:")
            for name, (s, (_, mr)) in zip(candidate_names, zip(scores, miner_results)):
                adj = s.get("adjustedScore", 0.0)
                fin = s.get("final_miner_score", 0.0)
                per_candidate_adjusted[name].append(float(adj))
                per_candidate_scores[name].append(float(fin))
                raw_n = len(mr.get("original_tags", []))
                clean_n = len(mr.get("tags", []))
                # Show first 6 clean tags to spot-check style
                sample = ", ".join(mr.get("tags", [])[:6])
                print(f"    {name:12s}: final={fin:.4f} adj={adj:.4f} raw={raw_n} clean={clean_n}  [{sample}]")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'candidate':<14} {'n':>4} {'mean_final':>11} {'median':>9} {'max':>7} {'mean_adj':>9} {'penalty%':>9}")
    baseline_mean = None
    for name in candidate_names:
        fs = per_candidate_scores[name]
        adjs = per_candidate_adjusted[name]
        if not fs:
            continue
        mean_f = statistics.mean(fs)
        med = statistics.median(fs)
        mx = max(fs)
        mean_a = statistics.mean(adjs)
        # penalty rate = (adjusted > final) fraction
        pen = sum(1 for f, a in zip(fs, adjs) if a - f > 1e-3) / len(fs) * 100
        print(f"{name:<14} {len(fs):>4} {mean_f:>11.4f} {med:>9.4f} {mx:>7.4f} {mean_a:>9.4f} {pen:>8.1f}%")
        if name == "default":
            baseline_mean = mean_f

    if baseline_mean is not None:
        print("\nLift vs default baseline (mean_final):")
        for name in candidate_names:
            if name == "default":
                continue
            fs = per_candidate_scores[name]
            if not fs:
                continue
            lift = statistics.mean(fs) - baseline_mean
            print(f"  {name:<14}: {lift:+.4f}  ({lift/baseline_mean*100:+.1f}%)")

    print("\nTop-miner-on-mainnet baseline: mean 0.528 / max 0.687 (UID 21 wandb)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convos", type=int, default=3, help="Number of conversations")
    ap.add_argument("--windows-per-convo", type=int, default=3)
    ap.add_argument("--candidates", nargs="+", default=list(CANDIDATES.keys()))
    args = ap.parse_args()

    for name in args.candidates:
        if name not in CANDIDATES:
            raise SystemExit(f"Unknown candidate: {name}. Known: {list(CANDIDATES)}")

    asyncio.run(run_bench(args.convos, args.windows_per_convo, args.candidates))


if __name__ == "__main__":
    main()
