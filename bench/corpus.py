"""Benchmark corpora.

Provenance matters more than volume here, so each source is labelled with how
faithful it is to what a validator actually sends:

conversation_tagging  -  ReadyAI's own retired podcast corpus
    ``ReadyAi/5000-podcast-conversations-with-metadata-and-embedding-dataset``
    (4,888 full transcripts). This is the only source that contains FULL
    conversations, which is what the validator builds ground truth from. Our own
    production logs only ever captured the 10-line window, so scoring against
    them would compare a window-derived answer to a window-derived target and
    flatter every strategy equally. Windows are cut here with the validator's
    own parameters (size 10, overlap 2).

webpage / NER  -  real captured mainnet tasks (data/by_task_type/*.jsonl)
    Highest possible fidelity: the validator ships the miner the same
    ``[:1000]`` document it built ground truth from, so replaying the captured
    window reproduces the real target.

survey_tagging  -  real captured questions and comments, synthetic choices
    Ground truth is the survey's literal ``selected_choices``, which is never
    transmitted to miners and therefore absent from our logs. We synthesize a
    plausible choice list per response. Absolute numbers are therefore
    indicative only; the ranking between strategies is still meaningful because
    every strategy faces the same synthetic choices.

skill_generation  -  synthetic skill documents
    The task type shipped 2026-07-10, after our last capture. Ground truth is
    reproducible from the same 1000 characters the miner receives, so the
    mechanism is faithful even though the documents are generated.
"""

from __future__ import annotations

import json
import os
import random
from typing import List, Optional, Sequence

from bench.harness import Case

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURED = os.path.join(ROOT, "data", "by_task_type")
READYAI = os.path.join(ROOT, "data", "readyai")

# Validator window parameters (ConversationTaggingTaskBundle._split_conversation_in_windows)
WINDOW_SIZE = 10
WINDOW_OVERLAP = 2
MAX_CONVO_LINES = 300  # env MAX_CONVO_LINES default


def split_overlap(lines: Sequence, size: int = WINDOW_SIZE, overlap: int = WINDOW_OVERLAP) -> List[list]:
    """Mirror Utils.split_overlap_array."""
    out = []
    i = 0
    while i < len(lines):
        out.append(list(lines[i : i + size]))
        i += size - overlap
    return out


def conversation_cases(
    n: int = 40, seed: int = 7, min_lines: int = 40, windows_per_convo: int = 1
) -> List[Case]:
    """Full ReadyAI conversations, split into validator-shaped windows."""
    import pandas as pd

    path = os.path.join(READYAI, "conversations_train.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing - run: python bench/fetch_corpus.py"
        )
    df = pd.read_parquet(path)
    rng = random.Random(seed)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)

    speakers_seen: dict = {}

    def speaker_idx(convo_key: str, speaker: str) -> int:
        key = (convo_key, speaker)
        if key not in speakers_seen:
            speakers_seen[key] = len({k for k in speakers_seen if k[0] == convo_key})
        return speakers_seen[key]

    cases: List[Case] = []
    for i in idxs:
        if len(cases) >= n:
            break
        row = df.iloc[i]
        transcript = row["transcript"]
        if transcript is None or len(transcript) < min_lines:
            continue
        guid = str(row["c_guid"])
        lines = [
            (speaker_idx(guid, e["speaker"]), str(e["text"]).strip())
            for e in transcript
            if str(e.get("text", "")).strip()
        ][:MAX_CONVO_LINES]
        if len(lines) < min_lines:
            continue
        windows = split_overlap(lines)
        # Skip the first and last window: intros and outros are boilerplate
        # ("welcome to the show", "thanks for listening") and score unlike the body.
        body = windows[1:-1] or windows
        for w in rng.sample(body, min(windows_per_convo, len(body))):
            cases.append(Case(kind="conversation_tagging", guid=f"{guid}-w{windows.index(w)}", full_lines=lines, window_lines=w))
            if len(cases) >= n:
                break
    return cases


async def add_synthetic_enrichment(cases: List[Case], per_case: int = 2, model: str = "gpt-5.2") -> None:
    """Attach simulated enrichment lines to conversation cases, in place.

    Conversation tasks have carried enrichment since 2026-06-12: the validator
    web-searches queries derived from the conversation and passes the resulting
    title+snippet pairs to BOTH its own ground-truth generator and the miner
    (ConversationTaggingTaskBundle._build_enrichment_lines_and_tags). Our corpus
    predates that, so without this the bench measures a task shape that is no
    longer sent.

    Fidelity limit, stated plainly: these lines are model-written imitations of
    search results, not real ones. What they reproduce faithfully is the
    *mechanism* - identical lines reaching ground truth and miner - which is
    what makes enrichment reproducible for a miner in the first place. Absolute
    scores under synthetic enrichment should not be compared to scores without
    it; only strategies within one setting are comparable.
    """
    import asyncio

    from sn33 import llm

    async def one(case: Case) -> None:
        text = "\n".join(t for _, t in case.window_lines[:10])
        prompt = (
            f"Write {per_case} realistic web search results that a researcher would find "
            f"when searching for the topics discussed in this excerpt.\n\n"
            f"Format each as: TITLE\\nSNIPPET (one or two sentences).\n"
            f"Separate results with a blank line. No commentary.\n\nExcerpt:\n{text[:3000]}"
        )
        raw = await llm.chat(prompt, model=model, timeout=90, temperature=None, use_cache=True, salt="enrich")
        if not raw:
            return
        blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
        case.enrichment = [b[:1000] for b in blocks[:per_case]]

    await asyncio.gather(*[one(c) for c in cases])


def captured_cases(kind: str, n: int = 30, seed: int = 7) -> List[Case]:
    """Replay real mainnet tasks for webpage / NER."""
    path = os.path.join(CAPTURED, f"{kind}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # Dedupe: the validator sends 5 identical copies of each webpage/NER task.
    seen = set()
    uniq = []
    for r in rows:
        window = (r.get("task_raw", {}).get("input", {}).get("data", {}) or {}).get("window") or []
        if not window:
            continue
        doc = window[0][1] if len(window[0]) >= 2 else ""
        if not doc.strip():
            continue
        key = doc[:400]
        if key in seen:
            continue
        seen.add(key)
        uniq.append((doc, [w[1] for w in window[1:] if len(w) >= 2 and w[1].strip()], r))

    rng = random.Random(seed)
    rng.shuffle(uniq)
    cases = []
    for doc, enrich, r in uniq[:n]:
        cats = r.get("task_raw", {}).get("input", {}).get("input_categories") or []
        cases.append(
            Case(
                kind=kind,
                guid=r.get("task_raw", {}).get("input", {}).get("guid", "cap"),
                document=doc,
                enrichment=enrich,
                coding=bool(cats and "coding" in cats),
                reference_tags=(r.get("result") or {}).get("tags") or [],
            )
        )
    return cases


def survey_cases(n: int = 20, seed: int = 7, choices_path: Optional[str] = None) -> List[Case]:
    """Real question/comment pairs; ground-truth choices loaded from a side file.

    ``choices_path`` is a JSON map ``{comment: [selected choices]}`` produced by
    ``bench/make_survey_choices.py``. Without it the case carries no ground
    truth and the runner will skip it rather than silently invent one.
    """
    path = os.path.join(CAPTURED, "survey_tagging.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    synthetic = {}
    if choices_path and os.path.exists(choices_path):
        synthetic = json.load(open(choices_path))

    seen = set()
    cases = []
    rng = random.Random(seed)
    rng.shuffle(rows)
    for r in rows:
        d = r.get("task_raw", {}).get("input", {}).get("data", {}) or {}
        q, cmt = d.get("survey_question") or "", d.get("comment") or ""
        if not cmt.strip() or cmt in seen:
            continue
        seen.add(cmt)
        cases.append(
            Case(
                kind="survey_tagging",
                guid=f"survey-{len(cases)}",
                question=q,
                comment=cmt,
                reference_tags=synthetic.get(cmt, []),
            )
        )
        if len(cases) >= n:
            break
    return cases


def skill_cases(n: int = 20, path: Optional[str] = None) -> List[Case]:
    """Skill-markdown documents from a generated corpus file."""
    path = path or os.path.join(ROOT, "data", "skills_corpus.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing - run: python bench/make_skill_corpus.py")
    cases = []
    for line in open(path):
        if not line.strip():
            continue
        rec = json.loads(line)
        md = rec["skill_markdown"][:1000]  # SkillGenerationTaskBundle.py:157
        cases.append(Case(kind="skill_generation", guid=rec.get("name", f"skill-{len(cases)}"), document=md))
        if len(cases) >= n:
            break
    return cases
