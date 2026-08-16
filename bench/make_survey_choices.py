#!/usr/bin/env python3
"""Synthesize ground truth for survey tasks.

Survey ground truth is the literal ``selected_choices`` list from the survey
record (SurveyTaggingTaskBundle.py:68) - no LLM involved. Miners never receive
it, so it is absent from our captured logs and there is nothing to replay.

What we do have is 53 real captured question/comment pairs, all from the same
Spanish banking study ("¿Por qué razones prefiere ese banco?", multiple choice,
spontaneous). This script asks a model to act as the survey coder and produce
the answer options such a respondent would have been coded against.

Fidelity: the *mechanism* is exact (ground truth is a short list of literal
option strings, scored by cosine against their centroid, penalties on,
min_tags=1). The *content* is synthetic, so absolute scores are indicative;
what the bench measures reliably is whether a strategy's tags survive
validation and land near option-style Spanish phrases.

    python bench/make_survey_choices.py --n 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import corpus
from sn33 import llm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "survey_choices.json")

PROMPT = """\
You are the coder for a market research survey.

Question: {question}

A respondent gave this free-text answer: "{comment}"

Multiple-choice answer options are pre-coded short noun phrases in the survey's
own language. List the option labels this respondent would have been coded
under - between 1 and 4 of them, most certain first.

Return ONLY a comma-delimited list of option labels. No commentary.
"""


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default="gpt-5.2")
    args = ap.parse_args()

    cases = corpus.survey_cases(n=args.n)
    print(f"generating coded choices for {len(cases)} captured survey responses")

    async def one(case):
        raw = await llm.chat(
            PROMPT.format(question=case.question, comment=case.comment),
            model=args.model,
            timeout=90,
            temperature=None,
            use_cache=True,
            salt="survey_gt",
        )
        if not raw:
            return case.comment, []
        # Ground truth keeps the survey's literal strings - accents and all.
        return case.comment, [c.strip() for c in raw.split(",") if c.strip()][:4]

    pairs = await asyncio.gather(*[one(c) for c in cases])
    payload = {comment: choices for comment, choices in pairs if choices}
    with open(OUT, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT} with {len(payload)} entries")
    for comment, choices in list(payload.items())[:5]:
        print(f"  {comment[:45]!r:50s} -> {choices}")


if __name__ == "__main__":
    asyncio.run(main())
