#!/usr/bin/env python3
"""Generate a skill-document corpus for the newest task type.

``skill_generation`` shipped 2026-07-10 (v2.35.71), after our last production
capture, so there is nothing to replay. Ground truth for it is built from the
same ``skill_markdown[:1000]`` the miner receives
(SkillGenerationTaskBundle.py:157,162), which means the *mechanism* is fully
faithful even when the documents are synthetic - the miner holds exactly the
text the target was made from, as it would in production.

The documents imitate what the subnet says it is collecting: skills for coding
agents ("Claude Code, OpenHands, Codex" harnesses), auto-tagged for RAG.

    python bench/make_skill_corpus.py --n 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sn33 import llm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "skills_corpus.jsonl")

TOPICS = [
    "parsing and writing Excel workbooks in Python",
    "debugging flaky CI pipelines in GitHub Actions",
    "writing Terraform modules for AWS VPC networking",
    "migrating a Django app from SQLite to PostgreSQL",
    "profiling memory leaks in Node.js services",
    "building retrieval-augmented generation over PDFs",
    "setting up Kubernetes horizontal pod autoscaling",
    "refactoring React class components to hooks",
    "instrumenting a Go service with OpenTelemetry",
    "writing property-based tests with Hypothesis",
    "optimizing slow PostgreSQL queries with EXPLAIN",
    "packaging a Rust CLI for multiple platforms",
    "implementing OAuth2 device flow in a CLI",
    "converting a REST API to GraphQL incrementally",
    "hardening a Dockerfile for production",
    "scraping and normalizing tabular web data",
    "designing idempotent webhook handlers",
    "setting up pre-commit hooks for a Python monorepo",
    "diagnosing TLS certificate chain failures",
    "writing a language server plugin for VS Code",
    "batch image processing with ImageMagick",
    "migrating Jenkins pipelines to GitLab CI",
    "building a CSV to Parquet conversion tool",
    "adding structured logging to a Flask app",
]

PROMPT = """\
Write an LLM agent skill document in Markdown about: {topic}

Follow the usual skill format: a title, a one-line description, a "When to use
this skill" section, and concrete step-by-step instructions with short code
snippets. Aim for roughly 1200 characters - it will be truncated at 1000.

Return ONLY the Markdown. No commentary.
"""


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--model", default="gpt-5.2")
    args = ap.parse_args()

    topics = TOPICS[: args.n]
    print(f"generating {len(topics)} skill documents with {args.model}")

    async def one(topic: str):
        md = await llm.chat(
            PROMPT.format(topic=topic), model=args.model, timeout=120,
            temperature=None, use_cache=True, salt="skillgen",
        )
        return {"name": topic, "skill_markdown": md} if md else None

    rows = [r for r in await asyncio.gather(*[one(t) for t in topics]) if r]
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT} with {len(rows)} documents")
    if rows:
        print("sample:", rows[0]["skill_markdown"][:200].replace("\n", " "))


if __name__ == "__main__":
    asyncio.run(main())
