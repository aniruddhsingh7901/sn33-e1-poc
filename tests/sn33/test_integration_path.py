"""End-to-end through the code the miner actually runs.

The bench calls ``sn33.pipeline`` directly, which leaves one gap: the real
request arrives as a raw dict on a synapse and reaches us via
``parse_task`` -> ``MinerLib.do_mining`` -> ``sn33.adapter``. This exercises
that path with **real captured validator payloads** so a field-name change
upstream fails here rather than in production.

Network is stubbed: this asserts wiring and output shape, not tag quality.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from conversationgenome.miner.MinerLib import MinerLib
from conversationgenome.task.task_factory import try_parse_task
from sn33 import llm, pipeline
from sn33.tags import survives_validation

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAPTURED = os.path.join(ROOT, "data", "by_task_type")

TASK_FILES = {
    "conversation_tagging": "conversation_tagging.jsonl",
    "webpage_metadata_generation": "webpage_metadata_generation.jsonl",
    "named_entities_extraction": "named_entities_extraction.jsonl",
    "survey_tagging": "survey_tagging.jsonl",
}


def _first_task(kind: str):
    path = os.path.join(CAPTURED, TASK_FILES[kind])
    if not os.path.exists(path):
        pytest.skip(f"no capture for {kind}")
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line).get("task_raw")
            if raw:
                return raw
    pytest.skip(f"no usable payload in {path}")


@pytest.fixture
def stub_llm(monkeypatch):
    """Deterministic, fast, offline stand-ins for both API calls."""

    async def chat(prompt, model, **kw):
        return "mortgage rates, housing market, rental yields, property data, urban planning, construction"

    async def embed(texts, **kw):
        # Distinct but similar unit-ish vectors so ranking has something to do.
        out = {}
        for i, t in enumerate(texts):
            out[t] = [1.0, 0.1 * (i % 5), 0.05 * (i % 3)]
        return out

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "embed", embed)


@pytest.mark.parametrize("kind", list(TASK_FILES))
def test_real_payload_through_minerlib(kind, stub_llm, monkeypatch):
    monkeypatch.setenv("SN33_ENABLED", "1")
    monkeypatch.setenv("SN33_DEADLINE_S", "8.0")

    raw = _first_task(kind)
    task = try_parse_task(raw)
    assert task is not None, f"upstream parse_task rejected a captured {kind} payload"
    assert task.type == kind

    result = asyncio.run(MinerLib().do_mining(task=task))

    assert isinstance(result, dict), "miner must return a dict payload"
    assert "tags" in result and isinstance(result["tags"], list)

    profile = pipeline.TASK_PROFILE[kind]
    assert len(result["tags"]) >= profile["min_tags"], f"{kind}: {result['tags']}"
    assert len(result["tags"]) <= pipeline.MAX_TAGS
    assert len(result["tags"]) == len(set(result["tags"]))
    for tag in result["tags"]:
        assert survives_validation(tag), f"{tag!r} would be deleted by the validator"


def test_layer_can_be_disabled(monkeypatch, stub_llm):
    """SN33_ENABLED=0 must hand control back to the stock miner."""
    monkeypatch.setenv("SN33_ENABLED", "0")
    from sn33 import adapter

    assert adapter.enabled() is False


def test_adapter_falls_back_when_below_floor(monkeypatch):
    """Too few tags -> return None so the stock miner runs instead of scoring 0."""
    from sn33 import adapter

    async def empty(*a, **k):
        return pipeline.Result(tags=[], source="none")

    monkeypatch.setattr(pipeline, "mine", empty)
    raw = _first_task("conversation_tagging")
    task = try_parse_task(raw)
    assert asyncio.run(adapter.mine(task)) is None
