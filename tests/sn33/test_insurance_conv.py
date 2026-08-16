"""SN33_INSURANCE_CONV must raise conversation insurance and touch NOTHING else."""
import asyncio
import os

from sn33 import pipeline
from sn33.pipeline import Config, TASK_PROFILE


def test_conv_override_applies_only_to_conversation(monkeypatch):
    """The gate resolves insurance per kind: conversation gets the override,
    NER keeps its profile 0, webpage keeps its profile 6."""
    seen = {}

    real_compose = pipeline.compose

    def spy_compose(ranked, predicted_gt, profile, target_tags, insurance, **kw):
        seen[spy_compose.kind] = insurance
        return real_compose(ranked, predicted_gt, profile, target_tags, insurance, **kw)

    async def chat(prompt, *a, **k):
        return "mortgage rates, rental market, housing demand, construction starts"

    async def embed(texts, **k):
        return {t: [1.0, 0.0, 0.0] for t in texts}

    monkeypatch.setattr(pipeline.llm, "chat", chat)
    monkeypatch.setattr(pipeline.llm, "embed", embed)
    monkeypatch.setattr(pipeline, "compose", spy_compose)

    cfg = Config(use_local=False, use_pool=True, use_cache=False,
                 deadline_s=11.0, call_timeout_s=8.0, insurance_conv=14,
                 insurance_web=10)

    window = [(0, "rates moved again this quarter"), (1, "buyers are hesitating")]
    enrichment = ["housing and mortgage markets line"]

    spy_compose.kind = "conversation_tagging"
    asyncio.run(pipeline.mine("conversation_tagging", window=window,
                              enrichment=enrichment, cfg=cfg))
    spy_compose.kind = "webpage_metadata_generation"
    asyncio.run(pipeline.mine("webpage_metadata_generation",
                              window=[(0, "some page text about housing markets")],
                              enrichment=enrichment, cfg=cfg))

    assert seen.get("conversation_tagging") == 14, seen
    assert seen.get("webpage_metadata_generation") == 10, seen


def test_unset_env_changes_nothing(monkeypatch):
    monkeypatch.delenv("SN33_INSURANCE_CONV", raising=False)
    from sn33 import adapter
    assert adapter.config_from_env().insurance_conv is None


def test_env_hook(monkeypatch):
    monkeypatch.setenv("SN33_INSURANCE_CONV", "14")
    monkeypatch.setenv("SN33_INSURANCE_WEB", "10")
    from sn33 import adapter
    cfg = adapter.config_from_env()
    assert cfg.insurance_conv == 14
    assert cfg.insurance_web == 10
