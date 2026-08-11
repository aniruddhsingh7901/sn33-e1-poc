"""Bridge between the upstream task objects and ``sn33.pipeline``.

Design constraint: the upstream repo must stay mergeable. ``git pull`` happens
often on this subnet (three mandatory upgrades in the last quarter), so the
integration is one hook in ``MinerLib.do_mining`` plus this file, rather than
edits spread across the five task classes.

Behaviour is fail-open: anything this layer cannot handle falls through to the
stock ``task.mine()`` implementation.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

from sn33 import pipeline
from sn33.pipeline import Config

try:
    import bittensor as bt
except Exception:  # pragma: no cover - miner always has bittensor
    bt = None


def _log(msg: str) -> None:
    if bt is not None:
        bt.logging.info(f"[sn33] {msg}")
    else:
        print(f"[sn33] {msg}")


def config_from_env() -> Config:
    """All tuning lives in the environment; no secrets or knobs in code."""
    def _f(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    def _i(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    cfg = Config(
        gt_model=os.environ.get("SN33_GT_MODEL", os.environ.get("OPENAI_MODEL", "gpt-5.2")),
        pool_model=os.environ.get("SN33_POOL_MODEL", os.environ.get("OPENAI_MODEL", "gpt-5.2")),
        deadline_s=_f("SN33_DEADLINE_S", 8.0),
        call_timeout_s=_f("SN33_CALL_TIMEOUT_S", 6.5),
        combine=os.environ.get("SN33_COMBINE", "llm"),
        pool_size=_i("SN33_POOL_SIZE", 40),
        dedup_threshold=_f("SN33_DEDUP", 0.93),
        use_pool=os.environ.get("SN33_USE_POOL", "1") != "0",
        use_local=os.environ.get("SN33_USE_LOCAL", "1") != "0",
    )
    if os.environ.get("SN33_TARGET_TAGS"):
        cfg.target_tags = _i("SN33_TARGET_TAGS", 12)
    if os.environ.get("SN33_INSURANCE"):
        cfg.insurance = _i("SN33_INSURANCE", 2)
    # Deeper per-enrichment-line extraction, sourcing more high-cosine unique
    # candidates for the wider 18-tag lists. Env-toggled so it can be turned off
    # without a redeploy if the extra ~1.2-1.6s pushes the source=pool
    # (truncation) rate up - that failure costs ~0.35 per affected task.
    if os.environ.get("SN33_DEEP_ENRICHMENT", "0") != "0":
        cfg.use_deep_enrichment = True
        cfg.deep_enrichment_tags = _i("SN33_DEEP_TAGS", 30)
    if os.environ.get("SN33_THEME_TAGS", "0") != "0":
        cfg.use_theme_tags = True
    if os.environ.get("SN33_TRANSLATE", "1") == "0":
        cfg.translate_non_english = False   # on by default; env can disable
    # Enrichment-first aiming (conversation) and NER composites. Both OFF by
    # default: each is the single variable of its own mainnet A/B (>=25 scored
    # tasks per arm before judging - SE of a task mean is ~0.016 at n=26).
    if os.environ.get("SN33_ENRICHMENT_FIRST", "0") != "0":
        cfg.enrichment_first = True
        cfg.enrichment_demote = _f("SN33_ENRICH_DEMOTE", 0.90)
        cfg.enrichment_anchor_cos = _f("SN33_ENRICH_ANCHOR_COS", 0.35)
    if os.environ.get("SN33_NER_COMBOS", "0") != "0":
        cfg.ner_combos = True
    if os.environ.get("SN33_ENRICHMENT_FIRST_WEBPAGE", "0") != "0":
        cfg.enrichment_first_webpage = True
    # Per-line quota + conditional demotion (2026-08-10 forensics). Each is its
    # own single-variable A/B; both no-op unless enrichment-first is active.
    if os.environ.get("SN33_LINE_QUOTA"):
        cfg.enrichment_line_quota = _i("SN33_LINE_QUOTA", 0)
    if os.environ.get("SN33_DEMOTE_CONDITIONAL", "0") != "0":
        cfg.demote_conditional = True
        cfg.demote_agree_cut = _f("SN33_DEMOTE_AGREE_CUT", 0.18)
    if os.environ.get("SN33_HEAD_CAP"):
        cfg.head_cap = _i("SN33_HEAD_CAP", 0)
    return cfg


def enabled() -> bool:
    return os.environ.get("SN33_ENABLED", "1") != "0"


def _window(task) -> List[Tuple[int, str]]:
    data = getattr(getattr(task, "input", None), "data", None)
    win = getattr(data, "window", None) or []
    return [tuple(w) for w in win if isinstance(w, (list, tuple)) and len(w) >= 2]


def _enrichment(task) -> Optional[List[str]]:
    """Conversation tasks carry enrichment in their own field, not the window."""
    data = getattr(getattr(task, "input", None), "data", None)
    lines = getattr(data, "enrichment_lines", None)
    if lines is None:
        return None
    return [str(l[1]) for l in lines if isinstance(l, (list, tuple)) and len(l) >= 2 and str(l[1]).strip()]


async def mine(task) -> Optional[dict]:
    """Return the miner payload for ``task``, or None to fall back to upstream."""
    kind = getattr(task, "type", None)
    if kind not in pipeline.TASK_PROFILE:
        return None

    data = getattr(getattr(task, "input", None), "data", None)
    categories = getattr(getattr(task, "input", None), "input_categories", None) or []
    t0 = time.perf_counter()

    result = await pipeline.mine(
        kind,
        window=_window(task),
        enrichment=_enrichment(task),
        question=getattr(data, "survey_question", "") or "",
        comment=getattr(data, "comment", "") or "",
        coding="coding" in categories,
        cfg=config_from_env(),
    )

    profile = pipeline.TASK_PROFILE[kind]
    if len(result.tags) < profile["min_tags"]:
        # Below the floor the score is a hard zero either way; let the stock
        # path have a try rather than submitting something guaranteed worthless.
        _log(
            f"{kind}: only {len(result.tags)} tags after {result.elapsed:.1f}s "
            f"(source={result.source}) - falling back to upstream miner"
        )
        return None

    _log(
        f"{kind}: {len(result.tags)} tags in {result.elapsed:.2f}s "
        f"source={result.source} degraded={result.degraded} timings={result.timings}"
    )
    # Vectors are discarded by the validator (it re-embeds every tag itself in
    # format_results), so computing them here would be pure latency.
    return {"tags": result.tags, "vectors": None}
