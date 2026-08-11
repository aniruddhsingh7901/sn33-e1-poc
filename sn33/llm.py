"""Async LLM access for the miner and the bench.

Differences from ``conversationgenome.llm.llm_openai`` that matter:

* **Batched embeddings.** The upstream client issues one HTTPS request per tag
  (``LlmLib.get_vector_embeddings_set``). The OpenAI embeddings endpoint accepts
  a list, so 30 tags cost one round trip instead of 30.
* **Async.** Every call is awaitable, so per-enrichment-line work runs
  concurrently inside the synapse budget instead of serially.
* **Bounded.** Every call takes a timeout and never raises past the wrapper -
  callers get ``None`` and fall back.
* **Cached.** Bench replays hit sqlite instead of the API.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence

from openai import AsyncOpenAI

from sn33 import cache

# The validator hardcodes this; matching it is the whole game for tag ranking.
# conversationgenome/llm/llm_openai.py:15
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536

_client: Optional[AsyncOpenAI] = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        # Keep connections warm; a cold TLS handshake is ~100-200ms we cannot spare.
        _client = AsyncOpenAI(api_key=api_key, max_retries=0)
    return _client


class Usage:
    """Per-process counters so the bench can report cost and the miner can log it."""

    chat_calls = 0
    chat_prompt_tokens = 0
    chat_completion_tokens = 0
    embed_calls = 0
    embed_tokens = 0
    cache_hits = 0

    @classmethod
    def reset(cls) -> None:
        cls.chat_calls = cls.chat_prompt_tokens = cls.chat_completion_tokens = 0
        cls.embed_calls = cls.embed_tokens = cls.cache_hits = 0

    @classmethod
    def snapshot(cls) -> dict:
        return {
            "chat_calls": cls.chat_calls,
            "chat_prompt_tokens": cls.chat_prompt_tokens,
            "chat_completion_tokens": cls.chat_completion_tokens,
            "embed_calls": cls.embed_calls,
            "embed_tokens": cls.embed_tokens,
            "cache_hits": cls.cache_hits,
        }


async def chat(
    prompt: str,
    model: str,
    timeout: float = 8.0,
    temperature: Optional[float] = 0.0,
    use_cache: bool = False,
    max_completion_tokens: Optional[int] = None,
    attempts: int = 1,
    salt: str = "",
) -> Optional[str]:
    """One chat completion. Returns None on any failure (never raises).

    ``salt`` separates cache namespaces. The bench needs it: the validator's
    ground-truth call and a miner's call can render the *same* prompt, and
    serving both from one cache entry would fake perfect agreement between
    miner and ground truth. In reality they are two independent samples.
    """
    ck = cache.key_for("chat", model, prompt, temperature, max_completion_tokens, salt)
    if use_cache:
        hit = cache.get(ck)
        if hit is not None:
            Usage.cache_hits += 1
            return hit

    params: Dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        params["temperature"] = temperature
    if max_completion_tokens is not None:
        params["max_completion_tokens"] = max_completion_tokens
    # Paid latency lever (2026-08-10): OpenAI priority processing generates
    # ~30-40% faster for ~2x token price. Our LLM calls are ~85% of the 8-11s
    # task time and a truncated task (source=pool) scores ~0.2-0.3 vs ~0.65, so
    # buying speed converts directly into score floor. Env-gated; if the API
    # rejects the tier (model/account ineligible) the call retries WITHOUT it -
    # a dropped call is a dropped task and never acceptable.
    tier = os.environ.get("SN33_SERVICE_TIER", "").strip()
    if tier:
        params["service_tier"] = tier

    last_exc = None
    attempt = 0
    total = max(1, attempts)
    while attempt < total:
        try:
            resp = await asyncio.wait_for(
                client().chat.completions.create(**params), timeout=timeout
            )
            text = resp.choices[0].message.content or ""
            try:
                Usage.chat_calls += 1
                Usage.chat_prompt_tokens += resp.usage.prompt_tokens
                Usage.chat_completion_tokens += resp.usage.completion_tokens
            except Exception:
                pass
            # Never cache an empty completion. A failed call returns None and is
            # not cached, but a *successful* call that returns "" would be, and
            # the entry is permanent - every later run replays the empty answer
            # from cache and looks like a miner producing no tags.
            if use_cache and text:
                cache.put(ck, text)
            return text
        except asyncio.TimeoutError:
            return None  # deadline is sacred: do not burn budget retrying
        except Exception as e:  # noqa: BLE001 - deliberately total
            last_exc = e
            # Tier rejection retries immediately WITHOUT the tier and does not
            # consume an attempt (different failure class than a flaky call).
            if params.pop("service_tier", None) is not None and "service_tier" in str(e):
                continue
            attempt += 1
            if attempt < total:
                await asyncio.sleep(0.15 * attempt)
    if last_exc is not None and os.environ.get("SN33_DEBUG"):
        print(f"[sn33.llm] chat failed model={model}: {type(last_exc).__name__}: {last_exc}")
    return None


async def embed(
    texts: Sequence[str],
    timeout: float = 6.0,
    use_cache: bool = True,
    batch_size: int = 256,
) -> Dict[str, List[float]]:
    """Embed many strings in as few HTTP calls as possible.

    Returns ``{text: vector}``; texts that fail are simply absent from the dict,
    so callers must treat a missing key as "unknown", never as a zero vector.
    """
    out: Dict[str, List[float]] = {}
    todo: List[str] = []
    seen = set()
    for t in texts:
        if not t or not t.strip() or t in seen:
            continue
        seen.add(t)
        if use_cache:
            hit = cache.get(cache.key_for("emb", EMBED_MODEL, EMBED_DIMS, t))
            if hit is not None:
                Usage.cache_hits += 1
                out[t] = hit
                continue
        todo.append(t)

    if not todo:
        return out

    async def _one_batch(batch: List[str]) -> None:
        try:
            resp = await asyncio.wait_for(
                client().embeddings.create(
                    input=[b.replace("\n", " ") for b in batch],
                    model=EMBED_MODEL,
                    dimensions=EMBED_DIMS,
                ),
                timeout=timeout,
            )
        except Exception as e:  # noqa: BLE001
            if os.environ.get("SN33_DEBUG"):
                print(f"[sn33.llm] embed failed: {type(e).__name__}: {e}")
            return
        try:
            Usage.embed_calls += 1
            Usage.embed_tokens += resp.usage.total_tokens
        except Exception:
            pass
        for item in resp.data:
            text = batch[item.index]
            out[text] = item.embedding
            if use_cache:
                cache.put(cache.key_for("emb", EMBED_MODEL, EMBED_DIMS, text), item.embedding)

    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    await asyncio.gather(*[_one_batch(b) for b in batches])
    return out


async def timed(coro, label: str, sink: Optional[dict] = None):
    """Await ``coro`` recording wall time into ``sink[label]``."""
    t0 = time.perf_counter()
    try:
        return await coro
    finally:
        if sink is not None:
            sink[label] = round(time.perf_counter() - t0, 3)
