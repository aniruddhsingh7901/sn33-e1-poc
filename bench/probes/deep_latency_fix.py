#!/usr/bin/env python3
"""Adversarial arm: harvest the deep results AFTER the combine call.

Production `replica.replicate` harvests the deep tasks BEFORE the combine call
(replica.py:251-263), so the grace sits on the critical path: combine starts
`grace` seconds later than it otherwise would, and the combine call's own
timeout, `min(call_timeout_s, deadline - elapsed)`, is `grace` seconds shorter.

The deep tasks are plain futures running concurrently on the loop, so they keep
making progress while combine is in flight. Harvesting after combine therefore
buys the same results for ~0 extra wall time. This module implements that
ordering and nothing else, so the two can be A/B'd on identical inputs.
"""

from __future__ import annotations

import asyncio
import time as _time
from typing import Dict, List, Sequence

from sn33 import llm, prompts, replica
from sn33.replica import Replica, _sampled_tags, _tags_from_prompt, local_combine


async def replicate_late(
    kind: str,
    *,
    document: str = "",
    convo_xml: str = "",
    enrichment: Sequence[str] = (),
    model: str = "gpt-5.2",
    timeout: float = 8.0,
    coding: bool = False,
    combine: str = "llm",
    use_cache: bool = False,
    samples: int = 1,
    deadline: float = 0.0,
    deep_enrichment: int = 0,
) -> Replica:
    t_start = _time.perf_counter()
    rep = Replica()
    timings: Dict[str, float] = {}

    deep_prompt = None
    if kind == "conversation_tagging":
        doc_prompt = prompts.gt_conversation(convo_xml, coding=coding)
        enrich_prompt = lambda t: prompts.gt_enrichment(t, coding=coding)  # noqa: E731
        deep_prompt = lambda t: prompts.gt_enrichment_deep(t, n=deep_enrichment, coding=coding)  # noqa: E731
    elif kind == "webpage_metadata_generation":
        doc_prompt = prompts.gt_website(document, coding=coding)
        enrich_prompt = lambda t: prompts.gt_enrichment(t, coding=coding)  # noqa: E731
        deep_prompt = lambda t: prompts.gt_enrichment_deep(t, n=deep_enrichment, coding=coding)  # noqa: E731
    elif kind == "named_entities_extraction":
        doc_prompt = prompts.gt_transcript_ner(document)
        enrich_prompt = prompts.gt_enrichment_ner
    elif kind == "skill_generation":
        doc_prompt = prompts.gt_skill(document)
        enrich_prompt = prompts.gt_enrichment
        deep_prompt = lambda t: prompts.gt_enrichment_deep(t, n=deep_enrichment)  # noqa: E731
    else:
        raise ValueError(f"no ground-truth replica for task kind {kind!r}")

    jobs = [llm.timed(_sampled_tags(doc_prompt, model, timeout, use_cache, samples), "doc", timings)]
    for i, text in enumerate(enrichment):
        jobs.append(llm.timed(_tags_from_prompt(enrich_prompt(text), model, timeout, use_cache),
                              f"enrich{i}", timings))

    deep_tasks: List[asyncio.Future] = []
    if deep_enrichment > 0 and deep_prompt is not None:
        deep_tasks = [
            asyncio.ensure_future(
                llm.timed(_tags_from_prompt(deep_prompt(text), model, timeout, use_cache, salt="deep"),
                          f"deep{i}", timings)
            )
            for i, text in enumerate(enrichment)
        ]

    fanout_budget = max(0.5, deadline - 2.0) if deadline else timeout

    def harvest(grace: float) -> None:
        seen = set(rep.deep_tags)
        for t in deep_tasks:
            if not t.done() or t.cancelled() or t.exception():
                t.cancel()
                continue
            for tag in t.result() or []:
                if tag not in seen:
                    seen.add(tag)
                    rep.deep_tags.append(tag)

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*jobs, return_exceptions=True), timeout=fanout_budget
        )
    except asyncio.TimeoutError:
        for t in deep_tasks:
            t.cancel()
        rep.degraded = True
        rep.timings = timings
        return rep

    clean: List[List[str]] = []
    for r in results:
        clean.append([] if isinstance(r, BaseException) or not r else r)
    rep.doc_tags = clean[0]
    rep.enrichment_tags = [c for c in clean[1:] if c]
    rep.degraded = not rep.doc_tags
    rep.timings = timings

    sets = rep.all_sets
    if not sets:
        harvest(0.0)
        return rep

    must_combine = kind != "conversation_tagging" or len(sets) > 1
    if not must_combine or combine == "none":
        rep.tags = sets[0]
        harvest(0.0)
        return rep
    if combine == "local":
        rep.tags = local_combine(sets)
        harvest(0.0)
        return rep

    if deadline:
        left = deadline - (_time.perf_counter() - t_start)
        if left < 1.2:
            rep.tags = local_combine(sets)
            rep.degraded = True
            rep.timings = timings
            harvest(0.0)
            return rep
        timeout = min(timeout, left)

    # ---- THE DIFFERENCE: combine first, deep harvested from its shadow ----
    combined = await llm.timed(
        _tags_from_prompt(prompts.gt_combine(sets), model, timeout, use_cache), "combine", timings
    )
    if combined:
        rep.tags = combined
    else:
        rep.tags = local_combine(sets)
        rep.degraded = True

    if deep_tasks:
        # Only whatever is still missing after combine already ran, and never
        # past the budget the fan-out had been allotted.
        grace = min(replica.DEEP_GRACE_S, fanout_budget - (_time.perf_counter() - t_start))
        if grace > 0 and not all(t.done() for t in deep_tasks):
            await asyncio.wait(deep_tasks, timeout=grace)
        harvest(grace)
    return rep
