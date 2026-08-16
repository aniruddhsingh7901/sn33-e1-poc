"""Candidate miner strategies, all scored against the same ground truth.

Three families:

``stock``
    What an unmodified clone of the repo does: the upstream ``.j2`` prompt,
    output submitted as-is. The floor everyone else must beat.

``prod``
    This repo's current working tree - the tuned prompts that were shipped to
    the Hetzner miners in May (v6 conversation, v4 webpage, ...). Reads the
    live ``conversationgenome/llm/prompts/*.j2`` files, so it tracks whatever
    is actually deployed.

``replica*``
    ``sn33.pipeline`` - rebuild the validator's ground truth from the input we
    were given, then submit the candidates closest to its centroid. Variants
    differ only by config, so the sweep measures one knob at a time.

Every strategy returns a plain tag list; the harness handles validation,
embedding and scoring identically for all of them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from jinja2 import Environment, FileSystemLoader

from bench.harness import Case, convo_xml
from sn33 import llm, pipeline, prompts
from sn33.replica import clean_like_validator
from sn33.tags import normalize_all, parse_tag_list

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prod_env = Environment(loader=FileSystemLoader(os.path.join(ROOT, "conversationgenome", "llm", "prompts")))


def prod_prompt(name: str, **kwargs) -> str:
    """Render the prompt file as currently deployed in the working tree."""
    return _prod_env.get_template(name).render(**kwargs)


@dataclass
class Strategy:
    name: str
    run: Callable          # async (case, gt) -> List[str]
    note: str = ""
    needs_gt: bool = False  # oracle variants peek at ground truth on purpose


# --------------------------------------------------------------------------
# stock / prod - single call, submit whatever comes back
# --------------------------------------------------------------------------

def _prompt_for(case: Case, family: str) -> Optional[str]:
    render = prompts.upstream if family == "stock" else prod_prompt
    coding = case.coding
    if case.kind == "conversation_tagging":
        tpl = "conversation_to_metadata_coding.j2" if coding else "conversation_to_metadata.j2"
        return render(tpl, conversation_to_analyze=convo_xml(case.window_lines))
    if case.kind == "webpage_metadata_generation":
        tpl = "website_to_metadata_coding.j2" if coding else "website_to_metadata.j2"
        return render(tpl, website_content=case.document)
    if case.kind == "named_entities_extraction":
        return render("raw_transcript_to_named_entities.j2", raw_transcript=case.document)
    if case.kind == "skill_generation":
        return render("skill_to_metadata.j2", skill_markdown=case.document)
    if case.kind == "survey_tagging":
        return render("survey_tag.j2", survey_question=case.question, free_form_comment=case.comment)
    return None


async def _single_call(case: Case, family: str, model: str, use_cache: bool = True) -> List[str]:
    """Reproduce the upstream miner: one prompt, one parse, submit.

    Enrichment is folded in the same way the task classes do it (per-line call
    then a combine call) so the comparison is like-for-like.
    """
    prompt = _prompt_for(case, family)
    if prompt is None:
        return []
    render = prompts.upstream if family == "stock" else prod_prompt
    raw = await llm.chat(prompt, model=model, timeout=90, use_cache=use_cache)
    main = clean_like_validator(raw)
    if not main:
        return []

    if not case.enrichment:
        return main

    sets = [main]
    for line in case.enrichment:
        if case.kind == "named_entities_extraction":
            p = render("enrichment_to_named_entities.j2", enrichment_content=line)
        else:
            tpl = "enrichment_to_metadata_coding.j2" if case.coding else "enrichment_to_metadata.j2"
            p = render(tpl, enrichment_content=line)
        got = clean_like_validator(await llm.chat(p, model=model, timeout=90, use_cache=use_cache))
        if got:
            sets.append(got)

    if len(sets) == 1:
        return main
    combined = clean_like_validator(
        await llm.chat(render("combine_named_entities_prompt.j2",
                              entities_str="".join(f"<set{i}>{', '.join(s)}</set{i}>\n" for i, s in enumerate(sets))),
                       model=model, timeout=90, use_cache=use_cache)
    )
    return combined or main


def stock(model: str = "gpt-5.2") -> Strategy:
    async def run(case: Case, gt=None) -> List[str]:
        return await _single_call(case, "stock", model)

    return Strategy("stock", run, "upstream prompts, submitted as-is")


def prod(model: str = "gpt-5.2") -> Strategy:
    async def run(case: Case, gt=None) -> List[str]:
        return await _single_call(case, "prod", model)

    return Strategy("prod_current", run, "working-tree tuned prompts (May config)")


def prod_capped(model: str = "gpt-5.2", cap: int = 19) -> Strategy:
    """Current prompts plus the two zero-cost hygiene fixes.

    Isolates how much of the gap is just "stop submitting 20+ tags" and
    "normalize before submitting", with no change to the prompt at all.
    """

    async def run(case: Case, gt=None) -> List[str]:
        return normalize_all(await _single_call(case, "prod", model))[:cap]

    return Strategy(f"prod+normalize+cap{cap}", run, "prod prompts, normalized, capped")


# --------------------------------------------------------------------------
# replica - the new pipeline
# --------------------------------------------------------------------------

def _bench_cfg(**overrides) -> dict:
    """Bench runs without a deadline: we are measuring tag quality, not latency.

    Latency is measured separately by bench/timing.py against the real 12s
    budget, because mixing the two would let a slow-but-good strategy look good
    here and time out in production.
    """
    cfg_kwargs = dict(use_cache=True, use_local=False, deadline_s=600.0, call_timeout_s=120.0)
    cfg_kwargs.update(overrides)
    return cfg_kwargs


async def _run_pipeline(case: Case, cfg_kwargs: dict):
    cfg = pipeline.Config(**cfg_kwargs)
    return await pipeline.mine(
        case.kind,
        window=case.miner_window(),
        enrichment=case.enrichment,
        question=case.question,
        comment=case.comment,
        coding=case.coding,
        cfg=cfg,
    )


def replica(name: str = "replica", **overrides) -> Strategy:
    cfg_kwargs = _bench_cfg(**overrides)

    async def run(case: Case, gt=None) -> List[str]:
        res = await _run_pipeline(case, cfg_kwargs)
        return res.tags

    note = ", ".join(
        f"{k}={v}" for k, v in overrides.items()
        if k not in ("use_cache", "use_local", "deadline_s", "call_timeout_s")
    )
    return Strategy(name, run, note)


def oracle(name: str = "ORACLE_selection", k: int = 21, **overrides) -> Strategy:
    """Upper bound: same candidate pool, ranked against the TRUE target.

    This is not deployable - it reads ground truth. It exists to split the
    remaining gap into two questions:

      oracle - replica  =  how much better selection could get us
      1.0    - oracle   =  how much the candidate pool itself is missing

    If oracle sits close to replica, better ranking is a dead end and the
    candidate generator is the thing to improve.
    """
    cfg_kwargs = _bench_cfg(**overrides)

    async def run(case: Case, gt=None) -> List[str]:
        res = await _run_pipeline(case, cfg_kwargs)
        if gt is None or gt.target is None or not res.candidates:
            return res.tags
        vectors = dict(res.vectors)
        missing = [t for t in res.candidates if t not in vectors]
        if missing:
            vectors.update(await llm.embed(missing, use_cache=True))
        best = scoring_oracle(gt, res.candidates, vectors, case.kind, k)
        return best or res.tags

    return Strategy(name, run, f"upper bound, k<={k}", needs_gt=True)


def scoring_oracle(gt, candidates: List[str], vectors: dict, kind: str, k: int) -> List[str]:
    from sn33 import scoring

    penalties = kind != "named_entities_extraction"
    min_tags = 3 if kind in ("conversation_tagging", "webpage_metadata_generation", "skill_generation") else 1
    best = scoring.oracle_best(
        gt.tags, gt.target, candidates, vectors, penalties=penalties, min_tags=min_tags, max_k=k
    )
    return best.get("tags", [])
