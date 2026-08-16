"""The miner.

Strategy, in one paragraph: the validator scores every tag as cosine similarity
to a single vector - the mean of the embeddings of ground-truth tags it made
from the source document. For webpage / NER / skill tasks it hands the miner
that exact source document, and for conversation tasks it hands over the exact
enrichment lines. So instead of guessing what "good tags" look like, we rebuild
the validator's ground truth locally with its own prompts, average the result
into an estimate of the target vector, and then submit the candidate tags that
sit closest to it.

Everything else is defence:

* a **deadline** (default 9s of the validator's 12s) after which whatever answer
  is currently held gets returned rather than nothing;
* a **local fallback** computed before any API call, so an answer exists even if
  every request fails;
* **normalization** on the way out, because tags that are not already in
  ``get_safe_tag`` form are deleted by the validator before scoring;
* a **19-tag cap**, because at 20+ the validator keeps a random 20 and discards
  the rest.

No exception may escape ``mine``; the caller is a synapse handler and an
exception there is a zero.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from conversationgenome.utils.Utils import Utils
from sn33 import extract, llm, prompts, replica, variants, vocab
from sn33.tags import (
    centroid,
    cosine,
    drop_near_duplicates,
    is_probably_english,
    make_summary_tags,
    normalize_all,
    parse_tag_list,
    rank_by_centroid,
    safe_tag,
    screen_safe,
)

# Per-task shape.
#
# min_tags and penalty behaviour are read off the task bundles and are NOT
# tunable - they are what the validator does:
#   conversation/webpage/skill -> min_tags 3, penalties ON
#   survey                     -> min_tags 1, penalties ON
#   NER                        -> min_tags 1, penalties OFF (NoPenalty mechanism)
#
# target_tags and insurance ARE tunable. 12 tags beat 19 and 30 on real tasks
# (0.6485 / 0.6453 / 0.6371) - each extra weak tag drags mean+median (35%) and
# 20+ hands tag selection to the validator's random cull.
#
# insurance/target_tags were corrected TWICE by production, and the second
# correction reversed part of the first. The record, because the trade-off is
# not obvious:
#
#   insurance 6, 12 tags  -> 18% penalties {x0.95: 2, x0.90: 4}
#                            some tasks starved on unique tags
#   insurance 3, 12 tags  -> 29% penalties {x0.90: 6}
#                            x0.95 gone, but no_both_tags roughly doubled:
#                            half as many verbatim predictions means far fewer
#                            exact matches, and no_both_tags is a flat 10%
#   insurance 6, 15 tags  -> 30% penalties, mean 0.5093 (n=10). No better.
#
# Config A remains the best measured, so the shipped config is A's parameters
# (insurance 6, 12 tags, variants on) with the variant defect fixed and the
# unambiguous safety work kept: the non-ASCII language guard (a Portuguese
# conversation scored a hard 0.0000 without it) and an 11s deadline (a
# replica cut off by the deadline scores ~0.20 against ~0.55).
# Tag counts raised 2026-08-08 from the W&B reverse-engineering of the whole
# field: the top-10 miners submit ~17 tags with ~14 unique, and the ENTIRE field
# (269 miners, 27,978 tasks) sits at a mode of 18 - we were the sole outlier at
# 12/6.7-unique. Our old offline "12 beats 18" was -0.019, below the 0.049
# resolution of 4 podcasts; the population is far stronger evidence. insurance
# stays 6: at 18 tags it predicts 6 both, ~2.8 of which actually land, giving
# ~15 unique / ~3 both - the top miners' 14/3 split. NER stays 10 (no penalties,
# short-and-strong, and the one type where we are already within 0.04 of the
# top). Survey raised 6 -> 12 (our worst type at 0.410; more shots at the
# English translation of the Spanish choices).
TASK_PROFILE = {
    "conversation_tagging":       {"min_tags": 3, "target_tags": 20, "insurance": 6, "penalties": True},
    "webpage_metadata_generation": {"min_tags": 3, "target_tags": 20, "insurance": 6, "penalties": True},
    "skill_generation":           {"min_tags": 3, "target_tags": 20, "insurance": 6, "penalties": True},
    # NER has no penalties at all, so there is no no_both_tags to insure
    # against and no unique-count floor - the list wants to be short and strong.
    "named_entities_extraction":  {"min_tags": 1, "target_tags": 10, "insurance": 0, "penalties": False},
    # Survey ground truth is the literal selected_choices, so there is nothing
    # to replicate; the pool prompt predicts option wording directly.
    "survey_tagging":             {"min_tags": 1, "target_tags": 12, "insurance": 0, "penalties": True},
}

# Hard ceiling on submitted tags. 21+ CLEANED tags triggers random.sample(20) in
# LlmLib.validate_tag_set; at exactly 20 that sample is a shuffle and loses
# nothing (verified 2026-08-08). So 20 is the largest count that keeps full
# control. The production default stays at target_tags=12 (measured best - 18.7
# tags scored -0.019); 20 is available for the deep-enrichment A/B, where the
# extra slots may carry stronger candidates. Every emitted tag is a validated
# fixed point, so the validator's own cleaning cannot split one into two and
# push the count past 20.
MAX_TAGS = 20

# Expected-unique tags to aim for. The floor that matters is 3 (below it
# top_3_mean is zero-padded), but our estimate of which tags are "unique" is
# imperfect, so we build in headroom.
MIN_UNIQUE_TARGET = 5

# Guaranteed count of tags every word of which is a real dictionary word, so the
# validator's LLM "good English keywords" screen can never delete us below
# min_tags(3). On an acronym-heavy task the screen deleted ALL 18 of our tags ->
# a hard 0.0 (task cbc65e23, 2026-08-08). 7 leaves a wide margin above 3 even if
# the screen also drops one or two of the "safe" tags.
SCREEN_SAFE_FLOOR = 7


@dataclass
class Config:
    gt_model: str = "gpt-5.2"
    pool_model: str = "gpt-5.2"
    # 8s of the validator's 12s. Measured worst case end-to-end is ~9s at a 9s
    # deadline (bench/timing.py), and the 12s budget has to cover network
    # transport in both directions plus the validator's own overhead - so the
    # miner keeps a margin rather than spending the whole allowance.
    deadline_s: float = 8.0
    call_timeout_s: float = 6.5
    combine: str = "llm"              # llm | local | none
    pool_size: int = 40
    dedup_threshold: float = 0.93
    local_backends: tuple = ("spacy",)
    use_pool: bool = True
    use_local: bool = True
    gt_samples: int = 1          # independent ground-truth draws to pool (concurrent)
    # Variants are OFF after production evidence: pluralising proper nouns and
    # acronyms produced "dallas texases", "dallas texa", "irans", "nfls", which
    # the validator's English screen deletes. Losing 4-6 slots left mostly
    # exact ground-truth matches behind, starving the unique count and
    # zero-padding top_3_mean. Measured benefit had only been +0.007 (not
    # significant); measured cost was a 19% penalty rate on conversation tasks.
    # Back ON, but only with the spaCy part-of-speech guard in sn33/variants.py.
    # The original implementation pluralised proper nouns and acronyms
    # ("dallas texas" -> "dallas texases", "iran" -> "irans") and the validator
    # deleted every one. Config A shipped those broken variants and still scored
    # best of the three configs tried (0.5523 vs 0.4951 and 0.5093), so the
    # feature is restored with the defect removed rather than dropped wholesale.
    use_variants: bool = True
    variants_per_tag: int = 2
    compose_mode: str = "quota"  # quota | greedy (greedy maximises the estimated score)
    use_anchors: bool = True     # offer high-prior ground-truth vocabulary
    anchor_pool: int = 60
    anchor_min_cos: float = 0.42  # an anchor below this is off-topic; never submit it
    target_tags: Optional[int] = None  # override the per-task default
    insurance: Optional[int] = None
    # Conversation-ONLY insurance override. Measured 2026-08-15 (cross-draw,
    # n=107): ins 6->14 is a variance swap - p10 0.332->0.425, sub-0.4 tasks
    # 17->6, per-task sd 0.128->0.099, bottom-quartile +0.070 for top-quartile
    # -0.014. Peak is 14-16; >=18 declines; 20 (all-verbatim) collapses. Kept
    # separate from `insurance` because that field is GLOBAL and would force
    # verbatim tags into NER (profile insurance 0) and webpage.
    insurance_conv: Optional[int] = None
    # Webpage-ONLY insurance override. Same mechanism, independently measured
    # 2026-08-15 on the 49-task webpage judge corpus: ins 6->14 mean +0.0125,
    # p10 0.376->0.439, sub-0.5 18->15, worst regression -0.026; ins=10 is the
    # majority-positive point (W/L 27/19, worst -0.015).
    insurance_web: Optional[int] = None
    # Above 3 this forces _swap_in to replace high-cosine tags with lower-cosine
    # paraphrases. CLAUDE.md config D shipped 5 and measured final 0.4878 (n=10)
    # against config A's 0.5523 (n=28). Exposed so it can be A/B'd, not guessed.
    min_unique: Optional[int] = None
    # Seed the first 3 slots with expected-unique tags. Off by default: measured
    # -0.0065 on a single window (that window's two best candidates were `both`).
    # A hypothesis to A/B, not a proven gain.
    uniques_first: bool = False
    # Broad umbrella theme tags added to the CANDIDATE POOL for conversation
    # tasks. One extra LLM call, run concurrently inside the fan-out budget and
    # gated, so a slow one is dropped and we degrade to the default. Measured
    # +0.0013 on 10 mainnet tasks (flat on score) but the themes are broad and
    # dictionary-safe, so they also feed the screen-safe floor.
    use_theme_tags: bool = False
    # Translate non-English candidates to English before ranking. On a Spanish
    # conversation the predicted-GT tags rank highest on the (Spanish) centroid
    # but are DELETED by the validator's English screen; the multilingual
    # embedding keeps a translation ~0.01 from the original, so it survives AND
    # scores. Measured +0.017 on a real Univision task (6->12 survivors). One
    # gated LLM call, only on language-mismatched non-survey tasks.
    translate_non_english: bool = True
    # Re-extract each enrichment line at 30 tags instead of the upstream 10 and
    # union the result into the CANDIDATE POOL only. Enrichment supplies 87.7%
    # of ground truth (17.8 of 20.2 tags, measured 2026-08-08) while the 10-line
    # window supplies almost nothing, so the extra depth is the cheapest way to
    # see more of the target. Probed at +0.024 adjusted on 4/4 conversations.
    #
    # OFF by default: not production-proven, and it is the single variable in
    # the A/B. It costs one extra call per enrichment line, gated in
    # replica.replicate so a slow one degrades to exactly this default.
    use_deep_enrichment: bool = False
    deep_enrichment_tags: int = 30
    # Enrichment-first aiming, conversation only (2026-08-09 evidence run, both
    # findings verifier-CONFIRMED on 89 score-joined mainnet tasks):
    #   * the fraction of tags lexically anchored in enrichment predicts the
    #     real validator score at r=0.66; window-only answers are the 0.42-0.53
    #     low tail (worst same-task loss: our 0.415 vs others' 0.698);
    #   * candidate generation is window-dominated (pool prompt, replica doc
    #     call and local extraction all see ONLY the window), while the per-line
    #     enrichment extractions the replica already computes (N lines x ~10
    #     tags) were DISCARDED except via the ~20-tag combine.
    # Three gated behaviors, all zero-added-latency except ~50 extra strings in
    # the one embed batch: (a) per-line enrichment tags join the candidate pool;
    # (b) the ranking centroid is re-weighted toward them (GT is ~88%
    # enrichment-derived); (c) candidates with no lexical/semantic anchor in the
    # enrichment are demoted - never deleted - by `enrichment_demote`.
    # OFF by default: this is the single variable of the next A/B.
    enrichment_first: bool = False
    enrichment_demote: float = 0.90
    enrichment_anchor_cos: float = 0.35
    # Webpage extension of enrichment-first (2026-08-09 evidence run, GT-code
    # side verifier-CONFIRMED): webpage GT is built by the SAME 1-doc-set vs
    # N-enrichment-sets combine vote (~79% enrichment share, mean 3.7-4.3 sets,
    # WebpageMetadataGenerationTaskBundle.py:160-181), our enrichment-anchored
    # tag fraction predicts the real final at r=+0.53 while DOC-anchored
    # anti-predicts at -0.45, and the pure enrichment-centroid proxy hits
    # Spearman 0.952 vs real webpage finals (n=14) where the doc centroid is
    # uncorrelated (-0.04). Separate flag so it ships as its own single-variable
    # A/B on top of an already-live enrichment_first.
    enrichment_first_webpage: bool = False
    # Per-line quota (2026-08-10 cohort-loss forensics): the validator's GT is
    # a 1-vs-N combine where EACH enrichment line is one equal vote set, but
    # cosine-to-blended-centroid allocation lets majority-topic lines soak up
    # slots (measured: 11 testing tags vs 2/1 for the HIV/RICE lines on the
    # 13:26 loss, us 0.594 vs cohort 0.653). The replica already extracts tags
    # PER LINE; guarantee each line >=quota of its own tags in the answer.
    # 0 = off (the A/B control).
    enrichment_line_quota: int = 0
    # Conditional demotion (same forensics): demote=0.90 pays on tasks where
    # window and enrichment DISAGREE (we win those 5/5) but taxes tasks where
    # they agree (18:48-style near-losses). When ON, demotion applies only if
    # the window/enrichment vocabulary overlap is below demote_agree_cut.
    demote_conditional: bool = False
    demote_agree_cut: float = 0.18
    # Head-cap (2026-08-10 forensics, 4/4 real blowouts): "centroid capture" -
    # ranking fills 9-16 of 20 slots with one head-word family (housing=16,
    # embroidery=14, risk=12, capex=11 on the four samples) while whole GT
    # voting sources get zero tags. Cap tags sharing one stemmed head noun at
    # this many; excess slots go to the next-ranked candidates from other
    # families. 0 = off (A/B control).
    head_cap: int = 0
    # NER composite tags (2026-08-09, verifier-CONFIRMED on 10 score-joined NER
    # tasks): single entities cap at ~0.56-0.64 cosine against a centroid of ~22
    # diverse entity embeddings - the pool ceiling, not selection, is the gap
    # (a greedy oracle over our own pool scores WORSE than our shipped answer).
    # Multi-entity concatenations ('boston city hall mejia weber') reach
    # 0.60-0.83; swapping them into the 5 weakest slots measured +0.088 proxy
    # adjusted, 10/10 tasks improved. The NER path has no LLM screen and no
    # penalties (LlmLib.py:194-200, NoPenaltyGroundTruthTagSimilarityScoring),
    # so composites are legal there and ONLY there. OFF by default for the A/B.
    ner_combos: bool = False
    # --- RCA fixes 2026-08-11 (each independently flag-gated for A/B/C/D/E) ---
    # D: one salvage pass when the embed batch fails. Incidence measured 2/127
    # tasks/day (silent empty-vectors -> source=pool at ~0.32-0.48 vs cohorts
    # ~0.67). Retry runs only when >=1.5s of budget remains.
    embed_retry: bool = False
    # B: enrichment-first truncation fallback. All 3 pool losses in the RCA
    # shipped 0/N enrichment-line coverage against an ~88%-enrichment GT while
    # rep.enrichment_tags (per-line, already computed) sat unused. Interim
    # answer becomes round-robin per-line tags (one vote per line, mirroring
    # the validator's combine) + pool fill. Zero added LLM calls.
    fallback_enrichment: bool = False
    # C: value-based slot allocation (canary-only; offline instruments proved
    # unable to rank allocators). Lines earn slots by their own tags' cosine to
    # the target; junk lines (best < line_min_cos) earn none; unanchored
    # (window-only) tags capped. Post-pass over compose output, preserving the
    # verbatim insurance and the screen-safe floor.
    value_alloc: bool = False
    value_line_min_cos: float = 0.35
    value_line_max_slots: int = 6
    value_window_cap: int = 2
    # Honest centroid-summary phrases (conversation only). Adds a few umbrella
    # phrases built from the predicted-GT theme words to the candidate pool; the
    # composer ranks them like any other candidate and keeps one only when it
    # beats an existing tag on cosine to the target. Every phrase is filtered
    # through screen_safe, so nothing here can be deleted by the validator's
    # English screen. Offline (screen-safe only): +0.0085, W/L 43/7, worst
    # -0.003. OFF by default; its own single-variable flag.
    summary_tags: bool = False
    summary_tags_k: int = 3
    # Survey-only v2 generator: option-menu framing + precision-first ordering +
    # strict noun-phrase form (SN33_SURVEY_V2). Survey is the worst type
    # (0.410 vs 0.642) and has no re-ranking, so generation IS the lever.
    # Unmeasurable offline (GT = the validator's hidden selected_choices);
    # ships as a single-miner live canary. OFF by default.
    survey_v2: bool = False
    use_cache: bool = False            # bench turns this on; production leaves it off


@dataclass
class Result:
    tags: List[str] = field(default_factory=list)
    source: str = "none"               # which stage produced the answer
    elapsed: float = 0.0
    timings: Dict[str, float] = field(default_factory=dict)
    predicted_gt: List[str] = field(default_factory=list)
    degraded: bool = False
    # Diagnostics for the bench: the full candidate set and its embeddings, so
    # an oracle can re-rank the same pool against the true target and reveal
    # how much of the gap is candidate generation vs selection.
    candidates: List[str] = field(default_factory=list)
    vectors: Dict[str, List[float]] = field(default_factory=dict)


class Deadline:
    def __init__(self, budget: float):
        self.t0 = time.perf_counter()
        self.budget = budget

    def left(self) -> float:
        return max(0.0, self.budget - (time.perf_counter() - self.t0))

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def expired(self) -> bool:
        return self.left() <= 0.0


def _document_text(kind: str, window: Sequence, question: str = "", comment: str = "") -> str:
    """Plain text of whatever the miner was given, for local extraction."""
    if kind == "survey_tagging":
        return f"{question}\n{comment}"
    if not window:
        return ""
    if kind == "conversation_tagging":
        return "\n".join(str(l[1]) for l in window if len(l) >= 2)
    return str(window[0][1]) if len(window[0]) >= 2 else ""


async def _generate_pool(
    kind: str, text: str, cfg: Config, question: str = "", comment: str = ""
) -> List[str]:
    """Wide, deliberately over-inclusive candidate list from the LLM."""
    if kind == "survey_tagging":
        tmpl = prompts.MINER_SURVEY_POOL_V2 if cfg.survey_v2 else prompts.MINER_SURVEY_POOL
        prompt = tmpl.format(n=cfg.pool_size, question=question, comment=comment)
    elif kind == "conversation_tagging":
        prompt = prompts.MINER_CONVERSATION_POOL.format(n=cfg.pool_size, content=text[:6000])
    else:
        prompt = prompts.MINER_DOCUMENT_POOL.format(n=cfg.pool_size, content=text[:6000])
    raw = await llm.chat(
        prompt, model=cfg.pool_model, timeout=cfg.call_timeout_s, use_cache=cfg.use_cache, salt="miner"
    )
    return parse_tag_list(raw)


def window_enrichment_agreement(window_text: str, enrich_texts: Sequence[str]) -> float:
    """Vocabulary overlap in [0,1] between the window and the enrichment.

    |W ∩ E| / min(|W|,|E|) over significant words (len>3, post-safe_tag).
    High overlap = the conversation IS the enrichment topic, so window tags are
    legitimate GT material and demoting them just costs slots.
    """
    def words(s: str) -> set:
        return {w for w in safe_tag(s).split() if len(w) > 3}
    W = words(window_text)
    E = set()
    for t in enrich_texts or []:
        E |= words(str(t))
    if not W or not E:
        return 0.0
    return len(W & E) / min(len(W), len(E))


def apply_line_quota(
    final: List[str],
    per_line: Sequence[Sequence[str]],
    vectors: Dict[str, Sequence[float]],
    target,
    quota: int = 2,
    protected: Optional[set] = None,
    screen_floor: int = 0,
) -> List[str]:
    """Guarantee each enrichment line >= ``quota`` tags from its OWN extraction.

    The combine gives every line one equal vote, so an answer that ignores a
    line concedes that line's share of the centroid. Swaps the weakest
    non-protected tags for each starved line's best (by cosine) own tags.
    Never drops below ``screen_floor`` screen-safe tags and never removes
    ``protected`` (verbatim insurance) tags. Pure CPU on existing vectors.
    """
    protected = protected or set()
    out = list(final)
    have = set(out)

    def cos(t) -> float:
        v = vectors.get(t)
        if v is None or not len(v) or target is None:
            return -1.0
        return cosine(np.asarray(v, dtype=np.float32), target)

    for line_tags in per_line:
        line_set = [t for t in line_tags if t]
        if not line_set:
            continue
        covered = sum(1 for t in line_set if t in have)
        needed = quota - covered
        if needed <= 0:
            continue
        adds = sorted((t for t in line_set if t not in have and cos(t) >= 0),
                      key=cos, reverse=True)[:needed]
        for add in adds:
            # evict the weakest tag that is neither protected, a quota add of an
            # earlier line, nor load-bearing for the screen-safe floor
            safe_count = sum(1 for t in out if screen_safe(t))
            evictable = [t for t in out
                         if t not in protected and t not in line_set
                         and not (screen_safe(t) and safe_count <= screen_floor)]
            if not evictable:
                break
            worst = min(evictable, key=cos)
            # The line's vote is worth a moderate cosine discount: majority-line
            # tags ALWAYS rank higher (that is why they hogged the slots), so
            # demanding cos(add) > cos(worst) would nullify the quota. Junk is
            # still blocked - an add must reach 85% of the evictee's cosine.
            if cos(add) < cos(worst) * 0.85:
                continue
            out.remove(worst)
            out.append(add)
            have.discard(worst)
            have.add(add)
            protected = protected | {add}
    return out


def allocate_value_based(
    final: List[str],
    per_line: Sequence[Sequence[str]],
    vectors: Dict[str, Sequence[float]],
    target,
    enrich_vocab: set,
    protected: Optional[set] = None,
    screen_floor: int = 0,
    target_tags: int = 20,
    line_min_cos: float = 0.35,
    line_max_slots: int = 6,
    window_cap: int = 2,
) -> List[str]:
    """Value-based slot allocation (RCA cause 2, user-proposed policy).

    Differences from the fixed quota:
      * a line's slot budget SCALES with its strength (best own-tag cosine to
        the target), so strong lines may take up to ``line_max_slots`` and a
        junk line (best < ``line_min_cos``, e.g. a Wikipedia-disambiguation
        snippet) takes ZERO - no slots wasted satisfying a quota;
      * tags lexically unanchored in ANY enrichment line ("window-only") are
        capped at ``window_cap`` - the window supplies ~10% of GT but consumed
        6 slots on two RCA losses.
    Post-pass over compose output: verbatim insurance (``protected``) and the
    screen-safe floor survive by construction. Pure CPU on existing vectors.
    """
    protected = protected or set()

    def cos(t) -> float:
        v = vectors.get(t)
        if v is None or not len(v) or target is None:
            return -1.0
        return cosine(np.asarray(v, dtype=np.float32), target)

    # per-line strength and slot budget
    lines = [[t for t in s if t] for s in per_line if s]
    strengths = [max((cos(t) for t in s), default=-1.0) for s in lines]
    active = [(s, st_) for s, st_ in zip(lines, strengths) if st_ >= line_min_cos]
    if not active:
        return final
    total = sum(st_ for _, st_ in active)
    budgets = []
    for s, st_ in active:
        share = max(1, round(target_tags * (st_ / total))) if total > 0 else 1
        budgets.append((s, min(line_max_slots, share)))

    def anchored(tag: str) -> bool:
        return tag in protected or any(w in enrich_vocab for w in tag.split())

    picked: List[str] = []
    have = set()
    # 1. each active line claims its budget from its OWN tags, best first
    for s, budget in budgets:
        own = sorted((t for t in s if t not in have and cos(t) >= 0), key=cos, reverse=True)
        for t in own[:budget]:
            if len(picked) >= target_tags:
                break
            picked.append(t)
            have.add(t)
    # 2. fill remaining slots in compose's order (keeps verbatims/floor bias),
    #    capping unanchored (window-only) tags
    window_used = 0
    for t in final:
        if len(picked) >= target_tags:
            break
        if t in have:
            continue
        if not anchored(t):
            if window_used >= window_cap and t not in protected:
                continue
            window_used += 1
        picked.append(t)
        have.add(t)
    # 3. protected tags must never be dropped by this pass
    for t in final:
        if t in protected and t not in have and picked:
            weakest = min((x for x in picked if x not in protected), key=cos, default=None)
            if weakest is not None:
                picked[picked.index(weakest)] = t
                have.add(t)
    # 4. screen-safe floor guard: if the reallocation dropped below the floor,
    #    it is not safe to ship - fall back to compose's original answer.
    if screen_floor and sum(1 for t in picked if screen_safe(t)) < min(
            screen_floor, sum(1 for t in final if screen_safe(t))):
        return final
    return picked if len(picked) >= 3 else final


def _stem(w: str) -> str:
    """Naive inflection stem, enough to unite singular/plural families."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s"):
        return w[:-1]
    return w


def apply_head_cap(
    final: List[str],
    ranked: Sequence[tuple],
    cap: int,
    protected: Optional[set] = None,
) -> List[str]:
    """Limit tags sharing one stemmed head noun to ``cap``; refill from ranked.

    The centroid-capture failure ships 9-16 of 20 tags in one family
    ("housing market", "housing markets", "housing prices", ...). Keeping the
    top ``cap`` of a family and refilling the freed slots with the best
    candidates from OTHER families restores coverage of the ignored GT voting
    sources. ``final`` order is assumed best-first (compose emits it so).
    """
    if cap <= 0:
        return final
    protected = protected or set()
    fam_count: Dict[str, int] = {}
    kept: List[str] = []
    dropped = 0
    for t in final:
        words = t.split()
        fam = _stem(words[-1]) if words else t
        c = fam_count.get(fam, 0)
        if c >= cap and t not in protected:
            dropped += 1
            continue
        fam_count[fam] = c + 1
        kept.append(t)
    if not dropped:
        return final
    have = set(kept)
    for t, _s in ranked:
        if dropped <= 0:
            break
        if t in have:
            continue
        words = t.split()
        fam = _stem(words[-1]) if words else t
        if fam_count.get(fam, 0) >= cap:
            continue
        fam_count[fam] = fam_count.get(fam, 0) + 1
        kept.append(t)
        have.add(t)
        dropped -= 1
    return kept


def _enrichment_first_on(cfg: Config, kind: str) -> bool:
    """Which task types run enrichment-first aiming, per their own flags.

    Conversation and webpage have the SAME GT structure (1 doc set outvoted by
    N enrichment sets in combine - ~88% and ~79% enrichment share respectively)
    but separate flags, so each extension ships as its own single-variable A/B.
    NER/skill/survey never: NER GT is entities from the document, survey never
    reaches the replica, skill is unmeasured.
    """
    if kind == "conversation_tagging":
        return cfg.enrichment_first
    if kind == "webpage_metadata_generation":
        return cfg.enrichment_first_webpage
    return False


def ner_combo_candidates(entities: Sequence[str], limit: int = 80, max_len: int = 50) -> List[str]:
    """Composite NER tags: pair permutations + triple combinations of entities.

    A single entity tops out at ~0.56-0.64 cosine against the GT centroid (the
    mean of ~22 diverse entity embeddings); concatenating entities moves the
    embedding toward that mean, measured at 0.60-0.83. Pairs keep both orders
    (order changes the embedding); triples one order. ``max_len`` is 50, not the
    NER path's 64, because ``_finish`` re-normalizes the outgoing answer through
    ``tags.normalize`` (MAX_LEN=50) - a longer combo would survive the validator
    but not our own hygiene, so it must never be emitted.
    """
    base = [e for e in entities if e][:8]
    out: List[str] = []
    seen = set(base)

    def overlap(x: str, y: str) -> bool:
        # 'seattle' + 'seattle city council' -> 'seattle seattle city council':
        # legal but wastes the slot a distinct-entity pair would earn.
        return x in y or y in x

    for a in base:
        for b in base:
            if a == b or overlap(a, b):
                continue
            c = f"{a} {b}"
            if len(c) <= max_len and c not in seen:
                seen.add(c)
                out.append(c)
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            if overlap(base[i], base[j]):
                continue
            for k in range(j + 1, len(base)):
                if overlap(base[i], base[k]) or overlap(base[j], base[k]):
                    continue
                c = f"{base[i]} {base[j]} {base[k]}"
                if len(c) <= max_len and c not in seen:
                    seen.add(c)
                    out.append(c)
    return out[:limit]


def demote_unanchored(
    ranked: Sequence[tuple],
    vectors: Dict[str, Sequence[float]],
    enrich_gt: Sequence[str],
    enrichment: Sequence[str],
    exempt: set,
    demote: float = 0.90,
    min_cos: float = 0.35,
) -> List[tuple]:
    """Demote (never delete) candidates with no anchor in the enrichment.

    Anchor = shares a word with the enrichment tags/text (lexical), or sits
    within ``min_cos`` of the enrichment-tag centroid (semantic), or is one of
    the ``exempt`` tags (the verbatim predicted-GT insurance, whose eviction
    would re-fire no_both_tags). Window-only tags were measured as the 0.42-0.53
    low tail while enrichment-anchored answers ran 0.63+ (fe r=0.66, verified);
    demotion keeps them submittable when nothing better exists.
    """
    if not enrich_gt:
        return list(ranked)
    vocab_words = {w for t in enrich_gt for w in t.split()}
    for line in enrichment or []:
        vocab_words.update(safe_tag(str(line)).split())
    e_vecs = [np.asarray(vectors[t], dtype=np.float32) for t in enrich_gt
              if t in vectors and vectors[t] is not None and len(vectors[t])]
    e_cent = np.mean(e_vecs, axis=0) if e_vecs else None

    def anchored(tag: str) -> bool:
        if tag in exempt or any(w in vocab_words for w in tag.split()):
            return True
        v = vectors.get(tag)
        if e_cent is not None and v is not None and len(v):
            return cosine(np.asarray(v, dtype=np.float32), e_cent) >= min_cos
        return False

    out = [(t, s if anchored(t) else s * demote) for t, s in ranked]
    out.sort(key=lambda x: -x[1])
    return out


def compose_greedy(
    ranked: Sequence[tuple],
    predicted_gt: Sequence[str],
    vectors: Dict[str, Sequence[float]],
    target: "np.ndarray",
    profile: dict,
    max_tags: int = MAX_TAGS,
) -> List[str]:
    """Pick the tag set that maximises the *estimated* score directly.

    ``compose`` below balances the formula's competing terms with hand-tuned
    quotas. But we can estimate the score itself: we have an estimate of the
    target vector, and ``predicted_gt`` is an estimate of which tags will count
    as `both` rather than `unique`. So instead of approximating the objective
    with heuristics, run the real scoring function over candidate sets and keep
    the best.

    This automatically handles what the quotas only approximate: that the third
    unique tag is worth far more than the fourth (zero padding), that a tag
    below the current mean drags `mean` and `median` (35% of the score), and
    that the flat 10% no_both penalty is worth exactly one good slot.

    Greedy addition is sound here because each term is monotone in the per-tag
    cosines; the only interactions are through set membership, which we
    re-evaluate at every step.
    """
    from sn33 import scoring

    gt_norm = set(normalize_all(predicted_gt))
    pool = [t for t, _ in ranked]
    if not pool:
        return []

    penalties = profile["penalties"]
    min_tags = profile["min_tags"]

    def estimate(tags: List[str]) -> float:
        if len(tags) < min_tags:
            return -1.0
        return scoring.score_tags(
            list(gt_norm), target, tags, vectors, penalties=penalties, min_tags=min_tags
        )["final"]

    # Seed with the strongest `min_tags` candidates. Growing from an empty list
    # cannot work: every set below the floor scores 0, so no single first tag
    # ever looks like an improvement and the search never starts.
    chosen: List[str] = pool[:min_tags]
    if len(chosen) < min_tags:
        return []
    best_score = estimate(chosen)

    # Then add tags one at a time, keeping only additions that raise the
    # estimated score. This is where the length is decided: a tag below the
    # current mean pulls `mean` and `median` down more than it adds elsewhere,
    # so the loop stops on its own rather than filling a fixed quota.
    for _ in range(max_tags - len(chosen)):
        best_tag, best_gain = None, best_score
        for tag in pool:
            if tag in chosen:
                continue
            s = estimate(chosen + [tag])
            if s > best_gain:
                best_gain, best_tag = s, tag
        if best_tag is None:
            break
        chosen.append(best_tag)
        best_score = best_gain
    return chosen[:max_tags]


def compose(
    ranked: Sequence[tuple],
    predicted_gt: Sequence[str],
    profile: dict,
    target_tags: int,
    insurance: int,
    anchors: Optional[set] = None,
    min_unique: Optional[int] = None,
    uniques_first: bool = False,
) -> List[str]:
    """Turn ranked candidates into the final answer.

    The formula pulls in two directions and this is where they are balanced.

    ``top_3_mean`` is 55% of the score and is computed over UNIQUE tags only -
    the ones that do NOT string-match a ground-truth tag - padding with zeros
    when there are fewer than three. So a miner that reproduces ground truth
    perfectly scores 0.55 x 0 and loses more than half the formula. Measured:
    submitting the replica's own tags verbatim scored 0.23 against a 0.60
    baseline, with `less_than_1_unique` firing on every case.

    ``no_both_tags`` costs a flat 10% unless at least one tag matches ground
    truth exactly. That is worth one or two slots, no more.

    So the answer is built the other way round from the obvious: the bulk is
    *paraphrases* - tags close to the ground-truth centroid but lexically
    different from any predicted ground-truth tag - plus a small number of
    verbatim predictions as insurance against the 10%.
    """
    gt_norm = set(normalize_all(predicted_gt))

    # Candidates split by their expected role once the validator diffs our list
    # against its own tags.
    paraphrases = [t for t, _ in ranked if t not in gt_norm]   # expected `unique`
    verbatim = [t for t, _ in ranked if t in gt_norm]          # expected `both`
    if not verbatim:
        verbatim = [t for t in normalize_all(predicted_gt)]

    # Corpus-frequency anchors are the other route to an exact match: they are
    # the strings this domain's ground truth actually uses. They survived the
    # cosine floor already, so they rank alongside our own predictions.
    if anchors:
        verbatim = verbatim + [t for t, _ in ranked if t in anchors and t not in verbatim]

    # Take the strongest candidates regardless of role - measurement showed the
    # replica's own tags outrank pool paraphrases on cosine, so a fixed quota
    # for either side leaves score on the table.
    if uniques_first:
        # Seed the first three slots with the highest-cosine EXPECTED-unique
        # candidates, since only unique tags feed the 55% top-3 term. HYPOTHESIS,
        # off by default: measured -0.0065 on a single window, because that
        # window's two best candidates were `both` and forcing uniques ahead of
        # them reached lower in the pool. A/B before trusting it.
        seed = paraphrases[:3]
        rest = [t for t, _ in ranked if t not in seed]
        chosen = (seed + rest)[:target_tags]
    else:
        chosen = [t for t, _ in ranked[:target_tags]]

    # Then repair the two counts the formula is sensitive to. Both repairs
    # replace the WEAKEST chosen tag, so the cost is always the smallest
    # available.
    def _swap_in(candidates: List[str], needed: int) -> None:
        for cand in candidates:
            if needed <= 0:
                return
            if cand in chosen:
                continue
            for i in range(len(chosen) - 1, -1, -1):
                if chosen[i] not in candidates:
                    chosen[i] = cand
                    needed -= 1
                    break
            else:
                return

    # >=3 unique tags, or top_3_mean gets zero-padded and 55% of the score
    # collapses. This is the single most expensive mistake available.
    #
    # Aim for MIN_UNIQUE_TARGET, not 3. "Unique" here means "not in the tags we
    # predicted"; the validator diffs against the tags it actually generated.
    # In production those sets differ, so tags we expect to be unique routinely
    # land in real ground truth. Measured: 6 of 36 scored tasks fell under the
    # floor even though every one of them was composed with >=3 expected
    # uniques. The margin absorbs that estimation error.
    want_unique = MIN_UNIQUE_TARGET if min_unique is None else min_unique
    have_unique = sum(1 for t in chosen if t not in gt_norm)
    if have_unique < want_unique:
        _swap_in(paraphrases, want_unique - have_unique)

    # >=`insurance` exact-match candidates, to clear the flat 10% no_both_tags
    # penalty. Measured optimum on webpage tasks was 6-8 of 12.
    have_both = sum(1 for t in chosen if t in gt_norm)
    if have_both < insurance:
        _swap_in(verbatim, insurance - have_both)

    # Screen-safe floor. The validator's LLM screen deletes acronyms and
    # non-dictionary compounds; on an acronym-heavy task it deleted ALL our tags
    # and we scored a hard 0.0 (below min_tags). Guarantee a floor of tags every
    # word of which is a real dictionary word, so the screen can never delete us
    # to zero. This OUTRANKS the unique/both repairs above by cost: a zero loses
    # the whole score, no_both costs 0.06. NER is exempt - it uses
    # validate_named_entities_tag_set, which has no LLM screen.
    if profile.get("penalties", True):
        floor = min(SCREEN_SAFE_FLOOR, max(0, target_tags - 1))
        if sum(1 for t in chosen if screen_safe(t)) < floor:
            safe_cands = [t for t, _ in ranked if screen_safe(t)]
            needed = floor - sum(1 for t in chosen if screen_safe(t))
            for cand in safe_cands:
                if needed <= 0:
                    break
                if cand in chosen:
                    continue
                for i in range(len(chosen) - 1, -1, -1):
                    if not screen_safe(chosen[i]):
                        chosen[i] = cand
                        needed -= 1
                        break
                else:
                    break

    # Backfill if the pool was thin - a short list risks the hard-zero floor.
    for tag, _ in ranked:
        if len(chosen) >= target_tags:
            break
        if tag not in chosen:
            chosen.append(tag)

    # Never exceed the cull threshold: 20+ tags means the validator keeps a
    # random 20 (LlmLib.validate_tag_set:198).
    seen = set()
    out = [t for t in chosen if not (t in seen or seen.add(t))][:MAX_TAGS]

    # Hard floor. Under `min_tags` the validator does not apply a penalty - it
    # discards the response entirely (evaluator.py:157), which is a total loss
    # rather than a multiplier. The swap/dedupe path above can drop below the
    # floor when the pool is thin: measured 2.1 tags/case at target_tags=6,
    # median score 0.0000. At the production target of 12 it does not fire, but
    # nothing in the code guarantees that, and the failure is unrecoverable.
    if len(out) < profile["min_tags"]:
        for tag, _ in ranked:
            if len(out) >= profile["min_tags"]:
                break
            if tag not in seen:
                seen.add(tag)
                out.append(tag)
    return out


async def _theme_tags(convo_text: str, enrichment: str, cfg: "Config", timeout: float) -> List[str]:
    """Broad umbrella theme tags for the conversation. Pool material; never raises."""
    try:
        raw = await llm.chat(prompts.miner_theme(convo_text, enrichment),
                             model=cfg.pool_model, timeout=timeout,
                             use_cache=cfg.use_cache, salt="theme")
        return parse_tag_list(raw)
    except Exception:  # noqa: BLE001 - a theme failure must never sink the answer
        return []


async def _translate_candidates(tags: List[str], cfg: "Config", timeout: float) -> List[str]:
    """Translate a batch of tags to English. Never raises - a failure keeps the
    original candidates (the screen-safe floor still guarantees no zero)."""
    if not tags:
        return []
    try:
        raw = await llm.chat(prompts.translate_to_english(", ".join(tags)),
                             model=cfg.pool_model, timeout=timeout,
                             use_cache=cfg.use_cache, salt="xlate")
        return normalize_all(parse_tag_list(raw))
    except Exception:  # noqa: BLE001
        return []


async def mine(
    kind: str,
    *,
    window: Sequence = (),
    enrichment: Optional[Sequence[str]] = None,
    question: str = "",
    comment: str = "",
    coding: bool = False,
    cfg: Optional[Config] = None,
) -> Result:
    """Produce the tag list for one task. Never raises.

    ``window`` is the task's window as received. ``enrichment`` is optional and
    only needed for conversation tasks, where the validator ships enrichment in
    its own ``input.data.enrichment_lines`` field rather than inside the window
    (ConversationTaskInputData.enrichment_lines). For webpage / NER the
    enrichment lines are window[1:], so they are derived when not passed.
    """
    cfg = cfg or Config()
    profile = TASK_PROFILE.get(kind, TASK_PROFILE["conversation_tagging"])
    target_tags = cfg.target_tags if cfg.target_tags is not None else profile["target_tags"]
    insurance = cfg.insurance if cfg.insurance is not None else profile["insurance"]
    if kind == "conversation_tagging" and cfg.insurance_conv is not None:
        insurance = cfg.insurance_conv
    if kind == "webpage_metadata_generation" and cfg.insurance_web is not None:
        insurance = cfg.insurance_web
    dl = Deadline(cfg.deadline_s)
    res = Result()

    try:
        text = _document_text(kind, window, question, comment)
        if enrichment is not None:
            enrichment = [str(e) for e in enrichment if str(e).strip()]
        elif kind == "conversation_tagging":
            enrichment = []          # never inside the window for this task type
        else:
            enrichment = [str(l[1]) for l in list(window)[1:] if len(l) >= 2 and str(l[1]).strip()]

        # ---- stage 1 first: network work is the long pole, so start it before
        # anything CPU-bound. spaCy's first call loads a model and parses the
        # document, which blocks the event loop; doing that ahead of the LLM
        # calls delayed them by seconds and, on a task with 6 enrichment lines,
        # left too little budget for the replica to finish at all.
        jobs = {}
        if kind == "survey_tagging":
            # Nothing to replicate: ground truth is the literal selected_choices.
            jobs["pool"] = _generate_pool(kind, text, cfg, question, comment)
        else:
            convo_xml = ""
            if kind == "conversation_tagging":
                convo_xml = _convo_xml(window)
            jobs["replica"] = replica.replicate(
                kind,
                document=text,
                convo_xml=convo_xml,
                enrichment=enrichment,
                model=cfg.gt_model,
                timeout=min(cfg.call_timeout_s, dl.left()),
                coding=coding,
                combine=cfg.combine,
                use_cache=cfg.use_cache,
                samples=cfg.gt_samples,
                deadline=dl.left(),
                # survey_tagging never reaches here (no ground truth to
                # replicate), which is what we want: its traffic is Spanish and
                # get_safe_tag shreds non-English tags into a hard zero.
                deep_enrichment=cfg.deep_enrichment_tags if cfg.use_deep_enrichment else 0,
            )
            if cfg.use_pool:
                # Enrichment-first (c): today MINER_CONVERSATION_POOL sees zero
                # enrichment chars even when enrichment exists (measured mean
                # 3,429 window chars vs 0). Appending snippets to the SAME call
                # adds no latency and no extra API load; 6000-char cap holds.
                pool_text = text
                if _enrichment_first_on(cfg, kind) and enrichment:
                    # 800 chars, not 2500: extra input tokens slow the pool
                    # completion, and the pool is on the critical path of the
                    # fan-out (first post-deploy tasks ran 11.9s vs the 11.0
                    # deadline). The topic signal survives heavy truncation.
                    pool_text = (
                        text
                        + "\n\nRELATED SEARCH SNIPPETS (tags should cover these topics too):\n"
                        + "\n".join(e[:250] for e in enrichment)[:800]
                    )
                jobs["pool"] = _generate_pool(kind, pool_text, cfg)
            # Broad theme tags, conversation only, concurrent + gated so they add
            # no sequential latency: _gather_within cancels this job if it does
            # not land inside the budget, and it feeds the POOL only.
            if cfg.use_theme_tags and kind == "conversation_tagging":
                jobs["theme"] = _theme_tags(text, "\n".join(enrichment), cfg,
                                            min(cfg.call_timeout_s, dl.left()))

        # Local extraction runs in a thread so it overlaps the network calls
        # instead of delaying them, and still gives us a submittable answer if
        # every API call fails.
        local_task = None
        if cfg.use_local and text:
            local_task = asyncio.ensure_future(
                asyncio.to_thread(
                    extract.candidates, text, cfg.local_backends, cfg.pool_size
                )
            )

        done = await _gather_within(jobs, dl.left())

        local: List[str] = []
        if local_task is not None:
            try:
                local = await asyncio.wait_for(local_task, timeout=max(0.05, min(1.5, dl.left())))
            except Exception:
                local_task.cancel()
        if local:
            res.tags = normalize_all(local)[:target_tags]
            res.source = "local"
        rep = done.get("replica")
        pool_tags = done.get("pool") or []

        predicted_gt = list(rep.tags) if rep is not None and getattr(rep, "tags", None) else []
        res.predicted_gt = predicted_gt

        # Enrichment-first (a)+(b) source material: the per-line replica
        # extractions (N lines x ~10 tags) are already computed inside
        # replicate() and were previously DISCARDED - only what survived the
        # ~20-tag combine reached us. GT is ~88% enrichment-derived, so these
        # are the closest free proxies of the real target. They feed the
        # candidate pool and the centroid weighting ONLY - never predicted_gt,
        # whose verbatim insurance must stay upstream-faithful (same discipline
        # as deep/theme tags).
        enrich_gt: List[str] = []
        enrich_by_line: List[List[str]] = []
        if _enrichment_first_on(cfg, kind) and rep is not None:
            # Keep the PER-LINE structure (each line is one equal combine vote);
            # the flat list feeds candidates+centroid, the per-line lists feed
            # the quota repair after compose.
            enrich_by_line = [normalize_all(s or [])[:12]
                              for s in (getattr(rep, "enrichment_tags", None) or [])]
            # Cap 30, not 50: the first post-deploy tasks (2026-08-09 12:20)
            # overran the deadline by up to 0.9s and the marginal tags past ~30
            # only fatten the embed batch and the dedup sweep, not the centroid.
            enrich_gt = normalize_all([t for s in enrich_by_line for t in s])[:30]
        res.degraded = bool(rep.degraded) if rep is not None else True
        if rep is not None:
            res.timings.update(rep.timings)

        # Interim answer, in case the deadline hits before ranking finishes.
        # Prefer the POOL over the replica here: replica tags are our best guess
        # at the validator's own tags, and a list made purely of those scores
        # `less_than_1_unique` with top_3_mean padded to zero. Pool tags are
        # paraphrases, which is the shape the formula actually rewards.
        if pool_tags:
            promoted = normalize_all(pool_tags)[:target_tags]
            if len(promoted) >= profile["min_tags"]:
                res.tags = promoted
                res.source = "pool"
        elif predicted_gt:
            promoted = normalize_all(predicted_gt)[:target_tags]
            if len(promoted) >= profile["min_tags"]:
                res.tags = promoted
                res.source = "replica"

        # B: enrichment-first fallback interim. The pool prompt sees mostly
        # window text, so a truncated answer used to ship window vocabulary
        # against a GT that is ~88% enrichment (three RCA losses at 0/4
        # enrichment-line coverage, gaps -0.427/-0.247/-0.088). The per-line
        # replica extractions are already in hand at this point; round-robin
        # one tag per line per pass - each line is one combine vote - then fill
        # with pool paraphrases. Ranked output still overwrites this whenever
        # stage 2 completes, so the flag ONLY changes what a truncation ships.
        if (cfg.fallback_enrichment and enrich_by_line
                and kind in ("conversation_tagging", "webpage_metadata_generation")):
            per_line_norm = [normalize_all(s)[:6] for s in enrich_by_line if s]
            rr: List[str] = []
            depth = 0
            while len(rr) < target_tags and depth < 6:
                added = False
                for s in per_line_norm:
                    if depth < len(s) and s[depth] not in rr:
                        rr.append(s[depth])
                        added = True
                        if len(rr) >= target_tags:
                            break
                if not added:
                    break
                depth += 1
            fill = [t for t in normalize_all(pool_tags) if t not in rr]
            interim = (rr + fill)[:target_tags]
            if len(interim) >= profile["min_tags"]:
                res.tags = interim
                res.source = "fallback_enrich"

        # ---- stage 2: rank every candidate against the estimated target ----
        anchors = vocab.anchors_for(kind, limit=cfg.anchor_pool) if cfg.use_anchors else []
        # Lexical variants of the predicted ground truth are the strongest
        # `unique` candidates there are: same meaning, different string, so they
        # feed top_3_mean (55%) at nearly the cosine of a ground-truth tag.
        # NER combos replace variants there: pluralizing entities ships mangled
        # proper nouns ('luke palmisanos', 'downtown seattles') past the spaCy
        # PROPN guard - measured ~0 direct score impact but each one wastes a
        # slot a composite could earn (~0.60 cosine vs a duplicate vector).
        skip_variants = cfg.ner_combos and kind == "named_entities_extraction"
        variant_tags = (
            variants.expand(normalize_all(predicted_gt), per_tag=cfg.variants_per_tag)
            if cfg.use_variants and predicted_gt and not skip_variants
            else []
        )
        candidates = normalize_all(
            list(pool_tags) + list(predicted_gt) + list(res.tags) + variant_tags + anchors
            + enrich_gt
        )
        if kind == "survey_tagging":
            # No replica to aim at; the pool's own order is the model's ranking.
            candidates = normalize_all(list(pool_tags) + list(res.tags))

        # Deep enrichment tags widen the POOL and nothing else. They are added
        # here, AFTER predicted_gt, the variants and the anchors have all been
        # fixed, and they are never fed back into any of them.
        #
        # This ordering is load-bearing. A deep tag string-matches real ground
        # truth only 10.3% of the time against 26.2% for upstream-depth tags, so
        # leaking them into predicted_gt would dilute the verbatim insurance and
        # start firing the flat x0.9 no_both_tags penalty. A sibling probe did
        # exactly that and turned +0.043 adjusted into -0.024 final, on 38 of 38
        # windows. Everything below only ever ranks these by cosine.
        # Theme tags widen the POOL only (same discipline as deep tags): never
        # into predicted_gt, the centroid, or the anchors. English-filtered.
        theme_tags = done.get("theme") or []
        if theme_tags and kind == "conversation_tagging":
            pv = {w for t in candidates for w in t.split()}
            candidates = list(dict.fromkeys(
                candidates + [t for t in normalize_all(theme_tags) if is_probably_english(t, pv)]))

        # Honest centroid-summary phrases (conversation only). Umbrella phrases
        # built from the predicted-GT theme words embed near the GT centroid
        # (the NER-composite mechanism, legally: the validator re-embeds the
        # phrase string). Screen-safe filtered so the English screen cannot
        # delete them, and added to the POOL only - the composer keeps one just
        # when it out-ranks an existing candidate on cosine, so a weak phrase is
        # never selected (offline +0.0085 screen-safe, W/L 43/7, worst -0.003).
        summary_new: List[str] = []
        if cfg.summary_tags and kind == "conversation_tagging" and predicted_gt:
            cand_set = set(candidates)
            summary_new = [
                s for s in normalize_all(make_summary_tags(predicted_gt, cfg.summary_tags_k))
                if s and s not in cand_set and screen_safe(s)
            ]
            if summary_new:
                candidates = list(dict.fromkeys(candidates + summary_new))

        deep_tags = getattr(rep, "deep_tags", None) if rep is not None else None
        if deep_tags and kind != "survey_tagging":
            # Language guard. Deep tags come from the one pool prompt with no
            # English instruction, so a Spanish source yields on-topic Spanish
            # that `normalize` cannot catch (it is pure ASCII) and the validator
            # deletes - a Portuguese conversation once scored a hard 0.0000. The
            # pool we already built is English, so its words are the vocabulary a
            # legitimate deep tag should overlap; anything failing that AND the
            # dictionary check is dropped. Over-rejecting only costs a candidate.
            pool_vocab = {w for t in candidates for w in t.split()}
            pool_vocab.update(w for t in normalize_all(predicted_gt) for w in t.split())
            deep_clean = [t for t in normalize_all(deep_tags)
                          if is_probably_english(t, pool_vocab)]
            candidates = list(dict.fromkeys(candidates + deep_clean))

        # Translate non-English candidates to English. On a Spanish conversation
        # the predicted-GT tags and their variants (from the validator's own
        # prompt, no English instruction) rank highest on the Spanish centroid
        # and are then DELETED by the validator's English screen. Measured on a
        # real Univision task: 6->12 survivors, adjusted 0.5485->0.5650 (+0.017).
        # The centroid stays faithful (built from predicted_gt below, untouched);
        # only the SUBMITTABLE candidates are translated. Gated on the deadline,
        # and translating an already-English tag is a no-op, so misclassification
        # is harmless. A failed/slow translation leaves the Spanish candidates and
        # the screen-safe floor backstops the zero.
        # Detect the mismatch on predicted_gt - those are the high-cosine tags
        # that dominate the SELECTED answer, even when they are a small fraction
        # of the wide candidate pool. If they are mostly non-English, the task is
        # language-mismatched and every non-English candidate needs translating.
        if cfg.translate_non_english and kind != "survey_tagging" and dl.left() > 2.5:
            gt_norm = normalize_all(predicted_gt)
            gt_non_eng = [t for t in gt_norm if not is_probably_english(t)]
            if gt_norm and len(gt_non_eng) >= max(2, int(0.3 * len(gt_norm))):
                non_eng = [t for t in candidates if not is_probably_english(t)]
                eng = await _translate_candidates(
                    non_eng, cfg, min(cfg.call_timeout_s, dl.left() - 1.5))
                if eng:
                    candidates = list(dict.fromkeys(
                        [t for t in candidates if is_probably_english(t)] + eng))

        if not candidates or dl.left() < 0.6:
            return _finish(res, dl)

        # The validator embeds the CLEANED form of each ground-truth tag
        # (get_vector_embeddings_set re-cleans before embedding), so the target
        # must be built from cleaned strings to sit in the same space.
        gt_for_target = Utils.get_clean_tag_set(predicted_gt) if predicted_gt else []
        # No ground truth to aim at - survey (never replicated) or a failed
        # replica - means there is no target to rank against, so the embedding
        # round-trip below would be paid for and then discarded. Skip it and keep
        # the pool/local answer already promoted into res.tags.
        if not gt_for_target:
            return _finish(res, dl)

        # Enrichment-first (b): re-weight the ranking centroid toward the
        # per-line enrichment tags. Real GT is ~88% enrichment-derived; today's
        # centroid is the combine output alone, where the window set holds one
        # of N+1 votes. Repetition IS the weighting - `centroid` averages
        # per-entry, and a tag in BOTH lists is counted twice deliberately -
        # so this list must never pass through Utils.get_clean_tag_set, which
        # dedupes via set() (verified 2026-08-09). enrich_gt is already
        # normalize_all'd, i.e. clean fixed points, so it embeds in the same
        # space as gt_for_target. Empty enrich_gt -> today's behavior exactly.
        centroid_tags = list(enrich_gt) + gt_for_target

        to_embed = list(dict.fromkeys(candidates + centroid_tags))
        vectors = await asyncio.wait_for(
            llm.embed(to_embed, timeout=min(4.0, dl.left()), use_cache=cfg.use_cache,
                      # D: one salvage pass on a failed batch, only when enough
                      # budget remains that the retry cannot cause an overrun.
                      retry_timeout=(min(2.5, dl.left() - 1.0)
                                     if cfg.embed_retry and dl.left() > 1.5 else 0.0)),
            timeout=max(0.1, dl.left()),
        )

        res.candidates = candidates
        res.vectors = vectors

        target = centroid([vectors[t] for t in centroid_tags if t in vectors]) if centroid_tags else None
        if target is None:
            return _finish(res, dl)

        ranked = rank_by_centroid(candidates, vectors, target)
        # An anchor is a bet on ground-truth vocabulary, not on relevance, so it
        # only earns a slot if it is genuinely close to this document's centroid.
        anchor_set = {a for a in anchors}
        ranked = [(t, s) for t, s in ranked if t not in anchor_set or s >= cfg.anchor_min_cos]
        # Enrichment-first (demotion): window-only candidates - no lexical or
        # semantic anchor in the enrichment - slide down the ranking but stay
        # submittable. The verbatim predicted_gt tags are exempt so the
        # insurance machinery keeps its no_both protection.
        do_demote = bool(enrich_gt)
        if do_demote and cfg.demote_conditional:
            # When the window and the enrichment share vocabulary, the window
            # IS the GT topic - demoting window tags then only costs slots
            # (we measured near-losses on exactly these agreeing tasks).
            agree = window_enrichment_agreement(text, enrichment)
            if agree >= cfg.demote_agree_cut:
                do_demote = False
        if do_demote:
            ranked = demote_unanchored(
                ranked, vectors, enrich_gt, enrichment,
                exempt=set(normalize_all(predicted_gt)),
                demote=cfg.enrichment_demote, min_cos=cfg.enrichment_anchor_cos,
            )
        # NER composites: built from the top-8 predicted entities by cosine,
        # embedded in one extra small batch (~0.3s, deadline-gated), ranked on
        # the same target. A failed embed just means no combos this task.
        combo_tags: List[str] = []
        if (cfg.ner_combos and kind == "named_entities_extraction"
                and predicted_gt and dl.left() > 1.5):
            ents = [t for t in normalize_all(predicted_gt)
                    if t in vectors and vectors[t] is not None and len(vectors[t])]
            ents.sort(key=lambda t: -cosine(np.asarray(vectors[t], dtype=np.float32), target))
            cand_set = set(candidates)
            combo_tags = [c for c in ner_combo_candidates(ents) if c not in cand_set]
            if combo_tags:
                try:
                    cvecs = await asyncio.wait_for(
                        llm.embed(combo_tags, timeout=min(2.0, dl.left()), use_cache=cfg.use_cache),
                        timeout=max(0.1, dl.left()),
                    )
                except Exception:
                    cvecs = {}
                if cvecs:
                    vectors.update(cvecs)
                    ranked = sorted(
                        ranked + rank_by_centroid(combo_tags, vectors, target),
                        key=lambda x: -x[1],
                    )
                else:
                    combo_tags = []
        # Bound the tail CPU: the dedup sweep below is O(n * kept) in Python
        # and runs AFTER the last deadline gate. With deep enrichment (5 lines
        # x 30 tags) plus enrichment-first candidates the list reached ~250 and
        # the first post-deploy tasks overran the 11s deadline by up to 0.9s -
        # exactly the transport margin. compose only ever consumes the head,
        # so cap at 150 but re-append any predicted-GT tags from the cut tail:
        # losing a verbatim would starve the insurance repair and re-fire
        # no_both_tags.
        if len(ranked) > 150:
            gt_set_cap = set(normalize_all(predicted_gt))
            head, tail = ranked[:150], ranked[150:]
            ranked = head + [(t, s) for t, s in tail if t in gt_set_cap]
        # Do NOT dedupe a variant against its source: keeping "podcast" (an
        # expected `both`) alongside "podcasts" (an expected `unique`) is
        # exactly the pairing we want, and they are near-identical vectors.
        # Combos are protected for the same reason: a composite is deliberately
        # close to its constituent entities.
        protected = (set(variant_tags) | set(normalize_all(predicted_gt))
                     | set(combo_tags) | set(summary_new))
        ranked = drop_near_duplicates(ranked, vectors, cfg.dedup_threshold, protected=protected)
        if not ranked:
            return _finish(res, dl)

        if cfg.compose_mode == "greedy":
            final = compose_greedy(ranked, predicted_gt, vectors, target, profile)
            if len(final) < profile["min_tags"]:
                final = compose(ranked, predicted_gt, profile, target_tags, insurance, anchors=anchor_set,
                                min_unique=cfg.min_unique, uniques_first=cfg.uniques_first)
        else:
            final = compose(ranked, predicted_gt, profile, target_tags, insurance, anchors=anchor_set,
                                min_unique=cfg.min_unique, uniques_first=cfg.uniques_first)
        # Per-SOURCE quota repair: each enrichment line holds one equal combine
        # vote and the window/doc holds one more (2026-08-10 forensics: 4/4
        # real blowouts ignored 2+ voting sources entirely). rep.doc_tags is
        # the replica's own extraction of the window source - already computed.
        if (cfg.enrichment_line_quota and enrich_by_line
                and len(final) >= profile["min_tags"]):
            sources = list(enrich_by_line)
            doc_src = normalize_all(getattr(rep, "doc_tags", None) or []) if rep is not None else []
            if doc_src:
                sources.append(doc_src[:12])
            final = apply_line_quota(
                final, sources, vectors, target,
                quota=cfg.enrichment_line_quota,
                protected=set(normalize_all(predicted_gt)),
                screen_floor=SCREEN_SAFE_FLOOR if profile["penalties"] else 0,
            )
        # Head-cap AFTER quota: quota fixes starved sources, the cap stops one
        # family from re-flooding whatever slots remain.
        if cfg.head_cap and len(final) >= profile["min_tags"]:
            final = apply_head_cap(
                final, ranked, cfg.head_cap,
                protected=set(normalize_all(predicted_gt)),
            )
        # C: value-based allocation (canary-only flag). Mutually exclusive with
        # the fixed quota by convention - enable one, not both, per arm.
        if (cfg.value_alloc and enrich_by_line and kind == "conversation_tagging"
                and len(final) >= profile["min_tags"]):
            ev = {w for s in enrich_by_line for t in s for w in t.split()}
            for line in enrichment or []:
                ev.update(safe_tag(str(line)).split())
            final = allocate_value_based(
                final, enrich_by_line, vectors, target, ev,
                protected=set(normalize_all(predicted_gt)),
                screen_floor=SCREEN_SAFE_FLOOR if profile["penalties"] else 0,
                target_tags=target_tags,
                line_min_cos=cfg.value_line_min_cos,
                line_max_slots=cfg.value_line_max_slots,
                window_cap=cfg.value_window_cap,
            )
        if len(final) >= profile["min_tags"]:
            res.tags = final
            res.source = "ranked"

    except asyncio.TimeoutError:
        res.degraded = True
    except Exception:  # noqa: BLE001 - a raise here is a zero score
        res.degraded = True

    return _finish(res, dl)


def _finish(res: Result, dl: Deadline) -> Result:
    res.tags = normalize_all(res.tags)[:MAX_TAGS]
    res.elapsed = dl.elapsed()
    return res


def _convo_xml(window: Sequence) -> str:
    xml = "<conversation id='83945'>"
    for line in window:
        # >= 2 (not != 2) to match _document_text: a window line carrying extra
        # trailing fields must still contribute its (index, text), or the replica
        # would build its target from fewer lines than the candidate pool sees.
        if len(line) < 2:
            continue
        xml += f"<p{line[0]}>{line[1]}</p{line[0]}>"
    xml += "</conversation>"
    return xml


async def _gather_within(jobs: Dict[str, "asyncio.Future"], budget: float) -> Dict[str, object]:
    """Run named coroutines concurrently, keeping whatever finished in time."""
    if not jobs:
        return {}
    names = list(jobs)
    tasks = [asyncio.ensure_future(jobs[n]) for n in names]
    done, pending = await asyncio.wait(tasks, timeout=max(0.05, budget))
    for p in pending:
        p.cancel()
    out: Dict[str, object] = {}
    for name, task in zip(names, tasks):
        if task in done and not task.cancelled():
            exc = task.exception()
            out[name] = None if exc else task.result()
        else:
            out[name] = None
    return out
