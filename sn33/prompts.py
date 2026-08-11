"""Prompt templates.

Two families live here and the distinction is the whole point:

``upstream(...)``
    Byte-identical copies of the validator's own ``.j2`` files, vendored from
    ``git show HEAD:conversationgenome/llm/prompts/*`` into
    ``sn33/upstream_prompts/``. These are used to *replicate the ground truth*.
    They must never be "improved" - the moment they drift from what the
    validator runs, the replica stops predicting the target.

``MINER_*``
    Our own prompts, used only to widen the candidate pool. These are free to
    change because they are always filtered by a centroid built from the
    upstream replica.

An earlier round of tuning in this repo edited the shared ``.j2`` files in
place, which meant the offline benchmark generated its "ground truth" with the
*tuned* prompt and scored the tuned miner against it. That measures self
agreement, not score. Keeping the two families in separate directories is what
prevents that class of mistake.
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream_prompts")
_env = Environment(loader=FileSystemLoader(_DIR), undefined=StrictUndefined)


def upstream(name: str, **kwargs) -> str:
    """Render a vendored upstream template exactly as the validator would."""
    return _env.get_template(name).render(**kwargs)


# --- upstream renderers, one per validator ground-truth path ---------------

def gt_conversation(convo_xml: str, coding: bool = False) -> str:
    tpl = "conversation_to_metadata_coding.j2" if coding else "conversation_to_metadata.j2"
    return upstream(tpl, conversation_to_analyze=convo_xml)


def gt_website(website_content: str, coding: bool = False) -> str:
    tpl = "website_to_metadata_coding.j2" if coding else "website_to_metadata.j2"
    return upstream(tpl, website_content=website_content)


def gt_enrichment(enrichment_content: str, coding: bool = False) -> str:
    tpl = "enrichment_to_metadata_coding.j2" if coding else "enrichment_to_metadata.j2"
    return upstream(tpl, enrichment_content=enrichment_content)


# Instruction 5 of the upstream enrichment template caps extraction at 10 tags.
# That cap is the binding constraint on how much of the target we can see:
# measured 2026-08-08, enrichment supplies 87.7% of the validator's ground-truth
# tags (17.8 of 20.2) while the 300-line conversation supplies 2.0.
_ENRICHMENT_LIMIT_LINE = {
    False: "5.  **Limit to most important:** Return at most 10 of the most important and relevant tags.",
    True: "6.  **Limit to most important:** Return at most 10 of the most important and relevant technical tags.",
}


def gt_enrichment_deep(enrichment_content: str, n: int = 30, coding: bool = False) -> str:
    """The upstream enrichment prompt with ONLY its tag cap widened to ``n``.

    This is deliberately not a ``MINER_*`` prompt: it keeps the validator's
    wording, role and output format so the extra tags come from the same
    distribution as the ground truth, just further down it.

    The output is **candidate-pool material only**. Deep tags string-match real
    ground truth 10.3% of the time against 26.2% for upstream-depth tags, so
    routing them into ``predicted_gt`` or the anchors dilutes the exact-match
    insurance and fires the flat x0.9 ``no_both_tags`` penalty. See
    ``pipeline.mine``.
    """
    base = gt_enrichment(enrichment_content, coding=coding)
    old = _ENRICHMENT_LIMIT_LINE[bool(coding)]
    if old not in base:
        # Fail loudly. If upstream renumbers or rewords this line, a silent
        # str.replace no-op would hand back the plain 10-tag prompt and the
        # feature would measure as inert rather than as broken.
        raise ValueError(
            "upstream enrichment template no longer contains the tag-cap line "
            f"{old!r}; re-vendor sn33/upstream_prompts/ and update _ENRICHMENT_LIMIT_LINE"
        )
    num = old.split(".", 1)[0]
    new = (
        f"{num}.  **Be exhaustive:** Return at most {n} tags. Start with the most "
        "important and relevant, then continue with broader themes, alternative "
        "phrasings of the same concepts, adjacent subtopics, and the field or "
        "industry the content belongs to."
    )
    return base.replace(old, new)


def gt_enrichment_ner(enrichment_content: str) -> str:
    return upstream("enrichment_to_named_entities.j2", enrichment_content=enrichment_content)


def gt_transcript_ner(raw_transcript: str) -> str:
    return upstream("raw_transcript_to_named_entities.j2", raw_transcript=raw_transcript)


def gt_webpage_ner(raw_webpage: str) -> str:
    return upstream("raw_webpage_to_named_entities.j2", raw_webpage=raw_webpage)


def gt_skill(skill_markdown: str) -> str:
    return upstream("skill_to_metadata.j2", skill_markdown=skill_markdown)


def gt_combine(tag_sets: Sequence[Sequence[str]]) -> str:
    """The validator's combine step.

    Note this template is entity-flavoured ("People, Organizations, Locations,
    Laws/Statutes, Budgets, Specific Projects") even when it is merging topic
    tags for a conversation - that bias is part of the target and the replica
    must keep it.
    """
    entities_str = ""
    for i, tags in enumerate(tag_sets):
        entities_str += f"<set{i}>{', '.join(tags)}</set{i}>\n"
    return upstream("combine_named_entities_prompt.j2", entities_str=entities_str)


def gt_survey(survey_question: str, free_form_comment: str) -> str:
    return upstream("survey_tag.j2", survey_question=survey_question, free_form_comment=free_form_comment)


# --------------------------------------------------------------------------
# Miner-side candidate generation
# --------------------------------------------------------------------------

# Conversation windows are a *slice* of a document whose ground truth was built
# from the whole thing, so the pool has to cover themes the slice only hints at.
MINER_CONVERSATION_POOL = """\
Below is an excerpt from a longer conversation or transcript.

Produce {n} candidate topic tags. They will be filtered afterwards, so breadth \
matters more than precision - include both the obvious subject-matter tags and \
plausible broader themes that would describe the FULL discussion this excerpt \
came from.

Cover, in this order:
1. The core subjects actually discussed in the excerpt.
2. The broader field / domain each subject belongs to.
3. Alternative phrasings a different writer would use for those same subjects.
4. Named entities that carry the topic (people, organizations, products, places).

Rules for every tag: lowercase, 1-4 words, plain English dictionary words, no \
punctuation, no hyphens, no abbreviations, no duplicates, no participant names \
or greetings.

CRITICAL: write every tag in ENGLISH, even when the source text is in another language. The validator strips accented characters before scoring (turning "construção" into "constru o") and then deletes anything that is not a good English keyword - so a tag in the source language is discarded and the slot is wasted. Translate the concept instead.

Return ONLY a comma-delimited list.

Excerpt:
{content}
"""

# Webpage / skill / transcript: we hold the *same* text the ground truth was
# built from, so the pool should stay tight to the document.
MINER_DOCUMENT_POOL = """\
Below is a document.

Produce {n} candidate topic tags describing what this document is about. They \
will be filtered afterwards, so include both the central topics and reasonable \
alternative phrasings of them.

Cover: the primary subject, the industry or field, the document's purpose, key \
entities named in it, and the concepts a librarian would file it under.

Rules for every tag: lowercase, 1-4 words, plain English dictionary words, no \
punctuation, no hyphens, no abbreviations, no duplicates.

CRITICAL: write every tag in ENGLISH, even when the source text is in another language. The validator strips accented characters before scoring (turning "construção" into "constru o") and then deletes anything that is not a good English keyword - so a tag in the source language is discarded and the slot is wasted. Translate the concept instead.

Return ONLY a comma-delimited list.

Document:
{content}
"""

# Survey ground truth is not LLM output at all - it is the literal
# `selected_choices` strings from the survey record
# (SurveyTaggingTaskBundle.py:68). So the task is to guess the wording of the
# answer options this respondent picked, not to tag the comment.
#
# THE TAGS MUST BE IN ENGLISH even when the survey is not. Before scoring, the
# validator runs every tag through `validate_tags.j2`, which asks a model to
# split them into "good English keywords" and "malformed keywords" and keeps
# only the former (LlmLib.validate_tag_set:202-213). Measured against the real
# screen: 0 of 6 Spanish tags survived; 6 of 6 English equivalents survived.
# The captured production traffic is a Spanish banking survey, and this repo's
# deployed miner answered it in Spanish - scoring a hard zero on every one.
#
# Ground truth stays in the survey's language, so the English tags are scored
# by cross-lingual similarity to that centroid. That is worth far more than a
# perfectly-worded Spanish tag that is deleted before it is ever embedded.
MINER_SURVEY_POOL = """\
A survey respondent was asked a question and left a free-text comment. The \
survey may be in any language.

The answer options were multiple-choice. Predict which options this respondent \
selected, then give close variants of each.

Produce {n} candidates, ordered most-likely first. Each should read like a \
survey answer option (a short noun phrase), not a summary of the comment.

CRITICAL: write every candidate in ENGLISH, even if the question and comment \
are in another language. Translate the concept rather than transliterating it.

Rules: lowercase, 1-5 words, ordinary English dictionary words, no accents, no \
punctuation, no duplicates.
Return ONLY a comma-delimited list.

Question: {question}
Comment: {comment}
"""


def miner_theme(convo_text: str, enrichment: str, n: int = 6) -> str:
    """Broad umbrella theme tags for the whole conversation.

    Pool-only material. Broad category labels sit nearer the ground-truth
    centroid than specific details do, and being plain dictionary words they are
    also screen-safe (the validator's LLM screen keeps them). Kept short and
    English so `is_probably_english` and `screen_safe` pass.
    """
    return (
        f"Name {n} BROAD theme tags for the overall subject areas of this "
        f"conversation - category labels a librarian would file it under, not "
        f"specific names or details. Each 1-3 plain English dictionary words, "
        f"lowercase, comma-separated, no commentary.\n\n"
        f"CONVERSATION:\n{convo_text[:1500]}\n\nCONTEXT:\n{enrichment[:1200]}"
    )


def translate_to_english(tags_csv: str) -> str:
    """Translate a comma-separated tag list to English, concept-for-concept.

    Language-mismatched tasks (Spanish conversations, all survey traffic) produce
    non-English predicted-GT tags that rank high on the centroid but are DELETED
    by the validator's English screen. text-embedding-3-small is multilingual, so
    an English translation sits ~0.01 from the original in cosine to the GT
    centroid (measured) while SURVIVING the screen - a large net win.
    """
    return (
        "Translate each tag to its natural English equivalent (the concept, not a "
        "literal word-by-word translation). Keep tags already in English unchanged. "
        "Return the SAME NUMBER of tags, same order, lowercase, comma-separated, "
        "1-4 plain English dictionary words each, no commentary.\n\n" + tags_csv
    )
