# sn33 — miner optimization layer for ReadyAI (netuid 33)

## What it does

The validator scores every tag you submit as cosine similarity to **one vector**:
the mean of the embeddings of the ground-truth tags it generated from the source
document. Everything here follows from that.

For three of the five task types the validator hands the miner **the exact text
it built ground truth from**:

| task | ground truth built from | miner receives |
|---|---|---|
| `webpage_metadata_generation` | `website_markdown[:1000]` + enrichment | the same 1000 chars + the same enrichment lines |
| `named_entities_extraction` | `transcript_text[:1000]` + enrichment | same |
| `skill_generation` | `skill_markdown[:1000]` | same |
| `conversation_tagging` | **full** conversation (≤300 lines) + enrichment | one ≤10-line window + the same enrichment lines |
| `survey_tagging` | literal `selected_choices` (no LLM) | question + comment only |

So the miner re-runs the validator's own prompts over that text, averages the
resulting tags into an estimate of the target vector, and submits the candidates
closest to it.

## The counterintuitive part

Reproducing ground truth *perfectly* is a disaster. `top_3_mean` is 55% of the
score and counts **only tags that do not string-match ground truth**, padding to
three entries with zeros. Measured: submitting the replica's tags verbatim
scored **0.23** where a paraphrase-heavy answer scored **0.60**.

So the answer is mostly *paraphrases* — semantically central, lexically
different — plus a few verbatim predictions to clear the flat 10%
`no_both_tags` penalty. `compose()` in `pipeline.py` implements exactly that.

## Layout

| file | role |
|---|---|
| `pipeline.py` | the miner: deadline, stages, composition |
| `replica.py` | rebuilds the validator's ground truth from our input |
| `prompts.py` | **upstream** prompts (for the replica) vs **miner** prompts (for candidates) |
| `tags.py` | normalization that survives `validate_tag_set`, centroid ranking |
| `scoring.py` | offline copy of the validator's formula (parity-tested) |
| `extract.py` | local spaCy/YAKE/RAKE fallback, no network |
| `llm.py` | async OpenAI, **batched** embeddings, disk cache |
| `adapter.py` | the single hook into `MinerLib.do_mining` |
| `upstream_prompts/` | vendored copies of the validator's `.j2` files |

`upstream_prompts/` must never be "improved" — the moment it drifts from what
the validator runs, the replica stops predicting the target. Tune
`MINER_*_POOL` in `prompts.py` instead.

## Hard constraints encoded in the code

| constraint | source | consequence if broken |
|---|---|---|
| tags must be lowercase alphanumeric, already normalized | `LlmLib.validate_tag_set:213` | silently deleted (cost this repo 38.5% of survey tags) |
| ≤19 tags | `validate_tag_set:198` | 20+ → validator keeps a random 20 |
| 3–50 chars | `get_clean_tag_set` + `:201` | dropped |
| ≥3 tags (conversation/webpage/skill) | `min_tags` | hard zero |
| ≥3 *unique* tags | `_calculate_stats:141` | top_3_mean zero-padded |
| ≥1 exact match | `PENALTIES.no_both_tags` | ×0.9 |
| answer within ~10s | `dendrite.forward` default 12s | scored as no answer |
| don't bother returning vectors | `format_results` re-embeds | wasted latency |

## Configuration

All via environment; nothing is hard-coded.

```bash
SN33_ENABLED=1            # 0 disables the layer entirely (stock miner runs)
SN33_GT_MODEL=gpt-5.2     # model for the ground-truth replica
SN33_POOL_MODEL=gpt-5.2   # model for candidate generation
SN33_DEADLINE_S=8.0       # global budget; returns best-held answer at expiry
SN33_CALL_TIMEOUT_S=6.5   # per-call ceiling
SN33_COMBINE=llm          # llm | local | none. `local` skips one round trip
                          # (~2s) but measured -0.025 score, 2W/15L. Latency lever,
                          # not a free win.
SN33_POOL_SIZE=40
SN33_TARGET_TAGS=12       # per-task defaults in TASK_PROFILE otherwise
SN33_INSURANCE=6          # verbatim ground-truth predictions to include
SN33_USE_LOCAL=1          # spaCy fallback
SN33_USE_POOL=1
```

## Safety properties

* `mine()` never raises — the caller is a synapse handler and an exception there
  is a zero.
* An answer exists before the first API call (local extraction), so a total
  provider outage still scores ~0.44 instead of 0.
* If the layer produces fewer tags than the task floor it returns `None` and the
  stock miner runs instead.
* Verified by `tests/sn33/test_pipeline_resilience.py`: dead LLM, hanging LLM,
  garbage output, flooded output, empty input.
