# Measured results

Every number here comes from `bench/run.py`. Ground truth is generated with the
**validator's own prompts** (vendored in `sn33/upstream_prompts/`, never the
miner's), at the API default temperature — because upstream
`llm_openai.basic_prompt` sets no temperature, so real ground truth carries
sampling variance. Miner and ground-truth calls use separate cache namespaces,
so they are independent samples rather than the same completion served twice.

Scoring runs the full validator path: `validate_tag_set` (clean → random-20 cull
→ 50-char truncate → the malformed-keyword LLM screen → raw-membership filter) →
re-embed with `text-embedding-3-small` → `GroundTruthTagSimilarityScoringMechanism`
(or the NoPenalty variant for NER).

Comparisons are **paired**: every strategy sees identical ground truth, and the
bootstrap CI is on the per-case difference. That matters — ground-truth noise is
larger than most of the effects being measured.

---

## Headline

| task | share of traffic¹ | current production | this miner | delta | verdict |
|---|---|---|---|---|---|
| `conversation_tagging` (enriched) | 66% | 0.5718 | **0.6408** | **+0.069** | 21W/3L, CI [+0.033, +0.125] **significant** |
| `webpage_metadata_generation` | 15% | 0.5760 | **0.6180** | **+0.042** | 14W/3L, CI [+0.010, +0.080] **significant** |
| `named_entities_extraction` | 12% | 0.5561 | **0.6247** | **+0.069** | 25W/5L, CI [+0.038, +0.100] **significant** |
| `survey_tagging` | 6% | **0.0000** | **0.4804** | **+0.480** | 30W/0L, CI [+0.448, +0.513] **significant** |
| `skill_generation` (new) | n/a² | 0.4711³ | **0.6292** | **+0.158** | 19W/1L, CI [+0.105, +0.212] **significant** |

**Run-to-run spread.** Repeating the same corpus and strategy moves the absolute
mean by roughly ±0.02 (webpage `replica` measured 0.618, 0.632, 0.635 and 0.644
across four runs), because ground truth is re-sampled and `validate_tag_set` is
itself an LLM call. The table quotes the highest-n run for each task. The
**paired** deltas within a run are far tighter than the absolute means and are
what the significance claims rest on; do not compare an absolute mean from one
run against another.

¹ measured from 913 captured production responses (May 2026).
² shipped 2026-07-10, after our last capture, so no traffic share is known.
³ compared against stock upstream, since no tuned production prompt exists for it.

The miner also submits **fewer tags** everywhere (10–12 vs 17–19), which is not
a compromise but part of the win: only 21 tags are ever scored, and `mean` +
`median` are 35% of the formula, so a weak tag is a liability.

---

## The three findings that produced most of the gain

### 1. Survey tasks were scoring a hard zero — every time

`validate_tags.j2` asks the model to keep only *"good English keywords"*, and
`validate_tag_set` returns only that list. All 53 captured survey tasks are a
**Spanish** banking survey whose ground truth is the Spanish `selected_choices`.

Direct measurement against the real screen with gpt-5.2:

```
SPANISH    submitted=6 survived=0  -> []
ENGLISH    submitted=6 survived=6  -> ['easy account opening', 'low fees', ...]
```

End-to-end, answering in Spanish scored `0.0000` with `all_tags_dropped` on
30/30 cases. Answering in English scores `0.4804`. This is a validator bug (see
`SN33_ATTACK_SURFACE.md` C6) — the subnet asks for tags on Spanish text and then
deletes Spanish answers. We comply because the alternative is a certain zero.

### 2. Reproducing ground truth *perfectly* is catastrophic

`top_3_mean` is 55% of the score and counts **only** tags that do *not*
string-match ground truth, padding to three entries with zeros. A strategy that
submitted the replica's tags verbatim scored **0.23** where a paraphrase-heavy
answer scored **0.60**, with `less_than_1_unique` firing on every case.

So the answer is mostly paraphrases — close to the ground-truth centroid,
lexically different — plus a few verbatim predictions to clear the flat 10%
`no_both_tags` penalty. Insurance sweep on webpage (n=17):

| insurance | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| mean | 0.609 | 0.619 | 0.631 | 0.632 | 0.632 |

Plateaus at 6; that is the shipped default.

### 3. Enrichment is the reproducible slice of ground truth

Conversation ground truth is built from the **full** conversation, which we
never see — but the enrichment lines are passed to ground truth and miner
*identically*, so running the upstream enrichment prompt on them reproduces that
part of the target exactly.

The effect is visible in the penalty counts. Same corpus, same strategy:

| | `no_both_tags` fired | mean |
|---|---|---|
| without enrichment (pre-2026-06-12 shape) | 17/24 | 0.5509 |
| with enrichment (current shape) | 3/24 | 0.6408 |

The replica's advantage over tuned prompts grows from +0.015 (not significant)
to +0.069 (significant) once enrichment is present — precisely because
enrichment is what a miner can reproduce.

---

## Calibration against real production scores — the harness is accurate to 0.001

Our own miners ran on netuid 33 as UIDs 17/150/198/253 during 8-15 May 2026, and
their replay logs captured every tag they submitted. The validators' W&B project
retains their real scores from the same window. That makes a direct check
possible: score the exact submitted tags offline, and compare with what the
validators actually recorded.

    harness prediction for the May config (traffic-weighted)  = 0.5111
    REAL mean of our miners with a 0% zero-rate               = 0.5120
    calibration error                                         = -0.0010

Real May results per uid (15 validator runs, 21,954 observations, 241 miners):

| uid | real mean | zero% | rank |
|---|---|---|---|
| 198 | 0.5495 | 0.0% | **10 / 241** |
| 253 | 0.5311 | 1.8% | 50 |
| 17 | 0.5263 | 3.3% | 68 |
| 150 | 0.5041 | 2.3% | 160 |

May population: mean 0.5069, best miner 0.5597, top-10 0.5541, median 0.5119.

Three conclusions, all load-bearing:

1. **The offline numbers can be trusted.** A 0.001 error on real submitted tags
   means the benchmark measures the same quantity the validators do.
2. **Rank is dominated by sampling noise.** Four miners running *identical*
   software spread from rank 10 to rank 160 (0.5041-0.5495). At ~70-110
   observations each, that 0.045 spread is task-draw luck, and it is larger than
   most of the optimizations in this document.
3. **The subnet moved between May and August.** Population mean 0.5069 -> 0.6039,
   best miner 0.5597 -> 0.6961. Comparing any May measurement to an August
   leaderboard is invalid; enrichment landed in June and lifted the whole field.

Calibrated, the config in this repo predicts **0.627** against the old config's
**0.511** - a **+0.116** improvement on the same scale. In May conditions that
would have been first outright. Against the August distribution it lands near
the median with the top cluster ~0.06 above; that offset cannot be verified
offline, because the reason the population shifted is not visible in the logs.

## Real production score distribution

`bench/wandb_scores.py` reads what validators actually recorded in their public
W&B project. Four netuid-33 runs from 2026-08-06, **6,738 scored responses across
258 miners**:

| | score |
|---|---|
| best miner (mean of 26 responses) | **0.6961** |
| top-10 miners | 0.6780 – 0.6961 |
| median miner | 0.6196 |
| all nonzero responses | mean 0.6365, median 0.6788, max 0.8124 |
| zero rate, all miners | 5.4% |
| zero rate, top-25 miners | **0.0–0.4%** |

Two things follow.

**The top cluster is statistically indistinguishable.** At ~26 observations per
miner with per-response sd ≈ 0.08, the standard error on a miner's mean is
**±0.016**. Rank 1 to rank 10 spans 0.018 — about one standard error. Combined
with rank-based weights where rank 10 pays within 5% of rank 1, the goal is to
*reach the top cluster*, not to be #1.

**Reliability is the visible differentiator.** Every miner in the top 25 has an
effectively zero zero-rate, against a 5.4% population average.

Weighting our per-task results by the observed traffic mix gives ~**0.627**
against a production baseline of ~**0.537** — placing the miner above the median
(0.6196) but still ~0.03–0.05 short of the top cluster. That residual is real
and I have not closed it; the failed attempts below are the evidence.

## Negative results

Recorded because they bound where the remaining gap is *not*.

| idea | result | keep? |
|---|---|---|
| Pool 3 or 5 independent ground-truth draws to denoise the centroid | −0.004 / +0.001, not significant | no (default 1; saves cost) |
| Greedy composer that maximises the *estimated* score directly | +0.004, 6W/11L, not significant — but halves variance (sd 0.080→0.040) and lifts the worst case (0.330→0.560) | available, not default |
| Corpus-frequency vocabulary anchors | +0.0013 | on (free, non-negative) |
| Lexical variants of predicted ground-truth tags | +0.007, 12W/5L, not significant | on (free, consistent direction) |
| `SN33_COMBINE=local` (drop one LLM call) | **−0.025**, 2W/15L, significant | no |
| Look up live conversations in ReadyAI's public corpus | **0 of 4,479 captured lines matched** — the released dataset is genuinely retired and disjoint from live traffic | dead end |
| KeyBERT for local extraction | unavailable: needs torch ≥2.4, repo pins `torch==2.1.1` | rejected |

The variants idea is worth a note because the isolated measurement was
convincing and the end-to-end result was not. Against a real centroid,
morphological variants scored 0.5895 mean cosine versus 0.5095 for semantic
paraphrases, implying +0.035 on the final score. End-to-end it delivered +0.007,
because the candidate pool already contained high-cosine options and the ranker
was already finding them. An isolated component win does not survive contact
with the rest of the pipeline unless the pipeline was actually short of that
component.

## Phase 1: local extraction libraries

Scored as a standalone fallback answer through the real validator path, not
judged by eye.

| backend | mean score | ms/doc | cold start | verdict |
|---|---|---|---|---|
| **spaCy** `en_core_web_sm` | **0.4400** | 30.5 | 1.1s | **chosen** — best output, and its NER covers a second task type |
| YAKE | 0.4343 | 7.8 | 0.13s | viable, fragmentary phrases |
| RAKE | 0.4321 | 1.3 | 1.2s | viable, most fragmentary |
| KeyBERT | **unavailable** | — | — | **rejected**: requires torch ≥2.4, repo pins `torch==2.1.1` |

The three working extractors are statistically indistinguishable. The number
that matters is that a **no-network fallback scores 0.44** against ~0.62 for the
full pipeline — 71% of full performance for 30ms and no API call. That converts
a provider outage from a zero into a scoring round, which matters more than the
difference between the libraries.

Sample output confirms the quality ordering (spaCy noun phrases vs RAKE
fragments like `real quick`, `go back`).

---

## Latency (bench/timing.py, cache off, as production runs)

The validator calls `dendrite.forward()` with no timeout argument, so
bittensor's **12s** default applies. `--neuron.timeout` (10) exists but is never
passed. A late answer scores exactly like no answer, so the tail is what
matters.

At the shipped 8s deadline:

| task | mean | p50 | p95 | max | >10s | >12s | tags | deadline hits |
|---|---|---|---|---|---|---|---|---|
| conversation_tagging | 5.26 | 5.32 | 7.12 | 7.12 | 0 | 0 | 12.0 | 0 |
| webpage_metadata_generation | 5.13 | 5.19 | 5.46 | 5.46 | 0 | 0 | 12.0 | 0 |
| named_entities_extraction | 6.03 | 5.87 | 7.30 | 7.30 | 0 | 0 | 10.0 | 0 |

**Worst observed 7.30s**, with `source=ranked` and `degraded=0` on every
request — the pipeline completes its full ranking pass rather than being cut
short, leaving ~4.7s of the 12s budget for network transport in both directions
and the validator's own overhead.

An earlier run at a 9s deadline showed a 9.00s maximum, which was the deadline
truncating work rather than a slow pipeline: tightening the budget to 8s made
the miner both faster *and* more complete, because it stopped racing the clock.
spaCy warm-up is 0.6s, paid once at startup.

Worst-case behaviour is covered by tests rather than measurement: dead provider,
hanging provider, garbage output and flooded output all still return a valid
list inside the deadline (`test_pipeline_resilience.py`).

### Cost levers, measured

| lever | latency saved | score cost |
|---|---|---|
| `SN33_COMBINE=local` (skip the combine call) | ~2s, one call | **−0.025** (2W/15L, significant) |

The combine call earns its keep; it is a latency lever, not a free win.

## Fidelity limits

Stated so the numbers are not over-read:

* **Ground truth is generated by our OpenAI account, not a validator's.** Same
  model (`gpt-5.2`, the committed default), same prompts, same temperature —
  but different samples. Paired comparison removes most of this; absolute
  values carry roughly ±0.01 of ground-truth noise.
* **Conversation enrichment is synthetic.** Our corpus predates enrichment, so
  the lines are model-written imitations of search results. The *mechanism*
  (identical lines to ground truth and miner) is exact; the content
  distribution is not. Scores with and without enrichment are not comparable to
  each other.
* **Survey ground truth is synthetic.** `selected_choices` is never sent to
  miners, so it cannot be replayed; choices were reconstructed by a model acting
  as the survey coder. The 0.00 → 0.48 result does not depend on this — it comes
  from tags surviving the validator's screen at all, which was measured directly
  against the real screen.
* **Skill documents are synthetic.** The task type postdates our capture.
  Ground truth is built from the same 1000 characters the miner receives, so the
  mechanism is faithful even though the documents are generated.
* **Sample sizes are 17–30 cases.** Enough for the effects reported as
  significant; not enough to resolve differences below ~0.02.

## Reproducing

```bash
pytest tests/sn33/ -q                     # 136 tests incl. scorer parity vs the validator's class
python bench/run.py --kind conversation_tagging --n 24 --enrichment 2 \
    --strategies prod,replica
python bench/run.py --kind webpage_metadata_generation --n 22 \
    --strategies prod,replica:insurance=2,replica:insurance=6
python bench/lib_bench.py --kind conversation_tagging --n 10
python bench/timing.py --n 5 --repeat 2
```

Results are cached in `data/sn33_cache/`, so a re-run costs almost nothing and
is directly comparable to the run before it.
