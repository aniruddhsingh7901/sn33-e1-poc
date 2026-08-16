# SN33 miner — project knowledge

Bittensor subnet 33 (ReadyAI / conversation-genome). We run a miner; everything
validator-side in this repo is read-only reference.

**Read `docs/SN33_README.md` first.** This file holds the facts and the mistakes
that are expensive to rediscover.

---

## The one number that matters

Measured 2026-08-07 from the validators' W&B, top-10 miners vs us:

```
raw tag quality (adjusted_score):  0.6929 vs 0.5347  -> 0.1582  = 96% of the gap
penalty loss                    :  0.0016 vs 0.0076  -> 0.0060  =  4% of the gap
TOTAL final gap                                         0.1642
```

**96% of the gap is raw tag quality.** Penalty tuning cannot close it — even a
perfect penalty rate is worth +0.006. Work on `adjusted_score`, which is set
entirely by how close our tags sit to the centroid of the validator's
ground-truth tag embeddings.

Do not spend another day on penalties. That mistake has already been made.

---

## Scoring, verified against the code

```
adjusted = 0.55*mean(top-3 UNIQUE) + 0.25*mean(all) + 0.10*median(all) + 0.10*max
final    = adjusted * penalties
```

* every tag is scored as cosine to ONE vector: the mean of the ground-truth tag
  embeddings (`text-embedding-3-small`, 1536 dims, hardcoded `llm_openai.py:15`)
* **unique** = our tags NOT string-matching a ground-truth tag; **both** = exact matches
* `top_3_mean` pads with zeros below 3 unique tags - the most expensive mistake available
* only 21 tags are ever scored; at >=20 cleaned tags the validator keeps a **random 20**
* validators set **no temperature** (API default 1.0), so ground truth is one
  random draw, not a fixed answer. Our replica cannot match it exactly.

Penalty multipliers (`utils/constants.py`): no exact match x0.9, max<0.2 x0.5,
<2 tags x0.2, <1/<2/<3 unique x0.85/x0.9/x0.95.

## Hard constraints on any tag we emit

| rule | source | if broken |
|---|---|---|
| already lowercase alphanumeric (fixed point of `get_safe_tag`) | `LlmLib.validate_tag_set:191` | silently deleted |
| <=20 tags | `validate_tag_set:176` | 21+ -> random cull. At exactly 20 `random.sample(range(20),20)` is a SHUFFLE and loses nothing; loss starts at 21. We ship 12 anyway - more tags dilute mean+median. |
| 3-50 chars | `get_clean_tag_set` + `:179` | dropped |
| **English only** | `validate_tags.j2` keeps "good English keywords" | non-English deleted -> hard zero |
| answer inside ~11s | `dendrite.forward` default 12s | scored as no answer |
| don't return vectors | validator re-embeds | wasted latency |

## Non-English is a trap

`get_safe_tag` strips everything outside `[a-zA-Z0-9\s]`:

```
construção   -> 'constru o'   (word shredded)
日本語のタグ   -> ''             (erased)
```

Measured against the real screen: **0 of 6** Spanish or Portuguese tags survived,
**6 of 6** English equivalents did. A Portuguese conversation scored a hard
**0.0000** before the guard existed. All survey traffic is Spanish.

So: always answer in English, whatever the source language. `no_both_tags`
(x0.9) is then unavoidable on those tasks - accept it, 0.43 beats 0.

Non-ASCII **symbols** (emoji) are harmless - they strip cleanly. Only non-ASCII
**letters** shred words. The guard keys on `isalpha()` for that reason.

---

## Configurations tried in production (all worse than the first)

| config | conversation n | final | adjusted | penalties |
|---|---|---|---|---|
| **A** insurance 6, 12 tags, variants on | 28 | **0.5523** | **0.5587** | 18% |
| B insurance 3, 12 tags, variants off | 14 | 0.4951 | 0.5101 | 36% |
| C insurance 6, 15 tags | 11 | 0.5149 | 0.5254 | 27% |
| D insurance 6, 12 tags + MIN_UNIQUE 5 | 10 | 0.4878 | 0.4958 | 20% |

**A is the best measured config.** Lessons paid for in production hours:

1. ~~**`MIN_UNIQUE_TARGET` above 3 lowers tag quality.**~~ **Retracted 2026-08-08.**
   Measured directly on 75 real-GT windows: 5 vs 4 vs 3 differ on 10 windows
   (13% - the branch does fire), and on exactly those the effect is a coin flip,
   mean -0.0004, best +0.0023, worst -0.0036, W/L 5/5. Config D's 0.4878 (n=10)
   was NOT caused by this knob. `MIN_UNIQUE_TARGET` is inert, like `insurance`;
   both are now `Config` fields so they can be A/B'd rather than believed.
2. **Cutting `insurance` raises `no_both_tags`.** Fewer verbatim ground-truth
   predictions means fewer exact matches. 6 worked; 3 doubled that penalty.
3. **Never change two variables at once.** B changed insurance *and* variants
   together and the result was uninterpretable for hours.
4. **~25 scored tasks minimum** before believing any comparison. A 3-sample mean
   swung 0.16 with two extra samples.

## Things that do not help (measured, do not redo)

| idea | result |
|---|---|
| pooling 3-5 ground-truth draws | -0.004 / +0.001, nothing |
| greedy score-maximising composer | +0.004, loses 11 of 17 |
| corpus vocabulary anchors | +0.0013 |
| lexical variants | +0.007 offline, not significant |
| `combine=local` (skip a call) | **-0.025**, significant loss |
| abstraction ladders as a family | -0.138 top-3 vs our own pool; miner already at the optimum |
| improving the centroid estimate | +0.0106 alignment -> **+0.0001** adjusted. A *perfect* centroid is +0.0036. Stop. |
| 3-way blends / LLM-fused phrases | -0.029 and -0.318; 3-blends blow the 50-char cap |
| sub-phrase transforms (drop_head etc.) | all worse than number inflection, 0/4 conversations each |
| maximising unique tags | +0.043 *adjusted* but **-0.024 final** - no_both x0.9 fires 38/38 |
| more candidates per se | +110 candidates: -0.0015. Provenance, not width |
| more shipped tags per se | 12 -> 18.7: **-0.019**; mean cosine 0.628 -> 0.559 |
| `variants.expand` per_tag>2 | 0.0 extra tags/window. Family exhausted |
| expanding the testnet corpus | 228 reserve calls, 4 query variants: **0 novel sources** |
| ReadyAI public HF corpus for lookup | **0 of 4,479** captured lines matched - retired data, dead end |
| KeyBERT | needs torch >=2.4, repo pins 2.1.1 |
| **ZSinvert** / zero-shot inversion (arxiv 2504.00147, checked 2026-08-08) | Genuinely embedding-agnostic, so it WOULD accept `text-embedding-3-small`. But: **90 seconds per inversion on an A40 GPU** (their number), beam 30 x top-k 30 encoder queries per token step (thousands of paid API calls per tag), needs a Qwen2.5-3B correction model, evaluated only on Contriever/GTE/GTE-Qwen/GTR - no OpenAI encoder - and reconstructs 32-token documents, not 2-word keywords. Off by ~4 orders of magnitude against an 11s CPU budget. Its objective `argmax_x cos(E(x), target)` is the same one our ranker already optimises, which confirms the architecture; buying a deeper search costs latency we do not have. |
| **vec2text** (embedding inversion, tested 2026-08-08) | supports ONLY `text-embedding-ada-002` and `gtr-base` (`vec2text/api.py:13-38`); `text-embedding-3-small` appears nowhere in the package. An ada-002 inverter recovers text aimed at ada-002's vector space, which says nothing about where it lands in 3-small's. Training our own is a GPU research project against a ~+0.03 ceiling, and inference is far too slow for the 11s budget. Also pulls torch 2.13 + the CUDA stack. |

## Latency

The replica getting cut off is the most expensive non-zero failure:

```
source=ranked (replica completed)  0.5517
source=pool   (replica cut off)    0.1994
```

Deadline 8s truncated 15% of tasks once OpenAI slowed; **11s** cut that to ~6%.
Measured overrun is 0.06s, so 11s leaves ~0.9s for transport inside the 12s
synapse budget. A timeout is not fatal - 408 is in `RETRY_STATUS_CODES` and
earns one retry - so spending the budget beats truncating.

---

## Discord competitive-intel sweep (6903 messages, resolved 2026-08-08)

Read the full miner Discord. Two things worth keeping:

**The 3-large vs 3-small scare - RESOLVED, do not re-flag.** The repo has
contradictory config: `ConfigLib.py:25` and `TaskLib.py:17` default to
`text-embedding-3-large`, and Kat (team) said the validator moved to 3-large in
ReadyAI 2.6.26 (Oct 2024). BUT the live scoring path is 3-small: `get_llm_backend()`
-> `LlmOpenAI()` -> `__init__` hardcodes `self.embedding_model =
"text-embedding-3-small"`, ignoring the config; `get_vector_embeddings(dimensions=1536)`.
The 3-large refs are vestigial 2024 config. Empirical clincher: our 3-small bench
calibrated to -0.001 vs real production - impossible in 3072-dim space. We are in
the correct geometry. GT model likewise defaults to gpt-5.2 (`llm_openai.py:14`),
matching ours; the Discord "GPT-4o/gpt-3.5" mentions are historical.

**The one untested lever (low prior): verbatim-GT-match count.** leaf's 2024 logs
(18/23 tags matching GT -> ADJ 0.990 vs 6 -> 0.953) hint that MORE both-tags help.
We only tested insurance 6 vs 3. BUT this is ada-002/pre-current-formula era, and
under the VERIFIED current formula more both-tags are barred from the 55%
top-3-UNIQUE term (all-verbatim measured 0.27, zero-pad). A cheap insurance level
6/10/14 bench would settle it; prior is it does not help. Everything else in the
Discord corroborates our approach or names an exploit we already handle.

## Statistics: what this corpus can and cannot resolve

The 75 windows are 4 conversations. `bench/harness.py:paired_delta` bootstraps
*windows*, so it treats 36 slices of one podcast as 36 independent draws.

```
                       delta      CI95              halfwidth
naive  (n=75)        -0.0675  [-0.0777,-0.0564]      0.0107
clustered t(3)       -0.0648  [-0.1141,-0.0155]      0.0493   <- honest
```

* naive is **4.6x too narrow at n=75, 10x at n=40** (ICC 0.507, design effect 8.7)
* **minimum resolvable effect: ~0.049** unpaired; ~0.017 for a strictly paired
  swap with low between-conversation spread
* below that and not to be quoted as measured: perfect centroid +0.0045, corpus
  anchors +0.0013, offline lexical variants +0.007, sub-phrases +0.012-0.016
* resolving a 0.012 effect needs **25-30 distinct conversations**. We have 4 and
  the testnet pool provably cannot supply more - mainnet A/B is the only route.

**Corpus bug, fixed 2026-08-08:** `expand_windows` emitted conversations
back-to-back (counts 36/21/16/2), so `--n 40` was 36 windows of one podcast with
the fourth conversation absent. Every `--n 40` result before this date was a
single-conversation experiment. Windows are now interleaved.

## What the ground truth is actually made of

Measured on all 4 conversations - of the ~20 ground-truth tags:

```
from enrichment only    17.8   87.7%
from the 300 lines       2.0    9.9%
invented by combine      0.5    2.5%
```

`combine_metadata_tags` sees ONE conversation tag set against FOUR-to-SIX
enrichment sets, so enrichment outvotes the transcript. The conversation
generated 40 candidate tags and 3 survived. This is why the window barely moves
our answer, and why enrichment work pays and window work does not.

(The exact 88/10 split is testnet-specific - its enrichment is unrelated
placeholder material, so `both` is 0. On mainnet enrichment is topical and the
sources overlap. The 1-vs-N vote ratio is in the code, not the data.)

## Non-English conversations: translate, do not just drop (2026-08-08)

Live data showed a Spanish conversation (Univision, Latino politics) where our
answer was ~13/20 Spanish tags (`partido republicano`, `residencia permanente`)
that the validator's English screen DELETES. The GT itself is mostly Spanish
(20/50 screen-safe), so Spanish tags rank highest on the centroid - and die.

Fix: `Config.translate_non_english` (default ON). When predicted_gt is >=30%
non-English, translate every non-English CANDIDATE to English via one gated LLM
call (`prompts.translate_to_english`); the centroid stays faithful (built from
the untranslated predicted_gt). text-embedding-3-small is multilingual, so a
translation sits ~0.01 from the original in cosine - measured on the real
Univision task: survivors 6->12, adjusted 0.5485->0.5650 (+0.017). Only helps
non-English conversations (a minority), so ~+0.003 average; gated on the
deadline (translating English is a no-op, so misclassification is harmless), and
a failed translation leaves the originals with the screen-safe floor backstopping
the zero. Detection is on predicted_gt, NOT the wide candidate pool - the few
Spanish GT tags dominate the selected answer even at <30% of all candidates.
Survey already answers in English via its own pool prompt.

## The zero cause and the screen-safe floor (fixed 2026-08-08)

We were scoring occasional hard **0.0** on conversation/webpage tasks. Root cause,
found by joining our submissions to the validator's `bt_log`:

```
task cbc65e23:  original tags: 18  ->  tags: 0  ->  score 0.0
```

The validator's LLM "good English keywords" screen (`validate_tags.j2`) deletes
"abbreviations, compound words not in the dictionary, and typos". On an
ACRONYM-HEAVY task (e.g. EV charging: nacs, ccs, ev, teslas) it deleted ALL of
our tags -> below min_tags(3) -> the response is DISCARDED (a hard 0.0, not a
penalty). The deterministic cleaning drops nothing (verified 0/270); the LLM
screen is the culprit, and it is non-deterministic so we cannot replicate it.

**Fix: a screen-safe FLOOR.** `tags.screen_safe(tag)` certifies a tag only if
EVERY word is a real dictionary word (bundled 64k lowercase wordlist) - acronyms,
non-dictionary compounds ("multifamily"), and modern words absent from the old
dictionary ("onboarding") are NOT certified (conservative by design). `compose`
guarantees `SCREEN_SAFE_FLOOR = 7` such tags in every answer (except NER, which
uses validate_named_entities_tag_set with no LLM screen). Verified on 12
acronym-heavy mainnet tasks: minimum 7 screen-safe tags -> even if the screen
deletes every acronym, >=7 survive -> a zero is now impossible.

Cost: forcing 7 dictionary tags may cost ~0.01 cosine on a task that was not
being screened, but that is trivial against a 0.0. Shipped with target_tags 20.

## Reverse-engineered from W&B: what top miners actually do (2026-08-08, n=27,978 scored tasks, 48h, 269 miners)

The validator's `bt_log` logs the FULL per-miner scoring breakdown - `ADJ SCORE`,
`Num Tags`, the whole `Unique Tag Scores` cosine list, median, mean. No tag
STRINGS, but the complete numeric signature of every miner's answer.

We ranked **233 of 269**. 0% zero-rate for us and the top - it is pure tag
quality, not reliability. Top-10 vs us, joined per task:

```
                TOP-10    US       gap
num_tags        17.3      9.5             <- the WHOLE field submits ~18; we are the outlier at 12
unique tags     13.8      6.7
TOP-3 unique    0.6891    0.5634   -0.126  <- the 55% term = the whole game
mean unique     0.6505    0.5180   -0.132
median(all)     0.6570    0.5417   -0.115
mean(all)       0.6445    0.5416   -0.103
```

Every tag they submit sits ~0.12 closer to the target - not just the top 3, the
entire distribution is shifted up. And they carry ~14 unique tags to our 6.7.

**Podcast-lookup is RULED OUT.** The gap is not concentrated in conversation; it
is present in every task type and is LARGER where there is no podcast to look up:

```
task_type                    top10    us     gap
survey_tagging               0.642   0.410  -0.233   <- our worst (Spanish->English)
skill_generation             0.660   0.490  -0.170
conversation_tagging         0.652   0.516  -0.136
webpage_metadata_generation  0.647   0.574  -0.073
named_entities_extraction    0.684   0.642  -0.042   <- our closest (NER, no penalties)
```

So it is general tag quality, which is REPLICABLE - not a conversation exploit.
Top-5 consistency: mean 0.65-0.67, sd 0.08-0.14, occasional 0.000 - looks like
robust generation, not a bimodal lookup.

**Implications, in priority order:**
1. **Submit ~18 tags with ~14 unique, not 12/6.7.** The entire field does; this
   is population-level evidence, not a below-resolution 4-podcast effect. Our
   offline "12 beats 18" was -0.019, below the 0.049 corpus resolution, on OUR
   weak pool. 18 only helps if tags 13-18 are strong uniques - so raise
   target_tags AND fill the slots from variants + deep enrichment. A/B 12 vs 18
   on mainnet is the #1 test.
2. Fix survey (0.410) and skill (0.490) - our worst types. Survey is the
   Spanish->English guard.
3. The 0.12 per-tag gap is the candidate POOL. deep enrichment (+0.024) is a
   start, not a close. Top miners generate uniformly-central tags; we scatter.

Raw data: data/wandb_deep/rows.jsonl (7,205 ADJ breakdowns).

## CHECKPOINT 2026-08-10: three flags live, and the next conversation level

Live on UID 33 since 08-09: enrichment-first (conv), NER combos, enrichment-first
(webpage), OpenAI priority tier. Night verdict (n=67): overall 0.612 -> 0.638,
NER +0.102 (n=34, proven), webpage +0.046 (flag-attribution pending), conv 0.585
abs / cohort-relative -0.031 -> -0.006 after excluding one truncation. Conv wins
hard tasks 5/5, loses narrowly on easy ones.

Forensics on the cohort losses found the NEXT LEVEL, one shared root cause:
**per-line allocation**. The validator's GT is a 1-vs-N combine where EACH
enrichment line is one equal vote set, but our composer allocates slots by
global cosine to a blended centroid - majority-topic lines soak up slots
(11 testing tags vs 2 for the HIV line and 1 for the RICE line on the 13:26
loss), minority lines go under-covered, and generic tags beat line-specific
ones (18:48 loss: generic 'investment optimization' cloud vs the enrichment's
'hedging in real estate finance'). The replica already computes PER-LINE tag
lists (rep.enrichment_tags is a list of lists) and we flatten them - keeping
them per-line enables a quota compose: >=2-3 tags from each line's own
extraction, rest by cosine. Offline A/B next; also test demote 0.90->
conditional (only when window/enrichment vocabularies disagree) and
variants_per_tag reduction on coherent-enrichment tasks (14:18 loss: 9
'real estate X' string variants crowded out 'vacancy rates'/'interest rates').

Penalties are a non-issue: 3/67 fired, all x0.9 no_both_tags, all on tasks
already broken by truncation or blurry GT (~0.0015 total on the mean).

MEASURED TRACK STATUS, final UID-33 record (flags on; every number carries its
n and window - no extrapolation):
* NER: 0.549 baseline -> 0.651 (n=34, night) / 0.659 (n=52, morning pull).
* webpage: 0.623 -> 0.669 (n=11 night) / 0.697 (n=11 morning pull); last two
  scored tasks 0.785 and 0.812, both above their cohort's best miner.
* survey: 4 scored tasks total across the period (0.660-0.768) - too few for
  any claim.
* conversation: absolute 0.585 (n=18) / 0.592 (n=14) vs 0.626 baseline, in
  windows where the FIELD conv mean was 0.610 (n=2,220 scores).
  Cohort-relative: -0.031 (n=18; one task was a source=pool truncation scoring
  0.295 vs cohort 0.761; excluding it -0.006) and -0.016 (n=14 window).
  By cohort difficulty: 5 tasks where the cohort mean was <0.60 -> we beat 5/5
  opponents on all 5; 9 tasks where the cohort mean was >=0.62 -> we lost the
  cohort majority on all 9, margins -0.005 to -0.05.
* Shelved by offline A/B (n=126 paired): per-line quota W/L 30/50 delta
  +0.002; conditional demotion changed 6/126 answers.
* Demote-strength isolation (2026-08-10, n=126 paired): 0.95 changed 0
  answers, demotion OFF changed 1 (delta -0.002). Demotion is inert at
  composition time; the easy-task-loss hypothesis is REFUTED and the knob is
  not worth tuning. Conversation cohort-relative excluding the one truncation
  is -0.006 (n=17) - no measured conversation-specific defect remains beyond
  source=pool truncations.

## RCA-FIX EVAL, 2026-08-12 (frozen fresh-GT judge, n=266) - the coverage law

Judge: one independent GT draw per task (gpt-5.2 via OpenRouter, no cache),
frozen to `data/frozen_judge.json`; real scoring formula. Baseline A = shipped
config, mean **0.5775** (conv 0.5760 n=210, webpage 0.5830 n=56).

**ARM B - enrichment-first truncation fallback: DEPLOY.** Forced-truncation
pairs, NEW (round-robin per-line replica tags + pool fill) vs OLD (pool only):
**+0.0928** (95CI +-0.0177), median +0.0978, **improved 142 / degraded 37**.
Enrichment-line coverage 0.363 -> 0.998. A truncated answer now retains **93%**
of the full ranked score vs 77% before. On 87 further tasks OLD produced NO
scoreable answer at all while NEW did. Worst regressions -0.27/-0.25/-0.23 (37
tasks) - real but outweighed 4:1. Live case idx 21 (pool, our 0.321 vs cohort
0.748): OLD 0.3534 -> NEW **0.6720 (+0.3185)**, window-only tags 15/20 -> 0/20.

**ARM C - value-based slot allocation: REJECTED (do not deploy, do not canary).**
**-0.0132** (95CI +-0.0032), **improved 29 / degraded 177**, worst -0.103.
Conversation-only -0.0167; webpage untouched (gate correct). This kills the
user-proposed value-allocation policy AND the older fixed quota with the same
evidence.

**THE COVERAGE LAW (why C fails and B wins - the durable lesson).**
Enrichment coverage helps only from a LOW base; near the ceiling it is negative:
* C raised coverage 0.915 -> 0.951 and score FELL. On the 32 tasks where C
  actually improved coverage, score delta was **-0.0023**; on the 178 where
  coverage was already complete, forcing reallocation cost **-0.0193**.
* B raised coverage 0.54 -> 0.999 from a truncated base and score ROSE +0.093,
  equally on low-coverage (+0.0937, n=69) and mid-coverage (+0.0922, n=110)
  pairs.
So: **fix coverage when it is broken (truncation); never trade a high-cosine tag
for a line-specific one when coverage is already ~0.9.** Slot provenance at
baseline is already 13.5 enrichment-anchored / 4.5 window-only / 1.9 neither per
~19.9 tags. The 2026-08-10 "centroid-capture" reading of the loss forensics is
therefore only half right: the *symptom* (a starved line) is real, but forcing
slots to that line is a net loss - three independent instruments now agree
(blended proxy W/L 30/50, replica-GT -0.0004, frozen judge -0.0132).

**ARM D - embed retry + always-log: DEPLOY (bug fix, not a scored gain).**
Fault injection: failure now always logged (was silent unless SN33_DEBUG), and
one salvage pass recovers 1/1 batches; no retry when the first call succeeds.
Observed incidence 2/127 tasks/day with the silent signature (source=pool,
degraded=False). NO offline score gain is claimed - corpus replay cannot
simulate API failures; the live truncation-rate is the only valid measure.

## LIVE B+D DEPLOY, 2026-08-11 19:11 UTC (commit 00d078d) - OPEN monitoring

Deployed on UID 24, md5-verified against the evaluated commit:
  `SN33_FALLBACK_ENRICH=1` (B)   `SN33_EMBED_RETRY=1` (D)
C (`SN33_VALUE_ALLOC`), `SN33_LINE_QUOTA`, `SN33_HEAD_CAP` deliberately ABSENT.
`SN33_ENRICHMENT_FIRST` unchanged. Rollback (flag-gated, no code revert):
  `sed -i '/SN33_FALLBACK_ENRICH\|SN33_EMBED_RETRY/d' /opt/sn33-miner/.env && pm2 restart sn33-miner --update-env`
Backups on server: `sn33.bak-BD-20260811`, `.env.bak-BD-20260811`.

PRE-DEPLOY BASELINE to compare against (2026-08-11 00:00-19:11 UTC): 122 tasks,
5 source=pool excl survey = **4.1%**, latency p90 9.61s / max 11.30s; W&B day
mean 0.545 / median 0.612 / cohort-relative -0.0248 (n=32 scored).

**THE B VERDICT IS OPEN AND NEEDS PATIENCE.** The fallback only fires on a
truncation (~4% of tasks) and only ~26% of answered tasks get a W&B score, so a
10-hour window yields ~2-3 fallback events - far too few to confirm or refute
the offline +0.0928. Do NOT change B, D, or any ranking logic on that sample;
keep collecting until enough real truncation events accumulate, then compare
live fallback-task scores against their cohorts and against the offline result.
Zero fallback events in a window is an honest outcome ("no fallback event
observed"), not a failure.

## THE CODE BASE POINT (set 2026-08-11)

Git branch `sn33-baseline-20260811`, commit **98ce191** = the known-good
reference. Certified: 260 tests pass; behaviorally identical to what runs on
UID 24; baseline mainnet score mean 0.605 / median 0.643 (n=131).
* Any future change is measured against this: `git diff sn33-baseline-20260811 -- sn33/`.
* Revert to it if a change regresses: `git checkout sn33-baseline-20260811 -- sn33/`.
* Contains: combine-overlap truncation fix (replica.py), enrichment-first
  (conv+webpage), NER combos, screen-safe floor, translation, survey pool.
  quota/head-cap/demote-conditional are PRESENT but env-gated OFF (shelved,
  offline-neutral). Server matches behaviorally; the only file diff
  (adapter.py, pipeline.py) is that dormant gated code, to sync at the next
  safe (non-immunity) deploy.

## UID-69 FROZEN BASELINE + verdicts (2026-08-11 audit, numbers only)

SURVIVAL: UID 69 survived its first 24h pruning cycle (immunity ended ~05:10
UTC; still registered at 05:39, incentive 0.0041, inc-rank 186/256, 70 UIDs
below). UID 33 died at this same point; UID 69 did not.

TRUNCATION FIX (combine-overlap, deployed 19:12 UTC 08-10): **PASS**.
  source=pool (excl survey), same-day slow-API window:
    PRE  13:00-19:11: 8/32  = 25%
    POST 19:12->05:39: 0/77 = 0%
  elapsed p90 11.18s -> 9.74s (max 11.66 -> 11.54, no regression). Post-fix
  absolute mean rose 0.594 -> 0.615 (truncations converted to ranked), cohort-
  relative +0.002 (n=8) - output-neutral confirmed.

PRIORITY TIER: already removed 08-10 (~17:45, with the credit-outage fix); .env
has 0 SN33_SERVICE_TIER lines. No 2x cost ongoing. Truncation benefit was nil
(the 25% pre-fix rate ran WITH the tier on) - it was a latency non-fix; the
combine-overlap reorder is what actually fixed truncations.

UID-69 BASELINE (frozen; n=131 scored, 05:13 UTC 08-10 -> 05:39 UTC 08-11):
  overall  mean 0.605  median 0.643  min 0.000  zeros 1 (17:24, credit outage)
  conversation_tagging         n=67  0.601
  named_entities_extraction    n=34  0.609
  webpage_metadata_generation  n=20  0.607
  skill_generation             n= 3  0.625
  survey_tagging               n= 4  0.560
  pre-overlap  n=68  0.594   |   post-overlap n=63  0.615
Health: axon rate-limit live (1928 scanner pkts dropped, 0 floods), credits OK,
pm2 stable (3 restarts, no crashes).

## CHECKPOINT 2026-08-10 night: combine-overlap latency fix DEPLOYED, awaiting slow-API proof

UID 33 pruned at 24h immunity (scoring 0.69). Re-registered as UID 69 (~05:10
UTC 2026-08-10, immunity ends ~05:10 UTC 2026-08-11). Same proven stack; the
per-line quota + head-cap fixes are SHELVED (4-arm offline A/B neutral -0.0004;
the blended proxy structurally rewards centroid-collapse so it CANNOT score that
fix - only a live canary can).

TONIGHT'S WORK:
* Priority tier REMOVED (2x cost, no measured truncation benefit).
* OpenAI credits went to zero ~17:20 UTC (priority + heavy offline testing);
  refunded ~17:45; one task scored a hard 0 (17:24, source=local). Credit-runway
  monitoring added to the daily audit. One task = the whole cost.
* Axon flood-protection: iptables per-source-IP rate-limit on 8091 (4/min sustained,
  burst 20) + drop of one confirmed non-validator scanner. Validator task confirmed
  flowing through it. NOT persistent across reboot (fail-safe). Rollback = iptables -D.
* Validator-IP WHITELIST RULED OUT with data: 5 of 9 validator-permit holders
  publish 0.0.0.0 (validate-only), incl. the #1-stake validator; a metagraph-IP
  whitelist would block 45% of validator stake. The axon already authenticates
  validators by hotkey signature (IP-agnostic), so IP filtering is redundant.

THE TRUNCATION FIX (deployed 2026-08-10 19:12 UTC, replica.py):
* Root cause: a slow deep-enrichment call (measured 6.8s) was awaited, THEN the
  ~2s combine ran - serialized. That pushed the replica to ~9s of an 11s
  deadline, starving embed+rank -> source=pool (~0.20 vs ~0.55). ~4-10% of
  afternoon tasks; the single biggest remaining bleed.
* Fix (user's "parallelize don't cut"): launch the combine CONCURRENTLY with the
  deep grace. combine never reads deep_tags (all_sets excludes them, pinned by
  test_deep_tags_never_reach_the_replica_combine_input), so it is a PURE
  reorder - identical inputs/outputs/fallbacks. 260 tests pass incl. 3 new
  invariance proofs (combine starts before a slow deep finishes; output
  identical with/without deep; tight-budget still degrades to local_combine).
* An earlier tail-reserve variant (dropping the deep straggler) was REVERTED:
  it changed behavior on edge tasks, violating the strict no-behavior-change
  mandate. Overlap achieves the same latency win with zero output change.
* EARLY LIVE SIGNAL (n=14, 19:13-21:30): 0 truncations, all source=ranked,
  cohort-relative +0.002 (quality untouched). NOT yet proven - truncations
  cluster in the slow-API afternoon; the real test is surviving that window.
  FINAL VERDICT SCHEDULED ~10:00 IST 2026-08-11: recount source=pool rate
  (excl. survey) pre vs post 19:12, and heavy-task elapsed p90.

LOSS LEDGER (25 losses / 76 scored since UID 69 start, data/uid69_losses.txt):
  7 truncation  -> combine-overlap (deployed)
  1 outage zero -> credit monitor (added)
  16 centroid-capture (soft) + 1 NER whisker -> quota+head-cap fix built but
     offline-neutral; needs the live canary to score. This is the whole
     remaining conversation gap (~-0.02 cohort-relative on clean tasks).

## IDEA SWEEP 2026-08-14: SELECTION IS SOLVED - GENERATION IS THE BOTTLENECK

Full validation of 4 optimization ideas (data/ideas/FINAL_VERDICTS.md; replay of
120 conversation tasks with reconstructed candidate pools, frozen judge, real
formula, adversarial verification; $0.37 total spend, 99.76% cache-served).

**THE CENTRAL RESULT - selection regret is ~ZERO.** Naive single-draw oracle
regret +0.079 is an overfit artifact: validated cross-draw (oracle on half the
judge GT, scored on the other half, n=118) it is **-0.041 to +0.009, majority of
tasks DEGRADED (43/74)**. Oracle sets from two halves of the SAME GT draw overlap
only 0.23 Jaccard - there is no stable "right subset" to find. 56% of oracle tags
were already in our answer; exclusions were rank-vs-own-centroid (the +0.0036
ceiling); only 15/563 lost to dedup/insurance/screen. Four instruments agree
(cross-draw oracle ~0, ARM C -0.0132, quota 30/50, greedy +0.004 ns):
**price any future ranking/selection/composer proposal at ~0. Generation only.**

Verdicts: idea1 oracle-regret REJECT; idea2 tail-detector REJECT (signal =
truncation confound, already observable as source!=ranked, corr -0.707; healthy
tail is target-vs-draw mismatch, invisible at inference by construction; the
slot-shift response made flagged tasks worse 0/5); idea3 concentration
KEEP-RESEARCHING-downgraded (real but validator-heterogeneous: r=0.71 in
5GQyFzvt, ~0 in 5DkzbwTn/5FZe9Mpo, Q p=0.043, pooled r=0.31; no lever - both
directions already tested); idea4 dup_pairs REJECT (it counts ONLY
singular/plural inflection pairs = replica-alignment diagnostic; era sign-flip
+0.43/-0.29; intervention -0.004 CI crosses 0; keep as free monitoring metric).

Decoy-cluster split (data/ideas/decoy_cluster_split.md): the 7 named decoy
counterexamples are outside the judge corpus (not computable - honest). On 8
in-corpus analogs: 3 GT-faithful, 2 generation, 3 "selection" ALL draw-confounded
(validator's own draw sided with our clusters, draw_gap +0.14..+0.41); the
under-covered cluster is ALWAYS the window (rejected ARM C shape), never a decoy
line. Decoy-cluster detection: NOT worth pursuing.

Ranked next actions (not started): 1) fix deep enrichment's 6 defects
(test_deep_enrichment_review.py is the spec) + paired judge A/B, expected
+0.02-0.03 - the oracle's source mix (deep lines at parity with replica lines)
is the only generation-side signal in the data; 2) per-type generation for
survey+skill (gaps -0.233/-0.170; see survey-zero lead below); 3) variant-
crowding cap on mismatch tasks (small, validator-stratified test mandatory).

**Survey zeros (UID 91, 2026-08-13):** both era zeros were survey tasks answered
fast with 12 tags - first-person phrase tags ("i do everything there") likely
killed by the validator LLM keyword screen below min_tags -> discard. Lead #3 in
data/ideas/followup_leads.md: bias survey pool toward noun phrases. UNTESTED.

**UID 91 live (n=89, first 16h): cohort-relative +0.0001 - first era at parity;
conversation +0.007 (n=40) vs historical -0.04/-0.05.** Consistent with
"no selection defect remains". NER -0.005 (n=28) vs historical +0.03 - watch.

## DEEP-ENRICHMENT VERDICT + GENERATION GAP, 2026-08-14 (data/ideas/DEEP_ENRICHMENT_VERDICT.md)

Paired deep-ON/OFF A/B, 109 conv + 52 webpage judge tasks, cross-draw, verified
(5/5 re-derivations to 1e-9, 12,540-text exclusion audit, 0 misclassifications):
* **Deep is worth +0.0041 conv (CI excl. zero, BELOW the 0.017 floor) / +0.0021
  webpage (neutral). The +0.024 prior is superseded — real value ~6x smaller.**
* Deep IS structurally additive (0.6% redundancy, 35% of shipped slots, largest
  unique-coverage source: 19.6% of covered GT concepts) but on average it
  substitutes for equally-good non-deep tags: honest ceiling delta -0.0044.
  Value concentrated in ~8% of tasks (proper-noun/geo/topic-family) with no
  prospective predictor. KEEP ON; stop investing.
* Old "deep disabled/not shippable" section below is STALE - defects fixed
  2026-08-08, live since 08-09.

**GENERATION GAP (data/ideas/generation_gap.md): the pool covers 96.1% of GT
concepts.** Generation is NOT the main bottleneck either - after idea1 (selection
optimal-within-noise) this closes the loop: the architecture is at its measured
optimum; remaining levers are small and additive. The 3.9% misses are patterned:
geo-generalization 31% ('united states' missed 17x - pool holds only finer geo),
in-window named entities 27% (verbatim in transcript, never lifted), bare heads
13% ('google' vs only 'google tools'). Perfect miss elimination is worth ~+0.005
overall (upper bound). Two zero-LLM fixes designed, NOT yet built:
  H1 geo-ladder gazetteer (city->country->region injection, targets 31%)
  H2 spaCy window-NER as candidate source (extend `local`, targets 27%)
Validation harness exists (replay + frozen judge, paired, n=172). Theme/anchor/
variant sources generate ~0 unique coverage (re-rankers, not generators).

## NEXT-GEN FIXES VERDICT, 2026-08-14 (data/ideas/NEXT_GENERATION_VERDICTS.md): NO DEPLOYABLE WINNER

4-arm H1/H2 injection A/B (156 scored judge tasks, verified, zero negative deltas):
* **H1 geo-ladder + H2 window-NER: REJECT.** They fix COVERAGE exactly as designed
  (geo misses 38->13, window-NE 40->8, overall 4.7%->2.7%) yet 152-154/156 answers
  are bit-identical - the ranker outranks injected tags with incumbents closer to
  the centroid. Score +0.0002 (~55x below floor). Selection-saturation confirmed
  from the candidate-supply side: filling pool holes does not win answer slots.
  Injectors + gazetteer kept in scratchpad/data/ideas for reuse; harm-free but
  pointless under the current ranking geometry.
* **Survey zeros: the suspected mechanism did NOT reproduce.** 13 direct
  validator-screen runs per known-zero tag set never went below 8 survivors
  (min_tags=3 never approached). The 2026-08-13 zero attributions are SUSPECT
  (one from a nearest-ts-join-adjacent path). First-person fragments ARE the
  weakest class (0.89 survival, correlated whole-class deletion) and the v2
  noun-phrase prompt removes structural exposure (worst-case run 5->8 survivors)
  at -0.0235 question-proximity (score-unknowable offline). HOLD: verify the two
  zeros against bt_log before any canary. survey_prompt_v2.j2 in data/ideas.
* **Skill: hypothesis refuted, NO ACTION.** Skill GT = ONE draw of
  skill_to_metadata.j2 over the same 1000 chars we receive (no enrichment, no
  voting); GT shape matches ours (1.96 vs 2.06 words/tag, 19% vs 23% gerunds);
  our answers beat an independent GT draw 40/41 (mean 0.645). The -0.03 cohort
  gap is not a generation-shape problem. Found incidentally: variants pluralizer
  ships non-words ('portfolio analysi') - -is/-us fall-through in
  sn33/variants.py; hygiene fix specced, needs offline A/B, low priority.

The program's arc, complete: truncation fixed -> selection proven optimal-within-
noise -> generation proven 96% saturated -> injection proven unable to convert
coverage into score. **The architecture is at its measured ceiling; remaining
edge = reliability, survival mechanics, and live A/Bs only a mainnet canary can
resolve (<0.017).**

## TAIL-STABILITY AUDIT 2026-08-14 (current config, uid24/12/91, n=140 scored)

Bad set = 32/140 (zero or rel<=-0.05), fully classified: 16 tight-cohort content
losses (cohort clustered above us - the real class), 7 wide-cohort draw variance
(irreducible), 5 mid-spread, 4 truncation-family (B rescued 2, capping losses at
-0.06/-0.13; 1 unrescued NER - B does not cover NER; 1 deadline-pressure), and
**1 cohort-wide zero: the 08-13 17:02 survey zero had cohort [0.00-0.00] - the
WHOLE cohort scored 0. Validator-side failure, NOT our tags. Survey-zero scare
CLOSED; survey v2 stays HOLD.** Conversation is the inconsistency source (sd
0.093, p10 0.419, 25/65 bad) vs NER (sd 0.067, p10 0.599, 1/48).

Reading the 16 tight losses vs 10 winner controls: NULL - losing answers are the
same species as winning ones (big same-head families, twins, dict-decoy tags all
appear equally in +0.08 winners; enrichment-following IS what winning looks
like). Gradients found and then killed by offline A/B on the judge corpus
(154 tasks, $0 cache-only, data/ideas/tail_tunings_ab.json):
* T1 pluralizer wordlist gate: -0.0008 mean, 13/29 W/L - REJECT (dropped twins
  were dedup-protected slot-holders; wordlist over-fires on valid modern words).
  The 11 truly-mangled live plurals remain a cosmetic non-issue (<any floor).
* T2 window-anchor floor 2 (conv): -0.0001 mean, 12/31 W/L, worst -0.034 -
  REJECT (forced-allocation cousin, as the prior predicted).
What separates -0.13 from +0.08 on identical-looking answers is where the GT
draw lands relative to our tag cloud - not a hygiene defect any config knob can
fix. VERDICT: KEEP CURRENT SETUP, no tunings deployed. Open live-gated items
only: survey v2 (gated on bt_log proof of a real screen discard), NER
fallback_enrich extension (1 case/48, designed, low priority).

## CHECKPOINT 2026-08-16: validator vuln audit (disclosed, not exploited) + SUMMARY_TAGS live

**SECURITY AUDIT - a real validator scoring bypass, REPORTED not exploited.**
Full writeup: `data/ideas/SN33_VULNERABILITY_DISCLOSURE.md` (send to ReadyAI).
Chain (E1): junk tags (<3 chars) -> `get_clean_tag_set` empties the list ->
`validate_tags_prompt` raises -> `ConversationTaggingTaskBundle.format_results`
never reassigns `tags`/`vectors` -> `validator.py:348` `except...continue` skips
LOGGING but NOT scoring -> `_calc_scores` reads `miner_result["vectors"]` (the
attacker's floats) off the wire. E2 primitive: text-embedding-3-small is
anisotropic, so one vector aimed at the global-mean sits ~0.75 cosine to any GT
centroid for free; a vector = mean of the attacker's OWN predicted-GT guesses
reaches ~0.90 (averaging cancels per-guess error). Verified through the REAL
`GroundTruthTagSimilarityScoringMechanism`: HONEST 0.548 vs CHEAT-smart 0.800,
**wins 50/50, +0.25**; worked example map_idx 1 honest 0.626 -> attack 0.852.
Same bug (D1) hard-zeros HONEST miners (vectors=None) when the validator's own
screen LLM fails. NOT exploited: scanned 7,139 live scoring rows, 0 show the
forged-vector signature (all tags score identically -> per-tag spread 0.0; min
observed 0.0179). Standing rule honored: report, never deploy forged vectors.
Fixes proposed: drop the response on format_results raise; never trust wire
`vectors` (re-embed server-side); fail closed on validator-LLM errors.

**SN33_SUMMARY_TAGS DEPLOYED on all 4 miners (uid5/127/175/82), ~12:24 UTC.**
The HONEST cousin of the NER-composite trick: `tags.make_summary_tags` builds
2-4 word umbrella phrases from the predicted-GT theme words (they embed near the
centroid, same mechanism NER concat rides), screen_safe-filtered, added to the
conversation candidate POOL only (compose keeps one just when it out-ranks an
incumbent). Flag-gated (`SN33_SUMMARY_TAGS=1`), conversation-only, OFF by default.
**Real-path gain +0.001** (verify_summary_realcompose.py: real rank+compose+dedup+
scoring class, n=60, W/L 18/13, worst -0.019) - the experiment's +0.0085 was an
artifact of FORCING 3 phrases into a weak baseline; the production compose already
ranks a 194-cand pool by the same centrality, so phrases get picked 48/100 and net
~0. It is BELOW the 0.017 floor and cannot be measured live; deployed per operator
decision as an honest, reversible, screen-safe change (cannot cause a zero).
Same collapse as quota/head-cap/value-alloc - selection is saturated; the coverage
law and IDEA SWEEP verdicts stand. Deploy verified: md5-matched, server import
smoke test, flag in all 4 process envs, post-restart tasks source=ranked
degraded=False, latencies healthy. Backups on server `*.bak-SUMMARY-20260816` +
`.env.bak-SUMMARY-20260816`. Pre-deploy conv health baseline (n=574): truncation
1.2%, 0 zeros, latency p90 9.91s / max 11.66s. Re-check + revert triggers:
`scratchpad/recheck_summary_deploy.py` (revert if trunc >5%, any zero, or p90 >10.5s).
ROLLBACK (flag-only): `sed -i '/SN33_SUMMARY_TAGS/d' /opt/sn33-miner/.env &&
pm2 restart sn33-miner2 sn33-miner3 sn33-miner4 sn33-miner5 --update-env`.

## THE QUALITY-GAP DECOMPOSITION, COMPLETE (2026-08-15) + insurance deploy

**GAP-TO-BEST decomposed (178 conv cohorts): -0.081 = -0.033 order-statistics
luck (uncloseable - "best of 5" beats the field mean by +0.033 on draw luck
alone; verified luck+deficit=-0.081 exactly) + -0.049 true deficit vs cohort
mean.** Beat-the-best rate 4% vs 17% for equals. Do NOT use gap-to-max as a
target; cohort-mean parity is also inflated by other miners' failures (wide
cohorts carry 0.57 failures vs 0.11 in tight ones - a -0.093 "tight-cohort
effect" was 91% this artifact; always check vs cohort max before believing
cohort-composition effects).

**DEPLOYED 2026-08-15 09:15 UTC: SN33_INSURANCE_CONV=14** (conversation-only
per-kind gate; the old global SN33_INSURANCE would poison NER). Offline paired
n=107: p10 0.332->0.425, sub-0.4 tasks 17->6, sd 0.128->0.099, cost -0.014 top
quartile; peak 14-16, >=18 declines, 20 collapses. Early live (n=27): deep-loss
18%->7%, p10 +0.096 (matches the offline +0.09 prediction), worst -0.44->-0.23.
**SN33_INSURANCE_WEB=10 was deployed same day and ROLLED BACK within hours**:
webpage was already at parity (-0.006), offline gain below floor, live 0/5 wins
rel -0.068. Lesson: do not extend a fix measured on type X to type Y whose
baseline is healthy.

**DRAW-CONSENSUS / MATCH-FORCING: REJECTED 2026-08-15 (the aim-channel leaf).**
Consensus tags from >=2-of-3 independent temp-1.0 GT-chain draws DO raise
exact-match count (+1.24/task, 6.95->8.19 vs the frozen judge - the mechanism
works) but the SCORE falls (W/L 17/20, worst -0.156): both-tags are excluded
from the 0.55*top-3-UNIQUE term, we already clear the no_both penalty at ~7
matches, and every extra match displaces a unique carrier. The "winners have
~5 both-tags, losers 0.5" decomposition finding is a CORRELATE of good aim,
not a causal channel - forcing matches optimizes the wrong formula term.
(Also: reordering predicted_gt is inert - compose picks insurance by cosine,
not list order.) Do not revisit match-count optimization.

**Score-decomposition facts (107 tasks, worst-vs-best quintile):** no formula
term collapses alone - top3u/mean/median/max all fall together (gaps +0.28..
+0.36); top3/mean ratio is HIGHER on bad tasks (1.19 vs 1.06); zero-pad cliff
never fires (0/107 below 3 unique). So bad tasks = whole-cloud aim misses; no
downstream detect-and-recompose can exist. The residual -0.02..-0.03 vs cohort
mean lives in UNIQUE-tag centrality = raw generation quality; every tested
lever (incl. stronger pool model, gpt-5.6-luna: pool +0.013 more central but
compose delta -0.002) fails to reach it. Conversation config knobs: CLOSED.

## THE TOP-MINER RECIPE, MEASURED (2026-08-14, 19,085 conv cohorts, 352 miners)

Per-miner conversation profiles from all W&B caches (254k score rows), no
assumptions (data/ideas/conv_miner_profiles.json):
* Top-10 consistency miners: rel mean +0.022-0.027, **rel sd 0.044-0.050,
  worst-ever single task -0.036..-0.045 over 190-483 tasks, bad-rate
  (rel<=-0.05) LITERALLY 0%.** 0.046 = the cohort-mean sampling-noise floor
  (~0.10/sqrt(5)) - their answers have ~ZERO idiosyncratic variance; ALL their
  rel noise is the cohort's own. THE recipe = uniformly draw-robust tag clouds,
  nothing else: penalties 12-15% (same as ours), n_tags 17-18, zeros similar.
* US: rel sd 0.103 -> idiosyncratic ~0.092. By era: mean SOLVED (-0.088 uid137
  -> +0.002 uid91) but idio sd persists (uid91 best: 0.066, still 1.4x floor);
  bad% 17-25% every era vs their 0%.
* **n_tags is NOT the lever**: offline A/B k=20/18/17/15 (107 judge tasks,
  paired): 18 = -0.0015 (58/49), 17 = -0.0019, 15 = -0.0035; cross-half
  draw-spread UNCHANGED (0.051 in all arms). Top miners' 17.5 is a correlate
  of their quality, not a cause. Do not re-trim tag count.
* Conclusion: the consistency gap IS the +0.12-per-tag pool-centrality gap
  viewed as variance. No config knob reaches it; only generation quality of a
  kind not achievable by injection (H1/H2 proved coverage doesn't convert).
  Idio-variance (bad% and rel sd vs the 0.046 floor) is THE metric to watch
  era-over-era - it is improving with each deployed fix generation.

## NEW TASK TYPE: skill_coverage_evaluation (v2.37.74, MANDATORY, mainnet 2026-08-18 19:30)

Merged PR#135 on branch `skill-coverage-upgrade` (2026-08-14). Miner receives
seed + section_map (WE SEE the section titles+descriptions being scored
against - unlike conversation GT); returns {skill, tdd_plan, section_tests:
{section_id: [{name, description, assertion}]}}. Scoring (verified from
SkillCoverageScoringMechanism):
  adjusted = 0.6*mean-over-sections(mean of top-2 judged-correct test cosines
             to the section-description embedding)
           + 0.4*cos(mean of all judged-correct test embeddings, OUR OWN skill
             text embedding)   <- self-referential, we control both sides
  Embedded text per test = description+assertion. LLM-judge stamps
  judged_correct (traceability: tests may only reference identifiers the skill
  defines); incorrect tests are EXCLUDED entirely, not down-scored.
  Penalties: <3 tests -> hard skip (0); <5 total x0.5; any test pair cos>=0.95
  x0.5; judged accuracy <50% x0.7; >6 tests in a section x0.6 (excess never
  embedded); 0 sections addressed -> 0.
  Optimal shape: every section covered, 2-6 distinct tests each (>=5 total),
  all judged-correct, aimed at the section description text.
Our path: MinerLib tries sn33 adapter (returns None for this type) -> falls
back to stock task.mine() FAST_MODE (1 call, skill_request_to_skill_bundle,
model gpt-5.6-luna + reasoning_effort low + service_tier PRIORITY (2x cost,
upstream's decorators) - watch spend).
**UPSTREAM BUG FIXED IN OUR FORK: basic_prompt hardcodes temperature=0 which
gpt-5.6-luna rejects (400) -> the whole chain silently returned None (empty
answers). Our llm_openai.py retries once without temperature. If upstream
fixes it differently, drop our patch at the next sync.** Smoke-verified: 9.4s,
full bundle, 2 tests/section. PR tests 44/44 + our 273 sn33 tests pass.

## The task-score join is TYPE-BLIND - do not trust nearest-ts joins (found 2026-08-09)

`scripts/sn33_join_scores.py:82-101` (and every ad-hoc join built the same way)
matches a task to the first same-validator score within 180s WITHOUT checking
task type. Audited against W&B task_id cohorts: **51% of
`data/our_24h_joined.json` rows (69/135) carry a score of a DIFFERENT task**
(webpage 11/17 wrong, conversation 35/89). Mis-joins attenuate real
correlations rather than create them - conversation's fe r=0.66 survived and
its paired A/B + live results are unaffected - but any MAGNITUDE claim built on
a nearest-ts join is unreliable until re-derived. The correct join: W&B logs
`task_id.{uid}` per ~6-miner cohort; identify our row by hotkey within the
cohort and take the cohort's task type. Paired offline A/Bs and per-type W&B
tables need no join and are the trustworthy instruments.

## Benchmark trustworthiness

`bench/` calibrated to **-0.001** against real production scores on real
submitted tags (May 2026 miners, uids 17/150/198/253). Trust it for *absolute*
score level.

**Do NOT trust it for the unique/both split.** It counts a tag as unique if it is
absent from ground truth *we* generate; the validator diffs against ground truth
*it* generates. Those overlap far more than the bench assumes, which is exactly
how the bad `insurance` recommendation happened.

Ground truth must be generated with the vendored **upstream** prompts
(`sn33/upstream_prompts/`) and a separate cache salt from miner calls, or the
bench scores a strategy against itself.

## Two real defects in the validator's tag screen (tested 2026-08-08)

Driving the real `LlmLib.validate_tag_set` with stubbed replies, not reasoning:

```
reply omits the word "malformed"   -> find() returns -1 -> content[0:-1]
   submitted 3 tags, SURVIVED 2, LOST 'multifamily properties'
a submitted tag contains "malformed" -> parse cut at the first occurrence
   submitted 4 tags, SURVIVED 1, LOST 3
```

Both cost the miner tags and hit every miner equally - they are defects, not
exploitable surface. Guarded in `tags.normalize` (reject any tag containing
"malformed"); the `[0:-1]` clipping is the validator's and we cannot avoid it.

Working as intended: tags we never submitted are rejected (`element in tags`),
and casing differences in the reply are re-cleaned correctly.

`{{tags_string}}` is interpolated into an LLM prompt and injection-shaped text
("ignore all previous instructions") survives cleaning as a valid fixed point.
UNTESTED whether the model obeys - and it would not pay: the screen only decides
which tags get scored, and a junk tag still scores ~0 on the cosine.

## deep enrichment: implemented, disabled, NOT shippable

`Config.use_deep_enrichment = False`. Probed at +0.024 adjusted (4/4
conversations, replicated on fresh ground truth), but adversarial review found
six defects and the score was never re-verified - the run stopped when the
OpenAI account ran dry. See `tests/sn33/test_deep_enrichment_review.py`, which
is xfail-marked and IS the specification for the fix. Worst two: ASCII Spanish
tags reach the answer (the non-ASCII guard only catches accented letters), and
`DEEP_GRACE_S = 1.6` against a measured 3.5s deep latency, so the gate always
spends its grace and never returns anything.

## Operations

Server `95.133.253.123`, `/opt/sn33-miner`, PM2. See `docs/SN33_MINER_RUNBOOK.md`.

* **bittensor must be 10.3.0**; `bittensor-drand==1.3.0` (2.0.0 removes
  `get_encrypted_commit`); `bittensor-wallet 4.1.0` reads btcli keyfiles
* keep `btcli` in a **separate venv** (`/opt/btcli`) - it drags in incompatible pins
* use `PYTHONPATH`, never `pip install -e .` (setup.py pins bittensor==9.0.0)
* **`--axon.ip` is mandatory**: `base/miner.py:118` commits `self.axon.ip`, which
  defaults to `[::]`. It publishes "successfully" and is unreachable. `run_miner.sh`
  refuses to start without it.
* data capture: `/var/log/sn33/tasks.jsonl` + `scores.jsonl`, joined by
  `scripts/sn33_join_scores.py` (by validator+time; the guid is masked)

## Economics

Rank 1 pays only **1.30x** the median and rank 10 is within 5% of rank 1. Four of
our own miners on identical software ranked 10, 50, 68 and 160 - rank is
dominated by sampling noise at ~26 observations (standard error +-0.016). Aim for
the top cluster and a 0% zero rate, not for rank 1.

Miner cost ~$0.009/task, ~$1/day at 5 tasks/hour. Output tokens are 60% of spend.
