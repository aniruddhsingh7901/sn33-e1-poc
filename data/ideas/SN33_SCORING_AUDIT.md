# SN33 scoring-path audit

**Date:** 2026-08-16
**Tree audited:** branch `skill-coverage-upgrade`, HEAD `e284c1e`. The validator-side code
under audit is upstream ReadyAI (PR #135, merge `c80668c`); our own commits touch only
`sn33/`, prompts and `neurons/miner.py`. One local exception is flagged inline.
**Scope:** local, defensive, read-only. No live validator, miner, mainnet or testnet endpoint
was contacted. No production file, process or config was modified. All new code lives in the
session scratchpad.
**Standing instruction honoured:** no withheld-GT reconstruction, no multi-key collusion, no
forged embeddings deployed, no interference with other miners. Everything classified EXPLOIT
below is written up for disclosure and marked DO-NOT-DEPLOY.

---

## 1. Executive summary

| Class | Count | Items |
|---|---|---|
| **EXPLOIT** (disclose, never deploy) | 2 | E1 format_results-raise bypass; E2 constant-vector payload |
| **DEFECT** (costs honest miners) | 6 | D1 LLM-failure zeroing; D2 normaliser asymmetry; D3 fixed-point filter; D4 `malformed` parser; D5 non-English shredding; D6 screen hard-zero |
| **LEGITIMATE** | 1 | L1 NER composite tags |
| **NOT REPRODUCED / corrected** | 4 | see §6 |

The single most important result: **E1 lets a miner skip the work entirely.** A response
carrying five junk tags and a forged `vectors` dict makes the validator's `format_results`
raise; the exception is caught and `continue`d, but `continue` only skips logging — the
response object stays in the list that is scored. The miner's tags are therefore never
screened and its embeddings are never regenerated, so cosine is computed against
attacker-chosen floats. Combined with E2 (a single fixed 1536-float vector sits at cosine
~0.75 against essentially any ground-truth centroid), a miner that never reads the
conversation outscores the entire measured live field.

**No evidence anyone is currently doing this.** All 7,139 per-tag cosine breakdowns in
`data/wandb_deep/rows.jsonl` with ≥3 unique scores have non-zero spread (min 0.0179, p1
0.0297, median 0.1324, **exactly zero rows at 0.0**), and a forged-single-vector answer
produces spread of exactly 0.0 by construction. Accidental incidence for honest miners is
also 0/879 in captured traffic. This is latent attack surface, not a live bleed — but it is
an incentive-alignment hole, and the right response is disclosure.

---

## 2. Findings that reproduced, ordered by expected gain

### E1 — Junk tags make `format_results` raise, so unscreened tags **and** miner-supplied embedding vectors reach the scorer

**Classification: EXPLOIT. DO NOT DEPLOY.**

**Code path** (every line read and confirmed verbatim on this tree):

| step | location | behaviour |
|---|---|---|
| 1 | `conversationgenome/utils/Utils.py:282` | `get_clean_tag_set` drops any tag whose `get_safe_tag` output is <3 or >64 chars |
| 2 | `conversationgenome/llm/LlmLib.py:217` | `validate_tag_set` calls `prompt_manager.validate_tags_prompt(clean_tag_list)` |
| 3 | `conversationgenome/llm/prompt_manager.py:49-50` | `if not tags: raise ValueError("tags cannot be empty.")` — **before** `basic_prompt`, so the attack costs the validator no API spend and needs no key |
| 4 | `conversationgenome/task_bundle/ConversationTaggingTaskBundle.py:161-162` | RHS raised ⇒ neither `miner_result['tags']` nor `miner_result['vectors']` is ever assigned |
| 5 | `neurons/validator.py:341-348` | `except Exception: bt.logging.error(...); continue` |
| 6 | `neurons/validator.py:369` | `await task_bundle.evaluate(miner_responses=responses)` — over the **unfiltered** list |
| 7 | `GroundTruthTagSimilarityScoringMechanism.py:_calc_scores` | `tag_vector_dict = miner_result["vectors"]` → cosine against attacker floats |

The `continue` at step 5 skips only `bt.logging.debug` and `vl.put_task`. It never removes the
response from `responses`.

**Reproduction** (mine, this session, no API calls):
`<scratchpad>/audit_verify_chain.py`, `<scratchpad>/audit_verify_chain2.py`. The second drives
the **real** `ConversationTaggingTaskBundle.format_results` and the **real** `prompt_manager`,
and transcribes the validator loop verbatim. Output:

```
conversation: RAISED ValueError: tags cannot be empty.
  tags after : ['ab', 'cd', 'ef', 'gh', 'ij']
  vectors untouched (identity): True | first float: 0.123
  'original_tags' assigned: True
  response still in `responses` for evaluate(): ['THE_RESPONSE_OBJECT']
```

Identity check `True` is the load-bearing fact: the forged dict object survives into scoring.

**Normal score:** 0.5580 final / 0.5645 adjusted (n=120 captured mainnet conversation tasks,
real shipped tags through the real chain, GT = independent frozen judge draw).

**Adversarial score** (n=120, same corpus, same chain):

| arm | forgery | adjusted | final | Δ final | wins |
|---|---|---|---|---|---|
| honest | — | 0.5645 | 0.5580 | — | — |
| X-gt | miner's predicted-GT centroid | 0.8978 | 0.8080 | **+0.2500** (95CI ±0.0150) | 120/120 |
| X-own | miner's own answer centroid | 0.7532 | 0.6779 | +0.1199 (±0.0082) | 120/120 |
| X-blind | **one fixed task-independent vector** | 0.7420 | 0.6678 | +0.1099 (±0.0181) | 104/120 |

**Expected gain:** +0.25 final per task with the strongest forgery the miner can already
build; +0.11 with zero task-specific work. Quote as a range, not a point — the adversarial
score is a property of the attacker's chosen forgery, not of the code. For context, the
measured live top-10 cohort sits at 0.6929 adjusted and our UID-69 baseline at 0.605 final,
so even the **blind** arm beats the entire measured field.

**Survives the real chain:** yes. Real bundle, real `prompt_manager`, real
`GroundTruthTagSimilarityScoringMechanism.evaluate()`, real penalty application (only
`no_both_tags` ×0.9 fires — junk tags never string-match GT, and that is already priced into
the finals above). The only exception raised across all arms is the expected
`ValueError: tags cannot be empty.`

**Survives unseen GT:** yes. The GT scored against is an independent judge draw the miner
cannot see; the miner's predicted-GT matches only 49.3% of its exact strings, yet the
*centroid* cosine is 0.898. Averaging cancels off-target directions, so a centroid-of-estimates
sits far closer to the centroid-of-GT than any individual tag can. Nothing depends on seeing GT.

**Incidence in real captured traffic** (`data/all_mainnet_tasks_with_scores.csv`, 879 answered
tasks; my own count):

```
exposed (call the raising validate_tag_set):  654 / 879 = 74.4%
  conversation_tagging 426 | webpage 146 | survey 46 | skill_generation 36
NOT exposed: named_entities_extraction 225
```

NER is not exposed — `NamedEntitiesExtractionTaskBundle.py:147` calls
`validate_named_entities_tag_set`, which never touches `prompt_manager`, returns `[]`,
`format_results` completes, and **vectors are overwritten**. Verified in my repro:
`NER: COMPLETED, tags=[], vectors OVERWRITTEN`. The response then scores 0.0 on
`min_tags=3`. Trigger conditions on the exposed types are 100% attacker-controlled.

**Accidental incidence for honest miners: 0/879.** No captured answer has every tag cleaned
away, so this costs honest miners nothing today.

**Payload is smaller than it looks:** because every junk tag carries the identical vector, and
`scoring_factors` sum to exactly 1.0 (`top_3_mean` 0.55 + `mean_score` 0.25 + `median_score`
0.10 + `max_score` 0.10 — confirmed at `GroundTruthTagSimilarityScoringMechanism.py:19-24`),
all four terms collapse to the same number and **adjusted = that one cosine**. Tag count is
therefore irrelevant: 3, 5 and 20 junk tags score identically to 4 d.p. `too_few_tags`
threshold is 2, so 3 tags clears it. `cgp_output` is `Optional[List[dict]]` with no schema, so
pydantic accepts the forged dict and round-trips the floats bit-exact.

---

### E2 — One hardcoded vector scores ~0.75 cosine against essentially every ground-truth centroid

**Classification: EXPLOIT (enabling primitive). DO NOT DEPLOY.**

**Code path:** `GroundTruthTagSimilarityScoringMechanism.py:162-180`
(`_calculate_semantic_neighborhood` — the centroid is a plain `np.mean` of the GT tag
embeddings) and `:260-275` (`_score_vector_similarity` — bare cosine, no provenance check).

**Reproduction (mine, this session, cache only, 0 API calls):** I split 80,000 cached
`text-embedding-3-small` vectors into two **disjoint** 40,000-vector halves, fitted a mean
direction to each independently, and scored both against 196 held-out GT centroids:

```
cos(mu_halfA, mu_halfB) = 0.9999          (disjoint samples)
halfA vs 196 GT centroids: mean 0.7545  p05 0.5992  min 0.5240  max 0.8898
halfB vs 196 GT centroids: mean 0.7544  p05 0.5992  min 0.5248  max 0.8899
```

The two independently-fitted directions are effectively the same vector, and both clear 0.75
mean against unseen centroids. This is not a fitting artifact.

**Normal score:** top-10 live miners average 0.6929 adjusted (n=27,978 scored tasks, 48h);
our live baseline 0.605 final.
**Adversarial score:** adjusted = the cosine exactly (see the collapse argument in E1), i.e.
~0.74 adjusted / ~0.67 final with **no knowledge of the task, the conversation or the GT**.
**Expected gain:** +0.11 final over honest, from zero work.
**Survives real chain:** only in combination with E1 or D1 — on the normal path the validator
re-embeds the tag strings and the forged vector is discarded. This is the payload, not the
entry point.

**The structural point for ReadyAI** (true even with E1 fixed): `text-embedding-3-small`
vectors share a large common component, and a GT centroid is the mean of ~20 of them, so it
regresses hard toward the global mean direction. Cosine-to-a-mean-of-embeddings therefore has
a **high content-independent floor of ~0.75**, which caps how much signal the metric carries
about tag quality. Honest miners spanning 0.55–0.69 are competing inside a narrow band above
a floor that requires no content at all.

---

### D1 — The same bypass fires with **zero** miner cooperation whenever the validator's own LLM call fails

**Classification: DEFECT (severe, asymmetric).**

**Code path:** `conversationgenome/llm/llm_openai.py:56` (`except → return None`) →
`LlmLib.py:220` (`len(response_content)` on `None` → `TypeError`) →
`ConversationTaggingTaskBundle.py:161` → `neurons/validator.py:341-348, 369`. Honest-miner
side: `sn33/adapter.py:191` ships `{"tags": ..., "vectors": None}` →
`GroundTruthTagSimilarityScoringMechanism.py:206` (`not tag in tag_vector_dict` on `None` →
`TypeError`) → caught by the broad except at `:76-80` → **0.0**.

**Reproduction:** hit accidentally and then deliberately —
`<scratchpad>/lens1_final.py` (G1), `<scratchpad>/lens1_honest_zero.py`. The scratchpad
OpenAI key is out of credits, so `basic_prompt` returned `None` on a real 429 and
`validate_tag_set` raised the exact `TypeError`. Not hypothetical.

**Normal score:** 0.5716 final (n=60, healthy validator screen).
**With the screen down and `vectors=None` on the wire** (exactly what our miner ships today):
**0.0000**.
**Adversarial score:** 0.7020 final (n=60) for a miner that pre-populates `vectors` for its
own real tags — wins 54/60 against its own healthy-screen score.
**Expected gain:** +0.1304 for the attacker and **−0.5716 (to a hard zero) for every honest
miner in the same batch**, per affected task.
**Survives real chain:** yes, drives the real classes.

**Incidence:** not measurable from our side — it is a function of the validator's own OpenAI
health (rate limits, quota exhaustion, timeouts, 5xx). Any such event converts the whole batch
to "miner tags + miner vectors, unscreened".

**Our exposure:** total, and there is no fair mitigation on the miner side. Shipping a
populated `vectors` dict would immunise us, but that is exactly the forged-embedding move the
operator's standing instruction forbids, and it would be indistinguishable from E1 to an
observer. **Recommendation: disclose, do not self-immunise.**

---

### D2 — GT and miner tags are normalised by different functions, so ~12.9% of GT tags can never be matched

**Classification: DEFECT (pure honest-miner loss, no adversarial side).**

**Code path:** `Utils.py:218-223` (`clean_tags` — strip/lower/de-quote only, **no** character
filter, **no** length bound; used for GT) vs `Utils.py:268-289` (`get_safe_tag` +
`get_clean_tag_set` — strips everything outside `[a-zA-Z0-9\s]`, bounds 3..64; used for miner
answers). Consumed by `Utils.compare_arrays:48-58` as an **exact set intersection**.

**Reproduction (mine, `<scratchpad>/audit_verify_chain.py` §D):**

```
GT='low-income housing tax credit'   miner-reachable=['low income housing tax credit']  matchable=False
GT="moody's analytics"               miner-reachable=['moody s analytics']              matchable=False
GT='node.js'                         miner-reachable=['node js']                        matchable=False
GT='u.s. office market'              miner-reachable=['u s office market']              matchable=False
GT='ai'                              miner-reachable=[]                                 matchable=False
```

**Corpus statistics** (7,870 real cached GT tag sets, 144,989 GT tags): a perfectly-playing
miner can reach at most **87.15%** of GT tags. Breakdown of the 18,636 unmatchable:
18,513 punctuation/non-ASCII, 110 shorter than 3 chars (`ai`, `ev`), 13 longer than 64.
**1.21% of tasks (95/7,870) have ZERO matchable GT tags** — the `no_both_tags` ×0.9 penalty is
then unavoidable no matter what the miner submits. Mean per-task unmatchable fraction 13.3%;
only 49.6% of tasks have a fully matchable GT set.

**Expected cost:** ~0.0012 on the mean final score from the forced penalty alone, plus the
loss of exact-match slots on the other 98.8%.
**Survives real chain:** yes — verified against the real `Utils` functions on real cached LLM
output. The mismatch is structural: two different normalisers on the two sides of a set
intersection.

**Our defence:** none possible. This is not defensible miner-side; it needs an upstream fix
(run GT through `get_clean_tag_set` too, or compare on a shared normal form).

---

### D3 — `validate_tag_set` compares the screen's output against the **raw** miner list, deleting tags the screen approved

**Classification: DEFECT.**

**Code path:** `conversationgenome/llm/LlmLib.py:228` —
`return [element for element in valid_tags if element in tags]`, where `valid_tags` has been
through `Utils.get_clean_tag_set` (line 227) but `tags` is the **unmodified** miner list from
line 209.

**Reproduction (mine, `<scratchpad>/audit_verify_chain.py` §C), with a perfect screen that
approves every tag:**

```
1/4 survive: ['Machine Learning','self-driving cars','u.s. politics','machine learning'] -> ['machine learning']
0/1 survive: ['MACHINE LEARNING']                                                        -> []
0/3 survive: ['research & development','input/output','e-commerce']                      -> []
```

**Expected gain (loss):** up to total loss of the answer. Dropping below `min_tags=3` makes
the response **discarded entirely** (`GroundTruthTagSimilarityScoringMechanism.py:110-119` →
final 0.0) — the same hard-zero failure mode as D6, but reachable with **no LLM
nondeterminism at all**.
**Survives real chain:** yes; only the network call was stubbed, to a maximally generous
screen. The deletion happens *after* the LLM has already said the tag is good English.

The intent of line 228 is clearly "do not let the LLM invent tags the miner never submitted",
and that part works. The bug is that the left side of the comparison is cleaned and the right
side is not, so the filter silently doubles as an undocumented "reject any tag that is not
already a fixed point of `get_safe_tag`" rule. A miner writing natural, correct English
keywords — capitalised proper nouns, hyphenated compounds, "U.S." — loses them all.

**Our defence: already complete.** `sn33/tags.normalize` emits only pre-normalised
lowercase-alphanumeric fixed points, so every tag we ship passes line 228 unchanged. This is
why the rule was worth discovering: it is invisible to us and lethal to a naive miner.

---

### D4, D5, D6 — previously-known defects: re-verified, still hold

| # | defect | status on this tree | our defence |
|---|---|---|---|
| D4 | `LlmLib.py:222-224` parses the reply by `find("malformed")`; absent ⇒ `-1` ⇒ `content[0:-1]` silently clips a tag. A **submitted** tag containing "malformed" cuts the parse at the first occurrence. | **Unchanged**, confirmed at `LlmLib.py:222-224` | `sn33/tags.normalize` rejects any tag containing "malformed". The `[0:-1]` clip is the validator's and cannot be avoided. |
| D5 | `Utils.get_safe_tag:269-272` strips non-`[a-zA-Z0-9\s]`, shredding accented and erasing non-Latin tags ⇒ non-English answers score ~0. | **Unchanged**, confirmed | `Config.translate_non_english` (on) translates candidates to English. |
| D6 | The `validate_tags.j2` "good English keywords" screen deletes acronyms/compounds nondeterministically; survivors <`min_tags`(3) ⇒ response **discarded** (hard 0.0, not a penalty). | **Unchanged**; `min_tags=3` confirmed at `GroundTruthTagSimilarityScoringMechanism.py:16` | screen-safe floor of 7 dictionary-certified tags per answer. |

---

### Two robustness notes (validator-side, not miner-exploitable)

* `NamedEntitiesExtractionTaskBundle.py:96` — `random.randint(1, min(3, len(results)))` raises
  `ValueError` when `results == []` (`randint(1, 0)`). This kills GT construction for the whole
  task, not any particular miner. Worth including in the disclosure as a one-line fix.
* `neurons/validator.py:121-124` — a length mismatch on `scattered_rewards[uids] = rewards` is
  swallowed. Not reproduced as harmful here; noted only.

---

## 3. DEFECTS: what our miner already defends, and what is exposed

| defect | our defence | residual exposure |
|---|---|---|
| D3 fixed-point filter | **complete** — we ship only `get_safe_tag` fixed points | none |
| D4 `malformed` parser | **partial** — we reject tags containing "malformed"; the `[0:-1]` clip still costs one tag whenever the reply omits the word | small, hits everyone equally |
| D5 non-English shredding | **complete** — translate-to-English on non-English candidates | none beyond the unavoidable `no_both_tags` ×0.9 on those tasks |
| D6 screen hard-zero | **complete** — screen-safe floor of 7 | none observed |
| D2 normaliser asymmetry | **none possible** | ~0.0012 mean, plus lost exact-match slots on ~50% of tasks. Upstream fix only. |
| D1 validator LLM failure | **none, and self-immunising would be unfair play** | total: a hard 0.0 on every task in an affected batch. Disclosure is the only fair route. |

The honest conclusion for the operator: our miner's defensive posture against the
*known* defects is essentially complete. The two residual items (D1, D2) are both upstream
problems that no fair miner-side change can address.

---

## 4. EXPLOITS — DO NOT DEPLOY

**E1 and E2 are flagged DO-NOT-DEPLOY.** Deploying either would be unfair play and is out of
scope per the operator's own standing instruction ("if you find a genuine vulnerability,
report it rather than exploiting it"). E1 in particular does not merely inflate our ranking —
it removes the requirement to do the work at all, which is a subnet-integrity issue rather
than a scoring-quality one. A subnet where the blind arm beats the honest field has no
incentive gradient left.

Note also that E1 and D1 are the same defect reached from two directions: one
miner-triggered, one triggered by the validator's own infrastructure. A single fix closes both.

### Draft responsible-disclosure note to the ReadyAI team

> **Subject: Miner-supplied embedding vectors can reach the scorer when `format_results` raises**
>
> **What.** On the current validator scoring path, a miner response whose tags are all removed
> by `Utils.get_clean_tag_set` (trivially achieved with tags under 3 characters) causes
> `prompt_manager.validate_tags_prompt` to raise `ValueError("tags cannot be empty.")`. Because
> the exception happens on the right-hand side of the assignments in
> `ConversationTaggingTaskBundle.format_results`, neither `miner_result['tags']` nor
> `miner_result['vectors']` is overwritten. `neurons/validator.py` catches the exception and
> `continue`s, but `continue` skips only logging and `put_task` — the response object remains
> in `responses`, which `evaluate()` then scores in full. The result is that the miner's raw
> tags bypass the "good English keywords" screen **and** the validator scores cosine against
> embedding vectors supplied by the miner rather than ones it generated itself.
>
> **Where.**
> `conversationgenome/utils/Utils.py:282` →
> `conversationgenome/llm/LlmLib.py:217` →
> `conversationgenome/llm/prompt_manager.py:49-50` →
> `conversationgenome/task_bundle/ConversationTaggingTaskBundle.py:161-162` →
> `neurons/validator.py:341-348` and `:369` →
> `GroundTruthTagSimilarityScoringMechanism._calc_scores`.
> The same path is reached with **no miner cooperation** whenever the validator's own LLM call
> fails: `llm_openai.py:56` returns `None` on any API exception, and `LlmLib.py:220` then does
> `len(None)` → `TypeError`. In that case honest miners that send `vectors: None` are scored
> 0.0 (a `TypeError` inside `_calc_scores`, caught and zeroed), while a miner that ships a
> populated `vectors` dict is scored normally or better.
>
> **Impact.** Measured on 120 captured mainnet conversation tasks against an independent
> ground-truth draw, using the real bundle, scorer and penalty code: honest 0.5580 final; a
> forged-centroid answer 0.8080 (+0.25, 120/120 wins). Most seriously, because
> `scoring_factors` sum to 1.0, a single repeated vector collapses all four scoring terms to
> one cosine — and one **fixed, task-independent** vector (the mean direction of any large
> sample of `text-embedding-3-small` tag embeddings) scores 0.7420 adjusted / 0.6678 final and
> beats the honest pipeline on 104/120 tasks while reading neither the conversation nor the
> task. Verified by split-half: two disjoint 40k-vector samples yield mean directions with
> cos 0.9999 that both score ~0.754 mean against 196 held-out GT centroids.
> Exposed task types are `conversation_tagging`, `webpage_metadata_generation`,
> `survey_tagging` and `skill_generation` — 74.4% of captured traffic.
> `named_entities_extraction` is not exposed (it uses `validate_named_entities_tag_set`, which
> returns `[]` rather than raising, so vectors are correctly overwritten).
>
> **We have not deployed this and do not intend to.** We are reporting it because it removes
> the incentive to do the work at all.
>
> **No evidence of exploitation.** We scanned 7,139 per-tag cosine breakdowns from validator
> `bt_log`: a forged-single-vector answer produces cosine spread of exactly 0.0, and the
> observed minimum spread is 0.0179 (p1 0.0297, median 0.1324) with zero rows at 0.0.
>
> **Suggested fixes**, in our order of preference:
> 1. **Never read a miner-supplied `vectors` field.** Have `_calc_scores` embed the tag
>    strings itself, or have `format_results` build the dict fresh. The miner has no
>    legitimate reason to supply vectors and the validator re-embeds anyway. This kills the
>    whole class, including the LLM-failure variant.
> 2. Drop (or zero) the response when `format_results` raises, rather than only skipping its
>    logging.
> 3. Let `basic_prompt` raise, or return a sentinel the caller checks, so an LLM outage becomes
>    "skip this task" instead of "score with whatever the miner sent".
> 4. Validate `cgp_output` against a strict schema; it is currently `Optional[List[dict]]`
>    with no shape constraints.
>
> **Free detection, no code change:** alert on a `Unique Tag Scores` list with zero spread.
> That never occurs naturally in 7,139 observed rows.
>
> **Two adjacent honest-miner defects worth fixing in the same pass:**
> (a) ground truth is normalised by `Utils.clean_tags` while miner tags go through
> `get_safe_tag`/`get_clean_tag_set`, and the two are then compared by exact set intersection —
> so 12.85% of GT tags (hyphenated, possessive, dotted, or <3 chars) are unmatchable by any
> miner, and 1.21% of tasks force the `no_both_tags` penalty unconditionally.
> (b) `LlmLib.py:228` filters the screen's cleaned output against the miner's **raw** tag list,
> so a screen-approved tag like `Machine Learning` or `e-commerce` is deleted anyway; enough
> such deletions drop the answer below `min_tags` and discard it entirely.

---

## 5. LEGITIMATE optimisations in use

**L1 — NER composite / multi-entity tags** (e.g. `jamie torres kh streetcar llc`), measured
+0.055 vs cohort on 50 live tasks.

**Evidence that this is fair play, not an exploit:**

1. Every tag is **genuinely derived from the transcript the validator supplied**. No withheld
   ground truth is reconstructed, nothing is forged, no other miner is affected.
2. It uses the **documented, intended** submission channel: real tag strings that the
   validator embeds itself. It survives the normal chain with all validator-side processing
   intact — verified, since NER's `format_results` completes and **overwrites** `vectors`
   (my repro: `NER: COMPLETED, tags=[], vectors OVERWRITTEN`).
3. The gain is a property of the metric the subnet chose to reward: GT is a centroid of
   entity embeddings, and a composite entity string lands nearer that centroid than any single
   entity. That is "answering the question that was asked well", the same category as
   translating to English or ranking candidates by cosine.
4. The absence of an LLM screen and of penalties on NER is a deliberate upstream choice
   (`NoPenaltyGroundTruthTagSimilarityScoringMechanism` carries the comment *"We skip the
   penalty for some task types (like NER)"*), not a bug being abused.

**Adjudication: LEGITIMATE.** It is nonetheless worth mentioning to ReadyAI as a *metric*
observation — it is the same centroid-geometry effect as E2 in benign form, and if they
consider composite entities off-spec they can say so in the prompt. Recommend disclosing it as
an observation, and continuing to use it unless they object.

Also legitimate and in use, for completeness: translate-to-English (D5 defence), the
screen-safe floor (D6 defence), pre-normalised tag emission (D3 defence), the
`malformed`-rejection guard (D4 defence), enrichment-first fallback and the combine-overlap
latency fix. All are fair-play responses to defects, and none depends on any of the above.

---

## 6. NOT REPRODUCED / corrected — do not re-investigate

| claim | verdict |
|---|---|
| E1 adversarial score is 0.7135 final / 0.7928 adjusted | **Corrected, not refuted.** An independent run on n=120 got 0.8080 / 0.8978 with a stronger forgery. The honest arms agree (0.5580 vs 0.5501, within 0.008). The adversarial figure is a property of the attacker's forgery, not of the code, so it must be quoted as a **range (+0.11 to +0.25)**, never as a point estimate. |
| `cos(mu_A, mu_B) = 0.9893` for split-half mean directions | **Corrected upward.** My independent split-half gives **0.9999** (40k/40k disjoint). The conclusion is unchanged and strengthened. |
| Tag length bounds are 3–50 chars (per `CLAUDE.md`) | **Wrong.** `Utils.py:282` bounds are **3..64**. The 50 is a separate truncation at `LlmLib.py:216` applied *after* the random-20 sample; a 51–64 char tag is then deleted by the `element in tags` filter at `:228`. Verified: 64 chars survives, 65 does not. |
| "Validators set no temperature (API default 1.0)" (per `CLAUDE.md`) | **Wrong on this tree.** `llm_openai.py:27` sets `temperature: 0`, so the GT draw and the screen are near-deterministic. (The temperature-400 retry at `:43-55` is *our* local change; upstream `basic_prompt` returning `None` on error is identical.) |
| Our miner ships forged vectors today | **No.** `sn33/adapter.py:191` returns `{"tags": ..., "vectors": None}` — the key is on the wire but the value is null. Does not change D1: null is exactly what gets us zeroed. |
| Anyone is currently exploiting E1 on mainnet | **No evidence.** 7,139 rows, zero at zero-spread; only 28/7,205 rows exceed adj 0.80 whereas the strong arm yields ~0.90 on nearly every task. |
| More `both` (verbatim-GT) tags help — leaf's 2024 logs | **Still not reproduced**, and structurally barred: `both` tags are excluded from the 55% top-3-**unique** term. Unchanged from prior audits. |

---

## 7. What this audit could NOT examine

* **Live validator behaviour.** Out of scope by instruction. Every result is local replay
  against captured tasks and a frozen independent GT draw. Real validator LLM health, real
  batch composition and real EMA dynamics were not observed.
* **Actual axon/dendrite wire transport.** Only the pydantic `CgSynapse` model was
  round-tripped locally (forged 1536 floats survive bit-exact). Whether a large forged payload
  clears real axon size/time limits was not tested — local-only scope. A 3-tag payload is
  59.1 KB, well inside plausible limits, but this is untested on the wire.
* **The E1 score arms were reproduced once, independently, not twice.** The *mechanics* (the
  raise, the surviving forged dict, the response staying in `responses`, NER's non-exposure)
  and the E2 constant-vector geometry were verified independently by me this session with zero
  API calls. The n=120 score magnitudes come from a single independent reproduction and should
  be treated as one measurement, not a replicated one.
* **`SkillCoverageScoringMechanism`** was read for penalty constants only. The
  `judge_section_tests.j2` judging path and `skill_coverage_evaluation` were **not** audited
  for injection or scoring defects. This is the largest unexamined surface and the obvious
  next target — `skill_generation` is one of our two weakest task types (0.490 vs 0.660).
* **Prompt injection through `{{tags_string}}`** (`validate_tags.j2`) remains **untested** for
  whether the model obeys. It was assessed as not worth pursuing — the screen only decides
  which submitted tags are scored, and a junk tag still scores ~0 on the cosine — but "not
  worth pursuing" is not the same as "does not work", and it would be an EXPLOIT if it did.
* **Strix / any external security tooling** was not installed or run. All analysis is manual
  code reading plus local Python reproduction.
* **OpenAI spend:** the scratchpad key is out of credits, so every result above came from the
  warm cache (`data/sn33_cache/cache.sqlite3`) or from `data/frozen_judge.json`. Total spend
  this session: $0.
