# SN33 validator scoring surface — source map

Defensive audit, phase 1: **map only**. No probing, no scoring runs, nothing
deployed. Every claim below is a read of the source at the tree checked out in
`/home/anirudh/bittensor-conversation-genome-project`.

Tree audited: branch `skill-coverage-upgrade`, HEAD `e284c1e`.
Validator-side code in this tree is upstream ReadyAI (PR#135 merge `c80668c`,
authored by JordanBourgault, also on `origin/Feat-Add-Skills-Cov-task`) — our
own commits (`98ce191`, `00d078d`, `ff353bf`, `e284c1e`) touch only miner-side
`sn33/`, prompts and `neurons/miner.py`. Verified with
`git log -S / git branch --contains`. The one exception worth knowing:
`conversationgenome/llm/llm_openai.py:43-55` (temperature-400 retry) is **ours**,
but the failure semantics that matter to this audit (`basic_prompt` returns
`None` on any exception) are identical upstream — see
`git show d4129ed:conversationgenome/llm/llm_openai.py:30-37`.

---

## 0. Common spine (all tag-scored types)

```
miner axon  --CgSynapse.cgp_output--> validator
neurons/validator.py:269   dendrite.forward(axons, synapse, deserialize=False)   [timeout 12.0s, dendrite default]
neurons/validator.py:285-302  status-code bookkeeping + commitment refresh
neurons/validator.py:308-319  conditional retry, responses[idx] overwritten
neurons/validator.py:330-332  `if not response.cgp_output: continue`   <- not scored here, but STILL in `responses`
neurons/validator.py:339      miner_result = response.cgp_output[0]
neurons/validator.py:340-348  miner_result = await task_bundle.format_results(miner_result)   [try/except -> continue]
neurons/validator.py:357-367  put_task() ships miner_result to the ReadyAI API
neurons/validator.py:369      (final_scores, rank_scores) = await task_bundle.evaluate(miner_responses=responses)
neurons/validator.py:402      self.update_scores(rank_scores, miner_uids)
```

`format_results` **mutates `response.cgp_output[0]` in place**; `evaluate` later
re-reads the very same object
(`GroundTruthTagSimilarityScoringMechanism.py:50,65`). That aliasing is the hinge
for hypothesis H1 below.

### 0.1 The two normalisation functions — they are NOT the same

| function | file:line | behaviour |
|---|---|---|
| `Utils.clean_tags` | `utils/Utils.py:219-223` | `strip().lower().replace('"','')`. **Keeps** hyphens, dots, apostrophes, accents, any length. |
| `Utils.get_safe_tag` | `utils/Utils.py:269-272` | `re.sub(r'\s{2,}|[^a-zA-Z0-9\s]', ' ', s)` then `re.sub(r'[^\w\s]|(?<=\s)\s*','',...).lower().strip()`. Destroys everything outside `[a-zA-Z0-9\s]`. |
| `Utils.get_clean_tag_set` | `utils/Utils.py:275-289` | `get_safe_tag` each tag, **drop if `len<3` or `len>64`**, collect into a `set()` -> **dedup + nondeterministic order**, return `list`. Bare `except -> []`. |

**Ground truth is built with `clean_tags`. Miner answers are filtered with
`get_safe_tag`.** Two different alphabets, compared with `==` (§0.4). Note also
`get_clean_tag_set` bounds are **3..64**, not 3..50 — the 50 comes from a
separate truncation in `validate_tag_set:216`.

### 0.2 `validate_tag_set` — the English keyword screen (conversation, webpage, survey, skill_generation)

`conversationgenome/llm/LlmLib.py:208-228`

```
208  if not tags: return None
212  clean_tag_list = Utils.get_clean_tag_set(tags)          # shred + drop <3/>64 + dedup + reorder
213-215 if len(clean_tag_list) >= 20: random.sample(range(n),20)   # at exactly 20 this is a SHUFFLE (lossless); loss starts at 21
216  clean_tag_list = [tag[:50] for tag in clean_tag_list]   # 50-char truncation AFTER the sample
217  prompt = prompt_manager.validate_tags_prompt(clean_tag_list)   # <- RAISES ValueError if list is empty (prompt_manager.py:48-51)
218  response_content = self.basic_prompt(prompt)            # returns "" or None on failure (llm_openai.py:56-59)
219  if len(response_content) == 0: return None              # <- TypeError if response_content is None
223-225 content_str = response_content.lower()
        malformed_pos = content_str.find("malformed")        # -1 when the word is absent
        good = content_str[0:malformed_pos]...               # -> content[0:-1], clips the last character
226-227 valid_tags = get_clean_tag_set(good.split(","))
228  return [element for element in valid_tags if element in tags]   # membership against the RAW submitted list
```

Silent drops / mutations, in order:
1. non-`[a-zA-Z0-9 ]` characters shredded (accented letters wreck the word; emoji strip cleanly);
2. anything <3 or >64 chars after shredding **deleted**;
3. duplicates collapsed, order randomised (`set`);
4. at >=21 clean tags, a uniformly random 20 kept;
5. truncation to 50 chars — a truncated tag can no longer satisfy `element in tags` at line 228, so **any 51..64-char tag is silently deleted at the last step**;
6. whatever the LLM screen calls "malformed";
7. anything the screen returns that was not literally in the raw submitted list.

Failure modes of the screen:
* **empty string reply** -> `return None` -> `miner_result['tags'] = None` -> `_has_enough_tags` raises -> caught -> **hard 0.0**.
* **`None` reply (API error/outage)** -> `len(None)` **TypeError** -> propagates out of `format_results` -> `validator.py:342` catches, `continue` -> **the response is still scored, with the miner's raw untouched `tags` and the miner's own `vectors`** (H1).
* **reply lacking the word "malformed"** -> `find()` = -1 -> last character clipped -> that tag fails `element in tags` -> lost. (Known finding #1, still present, unchanged.)
* **a submitted tag containing "malformed"** -> parse cut at the first occurrence -> everything after it lost. (Known finding #1.)
* **screen deletes everything** -> `[]` -> `len < min_tags` -> **hard 0.0 discard** (known finding #3).
* **all tags uncleanable** (all <3 or >64 chars, or all non-latin) -> `clean_tag_list == []` -> `validate_tags_prompt` raises **ValueError** -> same bypass as the `None` case (H1).

NER uses `validate_named_entities_tag_set` (`LlmLib.py:231-237`) instead: clean +
random-20 only, **no LLM screen, no `element in tags` filter, cannot raise**.

### 0.3 Embedding regeneration

`format_results` overwrites `miner_result['vectors']` with
`llml.get_vector_embeddings_set(tags)` (`LlmLib.py:66-80`), which re-cleans with
`get_clean_tag_set` and embeds with `text-embedding-3-small`, `dimensions=1536`
(`llm_openai.py:15,61-75`). Miner-supplied vectors are therefore **normally
discarded** — except on the exception path of §0.2, where they are not.
A failed embedding yields `{"vectors": []}` -> `_score_vector_similarity`
`np.dot` raises -> bare `except` -> score **0** (`GroundTruth...py:270-274`).

### 0.4 The scoring maths

`GroundTruthTagSimilarityScoringMechanism.py`

* **Target vector** (`:162-180`): `np.mean` of *all* GT tag vectors in
  `task_bundle.input.metadata.vectors`. One 1536-d centroid. `tag_count_ceiling`
  is never passed. Empty dict -> `None` -> every cosine 0.
* **Candidate set** (`:192`): `tag_set = list(set(tags))` — dedup, **arbitrary order**.
* **unique / both** (`:193`, `Utils.compare_arrays:48-58`): plain Python `set`
  intersection/difference of GT tag strings vs miner tag strings.
  `both = GT ∩ miner`, `unique = miner \ GT`. **Exact string equality**, across
  the two different normalisations of §0.1.
* **Scored subset** (`:195-198`): `if idx > self.max_scored_tags: break` with
  `max_scored_tags = 20` -> indices 0..20 -> **21 tags scored**, chosen by
  `set` iteration order. Unreachable on the normal path (both cleaners cap at
  20); reachable on the H1 bypass.
* **Per-tag score** (`:216`, `:260-275`): cosine(centroid, tag vector).
  All-zero vector -> 0. Any exception -> 0.
* **Stats** (`:121-152`): `mean/median/min/max` over **all scored** tags;
  `top_3_mean` over the **unique** ones only, sorted ascending, last 3, **zero-padded to 3**;
  `len(scores_unique)==0` -> `[0,0,0]`.
* **adjusted** (`:154-160`):
  `0.55*top_3_mean + 0.25*mean + 0.10*median + 0.10*max`.
* **Tie-breaks**: none anywhere. `np.sort` is a stable value sort; no tag identity is preserved.
* **Randomness in scoring**: (a) `random.sample` at `LlmLib.py:214/235`;
  (b) `set` iteration order at `Utils.py:287` and `GroundTruth...py:192`;
  (c) the GT itself is one LLM draw with `temperature=0` locally but the ReadyAI
  validator's own `basic_prompt` — upstream `d4129ed` — sends **no temperature**.

### 0.5 Penalties (`_calculate_penalty`, `:228-258`) — multiplicative, applied in this fixed order

| # | condition | source of the count | multiplier | constant |
|---|---|---|---|---|
| 1 | `num_both_tags == 0` | `len(diff['both'])` over the **full** tag set | **x0.9** | `constants.py:6-8` |
| 2 | `max_score < 0.2` | max over **scored** tags | **x0.5** | `constants.py:9-12` |
| 3 | `num_tags < 2` | `len(both)+len(unique)`, **full** set | **x0.2** | `constants.py:13-16` |
| 4 | `num_unique < 1` / `< 2` / `< 3` (elif chain, only one fires) | full set | **x0.85 / x0.9 / x0.95** | `constants.py:17-27` |

Key asymmetry: **penalty counts use the full submitted set (`:86-88`), the score
statistics use only the first 21 tags (`:195-198`).** A miner with 25 tags whose
21 scored happen to all be `both` gets `top_3_mean = 0` (55% of the score gone)
yet no unique-tag penalty.

NER replaces this whole method with `return score`
(`NoPenaltyGroundTruthTagSimilarityScoringMechanism.py:8-10`).

### 0.6 Hard-zero / discard paths (score exactly 0.0, not merely low)

| path | file:line |
|---|---|
| no `cgp_output` at all (timeout, 500, unreachable) | `validator.py:330`; the entry still lands at `GroundTruth...py:62-63` -> 0.0 |
| `cgp_output[0]` falsy | `GroundTruth...py:62-63` |
| `len(tags) < min_tags` (3; **1** for survey and NER) | `GroundTruth...py:66-68`, `:110-119` |
| `miner_result['tags']` missing / not a list -> exception in `_has_enough_tags` | `GroundTruth...py:115-118` |
| any exception inside `_calc_scores` (e.g. `miner_result['vectors']` missing -> `KeyError`) | `GroundTruth...py:76-80` |
| GT metadata has no vectors -> centroid `None` -> every cosine 0 -> adjusted 0 | `GroundTruth...py:176-180`, `:271` |
| screen deletes every tag | `LlmLib.py:228` -> empty list -> min_tags gate |
| screen returns empty string | `LlmLib.py:219-221` -> `tags=None` -> min_tags gate |

### 0.7 Timing / retry

* Deadline: `dendrite.forward` is called with **no `timeout` argument**
  (`validator.py:269-273`) -> bittensor default **12.0 s** end-to-end, including
  transport (`bittensor/core/dendrite.py:399,541`).
* A "failed response" = `status_code != 200` **or** empty `cgp_output`
  (`validator.py:291-292`).
* Any non-success triggers a commitment refresh (`:294-295`, debounced 5 min per
  UID at `:86-100`).
* **Exactly one** retry, and only for `status_code in {408, 422, 503, None}`
  (`:282`, `:297-319`) — i.e. timeout, unprocessable, unavailable, connection
  failure. A 500 from the miner earns **no** retry. The retry reuses the *same*
  synapse object, so the miner sees an identical task.
* Non-responders are still enumerated in `responses` and scored 0.0, which feeds
  `update_scores`.

### 0.8 After `final_miner_score`

`validator.py:402` -> `base/validator.py:466-490` -> `ValidatorLib.update_scores:97-160`:
`scattered_rewards[uids] = rewards` (wrapped in try/except at `:121-124` — a
length mismatch is swallowed and the EMA still advances on stale data), EMA with
`alpha` for non-zero rewards and **`alpha/2` for zero** rewards (`:134-140`),
normalise, `**3.0` non-linear power (`base/validator.py:107`), renormalise.

---

## 1. Per-type traces

### 1.1 `conversation_tagging`

* Bundle: `task_bundle/ConversationTaggingTaskBundle.py`
* GT build: `_generate_metadata:211-251` — one `conversation_to_metadata` over
  <=300 lines, plus 1-3 randomly sampled enrichment results **per query**
  (`_build_enrichment_lines_and_tags:253-282`, `random.randint`/`random.sample`
  at `:264-265`), each -> `enrichment_to_metadata`; then
  `combine_metadata_tags(all_tags, generateEmbeddings=True)` (`:236`).
  The combine sees **1 conversation set vs N enrichment sets** — enrichment
  outvotes the transcript by construction.
  Tags via `Utils.clean_tags`; centroid via `get_vector_embeddings_set` (i.e.
  `get_safe_tag`) -> **GT tags and GT vector keys are in different alphabets**,
  and any GT tag >64 or <3 chars contributes **no vector to the centroid** while
  still counting for `both`/`unique`.
* Task masking: `mask_task_for_miner:176-182` + `TaskBundle.py:60-68` — guid and
  `window_idx` hidden.
* `format_results:157-163`: `original_tags` <- `tags`; `tags` <- `validate_tag_set` (§0.2, LLM screen ON); `vectors` <- re-embedded.
* Scoring: `GroundTruthTagSimilarityScoringMechanism`, `min_tags = 3`, **all four penalties live**.
* Miner text -> LLM: only `validate_tags.j2 {{tags_string}}` (`prompt_manager.py:48-51`, `",".join(tags)`). The model's output decides which submitted tags are scored at all.

### 1.2 `webpage_metadata_generation`

* Bundle: `task_bundle/WebpageMetadataGenerationTaskBundle.py`
* GT build `_generate_metadata:154-202`: `website_markdown[:1000]` ->
  `website_to_metadata`; enrichment identical to conversation
  (`random.randint`/`random.sample` at `:169-170`, snippet+title capped 1000
  chars); `combine_metadata_tags(..., generateEmbeddings=True):189`.
  Miner window is rewritten to `[(0, website_markdown)] + enrichment_lines` (`:186`).
* `format_results:115-121` — **identical** to conversation (LLM screen ON, re-embed).
* Scoring: `GroundTruthTagSimilarityScoringMechanism`, `min_tags = 3`, all penalties.
* Injection surface: `validate_tags.j2` only.

### 1.3 `named_entities_extraction`

* Bundle: `task_bundle/NamedEntitiesExtractionTaskBundle.py`
* Not in the `TaskType` literal (`constants.py:3`) — the union member is
  declared on the bundle itself (`:70`) and reaches pydantic via
  `task_bundle_factory.py:21,26`.
* GT build `_generate_metadata:81-119`: `transcript_text[:1000]` ->
  `raw_transcript_to_named_entities`; enrichment -> `enrichment_to_NER`;
  `combine_named_entities(tags, generateEmbeddings=True):112`.
  **Note `:96` `random.randint(1, min(3, len(results)))` is not guarded against
  `results == []`** -> `randint(1,0)` raises `ValueError` inside `setup()`.
* `format_results:143-149`: `validate_named_entities_tag_set` — clean + random-20,
  **no LLM screen**, no `element in tags` filter, no 50-char truncation, cannot raise.
* Scoring: `NoPenaltyGroundTruthTagSimilarityScoringMechanism`, `evaluator.min_tags = 1` (`:160`).
  **Zero penalties.** So the 55%-weight `top_3_mean` term — computed over
  *unique* tags only — is the whole game and there is no `no_both_tags` x0.9 to
  pay for never matching a GT string. This is the mechanical reason concatenated
  multi-entity tags score well (known finding #5): a long entity string is
  guaranteed-unique, survives the no-screen path, and only has to beat the
  centroid cosine of two other tags.
* Injection surface: **none** — no miner text reaches any validator LLM call.

### 1.4 `survey_tagging`

* Bundle: `task_bundle/SurveyTaggingTaskBundle.py`
* GT build `_generate_metadata:63-74`: **no LLM at all.**
  `tags = parsed_json['selected_choices']` **verbatim**, centroid =
  `get_vector_embeddings_set(selected_choices)`.
* Verified against a captured real task (`data/testnet_corpus.jsonl`):
  `selected_choices = ["Los intereses para créditos son bajos",
  "App móvil fácil de manejar", "Me depositan sueldo/pagos en ese banco"]`.
  Consequences, both structural:
  1. GT tag strings are **capitalised, accented, punctuated, sentence-length**.
     Miner tags after `validate_tag_set` are lowercase ASCII `[a-z0-9 ]`.
     `set` intersection is **provably empty** -> `no_both_tags` **x0.9 fires on
     essentially every survey task**, for every miner, unavoidably.
  2. The centroid is built from `get_clean_tag_set(selected_choices)`, i.e. from
     **accent-shredded** strings (`"los intereses para cr ditos son bajos"`), and
     any choice exceeding 64 chars is **dropped from the centroid entirely**.
     The survey target vector is therefore a degraded version of the intended one.
* `format_results:100-105`: `validate_tag_set` (LLM screen ON) + re-embed.
* Scoring: `GroundTruthTagSimilarityScoringMechanism` with `evaluator.min_tags = 1` (`:116`), **all penalties live**.
* Injection surface: `validate_tags.j2` only.

### 1.5 `skill_generation`

* Bundle: `task_bundle/SkillGenerationTaskBundle.py`
* GT build `_generate_metadata:151-177`: `skill_markdown[:1000]` ->
  `skill_to_metadata` -> `combine_metadata_tags([tags], generateEmbeddings=True)`.
  **No enrichment** — the combine sees exactly one set, so the "1-vs-N vote"
  dilution of the conversation path does not apply here.
* `format_results:112-118`: identical to conversation (LLM screen ON, re-embed).
* Scoring: `GroundTruthTagSimilarityScoringMechanism`, `min_tags = 3`, all penalties.
* Injection surface: `validate_tags.j2` only.

### 1.6 `skill_coverage_evaluation`

Completely different mechanism. Bundle
`task_bundle/SkillCoverageEvaluationTaskBundle.py`, scorer
`scoring_mechanism/SkillCoverageScoringMechanism.py`.

**Setup** (`_generate_section_map:323-353`): `skill_request_to_section_map(seed)`
-> N sections; **`random.sample(result.sections, min(3, N))`** (`:342`,
`MAX_EVALUATED_SECTIONS = 3` at `:52`) picks the scored subset *once per bundle*, the
same for every miner; only those get `section_vectors` (embedding of
`"{title}: {description}"`). The miner receives the **full** section map and does
not learn which 3 count.

**`format_results:137-241`** — the only place miner data is sanitised:
1. `:146-147` scope to evaluated section ids;
2. `:155-158` `skill` coerced to `str`, truncated to `MAX_SKILL_CHARS = 6000` (`:63`);
3. `:160-167` `section_tests` coerced to `dict`, non-`list` values dropped, keys outside the evaluated set dropped;
4. `:243-263` `_sanitize_test` — non-dict -> `None` (dropped); `name` clamped to 200, `description`/`assertion` clamped to 1000 (`:64-65`);
5. `:174-177` **`PER_SECTION_TEST_CAP = 6`** (`:39`, shared with `PENALTIES["test_flooding"]["threshold"]`) — tests past it are never embedded and never judged, but remain visible with `vector=None, judged_correct=False` (`:225-230`);
6. `:178` `_judge_section_tests` (below);
7. `:182-215` one batched embedding call over `skill_text` + each eligible test's `f"{description}\nAssertion: {assertion}"`.

**The judge** (`_judge_section_tests:265-317`): `MAX_JUDGED_TESTS = 20` total
(`:44`) taken greedily in `dict` iteration order of `section_tests` — i.e. **in
the order the miner's own JSON listed the sections**. Prompt
`prompts/judge_section_tests.j2`; verdicts keyed
`(section_id, verdict.name)` at `:313-317`. **Fails closed**: no result -> `{}`
-> nothing credited (`:308-311`). `judge_section_tests_prompt` raises
`ValueError` if `skill_markdown` is blank (`prompt_manager.py:93-96`), but
`:292` already returns `{}` for blank skill text, so that is unreachable.

**Scoring** (`SkillCoverageScoringMechanism`):
* gate `_has_enough_tests:141-151`: total tests across sections `< min_tests = 3` (`:54`) -> **0.0**;
* `_score_sections:153-170`: per evaluated section, over tests with
  `judged_correct == True` **and** a non-null vector, cosine to the section
  vector, take `sorted(...)[:top_k_per_section = 2]` (`:55`) and **mean**;
  a section with no qualifying test scores **0.0** and still counts in the mean;
* `section_coverage = mean(section_scores)` (`:107`);
* `skill_coverage = _score_skill_coverage:172-186` = cosine( mean of the miner's
  own judged-correct test vectors , the miner's own `skill_vector` ).
  **No ground truth is involved in this term at all.**
* `adjusted = 0.6*section_coverage + 0.4*skill_coverage` (`:57-60`, `:188-192`).

**Penalties** (`_calculate_penalty:194-231`), in this order:
| # | condition | effect | constant |
|---|---|---|---|
| 0 | `sections_addressed == 0` (no evaluated section scored >0) | **return 0.0** (hard zero, short-circuits) | `:198-200` |
| 1 | `total_tests < 5` | x0.5 | `constants.py:29-32` |
| 2 | any two submitted vectors with cosine >= 0.95 | x0.5 | `constants.py:33-36`, `_has_near_duplicate_tests:245-256` |
| 3 | judged accuracy `< 0.5` (correct / **total incl. uncapped, unjudged** tests) | x0.7 | `constants.py:37-40`, `_judged_accuracy:237-243` |
| 4 | any section with `> 6` tests | x0.6 | `constants.py:43-46`, `_has_flooded_section:233-235` |

Note the interaction: exceeding the cap is punished **twice** — the excess
tests are unjudged (so they drag `_judged_accuracy` down toward penalty 3) and
then penalty 4 fires as well.

**Hard zeros**: `<3` tests; no cgp_output; exception in `_score_sections`/
`_score_skill_coverage` (`:109-112`); judge failure -> nothing correct ->
`sections_addressed == 0` -> 0.0.

**Injection surface — the largest in the codebase.** The miner's `skill`,
`name`, `description` and `assertion` are all interpolated into
`judge_section_tests.j2` (`:29,33,37`), and the model's boolean output is the
sole gate on **both** scoring signals. The prompt carries an explicit
`<security>` block (`:9-11`) instructing the model to treat that content as data
and to mark instruction-shaped tests `correct: false`, and the caller fails
closed — the design is aware of the risk. Whether the mitigation holds is an
empirical question, listed below as H4.

---

## 2. Divergence hypotheses — INTENDED vs ACTUAL scoring

Ranked by plausible reward gain to a miner. **Nothing here has been tested. No
score has been run. Nothing below is a recommendation to deploy — items marked
EXPLOIT are candidates for disclosure to ReadyAI if they reproduce.**

**H1 — `format_results` raises -> the response is scored on the miner's RAW tags
and MINER-SUPPLIED vectors.** *(hypothesised EXPLOIT + DEFECT)*
`validator.py:340-348` catches the exception and `continue`s the *put_task* loop,
but `responses` is unchanged and `:369` evaluates it anyway; because
`format_results` mutates in place, `miner_result['tags']` is still the raw
submitted list and `miner_result['vectors']` is still whatever the miner sent
(`GroundTruth...py:184-185`). The miner then bypasses, in one move: the English
keyword screen (so no hard-zero risk), `get_safe_tag` shredding, the 3..64 length
filter, the 50-char truncation, the random-20 cull (21 tags scored instead of 20,
`:195-198`), and validator-side embedding.
Two candidate triggers, both reachable from the wire:
  * (a) **every** submitted tag fails `get_clean_tag_set` — e.g. all tags longer
    than 64 characters — so `clean_tag_list == []` and
    `prompt_manager.validate_tags_prompt` raises `ValueError`
    (`prompt_manager.py:49-50`). The surviving raw tags are then long,
    semantically dense English phrases with the miner's own honest
    `text-embedding-3-small` vectors — plausibly *closer* to the GT centroid than
    2-word tags, and guaranteed `unique`, which is where 55% of the weight sits.
  * (b) `basic_prompt` returns `None` on an OpenAI error -> `len(None)` TypeError
    (`LlmLib.py:219`). Not miner-controllable, but it means **every miner in the
    batch is scored on self-reported vectors during an OpenAI outage** — a
    validator-integrity defect independent of any exploit.
To verify: does `set()` ordering leave the miner's chosen 21 in play; does the
long-phrase cosine actually beat the short-tag cosine; does `no_both_tags` x0.9
eat the gain. Requires driving the real classes end-to-end.

**H2 — 40% of `skill_coverage_evaluation` is self-graded.** *(hypothesised EXPLOIT + DESIGN DEFECT)*
`_score_skill_coverage:172-186` measures the miner's test-suite centroid against
the **miner's own skill text**. No validator artefact enters. A miner whose
`skill` document is (or embeds near) the concatenation of its own test
descriptions drives that cosine toward 1.0 and banks 0.4 of `adjusted` without
covering anything. The only gate is `judged_correct`, and the judge sees the same
self-authored skill document it is asked to check assertions against — a test can
be "correct given the skill" while the skill says nothing useful. Docstring
`:22-27` states the intent ("analogous to the semantic neighborhood"); the
divergence is that a neighborhood measured against a miner-supplied anchor is not
a coverage measurement.

**H3 — survey `no_both_tags` is structurally unavoidable, and the survey centroid
is accent-shredded.** *(DEFECT, costs every honest miner)*
`SurveyTaggingTaskBundle.py:68` stores raw `selected_choices` as GT tags while
`format_results:103` reduces miner tags to `[a-z0-9 ]`; the intersection at
`Utils.compare_arrays:54` can never be non-empty for the observed Spanish
corpus, so `PENALTIES["no_both_tags"]` x0.9 (`GroundTruth...py:233-235`) is a
flat tax on the type. Separately `:69` embeds `get_clean_tag_set(selected_choices)`,
so the target vector is built from strings with the accented letters blown out
and any >64-char choice missing entirely. Predicted effect: survey scores capped
~10% below the other types and the target itself mis-specified — consistent with
the measured 0.410 survey mean. Fix belongs upstream (compare on a common
normalisation; embed the original strings).

**H4 — judge prompt injection in `skill_coverage_evaluation`.** *(hypothesised EXPLOIT)*
Miner-authored `skill`/`name`/`description`/`assertion` reach
`judge_section_tests.j2:29,33,37`, and a forced `correct: true` unlocks both
scoring terms. The `<security>` block (`:9-11`) plus fail-closed handling
(`:308-311`) are a real mitigation and the design anticipates this; the open
question is only whether `gpt-5.6-luna` (`llm_openai.py:136-138`) honours it
under adversarial text. Note fail-closed makes the *cheap* attack
(blow the context, force an empty reply) self-defeating — it zeroes the attacker.

**H5 — verdict keys are miner-controlled `name` strings.** *(hypothesised EXPLOIT)*
`:313-317` builds `{(section_id, verdict.name): correct}` and `:236` looks each
test up by its own `name`. Two tests in one section sharing a `name` collapse to
**one** verdict — a single `correct: true` credits both. Since only the top-2 per
section count (`:55`, `:166`), crediting 2 tests from 1 honest verdict is exactly
the amount that matters. `_has_near_duplicate_tests` (>=0.95 cosine, `:254`) is
the only backstop and it keys on *vectors*, not names — different
descriptions/assertions under an identical `name` sail past it. Also a
`name`-less/`""`-named batch collapses to a single key.

**H6 — penalty counts and score statistics are computed over different tag sets.**
*(DEFECT, mostly latent)*
`:86-88` counts `both`/`unique` over the **full** deduped set, `:195-198` scores
only the first 21 in `set` order. Divergences: (a) a miner submitting >21 tags
can score `top_3_mean = 0` while dodging every unique-tag penalty; (b) which 21
get scored is `PYTHONHASHSEED`-dependent, so **the same submission can score
differently on two validators**. Latent today because both cleaners cap at 20 —
it becomes live under H1.

**H7 — suppressing the word "malformed" keeps the whole tag list.**
*(hypothesised EXPLOIT built on known defect #1)*
`LlmLib.py:224-225`: if the screen's reply never contains "malformed", the split
point is `-1` and *everything but the last character* is treated as "good".
Tags injected into `{{tags_string}}` (`prompt_manager.py:51`) that steer the model
away from emitting that literal word would preserve acronym-heavy answers that
the screen would otherwise delete — i.e. convert a hard 0.0 into a normal score.
Upside is bounded (it recovers score the miner "should" have had) but it is
mechanism-gaming, so: report, do not deploy. The honest defence for the same
failure is the existing screen-safe floor.

**H8 — `_get_axons_for_uids` can desynchronise scores from UIDs.** *(DEFECT)*
`validator.py:128-146` **skips** any UID whose index exceeds
`len(metagraph.axons)`/`hotkeys` (`:132-134`), returning a **shorter** axon list,
while `responses[i]` is later zipped against `miner_uids[i]`
(`:317-319`, `:402` -> `scattered_rewards[uids_array] = rewards`). After a
metagraph shrink/resync race, response *i* can be attributed to the wrong UID; a
length mismatch is swallowed at `ValidatorLib.py:121-124` and the EMA advances on
stale data regardless. Costs honest miners; not exploitable by one.

**H9 — GT tag strings and GT vector keys are in different alphabets.** *(DEFECT)*
Every tag-scored type builds GT tags with `Utils.clean_tags` (punctuation and
accents preserved) but the centroid with `get_vector_embeddings_set` ->
`get_safe_tag` + 3..64 filter. So a GT tag like `e-commerce` (a) can never be
matched by a miner tag (which is forced to `e commerce`), and (b) contributes a
vector under a *different* key; a GT tag over 64 chars contributes **no vector at
all** while still counting in the `both`/`unique` diff. Net effect: `both` is
rarer than the design assumes — which inflates `top_3_mean` (unique-only, 55%
weight) and simultaneously makes the x0.9 `no_both_tags` tax more common. This is
the same root cause as H3, generalised.

**H10 — `zero_score_mask` is dead code.** *(NON-ISSUE, note only)*
Allocated `:29`, written `:67,:79`, never read. Harmless; flagged so a future
reader does not assume masking happens.

---

## 3. What phase 2 must do for each of the above

Per the method requirement, nothing above graduates from "hypothesis" without:
code path -> runnable local repro -> normal score -> adversarial score ->
expected gain -> survives the **real** classes (drive
`ConversationTaggingTaskBundle.format_results` /
`GroundTruthTagSimilarityScoringMechanism.evaluate` /
`SkillCoverageScoringMechanism.evaluate` directly, never a reimplementation) ->
classification. Priority order for probing: **H1, H2, H3, H5, H4, H6, H7, H8, H9.**
H3, H8, H9 are defects to disclose and defend against; H1, H2, H4, H5, H7 are
exploit candidates — write-up for ReadyAI, explicitly do-not-deploy.
