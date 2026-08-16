# SN33 miner — complete context for optimization

You are advising the operator of a **Bittensor SN33 (ReadyAI / conversation-genome)**
miner. Below is exactly how the validator scores us, exactly what our miner does
with a query, exactly what we have already measured, and exactly what has already
been tried and failed.

**Your job:** find changes that measurably raise our score. Every recommendation
must be justified by the data below or by an experiment you specify. **Do not
propose anything from intuition alone** — we have a graveyard of reasonable-sounding
ideas that measured zero or negative, listed in §7. If you re-open one, you must
say what new evidence justifies it.

---

## 0. THE STAKES — why "reasonable-sounding" is not good enough

This subnet has **256 miner slots and constant registration pressure**. New UIDs get
**24 hours of immunity**; the moment it expires, the lowest-incentive non-immune UID
is pruned at the next registration event.

**We have been deregistered 7 times, each within ~24-30h of registering**, always
right after immunity lapsed:

```
UID 137 → 120 → 33 → 69 → 24 → 12 → 91 (current)
```

The killer detail: **UID 33 was scoring 0.69 (top-cluster quality) and was still
pruned.** Incentive is a slow EMA that starts at zero; in 24h a new UID cannot
out-accumulate established miners regardless of how well it scores. So:

* a single bad task hurts disproportionately — the EMA is short and the field is
  compressed (ranks 10→200 span only ~0.04 score);
* a zero, a truncated answer, or 20 minutes of downtime can decide a pruning cycle;
* **consistency beats peak quality.** The top-10 miners' distinguishing feature is
  that their *worst* task scores ~0.60, not that their best scores higher.

Optimize for **floor and reliability first**, average second.

---

## 1. HOW SCORING WORKS (verified against validator source)

The validator generates ~20 **ground-truth (GT) tags** per task, embeds them
(`text-embedding-3-small`, 1536 dims), and averages them into **one target vector**.
Every tag we submit is embedded and scored as **cosine similarity to that single
mean vector**.

```
adjusted = 0.55*mean(top-3 UNIQUE tags) + 0.25*mean(all) + 0.10*median(all) + 0.10*max
final    = adjusted * penalties
```

* **UNIQUE** = our tag is NOT string-identical to a GT tag. **BOTH** = exact match.
* `top-3 mean` **zero-pads** if we have fewer than 3 unique tags — the single most
  expensive failure mode.
* Only ~21 tags are scored; at ≥20 cleaned tags the validator keeps a random 20.
* GT is generated at **temperature 1.0** — it is ONE random draw. Two runs on the
  same task produce different GT. Cohort variance is irreducible.
* **Penalties** (multiplicative): no exact match ×0.9; max<0.2 ×0.5; <2 tags ×0.2;
  <1/<2/<3 unique ×0.85/×0.9/×0.95. Measured total impact ≈0.006 → **not a lever**.
* **Tag hygiene** (violations are silently deleted before scoring):
  lowercase alphanumeric only, 3–50 chars, must survive the validator's
  `get_safe_tag` unchanged, **English only** (an LLM screen deletes non-English
  words, acronyms, non-dictionary compounds). If fewer than `min_tags`(3) survive,
  the whole response is **discarded → hard 0.0**.
* **Deadline:** the validator's `dendrite.forward` allows **12 seconds total,
  including network both ways**. We use an 11.0s internal deadline.

### Where GT actually comes from (this is the crux)

Each task arrives with a **document/window** plus **N enrichment lines** (search-result
snippets). The validator builds GT by tagging *each source separately* and then
running a **combine** step where **each source set is one equal vote**:

| task type | GT composition |
|---|---|
| `conversation_tagging` | 1 set from the FULL ~300-line conversation + 1 set per enrichment line → **~88% enrichment**. **We only receive a 10-line window** of that conversation. |
| `webpage_metadata_generation` | 1 set from page text (first 1000 chars) + 1 set per enrichment line → **~79% enrichment**. We see everything GT saw. |
| `named_entities_extraction` | entities from the document. **No LLM screen, no penalties.** |
| `survey_tagging` | the literal selected answer choices — **which we never see**. |
| `skill_generation` | document-derived (never analyzed in depth). |

---

## 2. WHAT OUR MINER DOES WITH A QUERY (actual implementation)

On receiving a task, one async pipeline runs against an 11.0s deadline:

### Stage 1 — parallel fan-out (all launched simultaneously)

1. **Replica** (`replica.replicate`) — we re-run *the validator's own prompts* to
   predict its GT:
   * 1 doc call (conversation XML / page text / NER document),
   * 1 call **per enrichment line**,
   * optionally 1 **deep** call per enrichment line (30 tags instead of 10) —
     candidate-pool material only,
   * then a **combine** call merging those sets, mirroring the validator.
   Output: `predicted_gt` (the aim point), plus per-line tag lists.
2. **Pool** (`_generate_pool`) — a wide, deliberately over-inclusive candidate list
   (~40 tags) from one LLM call on the document (+ enrichment snippets appended
   when `enrichment_first` is on).
3. **Theme tags** — one extra call for broad umbrella topics (conversation only).
4. **Local extraction** — spaCy noun-phrase extraction in a thread, so an answer
   exists even if every API call fails.

`_gather_within` bounds the whole fan-out; slow jobs are dropped, **completed
results are kept** (never all-or-nothing).

### Stage 2 — target, rank, compose

5. **Target vector** = mean embedding of the cleaned `predicted_gt` (when
   `enrichment_first` is on, weighted toward per-line enrichment tags).
6. **Candidates** = pool + predicted_gt + variants (plural/lexical) + corpus anchors
   + deep tags + per-line enrichment tags, all normalized to validator-safe form.
7. **One batched embedding call** for every candidate + target tags.
8. **Rank** by cosine to the target; demote candidates with no lexical/semantic
   anchor in the enrichment; drop near-duplicates (cos ≥0.93) except protected ones.
9. **Compose** the final answer with repair passes:
   * `target_tags`: **20** for conversation/webpage/skill, **10** for NER, 12 survey,
   * **insurance = 6** verbatim `predicted_gt` tags (to avoid the ×0.9 no-exact-match
     penalty),
   * **MIN_UNIQUE_TARGET = 5** (protects the zero-padded top-3 term),
   * **SCREEN_SAFE_FLOOR = 7** — at least 7 tags where *every word* is a real
     dictionary word, so the English screen can never delete us below min_tags.

### Degradation ladder (what ships if time runs out)

```
ranked          full pipeline completed          ~0.60-0.65 typical
fallback_enrich truncated → round-robin per-line enrichment tags + pool fill
pool            truncated → raw pool list        ~0.20-0.35  ← the disaster case
local           all APIs failed → spaCy tags     ~0.2
```

### Live configuration

```
SN33_DEADLINE_S=11.0        SN33_CALL_TIMEOUT_S=9.5
SN33_ENRICHMENT_FIRST=1     SN33_ENRICHMENT_FIRST_WEBPAGE=1
SN33_NER_COMBOS=1           SN33_DEEP_ENRICHMENT=1     SN33_THEME_TAGS=1
SN33_FALLBACK_ENRICH=1      SN33_EMBED_RETRY=1
(SN33_VALUE_ALLOC / SN33_LINE_QUOTA / SN33_HEAD_CAP deliberately OFF — see §7)
```

---

## 3. OUR MEASURED PERFORMANCE

Comparison instrument: the validator samples **exactly 6 miners per task**, so for
each task we have our score and ~5 same-task opponents ("cohort"). Cohort-relative
score is the only difficulty-controlled metric.

**By task type** (cohort-relative, stable across three separate UID eras):

| type | vs cohort | note |
|---|---|---|
| NER | **+0.03** (win rate 72-74%) | our strength; composite entity tags gave +0.102 |
| survey | ~+0.05 | fixed from 0.410 → ~0.70 |
| webpage | −0.02 | |
| **conversation** | **−0.04 to −0.05** | **our weakness, and ~50-89% of traffic** |
| skill | −0.03 | never properly diagnosed |

**Top-10 miners:** mean ~0.66-0.70, and critically **p10 ≈ 0.55-0.60** — their worst
tasks are good. Our historic sub-0.5 rate was 18% vs their 3%.

**Gap decomposition on ~57 same-task comparisons:** excluding our own sub-0.5 tasks,
our outperformance was **+0.010 (top-10 grade)**. The entire deficit was the tail.

**Conversation loss taxonomy (103 scored tasks), by pipeline state:**

| class | n | cohort-relative | share of gap |
|---|---|---|---|
| A healthy `ranked` | 70 | −0.027 | 37% (72% in the current post-fix era) |
| B truncated/degraded | 9 | **−0.232** | 41% |
| C fallback | 1 | −0.361 | 7% |
| D unjoined | 23 | −0.035 | 15% |

**Conclusion: truncation used to dominate; after our latency fix it is ~5% of tasks,
and the remaining gap is now mostly class A — i.e. ranking/selection quality.**

---

## 4. THE DATASET AVAILABLE

`data/all_mainnet_tasks_with_scores.csv` — **879 mainnet tasks** across 7 UID eras
(Aug 6–13), each row containing:

```
timestamp, uid, validator, task_type, duration_sec,
window[], enrichment[], survey_question,      ← the exact validator query
our_tags[], n_tags,                            ← exactly what we answered
task_id, adjusted_score, final_score,
cohort_mean, cohort_max, cohort_scores[], beat, n_others, lag_s, score_available
```

* **879/879 have our submitted tags**; 196+ have validator scores (22.3%).
* Missing scores are **empty, never 0**.
* 443 conversation tasks, **100% carry enrichment lines**.
* `data/frozen_judge.json` — an independent GT draw per task for offline scoring
  with the real formula.

**HARD LIMIT you must respect:** the validator publishes only
`hotkey / adjusted_score / final_miner_score / task_id` to W&B. **We can never see
any other miner's tags.** Top miners can only be profiled *numerically*. Any advice
premised on "look at what top miners wrote" is impossible — say so instead.

---

## 5. THE COVERAGE LAW (our most important measured finding)

Enrichment coverage helps **only from a broken base**; near the ceiling it is
**negative**:

* raising coverage 0.36 → 0.998 from a *truncated* base: **+0.093** ✅
* raising coverage 0.915 → 0.951 on healthy answers: **−0.013** ❌
  (on the subset where coverage genuinely rose: −0.002; where it was already
  complete, forced reallocation cost −0.019)

Among healthy `ranked` answers, coverage carries **zero** information
(r = +0.069, p=0.57, CI crosses zero). **Do not propose coverage/quota/allocation
work again without new evidence.**

Related measured signature on healthy answers (observational, causality unproven):
**concentration beats spread** — tags piled on ONE enrichment line correlate
+0.486 with cohort-relative score (top-quartile answers put ~17.8 of 20 tags on one
line vs ~12.1 for bottom-quartile).

---

## 6. STATISTICAL RULES (violating these produces false conclusions)

* Minimum resolvable effect ≈ **0.049 unpaired**; per-task cohort SE ≈ **0.016**.
  Do not claim anything smaller without a paired design and n≥25 per arm.
* **Always control for task type first.** An apparent era-over-era decline was
  proven to be task-mix shift (one era was 62% NER, another 89% conversation).
* **Analyze healthy (`source=ranked`) tasks separately from truncated ones** —
  pooling them manufactures correlations that vanish under control.
* Report **tasks improved vs degraded**, never the mean alone. A mean gain with
  several collapsed tasks is a failure here, because of §0.

---

## 7. ALREADY MEASURED AND REJECTED — do not re-propose without new evidence

| idea | measured result |
|---|---|
| value-based slot allocation | **−0.0132**, improved 29 / degraded 177 (n=266) |
| fixed per-line quota (≥2 tags/line) | W/L 30/50, ≈0 |
| conditional demotion | changed 6/126 answers — inert |
| demote-strength tuning (0.90/0.95/off) | changed 0-1 answers — inert |
| head-word diversity cap | neutral |
| maximising unique tags | +0.043 adjusted but **−0.024 final** |
| more candidates | −0.0015 |
| more shipped tags (12 → 18.7) | −0.019 |
| better centroid estimate | a *perfect* centroid is worth only +0.0036 |
| 3-way blends / LLM-fused phrases | −0.029 / −0.318 |
| pooling 3-5 GT draws | ≈0 |
| corpus vocabulary anchors | +0.0013 |
| penalty tuning | ≤0.006 total |
| embedding inversion (vec2text/ZSinvert) | wrong embedding space and/or ~10⁴× too slow |
| OpenAI priority tier | 2× cost, no measured truncation benefit |

**What DID work:** enrichment-first aiming; NER composite tags (+0.102 live);
screen-safe floor (eliminated hard zeros); non-English translation; survey pool
prompt (0.410→0.70); running `combine` concurrently with the deep-enrichment grace
(truncation rate 25%→0% through a slow-API window); enrichment-first truncation
fallback (+0.093 offline).

---

## 8. WHAT I WANT FROM YOU

1. Given §1–§3, **where is the remaining recoverable score**, and why?
2. Concrete changes to the **ranking/selection** stage (§2 steps 5-9) that could move
   conversation from −0.045 toward cohort parity — the largest opportunity, since
   conversation is most of our traffic and truncation is largely solved.
3. For each proposal state: the mechanism, why it beats what we already do, the
   **offline experiment** that would validate it on the 879-task corpus (arms,
   metrics, dev/validation/holdout split), the expected effect size versus the
   0.049 resolution floor, and the **regression risk** — especially to NER, which is
   currently our only positive track.
4. Explicitly flag anything you cannot establish from this data.
5. Rank your proposals by **expected value × confidence × implementation risk**,
   remembering §0: a change that raises the mean but occasionally produces a 0.3
   task is likely net-negative for survival.

Do not suggest modifying production without an offline result. Do not suggest
anything in §7 unless you can point to evidence that overturns it.
