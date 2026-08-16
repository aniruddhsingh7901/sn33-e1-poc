# SN33 miner — offline research brief

You are doing **offline research** on a Bittensor SN33 (ReadyAI / conversation-genome)
miner. Goal: use a large historical dataset to discover what actually makes an
answer score highly on this subnet, and build a strategy that measurably beats
our current system. **Do not modify production.** Evidence first; my hypotheses
below are hypotheses, not requirements — reject any of them the data refutes.

---

## 0. TWO FACTS THAT INVALIDATE THE OBVIOUS RESEARCH PLAN — read first

**(a) The "3 deregistered miners" are OUR OWN previous registrations.**
Same hotkey `5GHciELXCD51y94VMLXqQxDuVBs7tdAwiQG9NPeXuf16yTMB` (wallet `sn33/m1`),
re-registered after each pruning: **UID 120 → 33 → 69 → 24 (current)**. They are
four eras of *our* miner, not other people's miners. There is no third-party
strategy to copy from them. They are useful as a **longitudinal record of our own
config changes**, nothing more.

**(b) We can NEVER see another miner's tags.**
The validator logs to W&B exactly four keys per scored response
(`neurons/validator.py:381-388`): `hotkey.{uid}`, `adjusted_score.{uid}`,
`final_miner_score.{uid}`, `task_id.{uid}` — plus `task_type` on a *separate*
row. **No tag strings are ever published.** So:
* ✅ possible: identify top miners by hotkey, measure their score distributions,
  compare same-task head-to-head, profile them by task type/consistency.
* ❌ impossible: read what top miners submitted, diff our tags against theirs,
  learn their vocabulary. Any "analysis of top miners' outputs" is fabrication.

Reverse-engineering top miners is therefore limited to their **numeric
signature** (level, variance, per-type profile, tail behaviour, task-difficulty
draw). That is still informative — do it — but never claim to know their content.

---

## 1. HOW SCORING WORKS (verified against validator source — do not re-derive)

Every submitted tag is embedded (`text-embedding-3-small`, 1536 dims, hardcoded
`conversationgenome/llm/llm_openai.py:15`) and scored as **cosine to ONE vector**:
the mean of the ~20 ground-truth tag embeddings.

```
adjusted = 0.55*mean(top-3 UNIQUE) + 0.25*mean(all) + 0.10*median(all) + 0.10*max
final    = adjusted * penalties
```
* `unique` = our tag NOT string-equal to a GT tag; `both` = exact match.
* `top_3_mean` **zero-pads** below 3 unique tags (most expensive failure mode).
* Only 21 tags are scored; at ≥20 cleaned tags the validator keeps a random 20.
* GT is generated at **temperature 1.0** — one random draw. Cohort variance is
  irreducible; never chase a <0.02 effect on a handful of tasks.
* Penalties (`utils/constants.py`): no exact match ×0.9, max<0.2 ×0.5, <2 tags
  ×0.2, <1/<2/<3 unique ×0.85/×0.9/×0.95. **Penalties are a measured dead end**
  (worth ~0.006 total); do not spend time there.
* Tag hygiene: must be a fixed point of `Utils.get_safe_tag` (lowercase
  alphanumeric), 3–50 chars, **English only** (an LLM screen deletes non-English
  and acronyms; dropping below `min_tags`=3 is a hard 0.0), answer within ~11s.

**Ground-truth composition (measured):** for `conversation_tagging`, GT is a
combine of 1 conversation-derived tag set + N enrichment-line sets, each line one
equal vote → **enrichment supplies ~88%** of GT, the 10-line window ~10%. For
`webpage_metadata_generation` it is ~79% enrichment. For NER, GT is entities from
the document (no LLM screen, **no penalties**). For survey, GT is the literal
selected answer choices, which the miner never sees.

---

## 2. THE DATA — what exists, where, and its limits

### 2.1 Our tasks + our answers (813 tasks, continuous 08-06 → now)
`data/by_uid/{uid120,uid33,uid69,uid24}_tasks.json`
Each record: `ts, dt, task_type, mode (mainnet|testnet), validator,
duration_sec, window[], enrichment[], survey_question, our_tags[], score{...}`.

```
era      tasks  mainnet  conv  webpage  NER  skill  survey
uid120    343      262    234      38     35    16     20
uid33     204      157     66      40     87     9      2
uid69     191      145     97      42     39     7      6
uid24      75       53     46       4     10     6      9
```
* **443/443 conversation tasks carry `enrichment_lines`** — this is what makes
  enrichment experiments possible.
* testnet tasks are scored by the same formula; usable, just labelled.

### 2.2 W&B scores for ~256 miners
Project **`afterparty/conversationgenome`** (hardcoded `analytics/WandbLib.py:29-30`;
the `template-validators` default in `utils/config.py` is vestigial — ignore it).
Local pulls: `data/wb_taskid.jsonl`, `data/wb_window33.jsonl`, `data/wb_gap.jsonl`
(fields: `validator, ts, hotkey, final, adjusted, task_id, wb_tt`).
The validator samples **exactly 6 miners per task**, so each `task_id` yields us
plus ~5 same-task opponents — this is the only unbiased comparison instrument.

### 2.3 FOUR DATA TRAPS — each already cost us real time

1. **netuid contamination.** The same W&B project also hosts **netuid 138**
   validators (`validator-160` = `5CyfmwLkHXoG`, `validator-243` = `5ChqLcW9N3zB`).
   **Always filter `run.config["netuid"] == 33`.** Ours are `validator-78`
   (`5GQyFzvtVMw9`), `validator-70` (`5DkzbwTnocec`), `validator-7` (`5FZe9Mpo5dLX`).
2. **A 4th validator publishes nothing.** `5FbGp2hED3Ef` sends us **20.9% of all
   tasks (27.6% of mainnet)** and has never appeared in W&B. Those tasks are
   scored on-chain but are **permanently unobservable**. Never interpret a missing
   score as a rejection or a failure.
3. **The task↔score join is type-blind.** Our miner never sees the task id (the
   validator masks it as `HIDDEN`), so joins must be inferred. A naive
   nearest-timestamp join was audited at **51% wrong**
   (`scripts/sn33_join_scores.py:82-101`). Correct join = same validator **+**
   score-after-task lag ∈ (0, 1200s] **+ task-type agreement**, one-to-one,
   smallest lag. Store `lag_s` and audit it.
4. **Only ~26% of answered tasks ever get a W&B score** (122 answered → 32 scored
   on a typical day). Absence of a score is the common case.

### 2.4 The offline judge (already built and frozen)
`data/frozen_judge.json` — one **independent** ground-truth draw per task
(gpt-5.2, no cache), 266 tasks with ≥5 GT tags. Score arms against this with the
real formula. Rationale: scoring an arm against ground truth it helped generate
is circular and produced a wrong verdict once already.
Do **not** use the enrichment-centroid "proxy" as the sole judge — it structurally
rewards centroid-collapse and therefore cannot rank slot-allocation strategies.

### 2.5 Cost constraints (hard)
* **OpenAI credits belong to the LIVE miner.** A credit outage on 2026-08-10 cost
  a production zero. Chat for research must route through OpenRouter
  (`SN33_CHAT_BASE_URL=https://openrouter.ai/api/v1`, `SN33_CHAT_MODEL_PREFIX=openai/`,
  exact model `openai/gpt-5.2`). **OpenRouter has no embeddings endpoint** —
  embeddings must use OpenAI, so lean on the sqlite cache
  (`data/proxy_cache.sqlite3`) and batch aggressively.
* Check credit balance before any large run; report cost.

---

## 3. WHAT HAS ALREADY BEEN MEASURED — do not rediscover

**Shipped and proven (live):** enrichment-first aiming for conversation+webpage;
NER composite tags (**+0.102 live, n=34**); screen-safe floor (guarantees ≥7
dictionary-word tags → hard zeros eliminated); non-English translation; a survey
pool prompt (0.410 → ~0.70); combine runs concurrently with the deep-enrichment
grace (truncation fix); enrichment-first truncation fallback (**+0.0928 offline**,
improved 142 / degraded 37); embed-failure retry + always-log.

**Measured and REJECTED — re-proposing these without new evidence is a waste:**

| idea | result |
|---|---|
| per-line fixed quota (≥2 tags/line) | W/L 30/50, ≈0 |
| **value-based slot allocation** | **−0.0132, improved 29 / degraded 177** (frozen judge, n=266) |
| conditional demotion | changed 6/126 answers, inert |
| demote-strength tuning (0.90/0.95/off) | changed 0–1 answers, inert |
| head-word cap | folded into the above, neutral |
| maximising unique tags | +0.043 adjusted but **−0.024 final** |
| more candidates per se | −0.0015 |
| more shipped tags per se (12→18.7) | −0.019 |
| improving the centroid estimate | a *perfect* centroid is worth +0.0036 |
| 3-way blends / LLM-fused phrases | −0.029 / −0.318 |
| pooling 3-5 GT draws | ≈0 |
| corpus vocabulary anchors | +0.0013 |
| penalty tuning | ≤0.006 total, irrelevant |
| embedding inversion (vec2text / ZSinvert) | wrong embedding space and/or ~4 orders of magnitude too slow |

**THE COVERAGE LAW (the most important durable finding).** Enrichment coverage
helps only from a **low** base; near the ceiling it is negative.
* Raising coverage 0.915 → 0.951 **lowered** score (−0.0132); on the subset where
  coverage genuinely improved the delta was −0.0023, and where coverage was
  already complete, forcing reallocation cost −0.0193.
* Raising coverage 0.36 → 0.998 from a *truncated* base **raised** score +0.0928.
Interpretation: fix coverage when it is broken; **never trade a high-cosine tag
for a line-specific one when coverage is already ~0.9.** Three independent
instruments agree.

**Statistical resolution.** With ~4 distinct conversations the minimum resolvable
effect is ~0.049 unpaired; per-task cohort SE ≈ 0.016. Effects below that cannot
be claimed. Use paired designs and ≥25 scored tasks per arm.

---

## 4. WHAT I ACTUALLY WANT YOU TO DO

### Step 1 — Identify the real top performers (numeric only)
From W&B netuid-33 rows: rank miners by mean `final_miner_score` with n ≥ 30.
Report the top cohort, subnet median/mean, p90/p95, and **which miners are
consistently strong across many tasks** (not one-task flukes). Include their
score-distribution *shape*: mean, median, sd, p10, worst-case, zero rate.
Known reference: top-10 sit ~0.66–0.70 with **p10 ≈ 0.55–0.60** — their edge is
that their worst tasks are good, not that their best are better.

### Step 2 — Quantify our gap, and decompose it honestly
For every task where we and a top miner are in the same 6-miner cohort, compute
the head-to-head margin. Then decompose the ladder gap into:
* **task-difficulty draw** (cohort mean excluding self — measures luck), vs
* **outperformance** (our score minus our own cohort mean — measures skill), vs
* **tail** (rate of sub-0.5 tasks).
Prior finding to verify or refute: on ~57 tasks our outperformance excluding our
own sub-0.5 tasks was **+0.010** (top-10 grade), while our sub-0.5 rate was
**18% vs their 3%** — i.e. the entire deficit was the tail, not typical quality.

### Step 3 — Feature discovery on OUR 813 answers
This is the part the data genuinely supports: correlate measurable features of
**our own** answers with **our own** scores, per task type. Candidate features
(measure, don't assume): tag count; enrichment-anchored vs window-only vs
unanchored fraction; per-line coverage and its distribution; head-noun
concentration; inflection/near-duplicate pairs; dictionary-word fraction;
mean/median cosine to the enrichment centroid; tag length; generic-vs-specific
vocabulary; overlap with the replica's predicted GT.
Report Pearson **and** Spearman with n, and **partial** correlations — several
features are collinear (coverage and anchoring especially). Beware confounds:
task difficulty (use cohort mean as a control), mode, validator, time-of-day.

### Step 4 — Per-type analysis
Separately for conversation / webpage / NER / skill / survey: our mean vs top-10
mean, gap, share of traffic, and **gap × share** (the only ranking that matters
for overall improvement). Note skill is the one type never properly diagnosed.

### Step 5 — Hypotheses → offline experiments
Derive hypotheses **from Step 3's measured correlations**, not from the list of
things already rejected. Split the 813 tasks into development / validation /
**holdout** and never tune on the holdout. For each arm report: mean, median,
win rate, tasks improved, tasks degraded, **worst regression**, per-type effect,
enrichment coverage, window-only counts, latency. A mean gain with several
collapsed tasks is a failure, not a win.

### Step 6 — Deliverables (before any production change)
1. Who the actual top performers are (hotkeys, n, mean, distribution shape).
2. Our gap to them, decomposed into difficulty / skill / tail.
3. Which task types create the largest **gap × share**.
4. Top 5 **data-backed** weaknesses, each with the measurement behind it.
5. What top miners plausibly do differently — **inferred only from numeric
   signatures**, explicitly labelled as inference, never as observed content.
6. Whether our own four UID eras show any config that outperformed the current
   one (this is the correct use of the "3 deregistered miners").
7. The strongest hypotheses, ranked by expected value.
8. Offline experiment results with the full metric set above.
9. Holdout results.
10. The single highest-confidence change to implement, with its expected effect
    size and the measurement that would confirm or refute it live.

---

## 5. RULES

* **Do not touch production.** No deploys, no config changes, no restarts. The
  live miner is mid-immunity on UID 24 and a restart risks a task.
* **No claim without a number.** Write "improved X from A to B on N tasks" or
  "this hypothesis was not supported", never "this should improve the score".
* **Report contradicting evidence explicitly**, including anything that refutes
  my framing above.
* Prefer the smallest change that tests a hypothesis; keep the current system as
  the baseline (git branch `sn33-baseline-20260811`, commit `98ce191`; current
  live code `00d078d`).
* If something cannot be established from the available data — especially
  anything about other miners' tag content — **say so plainly** instead of
  inferring it.
