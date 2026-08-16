# DEEP ENRICHMENT — FINAL VERDICT (2026-08-13)

Frozen-judge paired A/B, deep-ON vs deep-OFF, on the full replayed corpus:
109 independent conversation tasks + 52 webpage tasks (172-task judge map),
real scoring formula (`sn33.scoring`, penalties ON), same-draw AND cross-draw
(odd/even judge-half) protocol. All numbers below survived independent
adversarial verification (5/5 task re-derivations to 1e-9, exclusion audit of
all 12,540 deep-only/shared texts with 0 misclassifications, protocol-identity
check against idea1). Verifier corrections are applied inline and flagged.

---

## 1. The corrected premise

The CLAUDE.md line "deep enrichment: implemented, disabled, NOT shippable" is
**stale**. The six defects were fixed 2026-08-08, tests pass, and
`SN33_DEEP_ENRICHMENT=1` is **LIVE** on the miner. The open question was never
whether it runs — it is what it is *worth*: the +0.024 prior was probed once,
never re-verified (the run died when the OpenAI account ran dry). This A/B is
that verification, on real captured mainnet tasks per the mainnet-only testing
rule.

Idea-1's oracle fact that motivated the question: deep_line contributed 143/563
oracle-selected tags (25%), near-parity with replica_line (171). That fact is
real — but as shown below, oracle *share* is not validated *score*.

## 2. What deep enrichment is measured to be worth

### Conversation compose A/B (n=109 independent tasks)

| metric | same-draw | cross-draw (honest) |
|---|---|---|
| deep-ON mean | 0.5566 | 0.5330 |
| deep-OFF mean | 0.5503 | 0.5289 |
| **delta** | **+0.0064** [CI95 +0.0036, +0.0092] | **+0.0041** [CI95 +0.0011, +0.0071] |
| median | +0.0073 | +0.0047 |
| p10 | — | -0.0115 |
| improved / degraded | 78 / 31 | 75 / 34 |
| worst regression | — | -0.0442 (idx 64); best +0.0541 (idx 73) |

* Not translation-driven: translated tasks (n=29) +0.0036 vs non-translated
  (n=80) +0.0043.
* Robust to arm construction: the independently re-selected liveconfig arm
  (no stored-rank reuse) reproduces +0.0044 cross [CI +0.0015, +0.0072],
  W/L 77/32; the B2 variant/translation-residue sensitivity moved the delta
  +0.0001 with identical W/L. The verifier's decisive bias test passed.
* **Verifier correction (c):** this delta is nonzero by its own paired CI at
  n=109, but it does **NOT** clear the protocol's 0.017 paired resolution
  floor. Quote it as "CI excludes zero at n=109", never as floor-cleared.
* Verifier note (b): arm B reused the deep-inflated rank cap (~25 non-deep
  tail candidates/task absent); measured immaterial — compose selects from the
  top ~30% of the list, and the independent-arm reproduction above closes it.

### Webpage compose A/B (n=52; cross-draw n=49)

| metric | same-draw | cross-draw (honest) |
|---|---|---|
| deep-ON mean | 0.5698 | — |
| deep-OFF mean | 0.5685 | — |
| **delta** | +0.0013 (median +0.0047) | **+0.0021** (median +0.0034) |
| improved / degraded | 35 / 16 | 29 / 13 |
| worst regression | -0.1150 | -0.0305 |
| p10 | — | -0.0145 (sorted-index convention; np.percentile -0.0109) |

Deep touches **every** webpage answer (6.6 of 20 shipped tags, mean 73.9
deep-only candidates/task) and moves the score by nothing: both deltas are far
inside noise. **Score-NEUTRAL on webpage.** Verifier caveat (d): on the 20
translated tasks, translation residue leaks deep-derived strings into the
deep-OFF arm; the bias direction is toward zero, so the neutral verdict cannot
flip negative — the tiny positive could be marginally understated. One 402 at
idx 252 (translation call only; row otherwise clean, disclosed).

### Candidate-ceiling change (k>=15 constrained oracle, cross-draw)

**Deep does NOT expand the honest reachable ceiling — it slightly shrinks it.**

* Conversation, **corrected to the deduped n=107 set** (verifier correction (a):
  the original n=118 kept 11 duplicate-conversation rows the A/B dropped):
  cross-draw delta **-0.0044** [CI -0.0084, -0.0005], W/L 39/58.
  (Original -0.0052 [-0.0089, -0.0015] at n=118 — verdict unchanged, honest n
  is 107.) Same-draw was +0.0107 with 101/0 improved — pure draw-overfitting:
  extra candidates chase the evaluated half and generalize slightly worse.
* Webpage: same signature — same-draw +0.0128 (43/0), cross-draw **-0.0080**
  (15 improved / 28 degraded of 49, worst -0.0545).
* Verifier note (e): the odd/even halves are subsamples of ONE frozen draw,
  which biases oracle cross-deltas TOWARD deep-ON — so these negative ceiling
  findings are conservative and hold a fortiori. (For the compose A/Bs the
  bias is harmless: the arms never see the judge.)

This reconciles the 25%-oracle-share fact with the never-verified +0.024:
**deep tags get picked, but they mostly substitute for equally good non-deep
tags.** Value is concentrated, not average: 9/118 conversation tasks (~8%)
lift >=0.02 under cross-draw — proper-noun/geo-anchored tags ('immigration new
zealand', 'sgx', 'north carolina housing finance agency', 'austin
multifamily') and dense same-topic noun-phrase families the base pool lacks.
Only 3/49 webpage tasks show a cross-validated lift >=0.02.

### Does deep supply NEW high-value candidates? Yes — structurally additive

Per conversation task (n=109): 66.1 deep-only candidates reach ranking; 13.45
(20%) sit at or above the task's deep-OFF answer-mean cosine to the independent
judge-half centroid, and 13.3 of those are NOT near-dups of any non-deep
candidate; 7.1 are actually selected (35% of the 20 answer slots). Near-dup
rate of deep-only vs non-deep: **0.6%** (46/7,201) — deep almost never
re-finds what the pool already had. This corroborates the generation-gap
finding that deep is the **largest unique-coverage contributor** (19.6% of the
2,689 covered GT tags lose coverage if deep is removed, vs pool 15.5%, replica
14.4%). Deep is new central mass — it just is not *better* mass than what the
composer would otherwise pick.

### Latency / fallback risk (observed facts only)

Since the combine-overlap fix (deployed 2026-08-10 19:12 UTC), the combine
launches concurrently with the deep grace and never reads deep tags
(`test_deep_tags_never_reach_the_replica_combine_input` pins this), so deep is
off the critical path by construction; post-fix live truncation rate was 0/77
in the same-day slow-API window vs 8/32 pre-fix, elapsed p90 11.18s -> 9.74s.
The replay itself made zero chat calls and zero fresh embedding calls on the
conversation arm (16,756 cache hits); webpage spent $0.071 (12 OpenRouter
calls). No new latency claims are made beyond these logged numbers.

## 3. DECISION (per the stated gate)

**The gate is not met: there is no floor-cleared validated gain.**
Conversation +0.0041 cross-draw is below the 0.017 paired floor; webpage
+0.0021 is noise; the honest candidate ceiling moves -0.004 to -0.008.

**Disposition: KEEP DEEP ENRICHMENT ON, but close the investigation.**
Rationale: the conversation effect is a real small positive (CI excludes zero,
reproduced under three arm constructions), webpage is measured harmless, the
concentrated wins (~8% of tasks, entity/topic-family vocabulary) are worth
having, and since the overlap fix the cost is off-critical-path latency, not
score. But deep enrichment is **not a lever** — realistic value is
~+0.004-0.006, roughly 6x smaller than the +0.024 prior, and it cannot close
the 0.12 per-tag field gap. Do not spend further tuning effort on it.

**The next investment is NOT more candidate generation.** The generation-gap
audit (172 tasks, 2,803 GT tags) found **96.1% of GT tags already have a pool
candidate at cosine >=0.60**; 103/172 tasks miss zero; misses average
0.64/task. Perfect miss-elimination bounds at ~+0.005 overall — tail
robustness, not the field gap. The residual miss taxonomy (n=110) and the two
cheap, targeted generators worth validating:

1. **H1 — geo-ladder** (34/110 misses, 31%): deterministic gazetteer
   city/state -> country -> region injection ('austin texas' -> 'united
   states'; the single string 'united states' was missed 17 times). 0 LLM
   calls. Validate: paired replay on the same 172 tasks, same+cross-draw,
   0.017 floor, report miss-rate delta and injected-tag pick rate.
2. **H2 — spaCy-NER window pass** (~30/110, 27%): lift PERSON/ORG/GPE/PRODUCT
   spans verbatim as a 'local'-family source ('victor marks', 'greg lopez',
   'anthropic claude'). 0 LLM, ~50ms CPU. Accept only if selected-answer
   delta >= 0 (injected NEs are high-variance).

Remaining classes: bare-head demodification 14, numeric artifacts 5, other 27.
Worst miss classes by rate at 0.60: number-bearing 10.0%, GT-invented
(anchored-neither) 11.0%, domain_energy 12.8% — vs abstract 0.8%.

## 4. What cannot be concluded

* **Nothing about the live W&B score.** All numbers are frozen-judge replay;
  the judge is one independent GT draw, and cross-draw halves are subsamples
  of that one draw, not two independent validator draws. Live effect could be
  somewhat smaller or larger; only a mainnet A/B resolves below ~0.017.
* **No per-task-type deploy/rollback recommendation** beyond "keep ON": a
  webpage-only deep-OFF switch is not supported (neutral is not negative), and
  a conversation-only claim of +0.004 live is below the floor.
* **The concentrated-win tasks cannot be identified prospectively** from this
  data — no tested predictor separates the ~8% of tasks deep lifts >=0.02
  from the rest.
* **The +0.024 prior is neither confirmed nor explained** — it was a 4/4
  small-n probe on the testnet corpus (resolution 0.049); this measurement
  supersedes it.
* **Latency incidence of deep under API slowdowns** is not re-measured here;
  the 0/77 post-overlap truncation figure is the live evidence on record.

Data: `data/ideas/deep_ab_conv.json`, `data/ideas/deep_ab_webpage.json`,
`data/ideas/deep_ceiling.json`, `data/ideas/generation_gap.{md,json}`,
`data/replay_telemetry.jsonl`, `data/replay_webpage.jsonl`,
`data/frozen_judge.json`. Scripts in the session scratchpad
(`replay.py`, `deep_ceiling.py`, `replay_wp.py`, `deep_ab_wp.py`).
