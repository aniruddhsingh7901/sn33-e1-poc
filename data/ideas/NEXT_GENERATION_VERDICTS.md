# NEXT-GENERATION VERDICTS — 2026-08-13

Synthesis of the H1/H2 injection A/B (156 scored tasks: 107 conv + 49 webpage, cross-draw
protocol, verifier-confirmed exact reproduction), the survey screen-safety audit (46 mainnet
tasks, 296+ screen simulations), and the skill-generation deep-dive (41 unique docs, fresh
frozen judge). All work lived in scratchpad; zero production changes were made.

| Experiment | Cross-draw Δ | Median Δ | p10 Δ | Improved/Degraded | Worst Regression | Runtime | Verdict |
|---|---|---|---|---|---|---|---|
| H1 geo ladder | +0.00015 corpus (+0.00022 conv; web 0.0) | 0.0 | 0.0 | 2 / 0 (of 156) | 0.0 | 20.7 ms | REJECT — harm-free, ~55x below 0.017 floor |
| H2 window NER | +0.00022 corpus (+0.00031 conv; web 0.0) | 0.0 | 0.0 | 4 / 0 (of 156) | 0.0 | 116.1 ms | REJECT — harm-free, ~55-77x below floor |
| H1+H2 combined | +0.00022 corpus (+0.00031 conv; web 0.0) | 0.0 | 0.0 | 4 / 0 (of 156) | 0.0 | 136.8 ms | REJECT — not additive; same 4 tasks as H2 |
| Survey safety fix (prompt v2) | n/a (no offline GT) — structural zero-exposure 1/46 → 0/46; worst screen run 5 → 8 survivors | n/a | n/a | 13 / n-unk (comment-proximity −0.0235 cos, sign of score effect unknowable offline) | n/a | prompt-only | HOLD — ready canary candidate, gated on bt_log verification of the zeros |
| Skill generation | n/a (no offline GT for deploy) — our-vs-judge mean 0.6454; beats an independent GT draw 40/41 (+0.12) | n/a | n/a | n/a | n/a | n/a | NO ACTION — no generation gap; verb-phrase hypothesis REFUTED; one defect logged (pluralizer) |

All H1/H2/D numbers are verifier-reproduced to 1e-9 (arms, injections, anchor counts,
selections); regression audit scanned 156 rows x 3 arms: **zero negative deltas of any
magnitude**. Fresh spend this synthesis: $0. Cumulative session spend: ~$0.005 (H1H2 embeds)
+ <$0.50 (survey chat) + <$0.15 (skill judge); OpenRouter untouched.

---

## Answers to the mandate's questions

**1. Does H1 (geo ladder) deliver real score value, or only coverage?**
Only coverage. Geo-class GT misses fell 38/237 → 13/237 (overall miss-rate 4.73% → 3.72%),
but only 2/156 answers changed (idx 8 'middle east' +0.0126, idx 30 'united states' +0.0106).
Corpus-wide cross-draw delta +0.00015 — ~55x below the 0.017 paired resolvable floor. Only
30% (76/255) of injected geo parents even pass the production anchor rule; the rest are
demoted x0.90 and never win a slot. Coverage without selection is not score.

**2. Does H2 (window NER) deliver real score value, or only coverage?**
Only coverage, same shape. Window-NE-class misses fell 40/479 → 8/479 (overall 4.73% → 3.37%),
but 4/156 answers changed, selection rate 0.37% of 1,069 injected NEs, corpus-wide +0.00022.
The ranker (proven 161/161 parity with production) simply prefers incumbent pool candidates
that sit closer to the target centroid. The missed GT tags are misses of the whole pool
geometry, not of candidate supply.

**3. Is H1+H2 better than either alone?**
No. Combined coverage is best (miss-rate → 2.68%) but the changed-task set is identical to
H2's four tasks, and on idx 30 the NER tag 'sos' *displaces* H1's geo win (+0.0007 vs
B's +0.0106) — the combination is slightly worse than B on that task. Score delta identical
to C (+0.00022). The answer set is selection-saturated; injectors do not compose.

**4. Is the survey zero risk now effectively zero?**
Structurally yes, with an honest caveat. After prompt v2: 0/46 tasks can be zeroed by a
single correlated fragment-class deletion (before: 1/46 — the 2026-08-11 15:42 submission was
12/12 first-person fragments), minimum 7 noun-phrase/other tags per task, worst observed
screen run 8 survivors (was 5), min_tags(3) never approached in any of 296+ simulations.
CAVEAT (split verdict, verifier-endorsed): the cited mechanism for the two live zeros was
NOT reproduced — 13 direct screen runs per task never dropped below 8 survivors, and both
zero attributions are suspect (one CSV row carries no score; the other is a nearest-ts join,
a method documented as 51% type-blind-wrong). The class-level risk v2 removes is real and
measured (first-person fragments survive at 0.89 with correlated whole-class deletion);
whether it caused the live zeros is unproven.

**5. Did any fix worsen already-good tasks?**
No, per the verifier's regression audit: across all 156 tasks x 3 injection arms there is not
one negative cross-draw delta of ANY magnitude — worst regression exactly 0.0, 152-154/156
answers bit-identical to baseline, no baseline tag ever displaced at a net loss. Webpage:
0 changed tasks in 49, deltas exactly 0.0. The only measured cost anywhere is survey v2's
−0.0235 comment-proximity cosine (13/46 tasks improved on that metric) — an offline proxy
whose effect on the real answer-choice centroid is unknowable, not a demonstrated regression.

**6. What is the safest additive improvement available?**
Nothing clears the 0.017 floor. Ranked by safety-to-evidence ratio:
(a) **Pluralizer hygiene** (from the skill audit): 11 distinct dictionary-invalid
pluralizations shipped live ('analysises', 'real estates', 'splittings') via the -is
fall-through in `sn33/variants.py:_pluralize`. Requiring `tags.screen_safe` certification on
variants is a pure defect fix serving ~100% of traffic; risk is losing a variant slot, gain
is removing screen-deletion exposure invisible to every offline judge. Needs its designed
offline A/B before any deploy — DESIGN ONLY today.
(b) **Survey prompt v2**: eliminates the one structural zero-exposure at a small,
unquantifiable proximity cost; gated on bt_log verification (below).
(c) H1/H2 injection: strictly harm-free but buys +0.0002 — carrying dead code and 21-137 ms
for nothing fails the "negligible runtime for actual value" test. Not worth the diff.

**7. Is anything strong enough for a production canary?**
Not today. H1/H2/D fail the decision rule on "meaningful selected-answer performance"
(50x+ below floor) despite passing every safety criterion. Skill needs no fix (already beats
an independent GT draw 40/41; live calibration −0.015 cohort-relative at n=3, unresolvable).
Survey v2 is the only near-candidate but its trigger evidence is unverified: the recommended
gate — confirm from validator-side bt_log tag counts that the two live zeros were genuine
screen discards, not join artifacts — has not been run, offline score validation is
impossible (survey GT is the literal unseen answer choices), and survey yields ~4 scored
tasks/period so even a live verdict would be slow. Deploying a prompt change on an
unreproduced mechanism against a near-parity baseline violates the regression-risk rule.

---

## Decision-rule application

- H1 / H2 / H1+H2: intended-class miss reduction ✓, W/L ✓ (all-positive), worst regression ✓
  (0.0), runtime ✓ — but selected-answer gain +0.0002 is not "non-negative *generalizable*
  performance", it is noise 55-77x below resolution. **REJECT all three.** This corroborates,
  from the candidate-supply side, the established finding that selection is
  optimal-within-noise: the generation gap is not closable by deterministic injection into
  the existing ranking.
- Survey v2: structural fix sound, mechanism unproven, no offline score instrument. **HOLD**
  as a documented ready canary, gated on bt_log evidence. Not a deploy today.
- Skill: hypothesis refuted, no mismatch, ceiling not computable (no captured pools).
  **NO ACTION**; pluralizer hygiene experiment specced for a future offline A/B.
- Nothing here is a breakthrough and nothing reaches even the "small additive win"
  (+0.002-0.005) bar on a scored instrument.

NO DEPLOYABLE WINNER
