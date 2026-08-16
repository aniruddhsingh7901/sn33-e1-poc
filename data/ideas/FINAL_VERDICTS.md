# FINAL VERDICTS — idea sweep synthesis, 2026-08-13

Scope: 4 ideas, each independently evidenced, confound-checked, and (where possible)
intervention-tested against the frozen fresh-GT judge or fresh gpt-5.2 GT draws using the
real scoring formula (sn33.scoring, parity-tested vs the validator class). One idea
(idea 3) received a full adversarial verification; the verifier UPHELD the verdict
(holds=true) with corrections that are incorporated below. No verdict was refuted, so no
downgrades were forced; idea 3's evidence strength is downgraded per the verifier.

Replay meta: 120/120 conversation tasks replayed faithfully (0 errors, 99.76% cache-served,
$0.369 fresh-call spend, no unresolved 402). Only 6 replayed rows are confirmed-healthy
(hist_source=ranked); Rule 3 segmentation applied throughout.

---

## IDEA 1 — Candidate-pool oracle / selection regret
**Question:** do we generate the right candidates and select badly, or are the right candidates missing from the pool?

- **Evidence:** n=120 replayed conversation tasks vs frozen judge. Replay faithful
  (reproduced mean 0.5650 vs historical 0.5675 on the same judge). Naive single-draw
  oracle regret +0.0791 mean (+0.0265 median); oracle >= reproduced on 120/120 (scorer
  sane). Sanity: judge-GT-verbatim scores 0.223 (formula punishes all-both answers);
  judge tags' own mean cosine to their centroid is 0.554 vs our answers at 0.565 —
  selection already sits at the geometry's natural level. Decomposition: 56% of oracle
  tags already in our selected 20; excluded ones dropped by rank against our own centroid
  (known +0.0036 perfect-centroid ceiling); only 15/563 lost to dedup/insurance/screen
  combined; oracle-tag sources are the same families we already ship (no missing source).
- **Confounds checked:** oracle overfit to the single temp-1.0 judge draw (odd/even
  cross-draw validation — eliminates the entire naive regret); tiny-k exploitation
  (k>=15 constrained oracle); cross-regret right tail explained by judge-draw
  disagreement (spearman cross_regret vs repro score -0.617; vs live cohort-relative
  -0.108, null); Rule-3 segmentation (ranked/pool/unknown reported separately — the 6
  healthy-ranked rows show the MOST negative cross-regret); instrument validity;
  hygiene applied to pool; replay fidelity.
- **Reproduced?** Yes.
- **Causal intervention identifiable?** No — decomposition names no fixable mechanism.
- **Offline experiment possible?** Yes (was run).
- **Measured delta:** cross-draw-validated regret: unconstrained oracle **-0.0410 mean /
  -0.0703 median** (n=118); realistic k>=15 oracle **+0.0093 mean / -0.0205 median**.
  The naive +0.0791 is an overfit artifact. Neither meets the pre-registered >=0.04 bar.
- **Improved / degraded:** 43 / 74 (k>=15 oracle under cross-draw validation).
- **Worst regression:** -0.154 (k>=15 cross-draw); unconstrained oracle worst -0.226 —
  the oracle itself regresses when scored on an independent GT half-draw.
- **Holdout result:** odd/even judge-split cross-validation (the decisive instrument):
  oracle sets from the two halves overlap only 0.227 Jaccard — no stable "right subset"
  exists to find.
- **Verdict: REJECT.** Selection/ranking/composer tuning from the current pool is worth
  ~0 (point estimate -0.04 to +0.01, majority of tasks degraded). Any future
  selection-side proposal must be priced against this number.

---

## IDEA 2 — Tail-risk detector + slot-swap response for flagged tasks

- **Evidence:** pre-registered n floor FAILED: of 120 replayed rows only 6 are
  confirmed-healthy with 2 y_low positives (demoted to descriptive). In the
  label-contaminated unknown-source cohort, hist_mean_cos separates (AUC 0.052,
  precision 0.67 / recall 0.67 / FPR 0.06 at thr<0.55) — but that signal is
  re-detection of truncation: corr(hist_mean_cos, pool-only-tag fraction) = -0.707, and
  the pipeline already observes truncation directly as Result.source != "ranked"
  (arms B+D already respond to it). On the 6 confirmed-healthy rows the feature
  separates nothing — direction inverted (the 2 healthy positives sit ABOVE two
  negatives on own-target cosine).
- **Confounds checked:** truncation confound (proven — it IS the signal); Rule-3
  cohort pooling (ranked/pool/unknown strictly separate; separation exists only in the
  contaminated cohort); feature-answer mismatch (label-linked features computed on
  historical tags in replay cos space); both-arms-identical scoring for the intervention.
- **Reproduced?** Yes.
- **Causal intervention identifiable?** No — healthy-tail failures are target-mismatch
  vs the validator's GT draw, invisible by construction from inference-time features
  (all measured against our own target).
- **Offline experiment possible?** Yes (was run).
- **Measured delta:** response test (drop 4 lowest-cos, add 4 highest-cos unused
  candidates, paired n=120 vs frozen judge): flagged n=61 **-0.0027**; flagged
  confirmed-healthy n=5: **-0.0052, improved 0/5**. Unflagged n=59: +0.0013.
- **Improved / degraded:** 23 / 38 (flagged tasks).
- **Worst regression:** -0.0715 (flagged task, slot swap); on the 5 flagged
  confirmed-healthy tasks the intervention degraded all 5 (worst -0.0090).
- **Holdout result:** not run (n floor failed; 2 positives in the primary cohort).
- **Verdict: REJECT.** Two independent kills: the detector adds nothing over
  Result.source (predictable tail) and cannot see the healthy tail (GT-draw noise);
  even a perfect flag has no working response — the only inference-time lever made
  flagged tasks worse, consistent with the already-rejected greedy composer and
  max-unique results. Tail management is already as solved as it can be (B + D).

---

## IDEA 3 — per_line_max_frac (answer concentration) predicts cohort-relative score

- **Evidence:** correlation reproduced exactly: r=+0.486 on 70 confirmed-ranked rows
  (feature re-derived bit-exactly 70/70). Survives every measured control singly and
  jointly (joint partial r=0.389; decisive coherence test: partial r(plmf,rel|coh)=0.430
  vs r(coh,rel|plmf)=0.117 — not a coherence confound; window-enrichment mismatch,
  dominant-line share, redundancy all controlled). 7 high-concentration losers / 0
  low-concentration winners in 126. **Verifier corrections (holds=true, evidence
  weakened):** the signal is validator-heterogeneous — concentrated in one validator
  (5GQyFzvt r=0.712 n=16 uid69; 0.466 n=32 ext) while two others are null everywhere
  (5Dkz 0.015/-0.056; 5FZe 0.081); heterogeneity Q=9.85 p=0.043. Honest pooled
  estimate: within-validator fixed-effect r=0.311 p=0.0007, random-effects p=0.033,
  cluster-robust slope p=0.048 (core) / 0.124 (combined n=126). The headline p=2e-05
  must not be quoted alone. Verifier's additive control strengthens the core claim:
  partial r(plmf,rel|anchored_frac)=0.287 p=0.017 with reverse 0.035 p=0.77.
- **Confounds checked:** cohort_mean, era/uid dummies, validator dummies (level shifts
  only — slope heterogeneity found by verifier), n_enrichment_lines, enrichment/window
  chars, n_tags, coherence, mismatch, redundancy, dominant-line share. NOT
  controllable: candidate-pool alignment with the true GT topic (unmeasured latent;
  R2 of measured properties on plmf only 0.279) — the likely common cause.
- **Reproduced?** Yes (bit-exact).
- **Causal intervention identifiable?** No. Both directions are already tested:
  up-concentration = greedy-composer family (+0.004 ns, loses 11/17; demote inert);
  de-concentration = ARM C/quota family (REJECTED, -0.0132). The interventional record
  is an order of magnitude smaller than the observational slope (+0.044 implied by IQR)
  and of the wrong sign — the signature of confounding by the latent.
- **Offline experiment possible?** Partially — the blended proxy structurally rewards
  centroid-collapse and cannot adjudicate; only fresh-frozen-GT paired recompose or a
  live canary could, and the verifier assigns that follow-up a low prior.
- **Measured delta:** none — observational audit; no intervention run.
  Observational slope rel=+0.174*plmf (se 0.038-0.040).
- **Improved / degraded:** 0 / 0 (no intervention).
- **Worst regression:** n/a. Worst observational counterexample: plmf 0.80 at rel
  -0.159 (Risk-boardgame enrichment vs talk-radio window, decoy lines, variant
  crowding). Relevant prior interventional worst: ARM C -0.103.
- **Holdout result:** uid12/91 extension (n=56, source-unconfirmed): ext-only r=0.288
  p=0.03, combined r=0.340 — replicates weakly, but the replication is substantially
  validator composition (5GQy again). Verifier's era x validator cells are the honest
  holdout readout.
- **Verdict: KEEP RESEARCHING** (verifier upheld; priority DOWNGRADED per verifier).
  Concentration is a symptom of candidate-pool/GT-topic alignment, not a lever. Any
  future test MUST stratify by validator. The only follow-up retaining value is the
  narrow tail angle from the counterexamples: cap inflectional-variant crowding on
  window-enrichment-mismatch tasks (all 7 losers show 6-12 near-duplicate inflections)
  — distinct from the tested variants_per_tag knob.

---

## IDEA 4 — dup_pairs / semantic reinforcement

- **Evidence:** definition reverse-engineered with 443/443 exact reproduction:
  dup_pairs counts ONLY singular/plural inflection pairs — 99/99 pairs in healthy
  answers are the miner's own protected number-inflection variants; 0 semantic
  pairs exist. It is a diagnostic of replica alignment (variants self-rank into the
  top-20 when predicted-GT is on target), not a cause. Bench-era strong-pair partial
  +0.431 [+0.24,+0.59] survives all measured controls including answer-level cosine.
- **Confounds checked:** cohort_mean, plmf, anchored_frac, era dummies,
  n_enrichment_lines, answer-level quality (partial RISES under it), mutual
  strong/weak, other near-dup subtypes, cluster bootstrap (47 clusters/70 rows),
  healthy-only, extension duration-filtered + uid-controlled. NOT controllable:
  era-config differences — exactly where the correlation dies.
- **Reproduced?** Yes (443/443 exact; prior r reproduced).
- **Causal intervention identifiable?** No — the named intervention was built,
  tested, and failed; dedup relaxation is moot (variants already protected).
- **Offline experiment possible?** Yes (was run, with fresh GT draws).
- **Measured delta:** interventional (decisive): **-0.0042** final, CI95
  [-0.0147, +0.0041], n=15 paired fresh-GT tasks — CI excludes gains >= +0.005.
  Mechanism: the added variant beat the dropped tag only 13/29 times; "weak" tags by
  enrichment-centroid frequently carry real GT mass.
- **Improved / degraded:** 8 / 7.
- **Worst regression:** -0.065 (dropped 'signal messenger'/'it hiring', both closer to
  the real GT draw than the added plural variants).
- **Holdout result:** uid12/91 era extension (n=50): partial **-0.294**
  [-0.556,+0.101] — sign flip, CI excludes the bench point estimate; dose-response
  inverts (dup=0 is the best cell). Impossible if the pairs caused score.
- **Verdict: REJECT.** Keep dup_pairs only as a free monitoring indicator of replica
  alignment. Do not tune variants_per_tag on this basis.

---

# THE PRIORITY ANSWER

**We already generate the right candidates and we already select them about as well as
the geometry allows. Selection regret, validated honestly, is ~0:** cross-draw
unconstrained oracle regret **-0.041 mean / -0.070 median** (perfect hindsight
re-selection is WORSE than our shipped answer on an independent GT draw), realistic
k>=15 oracle **+0.009 mean / -0.021 median**, W/L 43/74. The naive +0.079 regret is
pure overfit to a single GT draw (oracle sets from two halves of the same judge overlap
only 0.23 Jaccard). 56% of oracle tags are already in our answer; our answers score
0.565 while the judge's own tags average 0.554 cosine to their own centroid.

**Therefore: the right candidates are NOT missing from the pool in a way selection can
fix, and no selection/ranking/composer/allocation work should be funded.** Four
instruments now agree: oracle cross-draw (~0), ARM C (-0.0132), quota (30/50), greedy
composer (+0.004 ns). The remaining ~0.12 per-tag gap to top miners (W&B, every task
type) can only live in **generation** — producing candidates that are near-central
under an INDEPENDENT GT draw — and in our two weakest task types (survey 0.410, skill
0.490). The next engineering effort goes to the generation side and to per-type fixes,
not to anything downstream of the candidate pool.

---

# RANKED NEXT ACTIONS (max 3)

1. **Fix and A/B deep enrichment (generation-side, the only positive signal left).**
   - Experiment: repair the six known defects (spec = tests/sn33/test_deep_enrichment_review.py,
     xfail-marked), then paired offline A/B on the frozen judge corpus (n=266,
     conversation+webpage) with deep ON vs OFF, live config otherwise, reporting
     improved/degraded + worst regression. Corroborating prior: probed +0.024 (4/4
     conversations, replicated on fresh GT); idea 1's oracle drew 25% of its tags from
     deep lines, near parity with replica lines (weak, draw-confounded, but the only
     generation-side signal in this sweep).
   - Expected effect vs the 0.049 floor: +0.02-0.03 — below the unpaired floor but
     resolvable paired on n=266 (paired resolution ~0.017).
   - Kill criterion: paired delta CI95 includes 0, or mean < +0.005, or worst-regression
     tail exceeds 4:1 loss ratio vs gains; also kill if latency pushes source=pool rate
     above the 4.1% pre-deploy baseline in any live canary.

2. **Per-type generation fix for survey (0.410) and skill (0.490).**
   - Experiment: capture >=25 mainnet survey + skill tasks; replay with type-specific
     pool prompts (survey already answers in English via its own pool prompt — audit
     whether the translation guard and screen-safe floor are actually firing on the
     replayed survey answers); score against fresh frozen-GT draws, paired vs live config.
   - Expected effect: the type gaps are -0.233 and -0.170 vs top-10 — far above the
     floor; even recovering a third clears 0.049 within-type.
   - Kill criterion: paired within-type delta < +0.02 on n>=25, or any increase in the
     hard-zero rate.

3. **Variant-crowding cap on window-enrichment-mismatch tasks (idea 3's surviving tail
   angle — low priority per verifier).**
   - Experiment: gate: when win_enr_sim is low (mismatch), cap inflectional variants to
     <=2 per head concept at compose time; paired recompose on captured mainnet tasks
     scored against fresh frozen-GT draws, STRATIFIED BY VALIDATOR (mandatory per
     verifier); report per-validator W/L.
   - Expected effect: tail-only — the 7 counterexample losers span -0.05 to -0.16; a
     ~+0.005 mean at best, below the floor overall but testable as a paired tail
     intervention on the mismatch stratum.
   - Kill criterion: W/L majority degraded on the mismatch stratum, or any degradation
     on the aligned stratum, or the effect appears only in validator 5GQyFzvt.

---

# WHAT CANNOT BE CONCLUDED FROM THIS DATA

- **Anything about other miners' tags.** W&B publishes numbers only; the ~0.12 per-tag
  gap's mechanism (their generation process) is unobservable.
- **Healthy-tail prediction.** Only 6 confirmed-healthy scored rows exist in the replay
  (2 positives); no classifier claim of any kind is supportable, and the healthy tail is
  consistent with pure GT-draw noise from our side.
- **Whether idea 3's concentration-score slope is causal.** The latent (candidate-pool
  alignment with the true GT topic) is unmeasured and plausibly causes both; the
  interventional record contradicts the slope in both directions. Also unresolvable
  here: why the signal lives in one validator (5GQyFzvt) — real validator behavioral
  difference vs small-cell noise.
- **The live B (fallback_enrich) verdict.** ~2-3 fallback events per 10h window; the
  offline +0.0928 cannot be confirmed or refuted yet. Do not touch B/D on this sample.
- **Per-task judge claims outside the mapped subset.** The frozen-judge idx->task
  mapping was lost and reconstructed (two reconstructions agree 110/266; only 6/70
  ranked rows mappable) — judge-based magnitudes for specific historical tasks are
  unreliable outside the replayed 120.
- **Any effect below ~0.017 paired / ~0.049 unpaired** on these corpus sizes; and any
  magnitude derived from a nearest-timestamp task-score join (51% mis-join rate, proven).
- **uid12/91 extension rows' health status** (no pm2 mirror covers Aug 12-13); all
  extension-based numbers are duration-proxy-filtered, not source-confirmed.

---

*Workings: data/ideas/idea1_oracle_regret.json, idea1_supplement_k15.json,
idea2_tail_detector.json, idea2_workings_features.json, idea2_intervention_rows.json,
idea3_concentration.json, idea3_rows.json, idea3_counterexamples.md,
idea4_dup_pairs.json, idea4_intervention_pilot.json, idea4_counterexamples.md,
data/replay_telemetry.jsonl. Total sweep spend: ~$0.62 OpenRouter chat, <$0.01
embeddings; no unresolved 402.*
