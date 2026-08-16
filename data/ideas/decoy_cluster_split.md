# Decoy-cluster split: selection vs generation vs GT-faithful (2026-08-14)

Mandate: for the idea3 "decoy-enrichment" counterexamples, decide per task whether
the wrong-cluster concentration was (1) a SELECTION failure (correct-cluster
candidates existed in the replayed pool at high cosine to the frozen-judge GT
centroid but selection went elsewhere), (2) a GENERATION failure (correct-cluster
candidates never entered the pool), or (null hypothesis) GT-FAITHFUL (the judge GT
itself follows the decoy lines, so the concentration was correct and the loss has
another cause, e.g. GT-draw disagreement).

Constraints honored: embeddings only, cache-first (`sn33.llm.embed`; run used
2,072 cache hits, 9 embed calls totaling 6,246 tokens for enrichment-line/window
texts never embedded before, and **0 chat calls**). No fresh GT was generated.
Workings: scratchpad `decoy_split.py` / `decoy_split_results.json`.

---

## Step 1 — overlap of the counterexamples with the replayed 120: **ZERO**

Timestamp intersection (±2s), both directions of the idea3 criteria:

* replayed-120 corpus (`data/replay_telemetry.jsonl`): 2026-08-08 02:22 -> 2026-08-10 13:52 UTC
* frozen-judge corpus (`data/judge_task_map.json`, 266 tasks): 2026-08-08 02:22 -> 2026-08-10 14:05 UTC, eras uid120/33/69
* the 7 high-conc losers: 2026-08-10 20:47 -> 2026-08-13 06:30 UTC (uid69-late/24/12 eras)
* low-conc winners (plmf<=0.5 & rel>=+0.03): **none exist** in idea3_rows at all, so that direction is empty by construction.

**None of the 7 named counterexamples is in the replay corpus or the judge corpus.**
The mandated split is therefore NOT COMPUTABLE for every named task: no replayed
candidate pool exists (pool not reconstructed) and no frozen-judge GT draw exists
(generating one would require chat calls, out of scope for this run).

### The named 7 — disposition table

| # | ts (UTC) | uid | plmf | rel | verdict | text-level note (all that is knowable without a pool or GT) |
|---|---|---|---|---|---|---|
| core19 | 08-10 20:47 | 69 | 0.85 | -0.125 | NOT COMPUTABLE | Warhammer window vs 4x real-estate lines. Answer 100% real estate, 0 window tags. No stray minority-line tags. Pool not reconstructed; no judge GT. |
| core57 | 08-11 10:23 | 24 | 0.95 | -0.073 | NOT COMPUTABLE | JS-testing window vs k8s + cloud-networking lines. 19/20 tags on the cloud line, `kubernetes` the only line-0 tag, 0 window (Mocha/WebDriver/Appium) tags. |
| core61 | 08-11 11:58 | 24 | 0.80 | -0.159 | NOT COMPUTABLE | Talk-radio window vs 3x Risk + Columbus + dictionary decoy. Strays `cosi`, `columbus ohio` show the generator did reach the Columbus line; window and dictionary lines 0. |
| ext19 | 08-13 00:43 | 12 | 1.00 | -0.091 | NOT COMPUTABLE | Career window vs 2x eval-SDK + 2x CASE-equipment homonym. Stray `equipment performance` reaches the CASE line; window 0. |
| ext20 | 08-13 00:44 | 12 | 0.95 | -0.103 | NOT COMPUTABLE | Same task family as ext19, same signature (stray `equipment performance`, window 0). |
| ext33 | 08-13 04:53 | 12 | 0.80 | -0.144 | NOT COMPUTABLE | Iran/inflation window vs 2x franchising + Precious-film + bullion decoys. Pure franchising monoculture; decoy lines and window 0. |
| ext36 | 08-13 06:30 | 12 | 1.00 | -0.074 | NOT COMPUTABLE | Cold-War chat window vs 3x insurance + 2x "Public" homonym decoys. 20/20 insurance; counts [20,20,20,1,0]. |

Text-level generation hint (weak, stated as such): in 4 of 7 answers a stray tag
from a minority enrichment line survived to the final 20, so the generator was
not fully blind to minority lines; the **window** cluster got 0 tags in all 7.
Whether window/minority candidates existed in those pools at competitive cosine
is exactly the question a pool would answer, and the pool was not captured.

Null-check for the named 7: not measurable (no GT draw). Prior only, labeled as
prior: the validator GT is ~88% enrichment-derived with one vote per line, so on
e.g. core61 (3 of 5 lines are Risk) a board-game-heavy GT is plausible and part
of our concentration was likely GT-faithful; but each task's cohort mean
(0.502-0.644) beat us, so the actual draws rewarded something our monoculture
missed. This is inference, not measurement.

---

## Step 2 — supplement: the split run where it IS computable (in-corpus analogs)

Since the named 7 are outside the corpus, the machinery was run on the nearest
in-corpus analogs: replayed+scored tasks with the same lexical fingerprint
(idea3's own `per_line_max_frac >= 0.8` on the historical answer) and rel < 0.
Honesty first: **no strict analog exists** — inside the replayed 120 the worst
high-plmf loss is rel -0.043, nowhere near the named tasks' -0.074..-0.159. The
strict decoy-loser fingerprint simply does not occur in the judge-corpus era.
These 8 analogs + 1 high-conc winner (control) are evidence about the mechanism,
not about the named tasks.

Method per task: judge GT = `frozen_judge[map_idx]` (one frozen draw, n=9..21
tags); centroid = mean of GT tag embeddings; GT reward per cluster = lexical
anchoring of GT tags to each enrichment line vs window (+ line/window text
cosine to the centroid); our concentration = idea3 lexical counts on the
historical answer; SELECTION iff the pool holds >=3 candidates from the
GT-rewarded under-covered cluster with cos-to-judge-centroid >= our answer's
mean; GT-FAITHFUL iff we cover the GT-dominant line with >=30% of our answer and
starved-line + window GT share < 0.25. `draw_gap` = live_final minus our
answer's mean cos to the judge centroid — a large positive value means the
validator's own draw liked our answer far more than the judge draw does.

| map_idx | uid | ts (UTC) | plmf | rel | class | draw_gap | evidence (one line) |
|---|---|---|---|---|---|---|---|
| 9 | 120 | 08-08 03:27 | 1.00 | -0.032 | GT-FAITHFUL | -0.003 | GT counts [1,6,8,10] sit on our own covered lines (our [0,5,18,18]); starved+window GT share 0.12 |
| 15 | 120 | 08-08 04:40 | 0.89 | -0.010 | GT-FAITHFUL | -0.077 | GT dom = our dom line 0 (share 0.45); starved GT share 0.00 |
| 20 | 120 | 08-08 05:31 | 0.83 | -0.031 | SELECTION (weak) | +0.181 | window carries 3/9 GT tags, we gave it 0; pool had 4 window candidates >= our mean 0.413 (top 0.476) — recoverable margin <0.01 |
| 60 | 120 | 08-08 12:30 | 0.83 | -0.028 | SELECTION (judge-draw outlier) | +0.412 | judge draw is 69% window-anchored, enrichment-line cos all <=0.20 — but live 0.682 vs judge-mean 0.270: the validator's draw sided WITH our clusters; on the draw that actually scored us there was no cluster failure |
| 188 | 33 | 08-09 11:12 | 0.85 | -0.015 | GENERATION | +0.118 | window carries 7/20 GT tags; pool's best window candidate 0.569 < our mean 0.620 — nothing selectable would have helped |
| 192 | 33 | 08-09 12:20 | 0.95 | -0.014 | SELECTION (small) | +0.135 | window carries 6/14 GT tags, our [19,19,19] all-enrichment; pool had 12 window candidates >= our mean 0.521 (top 0.636) — swap gain ~0.01-0.02, same order as the loss |
| 253 | n/a | 08-10 11:10 | 0.80 | -0.043 | GT-FAITHFUL | -0.142 | closest analog to the named 7 (CSM-collection window vs 2 real-estate lines): GT [7,5] on the two lines we covered [11,16]; 10/21 judge GT tags anchor to NEITHER source (combine-invented abstractions) |
| 257 | 69 | 08-10 12:40 | 0.80 | -0.007 | GENERATION | -0.112 | window carries 7/20 GT tags; best pool window candidate 0.626 < our mean 0.637 (borderline); rel -0.007 is noise |
| 182 (control) | 33 | 08-09 10:14 | 0.95 | **+0.064** | GT-FAITHFUL | -0.022 | high concentration WON: GT share 0.60 on our dominant line, window GT share 0.00 |

Aggregate over the 8 analog losers: **3 GT-faithful, 3 selection, 2 generation** —
with three qualifiers that carry the actual meaning:

1. **Every non-faithful case's under-covered GT cluster is the WINDOW**, never a
   rival/decoy enrichment line. In the measurable corpus, "decoy-cluster failure"
   degenerates to the already-known window-vote under-coverage, and forcing
   coverage there is exactly the shape of rejected ARM C (-0.0132).
2. **All 3 SELECTION verdicts are confounded by GT-draw disagreement**: their
   draw_gaps are +0.135/+0.181/+0.412 (the validator's own draw scored the same
   answer far above the judge centroid), while every GT-faithful row sits at
   -0.14..0.00. The "wrong cluster" reading flips with the draw you grade
   against; map 60 is the extreme case where the judge says catastrophic
   misallocation and the live validator draw says 0.682 (cohort 0.709).
3. **Recoverable margins are tiny.** Taking the judge draw at face value, the
   selection-class swaps are worth ~0.01-0.02 on tasks that lost 0.007-0.043 —
   the same order as noise, and consistent with Idea 1's cross-draw-validated
   selection regret of ~0.

Control task 182 confirms the null hypothesis fires in the wild: plmf 0.95
concentration was the WINNING move (+0.064) because the GT concentrated too.
Concentration is not the failure; the failure is when GT weight lands elsewhere,
and where that happens in measurable data it is window-weight, draw-dependent,
and small.

---

## Decision implication

The distinction the user asked for cannot be measured on the tasks that motivated
it: all 7 named decoy counterexamples fall outside the judge/replay corpus
(overlap 0/7), so their selection-vs-generation split is not computable without
new data capture (replayed pools + a GT draw for uid24/uid12-era tasks). Where
the machinery IS computable (8 in-corpus analogs sharing the plmf>=0.8 loser
fingerprint, losses -0.007..-0.043), the decoy subset **confirms, not
contradicts, Idea 1's verdict**: 3/8 are GT-faithful (the null hypothesis is
real — including map 253, the closest analog to the named tasks), 2/8 are
generation failures (nothing selectable existed), and the 3/8 nominal selection
failures are all draw-confounded with recoverable margins of ~0.01-0.02. No
detectable, fixable decoy-SELECTION failure mode appears anywhere in the
measurable data; the under-covered cluster, where real, is the window vote, and
forced reallocation to it is the already-rejected ARM C.

**Not worth pursuing** as an inference-time detection/selection fix. The one
honest open path is data collection, not analysis: capture candidate pools (and
optionally a judge draw) for uid12-era tasks going forward, which would make a
future split of fresh decoy losers computable — with the stated prior, from
three independent instruments (Idea 1 cross-draw regret ~0, ARM C -0.0132,
per-line quota 30/50), that the answer will again be generation/GT-noise.

Statistical honesty: 8 analog tasks + 1 control, one frozen judge draw of
varying size (n=9..21), descriptive only. No significance claims.

---

## Summary

```json
{"overlap_n": 0, "selection_n": 0, "generation_n": 0, "gt_faithful_n": 0,
 "not_computable_n": 7,
 "analog_supplement": {"n": 8, "gt_faithful": 3, "selection_draw_confounded": 3,
                        "generation": 2, "control_high_conc_winner": "GT-FAITHFUL +0.064",
                        "max_analog_loss": -0.043},
 "implication": "The named decoy failures are outside the judge/replay corpus, so their split is not computable; on the 8 measurable analogs the decoy subset confirms Idea 1 (regret ~0): 3 GT-faithful, 2 generation, and 3 selection verdicts that are all judge-vs-validator draw-confounded with ~0.01-0.02 recoverable margins. The under-covered GT cluster in every non-faithful case is the window, not a rival enrichment line, i.e. the already-rejected ARM C shape.",
 "worth_pursuing": "no — no detectable, fixable decoy-selection failure mode exists in measurable data; only new pool+GT capture for uid12-era tasks could ever settle the named 7, with a strong prior of generation/GT-noise"}
```
