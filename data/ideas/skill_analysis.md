# Skill-generation analysis (Phase 3, 2026-08-14)

Fresh-GT judge over all captured skill tasks; analysis-only, no production changes.

## Data

* 54 raw captures in `all_tasks_dedup.json` (36 mainnet + 18 testnet) -> **41 unique
  skill documents** (36 mainnet + 5 testnet) after dedup by document hash. All docs are
  exactly 1000 chars, 1 window line, 0 errors, 0 empty answers.
* Our shipped answers: 20 tags on 28 tasks, 18 on 8, 12 on 5 (older config).
* Live W&B joins exist for only **3** mainnet skill tasks: our 0.667/0.631/0.609 vs
  cohort means 0.700/0.565/0.687 -> cohort-relative mean **-0.015 (n=3, unresolvable)**.

## How the validator builds skill GT (verified in code)

`SkillGenerationTaskBundle._generate_metadata` (task_bundle/SkillGenerationTaskBundle.py:151-177):

1. `skill_markdown[:1000]` — the SAME 1000-char text the miner receives verbatim
   (`self.input.data.lines` is rewritten to it, line 162).
2. ONE `skill_to_metadata.j2` call (max **15** lowercase tags: "topics, technologies,
   capabilities"; vendored upstream prompt is byte-identical to local).
3. `combine_metadata_tags` over that **single** set (combine_named_entities_prompt.j2).

No enrichment, no multi-set voting — the only task type where GT is a single draw over
text we hold in full. Highest-replicability type in the subnet.

## Judge (validator's own prompt chain, gpt-5.2, no temperature, salt `skilljudge`)

One independent GT draw per task (stage-1 + combine), real formula, penalties ON.
82 chat calls (~34k prompt / ~4k completion tokens, well under $0.10 fresh spend);
embeddings cache-first.

```
n=41   mean final 0.6454   median 0.6486   min 0.3728   max 0.7584
mainnet-only n=36: 0.6427
excluding 2 GT-collapse tasks (nGT<=2): 0.6579
penalties fired: 2/41 (both no_both_tags, both on GT-collapse tasks)
n_both per task: mean 5.2 (insurance works)
calibration: live scored mean 0.6355 (n=3) vs judged 0.6454 — consistent
```

## The ceiling result (the headline)

A second fully independent validator-distribution draw (salt `skilljudge2`), shipped
as-is, scores:

```
draw-as-shipped (combined)   0.5255
draw stage-1 (15 tags)       0.5789
OUR shipped answers          0.6454   beats the raw draw on 40/41 tasks, +0.12 mean
```

**There is no generation gap on skill tasks.** Our composed answer (rank-to-centroid +
variants + 20 tags + insurance) is far above what perfectly replicating the validator's
GT distribution would score. The historical 0.490 (2026-08-08 W&B) was the old 12-tag
config; the current config sits at/near the top-10 level (top-10 skill was 0.660).

## GT concept characterization vs our answers

| | GT (519 tags) | ours (764 tags) |
|---|---|---|
| mean words/tag | 1.96 | 2.06 |
| single-word | 27% | 14% |
| gerund/verb-ish | 19% | 23% |

* The "GT wants skill-verb phrases, we ship topic nouns" hypothesis is **REFUTED** —
  shapes are nearly identical; GT is topic/technology nouns, exactly what we ship.
* The only distributional difference: GT carries more single-word technology/tool tokens
  (`dnf`, `winget`, `stryker`, `squash`, `kubernetes`).

Miss taxonomy (206/519 = 40% of GT tags have best-our cosine < 0.60; misses are NOT
differentially doc-anchored — 48% verbatim-in-doc for both missed and covered GT tags):

```
generalization_not_in_doc  100   (conflict resolution, 3d graphics, claims history)
tool_tech_ne_in_doc         46   (squash, fixup, dnf, pacman, winget, crdt, tdd)
domain_term_in_doc          46   (section 179, cam reconciliation, scoped css)
abstract_concept            14   (idempotence, commutativity, causality)
```

Under the coverage law this per-GT coverage deficit is not established as a score loss:
scoring is centroid-cosine, our mean-all cosine is 0.606, our tag distribution already
beats a perfect replica draw, and inflection-duplicate slots (13% of shipped tags) score
identically to non-duplicates (0.607 vs 0.606) — they are cosine-neutral, not weak.

## GT noise floor

* GT combine sometimes collapses the 15 stage-1 tags to 1-2 tags: 2/41 judge draws
  (nGT 1 and 2 -> our 0.373 and 0.430, the two worst tasks, both with an unavoidable
  no_both penalty); 4/41 draws have nGT<=5. This is validator noise, not a miner defect,
  and puts high variance on any single live skill score.

## The one real defect found (screen risk, invisible to the judge)

Our shipped answers contain **11 distinct dictionary-invalid inflections** made by the
variant pluralizer (`portfolio analysises`, `contract analysises`, `nonresidential real
estates`, `commit splittings` [pre-guard], `replica synchronizations` ...). The
`-ing/-ness/...` guard in `sn33/variants.py:_pluralize` exists, but words ending in
`-is` fall through (`analysis` -> `analysises`). The validator's `validate_tags.j2`
screen deletes non-dictionary words (measured live: "commit splittings" dropped), so
each such tag is a wasted slot. The offline judge does NOT simulate the LLM screen, so
this cost is invisible in every number above.

## Limits

* **No candidate pools** exist for skill tasks (capture stores only final answers), so
  per-GT best-CANDIDATE cosine and the candidate ceiling are **not computable** — the
  answer-vs-GT analysis above is the honest substitute.
* n=3 live scores means cohort-relative standing on skill is unresolvable from live data.

## Verdict + designed experiment (DESIGN ONLY, not run)

`mismatch_found = false` for any concept-class mismatch. Skill is ~5% of task volume,
already above the replica ceiling, near cohort parity on the (tiny) live sample.
Low priority. The smallest worthwhile offline experiment, if picked up:

**Skill A/B, scratchpad-only, judge-validated with cross-draw folds:**
1. Regenerate candidate pools by running the pipeline mirror offline on the 41 docs
   (replay.py machinery, skill branch).
2. Arm A = current config. Arm B = two additive changes:
   (i) *inflection hygiene*: refuse pluralization of words ending in `-is`/`-us`,
   and certify every emitted variant with the existing `tags.screen_safe` 64k wordlist
   before it may occupy a slot (fall back to next-ranked candidate);
   (ii) *doc tech-NE injection*: extract single-word tool/technology tokens verbatim
   from the 1000-char doc into the candidate pool (ranked normally, no forced slots —
   respecting the coverage law).
3. Score both arms on the `skilljudge` draw with the odd/even cross-draw protocol;
   success requires paired delta >= +0.017 (floor), no task degraded worse than -0.02,
   AND screen-unsafe shipped tags -> ~0 (reported as a separate metric, since the judge
   cannot see screen deletions).
Note (i) generalizes beyond skill — the same pluralizer serves conversation/webpage — so
its hygiene value, if validated, applies to ~100% of traffic, not 5%.

Artifacts: scratchpad `skill_judge.py`, `skill_judge_results.json`, `skill_ceiling.json`,
`skill_miss_tax.json`, `skill_tasks_uniq.json`; this file + `skill_analysis.json`.
