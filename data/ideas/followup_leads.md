# Follow-up leads — preserved, NOT yet experiments

## 1. Decoy-enrichment / wrong-cluster failures (from Idea 3 counterexamples, 2026-08-14)

Observation: the worst high-concentration loss (per_line_max_frac 0.80, rel −0.159,
uid24 2026-08-11 11:58) had DECOY enrichment lines — Risk-boardgame enrichment
against a talk-radio window. The answer concentrated hard on the WRONG cluster.
Concentration itself is not the defect; concentrating on the wrong cluster is.

USER DIRECTIVE (2026-08-14): do NOT open a new experiment yet. When Idea 1
oracle-regret lands, split the decoy-failure tasks by:

1. **Selection failure** — candidates anchored to the correct cluster existed in
   the replayed pool (high cosine to the frozen-judge GT) but ranked/composed out
   → decoy-cluster detection at compose time is worth pursuing.
2. **Generation failure** — the correct-cluster candidates never entered the pool
   → detection cannot help; the fix would be generation-side (and may not exist,
   since the validator's own GT inherits the same decoy enrichment).

Method: for each Idea-3 counterexample task (idea3_counterexamples.md), join
replay_telemetry.jsonl candidates → cosine to the frozen-judge centroid; check
whether the oracle-selected set (idea1_oracle_regret.json workings) draws from a
different enrichment cluster than our shipped answer.

Note the null hypothesis: if the validator's GT itself follows the decoy lines
(GT is ~88% enrichment — including decoys), then "wrong cluster" may actually be
GT-faithful and the loss comes from elsewhere. Check the judge GT's own cluster
alignment first.

## 2. Concentration signal (Idea 3 verdict: KEEP_RESEARCHING)

Real, era-stable, survives all controls incl. line-coherence (keeps 87% of r).
No deployable lever: more concentration = tested-neutral greedy/centroid family;
less = rejected ARM-C/quota family. Revisit only if the decoy split (lead 1)
identifies a conditional intervention (concentrate EXCEPT when cluster-conflict
is detected).

## 3. Survey zeros on UID 91 (found 2026-08-14, NOT yet diagnosed)

Both era zeros are SURVEY tasks to validator 5GQyFzvtVMw9:
* 08-13 14:47:38 — answered 3.7s, 12 tags ("only bank i use", "i do everything there", ...)
* 08-13 17:02:28 — answered 5.7s, 12 tags ("security of savings", "safer place for money", ...)

We answered fast with full 12-tag sets, all dictionary words — so the zeros are
validator-side discards, most plausibly the LLM "good English keywords" screen
deleting phrase-shaped tags (sentence fragments like "i do everything there" are
not keywords) below min_tags(3) -> response discarded. A third survey task at
17:09 with noun-phrase-shaped tags ("small business loans") was presumably fine.
Survey is ~5% of traffic but a zero drags the short EMA hard during survival.
Candidate fix (UNTESTED): bias the survey pool prompt toward NOUN-PHRASE tags,
never first-person fragments. Needs: pull the scored survey tasks + check which
survived vs zeroed before touching the prompt.
