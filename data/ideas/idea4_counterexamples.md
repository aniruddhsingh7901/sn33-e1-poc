# Idea 4 counterexamples (ranked conversation rows, conv_bench)

dup_pairs = plural-inflection pairs (reverse-engineered exact). strong = pair max cosine-to-enrichment-centroid > answer median.

## dup_pairs>=2 but rel<=-0.05 (reinforcement did not save the answer)
### row 5  rel=-0.064 final=0.587 cohort=0.651 era=uid69 strong=2 weak=0 ans_mean_cos=0.399
pairs: [('housing markets', 'housing market', 0.405), ('australian housing markets', 'australian housing market', 0.577)]
tags: ['housing markets', 'housing affordability', 'housing prices', 'migration and housing', 'housing market', 'cost of living', 'regional housing markets', 'australian housing markets', 'australian housing market', 'property market', 'urban economics', 'housing price', 'housing supply', 'renting affordability', 'wages', 'housing affordability australia', 'real estate economics', 'employment', 'socioeconomic consequences', 'renting']

### row 9  rel=-0.100 final=0.586 cohort=0.685 era=uid69 strong=1 weak=1 ans_mean_cos=0.366
pairs: [('regulatory requirements', 'regulatory requirement', 0.346), ('developer tools', 'developer tool', 0.26)]
tags: ['ai compliance', 'model explainability', 'model interpretability', 'explainable ai', 'regulatory requirements', 'technical standards', 'ai accountability', 'developer tools', 'responsible ai', 'explainability techniques', 'regulatory requirement', 'compliance standards', 'regulatory standards', 'compliance requirements', 'legal compliance', 'developer tool', 'accountability', 'conformity assessments', 'ai product requirements', 'high risk ai systems']

### row 63  rel=-0.090 final=0.563 cohort=0.653 era=uid24 strong=2 weak=2 ans_mean_cos=0.331
pairs: [('babel', 'babels', 0.215), ('smart contracts', 'smart contract', 0.293), ('optimistic rollups', 'optimistic rollup', 0.616), ('zk rollups', 'zk rollup', 0.532)]
tags: ['typescript', 'babel', 'smart contracts', 'webpack', 'smart contract', 'optimistic rollups', 'definition', 'transaction finality', 'definitions net', 'next js', 'smart contract development', 'javascript', 'babels', 'off chain computation', 'optimistic rollup', 'zk rollups', 'ethereum layer 2', 'ethereum scaling', 'zk rollup', 'securing']

## dup_pairs==0 but rel>=+0.03 (won without reinforcement)
(none in the 70 ranked bench rows -- every dup==0 row sits at rel<+0.03; dup==0 mean rel=-0.072.
BUT in the uid12/91 era (Aug 12-13, n=50 healthy-proxy) dup==0 is the BEST cell, mean rel=-0.003 --
the bench-era dose-response does not replicate out of era.)

## What the counterexamples show
- Row 9 and 63: plural pairs of LOW-cosine tags (0.26-0.35) -- reinforcing weak/off-target concepts.
  Consistent with the subtype split: only strong pairs correlate (+0.43), weak pairs are noise (+0.05).
- Row 5: two strong pairs on a diffuse answer (ans_mean_cos 0.399) that still lost by -0.064 --
  reinforcement cannot rescue a scattered candidate pool.

## Interventional pilot verdict (the decisive test)
Forcing the duplication (add plural variant of top-2 strong tags, drop 2 weakest;
n=15 paired tasks, fresh gpt-5.2 GT draws, real formula):
delta -0.0042, CI95 [-0.0147, +0.0041], improved 8 / degraded 7, worst -0.065.
Mechanism probe: the added variant beat the dropped tag under the judge only 13/29 times --
the 'weakest' tags by enrichment-centroid (often geographic/entity tags: china, japan, bahrain,
shoplifting) frequently carry real GT mass. dup_pairs is a SYMPTOM of a well-aligned predicted-GT
(its variants rank into the top-20 on their own), not a lever. REJECT.
