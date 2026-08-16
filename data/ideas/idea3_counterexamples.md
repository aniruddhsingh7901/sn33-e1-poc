# Idea 3 counterexamples — concentration (per_line_max_frac) vs cohort-relative score

Criteria: ranked conversation tasks with plmf>=0.8 & rel<=-0.05 ("high-conc losers")
and plmf<=0.5 & rel>=+0.03 ("low-conc winners").

Result across 126 rows (70 confirmed-ranked + 56 uid12/91 source-UNCONFIRMED):
**7 high-conc losers, 0 low-conc winners.** The absence of low-conc winners is itself
consistent with the positive relation — spreading below plmf 0.5 never beat the cohort.

## Confirmed-ranked core (conv_bench, n=70)

### 1. idx19 — uid69, 2026-08-10 20:47, plmf 0.85, rel -0.125 (0.486 vs cohort 0.611)
- Window: Warhammer 40k tournament talk (Aeldari, Fire Prisms, GW terrain).
- Enrichment: 4 lines, ALL commercial real estate (CBRE office report, Statista logistics,
  inflation/rates, logistics market size). Lexical line-coherence only 0.06 but topically one cluster.
- Our 20 tags: 100% real estate. Zero window tags. Heavy variant crowding:
  "real estate investment volume/volumes", "uk real estate market/markets",
  "commercial real estate/estates" — ~12 tags share the "real estate" head.
- Diagnosis: total window/enrichment mismatch. GT combine gives the (Warhammer) window
  1 vote vs 4 enrichment votes, but that 1 vote still shifts the GT centroid away from
  pure real estate; cohort (all ~0.61, tight) balanced better. Variant crowding diluted
  mean/median terms.

### 2. idx57 — uid24, 2026-08-11 10:23, plmf 0.95, rel -0.073 (0.571 vs cohort 0.644)
- Window: JS testing tooling (Mocha, WebDriver.io, Appium, Gulp, categorization of build tools).
- Enrichment: 2 lines — Kubernetes/Container-Attached-Storage, cloud networking/SDN.
- Our 20 tags: 19 anchored to the cloud-networking line; 14 start with "cloud".
  Zero testing-tools tags.
- Diagnosis: mismatch again; the window's dev-tooling vocabulary (adjacent to the
  enrichment's cloud topic) plausibly appears in GT; we submitted a near-monoculture
  "cloud X" block instead.

### 3. idx61 — uid24, 2026-08-11 11:58, plmf 0.80, rel -0.159 (0.421 vs cohort 0.579) — WORST
- Window: talk-radio call-in (California wildfires, Biden), zero board-game content.
- Enrichment: 5 lines — 3x Risk board game, 1x Columbus OH tourism, 1x Merriam-Webster
  dictionary decoy ("DO definition").
- Our 20 tags: board-game monoculture with variant crowding (strategy game/games,
  board games/tabletop games/tabletop gaming, risk/risk board game) + stray 'cosi',
  'columbus ohio'. per_line_counts [10,11,16,2,0].
- Diagnosis: mismatch + decoy lines. Each enrichment line is one equal GT vote, so the
  Columbus and dictionary lines contribute GT tags we barely covered — but NOTE: the
  frozen-judge eval already proved that forcing slots to starved lines loses on average
  (ARM C -0.0132). This task is the symptom, not a license for the rejected fix.

## Extended set (uid12, source UNCONFIRMED — no pm2 mirror for Aug 12-13)

### 4. 2026-08-13 00:44, plmf 1.00, rel -0.091 (0.415 vs 0.517)
- Window: career/personal conversation ("career debt", what happens to my company).
- Enrichment: 2x LLM-evaluation SDKs (NeMo Evaluator, Azure AI Foundry) + 2x CASE
  farm/construction equipment (homonym decoys for "Case").
- Our tags: eval-SDK monoculture with singular/plural spam ("containerized
  evaluation/evaluations", "evaluation harness/harnesses", "agent evaluation sdk/sdks",
  even the typo-ish "agent evaluate sdk") + one stray 'equipment performance'.

### 5. 2026-08-13 04:53, plmf 0.95, rel -0.103 (0.419 vs 0.564)
- Window: Iran/inflation political talk.
- Enrichment: 2x franchising/licensing + decoys: Precious (film) Wikipedia and
  bullion dealers ("precious metals" homonym).
- Our tags: franchising monoculture (franchisee/franchisees, franchisor/franchisors,
  franchise agreement/agreements, operation/operations).

### 6. 2026-08-13 06:30, plmf 0.80, rel -0.144 (0.428 vs 0.502)
- Window: conspiratorial chat (Cold War, stargate, Saddam).
- Enrichment: 3x insurance (GEICO, Dunn, State Farm) + 2x "Public" homonym decoys
  (Public.com investing app, public-relations Wikipedia).
- Our tags: 20/20 contain "insurance". Total monoculture.

### 7. 2026-08-13 10:30 (ts 1786602626), plmf 1.00, rel -0.074
- Same signature (dominant-line monoculture on a mismatch task).

## What actually distinguishes the losers

One shared fingerprint, all 7:
1. **Window↔enrichment topic mismatch** (window contributes a distinct GT vote we ignore), AND
2. **decoy/homonym enrichment lines** (search-result lines about an unrelated homonym), AND
3. **variant crowding**: 6-12 near-duplicate inflections of one head noun filling slots.

BUT: window-enrichment mismatch does NOT explain the overall correlation —
partial r(plmf, rel | win_enr_sim) = 0.484 (unchanged from raw 0.486), interaction ns,
and the plmf-rel slope is POSITIVE within both the mismatch half (r=0.442) and the
aligned half (r=0.571). The mismatch+decoy pattern characterizes the extreme tail only.
Fixing that tail by forced de-concentration is the already-REJECTED ARM C (-0.0132,
29 improved / 177 degraded). The only untested angle in the tail is variant crowding
on mismatch tasks (cap inflectional variants when window and enrichment disagree) —
related to but not identical to the tested variants_per_tag knob.
