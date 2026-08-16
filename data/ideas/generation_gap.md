# Generation-gap analysis: which GT concepts does our candidate pool miss?

2026-08-14. Instrument: 172 replayed tasks (120 conversation from
`data/replay_telemetry.jsonl`, 52 webpage from `data/replay_webpage.jsonl`; no
duplicate conversations found — the idx 115-136 repeat range is absent from
these files, max conversation-fingerprint Jaccard 0.55 across all pairs).
GT = `data/frozen_judge.json` (2,803 GT tags). For every GT tag: max cosine to
any candidate in the FULL replayed pool (deep + theme included). All 17,734
embeddings were cache hits — 0 API calls, $0. Classification is deterministic
heuristics + spaCy NER over the task's window/enrichment text
(`all_tasks_dedup.json`, joined by float timestamp, 172/172). Machine-readable
companion: `data/ideas/generation_gap.json`.

## Headline: the pool covers 96%. Generation is NOT the main bottleneck.

| threshold | missed / 2,803 | miss rate |
|---|---|---|
| best cos < 0.55 | 81 | 2.9% |
| **best cos < 0.60** | **110** | **3.9%** |
| best cos < 0.65 | 151 | 5.4% |

Conversation 4.7% (95/2,021) vs webpage 1.9% (15/782). Mean misses/task 0.64;
**103 of 172 tasks miss zero GT tags**; only 10 tasks (9 conversation) miss >=3.
The remaining conversation gap lives in selection, not generation — consistent
with idea-2/idea-4. The misses that do exist, however, are highly patterned and
mostly fixable WITHOUT an LLM.

## 1. Miss rate per class (threshold 0.60)

| class | n | missed | rate |
|---|---|---|---|
| abstract head (tion/ment/ity/... or abstract lexicon) | 788 | 6 | **0.8%** |
| concrete | 2,015 | 104 | 5.2% |
| named-entity-bearing | 1,453 | 53 | 3.7% |
| non-NE | 1,350 | 57 | 4.2% |
| number-bearing | 90 | 9 | **10.0%** |
| 1-word tag | 666 | 44 | **6.6%** |
| 2-word | 1,298 | 45 | 3.5% |
| 3-word | 609 | 12 | 2.0% |
| 4+-word | 230 | 9 | 3.9% |
| anchored in window only | 931 | 41 | 4.4% |
| anchored in enrichment only | 901 | 11 | **1.2%** |
| anchored in neither text ("invented" by GT combine) | 328 | 36 | **11.0%** |
| partial anchoring | 629 | 22 | 3.5% |
| domain: energy 12.8%, politics 6.4%, finance 4.5%, real-estate 4.1%, tech 1.7% | | | |

Covered-vs-missed composition (110 missed vs 110 random covered controls):
missed tags are far LESS abstract (5.5% vs 29.1%), more often 1-word (40% vs
21%), more often un-anchored in the source text (33% "neither" vs 9%), and more
number-bearing (8% vs 4%). NE share is similar (48% vs 44%) — it is not
"entities" generically, it is SPECIFIC KINDS of entities (below).

Enrichment-derived concepts are nearly saturated (1.2% miss) — replica + deep
own that space. The gap is concrete, short, window-side or GT-invented.

## 2. The four recurring miss patterns (taxonomy of the 110)

| pattern | n | share |
|---|---|---|
| geo-generalization (country/region names) | 34 | 31% |
| specific named entities (people/places/orgs, mostly in-window) | 30 | 27% |
| bare-head de-modification (GT bare word, pool only modified compounds) | 14 | 13% |
| numeric artifacts ('000' etc.) | 5 | 5% |
| other (niche products, orgs, misc) | 27 | 25% |

**Pattern 1 — geo-generalization.** The GT emits the country/region; our pool
holds only finer-grained or sibling geography. `united states` alone is missed
**17 times** (worst single GT string in the corpus); also china x3, canada x3,
iran x3, india x2, middle east, united kingdom, south korea, japan.

**Pattern 2 — specific in-window entities.** 24 window-only + 10 partially
anchored NEs the extractors never lifted, though they sit verbatim in the
transcript.

**Pattern 3 — bare heads.** GT emits the unmodified entity/noun; every pool
variant carries a modifier, and the modifier drags cosine below 0.60.

15 concrete examples (GT -> best candidate in pool, cosine):

| # | pattern | GT tag | best candidate | cos |
|---|---|---|---|---|
| 1 | geo | united states | us housing market | 0.380 |
| 2 | geo | united states | austin texas | 0.403 |
| 3 | geo | china | united states | 0.431 |
| 4 | geo | middle east | iran conflict | 0.403 |
| 5 | NE | telangana | technology | 0.225 |
| 6 | NE | hyderabad | heloc | 0.332 |
| 7 | NE | victor marks | polymarkets | 0.328 |
| 8 | NE | greg lopez | tom tancredo | 0.351 |
| 9 | NE | anthropic claude | artificial intelligence ads | 0.290 |
| 10 | NE | mario delgado | salinas pliego | 0.388 |
| 11 | NE | ted kaczynski | herbert marcuse | 0.293 |
| 12 | bare-head | google | google tools | 0.558 |
| 13 | bare-head | youtube | youtube tutorial | 0.544 |
| 14 | bare-head | mma | mma fighters | 0.541 |
| 15 | numeric | 000 (GT split of "400,000") | unit count | 0.361 |

(Full list: `all_missed` + `examples_worst_misses` in the JSON.)

## 3. Source affinity: who covers what

Share of covered GT tags (>=0.60) whose BEST-matching candidate carries each
source (candidates can be multi-source):

| class | replica | deep | pool | local | theme | variant |
|---|---|---|---|---|---|---|
| all covered | .390 | .318 | .216 | .053 | .012 | .009 |
| abstract | .373 | .368 | .210 | .015 | .026 | .008 |
| concrete | .398 | .296 | .218 | .070 | .006 | .009 |
| NE-bearing | .391 | .294 | .214 | .084 | .009 | .006 |
| window-anchored | .365 | .187 | .290 | **.144** | .007 | .004 |
| enrichment-anchored | .410 | **.403** | .166 | .000 | .009 | .011 |
| GT-invented ("neither") | .401 | .357 | .193 | .007 | .027 | .009 |
| 1-word | .427 | .266 | .188 | .095 | .017 | .002 |
| number-bearing | .472 | .317 | .146 | .024 | — | .024 |

Removal test — coverage LOST if the source (and its candidates) vanished, of
2,689 covered GT tags:

| source | uniquely covered | share |
|---|---|---|
| **deep** | **527** | **19.6%** |
| pool | 416 | 15.5% |
| replica | 387 | 14.4% |
| local | 129 | 4.8% |
| theme | 8 | 0.3% |
| variant | 2 | 0.1% |
| anchor | 0 | 0.0% |

Deep enrichment is the single largest UNIQUE coverage contributor at the pool
level (strongest on enrichment-anchored and abstract classes), corroborating
idea-1's oracle share (143/563). Theme/anchor/variant contribute essentially no
unique concept coverage — they are re-rankers/duplicates, not generators.
`local` is the only source with window-NE affinity (14.4% of window-anchored
best-matches) — exactly the class that patterns 1-2 say is under-generated.

## 4. Honest score cost of misses (descriptive, small n)

Grouping tasks by full-GT miss count at 0.60 and scoring
`reproduced_tags_liveconfig` with the real formula:

| instrument | high-miss group | low-miss group | diff |
|---|---|---|---|
| same-draw judge final (miss>=3 n=10 vs miss<=1 n=151) | mean .4905 / med .4763 | mean .5703 / med .6007 | **-0.080** |
| cross-draw (misses defined on ODD-half GT, scored on EVEN-half; >=2 n=10 vs <=1 n=162) | mean .4619 / med .4237 | mean .5462 / med .5653 | **-0.084** |
| live cohort-relative (scored subset only) | n=3, mean -.019 | n=51, mean -.039 | +0.020 (n too small, no signal) |

Caveats stated plainly: n=10 in the high-miss group; the difference is
confounded with task difficulty (a hard, entity-dense task both evades our
generators and scores low for everyone — the live instrument, at n=3, shows the
high-miss tasks are NOT losing to their cohorts). Upper bound on total mean
gain from perfect miss elimination: ~0.084 x (10/172) ~= **+0.005 overall**,
and only if selection actually ships the new candidates. This is a
tail-robustness fix, not the main gap.

## 5. Ranked generator-improvement hypotheses

**H1 — deterministic geo-ladder (targets 34/110 = 31% of misses).**
Gazetteer of ~60 entries mapping cities/states/regions -> country -> region
("austin texas" -> "united states"; "tehran" -> "iran" -> "middle east"). When
any window/enrichment/candidate geography matches, inject the parent names as
candidates. Zero LLM calls, one extra embed batch, <5 lines of latency.
*Validation:* re-run this replay pool + injected tags on the same 172 tasks;
paired same-draw AND cross-draw selection score, n=172, resolution floor 0.017
paired; also report miss-rate delta and how often selection actually picks the
injected tag. Predicted: miss rate 3.9% -> ~2.7%; score delta small (+0.002 to
+0.005) but concentrated on entity-heavy tasks.

**H2 — spaCy-NER window pass as a first-class candidate source (targets ~30
misses = 27%).** Lift every PERSON/ORG/GPE/PRODUCT span from the window
verbatim (lowercased, screen-cleaned) into the pool. Deterministic, no LLM,
~50ms CPU. `local` already half-does this (it owns the window-NE affinity
column) — extend it rather than add a new stage. *Validation:* same paired
replay harness; guard metric = selected-answer delta (injected NEs are
high-variance: they either hit a GT entity or sit far from the centroid, so
selection must stay cosine-gated). Predicted: covers telangana/hyderabad/victor
marks class; net score neutral-to-positive; accept only if paired delta >= 0.

**H3 — bare-head splitting (targets 14/110 = 13%).** For every multi-word
candidate, also emit its head/entity token when it is a known entity or
dictionary noun ("google tools" -> "google", "mma fighters" -> "mma").
Deterministic string op. *Risk:* floods the pool with generic 1-word tags that
selection might over-pick (1-word tags are also the field's weakest cosine
class). *Validation:* paired replay A/B; ship only if selected-answer score is
non-negative AND the 1-word share of the shipped answer does not grow by >2
tags.

**H4 — do NOT invest in more conceptual/LLM generation.** Abstract-class miss
rate is 0.8% and enrichment-anchored is 1.2% — replica + deep already saturate
the conceptual space. Any further LLM generation spend buys candidates in the
one region that is already full. (Anti-hypothesis; costs nothing to obey.)

**H5 — numeric artifacts: skip.** 5 misses, all GT tokenization noise ('000'
from "400,000"). Emitting comma-split fragments to chase them would add junk
tags that cost more on mean/median than the occasional exact match earns. Noted
for completeness, ranked last deliberately.

Priority order: H1 (biggest class, zero risk, zero cost) -> H2 (second class,
needs the selection guard) -> H3 (small, risky) . H1+H2+H3 together address
~71% of all misses with zero additional LLM calls; expected combined ceiling
~+0.005 mean with fatter gains on the 10 entity-dense tail tasks — worth doing
as tail-robustness, not as the route to the 0.12 field gap, which remains a
selection/centralness problem, not a coverage one.
