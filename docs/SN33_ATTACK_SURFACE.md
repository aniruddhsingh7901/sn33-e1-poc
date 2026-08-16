# SN33 attack surface — what a miner *could* do, and what we do

Written so we recognise these when we see them in the score distribution, and so
we know which lines we are not crossing. Every item is verified against the code
at `d4129ed` unless marked otherwise.

The brief's rule stands: optimising against the published reward function is the
intended activity; defeating the mechanism is not. Items marked **REPORT** are
genuine weaknesses that belong with the ReadyAI team rather than in a miner.

---

## A. Legitimate optimisation (what this miner does)

These use only the inputs the validator deliberately hands over.

| # | Technique | Why it is fair |
|---|---|---|
| A1 | **Replicate ground truth from the supplied document.** For webpage / NER / skill, `input.data.lines` *is* the `[:1000]` text ground truth was built from. Re-run the validator's prompts on it. | The validator chose to send it. Nothing withheld is recovered. |
| A2 | **Reproduce enrichment tags exactly.** The same enrichment lines go to ground truth and miner. Running the upstream `enrichment_to_metadata` prompt on them reproduces that slice of the target. | Explicitly encouraged by the team: *"Enrichment data is used by the validators to generate the ground truth. So you should use it too."* (papercirer, 2026-06-19) |
| A3 | **Rank candidates in the validator's embedding space** (`text-embedding-3-small`, 1536 dims, hardcoded `llm_openai.py:15`). | Public code, public model. |
| A4 | **Shape the answer to the formula**: ≥3 unique tags, 1–2 exact matches, ≤19 tags, normalized strings. | Published scoring maths. |
| A5 | **Vocabulary prior** from ReadyAI's public HuggingFace corpus. | Published dataset, released by the team. |
| A6 | **Cache identical tasks.** Webpage/NER/skill bundles emit 5 *identical* tasks (`to_mining_tasks`), so the same document can arrive repeatedly. | Answering the same question the same way is not an attack. |

---

## B. Grey zone — allowed by the rules, distorts the subnet

| # | Technique | Status |
|---|---|---|
| B1 | **Multi-key farming.** On-chain analysis (Bounty_debugger, 2026-07-25): 256 UIDs held by **30 coldkeys**; top 5 coldkeys = 56.8% of incentive; top-50 UIDs owned by 10 coldkeys. | **Explicitly permitted.** papercirer, 2026-07-30: *"running multiple hotkeys or coldkeys is not against the rules on SN33. It never has been… every key is scored independently… no bonus for volume."* Since weights are rank-based and nearly flat, N keys ≈ N× income — this, not tag quality, is what the leaderboard actually measures. |
| B2 | **Registration timing/bots.** Demand-priced registration since ~2026-07-22; operators openly batch-buy keys at price thresholds. | Permitted, purely economic. |
| B3 | **Tie shuffling.** `get_raw_weights` randomly shuffles tied scores before ranking (`ValidatorLib.py:214-221`). Identical EMA across many keys → free rank lottery each epoch. | Not exploitable in any directed way; noise, not leverage. |

---

## C. Real weaknesses — **REPORT, do not use**

| # | Weakness | Evidence | Impact |
|---|---|---|---|
| C1 | **Ground truth is POSTed to the write API before miners are dispatched.** `vl.put_task(...)` uploads `task_bundle.input.metadata` (which contains GT tags *and* vectors) to `db.conversations.xyz` *before* the masked tasks go out. | Acknowledged by papercirer 2026-07-30: *"There's no read exposure there, but… We'll move it after miner dispatch."* Fix promised, **not yet shipped** as of the current HEAD. | If any read path to that store exists or appears, a miner could fetch the exact target before answering and score ~1.0. This is the single highest-severity item. |
| C2 | **`validate_tag_set` crashes the scoring path on provider failure.** `LlmLib.validate_tag_set:204` does `len(response_content)` where `basic_prompt` returns `None` on any API error → `TypeError` inside the validator's `format_results`. | `llm_openai.py:38` returns `None`; `LlmLib.py:204` calls `len()` on it. | A validator's own OpenAI hiccup can except mid-scoring rather than degrading. Miners get collateral zeros. Explains some of the unexplained score drops discussed in Discord. |
| C3 | **`random.sample` cull is a coin flip on tag identity.** ≥20 cleaned tags → a random 20 kept (`:198`). Which 20 is luck, and `_calc_scores` then iterates `list(set(tags))`, so *which* 21 get scored depends on Python set ordering. | `LlmLib.py:198-200`, `GroundTruthTagSimilarityScoringMechanism.py:192-198`. | Adds pure variance to honest miners' scores. We avoid it by capping at 19 — but it means the subnet penalises verbosity randomly rather than deliberately. |
| C4 | **The `element in tags` membership filter deletes non-normalized tags silently.** | `LlmLib.py:213`. Measured on our own logs: 38.5% of survey tags destroyed, 5/53 responses zeroed. | Systematically penalises accented content — a correctness bug for a subnet that ships Spanish survey work, not just a miner footgun. |
| C6 | **Non-English tasks are unscoreable in their own language.** `validate_tags.j2` asks the model to keep only *"good English keywords"* and discard the rest, and `validate_tag_set` keeps only that list. All 53 captured survey tasks are a **Spanish** banking survey whose ground truth is the Spanish `selected_choices`. | Measured against the real screen with gpt-5.2: **0 of 6 Spanish tags survived; 6 of 6 English equivalents survived.** End-to-end bench over 30 captured survey responses: every Spanish-answering strategy scored **0.0000**, `all_tags_dropped` on 30/30. | The validator asks for tags on Spanish text and then deletes any Spanish answer. Every honest miner answering in the survey's own language scores zero; the only way to score is to answer a Spanish survey in English and rely on cross-lingual embedding similarity. That is a scoring bug, not a strategy — it should be reported. We comply with it (see `MINER_SURVEY_POOL`) because the alternative is a guaranteed zero, but the subnet is currently mis-measuring this entire task type. |
| C5 | **Commitment publish has no retry.** Published once at startup with `wait_for_inclusion`; on chain rate-limit there is no retry until process restart, while the metagraph axon is blackholed (`192.0.2.1:1234`). | `base/miner.py:101-138`. | A miner can be silently unreachable — and therefore unpaid — with a healthy-looking process. Operationally critical for us too (see runbook). |

---

## D. Things that do **not** work (checked, so we don't waste time)

| # | Idea | Why it fails |
|---|---|---|
| D1 | Forging embeddings | Miner `vectors` are discarded; the validator re-embeds every tag itself in `format_results`. |
| D2 | Reassembling documents from windows across keys | `mask_task_for_miner` blanks `guid`, `bundle_guid`, `input.guid` and sets `window_idx = -1`. We tested reconstruction on 576 real captured windows: only 2 chainable overlaps. Dead end. |
| D3 | Copying another miner's commitment | The hotkey is inside the sealed box; `decrypt_endpoint` raises on mismatch. |
| D4 | Gaming the non-linear transform | `np.power(scores, 3.0)` is monotonic and `get_raw_weights` uses `argsort` order only, so it cannot change payout. Confirmed by computing the curve. |
| D5 | Returning huge tag lists to cover more ground | 20+ triggers the random cull; only 21 are ever scored; and every weak tag drags `mean`+`median` (35% of the score). Measured: 12 tags beat 19. |

---

## E. Defensive posture for our own miner

* **DDoS.** May 2026 saw sustained 50–100 GB/s UDP floods against high-scoring miners' axon IPs. Encrypted commitments (PR #122) now hide the real endpoint — which is exactly why the commitment must be published successfully at every restart. Keep the axon firewalled to validator IPs; do not expose it publicly.
* **Provider dependence.** Discord attributes several cluster-wide score drops to OpenAI outages. Our local fallback scores ~0.44 with no network at all, which converts an outage from a zero into a mediocre round.
* **Never return `cgp_output=[None]` on error.** The working-tree `neurons/miner.py` did this; because `[None]` is truthy the validator reads `has_output=True`, skips its retry, and scores zero. Leaving the field unset earns a same-synapse retry instead.
