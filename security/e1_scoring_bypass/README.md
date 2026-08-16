# SN33 validator scoring-bypass (E1) — local reproduction

**Status:** responsibly disclosed. This is a **defensive, local** proof-of-concept.
It never connects to a network, registers a hotkey, or submits to any validator.
It builds the cheater payload in memory and runs it through this repo's **real**
scoring code so you can confirm the bug on your own machine and fix it.

## TL;DR

A miner can send 5 sub-3-character junk `tags` plus a hand-made `vectors` field.
That makes the validator's `format_results` raise **before** it recomputes the
embeddings, so the miner's forged `vectors` survive — and the scorer
(`GroundTruthTagSimilarityScoringMechanism`) reads them off the wire and grades
against them. The impact depends on **one** detail in your response loop:

| your `validator.py` response loop | impact of the payload |
|---|---|
| **no local `try/except` around `format_results`** (afterpartyai `main`, pre-2.37.74) | the raise propagates to the top-level handler → `evaluate()` never runs → **Denial-of-Service** (batch aborts, nobody scored). |
| **`try/except … continue`** (v2.37.74 / PR #135, commit `2741fa0e`, mainnet 2026-08-18) | the bad response is kept in the scored list → `evaluate()` grades the forged vectors → **score inflation ~0.85 (cap 0.90)** + honest miners zeroed (D1). |

**The catch-and-continue added in commit `2741fa0e` ("fixes and cleanup",
JordanBourgault, 2026-08-14) — intended to stop one bad miner from crashing the
batch — is exactly what converts the DoS into a score-inflation exploit**, because
it catches the error but does **not** remove the response from the list, and the
scorer trusts miner-supplied `vectors`.

## Verified facts (grep them yourself)

- `conversationgenome/scoring_mechanism/GroundTruthTagSimilarityScoringMechanism.py`
  `_calc_scores`: `tag_vector_dict = miner_result["vectors"]`;
  `tag_vectors = tag_vector_dict[tag]['vectors']` — reads miner vectors verbatim,
  no re-embed. Present on `main` **and** v2.37.74.
- `conversationgenome/task_bundle/ConversationTaggingTaskBundle.py` `format_results`:
  sets `original_tags`, then `tags = validate_tag_set(...)` (raises on empty clean
  list), then `vectors = _get_vector_embeddings_set(...)` (never reached on junk).
- `conversationgenome/utils/Utils.py` `get_clean_tag_set`: drops tags < 3 chars →
  empty list → the raise above.
- `neurons/validator.py` response loop: `main` calls `format_results` with **no**
  local catch; v2.37.74 wraps it in `try/except … continue` (commit `2741fa0e`).
- `conversationgenome/__init__.py`: `__version__ = "2.37.74"`.

## The cheater payload (`cheater_miner.py`)

```python
# HONEST miner returns:
[{"tags": [20 real tags], "vectors": None}]          # validator recomputes the vectors

# CHEATER miner returns:
[{"tags": ["ab","cd","ef","gh","ij"],                # junk (<3 chars) -> format_results raises
  "vectors": {"ab": {"vectors": FORGED}, ...}}]      # attacker-supplied; the scorer trusts it
# FORGED = mean of the attacker's own ~20 guess-tag embeddings (aimed at the GT centroid)
```

Why 5 junk tags: fewer than 3 would fail the scorer's `min_tags=3` gate. Why all
junk: one real tag makes the cleaned list non-empty, so `format_results` does not
raise, the validator re-embeds, and the forgery is discarded. So the `no_both`
×0.9 penalty is unavoidable → hard score cap **0.90**, realistically ~0.85.

## Run it

```bash
# from the repo root, with the project's venv/deps available:

# 1) transport proof: forged vectors survive the CgSynapse wire round-trip (no network)
PYTHONPATH="$PWD" python security/e1_scoring_bypass/synapse_roundtrip.py

# 2) scoring proof: forged vectors are graded by the REAL evaluate(), both variants
PYTHONPATH="security/e1_scoring_bypass:$PWD" python security/e1_scoring_bypass/reproduce.py

# realistic magnitudes (needs OPENAI_API_KEY; uses text-embedding-3-small):
PYTHONPATH="security/e1_scoring_bypass:$PWD" python security/e1_scoring_bypass/reproduce.py --real
```

`synapse_roundtrip.py` (wire) + `reproduce.py` (processing + scoring) prove the
**entire** chain locally — no chain, no axon, no live network, no third parties.
A shared-testnet miner is unnecessary and would answer other people's validators
(and DoS any still on v2.36.73), so it is not the clean test.

Expected (offline mock): `validate_tag_set(junk)` raises; VARIANT A → DoS (evaluate
never runs); VARIANT B → CHEATER 0.9000 vs HONEST ~0.26, un-aimed forgery ≈ 0,
D1 (`vectors=None`) → 0.0000. In `--real` mode the cheater lands ~0.85 vs an honest
~0.55. (The mid-run traceback in VARIANT B is the *real* validator code logging the
D1 `None` crash — it is expected and is itself the proof of D1.)

## The fix

Any one of the first two closes the score-inflation path; all three recommended.

1. **When `format_results` raises, remove the response from the list `evaluate()`
   scores** (build a `good_responses` list inside the loop; don't just `continue`
   the current one while leaving it in `responses`). This is the specific fix for
   commit `2741fa0e`: catch **and** drop.
2. **Never trust miner-supplied `vectors`.** In `_calc_scores`, always re-embed the
   cleaned tag strings server-side, or strip `vectors` from `cgp_output` on receipt.
3. **Fail closed on validator-LLM errors** (fixes D1): if the validator's own screen
   LLM fails, mark the task unscored that round instead of grading whatever remains.

A "flag any response whose per-tag cosine spread ≈ 0" monitor catches lazy
attackers (one identical vector on every tag) but a careful attacker adds per-tag
noise to evade it — so treat detection as a stopgap, not the fix.
