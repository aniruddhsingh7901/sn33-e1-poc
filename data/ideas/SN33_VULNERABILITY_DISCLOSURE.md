# Responsible Disclosure — SN33 (ReadyAI / conversation-genome) validator scoring bypass

**To:** ReadyAI / subnet-33 maintainers
**From:** an SN33 miner operator (defensive audit, local only)
**Date:** 2026-08-16
**Severity:** High — a miner can score above the entire honest field without producing a real answer
**Status:** Not exploited in the wild (verified, see §7). Reported, not deployed.

---

## 0. TL;DR

On the current validator scoring path a miner can:

1. send **5 sub-3-character junk tags** plus a **hand-crafted `vectors` dict**,
2. this makes `validate_tag_set` raise **before** the validator re-embeds anything,
3. the validator catches the exception and `continue`s — but `continue` does **not**
   remove the response from the list that gets scored,
4. so the scorer reads the miner's **own supplied embedding vectors** and computes
   cosine against them.

Because a single embedding vector aimed at the "average meaning" direction sits at
**cosine ~0.75 against almost any ground-truth centroid**, and a vector built from the
attacker's *own averaged guesses* reaches **~0.90**, the attacker scores **0.80 mean /
0.889 max** against an honest field that tops out at **~0.69** — **winning 50/50 tasks**
in our replay, having done none of the intended work.

The **same code path also fires with no attacker**, whenever the validator's *own* LLM
call fails: honest miners that send `vectors: None` are then scored a hard **0.0**.

Two one-line-ish fixes close both (§8).

---

## 0.5 CRITICAL UPDATE — the impact is version-dependent (verified 2026-08-16)

The 0.85 score-inflation above requires the validator's response loop to **catch
`format_results` and `continue`**. That local catch is **not** on `main`; it was added
in **commit `2741fa0e`** ("fixes and cleanup", JordanBourgault, 2026-08-14), part of
**PR #135 / v2.37.74** (`__version__ = "2.37.74"`) — the **mandatory mainnet upgrade for
2026-08-18**. The scorer's trust of miner-supplied `vectors` is identical on both.

| validator version | impact of the junk+forged payload |
|---|---|
| **`main` / pre-2.37.74 (no local catch)** | `format_results` raises → propagates to the top-level handler (`validator.py` "Top Level Validator Error") → `evaluate()` never runs → **Denial-of-Service** (batch aborts, nobody scored). |
| **v2.37.74 / PR #135 commit `2741fa0e` (try/except…continue)** | bad response retained → `evaluate()` grades the forged vectors → **score inflation ~0.85 (cap 0.90)** + honest miners zeroed (D1). |

**The catch-and-continue meant to stop one bad miner from crashing the batch is exactly
what converts the DoS into a score-inflation exploit** — it catches the error but leaves
the response in the scored list. The correct fix is catch **and drop from the list**
(§8 Fix 1) and/or never trust wire vectors (§8 Fix 2). Sections 4–5 below (the 0.80 /
50-of-50 measurements) therefore describe **VARIANT B (v2.37.74+)** behaviour.

A runnable local reproduction (both variants, real scorer, no network) is in
`security/e1_scoring_bypass/` — run `reproduce.py`.

---

## 1. Intended scoring (for reference)

`GroundTruthTagSimilarityScoringMechanism`:

```
adjusted = 0.55 * mean(top-3 UNIQUE tag cosines)
         + 0.25 * mean(all tag cosines)
         + 0.10 * median(all tag cosines)
         + 0.10 * max(all tag cosines)
final    = adjusted * penalties
```

Each tag's cosine is measured against `full_conversation_neighborhood` — the **mean of
the ~20 ground-truth tag embeddings** (`text-embedding-3-small`, 1536-dim).

**Design intent:** the validator *re-embeds the miner's tag strings itself* and runs an
LLM "good English keywords" screen first, so a miner is scored only on the semantic
quality of the words it actually submitted. The bug below defeats both protections.

---

## 2. The vulnerability chain (E1), line by line

Every step verified against the code on the current tree (branch `skill-coverage-upgrade`,
validator code is upstream PR #135). File paths are as shipped.

| # | file:line | what happens |
|---|---|---|
| 1 | `conversationgenome/utils/Utils.py` `get_clean_tag_set` / `get_safe_tag` | drops any tag whose cleaned form is **< 3** or **> 64** chars. Five tags like `ab, cd, ef, gh, ij` all vanish → `clean_tag_list = []` |
| 2 | `conversationgenome/llm/LlmLib.py:217` `validate_tag_set` | calls `prompt_manager.validate_tags_prompt(clean_tag_list)` |
| 3 | `conversationgenome/llm/prompt_manager.py:49-50` | `if not tags: raise ValueError("tags cannot be empty.")` — **raises before any LLM call**, so the attack costs the validator no tokens |
| 4 | `conversationgenome/task_bundle/ConversationTaggingTaskBundle.py:161-162` | `miner_result['tags'] = validate_tag_set(...)` raised on the RHS ⇒ **neither `tags` nor `vectors` is ever reassigned**. The dict still holds the attacker's wire values. |
| 5 | `neurons/validator.py:341-348` | `except Exception: bt.logging.error(...); continue` — the `continue` only skips `bt.logging.debug` and `vl.put_task`. **It never removes the response from `responses`.** |
| 6 | `neurons/validator.py:369` | `await task_bundle.evaluate(miner_responses=responses)` — evaluates the **unfiltered** list, including the response whose formatting just failed |
| 7 | `GroundTruthTagSimilarityScoringMechanism.py` `_calc_scores` | `tag_vector_dict = miner_result["vectors"]`, then `tag_vectors = tag_vector_dict[tag]['vectors']`, then cosine against the GT centroid — **using the attacker's floats** |

### The exact lines that make step 5 unsafe

```python
# neurons/validator.py
for response_idx, response in enumerate(responses):     # (:325) iterates the list
    ...
    try:
        miner_result = await task_bundle.format_results(miner_result)   # (:341) raises
    except Exception as e:
        bt.logging.error(...)                            # (:347)
        continue                                         # (:348) skips logging ONLY
    ...
(final_scores, rank_scores) = await task_bundle.evaluate(miner_responses=responses)  # (:369)
```

`continue` advances the loop but the `response` object is still in `responses`, so
`evaluate()` scores it.

### The exact line that trusts miner vectors

```python
# GroundTruthTagSimilarityScoringMechanism._calc_scores
tag_vector_dict = miner_result["vectors"]        # attacker-controlled dict
...
tag_vectors = tag_vector_dict[tag]['vectors']    # attacker-controlled floats
score = self._score_vector_similarity(full_conversation_neighborhood, tag_vectors, tag)
```

`_score_vector_similarity` is a bare cosine with no provenance check.

---

## 3. The enabling primitive (E2): why one fake vector scores ~0.75 for free

`text-embedding-3-small` vectors are **anisotropic** — they share a large common
direction. A ground-truth centroid is the mean of ~20 of them, which regresses even
harder toward that global-mean direction. Therefore **cosine(any-vector-aimed-at-the-
global-mean, any GT centroid) ≈ 0.75**, with no knowledge of the task.

Consequence for the metric itself (true even after E1 is fixed): honest miners live in a
narrow band (~0.55–0.69) sitting on top of a **~0.75 content-free floor**. See §6.

---

## 4. Worked example (real task, real numbers, through the real scoring class)

Task `map_idx=1` (Kubernetes autoscaling / LLM inference), scored with the **actual**
`GroundTruthTagSimilarityScoringMechanism` methods:

```
Ground truth (what the validator scores against), 20 tags:
  kubernetes, horizontal pod autoscaler, nvidia triton inference server,
  dynamic batching, kv cache management, ...

Attacker's OWN guesses (run the same prompts), 20 tags — different strings, same topic:
  kubernetes, cluster autoscaler, nvidia triton, llm serving, transformer models, ...

  cosine(single guess, GT centroid)          : avg 0.541, best 0.678, worst 0.441
  cosine(MEAN of the 20 guesses, GT centroid): 0.947    <-- higher than any single guess
    (averaging cancels each guess's random error, keeps the shared topic direction)

Payload the attacker submits:
  tags    = ["zzq1","zzq2","zzq3","zzq4","zzq5"]           # 5 junk tags, all < 3 chars? no:
                                                            # any set that get_clean_tag_set empties
  vectors = { "zzq1": {"vectors": FORGED}, ... }            # FORGED = mean(guess embeddings)

Result through the real class:
  HONEST answer (20 real tags):  adjusted 0.6260  final 0.6260
  ATTACK (junk + forged vector): adjusted 0.9467  final 0.8520
```

(The junk tags must be a set that `get_clean_tag_set` reduces to empty, which triggers the
raise. `ab,cd,ef,gh,ij` do this via the 3-char minimum.)

---

## 5. Measured impact (50 real tasks, validator's own scoring methods, independent GT draw)

```
                 mean     median   max     beats honest
HONEST           0.5484   0.620    0.750   --
CHEAT (blind)    0.5308   0.536    0.633   23/50   (one global-mean vector, zero task info)
CHEAT (smart)    0.7997   0.834    0.889   50/50   (mean of attacker's own guesses)
                                            gain +0.2513
```

- **Blind forgery is not enough** (0.53, tied) — a fixed generic vector only reaches the
  ~0.75 floor and loses the penalty edge.
- **Smart forgery wins every task** (0.80) — averaging the attacker's own guesses reaches
  ~0.90 cosine, far above the honest ceiling of ~0.69.

The honest column (0.548) matches the operator's real live conversation scores (~0.55),
confirming the harness is faithful.

Reproduction scripts (local, no network): `cheat_real_class.py` (drives the real class),
`cheat_walkthrough.py` (per-step numbers), `e1e2_50tasks.py` (50-task sweep).

---

## 6. D1 — the same bug zeros honest miners when the validator's own LLM fails

No attacker required. When the validator's screen LLM call fails (rate limit, quota,
timeout, 5xx):

```
llm_openai.py:56          basic_prompt catches the error, returns None
LlmLib.py:220             len(response_content) on None -> TypeError (same raise as E1)
validator.py:341-369      caught, continue, still evaluated
GroundTruthTag...:206     honest miner shipped vectors=None ->
                          `not tag in None` -> TypeError -> caught at :76-80 -> score 0.0
```

**Asymmetry:** in that same failure window, a miner shipping a populated (forged) `vectors`
dict scores normally (~0.70), while every honest miner shipping `vectors: None` is
**zeroed**. So a validator-side outage both breaks honest miners and rewards forgers.

We hit this accidentally during the audit when an OpenAI key ran out of credits — it is
not hypothetical.

---

## 7. We did **not** exploit this, and nobody currently is

- We did not deploy any forged-vector payload to any live validator. All testing was local.
- **Detection is trivial and we ran it:** a single forged vector makes every tag score
  *identically*, so the per-tag cosine spread is **exactly 0.0**. We scanned **7,139**
  real per-tag scoring breakdowns from validator W&B logs (`unique_scores` arrays):
  - rows with spread `== 0.0`: **0**
  - minimum spread observed: **0.0179** (p1 0.0297, median 0.1324)
  So no miner on the subnet currently shows the forged-vector signature.

We are reporting rather than using this, and recommend you add the 0.0-spread check as a
cheap monitoring signal until the root cause is fixed.

---

## 8. Suggested fixes (any one of the first two closes E1; both recommended)

1. **Drop the response from scoring when `format_results` raises.** The `continue` at
   `validator.py:348` should also exclude the response from `responses` (or evaluate only
   a `good_responses` list built inside the loop), so a formatting failure is a *skip*,
   never a *score-with-raw-wire-data*.

2. **Never trust miner-supplied vectors.** In `_calc_scores`, re-embed the (cleaned) tag
   strings server-side unconditionally and ignore any `vectors` key on the wire; or strip
   `vectors` from `cgp_output` on receipt. This also removes the incentive to ship vectors
   at all.

3. **Fail closed on validator-LLM errors (fixes D1).** `basic_prompt` should surface a
   sentinel the caller checks, and an LLM failure during `format_results` should mark the
   task *unscored for that round*, never "score with whatever the miner sent." Today an
   LLM outage converts the batch to "miner tags + miner vectors, unscreened."

4. **Schema-validate `cgp_output`.** It is currently `Optional[List[dict]]` with no schema,
   so pydantic round-trips arbitrary attacker floats bit-exact.

5. **Add the 0.0-spread detector** as monitoring: flag any scored response whose per-tag
   cosine spread is below ~0.005 (impossible for honest distinct tags).

### Structural note (independent of the bug)

Even with E1/D1 fixed, cosine-to-a-mean-of-embeddings has a ~0.75 content-free floor
(§2/§6), which compresses honest miners into a narrow band and limits how much signal the
metric carries about tag quality. Worth considering a normalization (e.g. subtract the
global mean direction before cosine, or z-score against a random-tag baseline) so honest
answer quality spreads across more of the 0–1 range.

---

## 9. Appendix — exact reproduction

All scripts are local, contact no network except OpenAI embeddings, and modify nothing.

```
cheat_real_class.py   # 50 tasks through the REAL GroundTruthTagSimilarityScoringMechanism
cheat_walkthrough.py  # one task, every intermediate number (guesses, centroid, cosine)
e1e2_50tasks.py       # blind vs smart forgery vs honest, 50 tasks
```

Honest baseline reproduces the operator's real live score (0.548 ≈ live 0.55), which
validates the harness before trusting the attack numbers.
