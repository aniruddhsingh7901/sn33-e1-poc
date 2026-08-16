"""
cheater_miner.py -- the miner-side payload of the E1 scoring-bypass, for DEFENSIVE
reproduction only.

This module builds the exact `cgp_output` a cheating miner would return. It is
NOT a runnable miner: there is deliberately NO axon/dendrite, NO wallet, NO
network submission, NO registration. It exists so the subnet owner can inspect
the payload and feed it into their OWN validator code locally (see reproduce.py)
to confirm the bug without attacking the live network.

Why the payload has this exact shape (all verified against the validator code):

  * `tags` are 5 tags of < 3 characters. The validator's Utils.get_clean_tag_set
    drops every tag shorter than 3 chars, so the cleaned list becomes empty and
    ConversationTaggingTaskBundle.format_results raises "tags cannot be empty"
    BEFORE it can recompute the embeddings. FIVE (not one) so the response still
    clears the scorer's min_tags=3 gate (_has_enough_tags).

  * `vectors` is a dict {tag: {"vectors": <1536 floats>}} the attacker fills in.
    An HONEST miner sends vectors=None because the validator is supposed to
    recompute them. The exploit makes format_results crash before that recompute,
    so this attacker-supplied dict survives and the scorer trusts it.

  * The one forged vector is the attacker's best estimate of the validator's
    secret ground-truth centroid: run the validator's own tag prompt yourself,
    embed your ~20 guess tags, and AVERAGE them (averaging cancels each guess's
    error and lands ~0.90 cosine to the real centroid). All tags carry the SAME
    vector, which is why a naive detector could flag it (per-tag score spread
    ~0); a careful attacker would add small per-tag noise to hide.
"""
from typing import List, Optional


JUNK_TAGS = ["ab", "cd", "ef", "gh", "ij"]   # each < 3 chars -> cleaned list is empty -> raise


def honest_miner_output(real_tags: List[str]) -> list:
    """What an HONEST miner returns as synapse.cgp_output (verified sn33/adapter.py:195)."""
    return [{"tags": real_tags, "vectors": None}]   # no vectors: the validator recomputes them


def cheater_miner_output(forged_vector) -> list:
    """What a CHEATING miner returns as synapse.cgp_output.

    forged_vector: one embedding (list/np.ndarray of 1536 floats) aimed at the
    validator's ground-truth centroid (= mean of the attacker's own guess embeds).
    """
    return [{
        "tags": list(JUNK_TAGS),                                   # the TRIGGER (crashes format_results)
        "vectors": {t: {"vectors": forged_vector} for t in JUNK_TAGS},  # the PAYLOAD (trusted by the scorer)
    }]


# For reference only -- what a cheating miner's forward() would look like. Network
# wiring (axon serve, wallet, dendrite) is intentionally omitted so this cannot be
# run against a live validator.
def forward_reference(synapse, forged_vector):
    """A cheating miner's forward() core. DO NOT wire this to a live axon."""
    synapse.cgp_output = cheater_miner_output(forged_vector)
    return synapse
