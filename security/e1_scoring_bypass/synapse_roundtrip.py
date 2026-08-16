#!/usr/bin/env python3
"""
Transport proof: does the forged `vectors` survive the WIRE?  (fully local, no network)

The only link reproduce.py doesn't exercise is the dendrite->axon wire hop. This
test constructs the REAL CgSynapse, sets cgp_output to the cheater payload,
serializes it EXACTLY as bittensor sends it over HTTP (pydantic JSON), then
deserializes it back on the "validator" side -- and checks the forged vectors
arrive byte-for-byte. No chain, no axon, no network, no third parties.

`cgp_output` is declared `Optional[List[dict]]` with NO schema, so arbitrary
attacker floats round-trip unchanged. This closes the last "is it an assumption"
gap: wire (here) + validator processing + scoring (reproduce.py) = full chain.
"""
import json

from conversationgenome.protocol import CgSynapse

JUNK = ["ab", "cd", "ef", "gh", "ij"]

# what actually goes on the wire is a plain list of floats (JSON-serializable),
# not a numpy array. 8 shown; a real payload is 1536.
FORGED = [0.0131, -0.0042, 0.0090, 0.0511, -0.0233, 0.0077, -0.0188, 0.0304]

payload = [{
    "tags": list(JUNK),
    "vectors": {t: {"vectors": FORGED} for t in JUNK},
}]

# 1) miner side: build the synapse the cheater returns
s = CgSynapse(cgp_input=[{"task": {"type": "conversation_tagging"}}], cgp_output=payload)

# 2) serialize -> deserialize exactly as the dendrite/axon HTTP body does (pydantic JSON)
wire_bytes = s.model_dump_json()
s2 = CgSynapse.model_validate_json(wire_bytes)

# 3) validator side: what the scoring code reads
got = s2.cgp_output[0]["vectors"]["ab"]["vectors"]
tags_got = s2.cgp_output[0]["tags"]

print("=== CgSynapse wire round-trip (local; no network) ===\n")
print(f"  cgp_output field type in protocol.py : Optional[List[dict]]  (NO schema validation)")
print(f"  sent   tags   : {payload[0]['tags']}")
print(f"  recv   tags   : {tags_got}")
print(f"  sent   vectors['ab']: {FORGED}")
print(f"  recv   vectors['ab']: {got}")
print(f"  wire size     : {len(wire_bytes)} bytes\n")

ok = (got == FORGED and tags_got == JUNK)
print("  RESULT:", "forged tags AND vectors survived the wire INTACT ✔"
      if ok else "payload was altered/stripped �’")
print("\n  => the wire carries attacker-supplied `vectors` unchanged into cgp_output[0];")
print("     combined with reproduce.py (real evaluate scores them), the E1 chain is")
print("     proven end-to-end with zero network and zero third-party impact.")
