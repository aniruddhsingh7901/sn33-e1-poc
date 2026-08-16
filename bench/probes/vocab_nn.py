#!/usr/bin/env python3
"""Would a precomputed vocabulary, searched by nearest-neighbour, beat our pool?

The pool is the bottleneck: perfect selection is worth +0.0012, a perfect
centroid +0.0036, but nothing we ship can beat the best tag we GENERATED
(0.6644 on the Joe Rogan window). A vocabulary lookup attacks that directly -
embed a large phrase list once, then at mine time just search it. Zero runtime
API calls, zero latency.

This tests the idea using only tags already in cache: build a vocabulary from
every conversation's ground truth and candidate pool, then for each conversation
rank the FOREIGN part of that vocabulary (everything not generated for this
conversation) against its real target.

    python bench/probes/vocab_nn.py
"""
from __future__ import annotations
import asyncio, os, statistics, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import load_cases, expand_windows, real_ground_truth
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline
from sn33.tags import cosine, normalize_all


async def main():
    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases)
    cfg = pipeline.Config(use_cache=True, use_local=False, deadline_s=600, call_timeout_s=180)

    per_convo = {}
    vocab_vecs = {}
    for c in cases:
        w = [x for cc, x in pairs if cc is c][0]
        gt = await real_ground_truth(c)
        if not gt.ok():
            continue
        res = await pipeline.mine(c.kind, window=w, enrichment=c.enrichment_lines, cfg=cfg)
        gtt = normalize_all(Utils.get_clean_tag_set(gt.tags))
        v = await llm.embed(gtt, use_cache=True)
        own = {t for t in gtt if t in v} | {t for t in res.candidates if t in res.vectors}
        for t in gtt:
            if t in v: vocab_vecs[t] = np.asarray(v[t])
        for t in res.candidates:
            if t in res.vectors: vocab_vecs[t] = np.asarray(res.vectors[t])
        tgt = np.array([v[t] for t in gtt if t in v]).mean(axis=0)
        per_convo[c.guid] = dict(head=str(c.full_lines[0][1])[:30], target=tgt, own=own,
                                 gtset=set(gtt), shipped=res.tags,
                                 pool={t: np.asarray(res.vectors[t]) for t in res.candidates
                                       if t in res.vectors})

    print(f"  vocabulary built from {len(cases)} conversations: {len(vocab_vecs)} distinct tags\n")
    print(f"  {'conversation':32s} {'our pool best':>13s} {'FOREIGN vocab best':>19s} {'gain':>8s}")
    print("  " + "-"*78)
    gains = []
    for guid, d in per_convo.items():
        T = d["target"]
        own_best = max(cosine(T, v) for v in d["pool"].values())
        foreign = {t: v for t, v in vocab_vecs.items() if t not in d["own"]}
        if not foreign:
            continue
        ranked = sorted(foreign.items(), key=lambda kv: -cosine(T, kv[1]))
        f_best = cosine(T, ranked[0][1])
        gains.append(f_best - own_best)
        print(f"  {d['head']:32s} {own_best:13.4f} {f_best:19.4f} {f_best-own_best:+8.4f}")
        for t, v in ranked[:3]:
            mark = "BOTH" if t in d["gtset"] else "unique"
            print(f"      would add: {t:34s} {cosine(T, v):.4f}  {mark}")
    print(f"\n  mean gain from searching a FOREIGN vocabulary: {statistics.mean(gains):+.4f}")
    print(f"  (positive = the vocabulary contained a better tag than we generated)")


if __name__ == "__main__":
    asyncio.run(main())
