#!/usr/bin/env python3
"""Does ADDING a vocabulary to the candidate pool raise the score?

vocab_nn.py asked "is the best foreign tag better than our best own tag" - the
wrong question, because extra candidates never displace ours by existing. The
right question is whether our OWN selection, run on a bigger pool using our own
ESTIMATED target, ends up shipping a better answer. That can go either way: a
foreign tag that flatters our estimate but sits far from the real target will
displace something good.
"""
from __future__ import annotations
import asyncio, os, statistics, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import load_cases, expand_windows, real_ground_truth
from conversationgenome.utils.Utils import Utils
from sn33 import llm, pipeline
from sn33.tags import centroid, cosine, normalize_all


async def main():
    cases = load_cases(kind="conversation_tagging")
    pairs = expand_windows(cases)[:40]
    cfg = pipeline.Config(use_cache=True, use_local=False, deadline_s=600, call_timeout_s=180)

    # Vocabulary: every tag we have a cached vector for, from every conversation.
    # Keyed by SOURCE conversation, so the one under test can be held out.
    # Without that hold-out the vocabulary contains the very ground-truth tags
    # we are trying to guess, which is not something a miner ever has.
    by_source = {}
    for c in cases:
        got = {}
        gt = await real_ground_truth(c)
        if gt.ok():
            gtt = normalize_all(Utils.get_clean_tag_set(gt.tags))
            v = await llm.embed(gtt, use_cache=True)
            for t in gtt:
                if t in v: got[t] = np.asarray(v[t])
        w = [x for cc, x in pairs if cc is c]
        if w:
            r = await pipeline.mine(c.kind, window=w[0], enrichment=c.enrichment_lines, cfg=cfg)
            for t in r.candidates:
                if t in r.vectors: got[t] = np.asarray(r.vectors[t])
        by_source[c.guid] = got
    print(f"  vocabulary: {sum(len(g) for g in by_source.values())} tags "
          f"from {len(by_source)} conversations (test conversation held out)\n")

    base_s, vocab_s = [], []
    for case, w in pairs:
        gt = await real_ground_truth(case)
        if not gt.ok(): continue
        res = await pipeline.mine(case.kind, window=w, enrichment=case.enrichment_lines, cfg=cfg)
        if not res.candidates: continue
        # HOLD OUT everything derived from this conversation.
        vocab = {}
        for guid, got in by_source.items():
            if guid == case.guid:
                continue
            vocab.update(got)
        own = {t for t in res.candidates}
        est_tags = [t for t in normalize_all(Utils.get_clean_tag_set(res.predicted_gt))
                    if t in res.vectors]
        if not est_tags: continue
        est = np.asarray(centroid([res.vectors[t] for t in est_tags]))

        # add the foreign part of the vocabulary, ranked by OUR estimate
        foreign = [(t, v) for t, v in vocab.items() if t not in own]
        top = sorted(foreign, key=lambda kv: -cosine(est, kv[1]))[:20]
        merged_vecs = dict(res.vectors); merged_vecs.update({t: v.tolist() for t, v in top})
        merged = list(res.candidates) + [t for t, _ in top]

        profile = pipeline.TASK_PROFILE[case.kind]
        from sn33.tags import rank_by_centroid
        ranked = rank_by_centroid(merged, merged_vecs, est)
        picked = pipeline.compose(ranked, res.predicted_gt, profile,
                                  target_tags=12, insurance=6)
        # Score locally against the REAL target. The validator's own path needs
        # an LLM screen call; this skips it and computes the same arithmetic, so
        # the comparison stays valid while the API is unavailable.
        gtt = normalize_all(Utils.get_clean_tag_set(gt.tags))
        gv = await llm.embed(gtt, use_cache=True)
        RT = np.array([gv[t] for t in gtt if t in gv]).mean(axis=0)
        L = np.linalg.norm(RT); GTS = set(gtt)
        allv = dict(merged_vecs); allv.update({t: gv[t] for t in gtt if t in gv})
        def adj(tags):
            sc = [float(np.asarray(allv[t]) @ RT)/L for t in tags if t in allv]
            uq = [float(np.asarray(allv[t]) @ RT)/L for t in tags if t in allv and t not in GTS]
            if not sc: return 0.0
            t3 = sorted(uq)[-3:]
            while len(t3) < 3: t3.append(0.0)
            return (0.55*statistics.mean(t3) + 0.25*statistics.mean(sc)
                    + 0.10*statistics.median(sc) + 0.10*max(sc))
        base_s.append(adj(res.tags)); vocab_s.append(adj(picked))

    d = [y - x for x, y in zip(base_s, vocab_s)]
    print(f"  n = {len(d)} windows")
    print(f"    baseline pool          adjusted {statistics.mean(base_s):.4f}")
    print(f"    pool + 20 vocab tags   adjusted {statistics.mean(vocab_s):.4f}")
    print(f"    delta                  {statistics.mean(d):+.4f}")
    print(f"    wins {sum(1 for x in d if x>1e-9)}  losses {sum(1 for x in d if x<-1e-9)}"
          f"  ties {sum(1 for x in d if abs(x)<=1e-9)}")


if __name__ == "__main__":
    asyncio.run(main())
