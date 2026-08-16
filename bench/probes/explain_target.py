#!/usr/bin/env python3
"""Show the target vector for a real conversation, with real numbers."""
from __future__ import annotations
import asyncio, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import load_cases, real_ground_truth
from conversationgenome.utils.Utils import Utils
from sn33 import llm
from sn33.tags import cosine


async def main():
    case = [c for c in load_cases(kind="conversation_tagging")
            if str(c.full_lines[0][1]).startswith("Joe Rogan")][0]
    gt = await real_ground_truth(case)
    tags = Utils.get_clean_tag_set(gt.tags)
    vecs = await llm.embed(tags, use_cache=True)
    M = np.array([vecs[t] for t in tags if t in vecs])
    tags = [t for t in tags if t in vecs]

    print(f"conversation : {str(case.full_lines[0][1])[:50]!r}")
    print(f"ground truth : {len(tags)} tags\n")

    print("STEP A - each tag becomes 1536 numbers")
    t0 = tags[0]
    v0 = np.array(vecs[t0])
    print(f"  embed({t0!r})")
    print(f"    shape  = {v0.shape}")
    print(f"    first 6 = [{', '.join(f'{x:+.4f}' for x in v0[:6])}, ...]")
    print(f"    length  = {np.linalg.norm(v0):.4f}   (OpenAI returns unit vectors)\n")

    print("STEP B - average all of them, dimension by dimension")
    target = M.mean(axis=0)
    print(f"  target = np.mean(all {len(tags)} vectors, axis=0)")
    for d in range(3):
        col = M[:, d]
        print(f"    dim {d}: ({col[0]:+.4f} + {col[1]:+.4f} + ... + {col[-1]:+.4f}) / {len(tags)} = {target[d]:+.4f}")
    print(f"    length of target = {np.linalg.norm(target):.4f}   <- SHORTER than 1.0")
    print("    the tags point in different directions, so averaging cancels them out\n")

    print("STEP C - how close is each ground-truth tag to that average?")
    cs = sorted(((cosine(target, np.array(vecs[t])), t) for t in tags), reverse=True)
    for c, t in cs[:5]:
        print(f"    {c:.4f}  {t}")
    print("      ...")
    for c, t in cs[-3:]:
        print(f"    {c:.4f}  {t}")
    arr = np.array([c for c, _ in cs])
    print(f"\n    best {arr.max():.4f}   mean {arr.mean():.4f}   worst {arr.min():.4f}")
    print("    NO ground-truth tag reaches 1.0 - the target is not any of them\n")

    print("STEP D - pairwise: how similar are the tags to EACH OTHER?")
    sims = [float(np.dot(M[i], M[j])) for i in range(len(M)) for j in range(i+1, len(M))]
    print(f"    mean tag-to-tag cosine = {np.mean(sims):.4f}")
    print(f"    mean tag-to-TARGET     = {arr.mean():.4f}   <- higher: the average sits 'between' them\n")

    print("STEP E - what beats a ground-truth tag? something more central.")
    probes = ["housing policy", "real estate", "housing", "multifamily housing market",
              "housing policy and real estate development", "ufc", "joe rogan"]
    pv = await llm.embed(probes, use_cache=True)
    for p in probes:
        c = cosine(target, np.array(pv[p]))
        mark = "  <- beats every gt tag" if c > arr.max() else ""
        print(f"    {c:.4f}  {p!r}{mark}")


if __name__ == "__main__":
    asyncio.run(main())
