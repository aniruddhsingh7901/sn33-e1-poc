#!/usr/bin/env python
"""Offline A/B: NER composite tags vs shipped config, on captured NER tasks.

Arm A = shipped Config, arm B = ner_combos=True. Both use_cache=True, so the
replica GT they AIM at is the same cached draw - paired.

Scoring is deliberately NOT that draw: a FRESH ground-truth draw
(use_cache=False, temperature-1, like the validator's own) is generated per
task and both arms are scored against its centroid with the real formula
(0.55*top3_unique + 0.25*mean + 0.10*median + 0.10*max, zero-padded top3,
unique = not string-in-GT). Scoring an arm against its own aim point would be
circular; an independent draw is exactly the validator's relationship to us.

    venv/bin/python scripts/ab_ner_combos.py [--n N] [--conc 2]
"""

import argparse
import asyncio
import json
import os
import statistics as st
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = ("/tmp/claude-1000/-home-anirudh-bittensor-conversation-genome-project/"
           "90b878db-c14b-471e-8482-0dddd9d0390f/scratchpad")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from proxy_eval import Embedder, _load_env_key  # noqa: E402

key = _load_env_key()
if not key:
    sys.exit("no OPENAI_API_KEY")
os.environ.setdefault("OPENAI_API_KEY", key)

import numpy as np  # noqa: E402

from sn33 import pipeline, replica  # noqa: E402
from sn33.pipeline import Config  # noqa: E402
from sn33.tags import normalize_all  # noqa: E402

from conversationgenome.utils.Utils import Utils  # noqa: E402


def adjusted(tags, gt_tags, vec):
    """The validator's formula against a GT centroid, penalties excluded (NER)."""
    gt_clean = Utils.get_clean_tag_set(gt_tags)
    gvecs = [vec[t] for t in gt_clean if t in vec]
    if not gvecs or not tags:
        return None
    cent = np.mean(np.asarray(gvecs, dtype=np.float32), axis=0)

    def cos(t):
        v = vec.get(t)
        if v is None:
            return None
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v) * np.linalg.norm(cent)
        return float(np.dot(v, cent) / n) if n else 0.0

    scores = [c for c in (cos(t) for t in tags) if c is not None]
    if not scores:
        return None
    gt_set = set(gt_clean)
    uniq = sorted((c for t, c in zip(tags, (cos(t) for t in tags))
                   if c is not None and t not in gt_set), reverse=True)
    top3 = (uniq + [0.0, 0.0, 0.0])[:3]
    return (0.55 * st.mean(top3) + 0.25 * st.mean(scores)
            + 0.10 * st.median(scores) + 0.10 * max(scores))


async def run_arm(tasks, cfg, conc):
    sem = asyncio.Semaphore(conc)
    out = [None] * len(tasks)

    async def one(i, t):
        async with sem:
            try:
                res = await pipeline.mine("named_entities_extraction",
                                          window=t["window"], cfg=cfg)
                out[i] = {"tags": res.tags, "source": res.source,
                          "elapsed": res.elapsed}
            except Exception as e:  # noqa: BLE001
                out[i] = {"tags": [], "source": f"error:{e}", "elapsed": 0.0}

    await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    return out


async def fresh_gt(tasks, conc):
    sem = asyncio.Semaphore(conc)
    out = [None] * len(tasks)

    async def one(i, t):
        async with sem:
            try:
                doc = str(t["window"][0][1]) if t["window"] else ""
                rep = await replica.replicate(
                    "named_entities_extraction", document=doc, convo_xml="",
                    enrichment=[], model="gpt-5.2", timeout=20.0,
                    use_cache=False, deadline=30.0)
                out[i] = normalize_all(rep.tags or [])
            except Exception:  # noqa: BLE001
                out[i] = []

    await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--conc", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "ab_ner_combos.json"))
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(os.path.join(SCRATCH, "srv_tasks_fresh.txt"))]
    tasks = []
    for r in recs:
        if r["task_type"] != "named_entities_extraction":
            continue
        win = r["task_raw"]["input"]["data"].get("window") or []
        win = [tuple(w) for w in win if isinstance(w, (list, tuple)) and len(w) >= 2]
        if win:
            tasks.append({"window": win, "ts": r["timestamp"]})
    if args.n:
        tasks = tasks[: args.n]
    print(f"NER tasks: {len(tasks)}", flush=True)

    t0 = time.perf_counter()
    arm_a = asyncio.run(run_arm(tasks, Config(use_cache=True), args.conc))
    print(f"arm A done {time.perf_counter()-t0:.0f}s", flush=True)
    t1 = time.perf_counter()
    arm_b = asyncio.run(run_arm(tasks, Config(use_cache=True, ner_combos=True), args.conc))
    print(f"arm B done {time.perf_counter()-t1:.0f}s", flush=True)
    t2 = time.perf_counter()
    gts = asyncio.run(fresh_gt(tasks, args.conc))
    print(f"fresh GT drawn {time.perf_counter()-t2:.0f}s", flush=True)

    emb = Embedder()
    texts = sorted({t for a, b, g in zip(arm_a, arm_b, gts)
                    for t in (a["tags"] + b["tags"] + list(Utils.get_clean_tag_set(g)))})
    vec = emb.embed(texts)

    paired = []
    for t, a, b, g in zip(tasks, arm_a, arm_b, gts):
        if not a["tags"] or not b["tags"] or len(g) < 3:
            continue
        sa = adjusted(a["tags"], g, vec)
        sb = adjusted(b["tags"], g, vec)
        if sa is None or sb is None:
            continue
        paired.append({"ts": t["ts"], "a": sa, "b": sb, "d": sb - sa,
                       "a_tags": a["tags"], "b_tags": b["tags"], "gt_n": len(g),
                       "a_elapsed": a["elapsed"], "b_elapsed": b["elapsed"]})

    d = [p["d"] for p in paired]
    print("\n================ RESULT (vs INDEPENDENT fresh GT draw) ================")
    print(f"paired: {len(paired)}")
    if d:
        print(f"est adjusted: A {st.mean(p['a'] for p in paired):.4f}"
              f" -> B {st.mean(p['b'] for p in paired):.4f}"
              f"  delta {st.mean(d):+.4f}  median {st.median(d):+.4f}"
              f"  W/L {sum(1 for x in d if x>0)}/{sum(1 for x in d if x<0)}")
        print(f"elapsed: A {st.mean(p['a_elapsed'] for p in paired):.2f}s"
              f"  B {st.mean(p['b_elapsed'] for p in paired):.2f}s")
        ex = max(paired, key=lambda p: p["d"])
        print(f"best win example B tags: {ex['b_tags'][:6]}")
    json.dump({"paired": paired}, open(args.out, "w"), indent=1)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
