# SN33 miner work — start here

Four documents, in reading order:

| document | what it answers |
|---|---|
| **[SN33_RESULTS.md](SN33_RESULTS.md)** | What was measured, against what baseline, with what fidelity limits |
| **[../sn33/README.md](../sn33/README.md)** | How the miner works and why it is shaped that way |
| **[SN33_MINER_RUNBOOK.md](SN33_MINER_RUNBOOK.md)** | Bare server → scoring miner, and what to watch |
| **[SN33_ATTACK_SURFACE.md](SN33_ATTACK_SURFACE.md)** | What miners could do, what we do, what to report upstream |

## Summary

| task | share | before | after | delta |
|---|---|---|---|---|
| conversation tagging | 66% | 0.5718 | 0.6408 | +0.069 |
| webpage metadata | 15% | 0.5760 | 0.6180 | +0.042 |
| named entities | 12% | 0.5561 | 0.6247 | +0.069 |
| survey tagging | 6% | **0.0000** | 0.4804 | +0.480 |
| skill generation | new | 0.4711 | 0.6292 | +0.158 |

All five significant on a paired bootstrap. Traffic-weighted, that is roughly
**+0.09 mean score**, most of it from fixing outright failures rather than from
better tags.

## Where that puts us against real miners

From the validators' own W&B logs (6,738 scored responses, 258 miners):
best miner **0.6961**, top-10 0.678–0.696, median **0.6196**. Our traffic-weighted
estimate is ~**0.627** — above the median, roughly 0.03–0.05 short of the top
cluster. That gap is real and not yet closed; `SN33_RESULTS.md` lists the four
things tried that did not close it.

Two facts worth internalising before optimising further: the top ~25 miners are
within one standard error of each other (±0.016 at ~26 samples each), and every
one of them has a **zero zero-rate** against a 5.4% population average.

## What was actually wrong

Three production defects, each measured rather than inferred:

1. **Survey tasks scored a hard zero, always.** The validator's tag screen keeps
   only "good English keywords"; all captured survey traffic is Spanish. 0 of 6
   Spanish tags survive, 6 of 6 English ones do.
2. **17% of webpage and NER responses returned no tags at all** (19/113 and
   29/171 in the captured logs) — each one a zero that decays the EMA 5%.
3. **38.5% of survey tags were deleted before scoring** by accent stripping, and
   98% of webpage responses submitted ≥20 tags, triggering the validator's
   random-20 cull.

## Two things to decide

* **We have no registered UIDs.** The May miners were deregistered. Registration
  is ~τ0.0358/UID with a 24h immunity window; the miner should be verified
  before spending it.
* **Rank 1 pays ~30% more than median**, and rank 10 sits within 5% of rank 1
  (measured on-chain). Given weights are rank-based and nearly flat, the
  economics favour *never scoring zero across several keys* over squeezing the
  last 0.01 of tag quality.

## Verify everything

```bash
pytest tests/sn33/ -q          # 142 tests, incl. scorer parity vs the validator's own class
python bench/run.py --kind webpage_metadata_generation --n 22 --strategies prod,replica
python bench/timing.py --n 5 --repeat 2
```
