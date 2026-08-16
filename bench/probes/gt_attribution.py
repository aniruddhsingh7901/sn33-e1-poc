#!/usr/bin/env python3
"""Of the ~20 ground-truth tags, how many come from the conversation and how
many from the enrichment lines?

The validator builds ground truth as
    combine_metadata_tags([ conversation_tags, enrich_tags_1, ... enrich_tags_N ])
`real_ground_truth` already returns those pre-combine sets, so every final tag
can be attributed back to the input set(s) that produced it.
"""
from __future__ import annotations
import asyncio, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bench.faithful import load_cases, real_ground_truth
from conversationgenome.utils.Utils import Utils
from sn33.tags import normalize_all


def norm(tags):
    return set(normalize_all(Utils.get_clean_tag_set(list(tags))))


async def main():
    cases = load_cases(kind="conversation_tagging")
    print(f"{len(cases)} distinct conversations\n")
    rows = []
    for c in cases:
        gt = await real_ground_truth(c)
        if not gt.ok() or not gt.sets:
            continue
        convo = norm(gt.sets[0])                       # from the FULL conversation
        enrich = set().union(*[norm(s) for s in gt.sets[1:]]) if len(gt.sets) > 1 else set()
        final = norm(gt.tags)

        both = final & convo & enrich
        only_c = (final & convo) - enrich
        only_e = (final & enrich) - convo
        neither = final - convo - enrich
        rows.append(dict(n_lines=len(c.full_lines), n_enrich=len(c.enrichment_lines),
                         convo=len(convo), enrich=len(enrich), final=len(final),
                         only_c=len(only_c), only_e=len(only_e),
                         both=len(both), neither=len(neither),
                         opens=str(c.full_lines[0][1])[:34]))
        print(f"--- {rows[-1]['opens']!r}  ({len(c.full_lines)} lines, {len(c.enrichment_lines)} enrichment) ---")
        print(f"  tags produced by the 300-line conversation : {len(convo)}")
        print(f"  tags produced by the enrichment lines      : {len(enrich)}")
        print(f"  FINAL ground-truth tags after combine      : {len(final)}")
        print(f"     from conversation only : {len(only_c):2d}   e.g. {sorted(only_c)[:3]}")
        print(f"     from enrichment only   : {len(only_e):2d}   e.g. {sorted(only_e)[:3]}")
        print(f"     in both                : {len(both):2d}   e.g. {sorted(both)[:3]}")
        print(f"     in neither (combine invented it): {len(neither):2d}   e.g. {sorted(neither)[:3]}")
        print()

    if not rows:
        raise SystemExit("no usable cases")
    m = lambda k: statistics.mean(r[k] for r in rows)
    tot = m('final')
    print("=" * 66)
    print(f"AVERAGE over {len(rows)} conversations  (final GT = {tot:.1f} tags)")
    print("=" * 66)
    for k, lbl in [('only_c', 'conversation only'), ('only_e', 'enrichment only'),
                   ('both', 'both sources'), ('neither', 'invented by combine')]:
        print(f"  {lbl:22s} {m(k):5.1f} tags   {100*m(k)/tot:5.1f}%")
    print()
    print(f"  reachable by a miner that only sees enrichment: "
          f"{100*(m('only_e')+m('both'))/tot:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
