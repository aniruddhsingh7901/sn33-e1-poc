#!/usr/bin/env python3
"""Pull real task bundles from the ReadyAI testnet API.

Why this exists: a miner only ever receives one ~10-line window of a
conversation, with the guid masked. The validator builds ground truth from the
FULL conversation. So every offline conversation benchmark we could build was
scored against a proxy ground truth generated from 3% of the source - and that
proxy already produced one measurably wrong recommendation.

The testnet API returns the *complete* bundle: all conversation lines, the real
guid, the enrichment payload, and (for surveys) the literal `selected_choices`
that are the ground truth. That makes a faithful harness possible.

The key ships in the repo (`testnet_readyai_api_data.json`, marked "will only
work on testnet") and needs no registration or stake - verified.

    python bench/fetch_testnet_corpus.py --n 40 --out data/testnet_corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://api.conversations.xyz:443/api/v1/conversation/reserve?cgp_version=2.36.73"


def api_key() -> str:
    with open(os.path.join(ROOT, "testnet_readyai_api_data.json")) as f:
        return json.load(f)["api_key"]


def reserve(key: str, timeout: float = 60.0):
    req = urllib.request.Request(
        URL,
        data=b"{}",
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def content_key(bundle: dict) -> str:
    """Fingerprint the SOURCE, not the bundle.

    The API reissues the same conversation under fresh guids - 75 bundles in
    one sweep turned out to be only 4 distinct conversations. Deduping on guid
    therefore massively overstates corpus size, so we hash the actual content.
    """
    import hashlib

    inp = bundle.get("input") or {}
    data = inp.get("data") or {}
    lines = data.get("lines") or []
    if not lines:
        return f"empty:{inp.get('guid')}"
    head = json.dumps(lines[:3], sort_keys=True)[:2000]
    return hashlib.md5(f"{bundle.get('type')}|{len(lines)}|{head}".encode()).hexdigest()


def describe(bundle: dict) -> dict:
    """Size up a bundle without keeping the whole thing in memory twice."""
    inp = bundle.get("input") or {}
    data = inp.get("data") or {}
    lines = data.get("lines") or []
    enrichment = data.get("enrichment") or {}
    searches = (enrichment or {}).get("search_results") or {}
    return {
        "type": bundle.get("type"),
        "guid": inp.get("guid"),
        "lines": len(lines),
        "enrichment_queries": len(searches),
        "enrichment_results": sum(len(v or []) for v in searches.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "testnet_corpus.jsonl"))
    ap.add_argument("--sleep", type=float, default=0.4, help="be polite to the API")
    ap.add_argument("--want", default="", help="only keep this task type")
    ap.add_argument("--until-dry", type=int, default=0,
                    help="stop after this many consecutive fetches with no NEW content")
    args = ap.parse_args()

    key = api_key()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Resume-friendly, deduped on CONTENT so reissued guids do not inflate it.
    seen = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    seen.add(content_key(json.loads(line)))
                except Exception:
                    continue
        print(f"existing corpus: {len(seen)} distinct sources")

    kinds = Counter()
    kept = 0
    dry = 0
    with open(args.out, "a") as out:
        for i in range(args.n):
            try:
                b = reserve(key)
            except urllib.error.HTTPError as e:
                print(f"  HTTP {e.code}: {e.read()[:120]!r}")
                break
            except Exception as e:
                print(f"  {type(e).__name__}: {str(e)[:90]}")
                time.sleep(2)
                continue

            d = describe(b)
            kinds[d["type"]] += 1
            ck = content_key(b)
            if ck in seen:
                dry += 1
                if args.until_dry and dry >= args.until_dry:
                    print(f"  pool exhausted: {dry} consecutive fetches with no new content")
                    break
                continue
            if args.want and d["type"] != args.want:
                continue
            dry = 0
            seen.add(ck)
            out.write(json.dumps(b) + "\n")
            out.flush()
            kept += 1
            print(f"  [{i+1}/{args.n}] {d['type']:28s} lines={d['lines']:4d} "
                  f"enrichment={d['enrichment_results']}")
            time.sleep(args.sleep)

    print(f"\nfetched {sum(kinds.values())}, kept {kept} NEW distinct sources -> {args.out}")
    print("task type mix (bundles served):", dict(kinds))
    print(f"distinct sources now held: {len(seen)}")


if __name__ == "__main__":
    main()
