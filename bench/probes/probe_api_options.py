#!/usr/bin/env python3
"""Does the reserve endpoint serve a DIFFERENT pool under other query params?

ApiLib.py:47 appends `&options=22` when CGP_API_OPTIONS contains "22", so the
endpoint takes at least one documented switch. If a switch reaches a different
pool it is worth far more than polling the default pool harder.

Read-only probe: it prints what each variant serves and never writes the corpus.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from bench.fetch_testnet_corpus import api_key, content_key, describe  # noqa: E402

BASE = "https://api.conversations.xyz:443/api/v1/conversation/reserve"

VARIANTS = [
    ("default", "?cgp_version=2.36.73"),
    ("options22", "?cgp_version=2.36.73&options=22"),
    ("no_version", ""),
    ("old_version", "?cgp_version=1.0.0"),
]


def call(url: str, key: str):
    req = urllib.request.Request(
        url, data=b"{}", method="POST",
        headers={"Accept": "application/json", "Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main() -> None:
    key = api_key()
    known = set()
    corpus = os.path.join(ROOT, "data", "testnet_corpus.jsonl")
    with open(corpus) as f:
        for line in f:
            try:
                known.add(content_key(json.loads(line)))
            except Exception:
                pass
    print(f"known content keys in shipped corpus: {len(known)}")

    for name, qs in VARIANTS:
        types = Counter()
        keys = set()
        novel = 0
        err = ""
        for _ in range(5):
            try:
                b = call(BASE + qs, key)
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code} {e.read()[:80]!r}"
                break
            except Exception as e:
                err = f"{type(e).__name__} {str(e)[:80]}"
                break
            d = describe(b)
            types[d["type"]] += 1
            ck = content_key(b)
            keys.add(ck)
            if ck not in known:
                novel += 1
            time.sleep(0.4)
        print(f"{name:12s} err={err or '-':30s} served={dict(types)} "
              f"distinct_content={len(keys)} novel_vs_corpus={novel}")


if __name__ == "__main__":
    main()
