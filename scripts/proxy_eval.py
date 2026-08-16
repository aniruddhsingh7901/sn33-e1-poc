#!/usr/bin/env python
"""Offline proxy scorer for SN33 conversation tasks.

Proxy for the validator's score: embed each enrichment line (truncated to 500
chars) with text-embedding-3-small (1536d), target = mean of the line vectors,
then score each of our tags as cosine to the target. Reports per-task
mean / top3 / max cosine.

Input format (list of tasks, e.g. data/conv_24h_full.json):
    [{"enrichment": [str, ...], "our_tags": [str, ...],
      "final": float|null, "dt": str, ...}, ...]

Embeddings are cached persistently in sqlite (data/proxy_cache.sqlite3,
key = sha1(model + "\\x00" + text)) so repeat runs cost nothing.

CLI:
    venv/bin/python scripts/proxy_eval.py data/conv_24h_full.json \
        [--out data/proxy_baseline.json]

Module use:
    from proxy_eval import score_tasks
    rows = score_tasks(tasks)   # adds proxy_mean/proxy_top3/proxy_max per task
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(REPO, ".env")
CACHE_PATH = os.path.join(REPO, "data", "proxy_cache.sqlite3")

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
ENRICHMENT_TRUNCATE = 500
BATCH_SIZE = 100

# ---------------------------------------------------------------- env / client


def _load_env_key(name="OPENAI_API_KEY"):
    """Return the key from the environment, falling back to .env
    (lines look like `export KEY=value`)."""
    if os.environ.get(name):
        return os.environ[name]
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):]
                if line.startswith(name + "="):
                    val = line.split("=", 1)[1].strip().strip("'\"")
                    if val:
                        return val
    raise RuntimeError(f"{name} not found in environment or {ENV_PATH}")


def _client():
    from openai import OpenAI
    return OpenAI(api_key=_load_env_key())


# --------------------------------------------------------------------- caching


def _cache_conn(path=CACHE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        " key TEXT PRIMARY KEY, model TEXT, vec BLOB)"
    )
    return conn


def _key(model, text):
    return hashlib.sha1((model + "\x00" + text).encode("utf-8")).hexdigest()


class Embedder:
    """Cached, batched text-embedding-3-small embedder.

    Vectors are stored as float32 blobs keyed by sha1(model + text).
    """

    def __init__(self, cache_path=CACHE_PATH, model=EMBED_MODEL,
                 dims=EMBED_DIMS, batch_size=BATCH_SIZE):
        self.conn = _cache_conn(cache_path)
        self.model = model
        self.dims = dims
        self.batch_size = batch_size
        self._client = None
        self.api_texts = 0        # texts actually sent to the API this run
        self.api_chars = 0

    def _get_cached(self, keys):
        out = {}
        CHUNK = 500
        for i in range(0, len(keys), CHUNK):
            chunk = keys[i:i + CHUNK]
            q = ("SELECT key, vec FROM embeddings WHERE key IN (%s)"
                 % ",".join("?" * len(chunk)))
            for k, blob in self.conn.execute(q, chunk):
                out[k] = np.frombuffer(blob, dtype=np.float32)
        return out

    def embed(self, texts):
        """Return {text: np.ndarray(float32, 1536)} for unique texts."""
        texts = list(dict.fromkeys(texts))          # unique, order-preserving
        keys = {t: _key(self.model, t) for t in texts}
        cached = self._get_cached(list(keys.values()))
        missing = [t for t in texts if keys[t] not in cached]

        for i in range(0, len(missing), self.batch_size):
            batch = missing[i:i + self.batch_size]
            if self._client is None:
                self._client = _client()
            resp = self._client.embeddings.create(
                model=self.model, input=batch, dimensions=self.dims)
            rows = []
            for text, item in zip(batch, resp.data):
                vec = np.asarray(item.embedding, dtype=np.float32)
                cached[keys[text]] = vec
                rows.append((keys[text], self.model, vec.tobytes()))
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?,?,?)", rows)
            self.conn.commit()
            self.api_texts += len(batch)
            self.api_chars += sum(len(t) for t in batch)

        return {t: cached[keys[t]] for t in texts}


# --------------------------------------------------------------------- scoring


def _cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_tasks(tasks, embedder=None):
    """Score each task dict in place and return per-task result rows.

    Adds proxy_mean / proxy_top3 / proxy_max: cosines of our_tags to the
    mean of the enrichment-line embeddings (lines truncated to 500 chars).
    Tasks with no enrichment or no tags get None.
    """
    if embedder is None:
        embedder = Embedder()

    all_texts = []
    for t in tasks:
        all_texts.extend(e[:ENRICHMENT_TRUNCATE] for e in t.get("enrichment", []))
        all_texts.extend(t.get("our_tags", []))
    vecs = embedder.embed(all_texts)

    rows = []
    for t in tasks:
        enr = [e[:ENRICHMENT_TRUNCATE] for e in t.get("enrichment", [])]
        tags = t.get("our_tags", [])
        row = {"dt": t.get("dt"), "final": t.get("final"),
               "proxy_mean": None, "proxy_top3": None, "proxy_max": None}
        if enr and tags:
            target = np.mean([vecs[e] for e in enr], axis=0)
            cos = sorted((_cos(vecs[g], target) for g in tags), reverse=True)
            row["proxy_mean"] = float(np.mean(cos))
            row["proxy_top3"] = float(np.mean(cos[:3]))
            row["proxy_max"] = cos[0]
        t.update({k: row[k] for k in ("proxy_mean", "proxy_top3", "proxy_max")})
        rows.append(row)
    return rows


def correlations(rows):
    """Pearson/Spearman of proxy_mean vs final over scored tasks."""
    from scipy import stats
    scored = [r for r in rows
              if r["final"] is not None and r["proxy_mean"] is not None]

    def corr(sub):
        x = [r["proxy_mean"] for r in sub]
        y = [r["final"] for r in sub]
        return {"n": len(sub),
                "pearson": float(stats.pearsonr(x, y)[0]),
                "spearman": float(stats.spearmanr(x, y)[0])}

    out = {"all_scored": corr(scored)}
    nz = [r for r in scored if r["final"] != 0.0]
    if len(nz) != len(scored):
        out["excluding_zero_final"] = corr(nz)
    return out


# ------------------------------------------------------------------------- CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tasks_json", help="input tasks JSON (conv_24h_full format)")
    ap.add_argument("--out", help="write per-task baseline JSON here")
    ap.add_argument("--cache", default=CACHE_PATH)
    args = ap.parse_args(argv)

    with open(args.tasks_json) as f:
        tasks = json.load(f)

    embedder = Embedder(cache_path=args.cache)
    rows = score_tasks(tasks, embedder)
    corr = correlations(rows)

    print(f"tasks: {len(rows)}  scored (final!=null): "
          f"{sum(1 for r in rows if r['final'] is not None)}")
    print(f"API calls this run: {embedder.api_texts} texts, "
          f"{embedder.api_chars} chars (~{embedder.api_chars // 4} tokens, "
          f"~${embedder.api_chars / 4 / 1e6 * 0.02:.4f})")
    for name, c in corr.items():
        print(f"{name}: n={c['n']}  pearson={c['pearson']:.4f}  "
              f"spearman={c['spearman']:.4f}")

    if args.out:
        baseline = {"source": os.path.abspath(args.tasks_json),
                    "model": EMBED_MODEL, "dims": EMBED_DIMS,
                    "enrichment_truncate": ENRICHMENT_TRUNCATE,
                    "correlations": corr,
                    "tasks": rows}
        with open(args.out, "w") as f:
            json.dump(baseline, f, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
