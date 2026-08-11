"""Content-addressed disk cache.

The bench replays the same conversations through dozens of strategy variants.
Without a cache every replay re-pays for ground-truth generation and embeddings;
with it, only genuinely new (model, prompt) pairs cost money.

Safe to import from the miner: if the cache cannot be opened for any reason the
functions degrade to no-ops rather than raising.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from typing import Any, Optional

_DEFAULT_DIR = os.environ.get("SN33_CACHE_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sn33_cache"
)

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_disabled = False


def key_for(*parts: Any) -> str:
    """Stable hash over arbitrary JSON-able parts."""
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _connect() -> Optional[sqlite3.Connection]:
    global _conn, _disabled
    if _disabled:
        return None
    if _conn is not None:
        return _conn
    try:
        os.makedirs(_DEFAULT_DIR, exist_ok=True)
        conn = sqlite3.connect(
            os.path.join(_DEFAULT_DIR, "cache.sqlite3"),
            check_same_thread=False,
            timeout=30,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        conn.commit()
        _conn = conn
        return _conn
    except Exception:
        _disabled = True
        return None


def get(k: str) -> Optional[Any]:
    conn = _connect()
    if conn is None:
        return None
    try:
        with _lock:
            row = conn.execute("SELECT v FROM kv WHERE k = ?", (k,)).fetchone()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


def put(k: str, value: Any) -> None:
    conn = _connect()
    if conn is None:
        return
    try:
        payload = json.dumps(value)
        with _lock:
            conn.execute("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (k, payload))
            conn.commit()
    except Exception:
        pass


def stats() -> dict:
    conn = _connect()
    if conn is None:
        return {"entries": 0, "disabled": True}
    try:
        with _lock:
            n = conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0]
        return {"entries": n, "path": _DEFAULT_DIR}
    except Exception:
        return {"entries": 0, "error": True}
