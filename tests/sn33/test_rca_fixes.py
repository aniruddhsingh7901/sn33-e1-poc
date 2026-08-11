"""RCA fixes 2026-08-11: embed retry (D), enrichment fallback (B), value alloc (C).

Evidence base (data/uid24_rca.json + verified synthesis):
  * D: 2/127 tasks/day hit a silent embed failure -> source=pool at ~0.32-0.48
    vs cohorts ~0.67, with degraded=False and no log line.
  * B: all 3 pool-truncation losses shipped 0/N enrichment-line coverage while
    rep.enrichment_tags sat computed and unused; GT is ~88% enrichment.
  * C: corpus-wide signature - (>=1 line uncovered OR >=3 window-only tags)
    -> margin -0.100 (n=13) vs -0.007 (n=16) without. Canary-only.
"""

import asyncio

import numpy as np
import pytest

from sn33 import llm
from sn33.pipeline import Config, allocate_value_based


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# D: embed retry + unconditional logging
# ---------------------------------------------------------------------------

class TestEmbedRetry:
    def _flaky_client(self, monkeypatch, fail_times):
        """Embeddings endpoint that fails the first `fail_times` calls."""
        calls = {"n": 0}

        class FakeData:
            def __init__(self, i):
                self.index = i
                self.embedding = [1.0, 0.0]

        class FakeResp:
            def __init__(self, batch):
                self.data = [FakeData(i) for i in range(len(batch))]
                self.usage = type("U", (), {"total_tokens": 1})()

        class FakeEmbeddings:
            async def create(self, input, model, dimensions):
                calls["n"] += 1
                if calls["n"] <= fail_times:
                    raise RuntimeError("boom")
                return FakeResp(input)

        class FakeClient:
            embeddings = FakeEmbeddings()

        monkeypatch.setattr(llm, "client", lambda: FakeClient())
        return calls

    def test_no_retry_by_default_and_failure_logged(self, monkeypatch, capsys):
        self._flaky_client(monkeypatch, fail_times=1)
        out = _run(llm.embed(["alpha"], use_cache=False))
        assert out == {}                                   # failed, no retry
        assert "embed FAILED" in capsys.readouterr().out   # ALWAYS logged now

    def test_retry_recovers_failed_batch(self, monkeypatch, capsys):
        calls = self._flaky_client(monkeypatch, fail_times=1)
        out = _run(llm.embed(["alpha"], use_cache=False, retry_timeout=1.0))
        assert "alpha" in out                              # salvage pass worked
        assert calls["n"] == 2
        logged = capsys.readouterr().out
        assert "embed FAILED" in logged and "recovered 1/1" in logged

    def test_retry_not_attempted_when_all_ok(self, monkeypatch):
        calls = self._flaky_client(monkeypatch, fail_times=0)
        out = _run(llm.embed(["alpha"], use_cache=False, retry_timeout=1.0))
        assert "alpha" in out
        assert calls["n"] == 1                             # no second call

    def test_config_gate_default_off(self):
        assert Config().embed_retry is False

    def test_adapter_env(self, monkeypatch):
        from sn33 import adapter
        monkeypatch.setenv("SN33_EMBED_RETRY", "1")
        assert adapter.config_from_env().embed_retry is True


# ---------------------------------------------------------------------------
# B: enrichment-first fallback (unit-level: the round-robin construction is
# exercised through mine() in integration tests; here we pin gating + config)
# ---------------------------------------------------------------------------

class TestFallbackEnrich:
    def test_config_gate_default_off(self):
        assert Config().fallback_enrichment is False

    def test_adapter_env(self, monkeypatch):
        from sn33 import adapter
        monkeypatch.setenv("SN33_FALLBACK_ENRICH", "1")
        assert adapter.config_from_env().fallback_enrichment is True


# ---------------------------------------------------------------------------
# C: value-based allocation
# ---------------------------------------------------------------------------

def _vecs(**kw):
    return {k: np.asarray(v, dtype=np.float32) for k, v in kw.items()}


TARGET = np.asarray([1.0, 0.0], dtype=np.float32)


class TestValueAlloc:
    def test_strong_line_gets_more_slots_junk_line_zero(self):
        # line A strong (cos~1), line B junk (cos~0) -> B earns nothing
        final = ["filler one", "filler two", "filler three", "filler four"]
        per_line = [["strong a", "strong b", "strong c"], ["junk x", "junk y"]]
        vectors = _vecs(**{"strong a": [1, 0], "strong b": [0.95, 0.05],
                           "strong c": [0.9, 0.1], "junk x": [0, 1], "junk y": [0, 1],
                           "filler one": [0.6, 0.4], "filler two": [0.5, 0.5],
                           "filler three": [0.4, 0.6], "filler four": [0.3, 0.7]})
        out = allocate_value_based(final, per_line, vectors, TARGET,
                                   enrich_vocab={"strong", "a", "b", "c"},
                                   target_tags=4, line_min_cos=0.35)
        assert "strong a" in out and "strong b" in out
        assert "junk x" not in out and "junk y" not in out   # no wasted quota

    def test_window_only_capped(self):
        # 4 unanchored (window-only) tags in compose order; cap=2 keeps only 2
        final = ["win one", "win two", "win three", "win four"]
        per_line = [["enrich tag"]]
        vectors = _vecs(**{"enrich tag": [1, 0], "win one": [0.9, 0.1],
                           "win two": [0.8, 0.2], "win three": [0.7, 0.3],
                           "win four": [0.6, 0.4]})
        out = allocate_value_based(final, per_line, vectors, TARGET,
                                   enrich_vocab={"enrich", "tag"},
                                   target_tags=4, window_cap=2)
        window_kept = [t for t in out if t.startswith("win")]
        assert len(window_kept) == 2
        assert "enrich tag" in out

    def test_protected_survive(self):
        final = ["verbatim gt", "win one", "win two", "win three"]
        per_line = [["line tag one", "line tag two", "line tag three"]]
        vectors = _vecs(**{"verbatim gt": [0.2, 0.8], "win one": [0.9, 0.1],
                           "win two": [0.8, 0.2], "win three": [0.7, 0.3],
                           "line tag one": [1, 0], "line tag two": [0.95, 0.05],
                           "line tag three": [0.9, 0.1]})
        out = allocate_value_based(final, per_line, vectors, TARGET,
                                   enrich_vocab={"line", "tag", "one", "two", "three"},
                                   protected={"verbatim gt"},
                                   target_tags=4, window_cap=1)
        assert "verbatim gt" in out

    def test_no_active_lines_is_identity(self):
        final = ["a tag", "b tag"]
        per_line = [["junk"]]
        vectors = _vecs(**{"junk": [0, 1], "a tag": [1, 0], "b tag": [0.9, 0.1]})
        assert allocate_value_based(final, per_line, vectors, TARGET,
                                    enrich_vocab=set(), line_min_cos=0.35) == final

    def test_config_gate_default_off(self):
        cfg = Config()
        assert cfg.value_alloc is False

    def test_adapter_env(self, monkeypatch):
        from sn33 import adapter
        monkeypatch.setenv("SN33_VALUE_ALLOC", "1")
        monkeypatch.setenv("SN33_VALUE_WINDOW_CAP", "3")
        cfg = adapter.config_from_env()
        assert cfg.value_alloc is True
        assert cfg.value_window_cap == 3
