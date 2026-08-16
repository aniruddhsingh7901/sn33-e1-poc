"""Per-line quota + conditional demotion (2026-08-10 cohort-loss forensics).

The validator's GT combine gives each enrichment line one equal vote; these
tests pin the repair that stops majority-topic lines from soaking up every
slot, and the agreement gate that stops demotion from taxing tasks where the
window IS the enrichment topic.
"""

import numpy as np
import pytest

from sn33.pipeline import (
    Config,
    apply_line_quota,
    window_enrichment_agreement,
)


def _vecs(**kw):
    return {k: np.asarray(v, dtype=np.float32) for k, v in kw.items()}


TARGET = np.asarray([1.0, 0.0], dtype=np.float32)


class TestApplyLineQuota:
    def test_starved_line_gets_its_tags(self):
        # final is all majority-topic; line 2's own tags exist with vectors
        final = ["testing one", "testing two", "testing three"]
        per_line = [["testing one", "testing two"], ["hiv prevention", "prevention access"]]
        vectors = _vecs(**{"testing one": [1, 0], "testing two": [0.9, 0.1],
                           "testing three": [0.1, 0.9],   # weakest vs TARGET
                           "hiv prevention": [0.8, 0.2], "prevention access": [0.7, 0.3]})
        out = apply_line_quota(final, per_line, vectors, TARGET, quota=2)
        assert "hiv prevention" in out and "prevention access" in out
        assert "testing three" not in out          # weakest evicted
        assert len(out) == len(final)              # size preserved

    def test_covered_line_untouched(self):
        final = ["a tag", "b tag"]
        per_line = [["a tag", "b tag"]]
        vectors = _vecs(**{"a tag": [1, 0], "b tag": [0.9, 0.1]})
        assert apply_line_quota(final, per_line, vectors, TARGET, quota=2) == final

    def test_protected_never_evicted(self):
        final = ["verbatim gt", "weak tag"]
        per_line = [["line only"]]
        vectors = _vecs(**{"verbatim gt": [0.1, 0.9], "weak tag": [0.5, 0.5],
                           "line only": [0.9, 0.1]})
        out = apply_line_quota(final, per_line, vectors, TARGET, quota=1,
                               protected={"verbatim gt"})
        assert "verbatim gt" in out
        assert "weak tag" not in out and "line only" in out

    def test_never_swaps_in_weaker_than_evicted(self):
        # the line's best tag is WEAKER than the current weakest -> no swap
        final = ["strong one", "strong two"]
        per_line = [["weak line tag"]]
        vectors = _vecs(**{"strong one": [1, 0], "strong two": [0.95, 0.05],
                           "weak line tag": [0.0, 1.0]})
        assert apply_line_quota(final, per_line, vectors, TARGET, quota=1) == final

    def test_screen_floor_respected(self):
        # 'housing policy' is the only screen-safe tag; floor=1 forbids evicting it
        final = ["housing policy", "verbatim gt"]
        per_line = [["line tag"]]
        vectors = _vecs(**{"housing policy": [0.1, 0.9], "verbatim gt": [0.2, 0.8],
                           "line tag": [1, 0]})
        out = apply_line_quota(final, per_line, vectors, TARGET, quota=1,
                               protected={"verbatim gt"}, screen_floor=1)
        assert "housing policy" in out and "verbatim gt" in out
        assert "line tag" not in out           # nothing evictable

    def test_no_vector_line_tags_skipped(self):
        final = ["a tag", "b tag"]
        per_line = [["unembedded"]]
        vectors = _vecs(**{"a tag": [1, 0], "b tag": [0.9, 0.1]})
        assert apply_line_quota(final, per_line, vectors, TARGET, quota=1) == final


class TestAgreement:
    def test_disagreeing_texts_low(self):
        a = window_enrichment_agreement(
            "warhammer detachment charge huron miniatures painting hobby",
            ["industrial logistics real estate vacancy rates inflation"])
        assert a < 0.1

    def test_agreeing_texts_high(self):
        a = window_enrichment_agreement(
            "machine learning deep neural networks image classification training",
            ["deep learning neural networks explained", "image classification training guide"])
        assert a > 0.3

    def test_empty_inputs_zero(self):
        assert window_enrichment_agreement("", ["something"]) == 0.0
        assert window_enrichment_agreement("something", []) == 0.0


class TestGatingDefaults:
    def test_defaults_off(self):
        cfg = Config()
        assert cfg.enrichment_line_quota == 0
        assert cfg.demote_conditional is False

    def test_env_toggles(self, monkeypatch):
        from sn33 import adapter
        monkeypatch.setenv("SN33_LINE_QUOTA", "2")
        monkeypatch.setenv("SN33_DEMOTE_CONDITIONAL", "1")
        monkeypatch.setenv("SN33_DEMOTE_AGREE_CUT", "0.25")
        cfg = adapter.config_from_env()
        assert cfg.enrichment_line_quota == 2
        assert cfg.demote_conditional is True
        assert cfg.demote_agree_cut == pytest.approx(0.25)


class TestHeadCap:
    def test_family_capped_and_refilled(self):
        from sn33.pipeline import apply_head_cap
        final = ["housing market", "housing markets", "housing prices",
                 "housing price", "wage policy"]
        ranked = [(t, 0.9) for t in final] + [("minimum wage", 0.6), ("immigration", 0.55)]
        out = apply_head_cap(final, ranked, cap=2)
        # housing family capped at 2 (market/markets stem to same fam? market vs price
        # are different HEADS - families are by head noun: market(2) price(2) ok at cap 2)
        assert len(out) == len(final)
        assert "wage policy" in out

    def test_monoculture_broken(self):
        from sn33.pipeline import apply_head_cap
        fam = [f"housing market {i}" for i in range(0)]  # noqa
        final = ["housing market", "housing markets", "regional housing market",
                 "australian housing market", "city housing markets"]
        ranked = [(t, 0.9) for t in final] + [("minimum wage", 0.6),
                                              ("superannuation", 0.55), ("payslips", 0.5)]
        out = apply_head_cap(final, ranked, cap=2)
        market_fam = [t for t in out if t.split()[-1].rstrip("s") == "market"]
        assert len(market_fam) == 2
        assert "minimum wage" in out and "superannuation" in out and "payslips" in out

    def test_protected_exempt(self):
        from sn33.pipeline import apply_head_cap
        final = ["housing market", "housing markets", "housing submarket"]
        out = apply_head_cap(final, [(t, 1) for t in final], cap=1,
                             protected={"housing markets"})
        assert "housing markets" in out

    def test_cap_zero_is_identity(self):
        from sn33.pipeline import apply_head_cap
        final = ["a tag", "b tag"]
        assert apply_head_cap(final, [], 0) == final

    def test_env_toggle(self, monkeypatch):
        from sn33 import adapter
        monkeypatch.setenv("SN33_HEAD_CAP", "8")
        assert adapter.config_from_env().head_cap == 8
