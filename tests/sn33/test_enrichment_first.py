"""Enrichment-first aiming + NER combos: unit tests for the 2026-08-09 fixes.

Evidence base (both verifier-CONFIRMED on score-joined mainnet tasks):
  * conversation: fraction of tags anchored in enrichment predicts the real
    validator score at r=0.66; the replica's per-line enrichment tags were
    computed and discarded; the centroid was combine-only.
  * NER: single entities cap at ~0.56-0.64 cosine to the GT centroid; composite
    tags reach 0.60-0.83, measured +0.088 proxy adjusted on 10/10 tasks.

These tests pin the pure-CPU pieces: combo generation, demotion, config
plumbing, and the off-by-default gating that makes each fix A/B-able.
"""

import numpy as np
import pytest

from sn33.pipeline import (
    Config,
    demote_unanchored,
    ner_combo_candidates,
)


# ---------------------------------------------------------------------------
# ner_combo_candidates
# ---------------------------------------------------------------------------

class TestNerComboCandidates:
    def test_pairs_keep_both_orders_triples_one(self):
        combos = ner_combo_candidates(["alpha", "beta", "gamma"])
        assert "alpha beta" in combos and "beta alpha" in combos
        assert "alpha beta gamma" in combos
        assert "gamma beta alpha" not in combos  # triples: one order only

    def test_max_len_50_matches_our_normalize_not_validator_64(self):
        # _finish re-normalizes through tags.normalize (MAX_LEN=50), so a combo
        # longer than 50 would be silently deleted from the outgoing answer even
        # though the NER validator path allows 64. The builder must not emit it.
        long_a = "a" * 30
        long_b = "b" * 30
        combos = ner_combo_candidates([long_a, long_b, "cc"])
        assert f"{long_a} {long_b}" not in combos          # 61 chars - dropped
        assert all(len(c) <= 50 for c in combos)

    def test_caps_base_at_8_and_output_at_limit(self):
        ents = [f"ent{i}" for i in range(20)]
        combos = ner_combo_candidates(ents, limit=80)
        assert len(combos) <= 80
        used = {w for c in combos for w in c.split()}
        assert used <= set(ents[:8])                        # only top-8 entities

    def test_never_duplicates_an_input_entity(self):
        combos = ner_combo_candidates(["boston", "city hall"])
        assert "boston" not in combos and "city hall" not in combos

    def test_empty_and_single_entity(self):
        assert ner_combo_candidates([]) == []
        assert ner_combo_candidates(["solo"]) == []


# ---------------------------------------------------------------------------
# demote_unanchored
# ---------------------------------------------------------------------------

def _vecs(**kw):
    return {k: np.asarray(v, dtype=np.float32) for k, v in kw.items()}


class TestDemoteUnanchored:
    def test_lexical_anchor_keeps_score(self):
        ranked = [("housing policy", 0.8), ("random topic", 0.7)]
        vectors = _vecs(**{"housing policy": [1, 0], "random topic": [0, 1],
                           "housing": [1, 0]})
        out = demote_unanchored(ranked, vectors, ["housing"], [], exempt=set(),
                                demote=0.5, min_cos=0.99)
        d = dict(out)
        assert d["housing policy"] == pytest.approx(0.8)   # shares 'housing'
        assert d["random topic"] == pytest.approx(0.35)    # demoted 0.7*0.5

    def test_semantic_anchor_via_enrichment_centroid(self):
        # 'lodging' shares no word with the enrichment but its vector sits on
        # the enrichment centroid - semantic anchor keeps it undemoted.
        ranked = [("lodging", 0.6), ("asteroids", 0.6)]
        vectors = _vecs(lodging=[1, 0], asteroids=[0, 1], hotels=[1, 0])
        out = demote_unanchored(ranked, vectors, ["hotels"], [], exempt=set(),
                                demote=0.5, min_cos=0.9)
        d = dict(out)
        assert d["lodging"] == pytest.approx(0.6)
        assert d["asteroids"] == pytest.approx(0.3)

    def test_exempt_verbatim_gt_never_demoted(self):
        ranked = [("window only tag", 0.9)]
        vectors = _vecs(**{"window only tag": [0, 1], "enrich tag": [1, 0]})
        out = demote_unanchored(ranked, vectors, ["enrich tag"], [],
                                exempt={"window only tag"}, demote=0.5, min_cos=0.99)
        assert dict(out)["window only tag"] == pytest.approx(0.9)

    def test_enrichment_text_words_count_as_anchor(self):
        ranked = [("charging stations", 0.7)]
        vectors = _vecs(**{"charging stations": [0, 1], "unrelated": [1, 0]})
        out = demote_unanchored(
            ranked, vectors, ["unrelated"],
            ["EV charging stations are expanding."], exempt=set(),
            demote=0.5, min_cos=0.99)
        assert dict(out)["charging stations"] == pytest.approx(0.7)

    def test_empty_enrich_gt_is_identity(self):
        ranked = [("a", 0.9), ("b", 0.1)]
        assert demote_unanchored(ranked, {}, [], [], exempt=set()) == ranked

    def test_reorders_by_demoted_score(self):
        ranked = [("windowish", 0.80), ("enriched", 0.78)]
        vectors = _vecs(windowish=[0, 1], enriched=[1, 0], core=[1, 0])
        out = demote_unanchored(ranked, vectors, ["core"], [], exempt=set(),
                                demote=0.5, min_cos=0.9)
        assert out[0][0] == "enriched"                      # 0.78 beats 0.40

    def test_demote_never_deletes(self):
        ranked = [("only option", 0.5)]
        vectors = _vecs(**{"only option": [0, 1], "core": [1, 0]})
        out = demote_unanchored(ranked, vectors, ["core"], [], exempt=set())
        assert len(out) == 1                                # still submittable


# ---------------------------------------------------------------------------
# Config / adapter gating: OFF by default, singly toggleable
# ---------------------------------------------------------------------------

class TestGating:
    def test_defaults_off(self):
        cfg = Config()
        assert cfg.enrichment_first is False
        assert cfg.ner_combos is False
        assert cfg.enrichment_first_webpage is False

    def test_enrichment_first_kind_routing(self):
        from sn33.pipeline import _enrichment_first_on
        conv_only = Config(enrichment_first=True)
        assert _enrichment_first_on(conv_only, "conversation_tagging") is True
        assert _enrichment_first_on(conv_only, "webpage_metadata_generation") is False
        wp = Config(enrichment_first_webpage=True)
        assert _enrichment_first_on(wp, "webpage_metadata_generation") is True
        assert _enrichment_first_on(wp, "conversation_tagging") is False
        both = Config(enrichment_first=True, enrichment_first_webpage=True)
        for k in ("named_entities_extraction", "survey_tagging", "skill_generation"):
            assert _enrichment_first_on(both, k) is False

    def test_webpage_env_toggle(self, monkeypatch):
        from sn33 import adapter
        monkeypatch.setenv("SN33_ENRICHMENT_FIRST_WEBPAGE", "1")
        cfg = adapter.config_from_env()
        assert cfg.enrichment_first_webpage is True

    def test_adapter_env_toggles(self, monkeypatch):
        from sn33 import adapter
        monkeypatch.setenv("SN33_ENRICHMENT_FIRST", "1")
        monkeypatch.setenv("SN33_ENRICH_DEMOTE", "0.8")
        monkeypatch.setenv("SN33_NER_COMBOS", "1")
        cfg = adapter.config_from_env()
        assert cfg.enrichment_first is True
        assert cfg.enrichment_demote == pytest.approx(0.8)
        assert cfg.ner_combos is True

    def test_adapter_defaults_off(self, monkeypatch):
        from sn33 import adapter
        monkeypatch.delenv("SN33_ENRICHMENT_FIRST", raising=False)
        monkeypatch.delenv("SN33_NER_COMBOS", raising=False)
        cfg = adapter.config_from_env()
        assert cfg.enrichment_first is False
        assert cfg.ner_combos is False
