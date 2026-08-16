"""SN33_SUMMARY_TAGS: honest centroid-summary phrases (conversation only).

Locks the invariants that matter regardless of whether the flag is deployed:
  * the flag is OFF by default (no behaviour change unless explicitly enabled),
  * make_summary_tags builds real umbrella phrases from the frequent theme words,
    capped at k, and never emits a single bare word,
  * only screen-safe phrases can reach the pool (the caller filters), so the
    validator's English screen can never delete a summary phrase.
"""
import os

from sn33.adapter import config_from_env
from sn33.tags import make_summary_tags, screen_safe


def test_flag_off_by_default(monkeypatch):
    monkeypatch.delenv("SN33_SUMMARY_TAGS", raising=False)
    assert config_from_env().summary_tags is False


def test_flag_on_and_k(monkeypatch):
    monkeypatch.setenv("SN33_SUMMARY_TAGS", "1")
    monkeypatch.setenv("SN33_SUMMARY_TAGS_K", "2")
    cfg = config_from_env()
    assert cfg.summary_tags is True
    assert cfg.summary_tags_k == 2


def test_make_summary_tags_builds_capped_multiword_phrases():
    pgt = ["housing market", "housing policy", "market rates", "policy makers",
           "rental market", "housing demand", "interest rates"]
    out = make_summary_tags(pgt, k=3)
    assert 1 <= len(out) <= 3
    # every phrase is multi-word (no bare single token) and drawn from theme words
    assert all(len(p.split()) >= 2 for p in out)
    assert "housing" in out[0]


def test_make_summary_tags_respects_k():
    pgt = ["alpha beta", "beta gamma", "gamma delta", "delta epsilon",
           "epsilon zeta", "zeta eta"]
    assert len(make_summary_tags(pgt, k=1)) == 1
    assert len(make_summary_tags(pgt, k=3)) <= 3


def test_short_words_ignored():
    # words <=3 chars ("the", "of", "ai") are not theme words
    pgt = ["the ai of housing", "the ai of markets", "housing markets"]
    out = make_summary_tags(pgt, k=3)
    for p in out:
        for w in p.split():
            assert len(w) > 3


def test_caller_only_ships_screen_safe():
    # The pipeline filters make_summary_tags through screen_safe. Dictionary
    # phrases pass; a phrase containing a non-dictionary token would be dropped.
    assert screen_safe("housing market policy") is True
    assert screen_safe("kubernetes inference") is False  # 'kubernetes' not in wordlist
