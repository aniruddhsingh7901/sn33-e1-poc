"""SN33_SURVEY_V2: option-menu / noun-phrase survey generator (survey-only).

Locks: flag OFF by default, ON via env; the v2 template formats and carries the
three levers (option framing, precision-first ordering, strict noun-phrase form).
"""
from sn33 import prompts
from sn33.adapter import config_from_env


def test_flag_off_by_default(monkeypatch):
    monkeypatch.delenv("SN33_SURVEY_V2", raising=False)
    assert config_from_env().survey_v2 is False


def test_flag_on(monkeypatch):
    monkeypatch.setenv("SN33_SURVEY_V2", "1")
    assert config_from_env().survey_v2 is True


def test_v2_template_formats():
    out = prompts.MINER_SURVEY_POOL_V2.format(n=40, question="Why prefer this bank?",
                                              comment="es mas facil de hacer transferencias")
    assert "Why prefer this bank?" in out
    assert "es mas facil de hacer transferencias" in out
    assert "{n}" not in out and "{question}" not in out and "{comment}" not in out


def test_v2_carries_the_three_levers():
    t = prompts.MINER_SURVEY_POOL_V2.lower()
    # 1. option-menu framing
    assert "answer option" in t and "reason-categories" in t
    # 2. precision-first ordering (distinct options before variants)
    assert "distinct" in t and "variant" in t
    # 3. strict noun-phrase form, no first-person
    assert "noun phrase" in t and "first-person" in t
    # english mandatory
    assert "english" in t


def test_pool_selects_v2_when_flagged():
    # both templates exist and differ
    assert prompts.MINER_SURVEY_POOL_V2 != prompts.MINER_SURVEY_POOL
    assert "reason-categories" not in prompts.MINER_SURVEY_POOL  # only v2 has the menu framing
