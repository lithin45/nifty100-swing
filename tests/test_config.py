import pytest
from pydantic import ValidationError

from config.loader import load_universe, universe_map
from config.schema import Settings, Stock


def test_weights_normalize_to_one(settings):
    w = settings.scoring.normalized_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in w.values())


def test_exit_threshold_must_be_below_entry():
    with pytest.raises(ValidationError):
        Settings(scoring={"entry_threshold": 50, "exit_threshold": 60})


def test_negative_weights_rejected():
    with pytest.raises(ValidationError):
        Settings(scoring={"weights": {"technical": -0.5, "macro": 0.5}})


def test_universe_loads_and_derives_upstox_key():
    uni = load_universe()
    assert len(uni) >= 50
    rel = universe_map()["RELIANCE"]
    assert rel.yf_ticker == "RELIANCE.NS"
    assert rel.resolved_upstox_key == "NSE_EQ|INE002A01018"


def test_blank_isin_gives_blank_upstox_key():
    s = Stock("FOO", "IT")
    assert s.resolved_upstox_key == ""
