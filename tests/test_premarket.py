import datetime as dt

import pandas as pd

from alerting.formatter import format_morning_brief
from analyzers.context import MarketContext
from config.loader import load_settings
from scheduler.run_eod import market_cues


def test_morning_brief_groups_ok_and_warn():
    reviews = [
        {"symbol": "RELIANCE", "sector": "Oil & Gas", "composite": 72, "status": "ok", "note": "stable"},
        {"symbol": "TCS", "sector": "IT", "composite": 40, "status": "warn",
         "note": "score fell to 40; negative overnight news (-0.5)"},
    ]
    msg = format_morning_brief(dt.date(2026, 6, 15), "S&P 500 +0.8% · India VIX 14.0", reviews)
    assert "Morning brief" in msg
    assert "Still good" in msg and "RELIANCE" in msg
    assert "Reconsider" in msg and "TCS" in msg
    assert "no new buy signals" in msg.lower()


def test_morning_brief_empty():
    msg = format_morning_brief(dt.date(2026, 6, 15), "Crude +1%", [])
    assert "No active positions" in msg


def test_market_cues_summary():
    s = load_settings()
    idx = pd.date_range("2026-06-01", periods=2, freq="B")
    mctx = MarketContext(
        as_of=dt.date(2026, 6, 15), settings=s,
        macro={
            "sp500": pd.Series([5000, 5100], index=idx),
            "nasdaq": pd.Series([16000, 15800], index=idx),
            "crude": pd.Series([80, 82], index=idx),
            "usdinr": pd.Series([83.0, 83.2], index=idx),
        },
    )
    cues = market_cues(mctx, {"vix": 13.5})
    assert "S&P 500 +2.0%" in cues
    assert "Nasdaq -1.2%" in cues
    assert "India VIX 13.5" in cues
