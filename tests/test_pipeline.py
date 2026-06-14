"""Integration test: run the full EOD pipeline with fake data providers."""
from __future__ import annotations

import pandas as pd

import scheduler.run_eod as r
from storage import db


class _Px:
    def __init__(self, df):
        self.df = df

    def get_history(self, symbol, start=None, end=None):
        return self.df


class _Vix:
    def get_history(self, lookback_days=60):
        idx = pd.date_range("2026-01-01", periods=5, freq="B")
        return pd.Series([14, 13.5, 13, 12.8, 12.5], index=idx)


class _Macro:
    def get_snapshot(self, lookback_days=60):
        return {}


class _Fii:
    def get_recent(self, lookback_days=5):
        return pd.DataFrame({"fii_net": [1500, 1200, 900], "dii_net": [200, 300, 100]})


class _Sector:
    """Benchmark rises moderately; sector indices rise more -> positive RS."""
    def get_index_close(self, ticker, lookback_days=90):
        import numpy as np

        idx = pd.date_range("2025-06-02", periods=lookback_days + 5, freq="B")
        slope = 0.10 if ticker == "^CNX100" else 0.25
        return pd.Series(100 * (1 + slope) ** (np.arange(len(idx)) / len(idx)), index=idx)


class _Events:
    def get_earnings_date(self, symbol):
        return None

    def get_fno_ban_list(self):
        return set()


class _Fund:
    def get_fundamentals(self, symbol):
        return {"roe": 25, "pe": 22, "de": 0.2, "earnings_growth": 18, "ps": 5}


class _News:
    def get_headlines(self, max_age_days=7):
        return [{"title": "Reliance Industries profit jumps, wins record order", "summary": ""}]


class _Validator:
    def validate(self, symbol, df, last_n=3, repair=True):
        return df, None


def test_run_eod_generates_persists_buy(monkeypatch, tmp_path, strong_uptrend_df):
    dbfile = str(tmp_path / "pipe.db")
    monkeypatch.setattr("config.loader.get_db_path", lambda: dbfile)

    monkeypatch.setattr(r, "get_price_provider", lambda settings=None: _Px(strong_uptrend_df))
    monkeypatch.setattr(r, "get_vix_provider", lambda: _Vix())
    monkeypatch.setattr(r, "get_macro_provider", lambda settings=None: _Macro())
    monkeypatch.setattr(r, "get_fii_dii_provider", lambda: _Fii())
    monkeypatch.setattr(r, "get_sector_provider", lambda: _Sector())
    monkeypatch.setattr(r, "get_events_provider", lambda: _Events())
    monkeypatch.setattr(r, "get_fundamentals_provider", lambda: _Fund())
    monkeypatch.setattr(r, "get_news_provider", lambda settings=None: _News())
    monkeypatch.setattr(r, "BhavcopyValidator", lambda *a, **k: _Validator())

    result = r.run_eod(limit=1, send=False, date_str="2026-06-12")

    assert result["buy"] >= 1, result
    signals = db.recent_signals(10, db_path=dbfile)
    assert any(s.action == "BUY" and s.symbol == "RELIANCE" for s in signals)
    # sub-scores persisted
    buy = next(s for s in signals if s.action == "BUY")
    assert len(buy.sub_scores) == 8
    # position opened
    assert "RELIANCE" in {p.symbol for p in db.get_open_positions(db_path=dbfile)}
    # run finished cleanly
    run = db.latest_run(db_path=dbfile)
    assert run.status == "success" and run.n_buy >= 1
