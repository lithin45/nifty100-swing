import datetime as dt

import numpy as np
import pandas as pd

from analyzers.context import MarketContext, StockContext
from scoring.gates import run_gates, trend_gate, market_regime_gate, liquidity_gate, event_gate


def _mctx(settings, asof, vix=14.0, bench_uptrend=True, sector_rs=None):
    idx = pd.date_range("2025-06-02", periods=260, freq="B")
    bench = pd.Series(np.linspace(20000, 24000, 260) if bench_uptrend
                      else np.linspace(24000, 20000, 260), index=idx)
    return MarketContext(
        as_of=asof, settings=settings,
        vix=pd.Series([vix], index=idx[-1:]),
        benchmark=bench, sector_rs=sector_rs or {},
    )


def test_all_gates_pass_for_strong_stock(settings, asof, strong_uptrend_df, reliance):
    mctx = _mctx(settings, asof)
    sctx = StockContext(reliance, strong_uptrend_df)
    report = run_gates(sctx, mctx)
    assert report.passed, report.summary()


def test_trend_gate_blocks_downtrend(settings, asof, downtrend_df, reliance):
    mctx = _mctx(settings, asof, sector_rs={"Oil & Gas": -0.5})
    sctx = StockContext(reliance, downtrend_df)
    assert not trend_gate(sctx, mctx).passed


def test_trend_gate_sector_override(settings, asof, downtrend_df, reliance):
    mctx = _mctx(settings, asof, sector_rs={"Oil & Gas": 0.4})  # sector outperforming
    sctx = StockContext(reliance, downtrend_df)
    assert trend_gate(sctx, mctx).passed


def test_market_regime_blocks_high_vix(settings, asof, reliance, strong_uptrend_df):
    mctx = _mctx(settings, asof, vix=30.0)  # above ceiling 22
    sctx = StockContext(reliance, strong_uptrend_df)
    assert not market_regime_gate(sctx, mctx).passed


def test_liquidity_gate_blocks_illiquid(settings, asof, reliance, make_df):
    thin = make_df(np.linspace(100, 175, 260), vols=np.full(260, 100.0))  # tiny volume
    mctx = _mctx(settings, asof)
    assert not liquidity_gate(StockContext(reliance, thin), mctx).passed


def test_event_gate_blocks_imminent_results(settings, asof, reliance, strong_uptrend_df):
    mctx = _mctx(settings, asof)
    sctx = StockContext(reliance, strong_uptrend_df, earnings_date=asof + dt.timedelta(days=2))
    assert not event_gate(sctx, mctx).passed
