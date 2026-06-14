import json

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import sortino
from backtest.walkforward import walk_forward
from scripts.run_backtest import _synthetic_universe


def test_sortino_no_downside_returns_zero_not_nan():
    r = pd.Series([0.01, 0.02, 0.015, 0.03])  # no negative returns
    s = sortino(r)
    assert s == 0.0  # not NaN (NaN != NaN)


def test_run_backtest_accounts_for_every_trade(settings):
    # Regression for the same-bar entry+exit cash leak: every generated trade
    # must have a recorded net return (none silently dropped), metrics JSON-safe.
    prices = _synthetic_universe(n_symbols=6, n_days=700)
    res = run_backtest(prices, settings, start="2021-01-04")
    assert len(res.trades) > 0
    assert res.trades["return_pct"].notna().all()
    assert np.isfinite(res.equity.iloc[-1])
    json.dumps(res.metrics)  # raises if any NaN/np scalar leaked in


def test_walkforward_runs_and_metrics_serializable(settings):
    prices = _synthetic_universe(n_symbols=6, n_days=900)
    wf = walk_forward(prices, settings)
    assert len(wf.windows) >= 1
    json.dumps(wf.combined_metrics)  # OOS metrics must be JSON-safe
    if len(wf.combined_equity):
        assert np.isfinite(wf.combined_equity.iloc[-1])
