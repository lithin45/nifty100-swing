"""Walk-forward validation.

Rolls in-sample (IS) / out-of-sample (OOS) windows across the history and runs
the backtest on each OOS slice, then stitches the OOS equity curves into one
combined out-of-sample track record. Because the live rules use fixed config
(no per-window curve-fitting), the IS window is where you'd hook parameter
optimisation; here it acts as the warm-up/context and OOS is what's scored —
which is the honest, look-ahead-free way to read these results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from dateutil.relativedelta import relativedelta

from backtest.engine import BacktestResult, run_backtest
from backtest.metrics import compute_metrics, plain_english_summary
from common.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class WindowResult:
    is_start: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    result: BacktestResult


@dataclass
class WalkForwardResult:
    windows: list[WindowResult] = field(default_factory=list)
    combined_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    combined_metrics: dict = field(default_factory=dict)
    summary: str = ""


def _bounds(prices: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts, ends = [], []
    for df in prices.values():
        if df is not None and len(df):
            starts.append(df.index.min())
            ends.append(df.index.max())
    return min(starts), max(ends)


def walk_forward(prices: dict[str, pd.DataFrame], settings) -> WalkForwardResult:
    wf = settings.backtest.walkforward
    if not wf.enabled or not prices:
        res = run_backtest(prices, settings)
        return WalkForwardResult([], res.equity, res.metrics, res.summary)

    data_start, data_end = _bounds(prices)
    cfg_start = pd.Timestamp(settings.backtest.start)
    start = max(data_start, cfg_start)

    windows: list[WindowResult] = []
    oos_returns: list[pd.Series] = []

    is_start = start
    while True:
        oos_start = is_start + relativedelta(months=wf.in_sample_months)
        oos_end = oos_start + relativedelta(months=wf.out_sample_months)
        if oos_start >= data_end:
            break
        oos_end = min(oos_end, data_end)

        # IS context begins at is_start so 200-DMA etc. are warm by oos_start.
        res = run_backtest(prices, settings, start=str(is_start.date()), end=str(oos_end.date()))
        # Keep only the OOS slice of the equity curve for stitching.
        if len(res.equity):
            oos_eq = res.equity[res.equity.index >= oos_start]
            if len(oos_eq) >= 2:
                oos_returns.append(oos_eq.pct_change().dropna())
        windows.append(WindowResult(is_start, oos_start, oos_end, res))

        is_start = is_start + relativedelta(months=wf.step_months)

    # Stitch OOS returns into one compounded equity curve.
    combined_equity = pd.Series(dtype=float)
    combined_metrics: dict = {"trades": 0}
    if oos_returns:
        stitched = pd.concat(oos_returns).sort_index()
        stitched = stitched[~stitched.index.duplicated(keep="first")]
        combined_equity = (1.0 + stitched).cumprod() * settings.backtest.initial_capital
        all_trades = pd.concat(
            [w.result.trades for w in windows if len(w.result.trades)], ignore_index=True
        ) if any(len(w.result.trades) for w in windows) else pd.DataFrame()
        combined_metrics = compute_metrics(combined_equity, all_trades)

    return WalkForwardResult(
        windows=windows,
        combined_equity=combined_equity,
        combined_metrics=combined_metrics,
        summary="Walk-forward (out-of-sample) results:\n" + plain_english_summary(combined_metrics),
    )
