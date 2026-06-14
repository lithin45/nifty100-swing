"""Backtest engine.

Primary path is a native, dependency-light event-driven simulator that mirrors
the live rules:

* **Entry signal** is the price-computable core of the live system (the bits that
  exist point-in-time historically): above the 200-DMA, an N-day breakout with
  volume confirmation, and RSI in a healthy band. (Fundamentals / news / FII are
  NOT replayed — we don't have reliable point-in-time history for them — so the
  backtest validates the *technical engine*; this is stated plainly in the README.)
* **Execution** decides on the close and fills at the next open (no look-ahead).
* **Exits**: ATR target, hard stop, ratcheting trailing stop, time stop, and a
  trend-reversal (close below the 50-DMA).
* **Portfolio**: caps concurrent positions, sizes each by % risk, and charges the
  full Indian round-trip cost stack at exit.

An optional :func:`run_backtest_vectorbt` is provided for those who install
vectorbt and want its portfolio analytics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from analyzers import indicators as I
from backtest.costs import compute_trade_costs
from backtest.metrics import compute_metrics, plain_english_summary
from common.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict = field(default_factory=dict)
    summary: str = ""


def _entry_signal(df: pd.DataFrame, settings) -> pd.Series:
    """Vectorised technical entry condition (decided on the bar's close)."""
    t = settings.technical
    close, vol = df["close"], df["volume"]
    sma200 = I.sma(close, 200)
    rsi = I.rsi(close, t.rsi_period)
    recent_high = I.rolling_high(close, t.breakout.lookback_high)
    avg_vol = I.sma(vol, t.breakout.volume_avg_period)

    above_trend = close > sma200
    breakout = close > recent_high
    vol_ok = vol > t.breakout.volume_multiple * avg_vol
    momentum_ok = (rsi > 50) & (rsi < 78)
    sig = above_trend & breakout & vol_ok & momentum_ok
    return sig.fillna(False)


def generate_trades(df: pd.DataFrame, settings, symbol: str = "",
                    regime_ok: Optional[pd.Series] = None) -> list[dict]:
    """Generate non-overlapping trades for one symbol (one position at a time).

    ``regime_ok`` (optional, date-indexed bool) gates entries to days when the
    broad market is healthy — mirroring the live market-regime gate.
    """
    if df is None or len(df) < 220:
        return []
    df = df.sort_index()
    risk = settings.risk
    exits_cfg = settings.exits

    sig = _entry_signal(df, settings)
    if regime_ok is not None:
        sig = sig & regime_ok.reindex(df.index).ffill().fillna(False)
    sig = sig.to_numpy()
    atr = I.atr(df, settings.technical.atr_period).to_numpy()
    sma50 = I.sma(df["close"], exits_cfg.trend_reversal_sma).to_numpy()
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    idx = df.index
    n = len(df)

    trades: list[dict] = []
    i = 0
    while i < n - 1:
        if not sig[i] or np.isnan(atr[i]) or atr[i] <= 0:
            i += 1
            continue

        e = i + 1  # execute next open
        entry_price = o[e]
        if entry_price <= 0:
            i += 1
            continue
        entry_date = idx[e]
        stop0 = entry_price - risk.atr_stop_multiple * atr[i]
        if stop0 <= 0 or stop0 >= entry_price:
            i += 1
            continue
        risk_ps = entry_price - stop0
        target = entry_price + risk.rr_target_multiple * risk_ps
        highest = entry_price
        cur_stop = stop0

        exit_price = c[-1]
        exit_date = idx[-1]
        reason = "eod"
        j = e
        while j < n:
            # ratchet trailing stop
            highest = max(highest, c[j])
            if risk.trailing.enabled and highest >= entry_price + risk.trailing.activate_at_r * risk_ps:
                if risk.trailing.type == "percent":
                    trail = highest * (1.0 - risk.trailing.percent / 100.0)
                else:
                    a = atr[j] if not np.isnan(atr[j]) else 0.0
                    trail = highest - risk.trailing.atr_multiple * a
                cur_stop = max(cur_stop, trail)

            # gap handling at open, then conservative intrabar (stop before target)
            if o[j] >= target:
                exit_price, exit_date, reason = o[j], idx[j], "target"; break
            if o[j] <= cur_stop:
                exit_price, exit_date, reason = o[j], idx[j], ("trailing" if cur_stop > stop0 else "stop"); break
            if low[j] <= cur_stop:
                exit_price, exit_date, reason = cur_stop, idx[j], ("trailing" if cur_stop > stop0 else "stop"); break
            if h[j] >= target:
                exit_price, exit_date, reason = target, idx[j], "target"; break
            if (j - e) >= risk.max_holding_days:  # bars held = trading days
                exit_price, exit_date, reason = c[j], idx[j], "time"; break
            if exits_cfg.trend_reversal_exit and not np.isnan(sma50[j]) and c[j] < sma50[j] and j > e:
                exit_price, exit_date, reason = c[j], idx[j], "trend"; break
            j += 1

        trades.append({
            "symbol": symbol,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": round(float(entry_price), 2),
            "exit_price": round(float(exit_price), 2),
            "stop_distance_pct": round(risk_ps / entry_price * 100.0, 3),
            "exit_reason": reason,
            "bars_held": int(min(j, n - 1) - e),
        })
        i = min(j, n - 1) + 1  # no re-entry before the exit bar

    return trades


def run_backtest(
    prices: dict[str, pd.DataFrame],
    settings,
    start: Optional[str] = None,
    end: Optional[str] = None,
    regime_filter: bool = True,
    benchmark: Optional[pd.Series] = None,
) -> BacktestResult:
    """Portfolio backtest across many symbols with position cap + costs.

    When ``regime_filter`` is True, entries are only allowed while the broad
    market is above its 200-DMA — mirroring the live market-regime gate. The
    market proxy is ``benchmark`` if supplied, else an equal-weight index built
    from the universe (offline-safe).
    """
    bt = settings.backtest
    start = start or bt.start
    end = end or bt.end

    # 1a) filter each symbol to [start, end]; collect closes for the panel.
    data: dict[str, pd.DataFrame] = {}
    close_panel: dict[str, pd.Series] = {}
    for sym, df in prices.items():
        if df is None or len(df) == 0:
            continue
        d = df.sort_index()
        if start:
            d = d[d.index >= pd.Timestamp(start)]
        if end:
            d = d[d.index <= pd.Timestamp(end)]
        if len(d) < 220:
            continue
        data[sym] = d
        close_panel[sym] = d["close"]

    if not close_panel:
        return BacktestResult(pd.DataFrame(), pd.Series(dtype=float), {"trades": 0},
                              "No symbols had enough history in this period.")

    panel = pd.DataFrame(close_panel).sort_index().ffill()

    # 1b) Market-regime proxy: only allow entries while the broad market is above
    #     its 200-DMA. Uses `benchmark` if given, else an equal-weight index of
    #     the universe (each stock normalised to its first close, then averaged).
    regime_ok = None
    if regime_filter:
        if benchmark is not None and len(benchmark):
            bench = benchmark.reindex(panel.index).ffill()
        else:
            firsts = {c: panel[c].loc[panel[c].first_valid_index()]
                      for c in panel.columns if panel[c].first_valid_index() is not None}
            bench = panel[list(firsts)].divide(pd.Series(firsts)).mean(axis=1)
        regime_ok = (bench > I.sma(bench, 200)).fillna(False)

    # 1c) generate per-symbol trades (entries gated by the regime filter).
    all_trades: list[dict] = []
    for sym, d in data.items():
        all_trades.extend(generate_trades(d, settings, sym, regime_ok=regime_ok))

    if not all_trades:
        return BacktestResult(pd.DataFrame(), pd.Series(dtype=float), {"trades": 0},
                              "No trades were generated in this period.")

    trades_df = pd.DataFrame(all_trades).sort_values("entry_date").reset_index(drop=True)

    # 2) chronological portfolio simulation with daily mark-to-market
    capital = float(bt.initial_capital)
    cash = capital
    max_pos = bt.max_open_positions
    max_size = settings.risk.max_position_pct / 100.0
    risk_per_trade = settings.risk.risk_per_trade_pct / 100.0

    entries_by_date: dict[pd.Timestamp, list[int]] = {}
    exits_by_date: dict[pd.Timestamp, list[int]] = {}
    for k, row in trades_df.iterrows():
        entries_by_date.setdefault(row["entry_date"], []).append(k)
        exits_by_date.setdefault(row["exit_date"], []).append(k)

    open_positions: dict[int, dict] = {}  # trade idx -> {qty, entry_price, alloc}
    trade_net_return: dict[int, float] = {}
    equity_curve: dict[pd.Timestamp, float] = {}

    def _close(k: int) -> None:
        nonlocal cash
        pos = open_positions.pop(k, None)
        if pos is None:
            return
        row = trades_df.loc[k]
        buy_val = pos["qty"] * pos["entry_price"]
        sell_val = pos["qty"] * row["exit_price"]
        cb = compute_trade_costs(buy_val, sell_val, settings.costs)
        net = sell_val - cb.total
        cash += net
        trade_net_return[k] = (net - buy_val) / pos["alloc"] * 100.0 if pos["alloc"] else 0.0

    for day in panel.index:
        # 1) exits for positions already open (free up slots/cash)
        for k in exits_by_date.get(day, []):
            _close(k)

        # 2) entries
        for k in entries_by_date.get(day, []):
            if len(open_positions) >= max_pos:
                continue
            row = trades_df.loc[k]
            equity_now = cash + sum(
                p["qty"] * _row_price(panel, k2, day, trades_df)
                for k2, p in open_positions.items()
            )
            stop_frac = row["stop_distance_pct"] / 100.0
            size_frac = min(max_size, risk_per_trade / stop_frac if stop_frac > 0 else 0.0)
            alloc = size_frac * equity_now
            if alloc > cash:
                alloc = cash
            if alloc <= 0 or row["entry_price"] <= 0:
                continue
            qty = alloc / row["entry_price"]
            cash -= alloc
            open_positions[k] = {"qty": qty, "entry_price": row["entry_price"], "alloc": alloc}

        # 3) same-bar trades: a position opened today whose exit is also today
        #    (gap/intrabar stop or target on the entry bar). Without this pass
        #    these positions would leak cash and never close.
        for k in exits_by_date.get(day, []):
            if k in open_positions:
                _close(k)

        # mark-to-market
        mtm = cash + sum(p["qty"] * _row_price(panel, k, day, trades_df)
                         for k, p in open_positions.items())
        equity_curve[day] = mtm

    equity = pd.Series(equity_curve).sort_index()
    trades_df["return_pct"] = trades_df.index.map(lambda k: round(trade_net_return.get(k, np.nan), 2))
    trades_df = trades_df.dropna(subset=["return_pct"]).reset_index(drop=True)

    metrics = compute_metrics(equity, trades_df)
    summary = plain_english_summary(metrics)
    return BacktestResult(trades_df, equity, metrics, summary)


def _row_price(panel: pd.DataFrame, trade_idx: int, day, trades_df: pd.DataFrame) -> float:
    sym = trades_df.loc[trade_idx, "symbol"]
    try:
        val = panel.at[day, sym]
        return float(val) if pd.notna(val) else float(trades_df.loc[trade_idx, "entry_price"])
    except (KeyError, TypeError):
        return float(trades_df.loc[trade_idx, "entry_price"])


def run_backtest_vectorbt(prices: dict[str, pd.DataFrame], settings):
    """Optional: portfolio backtest via vectorbt (returns its Portfolio object)."""
    try:
        import vectorbt as vbt
    except Exception:
        log.warning("vectorbt not installed; use run_backtest() instead")
        return None

    closes, entries = {}, {}
    for sym, df in prices.items():
        if df is None or len(df) < 220:
            continue
        d = df.sort_index()
        closes[sym] = d["close"]
        entries[sym] = _entry_signal(d, settings)
    if not closes:
        return None

    close = pd.DataFrame(closes)
    entry = pd.DataFrame(entries).reindex_like(close).fillna(False)
    cost_frac = settings.costs.slippage_pct / 100.0
    sl = settings.risk.atr_stop_multiple * 0.02  # rough ATR%->stop% proxy
    tp = sl * settings.risk.rr_target_multiple
    return vbt.Portfolio.from_signals(
        close, entries=entry, exits=None,
        sl_stop=sl, tp_stop=tp,
        fees=cost_frac, slippage=cost_frac,
        init_cash=settings.backtest.initial_capital, freq="1D",
    )
