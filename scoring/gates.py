"""Stage 1 — HARD GATES. All enabled gates must pass or no signal is generated.

Gates are deliberately conservative: when required market data is genuinely
unavailable a gate *passes with a note* rather than silently blocking the entire
universe — but a clear negative (price below 200-DMA, VIX above ceiling, results
imminent, F&O ban) always blocks.
"""
from __future__ import annotations

import math

import pandas as pd

from analyzers import indicators as I
from analyzers.context import MarketContext, StockContext
from common.calendar_nse import trading_days_until
from common.types import GateReport, GateResult


def _last(series: pd.Series) -> float:
    s = series.dropna() if series is not None else None
    return float(s.iloc[-1]) if s is not None and len(s) else math.nan


def liquidity_gate(sctx: StockContext, mctx: MarketContext) -> GateResult:
    cfg = mctx.settings.gates.liquidity
    df = sctx.price
    if df is None or len(df) == 0:
        return GateResult("liquidity", False, "no price data")
    turnover = (df["close"] * df["volume"]).tail(cfg.lookback_days).mean()
    turnover = float(turnover) if pd.notna(turnover) else 0.0
    passed = turnover >= cfg.min_avg_turnover_inr
    return GateResult(
        "liquidity", passed,
        f"avg turnover ₹{turnover/1e7:.1f} cr vs min ₹{cfg.min_avg_turnover_inr/1e7:.1f} cr",
        {"avg_turnover_inr": turnover},
    )


def trend_gate(sctx: StockContext, mctx: MarketContext) -> GateResult:
    cfg = mctx.settings.gates.trend
    df = sctx.price
    if df is None or len(df) < cfg.require_above_sma:
        return GateResult("trend", False, "insufficient history for trend filter")
    close = df["close"]
    sma = _last(I.sma(close, cfg.require_above_sma))
    last = _last(close)
    if not math.isnan(sma) and last > sma:
        return GateResult("trend", True, f"price {last:.1f} > {cfg.require_above_sma}-DMA {sma:.1f}")
    if cfg.allow_sector_rs_override:
        rs = mctx.sector_rs.get(sctx.sector)
        if rs is not None and rs > 0:
            return GateResult("trend", True,
                              f"below {cfg.require_above_sma}-DMA but sector RS {rs:+.2f} positive",
                              {"sector_rs": rs})
    return GateResult("trend", False, f"price {last:.1f} below {cfg.require_above_sma}-DMA {sma:.1f}")


def event_gate(sctx: StockContext, mctx: MarketContext) -> GateResult:
    cfg = mctx.settings.gates.event
    if cfg.block_fno_ban and sctx.symbol in mctx.fno_ban:
        return GateResult("event", False, "stock in F&O ban period")
    ed = sctx.earnings_date
    if ed is not None:
        days = trading_days_until(ed, mctx.as_of)
        if days is not None and 0 <= days <= cfg.no_entry_days_before_earnings:
            return GateResult("event", False, f"results in {days} trading day(s)",
                              {"earnings_date": ed.isoformat()})
    return GateResult("event", True, "no imminent results / not in F&O ban")


def market_regime_gate(sctx: StockContext, mctx: MarketContext) -> GateResult:
    cfg = mctx.settings.gates.market_regime
    notes: list[str] = []

    vix = _last(mctx.vix) if mctx.vix is not None else math.nan
    if not math.isnan(vix):
        if vix > cfg.vix_ceiling:
            return GateResult("market_regime", False,
                              f"India VIX {vix:.1f} above ceiling {cfg.vix_ceiling}",
                              {"vix": vix})
        notes.append(f"VIX {vix:.1f} ok")
    else:
        notes.append("VIX unavailable")

    bench = mctx.benchmark
    if bench is not None and len(bench.dropna()) >= cfg.require_index_above_sma:
        sma = _last(I.sma(bench, cfg.require_index_above_sma))
        last = _last(bench)
        if not math.isnan(sma) and last < sma:
            return GateResult("market_regime", False,
                              f"Nifty 100 {last:.0f} below {cfg.require_index_above_sma}-DMA {sma:.0f}",
                              {"index": last, "index_sma": sma})
        notes.append(f"index > {cfg.require_index_above_sma}-DMA")
    else:
        notes.append("benchmark unavailable")

    return GateResult("market_regime", True, "; ".join(notes))


_GATES = {
    "liquidity": liquidity_gate,
    "trend": trend_gate,
    "event": event_gate,
    "market_regime": market_regime_gate,
}


def run_gates(sctx: StockContext, mctx: MarketContext) -> GateReport:
    """Run all *enabled* gates and return a combined report."""
    gates_cfg = mctx.settings.gates
    report = GateReport()
    for name, fn in _GATES.items():
        if getattr(getattr(gates_cfg, name), "enabled", True):
            report.results.append(fn(sctx, mctx))
    return report
