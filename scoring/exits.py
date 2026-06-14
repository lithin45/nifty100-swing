"""EXIT logic for open positions.

Priority order (first match wins):
  1. target hit           (today's high >= target)
  2. hard stop hit        (today's low <= stop)
  3. trailing stop hit    (ratcheted ATR/%-stop, only after +activate_at_r)
  4. time exit            (held >= max_holding_days)
  5. signal decay         (composite < exit_threshold)
  6. trend reversal       (close < trend SMA, or bearish MACD cross)
  7. sector rollover      (sector RS turned negative)

:func:`evaluate_exit` always returns an :class:`ExitDecision` carrying the
updated trailing-stop bookkeeping, so the caller persists the ratchet even when
no exit fires.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from analyzers import indicators as I
from analyzers.context import MarketContext
from common.calendar_nse import trading_days_until
from common.types import ExitReason


@dataclass
class ExitDecision:
    should_exit: bool
    reason: Optional[ExitReason]
    price: float                 # suggested exit / reference price
    new_highest_close: float     # ratcheted high-water close
    new_current_stop: float      # ratcheted trailing stop
    pnl_pct: float
    holding_days: int
    detail: str = ""


def _last(series: pd.Series) -> float:
    s = series.dropna() if series is not None else None
    return float(s.iloc[-1]) if s is not None and len(s) else math.nan


def evaluate_exit(
    position,
    price_df: pd.DataFrame,
    composite_score: Optional[float],
    mctx: MarketContext,
) -> ExitDecision:
    """Decide whether to exit ``position`` given fresh prices + current composite.

    ``position`` is anything with the attributes: entry_price, stop_loss, target,
    atr, highest_close, current_stop, entry_date, sector (e.g. a storage Position).
    """
    settings = mctx.settings
    risk = settings.risk
    exits_cfg = settings.exits

    entry = float(position.entry_price)
    last_close = _last(price_df["close"]) if len(price_df) else entry
    last_high = _last(price_df["high"]) if len(price_df) else last_close
    last_low = _last(price_df["low"]) if len(price_df) else last_close
    holding_days = (mctx.as_of - position.entry_date).days          # calendar (for display)
    trading_held = trading_days_until(mctx.as_of, position.entry_date)  # NSE trading bars held

    # Ratchet the high-water close.
    new_high = max(float(position.highest_close or entry), last_close)

    # Recompute current ATR for the trailing stop.
    atr_now = _last(I.atr(price_df, settings.technical.atr_period)) if len(price_df) >= settings.technical.atr_period else (position.atr or 0.0)

    # Trailing stop (only ratchets up, only after activation profit).
    cur_stop = float(position.current_stop or position.stop_loss)
    risk_per_share = entry - float(position.stop_loss)
    activate_level = entry + risk.trailing.activate_at_r * risk_per_share
    new_stop = cur_stop
    if risk.trailing.enabled and new_high >= activate_level:
        if risk.trailing.type == "percent":
            trail = new_high * (1.0 - risk.trailing.percent / 100.0)
        else:  # atr
            trail = new_high - risk.trailing.atr_multiple * (atr_now or 0.0)
        new_stop = max(cur_stop, trail)

    def _decide(reason: ExitReason, price: float, detail: str) -> ExitDecision:
        pnl = (price - entry) / entry * 100.0 if entry else 0.0
        return ExitDecision(True, reason, round(price, 2), new_high, round(new_stop, 2),
                            round(pnl, 2), holding_days, detail)

    # 1. target
    if position.target and last_high >= float(position.target):
        return _decide(ExitReason.TARGET_HIT, float(position.target),
                       f"high {last_high:.1f} reached target {position.target:.1f}")
    # 2. hard stop
    if last_low <= float(position.stop_loss):
        return _decide(ExitReason.STOP_HIT, float(position.stop_loss),
                       f"low {last_low:.1f} hit stop {position.stop_loss:.1f}")
    # 3. trailing stop
    if risk.trailing.enabled and new_stop > float(position.stop_loss) and last_low <= new_stop:
        return _decide(ExitReason.TRAILING_STOP, new_stop,
                       f"low {last_low:.1f} hit trailing stop {new_stop:.1f}")
    # 4. time exit (compared in trading days, matching the "~1 trading month" config)
    if trading_held >= risk.max_holding_days:
        return _decide(ExitReason.TIME_EXIT, last_close,
                       f"held {trading_held} trading days >= max {risk.max_holding_days}")
    # 5. signal decay
    if composite_score is not None and composite_score < exits_cfg.signal_decay_threshold:
        return _decide(ExitReason.SIGNAL_DECAY, last_close,
                       f"composite {composite_score:.0f} < exit {exits_cfg.signal_decay_threshold:.0f}")
    # 6. trend reversal
    if exits_cfg.trend_reversal_exit and len(price_df) >= exits_cfg.trend_reversal_sma:
        sma = _last(I.sma(price_df["close"], exits_cfg.trend_reversal_sma))
        if not math.isnan(sma) and last_close < sma:
            return _decide(ExitReason.TREND_REVERSAL, last_close,
                           f"close {last_close:.1f} < {exits_cfg.trend_reversal_sma}-DMA {sma:.1f}")
    if exits_cfg.macd_reversal_exit and len(price_df) >= 35:
        macd_df = I.macd(price_df["close"], settings.technical.macd.fast,
                         settings.technical.macd.slow, settings.technical.macd.signal)
        if bool(I.crossed_below(macd_df["macd"], macd_df["signal"]).tail(2).any()):
            return _decide(ExitReason.TREND_REVERSAL, last_close, "bearish MACD cross")
    # 7. sector rollover
    if exits_cfg.sector_rollover_exit:
        rs = mctx.sector_rs.get(getattr(position, "sector", ""))
        if rs is not None and rs < -0.2:
            return _decide(ExitReason.SECTOR_ROLLOVER, last_close,
                           f"sector RS {rs:+.2f} rolled over")

    # No exit — return the (possibly ratcheted) bookkeeping for persistence.
    pnl = (last_close - entry) / entry * 100.0 if entry else 0.0
    return ExitDecision(False, None, round(last_close, 2), new_high, round(new_stop, 2),
                        round(pnl, 2), holding_days, "hold")
