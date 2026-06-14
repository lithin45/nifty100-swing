"""Technical analyzer: indicators + patterns -> a single ``technical`` SubScore.

Sub-checks (trend / momentum / breakout / volume / patterns) each yield a [0,1]
score and a phrase; they combine via ``technical.weights`` from settings. The
indicator snapshot is stashed in ``SubScore.details['indicators']`` so the signal
generator can reuse ATR, the breakout level and the pattern target without
recomputing.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from analyzers import indicators as I
from analyzers import patterns as P
from analyzers.context import MarketContext, StockContext
from common.types import SubScore


def _last(series: pd.Series) -> float:
    """Last finite value or NaN."""
    if series is None or len(series) == 0:
        return math.nan
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else math.nan


def _ok(x: float) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _rsi_score(r: float) -> float:
    """Map RSI to [0,1]: rewards 50–65, penalises overbought >75 and weak <50."""
    if not _ok(r):
        return 0.5
    if r <= 50:
        return max(0.0, r / 50.0 * 0.5)
    if r <= 65:
        return 0.5 + (r - 50) / 15.0 * 0.5
    if r <= 75:
        return 1.0 - (r - 65) / 10.0 * 0.3
    return max(0.2, 0.7 - (r - 75) / 25.0 * 0.5)


def _higher_highs_lows(close: pd.Series, window: int = 40) -> bool:
    if len(close) < window:
        return False
    recent = close.iloc[-window // 2 :]
    older = close.iloc[-window : -window // 2]
    return recent.max() > older.max() and recent.min() > older.min()


class TechnicalAnalyzer:
    key = "technical"

    def analyze(self, sctx: StockContext, mctx: MarketContext) -> SubScore:
        s = mctx.settings.technical
        df = sctx.price
        if df is None or len(df) < 30:
            return SubScore(self.key, 0.5, "Insufficient price history for technicals")

        close = df["close"]
        volume = df["volume"]

        # ---- indicator series ----
        sma20 = I.sma(close, 20)
        sma50 = I.sma(close, 50)
        sma200 = I.sma(close, 200)
        ema20 = I.ema(close, s.ema_periods[0] if s.ema_periods else 20)
        rsi = I.rsi(close, s.rsi_period)
        macd_df = I.macd(close, s.macd.fast, s.macd.slow, s.macd.signal)
        atr = I.atr(df, s.atr_period)
        stoch = I.stochastic(df, s.stoch.k, s.stoch.d, s.stoch.smooth)
        adx = I.adx(df, 14)
        bb = I.bollinger(close, s.bollinger.period, s.bollinger.std)
        recent_high = _last(I.rolling_high(close, s.breakout.lookback_high))
        avg_vol = _last(I.sma(volume, s.breakout.volume_avg_period))
        last_close = _last(close)
        last_vol = _last(volume)
        vol_ratio = (last_vol / avg_vol) if (_ok(avg_vol) and avg_vol) else math.nan

        v_sma20, v_sma50, v_sma200 = _last(sma20), _last(sma50), _last(sma200)
        v_rsi = _last(rsi)
        v_macd, v_sig, v_hist = _last(macd_df["macd"]), _last(macd_df["signal"]), _last(macd_df["hist"])
        v_k, v_d = _last(stoch["k"]), _last(stoch["d"])
        v_adx = _last(adx["adx"])

        # ---- TREND ----
        checks: list[tuple[float, bool]] = []
        if _ok(v_sma200):
            checks.append((0.25, last_close > v_sma200))
        if _ok(v_sma50):
            checks.append((0.20, last_close > v_sma50))
        if _ok(v_sma20):
            checks.append((0.15, last_close > v_sma20))
        if _ok(v_sma50) and _ok(v_sma200):
            checks.append((0.20, v_sma50 > v_sma200))  # golden-cross regime
        checks.append((0.20, _higher_highs_lows(close)))
        tw = sum(w for w, _ in checks) or 1.0
        trend = sum(w for w, ok in checks if ok) / tw

        # ---- MOMENTUM ----
        rising_rsi = _ok(v_rsi) and v_rsi > _last(rsi.shift(1))
        macd_bull = _ok(v_hist) and v_hist > 0 and v_macd > v_sig
        macd_cross = bool(I.crossed_above(macd_df["macd"], macd_df["signal"]).tail(3).any())
        stoch_bull = _ok(v_k) and _ok(v_d) and v_k > v_d and v_k < 85
        momentum = (
            0.45 * _rsi_score(v_rsi)
            + 0.15 * (1.0 if rising_rsi else 0.0)
            + 0.25 * (1.0 if macd_bull else (0.6 if macd_cross else 0.0))
            + 0.15 * (1.0 if stoch_bull else 0.0)
        )

        # ---- BREAKOUT ----
        is_breakout = _ok(recent_high) and last_close > recent_high
        vol_confirmed = _ok(vol_ratio) and vol_ratio >= s.breakout.volume_multiple
        if is_breakout and vol_confirmed:
            breakout = 1.0
        elif is_breakout:
            breakout = 0.65
        elif _ok(recent_high) and last_close > recent_high * 0.98:
            breakout = 0.45  # approaching
        else:
            breakout = 0.2
        if _ok(v_adx) and v_adx >= 25 and breakout > 0.3:
            breakout = min(1.0, breakout + 0.1)  # strong trend confirmation

        # ---- VOLUME ----
        if _ok(vol_ratio):
            volume_score = max(0.0, min(1.0, (vol_ratio - 0.5) / 1.5))
        else:
            volume_score = 0.5

        # ---- PATTERNS ----
        pats = P.detect_patterns(df, mctx.settings)
        best_bull = P.best_bullish_pattern(pats)
        best_bear = P.best_bearish_pattern(pats)
        pattern_score = best_bull.confidence if best_bull else 0.0

        # ---- combine ----
        w = mctx.settings.technical.weights
        wsum = sum(w.values()) or 1.0
        parts = {
            "trend": trend,
            "momentum": momentum,
            "breakout": breakout,
            "volume": volume_score,
            "patterns": pattern_score,
        }
        score = sum(w.get(k, 0.0) * v for k, v in parts.items()) / wsum

        reason = self._reason(parts, last_close, v_sma200, v_rsi, is_breakout,
                              vol_confirmed, vol_ratio, best_bull, best_bear)

        entry_level = recent_high if (is_breakout and _ok(recent_high)) else last_close
        details = {
            "subscores": {k: round(v, 3) for k, v in parts.items()},
            "indicators": {
                "close": last_close, "sma20": v_sma20, "sma50": v_sma50, "sma200": v_sma200,
                "ema20": _last(ema20), "rsi": v_rsi, "macd": v_macd, "macd_signal": v_sig,
                "macd_hist": v_hist, "atr": _last(atr), "stoch_k": v_k, "stoch_d": v_d,
                "adx": v_adx, "bb_upper": _last(bb["upper"]), "bb_lower": _last(bb["lower"]),
                "bb_pct_b": _last(bb["pct_b"]), "recent_high": recent_high,
                "avg_volume": avg_vol, "last_volume": last_vol, "vol_ratio": vol_ratio,
            },
            "entry_level": entry_level,
            "pattern": (
                {"name": best_bull.name, "confidence": round(best_bull.confidence, 3),
                 "target": best_bull.target, "neckline": best_bull.neckline}
                if best_bull else None
            ),
            "bearish_pattern": best_bear.name if best_bear else None,
            "macd_bearish": _ok(v_hist) and v_hist < 0 and v_macd < v_sig,
            "below_trend_sma": _ok(v_sma50) and last_close < v_sma50,
        }
        return SubScore(self.key, score, reason, details=details)

    @staticmethod
    def _reason(parts, close, sma200, rsi, is_breakout, vol_confirmed, vol_ratio,
                best_bull, best_bear) -> str:
        bits: list[str] = []
        if _ok(sma200):
            bits.append("above 200-day average" if close > sma200 else "below 200-day average")
        if _ok(rsi):
            bits.append(f"RSI {rsi:.0f}")
        if is_breakout:
            v = f" on {vol_ratio:.1f}x volume" if _ok(vol_ratio) else ""
            bits.append(f"breakout{' (vol-confirmed)' if vol_confirmed else ''}{v}")
        if best_bull:
            bits.append(best_bull.reason)
        if best_bear:
            bits.append(f"caution: {best_bear.name.replace('_', ' ')}")
        ranked = sorted(parts.items(), key=lambda kv: kv[1], reverse=True)
        lead = ranked[0][0]
        return f"Technical lead: {lead}. " + "; ".join(bits) if bits else f"Technical lead: {lead}"


def analyze_technical(sctx: StockContext, mctx: MarketContext) -> SubScore:
    return TechnicalAnalyzer().analyze(sctx, mctx)
