"""Macro factor (market-wide): USD/INR, crude, US indices, RBI repo rate.

Each component is scored to [-1,1] then combined with ``macro.weights``:
* USD/INR rising (rupee weakening) -> negative for equities.
* Crude rising -> negative (India is a net importer).
* US indices rising -> positive (risk-on spillover).
* Lower repo rate -> positive (cheaper money). 6.5% is treated as neutral.
"""
from __future__ import annotations

import math

import pandas as pd

from common.types import SubScore, bipolar_to_unit
from analyzers.context import MarketContext, StockContext


def _trend(series: pd.Series, lookback: int) -> float:
    if series is None:
        return 0.0
    s = series.dropna()
    if len(s) < 2:
        return 0.0
    n = min(lookback, len(s) - 1)
    base = s.iloc[-1 - n]
    return float(s.iloc[-1] / base - 1.0) if base else 0.0


def _squash(x: float, scale: float = 10.0) -> float:
    return math.tanh(x * scale)  # ~±1 for a few % move


def _rate_score(repo: float) -> float:
    # 5.0% -> +1, 6.5% -> 0, 8.0% -> -1
    return max(-1.0, min(1.0, (6.5 - repo) / 1.5))


class MacroFactorAnalyzer:
    key = "macro"

    def analyze(self, sctx: StockContext, mctx: MarketContext) -> SubScore:
        cfg = mctx.settings.macro
        lb = cfg.trend_lookback_days
        m = mctx.macro or {}

        usdinr_tr = _trend(m.get("usdinr"), lb)
        crude_tr = _trend(m.get("crude"), lb)
        us_tr = (_trend(m.get("sp500"), lb) + _trend(m.get("nasdaq"), lb)) / 2.0

        comp = {
            "usdinr": -_squash(usdinr_tr),
            "crude": -_squash(crude_tr),
            "us_indices": _squash(us_tr),
            "rates": _rate_score(cfg.rbi_repo_rate),
        }
        w = cfg.weights
        wsum = sum(w.values()) or 1.0
        raw = max(-1.0, min(1.0, sum(w.get(k, 0.0) * v for k, v in comp.items()) / wsum))

        drivers = sorted(comp.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
        driver_txt = ", ".join(f"{k} {v:+.2f}" for k, v in drivers)
        tone = "supportive" if raw > 0.1 else "headwind" if raw < -0.1 else "neutral"
        reason = f"Macro {tone} ({raw:+.2f}); drivers: {driver_txt}"
        return SubScore(self.key, bipolar_to_unit(raw), reason, raw=raw,
                        details={k: round(v, 3) for k, v in comp.items()})


def analyze_macro(sctx: StockContext, mctx: MarketContext) -> SubScore:
    return MacroFactorAnalyzer().analyze(sctx, mctx)
