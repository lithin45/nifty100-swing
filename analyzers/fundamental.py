"""Fundamental analyzer: turn a fundamentals dict into a ``fundamental`` SubScore.

Each available metric is scored to [0,1] with capital-agnostic thresholds, then
averaged over whatever metrics are present. Missing data -> neutral 0.5.
"""
from __future__ import annotations

from typing import Optional

from common.types import SubScore
from analyzers.context import MarketContext, StockContext


def _band(value: Optional[float], good: float, bad: float) -> Optional[float]:
    """Linear score in [0,1]; ``good`` maps to 1, ``bad`` to 0 (either direction)."""
    if value is None:
        return None
    if good == bad:
        return 0.5
    frac = (value - bad) / (good - bad)
    return max(0.0, min(1.0, frac))


class FundamentalAnalyzer:
    key = "fundamental"

    def analyze(self, sctx: StockContext, mctx: MarketContext) -> SubScore:
        f = sctx.fundamentals or {}
        scores: dict[str, float] = {}

        roe = _band(f.get("roe"), good=22.0, bad=5.0)          # higher better
        if roe is not None:
            scores["roe"] = roe
        pe = f.get("pe")
        if pe is not None:
            scores["pe"] = 0.2 if pe <= 0 else _band(pe, good=15.0, bad=45.0)  # lower better
        ps = _band(f.get("ps"), good=2.0, bad=12.0)            # lower better
        if ps is not None:
            scores["ps"] = ps
        de = _band(f.get("de"), good=0.3, bad=2.0)             # lower better
        if de is not None:
            scores["de"] = de
        growth = _band(f.get("earnings_growth") or f.get("profit_growth"),
                       good=20.0, bad=-5.0)                    # higher better
        if growth is not None:
            scores["growth"] = growth

        if not scores:
            return SubScore(self.key, 0.5, "No fundamentals available",
                            details={"metrics": {}})

        score = sum(scores.values()) / len(scores)
        best = max(scores, key=scores.get)
        worst = min(scores, key=scores.get)
        reason = (
            f"Fundamentals avg {score:.2f}: strongest {best} "
            f"({_fmt(f, best)}), weakest {worst} ({_fmt(f, worst)})"
        )
        return SubScore(self.key, score, reason,
                        details={"metrics": {k: round(v, 3) for k, v in scores.items()},
                                 "raw": {k: f.get(k) for k in
                                         ("roe", "pe", "ps", "de", "earnings_growth")}})


def _fmt(f: dict, key: str) -> str:
    mapping = {"growth": "earnings_growth"}
    raw_key = mapping.get(key, key)
    val = f.get(raw_key) if raw_key in f else f.get("profit_growth")
    return f"{val}" if val is not None else "n/a"


def analyze_fundamental(sctx: StockContext, mctx: MarketContext) -> SubScore:
    return FundamentalAnalyzer().analyze(sctx, mctx)
