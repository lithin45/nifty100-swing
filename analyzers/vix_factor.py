"""India VIX / market-regime favourability factor (market-wide).

Lower VIX = calmer market = friendlier for swing entries. A *falling* VIX gets a
small bonus. The hard VIX ceiling lives in the market-regime gate; this factor is
the graded version that feeds the composite.
"""
from __future__ import annotations

import math

from common.types import SubScore
from analyzers.context import MarketContext, StockContext


def vix_level_score(vix: float) -> float:
    if vix is None or math.isnan(vix):
        return 0.5
    if vix < 12:
        return 0.90
    if vix < 15:
        return 0.82
    if vix < 18:
        return 0.72
    if vix < 22:
        return 0.58
    if vix < 26:
        return 0.40
    if vix < 32:
        return 0.25
    return 0.10


class VixFactorAnalyzer:
    key = "vix"

    def analyze(self, sctx: StockContext, mctx: MarketContext) -> SubScore:
        vix = mctx.vix.dropna() if mctx.vix is not None else None
        if vix is None or len(vix) == 0:
            return SubScore(self.key, 0.5, "India VIX unavailable (neutral)")

        latest = float(vix.iloc[-1])
        score = vix_level_score(latest)
        falling = len(vix) >= 2 and latest < float(vix.iloc[-2])
        if falling:
            score = min(1.0, score + 0.05)
        trend = "falling" if falling else "rising/flat"
        reason = f"India VIX {latest:.1f} ({trend}) — {'calm' if latest < 18 else 'elevated' if latest < 26 else 'fearful'} regime"
        return SubScore(self.key, score, reason,
                        details={"vix": round(latest, 2), "falling": falling})


def analyze_vix(sctx: StockContext, mctx: MarketContext) -> SubScore:
    return VixFactorAnalyzer().analyze(sctx, mctx)
