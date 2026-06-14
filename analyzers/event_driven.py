"""Event / catalyst factor and risk-flag helper.

The hard event *gate* (no entry within N days of results, F&O ban) lives in
``scoring/gates.py``. This module produces the graded ``event`` sub-score and the
plain-English risk flags attached to each alert.
"""
from __future__ import annotations

from common.calendar_nse import trading_days_until
from common.types import SubScore
from analyzers.context import MarketContext, StockContext


class EventAnalyzer:
    key = "event"

    def analyze(self, sctx: StockContext, mctx: MarketContext) -> SubScore:
        gate_days = mctx.settings.gates.event.no_entry_days_before_earnings
        ed = sctx.earnings_date
        if ed is None:
            return SubScore(self.key, 0.6, "No scheduled results nearby",
                            details={"earnings_date": None})

        days = trading_days_until(ed, mctx.as_of)
        if days is None:
            return SubScore(self.key, 0.6, "Results date unknown")

        if 0 <= days <= gate_days:
            score, note = 0.40, f"Results in {days} trading day(s) — event risk"
        elif gate_days < days <= 20:
            score, note = 0.55, f"Results in {days} trading days"
        elif -5 <= days < 0:
            score, note = 0.65, f"Results {abs(days)} day(s) ago — fresh catalyst"
        else:
            score, note = 0.60, "No imminent catalyst"
        return SubScore(self.key, score, note,
                        details={"earnings_date": ed.isoformat(), "trading_days_to_earnings": days})


def event_risk_flags(sctx: StockContext, mctx: MarketContext) -> list[str]:
    """Plain-English risk flags surfaced on alerts."""
    flags: list[str] = []
    ed = sctx.earnings_date
    if ed is not None:
        days = trading_days_until(ed, mctx.as_of)
        if days is not None and 0 <= days <= 7:
            flags.append(f"Results in {days} trading day(s)")
    if sctx.symbol in mctx.fno_ban:
        flags.append("In F&O ban period")
    rs = mctx.sector_rs.get(sctx.sector)
    if rs is not None and rs < -0.2:
        flags.append(f"Sector '{sctx.sector}' lagging (RS {rs:+.2f})")
    if mctx.regime.get("vix") is not None and mctx.regime["vix"] >= mctx.settings.gates.market_regime.vix_ceiling - 3:
        flags.append(f"Elevated India VIX ({mctx.regime['vix']:.1f})")
    return flags


def analyze_event(sctx: StockContext, mctx: MarketContext) -> SubScore:
    return EventAnalyzer().analyze(sctx, mctx)
