"""BUY signal generation: assemble a concrete, capital-agnostic trade plan.

A BUY fires only when **all gates pass** and **composite >= entry_threshold**.
The plan uses the technical analyzer's indicator snapshot:
* entry  = breakout level (if breaking out) else last close
* stop   = entry - k×ATR
* target = max(R-multiple target, pattern measured-move) [if enabled]
* size % = risk_per_trade_pct / stop-distance%, capped at max_position_pct
"""
from __future__ import annotations

import math
from typing import Optional

from analyzers.context import MarketContext, StockContext
from analyzers.event_driven import event_risk_flags
from common.types import CompositeResult, GateReport, Signal, SignalAction, SubScore, TradePlan


def _subscores_to_dicts(subscores: list[SubScore], settings) -> list[dict]:
    """Serialise sub-scores (with weighted points) for storage/dashboard."""
    weights = settings.scoring.normalized_weights()
    out = []
    for ss in subscores:
        out.append(
            {
                "key": ss.key,
                "score": round(ss.score, 4),
                "raw": ss.raw,
                "weighted_points": round(weights.get(ss.key, 0.0) * ss.score * 100.0, 2),
                "reason": ss.reason,
                "details": ss.details,
            }
        )
    return out


def build_trade_plan(technical_details: dict, settings) -> Optional[TradePlan]:
    """Construct stop/target/size from the technical indicator snapshot."""
    ind = (technical_details or {}).get("indicators", {})
    atr = ind.get("atr")
    entry = (technical_details or {}).get("entry_level") or ind.get("close")
    if not entry or atr is None or (isinstance(atr, float) and math.isnan(atr)) or atr <= 0:
        return None

    risk = settings.risk
    stop = entry - risk.atr_stop_multiple * atr
    if stop <= 0 or stop >= entry:
        return None
    risk_per_share = entry - stop

    target = entry + risk.rr_target_multiple * risk_per_share
    pattern = (technical_details or {}).get("pattern")
    if risk.use_pattern_target and pattern and pattern.get("target"):
        pt = float(pattern["target"])
        if pt > entry:
            target = max(target, pt)

    reward_per_share = target - entry
    rr = reward_per_share / risk_per_share if risk_per_share else 0.0

    stop_distance_pct = risk_per_share / entry * 100.0
    size_pct = min(
        risk.max_position_pct,
        (risk.risk_per_trade_pct * 100.0 / stop_distance_pct) if stop_distance_pct else 0.0,
    )

    notes = (
        f"Risk {risk.risk_per_trade_pct:.1f}% of capital; stop {stop_distance_pct:.1f}% "
        f"below entry; size ~{size_pct:.1f}% of capital"
    )
    return TradePlan(
        entry_price=round(float(entry), 2),
        stop_loss=round(float(stop), 2),
        target=round(float(target), 2),
        atr=round(float(atr), 2),
        risk_per_share=round(float(risk_per_share), 2),
        reward_per_share=round(float(reward_per_share), 2),
        rr=round(float(rr), 2),
        position_size_pct=round(float(size_pct), 2),
        notes=notes,
    )


def top_reasons(composite: CompositeResult, n: int = 3) -> list[str]:
    """Plain-English reasons ranked by weighted contribution to the score."""
    by_key = {ss.key: ss for ss in composite.subscores}
    ranked = sorted(composite.contributions.items(), key=lambda kv: kv[1], reverse=True)
    reasons = []
    for key, _pts in ranked[:n]:
        ss = by_key.get(key)
        if ss and ss.reason:
            reasons.append(ss.reason)
    reasons.extend(composite.penalties)
    return reasons


def generate_signal(
    sctx: StockContext,
    mctx: MarketContext,
    composite: CompositeResult,
    gate_report: GateReport,
) -> Optional[Signal]:
    """Return a BUY Signal if gates pass and composite clears the threshold."""
    settings = mctx.settings
    if not gate_report.passed:
        return None
    if composite.score < settings.scoring.entry_threshold:
        return None

    tech = next((ss for ss in composite.subscores if ss.key == "technical"), None)
    plan = build_trade_plan(tech.details if tech else {}, settings)
    if plan is None:
        return None

    reasons = top_reasons(composite, settings.alerts.top_reasons)
    risk_flags = event_risk_flags(sctx, mctx)

    return Signal(
        symbol=sctx.symbol,
        sector=sctx.sector,
        action=SignalAction.BUY,
        as_of=mctx.as_of,
        composite=composite.score,
        plan=plan,
        reasons=reasons,
        risk_flags=risk_flags,
        details={
            "contributions": composite.contributions,
            "penalties": composite.penalties,
            "subscores": _subscores_to_dicts(composite.subscores, settings),
            "gate_summary": gate_report.summary(),
        },
    )
