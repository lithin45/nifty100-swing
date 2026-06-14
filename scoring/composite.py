"""Stage 2 — WEIGHTED COMPOSITE SCORE (0–100).

Runs every analyzer, combines their [0,1] sub-scores using the (normalised)
weights from settings, then applies conflict penalties:

* strong technical **and** clearly negative news -> multiply by
  ``penalty_multiplier`` (don't chase a good chart into bad news).
* near-term event risk (low event sub-score) -> multiply by
  ``event_risk_penalty_multiplier``.

Every penalty is recorded in ``CompositeResult.penalties`` and surfaced in the
alert's reasons so the user understands *why* a score was docked.
"""
from __future__ import annotations

from typing import Callable

from analyzers.context import MarketContext, StockContext
from analyzers.event_driven import analyze_event
from analyzers.fii_dii_factor import analyze_fii_dii
from analyzers.fundamental import analyze_fundamental
from analyzers.macro_factor import analyze_macro
from analyzers.sector_factor import analyze_sector
from analyzers.sentiment import analyze_sentiment
from analyzers.technical import analyze_technical
from analyzers.vix_factor import analyze_vix
from common.types import CompositeResult, SubScore

# key -> analyzer callable. Keys must match scoring.weights in settings.yaml.
ANALYZERS: dict[str, Callable[[StockContext, MarketContext], SubScore]] = {
    "technical": analyze_technical,
    "sector": analyze_sector,
    "fii_dii": analyze_fii_dii,
    "sentiment": analyze_sentiment,
    "fundamental": analyze_fundamental,
    "event": analyze_event,
    "macro": analyze_macro,
    "vix": analyze_vix,
}


def run_analyzers(sctx: StockContext, mctx: MarketContext) -> list[SubScore]:
    """Run every analyzer; a failing analyzer degrades to a neutral 0.5."""
    out: list[SubScore] = []
    for key, fn in ANALYZERS.items():
        try:
            out.append(fn(sctx, mctx))
        except Exception as exc:  # never let one analyzer kill the pipeline
            out.append(SubScore(key, 0.5, f"{key} analyzer error: {exc}"))
    return out


def compute_composite(subscores: list[SubScore], settings) -> CompositeResult:
    """Combine sub-scores into a 0–100 composite with conflict penalties."""
    weights = settings.scoring.normalized_weights()
    by_key = {ss.key: ss for ss in subscores}

    contributions: dict[str, float] = {}
    base = 0.0
    for key, w in weights.items():
        ss = by_key.get(key)
        if ss is None:
            continue
        pts = w * ss.score * 100.0
        contributions[key] = round(pts, 2)
        base += pts

    score = base
    penalties: list[str] = []
    cp = settings.scoring.conflict_penalty
    if cp.enabled:
        tech = by_key.get("technical")
        sent = by_key.get("sentiment")
        if (
            tech is not None
            and sent is not None
            and tech.score >= cp.strong_technical_threshold
            and sent.raw is not None
            and sent.raw <= cp.negative_sentiment_threshold
        ):
            score *= cp.penalty_multiplier
            penalties.append(
                f"Strong chart vs negative news (sentiment {sent.raw:+.2f}): "
                f"score ×{cp.penalty_multiplier}"
            )
        event = by_key.get("event")
        if event is not None and event.score <= 0.45:
            score *= cp.event_risk_penalty_multiplier
            penalties.append(f"Near-term event risk: score ×{cp.event_risk_penalty_multiplier}")

    score = max(0.0, min(100.0, score))
    lead = max(contributions, key=contributions.get) if contributions else "n/a"
    reason = f"Composite {score:.1f}/100 (lead factor: {lead})"
    if penalties:
        reason += " — penalties applied"

    return CompositeResult(
        score=round(score, 2),
        subscores=subscores,
        contributions=contributions,
        penalties=penalties,
        reason=reason,
    )
