"""News sentiment via FinBERT, with a finance-lexicon fallback.

``score_headlines`` returns a signed score in [-1, 1]. The analyzer maps that to
a [0, 1] SubScore (``raw`` keeps the signed value for the conflict-penalty logic
in the composite). FinBERT (transformers + torch) is heavy and optional — if it
can't load, we fall back to a small finance word-list scorer so the pipeline
still runs unattended.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from common.logging_config import get_logger
from common.types import SubScore, bipolar_to_unit
from analyzers.context import MarketContext, StockContext

log = get_logger(__name__)

_PIPELINE = None
_PIPELINE_FAILED = False

# Compact finance lexicon for the fallback scorer.
_POS = {
    "surge", "surges", "jump", "jumps", "rally", "rallies", "gain", "gains", "rise",
    "rises", "beat", "beats", "record", "profit", "profits", "growth", "upgrade",
    "upgrades", "outperform", "bullish", "strong", "soar", "soars", "high", "wins",
    "win", "order", "orders", "expansion", "dividend", "buyback", "approval",
}
_NEG = {
    "fall", "falls", "drop", "drops", "plunge", "plunges", "decline", "declines",
    "loss", "losses", "miss", "misses", "downgrade", "downgrades", "weak", "bearish",
    "slump", "slumps", "cut", "cuts", "fraud", "probe", "ban", "default", "lawsuit",
    "warning", "slowdown", "crash", "selloff", "resign", "resigns",
}
_WORD = re.compile(r"[a-z]+")


def _lexicon_score(text: str) -> float:
    words = _WORD.findall(text.lower())
    pos = sum(1 for w in words if w in _POS)
    neg = sum(1 for w in words if w in _NEG)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _get_pipeline(model: str):
    global _PIPELINE, _PIPELINE_FAILED
    if _PIPELINE is not None or _PIPELINE_FAILED:
        return _PIPELINE
    try:
        from transformers import pipeline

        _PIPELINE = pipeline("text-classification", model=model, top_k=None)
        log.info("FinBERT loaded: %s", model)
    except Exception as exc:
        log.warning("FinBERT unavailable (%s); using lexicon fallback", exc)
        _PIPELINE_FAILED = True
    return _PIPELINE


def _finbert_signed(results: list[dict]) -> float:
    """Convert FinBERT label scores -> signed [-1,1] (P(pos) - P(neg))."""
    scores = {r["label"].lower(): r["score"] for r in results}
    return float(scores.get("positive", 0.0) - scores.get("negative", 0.0))


def score_headlines(
    headlines: list[dict[str, Any]],
    model: str = "ProsusAI/finbert",
    max_headlines: int = 15,
) -> tuple[float, str]:
    """Return (signed score in [-1,1], method) averaged over recent headlines."""
    titles = [h.get("title", "") for h in headlines[:max_headlines] if h.get("title")]
    if not titles:
        return 0.0, "none"

    pipe = _get_pipeline(model)
    if pipe is not None:
        try:
            outputs = pipe(titles)
            vals = [_finbert_signed(o if isinstance(o, list) else [o]) for o in outputs]
            return (sum(vals) / len(vals), "finbert")
        except Exception as exc:
            log.warning("FinBERT scoring failed (%s); lexicon fallback", exc)

    vals = [_lexicon_score(t) for t in titles]
    return (sum(vals) / len(vals), "lexicon")


class SentimentAnalyzer:
    key = "sentiment"

    def analyze(self, sctx: StockContext, mctx: MarketContext) -> SubScore:
        cfg = mctx.settings.sentiment
        if not sctx.headlines:
            return SubScore(self.key, 0.5, "No recent news", raw=0.0,
                            details={"n_headlines": 0, "method": "none"})
        raw, method = score_headlines(sctx.headlines, cfg.model, cfg.max_headlines_per_stock)
        n = len(sctx.headlines)
        tone = "positive" if raw > 0.15 else "negative" if raw < -0.15 else "neutral"
        reason = f"{tone.capitalize()} news tone ({raw:+.2f}) from {n} headline(s) [{method}]"
        return SubScore(self.key, bipolar_to_unit(raw), reason, raw=raw,
                        details={"n_headlines": n, "method": method, "signed": round(raw, 3)})


def analyze_sentiment(sctx: StockContext, mctx: MarketContext) -> SubScore:
    return SentimentAnalyzer().analyze(sctx, mctx)
