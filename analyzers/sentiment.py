"""News sentiment via FinBERT, with a finance-lexicon fallback.

``score_headlines`` returns a signed score in [-1, 1]. The analyzer maps that to
a [0, 1] SubScore (``raw`` keeps the signed value for the conflict-penalty logic
in the composite). FinBERT (transformers + torch) is heavy and optional — if it
can't load, we fall back to a small finance word-list scorer so the pipeline
still runs unattended.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from common.logging_config import get_logger
from common.types import SubScore, bipolar_to_unit
from analyzers.context import MarketContext, StockContext

log = get_logger(__name__)

_PIPELINE = None
_PIPELINE_FAILED = False
_CLAUDE_SCORER = None
_CLAUDE_FAILED = False

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


_CLAUDE_SYSTEM = (
    "You are a financial news sentiment analyst for Indian (NSE) stocks. "
    "Given recent headlines about ONE stock, judge the overall sentiment for its "
    "near-term (1 day to 1 month) swing-trading outlook. Account for Indian market "
    "context (results, RBI/SEBI actions, promoter pledges, order wins, downgrades). "
    "Return a single overall score from -1.0 (very negative) to +1.0 (very positive) "
    "and a concise one-line reason. Be calibrated: routine/ambiguous news is near 0."
)
_CLAUDE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "overall sentiment, -1.0 to 1.0"},
        "reason": {"type": "string", "description": "one-line justification"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


class ClaudeSentimentScorer:
    """Score a stock's headlines with Claude (one structured-output call).

    Uses Haiku by default — sentiment is a simple classification, so the cheapest
    capable model is the right call. Returns None on any failure so callers fall
    back to FinBERT/lexicon. Needs ``ANTHROPIC_API_KEY``.
    """

    def __init__(self, model: str = "claude-haiku-4-5", api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: optional dependency

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def score(self, titles: list[str]) -> Optional[tuple[float, str]]:
        if not titles or not self.available:
            return None
        joined = "\n".join(f"- {t}" for t in titles[:15])
        try:
            resp = self._get_client().messages.create(
                model=self.model,
                max_tokens=200,
                system=_CLAUDE_SYSTEM,
                messages=[{"role": "user",
                           "content": f"Headlines:\n{joined}\n\nReturn the overall sentiment."}],
                output_config={"format": {"type": "json_schema", "schema": _CLAUDE_SCHEMA}},
            )
            text = next((b.text for b in resp.content if b.type == "text"), "")
            data = json.loads(text)
            score = max(-1.0, min(1.0, float(data["score"])))
            return score, str(data.get("reason", ""))[:140]
        except Exception as exc:  # network/SDK/quota/parse — degrade gracefully
            log.warning("Claude sentiment failed (%s); falling back", exc)
            return None


def _get_claude_scorer(model: str) -> Optional[ClaudeSentimentScorer]:
    global _CLAUDE_SCORER, _CLAUDE_FAILED
    if _CLAUDE_FAILED:
        return None
    if _CLAUDE_SCORER is None:
        scorer = ClaudeSentimentScorer(model=model)
        if not scorer.available:
            _CLAUDE_FAILED = True
            return None
        _CLAUDE_SCORER = scorer
    return _CLAUDE_SCORER


def score_headlines(
    headlines: list[dict[str, Any]],
    model: str = "ProsusAI/finbert",
    max_headlines: int = 15,
    prefer_provider: bool = True,
    provider: str = "finbert",
    claude_model: str = "claude-haiku-4-5",
) -> tuple[float, str]:
    """Return (signed score in [-1,1], method) for a stock's recent headlines.

    Resolution order:
    1. provider sentiment (e.g. Marketaux) if ``prefer_provider`` and present;
    2. the configured ``provider``: "claude" (Claude → FinBERT → lexicon),
       "finbert" (FinBERT → lexicon), or "lexicon".
    Every step degrades gracefully to the next, so the pipeline never breaks.
    """
    subset = headlines[:max_headlines]
    if prefer_provider:
        provided = [h["provider_sentiment"] for h in subset
                    if isinstance(h.get("provider_sentiment"), (int, float))]
        if provided:
            return max(-1.0, min(1.0, sum(provided) / len(provided))), "provider"

    titles = [h.get("title", "") for h in subset if h.get("title")]
    if not titles:
        return 0.0, "none"

    provider = (provider or "finbert").lower()

    # 1) Claude (cheap Haiku call) — falls through to FinBERT/lexicon on failure.
    if provider == "claude":
        scorer = _get_claude_scorer(claude_model)
        if scorer is not None:
            result = scorer.score(titles)
            if result is not None:
                return max(-1.0, min(1.0, result[0])), "claude"

    # 2) FinBERT (unless lexicon was explicitly requested).
    if provider != "lexicon":
        pipe = _get_pipeline(model)
        if pipe is not None:
            try:
                outputs = pipe(titles)
                vals = [_finbert_signed(o if isinstance(o, list) else [o]) for o in outputs]
                return (sum(vals) / len(vals), "finbert")
            except Exception as exc:
                log.warning("FinBERT scoring failed (%s); lexicon fallback", exc)

    # 3) Lexicon (always available).
    vals = [_lexicon_score(t) for t in titles]
    return (sum(vals) / len(vals), "lexicon")


class SentimentAnalyzer:
    key = "sentiment"

    def analyze(self, sctx: StockContext, mctx: MarketContext) -> SubScore:
        cfg = mctx.settings.sentiment
        if not sctx.headlines:
            return SubScore(self.key, 0.5, "No recent news", raw=0.0,
                            details={"n_headlines": 0, "method": "none"})
        raw, method = score_headlines(
            sctx.headlines, cfg.model, cfg.max_headlines_per_stock,
            prefer_provider=cfg.prefer_provider_sentiment,
            provider=cfg.provider, claude_model=cfg.claude_model,
        )
        n = len(sctx.headlines)
        tone = "positive" if raw > 0.15 else "negative" if raw < -0.15 else "neutral"
        reason = f"{tone.capitalize()} news tone ({raw:+.2f}) from {n} headline(s) [{method}]"
        return SubScore(self.key, bipolar_to_unit(raw), reason, raw=raw,
                        details={"n_headlines": n, "method": method, "signed": round(raw, 3)})


def analyze_sentiment(sctx: StockContext, mctx: MarketContext) -> SubScore:
    return SentimentAnalyzer().analyze(sctx, mctx)
