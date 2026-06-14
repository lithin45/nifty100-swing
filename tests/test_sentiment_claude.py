import analyzers.sentiment as sentiment
from analyzers.sentiment import ClaudeSentimentScorer, score_headlines


class _Block:
    type = "text"
    text = '{"score": 0.7, "reason": "record profit and order wins"}'


class _Resp:
    content = [_Block()]


class _Msgs:
    def create(self, **kwargs):
        return _Resp()


class _Client:
    messages = _Msgs()


def test_claude_scorer_parses_structured_output():
    s = ClaudeSentimentScorer(api_key="test-key")
    s._client = _Client()  # bypass the real anthropic client
    score, reason = s.score(["Company posts record profit"])
    assert abs(score - 0.7) < 1e-9
    assert "record profit" in reason


def test_claude_scorer_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = ClaudeSentimentScorer(api_key="")
    assert s.available is False
    assert s.score(["anything"]) is None


def test_score_headlines_uses_claude_when_available(monkeypatch):
    class FakeScorer:
        def score(self, titles):
            return (0.6, "fake reason")

    monkeypatch.setattr(sentiment, "_get_claude_scorer", lambda model: FakeScorer())
    raw, method = score_headlines([{"title": "x"}], provider="claude", prefer_provider=False)
    assert method == "claude" and abs(raw - 0.6) < 1e-9


def test_claude_provider_falls_back_without_key(monkeypatch):
    # No key -> Claude scorer is None -> FinBERT (not installed) -> lexicon.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(sentiment, "_CLAUDE_SCORER", None)
    monkeypatch.setattr(sentiment, "_CLAUDE_FAILED", False)
    raw, method = score_headlines(
        [{"title": "Company wins record order, profit jumps"}],
        provider="claude", prefer_provider=False,
    )
    assert method in ("finbert", "lexicon")  # never "claude" without a key
    assert raw > 0  # positive words still scored by the fallback
