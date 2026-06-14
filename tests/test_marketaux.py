import pytest

from analyzers.sentiment import score_headlines
from config.loader import load_settings
from config.schema import Stock
from data_ingestion.news import (
    MarketauxNewsProvider,
    RssNewsProvider,
    get_news_provider,
    match_headlines_to_symbols,
)

SAMPLE_ARTICLE = {
    "title": "Reliance Q4 profit jumps",
    "description": "RIL beats estimates; Infosys also gains",
    "url": "http://example.com/x",
    "published_at": "2026-06-12T08:00:00Z",
    "source": "ET Markets",
    "entities": [
        {"symbol": "RELIANCE.NSE", "exchange": "NSE", "sentiment_score": 0.6},
        {"symbol": "RELIANCE.BSE", "exchange": "BSE", "sentiment_score": 0.5},  # wrong exchange
        {"symbol": "INFY.NSE", "exchange": "NSE", "sentiment_score": -0.2},
        {"symbol": "NOTINUNIVERSE.NSE", "exchange": "NSE", "sentiment_score": 0.9},
    ],
}


def test_article_parsing_tags_and_sentiment():
    valid = {"RELIANCE", "INFY", "TCS"}
    h = MarketauxNewsProvider._article_to_headline(SAMPLE_ARTICLE, ".NSE", valid)
    assert h["symbols"] == ["INFY", "RELIANCE"]          # BSE + unknown filtered out
    assert abs(h["provider_sentiment"] - 0.2) < 1e-9      # (0.6 + -0.2) / 2
    assert h["title"] == "Reliance Q4 profit jumps"
    assert h["link"] == "http://example.com/x"


def test_matching_trusts_provider_tags():
    stocks = [Stock("RELIANCE", "Oil & Gas", name="Reliance Industries"),
              Stock("INFY", "IT", name="Infosys")]
    # Title mentions neither; tags should still route it correctly.
    heads = [{"title": "Big-cap movers today", "summary": "", "symbols": ["RELIANCE"]}]
    m = match_headlines_to_symbols(heads, stocks)
    assert m == {"RELIANCE": heads}


def test_score_headlines_prefers_provider_sentiment():
    heads = [{"title": "x", "provider_sentiment": 0.6},
             {"title": "y", "provider_sentiment": -0.2}]
    raw, method = score_headlines(heads, prefer_provider=True)
    assert method == "provider"
    assert abs(raw - 0.2) < 1e-9
    # When disabled it must NOT use provider sentiment.
    _, method2 = score_headlines(heads, prefer_provider=False)
    assert method2 != "provider"


def test_factory_falls_back_to_rss_without_key(monkeypatch):
    monkeypatch.delenv("MARKETAUX_KEY", raising=False)
    s = load_settings()
    s.news.provider = "marketaux"
    assert isinstance(get_news_provider(s), RssNewsProvider)


def test_factory_uses_marketaux_with_key(monkeypatch):
    monkeypatch.setenv("MARKETAUX_KEY", "test-token")
    s = load_settings()
    s.news.provider = "marketaux"
    provider = get_news_provider(s)
    assert isinstance(provider, MarketauxNewsProvider)
    assert provider.api_token == "test-token"
    assert len(provider.symbols) > 50  # loaded the universe
