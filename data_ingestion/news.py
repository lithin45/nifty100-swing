"""News ingestion (RSS) + headline -> stock matching.

The RSS fetch needs ``feedparser`` and network; the matching logic is pure and
unit-tested. An optional Marketaux/NewsData.io adapter can be slotted behind the
same :class:`~data_ingestion.base.NewsProvider` interface.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any, Iterable

from common.logging_config import get_logger
from data_ingestion.base import NewsProvider
from config.schema import Stock

log = get_logger(__name__)

# Words stripped when deriving a short company name for matching.
_STOP_WORDS = {
    "ltd", "limited", "the", "of", "and", "co", "company", "corporation", "corp",
    "industries", "india", "indian", "enterprises", "&",
}
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lower-case, strip punctuation (incl. ``&``/``-``), collapse whitespace.

    Applied to BOTH aliases and headline text so internal punctuation (e.g.
    "Larsen & Toubro") never blocks a match.
    """
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def normalize_name(name: str) -> str:
    """Like :func:`normalize_text` but also drops generic suffixes (Ltd, India…)."""
    tokens = [t for t in normalize_text(name).split() if t not in _STOP_WORDS]
    return " ".join(tokens).strip()


def build_aliases(stock: Stock) -> set[str]:
    """Punctuation-free match phrases: short name, full name, and a safe symbol."""
    aliases: set[str] = set()
    aliases.add(normalize_name(stock.name or stock.symbol))   # "reliance"
    if stock.name:
        aliases.add(normalize_text(stock.name))               # "larsen toubro"
    # Trust the raw symbol only when alphanumeric and >=3 chars (TCS, ITC, IOC).
    # Skips 2-letter tickers ("LT") and punctuated ones ("M&M", "BAJAJ-AUTO"),
    # which match via their company name instead.
    if stock.symbol.isalnum() and len(stock.symbol) >= 3:
        aliases.add(stock.symbol.lower())
    return {a for a in aliases if len(a) >= 3}


def match_headlines_to_symbols(
    headlines: Iterable[dict[str, Any]],
    stocks: Iterable[Stock],
) -> dict[str, list[dict[str, Any]]]:
    """Map each stock symbol -> list of headlines mentioning it.

    Matching is word-boundary aware over ``title + summary``.
    """
    alias_map: dict[str, list[tuple[str, re.Pattern]]] = {}
    for st in stocks:
        for alias in build_aliases(st):
            pat = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            alias_map.setdefault(st.symbol, []).append((alias, pat))

    result: dict[str, list[dict[str, Any]]] = {}
    for h in headlines:
        text = normalize_text(f"{h.get('title', '')} {h.get('summary', '')}")
        for symbol, patterns in alias_map.items():
            if any(pat.search(text) for _, pat in patterns):
                result.setdefault(symbol, []).append(h)
    return result


class RssNewsProvider(NewsProvider):
    def __init__(self, feeds: list[str]) -> None:
        self.feeds = feeds

    def get_headlines(self, max_age_days: int = 7) -> list[dict[str, Any]]:
        try:
            import feedparser
        except ImportError:
            log.error("feedparser not installed; no news")
            return []

        cutoff = dt.datetime.now() - dt.timedelta(days=max_age_days)
        out: list[dict[str, Any]] = []
        for url in self.feeds:
            try:
                parsed = feedparser.parse(url)
            except Exception as exc:
                log.debug("feed failed %s: %s", url, exc)
                continue
            source = parsed.feed.get("title", url) if hasattr(parsed, "feed") else url
            for e in parsed.entries:
                published = None
                if getattr(e, "published_parsed", None):
                    published = dt.datetime(*e.published_parsed[:6])
                if published and published < cutoff:
                    continue
                out.append(
                    {
                        "title": getattr(e, "title", ""),
                        "summary": re.sub("<[^<]+?>", "", getattr(e, "summary", "")),
                        "link": getattr(e, "link", ""),
                        "published": published.isoformat() if published else None,
                        "source": source,
                    }
                )
        return out


def get_news_provider(settings=None) -> NewsProvider:
    if settings is None:
        from config.loader import get_settings

        settings = get_settings()
    return RssNewsProvider(settings.news.rss_feeds)
