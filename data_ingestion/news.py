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

# Corporate suffixes to strip when building a specific (but not over-aggressive)
# match phrase — keeps "Indian Oil" while dropping "Ltd/Corporation".
_CORP_SUFFIXES = {"ltd", "limited", "the", "co", "company", "corporation", "corp",
                  "industries", "enterprises", "&", "and"}

# Generic words that are useless / dangerous as a standalone company alias: a name
# that reduces to one of these (or is built ENTIRELY from them) matches unrelated
# headlines (e.g. "Indian Oil" -> "oil" tags IOC to every crude-oil story). Such
# stocks are matched by their ticker symbol and full name instead.
_AMBIGUOUS_ALIASES = {
    "oil", "gas", "power", "grid", "energy", "steel", "metal", "metals", "coal",
    "cement", "finance", "financial", "services", "service", "bank", "banks",
    "auto", "motors", "motor", "chemicals", "chemical", "media", "retail",
    "ports", "port", "life", "general", "insurance", "consumer", "durables",
    "infra", "infrastructure", "industrial", "national", "international",
    "products", "product", "paints", "paint", "systems", "system",
    "technologies", "technology", "tech", "foods", "food", "healthcare",
    "health", "telecom", "aviation", "realty", "housing", "capital",
    "securities", "holdings", "global", "india", "indian", "hotels", "hotel",
    "electric", "electricals", "electronics", "mills", "petroleum",
}


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


def _suffix_stripped(name: str) -> str:
    """Drop only corporate suffixes (keep geographic/sector words): a specific
    multi-word phrase like 'indian oil' rather than the over-aggressive 'oil'."""
    tokens = [t for t in normalize_text(name).split() if t not in _CORP_SUFFIXES]
    return " ".join(tokens)


def _too_generic(alias: str) -> bool:
    """An alias is unusable if it's a single short/ambiguous word, or is built
    ENTIRELY from generic words (e.g. 'power grid', 'indian oil')."""
    tokens = alias.split()
    if not tokens:
        return True
    if len(tokens) == 1 and (alias in _AMBIGUOUS_ALIASES or len(alias) < 4):
        return True
    return all(t in _AMBIGUOUS_ALIASES for t in tokens)


def build_aliases(stock: Stock) -> set[str]:
    """Punctuation-free match phrases: short name, full name, and a safe symbol.

    Generic-only derived names ('oil', 'power grid', 'indian oil') are dropped to
    avoid tagging a stock to every commodity/sector headline; those stocks rely on
    their ticker symbol and full company name instead.
    """
    candidates: set[str] = set()
    candidates.add(normalize_name(stock.name or stock.symbol))   # "reliance"
    if stock.name:
        candidates.add(_suffix_stripped(stock.name))             # "indian oil"
        candidates.add(normalize_text(stock.name))               # full name (specific)
    aliases = {a for a in candidates if a and not _too_generic(a)}

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

    If a headline carries explicit ticker tags (``h["symbols"]``, e.g. from
    Marketaux), those are trusted directly; otherwise we fall back to
    word-boundary text matching over ``title + summary``.
    """
    valid = {st.symbol for st in stocks}
    alias_map: dict[str, list[tuple[str, re.Pattern]]] = {}
    for st in stocks:
        for alias in build_aliases(st):
            pat = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            alias_map.setdefault(st.symbol, []).append((alias, pat))

    result: dict[str, list[dict[str, Any]]] = {}
    for h in headlines:
        tagged = [s for s in (h.get("symbols") or []) if s in valid]
        if tagged:  # trust provider-supplied entity tags
            for sym in tagged:
                result.setdefault(sym, []).append(h)
            continue
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


class MarketauxNewsProvider(NewsProvider):
    """Marketaux news API — returns headlines pre-tagged with stock tickers and
    an entity sentiment score. Free tier: ~100 requests/day, 3 articles/request.

    Each returned headline dict additionally carries:
      * ``symbols`` — NSE tickers Marketaux tagged on the article (suffix stripped)
      * ``provider_sentiment`` — average entity sentiment in [-1, 1] (or None)
    """

    BASE = "https://api.marketaux.com/v1/news/all"

    def __init__(self, api_token: str, symbols: list[str] | None = None,
                 cfg=None) -> None:
        from config.schema import MarketauxCfg

        self.api_token = api_token
        self.symbols = [s.upper() for s in (symbols or [])]
        self.cfg = cfg or MarketauxCfg()

    @staticmethod
    def _article_to_headline(article: dict, suffix: str,
                             valid: set[str] | None) -> dict[str, Any]:
        """Convert a Marketaux article into our standard headline dict (testable)."""
        suffix = suffix.lstrip(".").upper()
        syms, sents = [], []
        for ent in article.get("entities", []) or []:
            raw = str(ent.get("symbol", "")).upper()
            base = raw.split(".")[0]  # RELIANCE.NSE -> RELIANCE
            # keep only equity entities on our exchange (or any if suffix blank)
            ex = str(ent.get("exchange", "")).upper()
            if suffix and ex and suffix not in ex and not raw.endswith(suffix):
                continue
            if valid is not None and base not in valid:
                continue
            syms.append(base)
            sc = ent.get("sentiment_score")
            if isinstance(sc, (int, float)):
                sents.append(float(sc))
        return {
            "title": article.get("title", ""),
            "summary": article.get("description") or article.get("snippet", ""),
            "link": article.get("url", ""),
            "published": article.get("published_at"),
            "source": article.get("source", "marketaux"),
            "symbols": sorted(set(syms)),
            "provider_sentiment": (sum(sents) / len(sents)) if sents else None,
        }

    def get_headlines(self, max_age_days: int = 7) -> list[dict[str, Any]]:
        if not self.api_token:
            log.error("MARKETAUX_KEY not set; no Marketaux news")
            return []
        import requests

        valid = set(self.symbols) or None
        published_after = (dt.datetime.utcnow() - dt.timedelta(days=max_age_days)).strftime(
            "%Y-%m-%dT%H:%M"
        )
        batch = self.cfg.max_symbols_per_request
        # Build symbol batches (Marketaux ticker = SYMBOL + suffix).
        suffix = self.cfg.exchange_suffix
        batches = ([self.symbols[i:i + batch] for i in range(0, len(self.symbols), batch)]
                   if self.symbols else [None])

        out: list[dict[str, Any]] = []
        requests_made = 0
        for syms in batches:
            if requests_made >= self.cfg.max_requests:
                log.warning("Marketaux request cap (%d) reached; stopping", self.cfg.max_requests)
                break
            params = {
                "api_token": self.api_token,
                "language": self.cfg.language,
                "filter_entities": "true",
                "published_after": published_after,
            }
            if syms:
                params["symbols"] = ",".join(f"{s}{suffix}" for s in syms)
            else:
                params["countries"] = self.cfg.countries
            try:
                resp = requests.get(self.BASE, params=params, timeout=20)
                requests_made += 1
                if resp.status_code in (402, 429):
                    log.warning("Marketaux quota/limit hit (%s); returning partial", resp.status_code)
                    break
                resp.raise_for_status()
                for art in resp.json().get("data", []) or []:
                    out.append(self._article_to_headline(art, suffix, valid))
            except Exception as exc:
                log.debug("Marketaux request failed: %s", exc)
        return out


def get_news_provider(settings=None) -> NewsProvider:
    """Build the configured news provider.

    Selects Marketaux when ``news.provider == 'marketaux'`` and a ``MARKETAUX_KEY``
    is available; otherwise falls back to free RSS feeds.
    """
    import os

    if settings is None:
        from config.loader import get_settings

        settings = get_settings()

    if settings.news.provider.lower() == "marketaux":
        token = os.getenv("MARKETAUX_KEY", "")
        if token:
            from config.loader import load_universe

            symbols = [s.symbol for s in load_universe()]
            return MarketauxNewsProvider(token, symbols=symbols, cfg=settings.news.marketaux)
        log.warning("news.provider=marketaux but MARKETAUX_KEY not set; using RSS")
    return RssNewsProvider(settings.news.rss_feeds)
