"""Fundamentals via screener.in (unofficial). Cached daily on disk.

Returns a normalized dict: ``roe, pe, ps, de, earnings_growth, profit_growth,
sales_growth``. Missing metrics are simply absent. Degrades to ``{}`` when the
network / parser libs are unavailable — the fundamental analyzer then returns a
neutral score.
"""
from __future__ import annotations

import re
from typing import Any

from common.logging_config import get_logger
from data_ingestion.base import DiskCache, FundamentalsProvider

log = get_logger(__name__)

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _to_float(text: str) -> float | None:
    m = _NUM.search(text.replace(",", ""))
    return float(m.group()) if m else None


class ScreenerFundamentalsProvider(FundamentalsProvider):
    """Scrape the top-ratios block from screener.in."""

    BASE = "https://www.screener.in/company/{sym}/consolidated/"
    _LABELS = {
        "stock p/e": "pe",
        "price to earning": "pe",
        "roe": "roe",
        "return on equity": "roe",
        "roce": "roce",
        "debt to equity": "de",
        "price to sales": "ps",
        "dividend yield": "dividend_yield",
        "eps": "eps",
    }

    def __init__(self, cache_ttl_hours: float = 22.0) -> None:
        self.cache = DiskCache("fundamentals", ttl_hours=cache_ttl_hours)

    def get_fundamentals(self, symbol: str) -> dict[str, Any]:
        symbol = symbol.upper()
        cached = self.cache.get(symbol)
        if cached is not None:
            return cached

        data: dict[str, Any] = {}
        try:
            import requests
            from bs4 import BeautifulSoup

            url = self.BASE.format(sym=symbol)
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if resp.status_code != 200:
                return {}
            soup = BeautifulSoup(resp.text, "lxml")
            for li in soup.select("#top-ratios li, ul#top-ratios li"):
                name = li.select_one(".name")
                value = li.select_one(".value")
                if not name or not value:
                    continue
                label = name.get_text(strip=True).lower()
                key = next((v for k, v in self._LABELS.items() if k in label), None)
                if key:
                    num = _to_float(value.get_text(strip=True))
                    if num is not None:
                        data[key] = num

            # Growth tables (best effort).
            for h in soup.find_all(string=re.compile("Compounded Profit Growth", re.I)):
                section = h.find_parent("table")
                if section:
                    cells = [c.get_text(strip=True) for c in section.find_all("td")]
                    nums = [_to_float(c) for c in cells if "%" in c]
                    if nums:
                        data["profit_growth"] = nums[0]
                    break

            data.setdefault("earnings_growth", data.get("profit_growth"))
        except Exception as exc:
            log.debug("screener fetch failed for %s: %s", symbol, exc)
            return {}

        if data:
            self.cache.set(symbol, data)
        return data


def get_fundamentals_provider() -> FundamentalsProvider:
    return ScreenerFundamentalsProvider()
