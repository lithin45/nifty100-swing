"""Sectoral / benchmark index close series (yfinance).

Relative-strength computation and scoring live in
``analyzers/sector_factor.py``; this module only fetches the raw series.
"""
from __future__ import annotations

import pandas as pd

from common.logging_config import get_logger
from data_ingestion.base import DiskCache, SectorProvider

log = get_logger(__name__)

# yfinance fallbacks for indices that are sometimes unavailable.
_FALLBACK = {"^CNX100": "^NSEI"}  # Nifty 100 -> Nifty 50 as a proxy benchmark


class YFinanceSectorProvider(SectorProvider):
    def __init__(self, cache_ttl_hours: float = 14.0) -> None:
        self.cache = DiskCache("sectors", ttl_hours=cache_ttl_hours)

    def get_index_close(self, index_symbol: str, lookback_days: int = 90) -> pd.Series:
        cached = self.cache.get(index_symbol)
        if cached is not None and len(cached) >= lookback_days:
            return cached.tail(lookback_days)

        series = self._download(index_symbol, lookback_days)
        if series.empty and index_symbol in _FALLBACK:
            log.debug("index %s empty; trying fallback %s", index_symbol, _FALLBACK[index_symbol])
            series = self._download(_FALLBACK[index_symbol], lookback_days)

        if len(series):
            self.cache.set(index_symbol, series)
        return series.tail(lookback_days)

    @staticmethod
    def _download(ticker: str, lookback_days: int) -> pd.Series:
        try:
            import yfinance as yf

            raw = yf.download(ticker, period=f"{max(lookback_days, 200)}d",
                              progress=False, threads=False, auto_adjust=True)
            if not len(raw):
                return pd.Series(dtype="float64")
            col = "Close" if "Close" in raw.columns else raw.columns[0]
            series = raw[col]
            if hasattr(series, "columns"):
                series = series.iloc[:, 0]
            series.index = pd.to_datetime(series.index).normalize()
            series.name = ticker
            return series
        except Exception as exc:
            log.debug("sector index fetch failed for %s: %s", ticker, exc)
            return pd.Series(dtype="float64")


def get_sector_provider() -> SectorProvider:
    return YFinanceSectorProvider()
