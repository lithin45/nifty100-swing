"""India VIX provider (^INDIAVIX via yfinance, nsepython fallback)."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from common.logging_config import get_logger
from data_ingestion.base import DiskCache, VixProvider

log = get_logger(__name__)


class IndiaVixProvider(VixProvider):
    def __init__(self, cache_ttl_hours: float = 12.0) -> None:
        self.cache = DiskCache("vix", ttl_hours=cache_ttl_hours)

    def get_history(self, lookback_days: int = 60) -> pd.Series:
        cached = self.cache.get("history")
        if cached is not None and len(cached) >= lookback_days:
            return cached.tail(lookback_days)

        series = pd.Series(dtype="float64")
        try:
            import yfinance as yf

            raw = yf.download("^INDIAVIX", period=f"{max(lookback_days, 120)}d",
                              progress=False, threads=False)
            if len(raw):
                col = "Close" if "Close" in raw.columns else raw.columns[0]
                series = raw[col]
                if hasattr(series, "columns"):  # MultiIndex slice
                    series = series.iloc[:, 0]
                series.index = pd.to_datetime(series.index).normalize()
                series.name = "india_vix"
        except Exception as exc:
            log.debug("yfinance VIX failed: %s", exc)

        if series.empty:
            try:
                from nsepython import nse_index_quote  # noqa

                # Fallback returns only the latest value.
                from nsepython import nsefetch

                data = nsefetch("https://www.nseindia.com/api/allIndices")
                for idx in data.get("data", []):
                    if "VIX" in str(idx.get("index", "")).upper():
                        val = float(idx.get("last"))
                        series = pd.Series([val], index=[pd.Timestamp(dt.date.today())],
                                           name="india_vix")
                        break
            except Exception as exc:
                log.debug("nsepython VIX failed: %s", exc)

        if len(series):
            self.cache.set("history", series)
        return series.tail(lookback_days)


def get_vix_provider() -> VixProvider:
    return IndiaVixProvider()
