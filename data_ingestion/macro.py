"""Macro factor series: USD/INR, crude, US indices (yfinance). RBI repo from config."""
from __future__ import annotations

import pandas as pd

from common.logging_config import get_logger
from data_ingestion.base import DiskCache, MacroProvider

log = get_logger(__name__)


class YFinanceMacroProvider(MacroProvider):
    def __init__(self, symbol_map: dict[str, str] | None = None,
                 cache_ttl_hours: float = 14.0) -> None:
        self.symbol_map = symbol_map or {
            "usdinr": "INR=X",
            "crude": "CL=F",
            "sp500": "^GSPC",
            "nasdaq": "^IXIC",
        }
        self.cache = DiskCache("macro", ttl_hours=cache_ttl_hours)

    def get_series(self, key: str, lookback_days: int = 60) -> pd.Series:
        ticker = self.symbol_map.get(key, key)
        cached = self.cache.get(ticker)
        if cached is not None and len(cached) >= lookback_days:
            return cached.tail(lookback_days)

        series = pd.Series(dtype="float64")
        try:
            import yfinance as yf

            raw = yf.download(ticker, period=f"{max(lookback_days, 120)}d",
                              progress=False, threads=False, auto_adjust=True)
            if len(raw):
                col = "Close" if "Close" in raw.columns else raw.columns[0]
                series = raw[col]
                if hasattr(series, "columns"):
                    series = series.iloc[:, 0]
                series.index = pd.to_datetime(series.index).normalize()
                series.name = key
        except Exception as exc:
            log.debug("macro fetch failed for %s: %s", ticker, exc)

        if len(series):
            self.cache.set(ticker, series)
        return series.tail(lookback_days)

    def get_snapshot(self, lookback_days: int = 30) -> dict[str, pd.Series]:
        return {k: self.get_series(k, lookback_days) for k in self.symbol_map}


def get_macro_provider(settings=None) -> MacroProvider:
    sym_map = None
    if settings is not None:
        sym_map = {k: v for k, v in settings.macro.symbols.items() if k != "vix"}
    return YFinanceMacroProvider(symbol_map=sym_map)
