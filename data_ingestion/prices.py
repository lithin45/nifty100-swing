"""Price providers: yfinance (default), Upstox V3, Dhan.

All heavy/optional third-party libs are imported lazily inside methods so the
module imports cleanly even when they are not installed.
"""
from __future__ import annotations

import datetime as dt
import os
import time
from typing import Optional
from urllib.parse import quote

import pandas as pd

from common.logging_config import get_logger
from data_ingestion.base import OHLCV, DiskCache, PriceProvider, normalize_ohlcv

log = get_logger(__name__)


def _yf_ticker(symbol: str) -> str:
    """Map a bare NSE symbol to a yfinance ticker (indices/FX pass through)."""
    if symbol.startswith("^") or symbol.endswith(".NS") or "=" in symbol:
        return symbol
    return f"{symbol}.NS"


class YFinancePriceProvider(PriceProvider):
    """Primary EOD source: yfinance ``<SYMBOL>.NS`` (split/dividend adjusted)."""

    name = "yfinance"

    def __init__(self, adjust: bool = True, cache_ttl_hours: float = 20.0,
                 history_days: int = 800) -> None:
        self.adjust = adjust
        self.history_days = history_days
        self.cache = DiskCache("prices_yf", ttl_hours=cache_ttl_hours)

    def get_history(
        self,
        symbol: str,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        ticker = _yf_ticker(symbol)
        key = f"{ticker}|{start}|{end}|{self.adjust}|{self.history_days}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        try:
            import yfinance as yf
        except ImportError:
            log.error("yfinance not installed; cannot fetch %s", ticker)
            return pd.DataFrame(columns=OHLCV)

        kwargs = dict(auto_adjust=self.adjust, progress=False, threads=False)
        if start is not None:
            kwargs["start"] = start
            kwargs["end"] = end or (dt.date.today() + dt.timedelta(days=1))
        else:
            kwargs["period"] = f"{self.history_days}d"

        try:
            raw = yf.download(ticker, **kwargs)
        except Exception as exc:
            log.warning("yfinance download failed for %s: %s", ticker, exc)
            return pd.DataFrame(columns=OHLCV)

        df = normalize_ohlcv(raw)
        if len(df):
            self.cache.set(key, df)
        return df


class UpstoxPriceProvider(PriceProvider):
    """Optional: Upstox Historical Candle V3 (daily data from Jan 2000).

    Endpoint: ``GET /v3/historical-candle/{key}/days/1/{to}/{from}``.
    Requires ``UPSTOX_ACCESS_TOKEN`` (expires daily at 03:30 IST — see README).
    Instrument keys come from the universe CSV (``NSE_EQ|<ISIN>``).
    """

    name = "upstox"
    BASE = "https://api.upstox.com"

    def __init__(self, access_token: Optional[str] = None,
                 key_resolver=None, cache_ttl_hours: float = 20.0,
                 history_days: int = 800) -> None:
        self.access_token = access_token or os.getenv("UPSTOX_ACCESS_TOKEN", "")
        # key_resolver: symbol -> "NSE_EQ|INE..."; defaults to the universe map.
        self.key_resolver = key_resolver
        self.history_days = history_days
        self.cache = DiskCache("prices_upstox", ttl_hours=cache_ttl_hours)

    def _resolve_key(self, symbol: str) -> str:
        if symbol.startswith("NSE_") or "|" in symbol:
            return symbol
        if self.key_resolver:
            return self.key_resolver(symbol)
        from config.loader import universe_map

        st = universe_map().get(symbol.upper())
        return st.resolved_upstox_key if st else ""

    def get_history(
        self,
        symbol: str,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        if not self.access_token:
            raise RuntimeError("UPSTOX_ACCESS_TOKEN not set; cannot use Upstox provider")
        key = self._resolve_key(symbol)
        if not key:
            raise ValueError(f"No Upstox instrument key for {symbol}")

        to_date = (end or dt.date.today()).isoformat()
        from_date = (start or dt.date.today() - dt.timedelta(days=self.history_days)).isoformat()
        cache_key = f"{key}|{from_date}|{to_date}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        import requests

        url = f"{self.BASE}/v3/historical-candle/{quote(key, safe='')}/days/1/{to_date}/{from_date}"
        headers = {"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        candles = resp.json().get("data", {}).get("candles", [])
        # candle = [timestamp, open, high, low, close, volume, oi]
        rows = [
            {"date": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5]}
            for c in candles
        ]
        df = normalize_ohlcv(pd.DataFrame(rows))
        if len(df):
            self.cache.set(cache_key, df)
        return df


class DhanPriceProvider(PriceProvider):
    """Optional: Dhan Data API (paid). Daily historical via ``/v2/charts/historical``."""

    name = "dhan"
    BASE = "https://api.dhan.co"

    def __init__(self, client_id: Optional[str] = None, access_token: Optional[str] = None,
                 cache_ttl_hours: float = 20.0, history_days: int = 800) -> None:
        self.client_id = client_id or os.getenv("DHAN_CLIENT_ID", "")
        self.access_token = access_token or os.getenv("DHAN_ACCESS_TOKEN", "")
        self.history_days = history_days
        self.cache = DiskCache("prices_dhan", ttl_hours=cache_ttl_hours)

    def get_history(
        self,
        symbol: str,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        if not (self.client_id and self.access_token):
            raise RuntimeError("DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set")
        from config.loader import universe_map

        st = universe_map().get(symbol.upper())
        if st is None or not st.isin:
            raise ValueError(f"No ISIN for {symbol}; Dhan requires a security id mapping")

        import requests

        payload = {
            "securityId": st.isin,  # NOTE: Dhan uses numeric securityId; map via Dhan scrip master
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "fromDate": (start or dt.date.today() - dt.timedelta(days=self.history_days)).isoformat(),
            "toDate": (end or dt.date.today()).isoformat(),
        }
        headers = {"access-token": self.access_token, "client-id": self.client_id}
        resp = requests.post(f"{self.BASE}/v2/charts/historical", json=payload,
                             headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(data.get("timestamp", []), unit="s"),
                "open": data.get("open", []),
                "high": data.get("high", []),
                "low": data.get("low", []),
                "close": data.get("close", []),
                "volume": data.get("volume", []),
            }
        )
        return normalize_ohlcv(df)


def get_price_provider(settings=None) -> PriceProvider:
    """Factory: build the configured price provider."""
    if settings is None:
        from config.loader import get_settings

        settings = get_settings()
    src = settings.data.primary_price_source.lower()
    kwargs = dict(cache_ttl_hours=settings.data.cache_ttl_hours, history_days=settings.data.history_days)
    if src == "upstox":
        return UpstoxPriceProvider(**kwargs)
    if src == "dhan":
        return DhanPriceProvider(**kwargs)
    return YFinancePriceProvider(adjust=settings.data.adjust_ohlc, **kwargs)


def fetch_universe_history(
    symbols: list[str],
    settings=None,
    throttle_s: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Fetch daily history for many symbols using the configured provider."""
    provider = get_price_provider(settings)
    out: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        try:
            out[sym] = provider.get_history(sym)
        except Exception as exc:
            log.warning("history fetch failed for %s: %s", sym, exc)
            out[sym] = pd.DataFrame(columns=OHLCV)
        if throttle_s:
            time.sleep(throttle_s)
    return out
