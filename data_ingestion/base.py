"""Swappable data-source interfaces + shared ingestion infrastructure.

Every concrete data source implements one of the abstract base classes here, so
the rest of the system depends on the *interface*, never on yfinance/NSE/Upstox
directly. Swap a provider by changing ``data.primary_price_source`` in
``settings.yaml`` (or by registering a new class).

Also provides:
* :data:`OHLCV` — canonical column order.
* :func:`normalize_ohlcv` — coerce any provider's frame into the canonical shape.
* :class:`DiskCache` — TTL pickle cache so we don't refetch within a trading day.
"""
from __future__ import annotations

import abc
import datetime as dt
import hashlib
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from common.logging_config import get_logger
from common.paths import CACHE_DIR, ensure_data_dir

log = get_logger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]

_COLUMN_ALIASES = {
    "open": "open", "high": "high", "low": "low", "close": "close",
    "adj close": "close", "adj_close": "close", "adjclose": "close",
    "volume": "volume", "vol": "volume",
    "ltp": "close", "last": "close", "prev close": "close",
}


def normalize_ohlcv(df: pd.DataFrame, *, drop_na: bool = True) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame into the canonical shape.

    * lower-cased, aliased columns -> ``open/high/low/close/volume``
    * a sorted, de-duplicated ``DatetimeIndex`` named ``date``
    * numeric dtypes; optional drop of all-NaN price rows
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OHLCV)

    out = df.copy()

    # Flatten yfinance MultiIndex columns (Price, Ticker).
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    out.columns = [str(c).strip().lower() for c in out.columns]
    out = out.rename(columns={c: _COLUMN_ALIASES[c] for c in out.columns if c in _COLUMN_ALIASES})

    # Index -> DatetimeIndex.
    if not isinstance(out.index, pd.DatetimeIndex):
        for cand in ("date", "datetime", "timestamp", "time"):
            if cand in out.columns:
                out = out.set_index(cand)
                break
    out.index = pd.to_datetime(out.index, errors="coerce")
    out.index.name = "date"
    # Strip intraday time component to a pure date-at-midnight index.
    out.index = out.index.normalize()

    for col in OHLCV:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[OHLCV]
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if drop_na:
        out = out.dropna(subset=["close"])
    return out


def _safe_key(key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", key)[:80]
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


class DiskCache:
    """Simple TTL pickle cache under ``data/cache/<namespace>/``."""

    def __init__(self, namespace: str, ttl_hours: float = 20.0) -> None:
        ensure_data_dir()
        self.dir = CACHE_DIR / namespace
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = dt.timedelta(hours=ttl_hours)

    def _path(self, key: str) -> Path:
        return self.dir / f"{_safe_key(key)}.pkl"

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        if not p.exists():
            return None
        age = dt.datetime.now() - dt.datetime.fromtimestamp(p.stat().st_mtime)
        if age > self.ttl:
            return None
        try:
            with open(p, "rb") as fh:
                return pickle.load(fh)
        except Exception as exc:  # corrupt cache entry
            log.warning("cache read failed for %s: %s", key, exc)
            return None

    def set(self, key: str, value: Any) -> None:
        try:
            with open(self._path(key), "wb") as fh:
                pickle.dump(value, fh)
        except Exception as exc:
            log.warning("cache write failed for %s: %s", key, exc)


# --------------------------------------------------------------------------- #
# Abstract provider interfaces                                                 #
# --------------------------------------------------------------------------- #
class PriceProvider(abc.ABC):
    """Daily OHLCV history for a single instrument."""

    name: str = "base"

    @abc.abstractmethod
    def get_history(
        self,
        symbol: str,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        """Return canonical OHLCV indexed by date (ascending)."""

    def get_history_batch(
        self,
        symbols: list[str],
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> dict[str, pd.DataFrame]:
        """Default sequential batch; providers may override for bulk endpoints."""
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                out[sym] = self.get_history(sym, start, end)
            except Exception as exc:
                log.warning("price fetch failed for %s: %s", sym, exc)
                out[sym] = pd.DataFrame(columns=OHLCV)
        return out


class FundamentalsProvider(abc.ABC):
    @abc.abstractmethod
    def get_fundamentals(self, symbol: str) -> dict[str, Any]:
        """Return a dict with at least: roe, pe, ps, de, earnings_growth."""


class FiiDiiProvider(abc.ABC):
    @abc.abstractmethod
    def get_recent(self, lookback_days: int = 5) -> pd.DataFrame:
        """Return recent FII/DII net cash flows (₹ crore) indexed by date."""


class VixProvider(abc.ABC):
    @abc.abstractmethod
    def get_history(self, lookback_days: int = 60) -> pd.Series:
        """India VIX close series indexed by date."""

    def get_latest(self) -> Optional[float]:
        s = self.get_history(lookback_days=10)
        return float(s.iloc[-1]) if len(s) else None


class MacroProvider(abc.ABC):
    @abc.abstractmethod
    def get_series(self, key: str, lookback_days: int = 60) -> pd.Series:
        """Close series for a macro symbol key (usdinr/crude/sp500/nasdaq)."""


class NewsProvider(abc.ABC):
    @abc.abstractmethod
    def get_headlines(self, max_age_days: int = 7) -> list[dict[str, Any]]:
        """Return headlines as dicts: title, summary, link, published, source."""


class EventsProvider(abc.ABC):
    @abc.abstractmethod
    def get_earnings_date(self, symbol: str) -> Optional[dt.date]:
        """Next results/earnings date for a symbol, if known."""

    def get_fno_ban_list(self) -> set[str]:
        return set()

    def is_in_circuit(self, symbol: str) -> bool:
        return False


class SectorProvider(abc.ABC):
    @abc.abstractmethod
    def get_index_close(self, index_symbol: str, lookback_days: int = 90) -> pd.Series:
        """Close series for a sectoral / benchmark index."""
