"""Input bundles passed to analyzers.

``MarketContext`` holds market-wide data fetched once per run; ``StockContext``
holds per-stock data. Every analyzer's ``analyze(sctx, mctx)`` reads what it
needs from these, which keeps analyzers individually testable (build a minimal
context with just the fields that analyzer uses).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from config.schema import Settings, Stock


def _empty_series() -> pd.Series:
    return pd.Series(dtype="float64")


@dataclass
class MarketContext:
    as_of: dt.date
    settings: Settings
    vix: pd.Series = field(default_factory=_empty_series)
    macro: dict[str, pd.Series] = field(default_factory=dict)
    fii_dii: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark: pd.Series = field(default_factory=_empty_series)
    sector_rs: dict[str, float] = field(default_factory=dict)  # sector label -> RS [-1,1]
    regime: dict[str, Any] = field(default_factory=dict)
    fno_ban: set[str] = field(default_factory=set)


@dataclass
class StockContext:
    stock: Stock
    price: pd.DataFrame
    fundamentals: dict[str, Any] = field(default_factory=dict)
    headlines: list[dict[str, Any]] = field(default_factory=list)
    earnings_date: Optional[dt.date] = None

    @property
    def symbol(self) -> str:
        return self.stock.symbol

    @property
    def sector(self) -> str:
        return self.stock.sector
