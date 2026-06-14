"""Sector / thematic strength: relative strength of a stock's sector vs Nifty 100.

``compute_sector_rs`` (called once per run by the pipeline) builds a map of
``sector label -> RS in [-1, 1]``; the analyzer just reads it from the
MarketContext. RS = sector index return minus benchmark return over the lookback,
clipped so a ~15% relative move saturates to ±1.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from common.logging_config import get_logger
from common.types import SubScore, bipolar_to_unit
from analyzers.context import MarketContext, StockContext

log = get_logger(__name__)

# Map universe sector labels -> a sectoral index key in settings.sectors.indices.
SECTOR_TO_INDEX: dict[str, str] = {
    "IT": "Nifty IT",
    "Bank": "Nifty Bank",
    "PSU Bank": "Nifty Bank",
    "Financial Services": "Nifty Financial Services",
    "Insurance": "Nifty Financial Services",
    "Pharma": "Nifty Pharma",
    "Healthcare": "Nifty Pharma",
    "Auto": "Nifty Auto",
    "FMCG": "Nifty FMCG",
    "Metal": "Nifty Metal",
    "Energy": "Nifty Energy",
    "Oil & Gas": "Nifty Energy",
    "Power": "Nifty Energy",
    "Realty": "Nifty Realty",
    "Media": "Nifty Media",
}

_RS_SATURATION = 0.15  # 15% relative outperformance -> RS = 1.0


def _ret(series: pd.Series, lookback: int) -> Optional[float]:
    s = series.dropna()
    if len(s) < 2:
        return None
    n = min(lookback, len(s) - 1)
    base = s.iloc[-1 - n]
    return float(s.iloc[-1] / base - 1.0) if base else None


def compute_sector_rs(provider, settings) -> dict[str, float]:
    """Return ``sector label -> RS [-1,1]`` using a SectorProvider."""
    lookback = settings.sectors.rs_lookback_days
    bench = provider.get_index_close(settings.sectors.benchmark, lookback + 30)
    bench_ret = _ret(bench, lookback)

    rs_by_index: dict[str, float] = {}
    for index_name, ticker in settings.sectors.indices.items():
        series = provider.get_index_close(ticker, lookback + 30)
        idx_ret = _ret(series, lookback)
        if idx_ret is None or bench_ret is None:
            continue
        rel = idx_ret - bench_ret
        rs_by_index[index_name] = max(-1.0, min(1.0, rel / _RS_SATURATION))

    # Map back to universe sector labels.
    rs_by_label: dict[str, float] = {}
    for label, index_name in SECTOR_TO_INDEX.items():
        if index_name in rs_by_index:
            rs_by_label[label] = rs_by_index[index_name]
    return rs_by_label


class SectorFactorAnalyzer:
    key = "sector"

    def analyze(self, sctx: StockContext, mctx: MarketContext) -> SubScore:
        label = sctx.sector
        rs = mctx.sector_rs.get(label)
        if rs is None:
            return SubScore(self.key, 0.5, f"No sector RS for '{label}' (neutral)",
                            raw=0.0, details={"sector": label})
        tone = "outperforming" if rs > 0.1 else "lagging" if rs < -0.1 else "in-line with"
        reason = f"Sector '{label}' {tone} Nifty 100 (RS {rs:+.2f})"
        return SubScore(self.key, bipolar_to_unit(rs), reason, raw=rs,
                        details={"sector": label, "rs": round(rs, 3)})


def is_sector_positive(sector_label: str, mctx: MarketContext) -> bool:
    """Used by the trend gate override and sector-rollover exit."""
    return mctx.sector_rs.get(sector_label, 0.0) > 0.0


def analyze_sector(sctx: StockContext, mctx: MarketContext) -> SubScore:
    return SectorFactorAnalyzer().analyze(sctx, mctx)
