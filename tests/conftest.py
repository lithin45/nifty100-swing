"""Shared pytest fixtures (deterministic synthetic market data)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from config.loader import load_settings
from config.schema import Stock


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def asof():
    return dt.date(2026, 6, 12)


def _make_df(closes, vols=None, start="2025-06-02"):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="B")
    high = closes * 1.012
    low = closes * 0.988
    openp = np.r_[closes[0], closes[:-1]]
    vol = vols if vols is not None else np.full(n, 3_000_000.0)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": closes, "volume": vol}, index=idx
    )


@pytest.fixture
def strong_uptrend_df():
    """260-bar uptrend ending in a volume-confirmed breakout (high turnover)."""
    n = 260
    closes = np.linspace(100, 175, n)
    closes[-1] = closes[-2] * 1.03
    vols = np.full(n, 4_000_000.0)
    vols[-1] = 9_000_000.0
    return _make_df(closes, vols)


@pytest.fixture
def downtrend_df():
    n = 260
    return _make_df(np.linspace(175, 100, n))


@pytest.fixture
def reliance():
    return Stock("RELIANCE", "Oil & Gas", isin="INE002A01018", name="Reliance Industries")


@pytest.fixture
def make_df():
    return _make_df
