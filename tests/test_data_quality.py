import pandas as pd

from data_ingestion.base import normalize_ohlcv
from data_ingestion.validators import sanity_checks


def test_normalize_handles_adj_close_without_crash():
    # auto_adjust=False yields BOTH 'Close' and 'Adj Close' -> must not collide.
    idx = pd.date_range("2026-01-01", periods=3, freq="B")
    df = pd.DataFrame(
        {"Open": [1, 2, 3], "High": [1, 2, 3], "Low": [1, 2, 3],
         "Close": [1, 2, 3], "Adj Close": [0.9, 1.8, 2.7], "Volume": [10, 20, 30]},
        index=idx,
    )
    out = normalize_ohlcv(df)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out["close"].tolist() == [1.0, 2.0, 3.0]  # raw close kept; no dup-column crash


def test_normalize_multiindex_with_adj_close():
    idx = pd.date_range("2026-01-01", periods=2, freq="B")
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["X.NS"]]
    )
    df = pd.DataFrame([[1, 1, 1, 1, 0.9, 10], [2, 2, 2, 2, 1.8, 20]], index=idx, columns=cols)
    out = normalize_ohlcv(df)
    assert "close" in out.columns and len(out) == 2


def test_sanity_flags_zero_volume_on_nonflat_bar():
    idx = pd.date_range("2026-01-01", periods=1, freq="B")
    df = pd.DataFrame({"open": [10], "high": [12], "low": [9], "close": [11], "volume": [0]}, index=idx)
    assert "bad_volume" in {i.kind for i in sanity_checks(df)}


def test_sanity_allows_zero_volume_on_flat_bar():
    idx = pd.date_range("2026-01-01", periods=1, freq="B")
    df = pd.DataFrame({"open": [10], "high": [10], "low": [10], "close": [10], "volume": [0]}, index=idx)
    kinds = {i.kind for i in sanity_checks(df)}
    assert "bad_volume" not in kinds  # zero volume OK on a flat (no-trade) bar
