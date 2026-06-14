import numpy as np
import pandas as pd

from analyzers import indicators as I


def test_sma_ema_basic():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert I.sma(s, 3).iloc[-1] == 4.0
    assert np.isnan(I.sma(s, 3).iloc[0])
    assert I.ema(s, 2).iloc[-1] > 0


def test_rsi_bounds():
    rising = pd.Series(np.arange(1, 30), dtype=float)
    falling = pd.Series(np.arange(30, 1, -1), dtype=float)
    assert round(I.rsi(rising, 14).iloc[-1], 2) == 100.0
    assert round(I.rsi(falling, 14).iloc[-1], 2) == 0.0


def test_atr_positive():
    df = pd.DataFrame({
        "high": [10, 11, 12, 11, 13, 14, 13, 15, 16, 15],
        "low": [9, 9.5, 10, 10, 11, 12, 12, 13, 14, 14],
        "close": [9.5, 10.5, 11.5, 10.5, 12.5, 13.5, 12.5, 14.5, 15.5, 14.5],
    })
    atr = I.atr(df, 3)
    assert atr.iloc[-1] > 0


def test_crossovers():
    fast = pd.Series([1, 2, 3, 4, 5], dtype=float)
    slow = pd.Series([3, 3, 3, 3, 3], dtype=float)
    assert bool(I.crossed_above(fast, slow).iloc[3])
    assert bool(I.crossed_below(slow, fast).iloc[3])


def test_rolling_high_excludes_current_bar():
    s = pd.Series([10, 12, 11, 15, 9], dtype=float)
    rh = I.rolling_high(s, 2)
    assert rh.iloc[3] == 12.0  # max of [12, 11], not including 15


def test_macd_bollinger_columns():
    s = pd.Series(np.linspace(100, 120, 60))
    assert list(I.macd(s).columns) == ["macd", "signal", "hist"]
    assert set(I.bollinger(s, 20).columns) == {"mid", "upper", "lower", "bandwidth", "pct_b"}
