"""Native technical indicators (numpy/pandas only).

Implemented here rather than via pandas-ta/TA-Lib so the whole analysis stack is
dependency-light, numpy-2 safe, and deterministically unit-testable. RSI/ATR/ADX
use Wilder's smoothing (RMA) to match standard charting platforms.

All functions take/return pandas objects aligned to the input index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing (a.k.a. RMA): EMA with alpha = 1/length."""
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _rma(gain, length)
    avg_loss = _rma(loss, length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # All-gains -> 100; all-losses -> 0.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(avg_gain != 0, out.where(avg_loss == 0, 0.0))
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def bollinger(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, length)
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    upper = mid + std * sd
    lower = mid - std * sd
    width = (upper - lower) / mid.replace(0.0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "bandwidth": width, "pct_b": pct_b}
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder's Average True Range (absolute price units)."""
    return _rma(true_range(df), length)


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3) -> pd.DataFrame:
    low_k = df["low"].rolling(k, min_periods=k).min()
    high_k = df["high"].rolling(k, min_periods=k).max()
    raw_k = 100.0 * (df["close"] - low_k) / (high_k - low_k).replace(0.0, np.nan)
    k_line = raw_k.rolling(smooth, min_periods=smooth).mean()
    d_line = k_line.rolling(d, min_periods=d).mean()
    return pd.DataFrame({"k": k_line, "d": d_line})


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI / -DI."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    tr = true_range(df)
    atr_ = _rma(tr, length)
    plus_di = 100.0 * _rma(plus_dm, length) / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * _rma(minus_dm, length) / atr_.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_ = _rma(dx, length)
    return pd.DataFrame({"adx": adx_, "plus_di": plus_di, "minus_di": minus_di})


def rolling_high(series: pd.Series, length: int) -> pd.Series:
    """Highest value over the *previous* ``length`` bars (excludes current)."""
    return series.shift(1).rolling(length, min_periods=1).max()


def rolling_low(series: pd.Series, length: int) -> pd.Series:
    return series.shift(1).rolling(length, min_periods=1).min()


def crossed_above(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on bars where ``fast`` crosses from below to above ``slow``."""
    prev = fast.shift(1) <= slow.shift(1)
    now = fast > slow
    return prev & now


def crossed_below(fast: pd.Series, slow: pd.Series) -> pd.Series:
    prev = fast.shift(1) >= slow.shift(1)
    now = fast < slow
    return prev & now


def pct_rank(series: pd.Series, length: int) -> pd.Series:
    """Rolling percentile rank of the latest value within the window [0,1]."""
    return series.rolling(length, min_periods=length).apply(
        lambda w: (w <= w.iloc[-1]).mean(), raw=False
    )
