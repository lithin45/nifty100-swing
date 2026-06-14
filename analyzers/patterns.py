"""Chart-pattern detection via scipy pivots + geometric rules.

Each detector returns a :class:`Pattern` with a *confidence* (0–1), not just a
boolean, plus an optional measured-move ``target`` and the ``neckline`` used.
Bullish patterns (double bottom, inverse H&S, ascending triangle, bull flag,
cup-and-handle) feed the technical score and the trade target; bearish ones
(double top, H&S) feed exits/penalties.

Heuristic by design — these confirm/strengthen a thesis, they are not a
standalone strategy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:  # scipy is a core dep, but degrade gracefully if absent
    from scipy.signal import argrelextrema

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


@dataclass
class Pattern:
    name: str
    bullish: bool
    confidence: float
    target: Optional[float] = None
    neckline: Optional[float] = None
    reason: str = ""


def _pivots(values: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of local maxima and minima."""
    if _HAS_SCIPY:
        maxs = argrelextrema(values, np.greater, order=order)[0]
        mins = argrelextrema(values, np.less, order=order)[0]
        return maxs, mins
    # Fallback: simple neighbour comparison.
    maxs, mins = [], []
    for i in range(order, len(values) - order):
        window = values[i - order : i + order + 1]
        if values[i] == window.max() and (window == values[i]).sum() == 1:
            maxs.append(i)
        if values[i] == window.min() and (window == values[i]).sum() == 1:
            mins.append(i)
    return np.array(maxs, dtype=int), np.array(mins, dtype=int)


def _similar(a: float, b: float, tol: float) -> bool:
    base = min(abs(a), abs(b)) or 1.0
    return abs(a - b) / base <= tol


def _double_bottom(close: np.ndarray, mins: np.ndarray, maxs: np.ndarray,
                   tol: float = 0.04) -> Optional[Pattern]:
    if len(mins) < 2:
        return None
    i1, i2 = mins[-2], mins[-1]
    b1, b2 = close[i1], close[i2]
    if not _similar(b1, b2, tol):
        return None
    between = [m for m in maxs if i1 < m < i2]
    if not between:
        return None
    neckline = max(close[m] for m in between)
    avg_bottom = (b1 + b2) / 2.0
    if neckline <= avg_bottom * 1.03:  # need a real intervening peak
        return None
    last = close[-1]
    breakout = last > neckline
    sim = 1.0 - abs(b1 - b2) / (avg_bottom or 1.0)
    conf = 0.45 + 0.25 * sim + (0.2 if breakout else 0.0)
    conf += 0.1 if (last > avg_bottom and not breakout) else 0.0
    target = neckline + (neckline - avg_bottom)
    state = "confirmed breakout" if breakout else "forming"
    return Pattern("double_bottom", True, min(conf, 0.95), target, neckline,
                   f"Double bottom near {avg_bottom:.1f}, neckline {neckline:.1f} ({state})")


def _double_top(close: np.ndarray, mins: np.ndarray, maxs: np.ndarray,
                tol: float = 0.04) -> Optional[Pattern]:
    if len(maxs) < 2:
        return None
    i1, i2 = maxs[-2], maxs[-1]
    t1, t2 = close[i1], close[i2]
    if not _similar(t1, t2, tol):
        return None
    between = [m for m in mins if i1 < m < i2]
    if not between:
        return None
    neckline = min(close[m] for m in between)
    avg_top = (t1 + t2) / 2.0
    breakdown = close[-1] < neckline
    conf = 0.5 + (0.25 if breakdown else 0.0)
    return Pattern("double_top", False, min(conf, 0.9), None, neckline,
                   f"Double top near {avg_top:.1f} ({'breakdown' if breakdown else 'forming'})")


def _inverse_hs(close: np.ndarray, mins: np.ndarray, maxs: np.ndarray,
                tol: float = 0.06) -> Optional[Pattern]:
    if len(mins) < 3:
        return None
    ls, head, rs = mins[-3], mins[-2], mins[-1]
    l, h, r = close[ls], close[head], close[rs]
    if not (h < l and h < r and _similar(l, r, tol)):
        return None
    peaks = [m for m in maxs if ls < m < rs]
    if len(peaks) < 1:
        return None
    neckline = float(np.mean([close[m] for m in peaks]))
    last = close[-1]
    breakout = last > neckline
    conf = 0.5 + 0.2 * (1.0 - abs(l - r) / (max(l, r) or 1.0)) + (0.2 if breakout else 0.0)
    target = neckline + (neckline - h)
    return Pattern("inverse_head_shoulders", True, min(conf, 0.95), target, neckline,
                   f"Inverse H&S, head {h:.1f}, neckline {neckline:.1f} "
                   f"({'confirmed' if breakout else 'forming'})")


def _head_shoulders(close: np.ndarray, mins: np.ndarray, maxs: np.ndarray,
                    tol: float = 0.06) -> Optional[Pattern]:
    if len(maxs) < 3:
        return None
    ls, head, rs = maxs[-3], maxs[-2], maxs[-1]
    l, h, r = close[ls], close[head], close[rs]
    if not (h > l and h > r and _similar(l, r, tol)):
        return None
    troughs = [m for m in mins if ls < m < rs]
    if not troughs:
        return None
    neckline = float(np.mean([close[m] for m in troughs]))
    breakdown = close[-1] < neckline
    conf = 0.5 + (0.2 if breakdown else 0.0)
    return Pattern("head_shoulders", False, min(conf, 0.9), None, neckline,
                   f"Head & shoulders top, neckline {neckline:.1f}")


def _ascending_triangle(close: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> Optional[Pattern]:
    if len(maxs) < 2 or len(mins) < 2:
        return None
    top1, top2 = close[maxs[-2]], close[maxs[-1]]
    low1, low2 = close[mins[-2]], close[mins[-1]]
    flat_top = _similar(top1, top2, 0.03)
    rising_lows = low2 > low1 * 1.01
    if not (flat_top and rising_lows):
        return None
    resistance = (top1 + top2) / 2.0
    breakout = close[-1] > resistance
    conf = 0.5 + (0.25 if breakout else 0.1)
    target = resistance + (resistance - low1)
    return Pattern("ascending_triangle", True, min(conf, 0.9), target, resistance,
                   f"Ascending triangle, resistance {resistance:.1f}, rising lows")


def _bull_flag(close: np.ndarray, lookback: int = 40) -> Optional[Pattern]:
    n = len(close)
    if n < lookback:
        return None
    pole = close[-lookback : -lookback // 3]
    flag = close[-lookback // 3 :]
    if len(pole) < 3 or len(flag) < 3:
        return None
    pole_gain = (pole[-1] - pole[0]) / (pole[0] or 1.0)
    flag_range = (flag.max() - flag.min()) / (flag.mean() or 1.0)
    drift = (flag[-1] - flag[0]) / (flag[0] or 1.0)
    if pole_gain > 0.10 and flag_range < 0.08 and -0.05 < drift <= 0.02:
        conf = 0.5 + min(pole_gain, 0.3)
        target = flag[-1] * (1 + pole_gain)
        return Pattern("bull_flag", True, min(conf, 0.9), target, None,
                       f"Bull flag: +{pole_gain*100:.0f}% pole, tight {flag_range*100:.0f}% flag")
    return None


def _cup_and_handle(close: np.ndarray, lookback: int = 90) -> Optional[Pattern]:
    n = len(close)
    if n < lookback:
        return None
    seg = close[-lookback:]
    rim_left = seg[: max(3, lookback // 10)].max()
    bottom = seg.min()
    bottom_idx = int(np.argmin(seg))
    # Cup: bottom roughly central, depth meaningful, right side recovers to rim.
    central = lookback * 0.25 < bottom_idx < lookback * 0.75
    depth = (rim_left - bottom) / (rim_left or 1.0)
    recovered = seg[-1] >= rim_left * 0.95
    handle = seg[-max(3, lookback // 12) :]
    handle_pullback = (handle.max() - handle[-1]) / (handle.max() or 1.0)
    if central and 0.10 < depth < 0.5 and recovered and 0.0 <= handle_pullback < 0.10:
        conf = 0.45 + min(depth, 0.3)
        target = rim_left + (rim_left - bottom)
        return Pattern("cup_and_handle", True, min(conf, 0.9), target, rim_left,
                       f"Cup & handle: {depth*100:.0f}% deep cup, small handle")
    return None


def detect_patterns(df: pd.DataFrame, settings=None) -> list[Pattern]:
    """Run all detectors over the recent window; return detected patterns."""
    if settings is None:
        from config.loader import get_settings

        settings = get_settings()
    lookback = settings.patterns.lookback
    order = settings.technical.pivots.window
    min_conf = settings.patterns.min_confidence

    if df is None or len(df) < max(order * 2 + 1, 20):
        return []

    seg = df.tail(lookback)
    close = seg["close"].to_numpy(dtype=float)
    maxs, mins = _pivots(close, order)

    candidates = [
        _double_bottom(close, mins, maxs),
        _double_top(close, mins, maxs),
        _inverse_hs(close, mins, maxs),
        _head_shoulders(close, mins, maxs),
        _ascending_triangle(close, mins, maxs),
        _bull_flag(close),
        _cup_and_handle(close),
    ]
    return [p for p in candidates if p is not None and p.confidence >= min_conf]


def best_bullish_pattern(patterns: list[Pattern]) -> Optional[Pattern]:
    bull = [p for p in patterns if p.bullish]
    return max(bull, key=lambda p: p.confidence) if bull else None


def best_bearish_pattern(patterns: list[Pattern]) -> Optional[Pattern]:
    bear = [p for p in patterns if not p.bullish]
    return max(bear, key=lambda p: p.confidence) if bear else None
