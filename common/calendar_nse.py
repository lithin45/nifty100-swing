"""NSE trading calendar — weekends + exchange holidays.

Used by walk-forward backtesting (no look-ahead, skip non-trading days), the
event gate ("results in N **trading** days"), and next-trading-day reference
prices.

IMPORTANT: NSE publishes its trading-holiday list annually. The set below is a
best-effort list for 2024–2026 and **must be verified/updated** each year from
the official circular:  https://www.nseindia.com/resources/exchange-communication-holidays
You can also append holidays at runtime via :func:`register_holidays`.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

# Trading holidays (exchange fully closed). Muhurat / special sessions are NOT
# listed here because the exchange is effectively open on those days.
_HOLIDAY_STRINGS: set[str] = {
    # ---- 2024 ----
    "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29", "2024-04-11",
    "2024-04-17", "2024-05-01", "2024-06-17", "2024-07-17", "2024-08-15",
    "2024-10-02", "2024-11-01", "2024-11-15", "2024-12-25",
    # ---- 2025 ----
    "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14",
    "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02",
    "2025-10-21", "2025-10-22", "2025-11-05", "2025-12-25",
    # ---- 2026 (verified June 2026 against NSE list via cleartax + groww) ----
    # Aug 15 (Independence Day) falls on a Saturday in 2026, so it is not a
    # separate weekday closure. Diwali Laxmi Pujan (Nov 8) is a Sunday (Muhurat).
    "2026-01-15", "2026-01-26", "2026-03-03", "2026-03-26", "2026-03-31",
    "2026-04-03", "2026-04-14", "2026-05-01", "2026-05-28", "2026-06-26",
    "2026-09-14", "2026-10-02", "2026-10-20", "2026-11-10", "2026-11-24",
    "2026-12-25",
}

NSE_HOLIDAYS: set[date] = {date.fromisoformat(s) for s in _HOLIDAY_STRINGS}


def register_holidays(days: Iterable[str | date]) -> None:
    """Add extra trading holidays (e.g. loaded from config)."""
    for d in days:
        NSE_HOLIDAYS.add(d if isinstance(d, date) else date.fromisoformat(str(d)))


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # 5 = Sat, 6 = Sun


def is_trading_day(d: date) -> bool:
    """True if NSE trades on ``d`` (not a weekend, not a holiday)."""
    return not is_weekend(d) and d not in NSE_HOLIDAYS


def next_trading_day(d: date, inclusive: bool = False) -> date:
    """Next NSE trading day strictly after ``d`` (or on/after if inclusive)."""
    cur = d if inclusive else d + timedelta(days=1)
    while not is_trading_day(cur):
        cur += timedelta(days=1)
    return cur


def previous_trading_day(d: date, inclusive: bool = False) -> date:
    """Previous NSE trading day strictly before ``d`` (or on/before if inclusive)."""
    cur = d if inclusive else d - timedelta(days=1)
    while not is_trading_day(cur):
        cur -= timedelta(days=1)
    return cur


def add_trading_days(d: date, n: int) -> date:
    """Advance ``n`` trading days from ``d`` (``n`` may be negative)."""
    if n == 0:
        return next_trading_day(d, inclusive=True)
    step = 1 if n > 0 else -1
    cur = d
    remaining = abs(n)
    while remaining > 0:
        cur += timedelta(days=step)
        if is_trading_day(cur):
            remaining -= 1
    return cur


def trading_days_between(start: date, end: date) -> list[date]:
    """Inclusive list of trading days in ``[start, end]``."""
    if end < start:
        return []
    out: list[date] = []
    cur = start
    while cur <= end:
        if is_trading_day(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


def trading_days_until(target: date, frm: date | None = None) -> int:
    """Number of trading days from ``frm`` (default today's logic supplied by
    caller) up to and including ``target``. Negative if target is in the past.

    Callers pass ``frm`` explicitly to keep this function pure/testable.
    """
    if frm is None:
        raise ValueError("frm must be provided explicitly for deterministic results")
    if target == frm:
        return 0
    sign = 1 if target > frm else -1
    lo, hi = (frm, target) if target > frm else (target, frm)
    # count trading days strictly after lo, up to and including hi
    count = len([d for d in trading_days_between(lo, hi) if d != lo])
    return sign * count
