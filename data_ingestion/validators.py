"""Data-quality validation and repair.

Two layers:

1. :func:`sanity_checks` — provider-agnostic checks that need no network:
   flat bars (O=H=L=C), non-positive volume on non-flat days, NaNs, duplicate /
   non-monotonic dates, and implausible single-day jumps. Fully unit-tested.

2. :class:`BhavcopyValidator` — cross-checks the latest bars against the official
   NSE bhavcopy (via jugaad-data / nsepython) and optionally repairs bars that
   deviate materially. Degrades gracefully when the NSE libs/network are absent.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from common.logging_config import get_logger
from data_ingestion.base import DiskCache, normalize_ohlcv

log = get_logger(__name__)


@dataclass
class DataIssue:
    date: Optional[dt.date]
    kind: str
    detail: str


@dataclass
class ValidationResult:
    symbol: str
    issues: list[DataIssue] = field(default_factory=list)
    repaired_dates: list[dt.date] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0

    @property
    def n_issues(self) -> int:
        return len(self.issues)


def sanity_checks(
    df: pd.DataFrame,
    *,
    max_daily_jump_pct: float = 35.0,
    recent_only: int = 0,
) -> list[DataIssue]:
    """Return a list of data-quality issues found in an OHLCV frame.

    Parameters
    ----------
    max_daily_jump_pct : flag a close-to-close move larger than this (likely a
        bad/unadjusted bar). Default 35% (above typical NSE 20% circuit + gap).
    recent_only : if >0, only inspect the last N rows (cheap EOD check).
    """
    issues: list[DataIssue] = []
    if df is None or len(df) == 0:
        issues.append(DataIssue(None, "empty", "no rows returned"))
        return issues

    frame = df.tail(recent_only) if recent_only else df

    # Monotonic, unique dates.
    if not frame.index.is_monotonic_increasing:
        issues.append(DataIssue(None, "unsorted", "index is not monotonically increasing"))
    if frame.index.has_duplicates:
        issues.append(DataIssue(None, "duplicate_dates", "duplicate dates present"))

    for ts, row in frame.iterrows():
        d = ts.date() if hasattr(ts, "date") else None
        o, h, l, c, v = row["open"], row["high"], row["low"], row["close"], row["volume"]

        if any(pd.isna(x) for x in (o, h, l, c)):
            issues.append(DataIssue(d, "nan", "NaN in OHLC"))
            continue
        # OHLC internal consistency.
        if h < max(o, c) or l > min(o, c) or h < l:
            issues.append(DataIssue(d, "ohlc_inconsistent", f"O={o} H={h} L={l} C={c}"))
        # Flat bar: all four equal (often a stale/placeholder bar).
        if o == h == l == c:
            issues.append(DataIssue(d, "flat_bar", f"O=H=L=C={c}"))
        # Volume sanity (allow 0 only on a flat bar / holiday-ish row).
        if pd.isna(v) or v < 0:
            issues.append(DataIssue(d, "bad_volume", f"volume={v}"))

    # Implausible close-to-close jumps.
    closes = pd.to_numeric(frame["close"], errors="coerce")
    pct = closes.pct_change().abs() * 100.0
    for ts, p in pct.items():
        if pd.notna(p) and p > max_daily_jump_pct:
            d = ts.date() if hasattr(ts, "date") else None
            issues.append(DataIssue(d, "jump", f"{p:.1f}% close-to-close move"))

    return issues


class BhavcopyValidator:
    """Cross-check & repair recent bars against the NSE bhavcopy."""

    def __init__(self, tolerance_pct: float = 1.0, cache_ttl_hours: float = 36.0) -> None:
        self.tolerance_pct = tolerance_pct
        self.cache = DiskCache("bhavcopy", ttl_hours=cache_ttl_hours)

    def fetch_bhavcopy(self, day: dt.date) -> pd.DataFrame:
        """Return the NSE bhavcopy for ``day`` indexed by symbol (or empty)."""
        cache_key = day.isoformat()
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        bc = pd.DataFrame()
        # Preferred: jugaad-data.
        try:
            from jugaad_data.nse import bhavcopy_save  # noqa: F401
            from jugaad_data.nse import bhavcopy_raw

            raw = bhavcopy_raw(day)  # returns CSV text
            from io import StringIO

            bc = pd.read_csv(StringIO(raw))
        except Exception as exc:
            log.debug("jugaad-data bhavcopy failed for %s: %s", day, exc)

        if bc.empty:
            try:
                from nsepython import get_bhavcopy

                bc = get_bhavcopy(day.strftime("%d-%m-%Y"))
            except Exception as exc:
                log.debug("nsepython bhavcopy failed for %s: %s", day, exc)

        if bc.empty:
            return bc

        bc.columns = [str(c).strip().upper() for c in bc.columns]
        sym_col = next((c for c in ("SYMBOL", "TCKRSYMB") if c in bc.columns), None)
        if sym_col:
            bc = bc[bc.get("SERIES", "EQ").astype(str).str.strip().isin(["EQ", ""])] \
                if "SERIES" in bc.columns else bc
            bc = bc.set_index(bc[sym_col].astype(str).str.strip())
            self.cache.set(cache_key, bc)
        return bc

    def validate(
        self,
        symbol: str,
        df: pd.DataFrame,
        *,
        last_n: int = 3,
        repair: bool = True,
    ) -> tuple[pd.DataFrame, ValidationResult]:
        """Compare the last ``last_n`` bars to the bhavcopy; repair if asked."""
        result = ValidationResult(symbol=symbol)
        result.issues.extend(sanity_checks(df, recent_only=max(last_n, 5)))

        if df is None or len(df) == 0:
            return df, result

        out = df.copy()
        for ts in out.index[-last_n:]:
            day = ts.date()
            bc = self.fetch_bhavcopy(day)
            if bc.empty or symbol.upper() not in bc.index:
                continue
            ref = bc.loc[symbol.upper()]

            def _ref(*names: str) -> Optional[float]:
                for n in names:
                    if n in ref.index and pd.notna(ref[n]):
                        return float(ref[n])
                return None

            checks = {
                "open": _ref("OPEN", "OPEN_PRICE"),
                "high": _ref("HIGH", "HIGH_PRICE"),
                "low": _ref("LOW", "LOW_PRICE"),
                "close": _ref("CLOSE", "CLOSE_PRICE", "LAST"),
            }
            for col, ref_val in checks.items():
                if ref_val is None or ref_val == 0:
                    continue
                cur = float(out.at[ts, col]) if pd.notna(out.at[ts, col]) else np.nan
                dev = abs(cur - ref_val) / ref_val * 100.0 if pd.notna(cur) else 100.0
                if dev > self.tolerance_pct:
                    result.issues.append(
                        DataIssue(day, "bhavcopy_deviation",
                                  f"{col} {cur} vs bhavcopy {ref_val} ({dev:.1f}%)")
                    )
                    if repair:
                        out.at[ts, col] = ref_val
                        if day not in result.repaired_dates:
                            result.repaired_dates.append(day)

        return normalize_ohlcv(out, drop_na=False), result
