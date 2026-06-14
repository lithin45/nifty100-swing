"""Event data: earnings dates, F&O ban list, circuit/corporate actions.

For unattended GitHub Actions runs the most reliable source is a user-maintained
``config/earnings_calendar.csv`` (columns: ``symbol,date``). The provider reads
that first, then *optionally* augments from NSE (nsepython). This keeps the event
gate working even when NSE endpoints are flaky.
"""
from __future__ import annotations

import csv
import datetime as dt
from typing import Optional

from common.logging_config import get_logger
from common.paths import CONFIG_DIR
from data_ingestion.base import DiskCache, EventsProvider

log = get_logger(__name__)


class NseEventsProvider(EventsProvider):
    def __init__(self, calendar_csv: Optional[str] = None, cache_ttl_hours: float = 12.0) -> None:
        self.calendar_path = calendar_csv or str(CONFIG_DIR / "earnings_calendar.csv")
        self.cache = DiskCache("events", ttl_hours=cache_ttl_hours)
        self._calendar = self._load_calendar()

    def _load_calendar(self) -> dict[str, dt.date]:
        cal: dict[str, dt.date] = {}
        try:
            with open(self.calendar_path, "r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
                for row in reader:
                    sym = str(row.get(cols.get("symbol", "symbol"), "")).strip().upper()
                    raw = str(row.get(cols.get("date", "date"), "")).strip()
                    if not sym or not raw:
                        continue
                    try:
                        cal[sym] = dt.date.fromisoformat(raw)
                    except ValueError:
                        log.debug("bad earnings date for %s: %s", sym, raw)
        except FileNotFoundError:
            log.debug("no earnings_calendar.csv at %s (event gate uses NSE only)",
                      self.calendar_path)
        return cal

    def get_earnings_date(self, symbol: str) -> Optional[dt.date]:
        symbol = symbol.upper()
        if symbol in self._calendar:
            return self._calendar[symbol]
        # Best-effort NSE augmentation (cached).
        nse_cal = self.cache.get("nse_results_calendar")
        if nse_cal is None:
            nse_cal = self._fetch_nse_results_calendar()
            self.cache.set("nse_results_calendar", nse_cal)
        return nse_cal.get(symbol)

    def _fetch_nse_results_calendar(self) -> dict[str, dt.date]:
        cal: dict[str, dt.date] = {}
        try:
            from nsepython import nsefetch

            data = nsefetch(
                "https://www.nseindia.com/api/event-calendar"
            )
            for ev in data or []:
                sym = str(ev.get("symbol", "")).strip().upper()
                raw = ev.get("date")
                if not sym or not raw:
                    continue
                try:
                    cal[sym] = dt.datetime.strptime(raw, "%d-%b-%Y").date()
                except (ValueError, TypeError):
                    continue
        except Exception as exc:
            log.debug("NSE event calendar failed: %s", exc)
        return cal

    def get_fno_ban_list(self) -> set[str]:
        cached = self.cache.get("fno_ban")
        if cached is not None:
            return cached
        banned: set[str] = set()
        try:
            from nsepython import nse_fno  # noqa: F401
            from nsepython import nsefetch

            data = nsefetch("https://www.nseindia.com/api/fnoBan")
            banned = {str(x).strip().upper() for x in (data or [])}
        except Exception as exc:
            log.debug("F&O ban fetch failed: %s", exc)
        self.cache.set("fno_ban", banned)
        return banned


def get_events_provider() -> EventsProvider:
    return NseEventsProvider()
