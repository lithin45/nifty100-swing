"""Load and cache ``settings.yaml`` and the Nifty-100 universe CSV.

The universe is parsed with the stdlib ``csv`` module (no pandas dependency) so
the config layer stays lightweight and importable in minimal environments.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import yaml

from common.calendar_nse import register_holidays
from common.paths import SETTINGS_FILE, UNIVERSE_FILE
from config.schema import Settings, Stock


def load_settings(path: str | Path | None = None) -> Settings:
    """Read and validate ``settings.yaml`` (fresh read, not cached)."""
    p = Path(path) if path else SETTINGS_FILE
    if not p.exists():
        # Missing file -> run on documented defaults.
        return Settings()
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    settings = Settings(**raw)
    # Allow extra trading holidays to be supplied via YAML.
    extra = (raw.get("calendar") or {}).get("extra_holidays") or []
    if extra:
        register_holidays(extra)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton (used by long-running processes/dashboard)."""
    return load_settings()


def reload_settings() -> Settings:
    """Clear the cache and reload (handy for the dashboard / tests)."""
    get_settings.cache_clear()
    return get_settings()


def load_universe(path: str | Path | None = None) -> list[Stock]:
    """Parse ``nifty100.csv`` into a list of :class:`Stock`.

    Expected header (case-insensitive): ``symbol, isin, upstox_key, sector``.
    An optional ``name`` column is used for display if present. Blank
    ISIN/upstox_key are tolerated — the yfinance price path only needs symbols.
    """
    p = Path(path) if path else UNIVERSE_FILE
    if not p.exists():
        raise FileNotFoundError(f"Universe CSV not found: {p}")

    stocks: list[Stock] = []
    with open(p, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return stocks
        cols = {c.lower().strip(): c for c in reader.fieldnames}

        def get(row: dict, *keys: str) -> str:
            for k in keys:
                if k in cols and row.get(cols[k]) is not None:
                    return str(row[cols[k]]).strip()
            return ""

        for row in reader:
            symbol = get(row, "symbol", "ticker").upper()
            if not symbol or symbol.startswith("#"):
                continue
            stocks.append(
                Stock(
                    symbol=symbol,
                    sector=get(row, "sector") or "Unknown",
                    isin=get(row, "isin"),
                    upstox_key=get(row, "upstox_key", "upstox_instrument_key", "upstox"),
                    name=get(row, "name", "company"),
                )
            )
    return stocks


def universe_map(path: str | Path | None = None) -> dict[str, Stock]:
    """Symbol -> :class:`Stock` lookup."""
    return {s.symbol: s for s in load_universe(path)}


def get_db_path() -> str:
    """Resolve the SQLite DB path (settings override or default)."""
    from common.paths import DEFAULT_DB_PATH, ensure_data_dir

    settings = get_settings()
    ensure_data_dir()
    return settings.storage.db_path or str(DEFAULT_DB_PATH)
