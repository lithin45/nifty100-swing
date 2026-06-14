"""Backfill / warm the local data caches.

Fetches daily history for the whole universe plus market data (benchmark,
sector indices, VIX) using the configured provider. Everything is cached to
``data/cache/`` so the first live ``run_eod`` and backtests are fast.

    python scripts/seed_data.py            # full universe
    python scripts/seed_data.py --limit 20 # quick test
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import time

from common.logging_config import get_logger, setup_logging
from config.loader import get_settings, load_universe
from data_ingestion.prices import get_price_provider
from data_ingestion.sectors import get_sector_provider
from data_ingestion.vix import get_vix_provider

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical data caches")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--throttle", type=float, default=0.2, help="seconds between requests")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    universe = load_universe()
    if args.limit:
        universe = universe[: args.limit]

    price_p = get_price_provider(settings)
    sector_p = get_sector_provider()
    vix_p = get_vix_provider()

    ok = 0
    for i, stock in enumerate(universe, 1):
        df = price_p.get_history(stock.symbol)
        n = len(df) if df is not None else 0
        if n:
            ok += 1
        log.info("[%d/%d] %s: %d bars", i, len(universe), stock.symbol, n)
        time.sleep(args.throttle)

    log.info("Fetching market data (benchmark, sector indices, VIX)...")
    sector_p.get_index_close(settings.sectors.benchmark, 260)
    for ticker in settings.sectors.indices.values():
        sector_p.get_index_close(ticker, 120)
        time.sleep(args.throttle)
    vix_p.get_history(120)

    log.info("Done. %d/%d symbols have data cached.", ok, len(universe))


if __name__ == "__main__":
    main()
