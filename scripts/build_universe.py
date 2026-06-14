"""Refresh ``config/nifty100.csv`` with the current Nifty 100 constituents + ISINs.

Pulls the live index composition from NSE (symbol, ISIN, industry). The shipped
CSV is a starting point; constituents rebalance ~semi-annually, so re-run this
periodically. ISINs are needed only for the optional Upstox/Dhan/bhavcopy paths
(the default yfinance path needs just the symbol).

    python scripts/build_universe.py            # writes config/nifty100.csv
    python scripts/build_universe.py --dry-run  # print, don't write
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import csv

from common.logging_config import get_logger, setup_logging
from common.paths import UNIVERSE_FILE

log = get_logger(__name__)

# Map NSE 'industry' strings to our compact sector labels (sector_factor uses these).
_INDUSTRY_TO_SECTOR = {
    "information technology": "IT", "it": "IT", "software": "IT",
    "financial services": "Financial Services", "banks": "Bank", "bank": "Bank",
    "insurance": "Insurance", "pharmaceuticals": "Pharma", "healthcare": "Healthcare",
    "automobile": "Auto", "automobiles": "Auto", "fast moving consumer goods": "FMCG",
    "fmcg": "FMCG", "metals & mining": "Metal", "metals": "Metal",
    "oil gas & consumable fuels": "Oil & Gas", "oil & gas": "Oil & Gas",
    "power": "Power", "realty": "Realty", "media entertainment & publication": "Media",
    "construction": "Infrastructure", "cement & cement products": "Cement",
    "chemicals": "Chemicals", "telecommunication": "Telecom",
    "consumer durables": "Consumer Durables", "consumer services": "Retail",
}


def _sector(industry: str | None) -> str:
    safe = (industry or "").strip()
    return _INDUSTRY_TO_SECTOR.get(safe.lower(), safe or "Unknown")


def fetch_constituents() -> list[dict]:
    from nsepython import nsefetch

    data = nsefetch("https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100")
    rows = []
    for d in data.get("data", []):
        sym = str(d.get("symbol", "")).strip().upper()
        if not sym or sym == "NIFTY 100":
            continue
        meta = d.get("meta", {}) or {}
        rows.append({
            "symbol": sym,
            # ISIN/name may live in `meta` or at the row top-level depending on endpoint.
            "isin": (meta.get("isin") or d.get("isin") or "").strip(),
            "upstox_key": "",
            "sector": _sector(d.get("industry") or meta.get("industry")),
            "name": (meta.get("companyName") or d.get("companyName") or "").strip(),
        })
    return sorted(rows, key=lambda r: r["symbol"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    setup_logging()

    try:
        rows = fetch_constituents()
    except Exception as exc:
        log.error("Failed to fetch from NSE (%s). Is nsepython installed / network up?", exc)
        sys.exit(1)

    if not rows:
        log.error("NSE returned no constituents; aborting.")
        sys.exit(1)

    log.info("Fetched %d constituents.", len(rows))
    if args.dry_run:
        for r in rows[:10]:
            print(r)
        print(f"... ({len(rows)} total)")
        return

    with open(UNIVERSE_FILE, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["symbol", "isin", "upstox_key", "sector", "name"])
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %s", UNIVERSE_FILE)


if __name__ == "__main__":
    main()
