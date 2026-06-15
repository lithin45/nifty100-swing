"""Refresh a universe CSV with an NSE index's current constituents + ISINs.

Supports the Nifty 100 (default) and the Nifty Midcap 150. Pulls the live index
composition (symbol, ISIN, industry/sector, name) from the official NSE index
file, falling back to the NSE quote API. The shipped CSVs are starting points;
constituents rebalance ~semi-annually, so re-run periodically. ISINs are needed
only for the optional Upstox/Dhan/bhavcopy paths (the default yfinance path needs
just the symbol).

    python scripts/build_universe.py                          # -> config/nifty100.csv
    python scripts/build_universe.py --index niftymidcap150   # -> config/niftymidcap150.csv
    python scripts/build_universe.py --index niftymidcap150 --dry-run
    python scripts/build_universe.py --index niftymidcap150 --out config/midcaps.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import csv
import io

from common.logging_config import get_logger, setup_logging
from common.paths import CONFIG_DIR

log = get_logger(__name__)

# Supported indices: key -> (niftyindices CSV slug, NSE API index name, default file, title).
INDEX_SPECS: dict[str, dict] = {
    "nifty100": {
        "slug": "nifty100", "nse": "NIFTY 100",
        "out": "nifty100.csv", "title": "Nifty 100",
    },
    "niftymidcap150": {
        "slug": "niftymidcap150", "nse": "NIFTY MIDCAP 150",
        "out": "niftymidcap150.csv", "title": "Nifty Midcap 150",
    },
}

# Map NSE 'industry' strings to our compact sector labels (sector_factor uses these).
_INDUSTRY_TO_SECTOR = {
    "information technology": "IT", "it": "IT", "software": "IT",
    "financial services": "Financial Services", "banks": "Bank", "bank": "Bank",
    "insurance": "Insurance", "pharmaceuticals": "Pharma", "healthcare": "Healthcare",
    "automobile": "Auto", "automobiles": "Auto",
    "automobile and auto components": "Auto", "fast moving consumer goods": "FMCG",
    "fmcg": "FMCG", "metals & mining": "Metal", "metals": "Metal",
    "oil gas & consumable fuels": "Oil & Gas", "oil & gas": "Oil & Gas",
    "power": "Power", "realty": "Realty", "media entertainment & publication": "Media",
    "media": "Media", "construction": "Infrastructure",
    "cement & cement products": "Cement", "construction materials": "Cement",
    "chemicals": "Chemicals", "telecommunication": "Telecom",
    "consumer durables": "Consumer Durables", "consumer services": "Retail",
    "capital goods": "Capital Goods", "services": "Services", "textiles": "Textiles",
    "diversified": "Diversified",
}

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"),
    "Referer": "https://niftyindices.com/",
    "Accept": "text/csv,application/csv,*/*",
}


def _sector(industry: str | None) -> str:
    safe = (industry or "").strip()
    return _INDUSTRY_TO_SECTOR.get(safe.lower(), safe or "Unknown")


def fetch_from_niftyindices(slug: str) -> list[dict]:
    """Primary source: the official niftyindices.com constituent CSV
    (columns: Company Name, Industry, Symbol, Series, ISIN Code)."""
    import requests

    url = f"https://niftyindices.com/IndexConstituent/ind_{slug}list.csv"
    resp = requests.get(url, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}

    def col(*names: str) -> str | None:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    sym_c, name_c = col("symbol"), col("company name", "company")
    ind_c, isin_c, ser_c = col("industry"), col("isin code", "isin"), col("series")
    rows: list[dict] = []
    for d in reader:
        if ser_c:
            series = str(d.get(ser_c, "")).strip().upper()
            if series and series != "EQ":   # keep cash-segment rows only
                continue
        sym = str(d.get(sym_c, "")).strip().upper() if sym_c else ""
        if not sym:
            continue
        rows.append({
            "symbol": sym,
            "isin": (str(d.get(isin_c, "")).strip() if isin_c else ""),
            "upstox_key": "",
            "sector": _sector(d.get(ind_c) if ind_c else ""),
            "name": (str(d.get(name_c, "")).strip().rstrip(".") if name_c else ""),
        })
    return sorted(rows, key=lambda r: r["symbol"])


def fetch_from_nse_api(index_name: str) -> list[dict]:
    """Fallback source: the NSE equity-stockIndices quote API (via nsepython)."""
    from urllib.parse import quote

    from nsepython import nsefetch

    data = nsefetch(f"https://www.nseindia.com/api/equity-stockIndices?index={quote(index_name)}")
    rows: list[dict] = []
    for d in data.get("data", []):
        sym = str(d.get("symbol", "")).strip().upper()
        if not sym or sym == index_name.upper():
            continue
        meta = d.get("meta", {}) or {}
        rows.append({
            "symbol": sym,
            "isin": (meta.get("isin") or d.get("isin") or "").strip(),
            "upstox_key": "",
            "sector": _sector(d.get("industry") or meta.get("industry")),
            "name": (meta.get("companyName") or d.get("companyName") or "").strip(),
        })
    return sorted(rows, key=lambda r: r["symbol"])


def fetch_constituents(spec: dict) -> list[dict]:
    """Try the official CSV first, then the NSE API."""
    try:
        rows = fetch_from_niftyindices(spec["slug"])
        if rows:
            log.info("Fetched %d constituents from niftyindices.com", len(rows))
            return rows
        log.warning("niftyindices returned no rows; trying the NSE API...")
    except Exception as exc:
        log.warning("niftyindices fetch failed (%s); trying the NSE API...", exc)
    rows = fetch_from_nse_api(spec["nse"])
    log.info("Fetched %d constituents from the NSE API", len(rows))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh an NSE index universe CSV")
    parser.add_argument("--index", choices=sorted(INDEX_SPECS), default="nifty100")
    parser.add_argument("--out", type=str, default=None, help="output CSV path (overrides default)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    setup_logging()

    spec = INDEX_SPECS[args.index]
    out_path = Path(args.out) if args.out else (CONFIG_DIR / spec["out"])

    try:
        rows = fetch_constituents(spec)
    except Exception as exc:
        log.error("Failed to fetch %s (%s). Is requests/nsepython installed / network up?",
                  spec["title"], exc)
        sys.exit(1)

    if not rows:
        log.error("No constituents returned for %s; aborting.", spec["title"])
        sys.exit(1)

    log.info("%s: %d constituents.", spec["title"], len(rows))
    if args.dry_run:
        for r in rows[:10]:
            print(r)
        print(f"... ({len(rows)} total)")
        return

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"symbol,isin,upstox_key,sector,name\n")
        fh.write(f"# {spec['title']} constituents. Cols: symbol, ISIN, Upstox key, sector, name.\n")
        fh.write(f"# yfinance path needs only `symbol` (used as <symbol>.NS). Blank upstox_key is\n")
        fh.write(f"# auto-derived as \"NSE_EQ|<ISIN>\". REFRESH with: python scripts/build_universe.py\n")
        fh.write(f"#   --index {args.index}   (constituents rebalance ~semi-annually).\n")
        writer = csv.DictWriter(fh, fieldnames=["symbol", "isin", "upstox_key", "sector", "name"])
        writer.writerows(rows)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
