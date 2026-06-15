"""Run a portfolio backtest and write metrics + a QuantStats tearsheet.

    # On real data (needs yfinance + cached/seeded prices):
    python scripts/run_backtest.py --limit 40 --start 2019-01-01 --tearsheet

    # Self-contained demo on synthetic data (no network needed):
    python scripts/run_backtest.py --synthetic --tearsheet

Outputs go to ``backtest/output/``: tearsheet.html, trades.csv, equity.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.metrics import generate_tearsheet
from backtest.walkforward import walk_forward
from common.logging_config import get_logger, setup_logging
from common.paths import BACKTEST_OUTPUT_DIR
from config.loader import get_settings, load_universe
from data_ingestion.prices import get_price_provider

log = get_logger(__name__)


def _synthetic_universe(n_symbols: int = 12, n_days: int = 1100) -> dict[str, pd.DataFrame]:
    """Deterministic trending+noisy stocks so the demo generates trades."""
    out: dict[str, pd.DataFrame] = {}
    idx = pd.date_range("2021-01-04", periods=n_days, freq="B")
    for i in range(n_symbols):
        rng = np.random.default_rng(i)
        drift = 0.0004 + 0.00015 * i
        steps = rng.normal(drift, 0.014, n_days).cumsum()
        close = 100 * np.exp(steps) * (1 + rng.normal(0, 0.004, n_days))
        high = close * (1 + np.abs(rng.normal(0, 0.009, n_days)))
        low = close * (1 - np.abs(rng.normal(0, 0.009, n_days)))
        openp = np.r_[close[0], close[:-1]]
        vol = rng.uniform(2e6, 4e6, n_days)
        up = close > np.r_[close[0], close[:-1]]
        vol[up] *= 1.9  # volume spikes on up days -> breakout confirmation
        out[f"SYNTH{i:02d}"] = pd.DataFrame(
            {"open": openp, "high": high, "low": low, "close": close, "volume": vol}, index=idx
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio backtest")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true", help="use synthetic data (no network)")
    parser.add_argument("--universe", type=str, default=None,
                        help="universe CSV path (e.g. config/niftymidcap150.csv); default Nifty 100")
    parser.add_argument("--walkforward", action="store_true", help="run walk-forward validation")
    parser.add_argument("--tearsheet", action="store_true", help="write QuantStats HTML")
    parser.add_argument("--no-regime", action="store_true",
                        help="disable the market-regime entry filter (broad market > 200-DMA)")
    parser.add_argument("--entry", choices=["breakout", "pullback"], default="breakout",
                        help="entry rule: breakout (default) or pullback (buy-the-dip)")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        log.info("Generating synthetic universe...")
        prices = _synthetic_universe()
    else:
        provider = get_price_provider(settings)
        symbols = [s.symbol for s in load_universe(args.universe)[: args.limit]]
        # Fetch the FULL backtest range, not the live ~800-bar cap. Without an
        # explicit start, get_history() only returns data.history_days of bars,
        # which silently truncates a multi-year --start.
        import datetime as _dt
        bt_start = _dt.date.fromisoformat(args.start or settings.backtest.start)
        bt_end = _dt.date.fromisoformat(args.end) if args.end else None
        log.info("Fetching history for %d symbols from %s...", len(symbols), bt_start)
        prices = {s: provider.get_history(s, start=bt_start, end=bt_end) for s in symbols}
        prices = {k: v for k, v in prices.items() if v is not None and len(v) >= 220}
        if not prices:
            log.error("No price data — install yfinance and/or run scripts/seed_data.py first.")
            sys.exit(1)
        spans = [v.index.min() for v in prices.values()]
        log.info("Loaded %d symbols; earliest data from %s", len(prices), min(spans).date())

    if args.walkforward:
        log.info("Running walk-forward validation...")
        wf = walk_forward(prices, settings)
        print("\n" + "=" * 60 + "\n" + wf.summary + "\n" + "=" * 60)
        print("Combined OOS metrics:", wf.combined_metrics)
        equity = wf.combined_equity
        trades = pd.concat([w.result.trades for w in wf.windows if len(w.result.trades)],
                           ignore_index=True) if wf.windows else pd.DataFrame()
    else:
        log.info("Running portfolio backtest (entry: %s, regime filter: %s)...",
                 args.entry, "off" if args.no_regime else "on")
        res = run_backtest(prices, settings, start=args.start, end=args.end,
                           regime_filter=not args.no_regime, entry_mode=args.entry)
        print("\n" + "=" * 60)
        print("METRICS:", res.metrics)
        print("-" * 60)
        print(res.summary)
        print("=" * 60)
        equity, trades = res.equity, res.trades

    if len(trades):
        trades.to_csv(BACKTEST_OUTPUT_DIR / "trades.csv", index=False)
        log.info("Wrote %s", BACKTEST_OUTPUT_DIR / "trades.csv")
    if len(equity):
        equity.to_csv(BACKTEST_OUTPUT_DIR / "equity.csv")
        if args.tearsheet:
            path = str(BACKTEST_OUTPUT_DIR / "tearsheet.html")
            if generate_tearsheet(equity, path, title="Nifty100 Swing"):
                log.info("Wrote QuantStats tearsheet -> %s", path)
            else:
                log.warning("quantstats not installed; skipped tearsheet "
                            "(pip install quantstats). Metrics above are still valid.")


if __name__ == "__main__":
    main()
