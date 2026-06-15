"""Does the score actually predict future returns? (technical-only, honest version)

The live system buys when the composite score >= 65, on the assumption that a
higher score => a better stock. NOBODY HAS EVER CHECKED THAT. This script does.

What it tests
-------------
Only the *technical* sub-score (35% of the composite) can be reconstructed
point-in-time from free price/volume history. News / FII-DII / fundamentals have
no reliable free history, so this study honestly measures the biggest computable
chunk of the score. To make the 65 threshold comparable, each technical sub-score
is mapped to a "composite-equivalent" = the score the live system would produce
if every NON-technical factor sat at its neutral 0.5 fallback (which is exactly
what happens on degraded-data days). With default weights that is:

    composite_equiv = technical_weight * tech * 100 + (1 - technical_weight) * 50

For thousands of (stock, day) points it records:
  * the point-in-time technical score (no look-ahead: only bars up to that close),
  * the forward return of entering at the NEXT day's open and exiting ~1 month
    (max_holding_days) later — the live execution + horizon.

Then it asks three questions and prints a plain-English verdict:
  1. Does average forward return RISE across score deciles? (the "staircase")
  2. Rank correlation (information coefficient) between score and forward return.
  3. Do 65+ signals beat sub-65 signals — and beat just holding the market?

It writes NOTHING to the live DB and changes no live behaviour.

Run
---
    # Real data (needs yfinance; reuses the price cache):
    python scripts/score_study.py --limit 50 --start 2019-01-01

    # Faster sample (every 5th day), or a no-network smoke test:
    python scripts/score_study.py --limit 30 --stride 5
    python scripts/score_study.py --synthetic
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from analyzers.context import MarketContext, StockContext
from analyzers.technical import analyze_technical
from backtest.costs import compute_trade_costs
from common.logging_config import get_logger, setup_logging
from common.paths import BACKTEST_OUTPUT_DIR
from config.loader import get_settings, load_universe
from config.schema import Stock
from data_ingestion.prices import get_price_provider
from scripts.run_backtest import _synthetic_universe

log = get_logger(__name__)

MIN_HISTORY = 200  # need the 200-DMA to exist, matching the live trend checks


def _round_trip_cost_pct(settings) -> float:
    """Estimated round-trip cost as a % of capital (for a net-of-costs view)."""
    notional = 100_000.0
    cb = compute_trade_costs(notional, notional, settings.costs)
    return cb.total / notional * 100.0


def collect_observations(prices: dict[str, pd.DataFrame], settings,
                         horizon: int, stride: int, start: dt.date | None) -> pd.DataFrame:
    """Walk each stock day-by-day, scoring point-in-time and recording the
    forward return. Returns a tidy frame: symbol, date, tech, composite_equiv,
    fwd_return_pct."""
    tw = settings.scoring.normalized_weights().get("technical", 0.35)
    rows: list[dict] = []
    n_sym = len(prices)
    for s_i, (sym, df) in enumerate(prices.items(), 1):
        if df is None or len(df) < MIN_HISTORY + horizon + 2:
            continue
        df = df.sort_index()
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        n = len(df)
        if n < MIN_HISTORY + horizon + 2:
            continue
        opens = df["open"].to_numpy(float)
        closes = df["close"].to_numpy(float)
        idx = df.index
        stock = Stock(sym, "")
        # Evaluate from the first day the 200-DMA exists until we still have a
        # full forward window. Sample every `stride` days to keep it quick.
        last_eval = n - horizon - 2          # need open[i+1] and close[i+1+horizon]
        for i in range(MIN_HISTORY, last_eval + 1, stride):
            entry = opens[i + 1]
            exit_px = closes[i + 1 + horizon]
            if not (entry > 0 and exit_px > 0):
                continue
            sctx = StockContext(stock=stock, price=df.iloc[: i + 1])  # only up to today's close
            mctx = MarketContext(as_of=idx[i].date(), settings=settings, sector_rs={})
            tech = float(analyze_technical(sctx, mctx).score)
            comp_equiv = tw * tech * 100.0 + (1.0 - tw) * 50.0
            fwd = (exit_px / entry - 1.0) * 100.0
            rows.append({"symbol": sym, "date": idx[i].date(), "tech": tech,
                         "composite_equiv": comp_equiv, "fwd_return_pct": fwd})
        log.info("scored %s (%d/%d) — %d observations so far", sym, s_i, n_sym, len(rows))
    return pd.DataFrame(rows)


def analyze_observations(obs: pd.DataFrame, settings, horizon: int) -> str:
    """Bucket by score, compute the IC + the 65 / market comparisons, and return
    a plain-English report."""
    if obs.empty:
        return "No observations were collected — not enough price history."

    try:
        from scipy.stats import spearmanr
        ic, _p = spearmanr(obs["composite_equiv"], obs["fwd_return_pct"])
    except Exception:
        ic = float(obs["composite_equiv"].corr(obs["fwd_return_pct"], method="spearman"))

    # Market baseline = equal-weight cross-section: the average forward return of
    # ALL stocks on the same day. Excess = how much a score beat that day's market.
    market_fwd = obs.groupby("date")["fwd_return_pct"].transform("mean")
    obs = obs.assign(excess_pct=obs["fwd_return_pct"] - market_fwd)
    cost = _round_trip_cost_pct(settings)
    entry_thr = settings.scoring.entry_threshold

    # Decile staircase on the composite-equivalent score.
    try:
        obs = obs.assign(bucket=pd.qcut(obs["composite_equiv"], 10, labels=False, duplicates="drop"))
    except ValueError:
        obs = obs.assign(bucket=pd.cut(obs["composite_equiv"], 10, labels=False))
    grp = obs.groupby("bucket")
    table = grp.agg(
        score_lo=("composite_equiv", "min"), score_hi=("composite_equiv", "max"),
        n=("fwd_return_pct", "size"), avg_fwd=("fwd_return_pct", "mean"),
        win_rate=("fwd_return_pct", lambda x: (x > 0).mean() * 100.0),
        avg_excess=("excess_pct", "mean"),
    ).reset_index()

    lo_ret = table.iloc[0]["avg_fwd"]
    hi_ret = table.iloc[-1]["avg_fwd"]
    spread = hi_ret - lo_ret

    above = obs[obs["composite_equiv"] >= entry_thr]
    below = obs[obs["composite_equiv"] < entry_thr]
    n_above = len(above)
    above_net = above["fwd_return_pct"].mean() - cost if n_above else float("nan")
    above_excess = above["excess_pct"].mean() if n_above else float("nan")
    below_net = below["fwd_return_pct"].mean() - cost if len(below) else float("nan")

    # ---- assemble the report ----
    lines: list[str] = []
    lines.append("=" * 66)
    lines.append("DOES THE SCORE PREDICT FUTURE RETURNS? (technical-only study)")
    lines.append("=" * 66)
    lines.append(f"Observations: {len(obs):,}  |  stocks: {obs['symbol'].nunique()}  "
                 f"|  horizon: {horizon} trading days  |  round-trip cost: {cost:.2f}%")
    lines.append("")
    lines.append("Score bucket (composite-equivalent)  ->  avg next-month return")
    lines.append("-" * 66)
    lines.append(f"{'bucket':>6} {'score range':>16} {'n':>7} {'avg ret':>9} "
                 f"{'win%':>6} {'vs mkt':>8}")
    for _, r in table.iterrows():
        bar = "#" * max(0, int(round(r["avg_fwd"] * 2)))
        lines.append(f"{int(r['bucket'])+1:>6} {r['score_lo']:>6.1f}-{r['score_hi']:<6.1f}   "
                     f"{int(r['n']):>7,} {r['avg_fwd']:>8.2f}% {r['win_rate']:>5.0f}% "
                     f"{r['avg_excess']:>+7.2f}%  {bar}")
    lines.append("-" * 66)
    lines.append(f"Information coefficient (rank corr, score vs forward return): {ic:+.3f}")
    lines.append(f"Lowest-decile avg {lo_ret:+.2f}%  ->  highest-decile avg {hi_ret:+.2f}%  "
                 f"(spread {spread:+.2f}%)")
    lines.append("")
    lines.append(f"At the live 65 cutoff (composite_equiv >= {entry_thr:.0f}):")
    if n_above == 0:
        lines.append(f"  • ZERO of {len(obs):,} observations ever reached {entry_thr:.0f} on "
                     f"technicals alone — the threshold is effectively unreachable when the "
                     f"non-technical factors sit at neutral (the common degraded-data case).")
    else:
        lines.append(f"  • {n_above:,} signals ({n_above/len(obs)*100:.1f}%) cleared {entry_thr:.0f}.")
        lines.append(f"  • their avg next-month return (net of costs): {above_net:+.2f}%")
        lines.append(f"  • vs sub-65 signals (net): {below_net:+.2f}%")
        lines.append(f"  • vs just holding the market that month: {above_excess:+.2f}% excess")
    lines.append("")
    lines.append("VERDICT:")
    lines.append(_verdict(ic, spread, n_above, above_excess, len(obs)))
    lines.append("")
    lines.append("Caveats: technical factor only (35% of the live score); current Nifty-100 "
                 "list (survivorship); overlapping windows make this directional, not precise. "
                 "A flat staircase + near-zero IC means the score level carries little signal.")
    lines.append("=" * 66)
    return "\n".join(lines)


def _verdict(ic: float, spread: float, n_above: int, above_excess: float, n_obs: int) -> str:
    aic = abs(ic)
    if aic < 0.02 and abs(spread) < 0.5:
        strength = ("  ✗ The score shows NO meaningful predictive power: higher-scoring stocks "
                    "did not earn higher forward returns. The 65 cutoff is essentially cosmetic.")
    elif aic < 0.05:
        strength = ("  ~ The score shows WEAK predictive power. There may be a faint signal, but "
                    "it is small relative to noise and costs — not a reliable edge on its own.")
    else:
        strength = ("  ✓ The score shows MODEST predictive power: higher scores did tend to "
                    "precede higher returns. Worth developing — but still validate net of costs.")
    if n_above == 0:
        thr = ("\n  The 65 threshold is unreachable on technicals alone, so in practice the live "
               "system only fires when (rare) non-technical factors lift the score — exactly the "
               "part this study cannot verify. Treat live BUYs with extra caution.")
    elif np.isnan(above_excess) or above_excess <= 0:
        thr = ("\n  Signals at/above 65 did NOT beat simply holding the market — the threshold is "
               "not selecting winners over the index.")
    else:
        thr = ("\n  Signals at/above 65 did beat the market on average — the threshold is "
               "selecting something, but confirm it survives costs and out-of-sample.")
    return strength + thr


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether the score predicts forward returns")
    parser.add_argument("--limit", type=int, default=50, help="number of stocks to study")
    parser.add_argument("--start", type=str, default="2019-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=None,
                        help="forward holding days (default: risk.max_holding_days)")
    parser.add_argument("--stride", type=int, default=3,
                        help="evaluate every Nth trading day (higher = faster, coarser)")
    parser.add_argument("--universe", type=str, default=None,
                        help="universe CSV path (e.g. config/niftymidcap150.csv); default Nifty 100")
    parser.add_argument("--synthetic", action="store_true", help="no-network smoke test")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    horizon = args.horizon or settings.risk.max_holding_days
    start = dt.date.fromisoformat(args.start) if args.start else None
    BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        log.info("Generating synthetic universe (no network)...")
        prices = _synthetic_universe(n_symbols=min(args.limit, 12), n_days=1100)
        start = None
    else:
        provider = get_price_provider(settings)
        symbols = [s.symbol for s in load_universe(args.universe)[: args.limit]]
        bt_start = dt.date.fromisoformat(args.start) if args.start else settings.backtest.start
        bt_end = dt.date.fromisoformat(args.end) if args.end else None
        log.info("Fetching history for %d symbols from %s...", len(symbols), bt_start)
        prices = {s: provider.get_history(s, start=bt_start, end=bt_end) for s in symbols}
        prices = {k: v for k, v in prices.items() if v is not None and len(v) >= MIN_HISTORY}
        if not prices:
            log.error("No price data — install yfinance and/or run scripts/seed_data.py first.")
            sys.exit(1)

    log.info("Scoring point-in-time (stride=%d, horizon=%d)... this can take a minute.",
             args.stride, horizon)
    obs = collect_observations(prices, settings, horizon=horizon, stride=args.stride, start=start)

    report = analyze_observations(obs, settings, horizon=horizon)
    print("\n" + report + "\n")

    if not obs.empty:
        out = BACKTEST_OUTPUT_DIR / "score_study.csv"
        obs.to_csv(out, index=False)
        log.info("Wrote %d observations -> %s", len(obs), out)


if __name__ == "__main__":
    main()
