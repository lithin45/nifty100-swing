"""End-of-day pipeline — the entry point for the GitHub Actions cron job.

Flow:
  1. build market context once (VIX, macro, FII/DII, benchmark, sector RS, F&O ban)
  2. for each Nifty-100 stock: fetch+validate prices, fundamentals, news, earnings
     -> run hard gates -> (if passed) composite -> BUY signal
  3. for each OPEN position: recompute composite -> evaluate exit (or ratchet trail)
  4. persist runs / signals / sub-scores / gate records / positions to SQLite
  5. send Telegram alerts (exits first, then BUYs by conviction)

Run:
  python -m scheduler.run_eod              # full run
  python -m scheduler.run_eod --limit 15   # quick test on first 15 stocks
  python -m scheduler.run_eod --no-send    # don't push Telegram
  python -m scheduler.run_eod --date 2026-06-12
"""
from __future__ import annotations

# --- flat-layout import bootstrap ---
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ------------------------------------

import argparse
import datetime as dt
from typing import Optional

from analyzers.context import MarketContext, StockContext
from analyzers.sector_factor import compute_sector_rs
from common.calendar_nse import is_trading_day, previous_trading_day
from common.logging_config import get_logger, setup_logging
from common.types import Signal, SignalAction
from config.loader import get_settings, load_universe, universe_map
from config.schema import Stock
from data_ingestion.events import get_events_provider
from data_ingestion.fii_dii import get_fii_dii_provider
from data_ingestion.fundamentals import get_fundamentals_provider
from data_ingestion.macro import get_macro_provider
from data_ingestion.news import get_news_provider, match_headlines_to_symbols
from data_ingestion.prices import get_price_provider
from data_ingestion.sectors import get_sector_provider
from data_ingestion.validators import BhavcopyValidator
from data_ingestion.vix import get_vix_provider
from scoring.composite import compute_composite, run_analyzers
from scoring.exits import evaluate_exit
from scoring.gates import run_gates
from scoring.signal import generate_signal
from storage import db

log = get_logger(__name__)


def resolve_as_of(date_str: Optional[str]) -> dt.date:
    if date_str:
        return dt.date.fromisoformat(date_str)
    today = dt.date.today()
    return today if is_trading_day(today) else previous_trading_day(today)


def build_market_context(settings, as_of: dt.date) -> tuple[MarketContext, dict]:
    """Fetch market-wide data once and assemble the MarketContext + regime dict."""
    vix_p = get_vix_provider()
    macro_p = get_macro_provider(settings)
    fii_p = get_fii_dii_provider()
    sector_p = get_sector_provider()
    events_p = get_events_provider()

    vix = vix_p.get_history(60)
    macro = macro_p.get_snapshot(60)
    fii_dii = fii_p.get_recent(settings.fii_dii.lookback_days)
    benchmark = sector_p.get_index_close(settings.sectors.benchmark, 260)
    sector_rs = compute_sector_rs(sector_p, settings)
    fno_ban = events_p.get_fno_ban_list()

    # Market regime snapshot.
    regime: dict = {}
    if vix is not None and len(vix.dropna()):
        regime["vix"] = round(float(vix.dropna().iloc[-1]), 2)
    if benchmark is not None and len(benchmark.dropna()) >= 200:
        from analyzers import indicators as I

        sma = I.sma(benchmark, 200).dropna()
        if len(sma):
            regime["nifty_above_200dma"] = bool(benchmark.dropna().iloc[-1] > sma.iloc[-1])
    if fii_dii is not None and len(fii_dii) and "fii_net" in fii_dii.columns:
        regime["fii_net_cr"] = round(float(fii_dii["fii_net"].sum()), 1)

    mctx = MarketContext(
        as_of=as_of, settings=settings, vix=vix, macro=macro, fii_dii=fii_dii,
        benchmark=benchmark, sector_rs=sector_rs, regime=regime, fno_ban=fno_ban,
    )
    return mctx, regime


def _build_stock_context(stock: Stock, mctx, providers, news_map) -> Optional[StockContext]:
    price_p, fund_p, events_p, validator = providers
    df = price_p.get_history(stock.symbol)
    if df is None or len(df) == 0:
        return None
    if validator is not None:
        try:
            df, _vr = validator.validate(stock.symbol, df, last_n=3, repair=True)
        except Exception as exc:
            log.debug("bhavcopy validation skipped for %s: %s", stock.symbol, exc)
    try:
        fundamentals = fund_p.get_fundamentals(stock.symbol)
    except Exception:
        fundamentals = {}
    try:
        earnings = events_p.get_earnings_date(stock.symbol)
    except Exception:
        earnings = None
    return StockContext(
        stock=stock, price=df, fundamentals=fundamentals,
        headlines=news_map.get(stock.symbol, []), earnings_date=earnings,
    )


def process_entries(universe, mctx, providers, news_map, run_id, open_symbols) -> tuple[list[Signal], int]:
    signals: list[Signal] = []
    n_eval = 0
    for stock in universe:
        try:
            sctx = _build_stock_context(stock, mctx, providers, news_map)
            if sctx is None:
                continue
            n_eval += 1
            gate_report = run_gates(sctx, mctx)
            db.save_gate_records(run_id, stock.symbol, mctx.as_of, gate_report)

            if not gate_report.passed:
                continue
            if stock.symbol in open_symbols:
                continue  # already holding — exits handle it

            subs = run_analyzers(sctx, mctx)
            composite = compute_composite(subs, mctx.settings)
            sig = generate_signal(sctx, mctx, composite, gate_report)
            if sig is not None:
                sig.details["regime"] = mctx.regime
                signals.append(sig)
        except Exception as exc:
            log.warning("entry processing failed for %s: %s", stock.symbol, exc)
    return signals, n_eval


def process_exits(mctx, providers, run_id) -> list[Signal]:
    price_p = providers[0]
    umap = universe_map()
    exit_signals: list[Signal] = []
    for pos in db.get_open_positions():
        try:
            stock = umap.get(pos.symbol) or Stock(pos.symbol, pos.sector)
            df = price_p.get_history(pos.symbol)
            if df is None or len(df) == 0:
                continue
            sctx = StockContext(stock=stock, price=df)
            subs = run_analyzers(sctx, mctx)
            composite = compute_composite(subs, mctx.settings)
            decision = evaluate_exit(pos, df, composite.score, mctx)

            # Always persist the ratcheted trailing-stop bookkeeping.
            db.update_position_fields(
                pos.id, highest_close=decision.new_highest_close,
                current_stop=decision.new_current_stop,
                last_composite=composite.score, last_price=decision.price,
            )

            if decision.should_exit:
                sig = Signal(
                    symbol=pos.symbol, sector=pos.sector, action=SignalAction.EXIT,
                    as_of=mctx.as_of, composite=composite.score, exit_reason=decision.reason,
                    reasons=[decision.detail],
                    details={
                        "entry_price": pos.entry_price, "current_price": decision.price,
                        "pnl_pct": decision.pnl_pct, "holding_days": decision.holding_days,
                        "detail": decision.detail, "regime": mctx.regime,
                    },
                )
                sid = db.save_signal(sig, run_id=run_id)
                db.close_position(pos.id, mctx.as_of, decision.price, decision.reason.value)
                exit_signals.append(sig)
        except Exception as exc:
            log.warning("exit processing failed for %s: %s", pos.symbol, exc)
    return exit_signals


def run_eod(limit: Optional[int] = None, send: bool = True,
            date_str: Optional[str] = None, force: bool = False) -> dict:
    settings = get_settings()
    setup_logging(settings.logging.level)
    # On weekday NSE holidays there is no fresh bar — skip to avoid duplicate runs.
    if date_str is None and not force and not is_trading_day(dt.date.today()):
        log.info("Today is not an NSE trading day; skipping. Use --force to override.")
        return {"skipped": True, "reason": "non-trading day"}
    as_of = resolve_as_of(date_str)
    log.info("=== EOD run for %s ===", as_of)

    db.init_db()
    universe = load_universe()
    if limit:
        universe = universe[:limit]

    run_id = db.create_run(as_of)
    try:
        mctx, regime = build_market_context(settings, as_of)
        log.info("Market regime: %s", regime)

        # News once, mapped to symbols.
        try:
            headlines = get_news_provider(settings).get_headlines(settings.news.max_age_days)
            news_map = match_headlines_to_symbols(headlines, universe)
        except Exception as exc:
            log.warning("news fetch failed: %s", exc)
            headlines, news_map = [], {}

        validator = BhavcopyValidator() if settings.data.validate_with_bhavcopy else None
        providers = (get_price_provider(settings), get_fundamentals_provider(),
                     get_events_provider(), validator)

        open_symbols = db.get_open_symbols()
        exit_signals = process_exits(mctx, providers, run_id)

        open_symbols = db.get_open_symbols()  # refresh after exits
        buy_signals, n_eval = process_entries(universe, mctx, providers, news_map, run_id, open_symbols)

        # Persist BUY signals + open positions.
        for sig in buy_signals:
            sid = db.save_signal(sig, run_id=run_id)
            db.open_position(sig, entry_signal_id=sid)
        freshness = {
            "price_source": settings.data.primary_price_source,
            "n_with_news": len(news_map),
            "n_headlines": len(headlines),
            "n_evaluated": n_eval,
        }
        db.finish_run(run_id, status="success", market_regime=regime,
                      data_freshness=freshness, n_evaluated=n_eval)

        log.info("Generated %d BUY, %d EXIT", len(buy_signals), len(exit_signals))

        if send:
            from alerting.telegram import get_notifier

            notifier = get_notifier(settings)
            sent = notifier.send_signals(exit_signals + buy_signals, settings, header_as_of=as_of)
            log.info("Telegram: %d message(s) sent", sent)

        return {"as_of": str(as_of), "buy": len(buy_signals), "exit": len(exit_signals),
                "evaluated": n_eval, "regime": regime}
    except Exception as exc:
        log.exception("EOD run failed")
        db.finish_run(run_id, status="error", error=str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Nifty 100 swing EOD pipeline")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only first N stocks")
    parser.add_argument("--no-send", action="store_true", help="skip Telegram alerts")
    parser.add_argument("--date", type=str, default=None, help="override trading date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="run even on a non-trading day")
    parser.add_argument("--log-level", type=str, default=None)
    args = parser.parse_args()

    setup_logging(args.log_level or "INFO")
    result = run_eod(limit=args.limit, send=not args.no_send, date_str=args.date, force=args.force)
    print(result)


if __name__ == "__main__":
    main()
