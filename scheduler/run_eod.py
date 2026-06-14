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
from scoring.signal import generate_signal, top_reasons
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


def _make_watch_item(stock, mctx, composite, failed) -> dict:
    entry = mctx.settings.scoring.entry_threshold
    return {
        "symbol": stock.symbol, "sector": stock.sector, "as_of": mctx.as_of,
        "composite": round(composite.score, 2),
        "distance": round(max(0.0, entry - composite.score), 2),
        "gates_passed": len(failed) == 0,
        "blocking_gate": failed[0].name if failed else None,
        "status": "near_miss" if not failed else "blocked",
        "reasons": top_reasons(composite, mctx.settings.alerts.top_reasons),
    }


def process_entries(universe, mctx, providers, news_map, run_id, open_symbols):
    """Returns (buy_signals, watch_items, n_evaluated).

    A stock is scored if it passes all gates, or (for the watchlist) fails exactly
    one. It becomes a BUY if gates pass and composite >= entry; otherwise, if its
    composite >= watchlist.composite_min, it's captured as a near-miss watch item.
    """
    settings = mctx.settings
    wl = settings.watchlist
    signals: list[Signal] = []
    watch: list[dict] = []
    n_eval = 0
    for stock in universe:
        try:
            sctx = _build_stock_context(stock, mctx, providers, news_map)
            if sctx is None:
                continue
            n_eval += 1
            gate_report = run_gates(sctx, mctx)
            db.save_gate_records(run_id, stock.symbol, mctx.as_of, gate_report)
            failed = gate_report.failed

            score_it = len(failed) == 0 or (wl.enabled and wl.include_one_gate_away and len(failed) == 1)
            if not score_it or stock.symbol in open_symbols:
                continue

            subs = run_analyzers(sctx, mctx)
            composite = compute_composite(subs, settings)

            if len(failed) == 0:
                sig = generate_signal(sctx, mctx, composite, gate_report)
                if sig is not None:
                    sig.details["regime"] = mctx.regime
                    signals.append(sig)
                    continue
            # Not a BUY -> consider for the "almost there" watchlist.
            if wl.enabled and composite.score >= wl.composite_min:
                watch.append(_make_watch_item(stock, mctx, composite, failed))
        except Exception as exc:
            log.warning("entry processing failed for %s: %s", stock.symbol, exc)

    watch.sort(key=lambda w: w["composite"], reverse=True)
    return signals, watch[: wl.max_items], n_eval


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
        buy_signals, watch_items, n_eval = process_entries(
            universe, mctx, providers, news_map, run_id, open_symbols)

        # Persist BUY signals + open positions.
        for sig in buy_signals:
            sid = db.save_signal(sig, run_id=run_id)
            db.open_position(sig, entry_signal_id=sid)

        # Persist the "almost there" watchlist.
        if watch_items:
            db.save_watchlist(run_id, watch_items)
        freshness = {
            "price_source": settings.data.primary_price_source,
            "n_with_news": len(news_map),
            "n_headlines": len(headlines),
            "n_evaluated": n_eval,
            "n_watchlist": len(watch_items),
        }
        db.finish_run(run_id, status="success", market_regime=regime,
                      data_freshness=freshness, n_evaluated=n_eval)

        log.info("Generated %d BUY, %d EXIT, %d watchlist", len(buy_signals),
                 len(exit_signals), len(watch_items))

        if send:
            from alerting.telegram import get_notifier

            notifier = get_notifier(settings)
            sent = notifier.send_signals(exit_signals + buy_signals, settings, header_as_of=as_of)
            log.info("Telegram: %d message(s) sent", sent)

        return {"as_of": str(as_of), "buy": len(buy_signals), "exit": len(exit_signals),
                "watchlist": len(watch_items), "evaluated": n_eval, "regime": regime}
    except Exception as exc:
        log.exception("EOD run failed")
        db.finish_run(run_id, status="error", error=str(exc))
        raise


def market_cues(mctx, regime: dict) -> str:
    """One-line overnight market summary (US session, crude, USD/INR, VIX)."""
    m = mctx.macro or {}

    def chg(key: str) -> Optional[float]:
        s = m.get(key)
        if s is None:
            return None
        s = s.dropna()
        return (s.iloc[-1] / s.iloc[-2] - 1) * 100 if len(s) >= 2 else None

    bits = []
    for key, label in [("sp500", "S&P 500"), ("nasdaq", "Nasdaq"),
                       ("crude", "Crude"), ("usdinr", "USD/INR")]:
        c = chg(key)
        if c is not None:
            bits.append(f"{label} {c:+.1f}%")
    if regime.get("vix") is not None:
        bits.append(f"India VIX {regime['vix']:.1f}")
    return " · ".join(bits)


def run_premarket(date_str: Optional[str] = None, send: bool = True,
                  force: bool = False) -> dict:
    """Pre-open 'morning brief': refresh overnight news/sentiment/macro and
    re-check active positions. Does NOT create new technical signals (no fresh
    NSE prices before the 9:15 open)."""
    settings = get_settings()
    setup_logging(settings.logging.level)
    if date_str is None and not force and not is_trading_day(dt.date.today()):
        log.info("Not an NSE trading day; skipping pre-market brief.")
        return {"skipped": True, "reason": "non-trading day"}
    as_of = resolve_as_of(date_str)
    log.info("=== PRE-MARKET brief for %s ===", as_of)

    db.init_db()
    run_id = db.create_run(as_of)
    try:
        mctx, regime = build_market_context(settings, as_of)
        cues = market_cues(mctx, regime)
        positions = db.get_open_positions()
        umap = universe_map()

        news_map: dict = {}
        if positions:
            try:
                headlines = get_news_provider(settings).get_headlines(settings.news.max_age_days)
                stocks = [umap.get(p.symbol) or Stock(p.symbol, p.sector) for p in positions]
                news_map = match_headlines_to_symbols(headlines, stocks)
            except Exception as exc:
                log.warning("news fetch failed: %s", exc)

        providers = (get_price_provider(settings), get_fundamentals_provider(),
                     get_events_provider(), None)  # no bhavcopy repair pre-open
        cp = settings.scoring.conflict_penalty
        reviews: list[dict] = []
        for pos in positions:
            try:
                stock = umap.get(pos.symbol) or Stock(pos.symbol, pos.sector)
                sctx = _build_stock_context(stock, mctx, providers, news_map)
                if sctx is None:
                    continue
                subs = run_analyzers(sctx, mctx)
                comp = compute_composite(subs, settings)
                flags = event_risk_flags(sctx, mctx)
                sent = next((s for s in subs if s.key == "sentiment"), None)
                notes: list[str] = []
                if comp.score < settings.scoring.exit_threshold:
                    notes.append(f"score fell to {comp.score:.0f}")
                if sent is not None and sent.raw is not None and sent.raw <= cp.negative_sentiment_threshold:
                    notes.append(f"negative overnight news ({sent.raw:+.2f})")
                notes.extend(flags)
                reviews.append({
                    "symbol": pos.symbol, "sector": pos.sector, "composite": comp.score,
                    "status": "warn" if notes else "ok", "note": "; ".join(notes) or "stable",
                })
                db.update_position_fields(pos.id, last_composite=comp.score)
            except Exception as exc:
                log.warning("pre-market review failed for %s: %s", pos.symbol, exc)

        db.finish_run(run_id, status="success",
                      market_regime={**regime, "mode": "premarket"},
                      data_freshness={"mode": "premarket", "cues": cues},
                      n_evaluated=len(positions))
        log.info("Pre-market: reviewed %d position(s). Cues: %s", len(reviews), cues)

        watch = db.latest_watchlist()
        if send:
            from alerting.formatter import format_morning_brief
            from alerting.telegram import get_notifier

            get_notifier(settings).send_message(format_morning_brief(as_of, cues, reviews, watch))
        return {"as_of": str(as_of), "mode": "premarket", "reviewed": len(reviews),
                "watchlist": len(watch), "cues": cues}
    except Exception as exc:
        log.exception("Pre-market run failed")
        db.finish_run(run_id, status="error", error=str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Nifty 100 swing EOD pipeline")
    parser.add_argument("--mode", choices=["eod", "premarket"], default="eod",
                        help="eod = full scan + signals (default); premarket = morning news brief")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only first N stocks")
    parser.add_argument("--no-send", action="store_true", help="skip Telegram alerts")
    parser.add_argument("--date", type=str, default=None, help="override trading date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="run even on a non-trading day")
    parser.add_argument("--log-level", type=str, default=None)
    args = parser.parse_args()

    setup_logging(args.log_level or "INFO")
    if args.mode == "premarket":
        result = run_premarket(send=not args.no_send, date_str=args.date, force=args.force)
    else:
        result = run_eod(limit=args.limit, send=not args.no_send, date_str=args.date, force=args.force)
    print(result)


if __name__ == "__main__":
    main()
