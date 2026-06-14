"""Streamlit dashboard — password-protected, beginner-friendly.

Reads only from SQLite (the scheduled EOD job writes). Deploy on Streamlit
Community Cloud with main file ``dashboard/app.py`` and secrets set in the app's
Advanced settings. Run locally:  ``streamlit run dashboard/app.py``
"""
from __future__ import annotations

# --- flat-layout import bootstrap (must precede project imports) ---
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# -------------------------------------------------------------------

import datetime as dt

import streamlit as st

from common.logging_config import setup_logging
from config.loader import get_settings
from dashboard.auth import require_password
from dashboard.components import (
    candlestick_chart,
    glossary_expander,
    positions_table,
    regime_banner,
    signal_card,
)
from storage import db
from storage.models import GateRecord

setup_logging()
settings = get_settings()

st.set_page_config(page_title=settings.dashboard.title, page_icon="📈", layout="wide")

if not require_password():
    st.stop()

db.init_db()  # ensure tables exist (no-op if present)

st.title(f"📈 {settings.dashboard.title}")
st.caption("Signal-only research tool. It never places orders — you decide and execute manually. "
           "Not investment advice.")

run = db.latest_run()
regime_banner(run)
if settings.dashboard.show_terms_glossary:
    glossary_expander()

tab_signals, tab_positions, tab_charts, tab_status = st.tabs(
    ["🟢 Today's Signals", "💼 Open Positions", "📊 Charts", "⚙️ System Status"]
)

# --------------------------------------------------------------------------- #
# Today's signals                                                             #
# --------------------------------------------------------------------------- #
with tab_signals:
    as_of = (run.trading_date if run else None) or dt.date.today()
    signals = db.signals_for_date(as_of)
    buys = [s for s in signals if s.action == "BUY"]
    exits = [s for s in signals if s.action == "EXIT"]

    st.subheader(f"New BUY ideas for {as_of} ({len(buys)})")
    if not buys:
        st.info("No BUY signals passed all gates today. That's normal — quality over quantity.")
    for sig in buys:
        signal_card(sig)

    if exits:
        st.subheader(f"EXIT alerts ({len(exits)})")
        for s in exits:
            pnl = (s.details or {}).get("pnl_pct")
            extra = f" · P&L {pnl:+.1f}%" if pnl is not None else ""
            st.markdown(f"🔴 **{s.symbol}** ({s.sector}) — {s.exit_reason}{extra}")

# --------------------------------------------------------------------------- #
# Open positions                                                              #
# --------------------------------------------------------------------------- #
with tab_positions:
    st.subheader("Open positions / watchlist")
    positions = db.get_open_positions()
    if not positions:
        st.info("No open positions tracked yet.")
    else:
        df = positions_table(positions)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P&L %": st.column_config.NumberColumn(format="%.1f%%"),
                "→ Stop %": st.column_config.NumberColumn(
                    "→ Stop %", help="How far price is above the stop (smaller = closer to exit)"),
                "→ Target %": st.column_config.NumberColumn(
                    "→ Target %", help="How far price is below the target"),
            },
        )
        st.caption("Tip: a small '→ Stop %' means the stock is close to its safety exit.")

    closed = db.closed_positions(limit=50)
    if closed:
        with st.expander(f"Recently closed ({len(closed)})"):
            import pandas as pd

            cdf = pd.DataFrame([{
                "Stock": c.symbol, "Entry": round(c.entry_price, 2),
                "Exit": round(c.exit_price or 0, 2), "P&L %": round(c.pnl_pct or 0, 1),
                "Reason": c.exit_reason, "Held (d)": c.holding_days,
                "Exited": c.exit_date,
            } for c in closed])
            st.dataframe(cdf, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# Charts                                                                      #
# --------------------------------------------------------------------------- #
with tab_charts:
    st.subheader("Annotated price chart")
    universe_syms = sorted({s.symbol for s in db.signals_for_date(
        (run.trading_date if run else dt.date.today()))} |
        {p.symbol for p in db.get_open_positions()})
    if not universe_syms:
        from config.loader import load_universe

        universe_syms = [s.symbol for s in load_universe()[:25]]
    symbol = st.selectbox("Choose a stock", universe_syms)
    if symbol:
        try:
            from data_ingestion.prices import get_price_provider

            price_df = get_price_provider(settings).get_history(symbol)
            if price_df is None or len(price_df) == 0:
                st.warning("No price data available for this stock right now.")
            else:
                # overlay the latest signal's levels if present
                sigs = [s for s in db.signals_for_date(
                    (run.trading_date if run else dt.date.today())) if s.symbol == symbol]
                entry = sigs[0].entry_price if sigs else None
                stop = sigs[0].stop_loss if sigs else None
                target = sigs[0].target if sigs else None
                fig = candlestick_chart(price_df, title=symbol, entry=entry, stop=stop, target=target)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not load chart: {exc}")

# --------------------------------------------------------------------------- #
# System status                                                               #
# --------------------------------------------------------------------------- #
with tab_status:
    st.subheader("System status & data freshness")
    if run is None:
        st.warning("No runs recorded yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Last run status", run.status)
        c1.metric("Started (UTC)", run.started_at.strftime("%Y-%m-%d %H:%M"))
        c2.metric("Stocks evaluated", run.n_evaluated)
        c2.metric("BUY / EXIT", f"{run.n_buy} / {run.n_exit}")
        if run.finished_at:
            c3.metric("Finished (UTC)", run.finished_at.strftime("%Y-%m-%d %H:%M"))
        if run.error:
            st.error(f"Last run error: {run.error}")
        if run.data_freshness:
            st.json(run.data_freshness)

        # Gate-block explainer
        with st.expander("Why were some stocks blocked? (gate outcomes)"):
            with db.session_scope() as s:
                from sqlalchemy import select

                recs = list(s.scalars(
                    select(GateRecord).where(GateRecord.run_id == run.id, GateRecord.passed == False)  # noqa: E712
                    .limit(200)
                ))
                if not recs:
                    st.write("No gate blocks recorded for the last run.")
                else:
                    import pandas as pd

                    gdf = pd.DataFrame([{"Stock": r.symbol, "Gate": r.name, "Reason": r.reason}
                                        for r in recs])
                    st.dataframe(gdf, use_container_width=True, hide_index=True)

st.divider()
st.caption("Built for personal use. Past performance and backtests do not guarantee future results.")
