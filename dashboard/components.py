"""Reusable, beginner-friendly Streamlit UI components.

Everything here assumes the reader has NO finance background: jargon gets a
tooltip via the GLOSSARY, and status uses green/amber/red cues.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd
import streamlit as st

GLOSSARY = {
    "Composite / Conviction": "An overall 0–100 confidence score combining chart, "
        "sector, news, fundamentals and market mood. Higher = stronger setup.",
    "Entry": "The reference price to buy around (a breakout level or recent close).",
    "Stop-loss": "A safety exit price. If the stock falls here, you sell to cap the loss.",
    "Target": "The price where the plan suggests taking profit.",
    "R:R (Reward:Risk)": "How much you aim to gain versus risk. 2.0 means you target "
        "twice what you'd lose if stopped out.",
    "Position size": "Suggested portion of your capital for this trade, set so a stop-out "
        "only costs a small, fixed % of your account.",
    "ATR": "Average True Range — how much the stock typically moves per day. Used to place "
        "a stop that respects the stock's normal wiggle.",
    "200-DMA": "The average price over the last 200 days. Trading above it = long-term uptrend.",
    "India VIX": "The market's 'fear gauge'. Low = calm, high = nervous markets.",
    "FII / DII": "Foreign and domestic big institutions. Their net buying/selling moves the market.",
    "RS (Relative Strength)": "Whether a sector is outperforming the broad market.",
}


def color_for(value: float, good: float, bad: float) -> str:
    """Return 🟢/🟡/🔴 based on thresholds (direction inferred from good vs bad)."""
    if good >= bad:
        return "🟢" if value >= good else "🔴" if value <= bad else "🟡"
    return "🟢" if value <= good else "🔴" if value >= bad else "🟡"


def glossary_expander() -> None:
    with st.expander("📖 What do these terms mean? (tap to learn)"):
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}** — {definition}")


def regime_banner(run) -> None:
    """Top-of-page market mood banner."""
    if run is None:
        st.info("No analysis run found yet. Run the end-of-day job to populate signals.")
        return
    regime = run.market_regime or {}
    vix = regime.get("vix")
    nifty_ok = regime.get("nifty_above_200dma")
    fii = regime.get("fii_net_cr")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Last run", str(run.trading_date or run.started_at.date()),
                  help="The trading day these signals are for.")
    with c2:
        if vix is not None:
            st.metric(f"{color_for(vix, 14, 24)} India VIX", f"{vix:.1f}",
                      help=GLOSSARY["India VIX"])
        else:
            st.metric("India VIX", "n/a")
    with c3:
        if nifty_ok is not None:
            st.metric("Market trend", "🟢 Uptrend" if nifty_ok else "🔴 Weak",
                      help="Is Nifty 100 above its 200-day average?")
        else:
            st.metric("Market trend", "n/a")
    with c4:
        if fii is not None:
            st.metric(f"{color_for(fii, 0, -2000)} FII flow (₹cr)", f"{fii:,.0f}",
                      help=GLOSSARY["FII / DII"])
        else:
            st.metric("FII flow", "n/a")


def signal_card(sig) -> None:
    """One BUY signal with a 'Why this fired' expander."""
    badge = color_for(sig.composite, 75, 60)
    header = f"{badge} **{sig.symbol}** · {sig.sector} — Conviction {sig.composite:.0f}/100"
    with st.container(border=True):
        st.markdown(header)
        if sig.entry_price:
            cols = st.columns(4)
            cols[0].metric("Entry", f"₹{sig.entry_price:,.2f}", help=GLOSSARY["Entry"])
            stop_pct = (sig.stop_loss - sig.entry_price) / sig.entry_price * 100 if sig.entry_price else 0
            cols[1].metric("Stop-loss", f"₹{sig.stop_loss:,.2f}", f"{stop_pct:.1f}%",
                           help=GLOSSARY["Stop-loss"])
            tgt_pct = (sig.target - sig.entry_price) / sig.entry_price * 100 if sig.entry_price else 0
            cols[2].metric("Target", f"₹{sig.target:,.2f}", f"{tgt_pct:+.1f}%",
                           help=GLOSSARY["Target"])
            cols[3].metric("Size / R:R", f"{sig.position_size_pct:.1f}% · {sig.rr:.1f}R",
                           help=GLOSSARY["Position size"])
        if sig.risk_flags:
            st.warning("⚠️ " + " · ".join(sig.risk_flags))
        with st.expander("Why this fired"):
            for r in (sig.reasons or []):
                st.markdown(f"- {r}")
            if getattr(sig, "sub_scores", None):
                st.caption("Factor breakdown (0–100% strength):")
                for ss in sorted(sig.sub_scores, key=lambda x: x.weighted_points or 0, reverse=True):
                    st.progress(min(1.0, max(0.0, ss.score)),
                                text=f"{ss.key}: {ss.score*100:.0f}% · {ss.reason[:70]}")


def positions_table(positions: list, prices: Optional[dict] = None) -> pd.DataFrame:
    """Open positions with live-ish P&L and 'distance to stop/target'."""
    rows = []
    for p in positions:
        last = (prices or {}).get(p.symbol) or p.last_price or p.entry_price
        pnl = (last - p.entry_price) / p.entry_price * 100 if p.entry_price else 0.0
        to_stop = (last - (p.current_stop or p.stop_loss)) / last * 100 if last else 0.0
        to_target = (p.target - last) / last * 100 if last and p.target else 0.0
        rows.append({
            "Stock": p.symbol,
            "Sector": p.sector,
            "Entry": round(p.entry_price, 2),
            "Now": round(last, 2),
            "P&L %": round(pnl, 1),
            "Stop": round(p.current_stop or p.stop_loss, 2),
            "→ Stop %": round(to_stop, 1),
            "Target": round(p.target, 2) if p.target else None,
            "→ Target %": round(to_target, 1),
            "Held (d)": (dt.date.today() - p.entry_date).days,
        })
    df = pd.DataFrame(rows)
    return df


def paper_trading_section(open_positions: list, closed: list) -> None:
    """Render the paper-trading track record: stats, cumulative P&L, full ledger.

    Every BUY signal is logged as a simulated position (no real money). This view
    summarises that running record. It populates over time as positions exit.
    """
    import plotly.graph_objects as go

    st.caption("Every BUY signal is logged as a **simulated** trade — no real money. "
               "This is the system's running track record; it fills in as positions "
               "hit their target, stop, or time exit over the coming days.")

    n_open, n_closed = len(open_positions), len(closed)
    wins = [c for c in closed if (c.pnl_pct or 0) > 0]
    losses = [c for c in closed if (c.pnl_pct or 0) <= 0]
    win_rate = (len(wins) / n_closed * 100) if n_closed else 0.0
    # Portfolio-style P&L: each trade's % result weighted by its position size %.
    realized = sum((c.pnl_pct or 0) * (c.size_pct or 0) / 100 for c in closed)
    unrealized = 0.0
    for p in open_positions:
        last = p.last_price or p.entry_price
        if p.entry_price:
            unrealized += (last - p.entry_price) / p.entry_price * 100 * (p.size_pct or 0) / 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open / Closed", f"{n_open} / {n_closed}")
    c2.metric("Win rate", f"{win_rate:.0f}%", help="Share of closed trades that made a profit")
    c3.metric("Realized P&L (est.)", f"{realized:+.2f}%",
              help="Each closed trade's % result, weighted by its position size — "
                   "an estimate of how the paper account has grown")
    c4.metric("Open P&L (est.)", f"{unrealized:+.2f}%",
              help="Unrealized P&L on the trades still open")

    if not n_closed:
        st.info("No closed paper trades yet. Your open trades show in the **Open Positions** "
                "tab; this record fills in once they hit a target, stop, or time exit.")
        return

    # Use (pnl_pct or 0) consistently: a closed trade can carry pnl_pct=None and
    # the loss bucket includes it, so summing the raw attribute would TypeError.
    avg_w = (sum(c.pnl_pct or 0 for c in wins) / len(wins)) if wins else 0.0
    avg_l = (sum(c.pnl_pct or 0 for c in losses) / len(losses)) if losses else 0.0
    gross_win = sum(c.pnl_pct or 0 for c in wins)
    gross_loss = abs(sum(c.pnl_pct or 0 for c in losses))
    pf = (gross_win / gross_loss) if gross_loss else (float("inf") if gross_win else 0.0)
    d1, d2, d3 = st.columns(3)
    d1.metric("Avg win", f"{avg_w:+.1f}%")
    d2.metric("Avg loss", f"{avg_l:+.1f}%")
    d3.metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}",
              help=">1 means winners outweigh losers")

    rows = sorted(closed, key=lambda c: (c.exit_date or dt.date.min))
    cum, series = 0.0, []
    for c in rows:
        cum += (c.pnl_pct or 0) * (c.size_pct or 0) / 100
        series.append({"date": c.exit_date, "cumulative": round(cum, 3)})
    cdf = pd.DataFrame(series)
    fig = go.Figure(go.Scatter(x=cdf["date"], y=cdf["cumulative"], mode="lines+markers",
                               line=dict(color="#2ca02c"), name="Paper P&L"))
    fig.update_layout(title="Cumulative paper P&L (% of capital, estimate)", height=300,
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Full closed-trade ledger**")
    ledger = pd.DataFrame([{
        "Stock": c.symbol, "Sector": c.sector,
        "Entry": round(c.entry_price, 2), "Exit": round(c.exit_price or 0, 2),
        "P&L %": round(c.pnl_pct or 0, 1), "Reason": c.exit_reason,
        "Held (d)": c.holding_days, "Entered": c.entry_date, "Exited": c.exit_date,
    } for c in rows[::-1]])  # newest first
    st.dataframe(ledger, use_container_width=True, hide_index=True,
                 column_config={"P&L %": st.column_config.NumberColumn(format="%.1f%%")})


def candlestick_chart(df: pd.DataFrame, *, title: str = "",
                      entry: float = None, stop: float = None, target: float = None,
                      lookback: int = 180):
    """Annotated candlestick + SMAs + volume (plotly figure)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from analyzers import indicators as I

    d = df.tail(lookback).copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"],
                                 close=d["close"], name="Price"), row=1, col=1)
    for length, color in [(20, "#1f77b4"), (50, "#ff7f0e"), (200, "#2ca02c")]:
        if len(df) >= length:
            fig.add_trace(go.Scatter(x=d.index, y=I.sma(df["close"], length).tail(lookback),
                                     line=dict(width=1, color=color), name=f"SMA{length}"),
                          row=1, col=1)
    for level, label, color in [(entry, "Entry", "#888"), (stop, "Stop", "#d62728"),
                                (target, "Target", "#2ca02c")]:
        if level:
            fig.add_hline(y=level, line_dash="dash", line_color=color, row=1, col=1,
                          annotation_text=label, annotation_position="right")
    fig.add_trace(go.Bar(x=d.index, y=d["volume"], name="Volume", marker_color="#aaa"),
                  row=2, col=1)
    fig.update_layout(title=title, height=520, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


def watchlist_table(items: list) -> None:
    """'Almost there' stocks — close to firing a BUY. Keep an eye on these."""
    st.caption("Stocks **close** to firing a BUY — either scoring just under the 65 cutoff, "
               "or strong but blocked by a single safety gate. Not buys yet — watch them.")
    if not items:
        st.info("Nothing on the watchlist right now. It fills in after an evening scan finds "
                "stocks scoring 55–64 (or one gate away). In a falling market it stays empty "
                "by design — the system waits for conditions to improve.")
        return
    rows = []
    for w in items:
        rows.append({
            "Stock": w.symbol,
            "Sector": w.sector,
            "Score": round(w.composite, 1),
            "Needs +": round(w.distance, 1),
            "Status": "Near miss" if w.gates_passed else f"Blocked: {w.blocking_gate}",
            "Why it's close": "; ".join((w.reasons or [])[:2]),
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.1f", help="0–100 conviction; 65 = BUY"),
            "Needs +": st.column_config.NumberColumn("Needs +", help="Points below the 65 BUY threshold"),
        },
    )
