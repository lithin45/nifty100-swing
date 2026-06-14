"""Format signals into Telegram-Markdown alert messages.

Entry alert: stock + sector, score, entry, ATR stop, target + R:R, position size,
top reasons, risk flags. Exit alert: stock, reason, entry vs current, P&L %,
holding days. Dynamic text is escaped so stray ``_ * [`` `` ` `` don't break
Markdown rendering.
"""
from __future__ import annotations

from common.types import ExitReason, Signal, SignalAction

_MD_SPECIALS = ("_", "*", "`", "[")

_EXIT_LABELS = {
    ExitReason.TARGET_HIT: "🎯 Target hit",
    ExitReason.STOP_HIT: "🛑 Stop-loss hit",
    ExitReason.TRAILING_STOP: "🪜 Trailing stop hit",
    ExitReason.TIME_EXIT: "⏳ Max holding reached",
    ExitReason.SIGNAL_DECAY: "📉 Signal weakened",
    ExitReason.TREND_REVERSAL: "🔄 Trend reversed",
    ExitReason.SECTOR_ROLLOVER: "🏭 Sector rolled over",
}

_FOOTER = "_Signal-only — execute manually on Upstox._"


def _esc(text: str) -> str:
    out = str(text)
    for ch in _MD_SPECIALS:
        out = out.replace(ch, "\\" + ch)
    return out


def _pct(a: float, b: float) -> str:
    """Signed % move from a to b."""
    if not a:
        return ""
    return f"{(b - a) / a * 100:+.1f}%"


def format_entry_alert(sig: Signal, top_reasons: int = 3) -> str:
    p = sig.plan
    lines = [f"🟢 *BUY — {_esc(sig.symbol)}*  _{_esc(sig.sector)}_",
             f"*Conviction:* {sig.composite:.0f}/100"]
    if p:
        lines += [
            "",
            f"*Entry (ref):* ₹{p.entry_price:,.2f}",
            f"*Stop-loss:* ₹{p.stop_loss:,.2f}  ({_pct(p.entry_price, p.stop_loss)})",
            f"*Target:* ₹{p.target:,.2f}  ({_pct(p.entry_price, p.target)})  •  *R:R* {p.rr:.1f}",
            f"*Suggested size:* ~{p.position_size_pct:.1f}% of capital",
        ]
    if sig.reasons:
        lines.append("")
        lines.append("*Why this fired:*")
        for i, r in enumerate(sig.reasons[:top_reasons], 1):
            lines.append(f"{i}. {_esc(r)}")
    if sig.risk_flags:
        lines.append("")
        lines.append("⚠️ *Risk flags:* " + "; ".join(_esc(f) for f in sig.risk_flags))
    lines += ["", _FOOTER]
    return "\n".join(lines)


def format_exit_alert(sig: Signal) -> str:
    d = sig.details or {}
    label = _EXIT_LABELS.get(sig.exit_reason, "Exit") if sig.exit_reason else "Exit"
    entry = d.get("entry_price")
    current = d.get("current_price")
    pnl = d.get("pnl_pct")
    held = d.get("holding_days")

    lines = [f"🔴 *EXIT — {_esc(sig.symbol)}*  _{_esc(sig.sector)}_",
             f"*Reason:* {label}"]
    if entry is not None and current is not None:
        lines.append(f"*Entry:* ₹{entry:,.2f}  →  *Now:* ₹{current:,.2f}")
    if pnl is not None:
        emoji = "🟩" if pnl >= 0 else "🟥"
        held_txt = f"  •  *Held:* {held} day(s)" if held is not None else ""
        lines.append(f"*P&L:* {emoji} {pnl:+.1f}%{held_txt}")
    if d.get("detail"):
        lines.append(f"_{_esc(d['detail'])}_")
    lines += ["", _FOOTER]
    return "\n".join(lines)


def format_alert(sig: Signal, top_reasons: int = 3) -> str:
    return (format_entry_alert(sig, top_reasons) if sig.action == SignalAction.BUY
            else format_exit_alert(sig))


def format_run_header(as_of, n_buy: int, n_exit: int, regime: dict | None = None) -> str:
    parts = [f"📊 *Nifty 100 Swing — {as_of}*",
             f"{n_buy} new BUY • {n_exit} EXIT"]
    if regime:
        vix = regime.get("vix")
        nifty_ok = regime.get("nifty_above_200dma")
        bits = []
        if vix is not None:
            bits.append(f"VIX {vix:.1f}")
        if nifty_ok is not None:
            bits.append("Nifty " + ("uptrend ✅" if nifty_ok else "below 200-DMA ⚠️"))
        if bits:
            parts.append("Regime: " + " • ".join(bits))
    return "\n".join(parts)
