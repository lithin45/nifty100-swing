"""Performance metrics + plain-English summary + optional QuantStats tearsheet.

Metrics are computed natively (no hard dependency) so the backtest always
reports numbers; if ``quantstats`` is installed, :func:`generate_tearsheet`
writes a full HTML report too.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _to_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    if returns.std(ddof=0) == 0 or len(returns) < 2:
        return 0.0
    excess = returns - rf / TRADING_DAYS
    return math.sqrt(TRADING_DAYS) * excess.mean() / returns.std(ddof=0)


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    downside = returns[returns < 0]
    dd = downside.std(ddof=0)
    if dd == 0 or len(returns) < 2:
        return 0.0
    excess = returns - rf / TRADING_DAYS
    return math.sqrt(TRADING_DAYS) * excess.mean() / dd


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """Headline metrics from an equity curve and a trades table.

    ``trades`` must have a ``return_pct`` column (net % per trade) and optionally
    ``bars_held``.
    """
    out: dict[str, float] = {}
    if equity is not None and len(equity) >= 2:
        returns = _to_returns(equity)
        out["cagr_pct"] = round(cagr(equity) * 100, 2)
        out["total_return_pct"] = round((equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2)
        out["sharpe"] = round(sharpe(returns), 2)
        out["sortino"] = round(sortino(returns), 2)
        out["max_drawdown_pct"] = round(max_drawdown(equity) * 100, 2)
        out["volatility_pct"] = round(returns.std(ddof=0) * math.sqrt(TRADING_DAYS) * 100, 2)

    if trades is not None and len(trades):
        r = trades["return_pct"]
        wins = r[r > 0]
        losses = r[r <= 0]
        out["trades"] = int(len(trades))
        out["win_rate_pct"] = round(len(wins) / len(trades) * 100, 2)
        out["avg_win_pct"] = round(wins.mean(), 2) if len(wins) else 0.0
        out["avg_loss_pct"] = round(losses.mean(), 2) if len(losses) else 0.0
        gross_win = wins.sum()
        gross_loss = abs(losses.sum())
        out["profit_factor"] = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")
        out["expectancy_pct"] = round(r.mean(), 2)
        if "bars_held" in trades.columns and len(trades):
            out["avg_holding_days"] = round(float(trades["bars_held"].mean()), 1)
    else:
        out["trades"] = 0

    # Coerce numpy scalars -> native python types (JSON-safe for storage/dashboard).
    def _native(v):
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    return {k: _native(v) for k, v in out.items()}


def plain_english_summary(metrics: dict) -> str:
    """Beginner-friendly narrative of the headline metrics."""
    if not metrics or metrics.get("trades", 0) == 0:
        return "No trades were generated in this period, so there is nothing to evaluate."

    lines = []
    cagr_ = metrics.get("cagr_pct")
    if cagr_ is not None:
        lines.append(f"• Grew about {cagr_:.1f}% per year on average (CAGR).")
    if "max_drawdown_pct" in metrics:
        lines.append(
            f"• Worst peak-to-trough drop was {abs(metrics['max_drawdown_pct']):.1f}% "
            f"(the biggest scary dip you'd have sat through)."
        )
    if "sharpe" in metrics:
        q = ("good" if metrics["sharpe"] >= 1 else "modest" if metrics["sharpe"] >= 0.5 else "weak")
        lines.append(f"• Risk-adjusted return (Sharpe) was {metrics['sharpe']:.2f} — {q}.")
    lines.append(
        f"• Won {metrics.get('win_rate_pct', 0):.0f}% of {metrics.get('trades', 0)} trades; "
        f"profit factor {metrics.get('profit_factor', 0)} "
        f"(>1 means winners outweigh losers)."
    )
    lines.append(
        f"• Average winner +{metrics.get('avg_win_pct', 0):.1f}%, "
        f"average loser {metrics.get('avg_loss_pct', 0):.1f}%, "
        f"held ~{metrics.get('avg_holding_days', 0)} days."
    )
    return "\n".join(lines)


def generate_tearsheet(equity: pd.Series, output_path: str,
                       benchmark: Optional[pd.Series] = None, title: str = "Strategy") -> bool:
    """Write a QuantStats HTML tearsheet. Returns False if quantstats absent."""
    try:
        import quantstats as qs
    except Exception:
        return False
    returns = _to_returns(equity)
    returns.index = pd.to_datetime(returns.index)
    bench_ret = _to_returns(benchmark) if benchmark is not None else None
    try:
        qs.reports.html(returns, benchmark=bench_ret, output=output_path, title=title)
        return True
    except Exception:
        return False
