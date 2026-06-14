"""Indian equity round-trip transaction costs.

Models the real cost stack so the backtest is honest: brokerage, STT, exchange
transaction charge, SEBI fee, stamp duty (buy only), 18% GST on
(brokerage + txn + SEBI), plus configurable slippage on both legs.

All rates come from ``settings.costs`` and are expressed as **percent** values.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostBreakdown:
    brokerage: float
    stt: float
    exchange_txn: float
    sebi: float
    stamp_duty: float
    gst: float
    slippage: float

    @property
    def total(self) -> float:
        return (self.brokerage + self.stt + self.exchange_txn + self.sebi
                + self.stamp_duty + self.gst + self.slippage)

    def as_dict(self) -> dict[str, float]:
        d = {k: round(v, 4) for k, v in self.__dict__.items()}
        d["total"] = round(self.total, 4)
        return d


def _leg_brokerage(value: float, pct: float, flat: float) -> float:
    charge = pct / 100.0 * value
    if flat and flat > 0:
        return min(charge, flat)
    return charge


def compute_trade_costs(buy_value: float, sell_value: float, costs_cfg) -> CostBreakdown:
    """Round-trip cost in absolute currency for one buy + one sell leg."""
    delivery = costs_cfg.segment.lower() == "delivery"
    turnover = buy_value + sell_value
    b = costs_cfg.brokerage

    if delivery:
        brokerage = (_leg_brokerage(buy_value, b.delivery_pct, b.delivery_flat)
                     + _leg_brokerage(sell_value, b.delivery_pct, b.delivery_flat))
        stt = (costs_cfg.stt.delivery_buy_pct / 100.0 * buy_value
               + costs_cfg.stt.delivery_sell_pct / 100.0 * sell_value)
    else:
        brokerage = (_leg_brokerage(buy_value, b.intraday_pct, b.intraday_flat)
                     + _leg_brokerage(sell_value, b.intraday_pct, b.intraday_flat))
        stt = costs_cfg.stt.intraday_sell_pct / 100.0 * sell_value

    exchange_txn = costs_cfg.exchange_txn_pct / 100.0 * turnover
    sebi = costs_cfg.sebi_pct / 100.0 * turnover
    stamp_duty = costs_cfg.stamp_duty_buy_pct / 100.0 * buy_value
    gst = costs_cfg.gst_pct / 100.0 * (brokerage + exchange_txn + sebi)
    slippage = costs_cfg.slippage_pct / 100.0 * turnover

    return CostBreakdown(brokerage, stt, exchange_txn, sebi, stamp_duty, gst, slippage)


def round_trip_cost_pct(price: float, costs_cfg, quantity: float = 1.0) -> float:
    """Round-trip cost as a % of invested value (entry == exit price proxy).

    Useful for quick net-return adjustments. Slippage dominates for liquid stocks.
    """
    value = price * quantity
    if value <= 0:
        return 0.0
    cb = compute_trade_costs(value, value, costs_cfg)
    return cb.total / value * 100.0
