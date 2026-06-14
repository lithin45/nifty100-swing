"""FII/DII flow factor (market-wide).

Net institutional cash flow over the lookback window, mapped to [-1,1] using the
strong-inflow / strong-outflow thresholds from settings. FII flows are weighted
more heavily than DII (they move Nifty large-caps more).
"""
from __future__ import annotations

import pandas as pd

from common.types import SubScore, bipolar_to_unit
from analyzers.context import MarketContext, StockContext


class FiiDiiFactorAnalyzer:
    key = "fii_dii"

    def analyze(self, sctx: StockContext, mctx: MarketContext) -> SubScore:
        cfg = mctx.settings.fii_dii
        df = mctx.fii_dii
        if df is None or len(df) == 0 or "fii_net" not in df.columns:
            return SubScore(self.key, 0.5, "FII/DII data unavailable (neutral)", raw=0.0)

        recent = df.tail(cfg.lookback_days)
        fii = float(recent["fii_net"].sum())
        dii = float(recent.get("dii_net", pd.Series(dtype=float)).sum())

        def _norm(v: float) -> float:
            if v >= 0:
                return min(1.0, v / cfg.strong_inflow_cr) if cfg.strong_inflow_cr else 0.0
            return max(-1.0, v / abs(cfg.strong_outflow_cr)) if cfg.strong_outflow_cr else 0.0

        raw = max(-1.0, min(1.0, 0.7 * _norm(fii) + 0.3 * _norm(dii)))
        tone = "buying" if raw > 0.1 else "selling" if raw < -0.1 else "flat"
        reason = (f"Institutions net {tone}: FII ₹{fii:,.0f} cr, DII ₹{dii:,.0f} cr "
                  f"over {len(recent)}d")
        return SubScore(self.key, bipolar_to_unit(raw), reason, raw=raw,
                        details={"fii_net_cr": round(fii, 1), "dii_net_cr": round(dii, 1)})


def analyze_fii_dii(sctx: StockContext, mctx: MarketContext) -> SubScore:
    return FiiDiiFactorAnalyzer().analyze(sctx, mctx)
