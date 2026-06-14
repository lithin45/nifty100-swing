"""FII/DII daily net cash flows (₹ crore) from NSE via nsepython/nsefin."""
from __future__ import annotations

import pandas as pd

from common.logging_config import get_logger
from data_ingestion.base import DiskCache, FiiDiiProvider, normalize_ohlcv  # noqa: F401

log = get_logger(__name__)


class NseFiiDiiProvider(FiiDiiProvider):
    """Recent FII/DII activity. Returns a frame indexed by date with columns
    ``fii_net`` and ``dii_net`` (₹ crore)."""

    def __init__(self, cache_ttl_hours: float = 18.0) -> None:
        self.cache = DiskCache("fii_dii", ttl_hours=cache_ttl_hours)

    def get_recent(self, lookback_days: int = 5) -> pd.DataFrame:
        cached = self.cache.get("recent")
        if cached is not None:
            return cached.tail(lookback_days)

        df = pd.DataFrame(columns=["fii_net", "dii_net"])
        try:
            from nsepython import nse_fiidii

            raw = nse_fiidii()
            rows = []
            for rec in raw:
                cat = str(rec.get("category", "")).upper()
                date = rec.get("date")
                net = rec.get("netValue") or rec.get("net")
                try:
                    net = float(str(net).replace(",", ""))
                except (TypeError, ValueError):
                    continue
                rows.append({"date": date, "category": cat, "net": net})
            if rows:
                tmp = pd.DataFrame(rows)
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce", dayfirst=True)
                pivot = tmp.pivot_table(index="date", columns="category", values="net", aggfunc="last")
                df = pd.DataFrame(index=pivot.index)
                df["fii_net"] = pivot.filter(regex="FII|FPI").sum(axis=1)
                df["dii_net"] = pivot.filter(regex="DII").sum(axis=1)
                df = df.sort_index()
        except Exception as exc:
            log.debug("nse_fiidii failed: %s", exc)

        if len(df):
            self.cache.set("recent", df)
        return df.tail(lookback_days)


def get_fii_dii_provider() -> FiiDiiProvider:
    return NseFiiDiiProvider()
