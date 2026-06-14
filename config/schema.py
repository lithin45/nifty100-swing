"""Pydantic v2 schema for ``settings.yaml``.

Every tunable lives here with a sensible default, so a partial YAML still
validates and the system runs out-of-the-box. The schema also enforces ranges
(thresholds in [0, 100], non-negative weights/percentages, positive ATR
multiples) and surfaces config typos early.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Base(BaseModel):
    # Ignore unknown keys so adding experimental YAML keys does not crash the
    # app, but validate everything we *do* know about.
    model_config = ConfigDict(extra="ignore")


# --------------------------------------------------------------------------- #
# Universe row (loaded from nifty100.csv, not from YAML)                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stock:
    """One Nifty-100 constituent."""

    symbol: str
    sector: str
    isin: str = ""
    upstox_key: str = ""
    name: str = ""

    @property
    def yf_ticker(self) -> str:
        """yfinance ticker for an NSE equity."""
        return f"{self.symbol}.NS"

    @property
    def resolved_upstox_key(self) -> str:
        """Upstox instrument key. NSE equity keys are ``NSE_EQ|<ISIN>``."""
        if self.upstox_key:
            return self.upstox_key
        return f"NSE_EQ|{self.isin}" if self.isin else ""


# --------------------------------------------------------------------------- #
# Project / data                                                              #
# --------------------------------------------------------------------------- #
class ProjectCfg(_Base):
    name: str = "Nifty100 Swing System"
    timezone: str = "Asia/Kolkata"
    base_currency: str = "INR"


class DataCfg(_Base):
    primary_price_source: str = "yfinance"  # yfinance | upstox | dhan
    validate_with_bhavcopy: bool = True
    history_days: int = 800  # enough for a 200-day SMA + warm-up
    cache_ttl_hours: int = 20
    adjust_ohlc: bool = True


# --------------------------------------------------------------------------- #
# Gates                                                                       #
# --------------------------------------------------------------------------- #
class LiquidityGateCfg(_Base):
    enabled: bool = True
    min_avg_turnover_inr: float = 250_000_000.0  # ₹25 cr/day
    lookback_days: int = 20


class TrendGateCfg(_Base):
    enabled: bool = True
    require_above_sma: int = 200
    allow_sector_rs_override: bool = True


class EventGateCfg(_Base):
    enabled: bool = True
    no_entry_days_before_earnings: int = 5
    block_fno_ban: bool = True
    block_circuit: bool = True


class MarketRegimeGateCfg(_Base):
    enabled: bool = True
    vix_ceiling: float = 22.0
    require_index_above_sma: int = 200


class GatesCfg(_Base):
    liquidity: LiquidityGateCfg = Field(default_factory=LiquidityGateCfg)
    trend: TrendGateCfg = Field(default_factory=TrendGateCfg)
    event: EventGateCfg = Field(default_factory=EventGateCfg)
    market_regime: MarketRegimeGateCfg = Field(default_factory=MarketRegimeGateCfg)


# --------------------------------------------------------------------------- #
# Scoring                                                                     #
# --------------------------------------------------------------------------- #
class ConflictPenaltyCfg(_Base):
    enabled: bool = True
    strong_technical_threshold: float = 0.70   # technical sub-score [0,1]
    negative_sentiment_threshold: float = -0.30  # raw sentiment [-1,1]
    penalty_multiplier: float = 0.85
    event_risk_penalty_multiplier: float = 0.90


_DEFAULT_WEIGHTS = {
    "technical": 0.35,
    "sector": 0.15,
    "fii_dii": 0.10,
    "sentiment": 0.10,
    "fundamental": 0.10,
    "event": 0.08,
    "macro": 0.07,
    "vix": 0.05,
}


class ScoringCfg(_Base):
    entry_threshold: float = 65.0
    exit_threshold: float = 45.0
    weights: dict[str, float] = Field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    conflict_penalty: ConflictPenaltyCfg = Field(default_factory=ConflictPenaltyCfg)

    @field_validator("entry_threshold", "exit_threshold")
    @classmethod
    def _in_0_100(cls, v: float) -> float:
        if not 0 <= v <= 100:
            raise ValueError("thresholds must be in [0, 100]")
        return v

    @field_validator("weights")
    @classmethod
    def _non_negative(cls, v: dict[str, float]) -> dict[str, float]:
        if any(w < 0 for w in v.values()):
            raise ValueError("composite weights must be non-negative")
        if sum(v.values()) <= 0:
            raise ValueError("composite weights must sum to a positive number")
        return v

    @model_validator(mode="after")
    def _exit_below_entry(self) -> "ScoringCfg":
        # Not fatal, but exit >= entry would make exits fire immediately.
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("exit_threshold should be below entry_threshold")
        return self

    def normalized_weights(self) -> dict[str, float]:
        """Weights rescaled to sum to 1 (composite uses relative weights)."""
        total = sum(self.weights.values())
        return {k: v / total for k, v in self.weights.items()}


# --------------------------------------------------------------------------- #
# Technical                                                                   #
# --------------------------------------------------------------------------- #
class MacdCfg(_Base):
    fast: int = 12
    slow: int = 26
    signal: int = 9


class StochCfg(_Base):
    k: int = 14
    d: int = 3
    smooth: int = 3


class BollingerCfg(_Base):
    period: int = 20
    std: float = 2.0


class BreakoutCfg(_Base):
    lookback_high: int = 20
    volume_multiple: float = 1.5
    volume_avg_period: int = 20


class PivotsCfg(_Base):
    window: int = 5  # argrelextrema order


_DEFAULT_TECH_WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.25,
    "breakout": 0.20,
    "volume": 0.10,
    "patterns": 0.15,
}


class TechnicalCfg(_Base):
    sma_periods: list[int] = Field(default_factory=lambda: [20, 50, 200])
    ema_periods: list[int] = Field(default_factory=lambda: [20, 50])
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd: MacdCfg = Field(default_factory=MacdCfg)
    stoch: StochCfg = Field(default_factory=StochCfg)
    bollinger: BollingerCfg = Field(default_factory=BollingerCfg)
    atr_period: int = 14
    breakout: BreakoutCfg = Field(default_factory=BreakoutCfg)
    pivots: PivotsCfg = Field(default_factory=PivotsCfg)
    weights: dict[str, float] = Field(default_factory=lambda: dict(_DEFAULT_TECH_WEIGHTS))


class PatternsCfg(_Base):
    min_confidence: float = 0.5
    lookback: int = 120


# --------------------------------------------------------------------------- #
# Risk / exits                                                                #
# --------------------------------------------------------------------------- #
class TrailingCfg(_Base):
    enabled: bool = True
    type: str = "atr"  # atr | percent
    atr_multiple: float = 2.5
    percent: float = 8.0
    activate_at_r: float = 1.0  # begin trailing after +1R


class RiskCfg(_Base):
    atr_stop_multiple: float = 2.0   # k in entry - k*ATR
    rr_target_multiple: float = 2.0  # target = entry + rr * risk
    use_pattern_target: bool = True
    max_holding_days: int = 22       # ~1 trading month
    risk_per_trade_pct: float = 1.0  # % of capital risked per trade
    max_position_pct: float = 10.0   # cap on any single position
    trailing: TrailingCfg = Field(default_factory=TrailingCfg)

    @field_validator("atr_stop_multiple", "rr_target_multiple")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("ATR/RR multiples must be positive")
        return v


class ExitsCfg(_Base):
    signal_decay_threshold: float = 45.0
    macd_reversal_exit: bool = True
    trend_reversal_exit: bool = True   # close below 50-DMA
    trend_reversal_sma: int = 50
    sector_rollover_exit: bool = True


# --------------------------------------------------------------------------- #
# Costs (Indian round-trip)                                                   #
# --------------------------------------------------------------------------- #
class BrokerageCfg(_Base):
    delivery_pct: float = 0.0
    delivery_flat: float = 0.0
    intraday_pct: float = 0.03   # % per leg
    intraday_flat: float = 20.0  # min(flat, pct) per leg


class SttCfg(_Base):
    delivery_buy_pct: float = 0.1
    delivery_sell_pct: float = 0.1
    intraday_sell_pct: float = 0.025


class CostsCfg(_Base):
    segment: str = "delivery"  # delivery | intraday
    brokerage: BrokerageCfg = Field(default_factory=BrokerageCfg)
    stt: SttCfg = Field(default_factory=SttCfg)
    exchange_txn_pct: float = 0.00297  # NSE equity
    sebi_pct: float = 0.0001
    stamp_duty_buy_pct: float = 0.015
    gst_pct: float = 18.0          # on (brokerage + txn + sebi)
    slippage_pct: float = 0.05     # per leg


# --------------------------------------------------------------------------- #
# Backtest                                                                    #
# --------------------------------------------------------------------------- #
class WalkForwardCfg(_Base):
    enabled: bool = True
    in_sample_months: int = 18
    out_sample_months: int = 6
    step_months: int = 6


class BacktestCfg(_Base):
    start: str = "2019-01-01"
    end: Optional[str] = None
    initial_capital: float = 1_000_000.0
    max_open_positions: int = 10
    execution: str = "next_open"  # decide on close, execute next open
    walkforward: WalkForwardCfg = Field(default_factory=WalkForwardCfg)


# --------------------------------------------------------------------------- #
# External factors                                                            #
# --------------------------------------------------------------------------- #
class SentimentCfg(_Base):
    model: str = "ProsusAI/finbert"
    max_headlines_per_stock: int = 15
    recency_days: int = 7
    fallback: str = "lexicon"  # used when transformers/torch unavailable
    # If headlines arrive with a provider sentiment score (e.g. Marketaux),
    # prefer averaging those over running FinBERT/lexicon on the title text.
    prefer_provider_sentiment: bool = True


class MarketauxCfg(_Base):
    exchange_suffix: str = ".NSE"          # Marketaux NSE entity suffix
    countries: str = "in"
    language: str = "en"
    max_symbols_per_request: int = 20      # Marketaux allows batched symbols
    max_requests: int = 40                 # cap to respect the free tier (100/day)
    use_provider_sentiment: bool = True     # attach Marketaux's entity sentiment


class NewsCfg(_Base):
    provider: str = "rss"                   # rss | marketaux
    rss_feeds: list[str] = Field(
        default_factory=lambda: [
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "https://www.moneycontrol.com/rss/marketreports.xml",
            "https://www.moneycontrol.com/rss/business.xml",
            "https://www.business-standard.com/rss/markets-106.rss",
            "https://www.livemint.com/rss/markets",
            "https://pulse.zerodha.com/feed.php",
        ]
    )
    max_age_days: int = 7
    marketaux: MarketauxCfg = Field(default_factory=MarketauxCfg)


class FiiDiiCfg(_Base):
    lookback_days: int = 5
    strong_inflow_cr: float = 2000.0   # ₹ crore net (cash market)
    strong_outflow_cr: float = -2000.0


class MacroCfg(_Base):
    symbols: dict[str, str] = Field(
        default_factory=lambda: {
            "usdinr": "INR=X",
            "crude": "CL=F",
            "sp500": "^GSPC",
            "nasdaq": "^IXIC",
            "vix": "^INDIAVIX",
        }
    )
    rbi_repo_rate: float = 6.50  # update on MPC days
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "usdinr": 0.30,
            "crude": 0.20,
            "us_indices": 0.30,
            "rates": 0.20,
        }
    )
    trend_lookback_days: int = 21


class SectorsCfg(_Base):
    rs_lookback_days: int = 63  # ~3 months relative strength
    benchmark: str = "^CNX100"
    indices: dict[str, str] = Field(
        default_factory=lambda: {
            "Nifty IT": "^CNXIT",
            "Nifty Bank": "^NSEBANK",
            "Nifty Pharma": "^CNXPHARMA",
            "Nifty Auto": "^CNXAUTO",
            "Nifty FMCG": "^CNXFMCG",
            "Nifty Metal": "^CNXMETAL",
            "Nifty Energy": "^CNXENERGY",
            "Nifty Realty": "^CNXREALTY",
            "Nifty Financial Services": "NIFTY_FIN_SERVICE.NS",
            "Nifty Media": "^CNXMEDIA",
        }
    )


# --------------------------------------------------------------------------- #
# Alerts / dashboard / infra                                                  #
# --------------------------------------------------------------------------- #
class TelegramCfg(_Base):
    enabled: bool = True
    parse_mode: str = "Markdown"
    max_signals_per_run: int = 15
    disable_web_page_preview: bool = True


class AlertsCfg(_Base):
    telegram: TelegramCfg = Field(default_factory=TelegramCfg)
    include_position_size: bool = True
    top_reasons: int = 3


class DashboardCfg(_Base):
    title: str = "Nifty 100 Swing Signals"
    refresh_minutes: int = 30
    show_terms_glossary: bool = True


class LoggingCfg(_Base):
    level: str = "INFO"


class StorageCfg(_Base):
    db_path: Optional[str] = None  # None -> data/swing.db


# --------------------------------------------------------------------------- #
# Root                                                                        #
# --------------------------------------------------------------------------- #
class Settings(_Base):
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    universe_csv: str = "nifty100.csv"
    gates: GatesCfg = Field(default_factory=GatesCfg)
    scoring: ScoringCfg = Field(default_factory=ScoringCfg)
    technical: TechnicalCfg = Field(default_factory=TechnicalCfg)
    patterns: PatternsCfg = Field(default_factory=PatternsCfg)
    risk: RiskCfg = Field(default_factory=RiskCfg)
    exits: ExitsCfg = Field(default_factory=ExitsCfg)
    costs: CostsCfg = Field(default_factory=CostsCfg)
    backtest: BacktestCfg = Field(default_factory=BacktestCfg)
    sentiment: SentimentCfg = Field(default_factory=SentimentCfg)
    news: NewsCfg = Field(default_factory=NewsCfg)
    fii_dii: FiiDiiCfg = Field(default_factory=FiiDiiCfg)
    macro: MacroCfg = Field(default_factory=MacroCfg)
    sectors: SectorsCfg = Field(default_factory=SectorsCfg)
    alerts: AlertsCfg = Field(default_factory=AlertsCfg)
    dashboard: DashboardCfg = Field(default_factory=DashboardCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
