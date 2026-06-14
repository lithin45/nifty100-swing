import datetime as dt

import numpy as np
import pandas as pd

from analyzers.context import MarketContext, StockContext
from common.types import SubScore
from scoring.composite import compute_composite
from scoring.signal import build_trade_plan


def _subs(tech=0.7, sentiment_raw=0.4, **overrides):
    base = {
        "technical": SubScore("technical", tech, "tech"),
        "sector": SubScore("sector", 0.7, "sector", raw=0.4),
        "fii_dii": SubScore("fii_dii", 0.6, "fii", raw=0.2),
        "sentiment": SubScore("sentiment", (sentiment_raw + 1) / 2, "news", raw=sentiment_raw),
        "fundamental": SubScore("fundamental", 0.6, "fund"),
        "event": SubScore("event", 0.6, "event"),
        "macro": SubScore("macro", 0.55, "macro", raw=0.1),
        "vix": SubScore("vix", 0.7, "vix"),
    }
    base.update(overrides)
    return list(base.values())


def test_composite_in_range_and_weighted(settings):
    res = compute_composite(_subs(), settings)
    assert 0 <= res.score <= 100
    assert res.contributions["technical"] == max(res.contributions.values())


def test_conflict_penalty_applies_for_strong_tech_negative_news(settings):
    clean = compute_composite(_subs(tech=0.9, sentiment_raw=0.5), settings).score
    conflict = compute_composite(_subs(tech=0.9, sentiment_raw=-0.6), settings)
    # negative news with strong chart -> penalty multiplier recorded + lower score
    assert conflict.penalties
    assert conflict.score < clean


def test_build_trade_plan_stop_target_size(settings):
    details = {
        "indicators": {"close": 100.0, "atr": 2.0},
        "entry_level": 100.0,
        "pattern": None,
    }
    plan = build_trade_plan(details, settings)
    assert plan is not None
    # stop = 100 - 2.0*ATR(2.0) = 96 ; risk 4 ; target = 100 + 2R = 108
    assert abs(plan.stop_loss - 96.0) < 1e-6
    assert abs(plan.target - 108.0) < 1e-6
    assert abs(plan.rr - 2.0) < 1e-6
    # size: risk 1% / stop-distance 4% = 25% -> capped at max_position_pct (10%)
    assert plan.position_size_pct == settings.risk.max_position_pct


def test_pattern_target_overrides_when_higher(settings):
    details = {
        "indicators": {"close": 100.0, "atr": 2.0},
        "entry_level": 100.0,
        "pattern": {"name": "double_bottom", "target": 130.0},
    }
    plan = build_trade_plan(details, settings)
    assert plan.target == 130.0  # measured-move target beats 2R (108)


def test_full_scoring_produces_buy(settings, asof, strong_uptrend_df, reliance):
    from scoring.composite import run_analyzers
    from scoring.gates import run_gates
    from scoring.signal import generate_signal

    idx = pd.date_range("2025-06-02", periods=260, freq="B")
    mctx = MarketContext(
        as_of=asof, settings=settings,
        vix=pd.Series([13.0], index=idx[-1:]),
        benchmark=pd.Series(np.linspace(20000, 24000, 260), index=idx),
        fii_dii=pd.DataFrame({"fii_net": [1500, 1200, 900], "dii_net": [200, 300, 100]}),
        sector_rs={"Oil & Gas": 0.5}, regime={"vix": 13.0},
    )
    sctx = StockContext(reliance, strong_uptrend_df,
                        fundamentals={"roe": 25, "pe": 22, "de": 0.2, "earnings_growth": 18},
                        headlines=[{"title": "Reliance Industries profit jumps, wins record order"}])
    report = run_gates(sctx, mctx)
    composite = compute_composite(run_analyzers(sctx, mctx), settings)
    sig = generate_signal(sctx, mctx, composite, report)
    assert sig is not None
    assert sig.composite >= settings.scoring.entry_threshold
    assert sig.plan.rr > 0 and sig.plan.position_size_pct > 0
    assert len(sig.reasons) >= 1
