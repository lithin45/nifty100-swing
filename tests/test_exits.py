import datetime as dt
from types import SimpleNamespace

import pandas as pd

from analyzers.context import MarketContext
from common.types import ExitReason
from scoring.exits import evaluate_exit


def _pos(**kw):
    base = dict(entry_price=150.0, stop_loss=144.0, target=165.0, atr=3.0,
                highest_close=150.0, current_stop=144.0,
                entry_date=dt.date(2026, 6, 1), sector="IT")
    base.update(kw)
    return SimpleNamespace(**base)


def _bar(hi, lo, cl, asof=dt.date(2026, 6, 12)):
    return pd.DataFrame({"open": [cl], "high": [hi], "low": [lo], "close": [cl], "volume": [1e6]},
                        index=[pd.Timestamp(asof)])


def _mctx(settings, asof=dt.date(2026, 6, 12), sector_rs=None):
    return MarketContext(as_of=asof, settings=settings, sector_rs=sector_rs or {})


def test_target_hit(settings):
    d = evaluate_exit(_pos(), _bar(166, 160, 165), 70, _mctx(settings))
    assert d.should_exit and d.reason == ExitReason.TARGET_HIT
    assert d.pnl_pct > 0


def test_stop_hit(settings):
    d = evaluate_exit(_pos(), _bar(148, 143, 144), 70, _mctx(settings))
    assert d.should_exit and d.reason == ExitReason.STOP_HIT
    assert d.pnl_pct < 0


def test_signal_decay(settings):
    d = evaluate_exit(_pos(), _bar(152, 149, 151), 40, _mctx(settings))
    assert d.should_exit and d.reason == ExitReason.SIGNAL_DECAY


def test_time_exit(settings):
    pos = _pos(target=9999.0, entry_date=dt.date(2026, 6, 12) - dt.timedelta(days=40))
    d = evaluate_exit(pos, _bar(152, 149, 151), 70, _mctx(settings))
    assert d.should_exit and d.reason == ExitReason.TIME_EXIT


def test_hold_returns_no_exit(settings):
    d = evaluate_exit(_pos(), _bar(153, 150, 152), 70, _mctx(settings))
    assert not d.should_exit
    assert d.reason is None


def test_trailing_stop_ratchets_up(settings):
    # Price has run far above entry -> trailing stop should rise above original stop.
    pos = _pos(highest_close=170.0)
    d = evaluate_exit(pos, _bar(172, 169, 171), 70, _mctx(settings))
    assert d.new_current_stop > pos.stop_loss  # ratcheted upward
