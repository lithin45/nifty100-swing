import datetime as dt

from alerting.formatter import format_morning_brief
from storage import db


def test_save_and_load_watchlist(tmp_path):
    dbf = str(tmp_path / "w.db")
    db.init_db(dbf)
    rid = db.create_run(dt.date(2026, 6, 12), db_path=dbf)
    items = [
        {"symbol": "INFY", "sector": "IT", "as_of": dt.date(2026, 6, 12), "composite": 63.0,
         "distance": 2.0, "gates_passed": True, "blocking_gate": None,
         "status": "near_miss", "reasons": ["above 200-DMA"]},
        {"symbol": "TCS", "sector": "IT", "as_of": dt.date(2026, 6, 12), "composite": 68.0,
         "distance": 0.0, "gates_passed": False, "blocking_gate": "market_regime",
         "status": "blocked", "reasons": ["strong chart"]},
    ]
    db.save_watchlist(rid, items, db_path=dbf)

    wl = db.latest_watchlist(dbf)
    assert len(wl) == 2
    assert wl[0].symbol == "TCS"          # sorted by composite desc (68 > 63)
    assert wl[0].blocking_gate == "market_regime" and wl[0].gates_passed is False
    assert wl[1].symbol == "INFY" and wl[1].gates_passed is True


def test_morning_brief_includes_watchlist():
    watch = [
        {"symbol": "INFY", "composite": 63, "gates_passed": True, "blocking_gate": None},
        {"symbol": "TCS", "composite": 68, "gates_passed": False, "blocking_gate": "market_regime"},
    ]
    msg = format_morning_brief(dt.date(2026, 6, 15), "India VIX 14", [], watch=watch)
    assert "Watchlist" in msg
    assert "INFY" in msg and "TCS" in msg
    assert "blocked: market_regime" in msg
