"""Database access layer: engine/session management + high-level helpers.

The scheduled job calls the ``create_run`` / ``save_*`` / position helpers; the
Streamlit dashboard calls the read helpers. All helpers open their own short
session unless one is passed in, so callers don't manage transactions.
"""
from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, selectinload, sessionmaker

from common.types import GateReport, Signal as SignalDTO, SignalAction
from storage.models import (
    Base,
    GateRecord,
    Position,
    Run,
    Signal,
    SubScoreRecord,
    WatchItem,
    utcnow,
)

_ENGINES: dict[str, Engine] = {}
_SESSION_FACTORIES: dict[str, sessionmaker] = {}


def _resolve_db_path(db_path: Optional[str]) -> str:
    if db_path:
        return db_path
    from config.loader import get_db_path

    return get_db_path()


def get_engine(db_path: Optional[str] = None) -> Engine:
    """Return (and cache) an Engine for the given SQLite path."""
    path = _resolve_db_path(db_path)
    if path not in _ENGINES:
        _ENGINES[path] = create_engine(
            f"sqlite:///{path}",
            future=True,
            connect_args={"check_same_thread": False},
        )
    return _ENGINES[path]


def get_session_factory(db_path: Optional[str] = None) -> sessionmaker:
    path = _resolve_db_path(db_path)
    if path not in _SESSION_FACTORIES:
        _SESSION_FACTORIES[path] = sessionmaker(
            bind=get_engine(path), expire_on_commit=False, future=True
        )
    return _SESSION_FACTORIES[path]


def init_db(db_path: Optional[str] = None) -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(get_engine(db_path))


@contextmanager
def session_scope(db_path: Optional[str] = None) -> Iterator[Session]:
    """Transactional session context manager."""
    session = get_session_factory(db_path)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Runs                                                                        #
# --------------------------------------------------------------------------- #
def create_run(trading_date: dt.date, db_path: Optional[str] = None) -> int:
    """Open a new run row and return its id."""
    with session_scope(db_path) as s:
        run = Run(trading_date=trading_date, status="running", started_at=utcnow())
        s.add(run)
        s.flush()
        return run.id


def finish_run(
    run_id: int,
    status: str = "success",
    market_regime: Optional[dict] = None,
    data_freshness: Optional[dict] = None,
    error: Optional[str] = None,
    n_evaluated: Optional[int] = None,
    db_path: Optional[str] = None,
) -> None:
    with session_scope(db_path) as s:
        run = s.get(Run, run_id)
        if run is None:
            return
        run.status = status
        run.finished_at = utcnow()
        if n_evaluated is not None:
            run.n_evaluated = n_evaluated
        if market_regime is not None:
            run.market_regime = market_regime
        if data_freshness is not None:
            run.data_freshness = data_freshness
        if error is not None:
            run.error = error
        # Refresh counts from persisted signals.
        run.n_buy = sum(1 for sig in run.signals if sig.action == SignalAction.BUY.value)
        run.n_exit = sum(1 for sig in run.signals if sig.action == SignalAction.EXIT.value)


def latest_run(db_path: Optional[str] = None, exclude_premarket: bool = True) -> Optional[Run]:
    """Most recent run. By default skips pre-market brief runs so the dashboard's
    'today's signals' panel keeps showing the last real EOD scan (the pre-market
    run produces no BUY/EXIT signals and would otherwise blank the panel)."""
    with session_scope(db_path) as s:
        rows = s.scalars(
            select(Run)
            .options(selectinload(Run.signals).selectinload(Signal.sub_scores))
            .order_by(Run.started_at.desc())
            .limit(10)
        ).all()
        for run in rows:
            if (exclude_premarket and isinstance(run.market_regime, dict)
                    and run.market_regime.get("mode") == "premarket"):
                continue
            return run
        return rows[0] if rows else None


# --------------------------------------------------------------------------- #
# Signals + sub-scores                                                        #
# --------------------------------------------------------------------------- #
def save_signal(sig: SignalDTO, run_id: Optional[int] = None, db_path: Optional[str] = None) -> int:
    """Persist a :class:`common.types.Signal` and its sub-score breakdown."""
    plan = sig.plan
    with session_scope(db_path) as s:
        row = Signal(
            run_id=run_id,
            symbol=sig.symbol,
            sector=sig.sector,
            action=sig.action.value,
            as_of=sig.as_of,
            composite=float(sig.composite),
            entry_price=plan.entry_price if plan else None,
            stop_loss=plan.stop_loss if plan else None,
            target=plan.target if plan else None,
            atr=plan.atr if plan else None,
            risk_per_share=plan.risk_per_share if plan else None,
            reward_per_share=plan.reward_per_share if plan else None,
            rr=plan.rr if plan else None,
            position_size_pct=plan.position_size_pct if plan else None,
            reasons=list(sig.reasons),
            risk_flags=list(sig.risk_flags),
            exit_reason=sig.exit_reason.value if sig.exit_reason else None,
            details=dict(sig.details or {}),
        )
        for ss in sig.details.get("subscores", []):
            row.sub_scores.append(
                SubScoreRecord(
                    key=ss.get("key", ""),
                    score=float(ss.get("score", 0.0)),
                    raw=ss.get("raw"),
                    weighted_points=ss.get("weighted_points"),
                    reason=ss.get("reason", ""),
                    details=ss.get("details", {}) or {},
                )
            )
        s.add(row)
        s.flush()
        return row.id


def save_gate_records(
    run_id: Optional[int],
    symbol: str,
    as_of: dt.date,
    report: GateReport,
    db_path: Optional[str] = None,
) -> None:
    """Persist every hard-gate outcome for a stock (audit / dashboard)."""
    with session_scope(db_path) as s:
        for r in report.results:
            s.add(
                GateRecord(
                    run_id=run_id,
                    symbol=symbol,
                    as_of=as_of,
                    name=r.name,
                    passed=bool(r.passed),
                    reason=r.reason,
                )
            )


def save_watchlist(run_id: Optional[int], items: list[dict], db_path: Optional[str] = None) -> None:
    """Persist 'almost there' watch items for a run."""
    with session_scope(db_path) as s:
        for it in items:
            s.add(WatchItem(
                run_id=run_id, symbol=it["symbol"], sector=it.get("sector", ""),
                as_of=it["as_of"], composite=float(it["composite"]),
                distance=float(it.get("distance", 0.0)),
                gates_passed=bool(it.get("gates_passed", False)),
                blocking_gate=it.get("blocking_gate"),
                status=it.get("status", "near_miss"),
                reasons=list(it.get("reasons", [])),
            ))


def latest_watchlist(db_path: Optional[str] = None) -> list[WatchItem]:
    """Watch items from the single most recent run that produced any.

    Filters by the latest run_id (not by date): two runs on the SAME trading day
    — e.g. a re-run, or a manual scan after the scheduled one — would otherwise
    have their watch items merged and shown together as duplicates.
    """
    with session_scope(db_path) as s:
        latest = s.scalars(
            select(WatchItem).order_by(WatchItem.created_at.desc()).limit(1)
        ).first()
        if latest is None:
            return []
        return list(s.scalars(
            select(WatchItem).where(WatchItem.run_id == latest.run_id)
            .order_by(WatchItem.composite.desc())
        ))


def recent_signals(limit: int = 50, db_path: Optional[str] = None) -> list[Signal]:
    with session_scope(db_path) as s:
        return list(
            s.scalars(
                select(Signal)
                .options(selectinload(Signal.sub_scores))
                .order_by(Signal.created_at.desc())
                .limit(limit)
            )
        )


def signals_for_date(as_of: dt.date, db_path: Optional[str] = None) -> list[Signal]:
    with session_scope(db_path) as s:
        return list(
            s.scalars(
                select(Signal)
                .options(selectinload(Signal.sub_scores))
                .where(Signal.as_of == as_of)
                .order_by(Signal.composite.desc())
            )
        )


# --------------------------------------------------------------------------- #
# Positions                                                                   #
# --------------------------------------------------------------------------- #
def get_open_positions(db_path: Optional[str] = None) -> list[Position]:
    with session_scope(db_path) as s:
        return list(s.scalars(select(Position).where(Position.status == "open")))


def get_open_symbols(db_path: Optional[str] = None) -> set[str]:
    return {p.symbol for p in get_open_positions(db_path)}


def open_position(
    sig: SignalDTO,
    entry_signal_id: Optional[int] = None,
    db_path: Optional[str] = None,
) -> int:
    """Open a position from a BUY signal's trade plan."""
    if sig.plan is None:
        raise ValueError("Cannot open a position without a trade plan")
    plan = sig.plan
    with session_scope(db_path) as s:
        pos = Position(
            symbol=sig.symbol,
            sector=sig.sector,
            status="open",
            entry_signal_id=entry_signal_id,
            entry_date=sig.as_of,
            entry_price=plan.entry_price,
            stop_loss=plan.stop_loss,
            target=plan.target,
            atr=plan.atr,
            size_pct=plan.position_size_pct,
            highest_close=plan.entry_price,
            current_stop=plan.stop_loss,
            last_composite=sig.composite,
            last_price=plan.entry_price,
        )
        s.add(pos)
        s.flush()
        return pos.id


def update_position_fields(position_id: int, db_path: Optional[str] = None, **fields) -> None:
    with session_scope(db_path) as s:
        pos = s.get(Position, position_id)
        if pos is None:
            return
        for k, v in fields.items():
            setattr(pos, k, v)


def close_position(
    position_id: int,
    exit_date: dt.date,
    exit_price: float,
    exit_reason: str,
    db_path: Optional[str] = None,
) -> None:
    with session_scope(db_path) as s:
        pos = s.get(Position, position_id)
        if pos is None:
            return
        pos.status = "closed"
        pos.exit_date = exit_date
        pos.exit_price = exit_price
        pos.exit_reason = exit_reason
        pos.pnl_pct = (
            (exit_price - pos.entry_price) / pos.entry_price * 100.0 if pos.entry_price else None
        )
        pos.holding_days = (exit_date - pos.entry_date).days


def closed_positions(limit: Optional[int] = 200, db_path: Optional[str] = None) -> list[Position]:
    """Closed positions, newest first. ``limit=None`` returns the full history."""
    with session_scope(db_path) as s:
        q = (
            select(Position)
            .where(Position.status == "closed")
            .order_by(Position.exit_date.desc())
        )
        if limit is not None:
            q = q.limit(limit)
        return list(s.scalars(q))
