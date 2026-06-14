"""SQLAlchemy 2.0 ORM models for the SQLite store.

Tables
------
runs          one row per EOD pipeline execution (status, market regime, counts)
signals       every emitted BUY / EXIT alert
sub_scores    per-analyzer breakdown for an emitted signal (audit + dashboard)
gate_records  per-stock hard-gate outcomes (explains why a stock was blocked)
positions     open/closed position lifecycle (drives exits + dashboard P&L)

The scheduled job writes; the dashboard reads.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> dt.datetime:
    """Naive UTC timestamp (SQLite-friendly)."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running|success|error
    trading_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)

    n_evaluated: Mapped[int] = mapped_column(Integer, default=0)
    n_buy: Mapped[int] = mapped_column(Integer, default=0)
    n_exit: Mapped[int] = mapped_column(Integer, default=0)

    market_regime: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    data_freshness: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    signals: Mapped[list["Signal"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    gate_records: Mapped[list["GateRecord"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    sector: Mapped[str] = mapped_column(String(48), default="")
    action: Mapped[str] = mapped_column(String(8))  # BUY | EXIT
    as_of: Mapped[dt.date] = mapped_column(Date, index=True)
    composite: Mapped[float] = mapped_column(Float, default=0.0)

    # Trade plan (BUY); nullable for EXIT alerts.
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    target: Mapped[float | None] = mapped_column(Float, nullable=True)
    atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_per_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    reward_per_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    rr: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_size_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    reasons: Mapped[list] = mapped_column(JSON, default=list)
    risk_flags: Mapped[list] = mapped_column(JSON, default=list)
    exit_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

    run: Mapped["Run"] = relationship(back_populates="signals")
    sub_scores: Mapped[list["SubScoreRecord"]] = relationship(
        back_populates="signal", cascade="all, delete-orphan"
    )


class SubScoreRecord(Base):
    __tablename__ = "sub_scores"

    id: Mapped[int] = mapped_column(primary_key=True)
    signal_id: Mapped[int] = mapped_column(ForeignKey("signals.id"), index=True)
    key: Mapped[str] = mapped_column(String(32))
    score: Mapped[float] = mapped_column(Float)
    raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    weighted_points: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    details: Mapped[dict] = mapped_column(JSON, default=dict)

    signal: Mapped["Signal"] = relationship(back_populates="sub_scores")


class GateRecord(Base):
    __tablename__ = "gate_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    as_of: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    name: Mapped[str] = mapped_column(String(32))
    passed: Mapped[bool] = mapped_column(Boolean)
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

    run: Mapped["Run"] = relationship(back_populates="gate_records")


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    sector: Mapped[str] = mapped_column(String(48), default="")
    status: Mapped[str] = mapped_column(String(8), default="open", index=True)  # open|closed

    entry_signal_id: Mapped[int | None] = mapped_column(ForeignKey("signals.id"), nullable=True)
    entry_date: Mapped[dt.date] = mapped_column(Date)
    entry_price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    target: Mapped[float] = mapped_column(Float)
    atr: Mapped[float | None] = mapped_column(Float, nullable=True)
    size_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Trailing-stop bookkeeping (ratchets up only).
    highest_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_stop: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_composite: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_price: Mapped[float | None] = mapped_column(Float, nullable=True)

    exit_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    holding_days: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
