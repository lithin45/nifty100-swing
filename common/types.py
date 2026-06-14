"""Shared domain types used across analyzers, scoring, storage and alerting.

Centralised here to avoid circular imports. The two normalisation conventions
used throughout the codebase:

* Every analyzer returns a :class:`SubScore` whose ``score`` is in **[0, 1]**
  (this is what the weighted composite consumes). Analyzers whose natural
  output is bipolar (sentiment, FII/DII, macro) also expose ``raw`` in
  **[-1, 1]** for display/debugging; ``score`` is then ``(raw + 1) / 2``.
* Gates are pure booleans wrapped in :class:`GateResult`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Optional


class SignalAction(str, Enum):
    """Direction of an emitted alert. The system is long-only / signal-only."""

    BUY = "BUY"
    EXIT = "EXIT"


class ExitReason(str, Enum):
    """Why an EXIT alert fired (see ``scoring/exits.py``)."""

    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    SIGNAL_DECAY = "signal_decay"
    TREND_REVERSAL = "trend_reversal"
    SECTOR_ROLLOVER = "sector_rollover"


def clamp01(x: float) -> float:
    """Clamp to [0, 1]."""
    return max(0.0, min(1.0, float(x)))


def bipolar_to_unit(x: float) -> float:
    """Map a value in [-1, 1] to [0, 1]."""
    return clamp01((float(x) + 1.0) / 2.0)


@dataclass
class SubScore:
    """A single analyzer's normalized contribution.

    Attributes
    ----------
    key:    machine key matching a weight in ``settings.yaml`` (e.g. "technical").
    score:  normalized value in [0, 1] consumed by the composite.
    reason: plain-English explanation suitable for a non-financial reader.
    raw:    optional natural-scale value (e.g. [-1, 1] for sentiment).
    details: arbitrary structured detail (indicator values, sub-checks…).
    """

    key: str
    score: float
    reason: str = ""
    raw: Optional[float] = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = clamp01(self.score)


@dataclass
class GateResult:
    """Outcome of a single hard gate."""

    name: str
    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateReport:
    """Collection of gate results. The signal is blocked unless all pass."""

    results: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.results) > 0 and all(r.passed for r in self.results)

    @property
    def failed(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        if self.passed:
            return "All gates passed"
        return "; ".join(f"{r.name} blocked ({r.reason})" for r in self.failed)


@dataclass
class TradePlan:
    """Concrete, capital-agnostic trade parameters for a BUY signal."""

    entry_price: float
    stop_loss: float
    target: float
    atr: float
    risk_per_share: float
    reward_per_share: float
    rr: float
    position_size_pct: float
    notes: str = ""


@dataclass
class CompositeResult:
    """Result of the Stage-2 weighted composite."""

    score: float  # 0..100
    subscores: list[SubScore] = field(default_factory=list)
    contributions: dict[str, float] = field(default_factory=dict)  # key -> weighted points
    penalties: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class Signal:
    """A fully-formed BUY or EXIT alert, ready to persist and send."""

    symbol: str
    sector: str
    action: SignalAction
    as_of: date
    composite: float
    plan: Optional[TradePlan] = None
    reasons: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    exit_reason: Optional[ExitReason] = None
    details: dict[str, Any] = field(default_factory=dict)
