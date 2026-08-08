"""Shared deadline, progress, and clean-stop contracts for construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from time import monotonic
from typing import Callable


Clock = Callable[[], float]
ConstructionProgressSink = Callable[[dict[str, object]], None]
CONSTRUCTION_EVENT_SCHEMA_VERSION = 1


class ConstructionPhase(str, Enum):
    CACHE_VALIDATION = "cache_validation"
    ROUTING_PREPARATION = "routing_preparation"
    SUPPORT_DISCOVERY = "support_discovery"
    PLANNING = "planning"
    SHARD_VALIDATION = "shard_validation"
    SHARD_CONSTRUCTION = "shard_construction"
    TEMPORAL_BLOCK_ASSEMBLY = "temporal_block_assembly"
    FINAL_VALIDATION = "final_validation"
    PERSISTENCE = "persistence"
    COMPLETED = "completed"


class ConstructionTerminalStatus(str, Enum):
    COMPLETED = "completed"
    DECLINED = "declined"
    CACHE_REUSED = "cache_reused"
    DEADLINE_STOPPED = "deadline_stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ConstructionDeadline:
    """One monotonic budget shared by every construction phase."""

    started_at: float
    absolute_deadline: float | None
    safety_margin_seconds: float = 0.0
    clock: Clock = monotonic

    def __post_init__(self) -> None:
        if not math.isfinite(self.started_at):
            raise ValueError("started_at must be finite.")
        if self.absolute_deadline is not None and not math.isfinite(
            self.absolute_deadline
        ):
            raise ValueError("absolute_deadline must be finite when provided.")
        if (
            not math.isfinite(self.safety_margin_seconds)
            or self.safety_margin_seconds < 0.0
        ):
            raise ValueError("safety_margin_seconds must be finite and nonnegative.")

    @classmethod
    def unlimited(
        cls, *, safety_margin_seconds: float = 0.0, clock: Clock = monotonic
    ) -> ConstructionDeadline:
        return cls(clock(), None, safety_margin_seconds, clock)

    @classmethod
    def from_budget(
        cls,
        seconds: float | None,
        *,
        safety_margin_seconds: float = 0.0,
        clock: Clock = monotonic,
    ) -> ConstructionDeadline:
        started = clock()
        if seconds is None:
            return cls(started, None, safety_margin_seconds, clock)
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError("time budget must be finite and nonnegative.")
        return cls(started, started + seconds, safety_margin_seconds, clock)

    @classmethod
    def from_absolute(
        cls,
        absolute_deadline: float | None,
        *,
        safety_margin_seconds: float = 0.0,
        clock: Clock = monotonic,
    ) -> ConstructionDeadline:
        return cls(clock(), absolute_deadline, safety_margin_seconds, clock)

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    @property
    def remaining_seconds(self) -> float | None:
        if self.absolute_deadline is None:
            return None
        return max(0.0, self.absolute_deadline - self.clock())

    @property
    def expired(self) -> bool:
        return (
            self.absolute_deadline is not None
            and self.clock() >= self.absolute_deadline
        )

    def may_start(self, predicted_seconds: float | None = None) -> bool:
        """Return whether one more indivisible operation fits safely."""
        predicted = 0.0 if predicted_seconds is None else float(predicted_seconds)
        if not math.isfinite(predicted) or predicted < 0.0:
            raise ValueError("predicted_seconds must be finite and nonnegative.")
        if self.absolute_deadline is None:
            return True
        remaining = self.remaining_seconds
        assert remaining is not None
        return predicted + self.safety_margin_seconds <= remaining


@dataclass(frozen=True, slots=True)
class ConstructionTermination:
    status: ConstructionTerminalStatus
    phase: ConstructionPhase
    reason: str
    elapsed_seconds: float
    remaining_seconds: float | None
    completed_units: int | None = None
    total_units: int | None = None
    next_resumable_position: str | None = None
    checkpoint_location: str | None = None
    artifact_location: str | None = None
    checkpoint_reusable: bool = False
    predicted_next_seconds: float | None = None
    deadline_overshoot_seconds: float = 0.0


class ConstructionDeadlineStop(RuntimeError):
    """Internal control-flow signal for an expected, clean bounded stop."""

    def __init__(self, termination: ConstructionTermination):
        super().__init__(termination.reason)
        self.termination = termination


def deadline_stop(
    deadline: ConstructionDeadline,
    *,
    phase: ConstructionPhase,
    reason: str,
    completed_units: int | None = None,
    total_units: int | None = None,
    next_resumable_position: str | None = None,
    checkpoint_location: str | None = None,
    artifact_location: str | None = None,
    checkpoint_reusable: bool = False,
    predicted_next_seconds: float | None = None,
) -> ConstructionDeadlineStop:
    overshoot = (
        0.0
        if deadline.absolute_deadline is None
        else max(0.0, deadline.clock() - deadline.absolute_deadline)
    )
    return ConstructionDeadlineStop(
        ConstructionTermination(
            status=ConstructionTerminalStatus.DEADLINE_STOPPED,
            phase=phase,
            reason=reason,
            elapsed_seconds=deadline.elapsed_seconds,
            remaining_seconds=deadline.remaining_seconds,
            completed_units=completed_units,
            total_units=total_units,
            next_resumable_position=next_resumable_position,
            checkpoint_location=checkpoint_location,
            artifact_location=artifact_location,
            checkpoint_reusable=checkpoint_reusable,
            predicted_next_seconds=predicted_next_seconds,
            deadline_overshoot_seconds=overshoot,
        )
    )


@dataclass(slots=True)
class ConstructionProgressReporter:
    """Emit versioned progress dictionaries with deterministic throttling."""

    deadline: ConstructionDeadline
    sink: ConstructionProgressSink | None = None
    minimum_interval_seconds: float = 1.0
    clock: Clock = monotonic
    _last_emitted_at: float | None = None

    def emit(
        self,
        *,
        phase: ConstructionPhase,
        status: str,
        force: bool = False,
        completed_units: int | None = None,
        total_units: int | None = None,
        current_unit: str | None = None,
        recent_unit_seconds: float | None = None,
        predicted_remaining_seconds: float | None = None,
        checkpoint_location: str | None = None,
        cache_hits: int | None = None,
        cache_misses: int | None = None,
        peak_resident_memory_bytes: int | None = None,
        terminal_reason: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        if self.sink is None:
            return
        now = self.clock()
        if (
            not force
            and self._last_emitted_at is not None
            and now - self._last_emitted_at < self.minimum_interval_seconds
        ):
            return
        self._last_emitted_at = now
        event = {
            "schema_version": CONSTRUCTION_EVENT_SCHEMA_VERSION,
            "phase": phase.value,
            "status": status,
            "elapsed_seconds": self.deadline.elapsed_seconds,
            "remaining_seconds": self.deadline.remaining_seconds,
            "safety_margin_seconds": self.deadline.safety_margin_seconds,
            "completed_units": completed_units,
            "total_units": total_units,
            "current_unit": current_unit,
            "recent_unit_seconds": recent_unit_seconds,
            "predicted_remaining_seconds": predicted_remaining_seconds,
            "checkpoint_location": checkpoint_location,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "peak_resident_memory_bytes": peak_resident_memory_bytes,
            "terminal_reason": terminal_reason,
        }
        if details:
            event.update(details)
        self.sink(event)

    def terminal(self, termination: ConstructionTermination) -> None:
        self.emit(
            phase=termination.phase,
            status=termination.status.value,
            force=True,
            completed_units=termination.completed_units,
            total_units=termination.total_units,
            current_unit=termination.next_resumable_position,
            predicted_remaining_seconds=termination.predicted_next_seconds,
            checkpoint_location=termination.checkpoint_location,
            terminal_reason=termination.reason,
        )


def termination_payload(termination: ConstructionTermination) -> dict[str, object]:
    """Return a JSON-ready structured terminal record."""
    payload = asdict(termination)
    payload["status"] = termination.status.value
    payload["phase"] = termination.phase.value
    return payload
