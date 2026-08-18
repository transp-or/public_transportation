"""Shared deadline, progress, and clean-stop contracts for construction."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import math
from statistics import median
import threading
from time import monotonic
from typing import Callable, Iterable, Iterator


Clock = Callable[[], float]
ConstructionProgressSink = Callable[[dict[str, object]], None]
CONSTRUCTION_EVENT_SCHEMA_VERSION = 1


class ConstructionPhase(str, Enum):
    MEASUREMENT_SUPPORT_PREFLIGHT = "measurement_support_preflight"
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


@dataclass(frozen=True, slots=True)
class ConstructionETA:
    """A conservative ETA derived only from completed units."""

    predicted_remaining_seconds: float | None
    eta_confidence: str
    estimated_completion_at_utc: str | None
    eta_reason: str | None


def estimate_completed_unit_eta(
    durations: Iterable[float],
    *,
    completed_units: int,
    total_units: int | None,
    parallelism: int = 1,
    minimum_observations: int = 3,
) -> ConstructionETA:
    """Estimate remaining work without using in-flight or merely buffered units.

    The caller supplies durations for units that are already persisted or
    reusable.  A robust median of the most recent observations is used so one
    unusually slow shard cannot dominate the estimate.  Early estimates are
    deliberately marked unavailable/low confidence rather than presenting a
    precise-looking number based on too little evidence.
    """

    if total_units is None:
        return ConstructionETA(None, "unavailable", None, "total units unknown")
    completed = max(0, int(completed_units))
    total = max(0, int(total_units))
    remaining = max(0, total - completed)
    if remaining == 0:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return ConstructionETA(0.0, "high", now, "all units completed")
    samples = [
        float(value)
        for value in durations
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    if len(samples) < max(1, int(minimum_observations)):
        return ConstructionETA(
            None,
            "unavailable",
            None,
            f"fewer than {max(1, int(minimum_observations))} completed observations",
        )
    recent = samples[-9:]
    typical = float(median(recent))
    deviations = [abs(value - typical) for value in recent]
    mad = float(median(deviations))
    relative_mad = mad / max(abs(typical), np_finfo_eps())
    observations = len(recent)
    if observations >= 8 and relative_mad <= 0.20:
        confidence = "high"
    elif observations >= 5 and relative_mad <= 0.75:
        confidence = "medium"
    else:
        confidence = "low"
    workers = max(1, int(parallelism))
    prediction = max(0.0, remaining * typical / workers)
    completion = (
        (datetime.now(timezone.utc) + timedelta(seconds=prediction))
        .isoformat()
        .replace("+00:00", "Z")
    )
    reason = (
        "robust median of completed-unit durations"
        if confidence in {"high", "medium"}
        else "heterogeneous or still sparse completed-unit durations"
    )
    return ConstructionETA(prediction, confidence, completion, reason)


def np_finfo_eps() -> float:
    """Small local epsilon without importing NumPy into this control module."""

    return 1.0e-12


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
    _emit_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

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
        eta_confidence: str = "unavailable",
        estimated_completion_at_utc: str | None = None,
        eta_reason: str | None = None,
        checkpoint_location: str | None = None,
        cache_hits: int | None = None,
        cache_misses: int | None = None,
        peak_resident_memory_bytes: int | None = None,
        terminal_reason: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        if self.sink is None:
            return
        with self._emit_lock:
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
                "eta_confidence": eta_confidence,
                "estimated_completion_at_utc": estimated_completion_at_utc,
                "eta_reason": eta_reason,
                "checkpoint_location": checkpoint_location,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "peak_resident_memory_bytes": peak_resident_memory_bytes,
                "terminal_reason": terminal_reason,
            }
            if details:
                event.update(details)
            self.sink(event)

    @contextmanager
    def heartbeat_scope(
        self,
        *,
        current_unit: str,
        completed_units: int | None = None,
        total_units: int | None = None,
        details: dict[str, object] | None = None,
        interval_seconds: float | None = None,
    ) -> Iterator[None]:
        """Emit throttled observations while an indivisible operation runs.

        The operation deliberately reports no invented unit count or ETA.  It
        is useful for opaque factories and JAX tracing/compilation, where a
        callback cannot observe shard boundaries.
        """

        if self.sink is None:
            yield
            return
        interval = (
            self.minimum_interval_seconds
            if interval_seconds is None
            else max(0.01, float(interval_seconds))
        )
        stop = threading.Event()

        def emit_heartbeat() -> None:
            self.emit(
                phase=ConstructionPhase.ROUTING_PREPARATION,
                status="running",
                completed_units=completed_units,
                total_units=total_units,
                current_unit=current_unit,
                predicted_remaining_seconds=None,
                eta_confidence="unavailable",
                eta_reason="operation does not expose completed-unit timing",
                details=details,
            )

        def worker() -> None:
            while not stop.wait(interval):
                emit_heartbeat()

        thread = threading.Thread(
            target=worker, name="construction-progress-heartbeat", daemon=True
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=max(1.0, interval * 2.0))

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
