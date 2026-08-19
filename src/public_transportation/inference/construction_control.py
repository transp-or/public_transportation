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
from typing import Callable, Iterable, Iterator, Mapping, Sequence


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
    phase_elapsed_seconds: float | None = None
    predicted_job_remaining_seconds: float | None = None
    eta_confidence: str = "unavailable"
    estimated_completion_at_utc: str | None = None
    reused_units: int | None = None
    rebuilt_units: int | None = None


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
    eta_lower_seconds: float | None = None
    eta_upper_seconds: float | None = None
    throughput_units_per_second: float | None = None
    completed_weight: float | None = None
    total_weight: float | None = None
    weighted_fraction: float | None = None
    throughput_weight_per_second: float | None = None


@dataclass(frozen=True, slots=True)
class ProgressWorkUnit:
    """A serialisable unit in a hierarchical progress stack.

    Work units are descriptive only.  They never participate in the numerical
    identity of an artefact and may therefore be added to an existing run
    without invalidating checkpoints.
    """

    name: str
    completed_units: int | None = None
    total_units: int | None = None
    current_unit: str | None = None
    completed_weight: float | None = None
    total_weight: float | None = None
    status: str | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("work-unit name must be nonempty.")
        if self.completed_units is not None and self.completed_units < 0:
            raise ValueError("completed_units must be non-negative.")
        if self.total_units is not None and self.total_units < 0:
            raise ValueError("total_units must be non-negative.")
        if (
            self.completed_units is not None
            and self.total_units is not None
            and self.completed_units > self.total_units
        ):
            raise ValueError("completed_units must not exceed total_units.")
        for field_name, value in (
            ("completed_weight", self.completed_weight),
            ("total_weight", self.total_weight),
        ):
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise ValueError(f"{field_name} must be finite and non-negative.")
        if (
            self.completed_weight is not None
            and self.total_weight is not None
            and self.completed_weight > self.total_weight
        ):
            raise ValueError("completed_weight must not exceed total_weight.")

    def as_dict(self) -> dict[str, object]:
        """Return stable JSON-ready fields, omitting no compatibility keys."""

        return {
            "name": self.name,
            "completed_units": self.completed_units,
            "total_units": self.total_units,
            "current_unit": self.current_unit,
            "completed_weight": self.completed_weight,
            "total_weight": self.total_weight,
            "weighted_fraction": (
                None
                if self.completed_weight is None
                or self.total_weight in (None, 0.0)
                else self.completed_weight / self.total_weight
            ),
            "status": self.status,
        }


def _finite_positive_samples(values: Iterable[float]) -> list[float]:
    """Return finite positive samples without allowing an in-flight unit."""

    result: list[float] = []
    for value in values:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate) and candidate > 0.0:
            result.append(candidate)
    return result


def _eta_interval(samples: Sequence[float], typical: float) -> tuple[float, float]:
    """Return a conservative duration interval from completed observations."""

    ordered = sorted(samples)
    if len(ordered) < 3:
        return typical, typical
    deviations = [abs(value - typical) for value in ordered]
    mad = median(deviations)
    lower = max(0.0, typical - 2.5 * mad, ordered[0])
    upper = max(lower, typical + 2.5 * mad, ordered[-1])
    return float(lower), float(upper)


def estimate_completed_unit_eta(
    durations: Iterable[float],
    *,
    completed_units: int,
    total_units: int | None,
    parallelism: int = 1,
    minimum_observations: int = 3,
    completed_weight: float | None = None,
    total_weight: float | None = None,
    weight_durations: Iterable[float] | None = None,
    elapsed_seconds: float | None = None,
) -> ConstructionETA:
    """Estimate remaining work without using in-flight or merely buffered units.

    The caller supplies durations for units that are already persisted or
    reusable.  A robust median of the most recent observations is used so one
    unusually slow shard cannot dominate the estimate.  Early estimates are
    deliberately marked unavailable/low confidence rather than presenting a
    precise-looking number based on too little evidence.
    """

    if total_units is None and total_weight is None:
        return ConstructionETA(None, "unavailable", None, "total units unknown")
    completed = max(0, int(completed_units))
    total = None if total_units is None else max(0, int(total_units))
    remaining = None if total is None else max(0, total - completed)
    valid_completed_weight = (
        None
        if completed_weight is None
        else float(completed_weight)
    )
    valid_total_weight = None if total_weight is None else float(total_weight)
    weighted_remaining = None
    if valid_completed_weight is not None and valid_total_weight is not None:
        if not (
            math.isfinite(valid_completed_weight)
            and math.isfinite(valid_total_weight)
            and valid_completed_weight >= 0.0
            and valid_total_weight >= valid_completed_weight
        ):
            raise ValueError("progress weights must be finite and ordered.")
        weighted_remaining = max(0.0, valid_total_weight - valid_completed_weight)
    if (remaining == 0 if remaining is not None else False) or (
        weighted_remaining is not None and weighted_remaining == 0.0
    ):
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return ConstructionETA(
            0.0,
            "high",
            now,
            "all units completed",
            eta_lower_seconds=0.0,
            eta_upper_seconds=0.0,
            throughput_units_per_second=(
                completed / elapsed_seconds
                if elapsed_seconds is not None and elapsed_seconds > 0.0
                else None
            ),
            completed_weight=valid_completed_weight,
            total_weight=valid_total_weight,
            weighted_fraction=(
                None
                if valid_total_weight in (None, 0.0)
                else valid_completed_weight / valid_total_weight
            ),
        )
    samples = _finite_positive_samples(durations)
    weight_samples = (
        [] if weight_durations is None else _finite_positive_samples(weight_durations)
    )
    required_observations = max(1, int(minimum_observations))
    can_use_wall_clock = (
        elapsed_seconds is not None
        and elapsed_seconds > 0.0
        and completed > 0
        and total is not None
    )
    if (
        len(samples) < required_observations
        and len(weight_samples) < required_observations
        and not can_use_wall_clock
    ):
        return ConstructionETA(
            None,
            "unavailable",
            None,
            f"fewer than {required_observations} completed observations",
            completed_weight=valid_completed_weight,
            total_weight=valid_total_weight,
            weighted_fraction=(
                None
                if valid_completed_weight is None
                or valid_total_weight in (None, 0.0)
                else valid_completed_weight / valid_total_weight
            ),
        )
    recent = (samples if samples else weight_samples)[-9:]
    typical = float(median(recent)) if recent else 0.0
    lower_unit, upper_unit = _eta_interval(recent, typical)
    deviations = [abs(value - typical) for value in recent]
    mad = float(median(deviations)) if deviations else 0.0
    relative_mad = mad / max(abs(typical), np_finfo_eps())
    observations = len(recent)
    if observations >= 8 and relative_mad <= 0.20:
        confidence = "high"
    elif observations >= 5 and relative_mad <= 0.75:
        confidence = "medium"
    else:
        confidence = "low"
    workers = max(1, int(parallelism))
    if elapsed_seconds is not None and elapsed_seconds > 0.0 and completed > 0:
        unit_throughput = completed / float(elapsed_seconds)
        prediction = (
            None
            if total is None
            else max(0.0, remaining / max(unit_throughput, np_finfo_eps()))
        )
        if prediction is None:
            lower_prediction = upper_prediction = None
        elif samples:
            lower_prediction = max(
                0.0,
                remaining * lower_unit / max(workers, 1),
            )
            upper_prediction = max(
                lower_prediction,
                remaining * upper_unit / max(workers, 1),
            )
        else:
            # A wall-clock rate with no completed-unit samples is useful for
            # monitoring, but deliberately carries a broad low-confidence
            # interval rather than presenting false precision.
            lower_prediction = prediction * 0.5
            upper_prediction = prediction * 2.0
    elif weighted_remaining is not None and weight_samples:
        if weight_samples:
            typical_weight = float(median(weight_samples[-9:]))
            lower_weight, upper_weight = _eta_interval(
                weight_samples[-9:], typical_weight
            )
            prediction = weighted_remaining / max(typical_weight * workers, np_finfo_eps())
            lower_prediction = weighted_remaining / max(upper_weight * workers, np_finfo_eps())
            upper_prediction = weighted_remaining / max(lower_weight * workers, np_finfo_eps())
            unit_throughput = (
                valid_completed_weight / elapsed_seconds
                if elapsed_seconds is not None
                and elapsed_seconds > 0.0
                and valid_completed_weight is not None
                else None
            )
        else:
            prediction = None
            lower_prediction = upper_prediction = None
            unit_throughput = None
    else:
        prediction = (
            None if total is None else max(0.0, remaining * typical / workers)
        )
        lower_prediction = (
            None if total is None else max(0.0, remaining * lower_unit / workers)
        )
        upper_prediction = (
            None if total is None else max(0.0, remaining * upper_unit / workers)
        )
        unit_throughput = 1.0 / max(typical / workers, np_finfo_eps())
    if prediction is None:
        return ConstructionETA(
            None,
            "unavailable",
            None,
            "total work weight or unit count is unavailable",
            completed_weight=valid_completed_weight,
            total_weight=valid_total_weight,
            weighted_fraction=(
                None
                if valid_completed_weight is None
                or valid_total_weight in (None, 0.0)
                else valid_completed_weight / valid_total_weight
            ),
        )
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
    weighted_fraction = (
        None
        if valid_completed_weight is None
        or valid_total_weight in (None, 0.0)
        else valid_completed_weight / valid_total_weight
    )
    return ConstructionETA(
        prediction,
        confidence,
        completion,
        reason,
        eta_lower_seconds=lower_prediction,
        eta_upper_seconds=upper_prediction,
        throughput_units_per_second=unit_throughput,
        completed_weight=valid_completed_weight,
        total_weight=valid_total_weight,
        weighted_fraction=weighted_fraction,
        throughput_weight_per_second=(
            None
            if valid_completed_weight is None
            or elapsed_seconds is None
            or elapsed_seconds <= 0.0
            else valid_completed_weight / elapsed_seconds
        ),
    )


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
    _job_started_at: float | None = field(default=None, init=False, repr=False)
    _phase_started_at: float | None = field(default=None, init=False, repr=False)
    _current_phase: ConstructionPhase | None = field(default=None, init=False, repr=False)
    _phase_plan: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _work_stack: list[ProgressWorkUnit] = field(
        default_factory=list, init=False, repr=False
    )
    _last_event: dict[str, object] | None = field(
        default=None, init=False, repr=False
    )
    _emit_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.minimum_interval_seconds))
            or self.minimum_interval_seconds < 0.0
        ):
            raise ValueError("minimum_interval_seconds must be finite and non-negative.")

    def set_phase_plan(self, phase_plan: Mapping[str, float] | None) -> None:
        """Set optional expected durations for future phases.

        The plan is reporting metadata only.  It is deliberately not included
        in any computational fingerprint.  A missing phase has an unknown
        duration and consequently prevents a fabricated job-level ETA.
        """

        normalized: dict[str, float] = {}
        for name, value in (phase_plan or {}).items():
            seconds = float(value)
            if not math.isfinite(seconds) or seconds < 0.0:
                raise ValueError(
                    f"phase duration for {name!r} must be finite and non-negative."
                )
            normalized[str(name)] = seconds
        with self._emit_lock:
            self._phase_plan = normalized

    @property
    def work_stack(self) -> tuple[ProgressWorkUnit, ...]:
        """Return a snapshot of the currently active nested work units."""

        with self._emit_lock:
            return tuple(self._work_stack)

    @property
    def last_event(self) -> dict[str, object] | None:
        """Return a copy of the last emitted event for manifest writers."""

        with self._emit_lock:
            return None if self._last_event is None else dict(self._last_event)

    def summary(self) -> dict[str, object]:
        """Return the latest progress fields suitable for a stage manifest."""

        event = self.last_event
        if event is None:
            return {
                "schema_version": CONSTRUCTION_EVENT_SCHEMA_VERSION,
                "status": "not_started",
                "elapsed_seconds": 0.0,
                "phase": None,
            }
        return event

    def push_work(
        self,
        name: str,
        *,
        completed_units: int | None = None,
        total_units: int | None = None,
        current_unit: str | None = None,
        completed_weight: float | None = None,
        total_weight: float | None = None,
        status: str | None = "running",
    ) -> ProgressWorkUnit:
        """Push one nested work unit onto the progress stack."""

        unit = ProgressWorkUnit(
            name=name,
            completed_units=completed_units,
            total_units=total_units,
            current_unit=current_unit,
            completed_weight=completed_weight,
            total_weight=total_weight,
            status=status,
        )
        with self._emit_lock:
            self._work_stack.append(unit)
        return unit

    def update_work(self, **updates: object) -> ProgressWorkUnit:
        """Update the innermost work unit and return its new snapshot."""

        with self._emit_lock:
            if not self._work_stack:
                raise RuntimeError("cannot update work: the progress stack is empty.")
            current = self._work_stack[-1]
            allowed = {
                "name",
                "completed_units",
                "total_units",
                "current_unit",
                "completed_weight",
                "total_weight",
                "status",
            }
            unknown = set(updates).difference(allowed)
            if unknown:
                raise TypeError(
                    "unknown work-unit fields: " + ", ".join(sorted(unknown))
                )
            values = {
                "name": current.name,
                "completed_units": current.completed_units,
                "total_units": current.total_units,
                "current_unit": current.current_unit,
                "completed_weight": current.completed_weight,
                "total_weight": current.total_weight,
                "status": current.status,
            }
            values.update(updates)
            replacement = ProgressWorkUnit(**values)
            self._work_stack[-1] = replacement
            return replacement

    def pop_work(self) -> ProgressWorkUnit:
        """Pop and return the innermost work unit."""

        with self._emit_lock:
            if not self._work_stack:
                raise RuntimeError("cannot pop work: the progress stack is empty.")
            return self._work_stack.pop()

    @contextmanager
    def work_scope(
        self,
        name: str,
        *,
        completed_units: int | None = None,
        total_units: int | None = None,
        current_unit: str | None = None,
        completed_weight: float | None = None,
        total_weight: float | None = None,
        status: str | None = "running",
    ) -> Iterator[ProgressWorkUnit]:
        """Temporarily expose a nested unit while an operation is running."""

        unit = self.push_work(
            name,
            completed_units=completed_units,
            total_units=total_units,
            current_unit=current_unit,
            completed_weight=completed_weight,
            total_weight=total_weight,
            status=status,
        )
        try:
            yield unit
        finally:
            self.pop_work()

    def _job_eta(
        self,
        *,
        phase: ConstructionPhase,
        phase_remaining_seconds: float | None,
        phase_eta_confidence: str,
        now: float,
    ) -> tuple[float | None, str | None, str | None]:
        """Compute a job ETA without guessing durations of unknown phases."""

        if phase_remaining_seconds is None:
            return None, None, "current phase ETA is unavailable"
        phase_name = phase.value
        if not self._phase_plan:
            # For one-phase operations the phase is the complete observable job.
            return phase_remaining_seconds, phase_eta_confidence, None
        if phase_name not in self._phase_plan:
            return None, None, "phase is absent from the declared phase plan"
        names = list(self._phase_plan)
        index = names.index(phase_name)
        future = sum(self._phase_plan[name] for name in names[index + 1 :])
        return (
            phase_remaining_seconds + future,
            phase_eta_confidence,
            "current phase estimate plus declared future phase durations",
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
        work_stack: Sequence[Mapping[str, object]] | None = None,
        inner_work: Mapping[str, object] | None = None,
        active_units: Sequence[str] | None = None,
        queued_units: int | None = None,
        queued_unit_ids: Sequence[str] | None = None,
        active_workers: int | None = None,
        requested_workers: int | None = None,
        current_unit_elapsed_seconds: float | None = None,
        current_unit_predicted_remaining_seconds: float | None = None,
        completed_weight: float | None = None,
        total_weight: float | None = None,
        throughput_units_per_second: float | None = None,
        throughput_weight_per_second: float | None = None,
        eta_lower_seconds: float | None = None,
        eta_upper_seconds: float | None = None,
        reused_units: int | None = None,
        rebuilt_units: int | None = None,
        next_resumable_position: str | None = None,
        checkpoint_reusable: bool | None = None,
        deadline_margin_seconds: float | None = None,
        will_finish_before_deadline: bool | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        if self.sink is None:
            return
        with self._emit_lock:
            now = self.clock()
            for name, value in (
                ("completed_weight", completed_weight),
                ("total_weight", total_weight),
            ):
                if value is not None and (
                    not math.isfinite(float(value)) or float(value) < 0.0
                ):
                    raise ValueError(f"{name} must be finite and non-negative.")
            if (
                completed_weight is not None
                and total_weight is not None
                and completed_weight > total_weight
            ):
                raise ValueError("completed_weight must not exceed total_weight.")
            if self._job_started_at is None:
                self._job_started_at = now
            if self._current_phase is not phase:
                self._current_phase = phase
                self._phase_started_at = now
            if (
                not force
                and self._last_emitted_at is not None
                and now - self._last_emitted_at < self.minimum_interval_seconds
            ):
                return
            self._last_emitted_at = now
            elapsed_seconds = self.deadline.elapsed_seconds
            if work_stack is not None:
                effective_stack = list(work_stack)
            elif self._work_stack:
                effective_stack = [unit.as_dict() for unit in self._work_stack]
            else:
                # Every event has at least one named unit, while retaining the
                # legacy flat completed/total fields for old consumers.
                effective_stack = [
                    {
                        "name": phase.value,
                        "completed_units": completed_units,
                        "total_units": total_units,
                        "current_unit": current_unit,
                        "status": status,
                    }
                ]
            job_remaining, job_confidence, job_reason = self._job_eta(
                phase=phase,
                phase_remaining_seconds=predicted_remaining_seconds,
                phase_eta_confidence=eta_confidence,
                now=now,
            )
            eta_completion = None
            if job_remaining is not None and math.isfinite(job_remaining):
                eta_completion = (
                    datetime.now(timezone.utc) + timedelta(seconds=max(0.0, job_remaining))
                ).isoformat().replace("+00:00", "Z")
            deadline_remaining = self.deadline.remaining_seconds
            if deadline_remaining is None or job_remaining is None:
                margin = deadline_margin_seconds
                finish_before = will_finish_before_deadline
            else:
                margin = deadline_remaining - max(0.0, job_remaining)
                finish_before = margin >= self.deadline.safety_margin_seconds
            weighted_fraction = (
                None
                if completed_weight is None or total_weight in (None, 0.0)
                else float(completed_weight) / float(total_weight)
            )
            event = {
                "schema_version": CONSTRUCTION_EVENT_SCHEMA_VERSION,
                "phase": phase.value,
                "status": status,
                "elapsed_seconds": elapsed_seconds,
                "job_elapsed_seconds": elapsed_seconds,
                "phase_elapsed_seconds": max(
                    0.0,
                    now - (self._phase_started_at if self._phase_started_at is not None else now),
                ),
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
                "work_stack": effective_stack,
                "inner_work": None if inner_work is None else dict(inner_work),
                "active_units": None if active_units is None else list(active_units),
                "queued_units": queued_units,
                "queued_unit_ids": (
                    None if queued_unit_ids is None else list(queued_unit_ids)
                ),
                "active_workers": active_workers,
                "requested_workers": requested_workers,
                "current_unit_elapsed_seconds": current_unit_elapsed_seconds,
                "current_unit_predicted_remaining_seconds": (
                    current_unit_predicted_remaining_seconds
                ),
                "completed_weight": completed_weight,
                "total_weight": total_weight,
                "weighted_fraction": weighted_fraction,
                "throughput_units_per_second": throughput_units_per_second,
                "throughput_weight_per_second": throughput_weight_per_second,
                "eta_lower_seconds": eta_lower_seconds,
                "eta_upper_seconds": eta_upper_seconds,
                "predicted_job_remaining_seconds": job_remaining,
                "job_eta_confidence": job_confidence,
                "job_eta_reason": job_reason,
                "estimated_job_completion_at_utc": eta_completion,
                "reused_units": reused_units,
                "rebuilt_units": rebuilt_units,
                "next_resumable_position": next_resumable_position,
                "checkpoint_reusable": checkpoint_reusable,
                "deadline_margin_seconds": margin,
                "will_finish_before_deadline": finish_before,
            }
            if details:
                event.update(details)
            self._last_event = dict(event)
            if self.sink is not None:
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
        phase: ConstructionPhase = ConstructionPhase.ROUTING_PREPARATION,
    ) -> Iterator[None]:
        """Emit throttled observations while an indivisible operation runs.

        The operation deliberately reports no invented unit count or ETA.  It
        is useful for opaque factories and JAX tracing/compilation, where a
        callback cannot observe shard boundaries.
        """

        interval = (
            self.minimum_interval_seconds
            if interval_seconds is None
            else max(0.01, float(interval_seconds))
        )
        if self.sink is None:
            yield
            return
        stop = threading.Event()
        self.push_work(
            current_unit,
            completed_units=completed_units,
            total_units=total_units,
            current_unit=current_unit,
            status="running",
        )

        def emit_heartbeat() -> None:
            self.emit(
                phase=phase,
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
            self.pop_work()

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
            next_resumable_position=termination.next_resumable_position,
            checkpoint_reusable=termination.checkpoint_reusable,
            terminal_reason=termination.reason,
        )


def termination_payload(termination: ConstructionTermination) -> dict[str, object]:
    """Return a JSON-ready structured terminal record."""
    payload = asdict(termination)
    payload["status"] = termination.status.value
    payload["phase"] = termination.phase.value
    return payload
