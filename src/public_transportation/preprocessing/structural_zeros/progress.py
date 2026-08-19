"""Dependency-free progress contracts and an optional tqdm presentation layer."""

from __future__ import annotations

import sys
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Callable, TextIO


@dataclass(frozen=True, slots=True)
class StructuralZeroProgress:
    """One immutable observation of structural-zero preprocessing progress."""

    phase: str
    completed: int
    total: int
    elapsed_seconds: float
    message: str | None = None
    phase_elapsed_seconds: float | None = None
    work_stack: tuple[dict[str, object], ...] = ()
    active_units: tuple[str, ...] = ()
    queued_units: int | None = None
    active_workers: int | None = None
    requested_workers: int | None = None
    completed_weight: float | None = None
    total_weight: float | None = None
    eta_lower_seconds: float | None = None
    eta_upper_seconds: float | None = None
    checkpoint_location: str | None = None
    checkpoint_reusable: bool | None = None
    reused_units: int | None = None
    rebuilt_units: int | None = None
    next_resumable_position: str | None = None
    job_elapsed_seconds: float | None = None
    predicted_job_remaining_seconds: float | None = None
    job_eta_confidence: str = "unavailable"
    job_eta_reason: str | None = None
    estimated_job_completion_at_utc: str | None = None

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("phase must be nonempty.")
        if self.completed < 0 or self.total < 0 or self.completed > self.total:
            raise ValueError("progress counts must satisfy 0 <= completed <= total.")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative.")
        if self.phase_elapsed_seconds is not None and self.phase_elapsed_seconds < 0:
            raise ValueError("phase_elapsed_seconds must be non-negative.")
        for name, value in (
            ("completed_weight", self.completed_weight),
            ("total_weight", self.total_weight),
        ):
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative.")
        if (
            self.completed_weight is not None
            and self.total_weight is not None
            and self.completed_weight > self.total_weight
        ):
            raise ValueError("completed_weight must not exceed total_weight.")

    @property
    def throughput_units_per_second(self) -> float | None:
        """Observed throughput, once at least one unit has completed."""
        if self.completed == 0 or self.elapsed_seconds <= 0.0:
            return None
        return self.completed / self.elapsed_seconds

    @property
    def estimated_remaining_seconds(self) -> float | None:
        """Linear ETA from completed work, or ``None`` before calibration."""
        throughput = self.throughput_units_per_second
        if throughput is None:
            return None
        estimate = (self.total - self.completed) / throughput
        return estimate if math.isfinite(estimate) else None

    @property
    def eta_confidence(self) -> str:
        """Conservative confidence label for the linear ETA."""
        if self.completed == self.total and self.total > 0:
            return "complete"
        if self.completed == 0 or self.total == 0:
            return "unavailable"
        fraction = self.completed / self.total
        if self.completed < 3 or fraction < 0.01:
            return "low"
        if self.completed < 10 or fraction < 0.1:
            return "medium"
        return "high"

    @property
    def weighted_fraction(self) -> float | None:
        if self.completed_weight is None or self.total_weight in (None, 0.0):
            return None
        return self.completed_weight / self.total_weight

    @property
    def throughput_weight_per_second(self) -> float | None:
        if self.completed_weight is None or self.elapsed_seconds <= 0.0:
            return None
        return self.completed_weight / self.elapsed_seconds


StructuralZeroProgressCallback = Callable[[StructuralZeroProgress], None]


class ProgressEmitter:
    """Low-overhead, deterministic count/time throttling for one loop phase."""

    def __init__(
        self,
        progress: StructuralZeroProgressCallback | None,
        *,
        phase: str,
        total: int,
        message: str | None = None,
        work_name: str | None = None,
        total_weight: float | None = None,
        progress_interval_seconds: float = 10.0,
    ) -> None:
        self.progress = progress
        self.phase = phase
        self.total = total
        self.message = message
        self.work_name = work_name or phase
        self.total_weight = total_weight
        if (
            not math.isfinite(float(progress_interval_seconds))
            or progress_interval_seconds <= 0.0
        ):
            raise ValueError("progress_interval_seconds must be positive and finite.")
        self.progress_interval_seconds = float(progress_interval_seconds)
        self.started = perf_counter()
        self.last_completed = -1
        self.last_emitted_at = self.started
        self.count_interval = 1 if total <= 100 else max(25, (total + 99) // 100)

    def start(self) -> None:
        self.emit(0)

    def update(self, completed: int) -> None:
        if self.progress is None:
            return
        now = perf_counter()
        if (
            completed == self.total
            or completed - self.last_completed >= self.count_interval
            or now - self.last_emitted_at >= self.progress_interval_seconds
        ):
            self._emit(completed, now)

    def emit(self, completed: int) -> None:
        if self.progress is not None:
            self._emit(completed, perf_counter())

    def _emit(self, completed: int, now: float) -> None:
        if completed == self.last_completed:
            return
        callback = self.progress
        if callback is None:
            return
        callback(
            StructuralZeroProgress(
                phase=self.phase,
                completed=completed,
                total=self.total,
                elapsed_seconds=now - self.started,
                job_elapsed_seconds=now - self.started,
                predicted_job_remaining_seconds=(
                    (self.total - completed)
                    / (completed / max(now - self.started, 1.0e-12))
                    if completed > 0 and completed < self.total
                    else (0.0 if completed == self.total else None)
                ),
                job_eta_confidence=(
                    "high"
                    if completed >= 10
                    else ("medium" if completed >= 3 else "unavailable")
                ),
                job_eta_reason=(
                    None
                    if completed > 0
                    else "no completed units are available"
                ),
                message=self.message,
                phase_elapsed_seconds=now - self.started,
                work_stack=(
                    {
                        "name": self.work_name,
                        "completed_units": completed,
                        "total_units": self.total,
                        "current_unit": f"{self.work_name}-{completed:06d}",
                        "completed_weight": (
                            None
                            if self.total_weight is None
                            else float(self.total_weight) * completed / max(self.total, 1)
                        ),
                        "total_weight": self.total_weight,
                        "status": "completed" if completed == self.total else "running",
                    },
                ),
                active_units=(
                    () if completed >= self.total else (f"{self.work_name}-{completed:06d}",)
                ),
                eta_lower_seconds=(
                    None
                    if completed == 0
                    else ((self.total - completed) / (completed / max(now - self.started, 1.0e-12)))
                ),
                eta_upper_seconds=(
                    None
                    if completed == 0
                    else ((self.total - completed) / (completed / max(now - self.started, 1.0e-12)))
                ),
                completed_weight=(
                    None
                    if self.total_weight is None
                    else float(self.total_weight) * completed / max(self.total, 1)
                ),
                total_weight=self.total_weight,
            )
        )
        self.last_completed = completed
        self.last_emitted_at = now


def emit_phase(
    progress: StructuralZeroProgressCallback | None,
    phase: str,
    *,
    completed: int,
    total: int = 1,
    started: float | None = None,
    message: str | None = None,
) -> None:
    """Emit a non-loop phase transition without requiring an emitter object."""
    if progress is None:
        return
    elapsed = 0.0 if started is None else perf_counter() - started
    progress(StructuralZeroProgress(phase, completed, total, elapsed, message))


class StructuralZeroTqdmProgress:
    """Context-managed callback that displays one tqdm progress bar at a time."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        file: TextIO | None = None,
    ) -> None:
        self.enabled = enabled
        self.file = sys.stderr if file is None else file
        self._bar = None
        self._phase: str | None = None

    def __enter__(self) -> StructuralZeroTqdmProgress:
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def __call__(self, event: StructuralZeroProgress) -> None:
        if not self.enabled:
            return
        if self._phase != event.phase:
            self.close()
            try:
                from tqdm import tqdm  # type: ignore[import-untyped]
            except ImportError as error:
                raise RuntimeError(
                    "tqdm progress was requested but tqdm is not installed."
                ) from error
            self._bar = tqdm(
                total=event.total,
                initial=event.completed,
                desc=event.phase,
                file=self.file,
                leave=True,
            )
            self._phase = event.phase
        else:
            assert self._bar is not None
            delta = event.completed - self._bar.n
            if delta > 0:
                self._bar.update(delta)
        if event.message and self._bar is not None:
            self._bar.set_postfix_str(event.message, refresh=False)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
        self._bar = None
        self._phase = None


def structural_zero_tqdm_progress(
    *, enabled: bool = True, file: TextIO | None = None
) -> StructuralZeroTqdmProgress:
    """Return the optional tqdm adapter as a context manager and callback."""
    return StructuralZeroTqdmProgress(enabled=enabled, file=file)
