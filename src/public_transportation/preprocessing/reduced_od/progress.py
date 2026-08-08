"""Shared progress and ETA reporting for reduced-OD preprocessing."""

from __future__ import annotations

import math
from time import perf_counter
from typing import Callable, Mapping

ReducedODProgress = Callable[[dict[str, object]], None]


class ReducedODProgressEmitter:
    """Emit JSON-compatible progress at bounded count and time intervals."""

    def __init__(
        self,
        progress: ReducedODProgress | None,
        *,
        phase: str,
        total: int,
        count_interval: int | None = None,
        time_interval_seconds: float = 10.0,
    ) -> None:
        if total < 0:
            raise ValueError("total must be non-negative.")
        if time_interval_seconds <= 0.0:
            raise ValueError("time_interval_seconds must be positive.")
        self.progress = progress
        self.phase = phase
        self.total = total
        self.count_interval = (
            max(1, (total + 99) // 100)
            if count_interval is None
            else max(1, count_interval)
        )
        self.time_interval_seconds = time_interval_seconds
        self.started = perf_counter()
        self.last_completed = -1
        self.last_emitted_at = self.started

    def start(self, *, details: Mapping[str, object] | None = None) -> None:
        self._emit(0, self.started, details=details)

    def update(
        self,
        completed: int,
        *,
        current_unit: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        if self.progress is None:
            return
        now = perf_counter()
        if (
            completed == self.total
            or completed - self.last_completed >= self.count_interval
            or now - self.last_emitted_at >= self.time_interval_seconds
        ):
            self._emit(
                completed,
                now,
                current_unit=current_unit,
                details=details,
            )

    def _emit(
        self,
        completed: int,
        now: float,
        *,
        current_unit: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        if self.progress is None or completed == self.last_completed:
            return
        if completed < 0 or completed > self.total:
            raise ValueError("completed must satisfy 0 <= completed <= total.")
        elapsed = max(0.0, now - self.started)
        throughput = (
            None if completed == 0 or elapsed == 0.0 else completed / elapsed
        )
        remaining = (
            None
            if throughput is None
            else (self.total - completed) / throughput
        )
        if remaining is not None and not math.isfinite(remaining):
            remaining = None
        event: dict[str, object] = {
            "phase": self.phase,
            "status": "completed" if completed == self.total else (
                "started" if completed == 0 else "in_progress"
            ),
            "completed_units": completed,
            "total_units": self.total,
            "elapsed_seconds": elapsed,
            "throughput_units_per_second": throughput,
            "estimated_remaining_seconds": remaining,
            "eta_confidence": self._confidence(completed),
            "current_unit": current_unit,
        }
        if details is not None:
            event.update(details)
        self.progress(event)
        self.last_completed = completed
        self.last_emitted_at = now

    def _confidence(self, completed: int) -> str:
        if completed == self.total and self.total > 0:
            return "complete"
        if completed == 0 or self.total == 0:
            return "unavailable"
        fraction = completed / self.total
        if completed < 3 or fraction < 0.01:
            return "low"
        if completed < 10 or fraction < 0.1:
            return "medium"
        return "high"
