"""Dependency-free progress contracts and an optional tqdm presentation layer."""

from __future__ import annotations

import sys
from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("phase must be nonempty.")
        if self.completed < 0 or self.total < 0 or self.completed > self.total:
            raise ValueError("progress counts must satisfy 0 <= completed <= total.")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative.")


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
    ) -> None:
        self.progress = progress
        self.phase = phase
        self.total = total
        self.message = message
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
            or now - self.last_emitted_at >= 10.0
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
                message=self.message,
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
