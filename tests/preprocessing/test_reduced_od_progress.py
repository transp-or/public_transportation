from __future__ import annotations

import pytest

from public_transportation.preprocessing.reduced_od.progress import (
    ReducedODProgressEmitter,
)


def test_reduced_od_progress_reports_monotonic_counts_and_eta(monkeypatch) -> None:
    clock = iter((10.0, 12.0, 14.0))
    monkeypatch.setattr(
        "public_transportation.preprocessing.reduced_od.progress.perf_counter",
        lambda: next(clock),
    )
    events: list[dict[str, object]] = []
    emitter = ReducedODProgressEmitter(events.append, phase="work", total=2)
    emitter.start()
    emitter.update(1)
    emitter.update(2)

    assert [event["status"] for event in events] == [
        "started",
        "in_progress",
        "completed",
    ]
    assert events[1]["throughput_units_per_second"] == pytest.approx(0.5)
    assert events[1]["estimated_remaining_seconds"] == pytest.approx(2.0)
    assert events[-1]["estimated_remaining_seconds"] == pytest.approx(0.0)
    assert events[-1]["eta_confidence"] == "complete"


def test_reduced_od_progress_throttles_by_count(monkeypatch) -> None:
    clock = iter((0.0, 0.1, 0.2, 0.3))
    monkeypatch.setattr(
        "public_transportation.preprocessing.reduced_od.progress.perf_counter",
        lambda: next(clock),
    )
    events: list[dict[str, object]] = []
    emitter = ReducedODProgressEmitter(
        events.append, phase="work", total=100, count_interval=10
    )
    emitter.start()
    emitter.update(1)
    emitter.update(10)

    assert [event["completed_units"] for event in events] == [0, 10]
