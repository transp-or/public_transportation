from __future__ import annotations

import pytest

from public_transportation.inference.construction_control import (
    ConstructionDeadline,
    ConstructionPhase,
    ConstructionProgressReporter,
    deadline_stop,
)


class FakeClock:
    def __init__(self, value: float = 10.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_deadline_from_budget_and_absolute_time():
    clock = FakeClock()
    budget = ConstructionDeadline.from_budget(
        12.0, safety_margin_seconds=2.0, clock=clock
    )
    assert budget.absolute_deadline == 22.0
    assert budget.remaining_seconds == 12.0
    assert budget.may_start(10.0)
    assert not budget.may_start(10.01)
    clock.value = 22.5
    assert budget.expired
    assert budget.elapsed_seconds == 12.5
    assert budget.remaining_seconds == 0.0

    absolute = ConstructionDeadline.from_absolute(30.0, clock=clock)
    assert absolute.started_at == 22.5
    assert absolute.remaining_seconds == 7.5


def test_unlimited_deadline_accepts_any_finite_prediction():
    clock = FakeClock()
    deadline = ConstructionDeadline.unlimited(clock=clock)
    assert deadline.remaining_seconds is None
    assert not deadline.expired
    assert deadline.may_start(1.0e12)
    with pytest.raises(ValueError, match="predicted_seconds"):
        deadline.may_start(-1.0)


def test_structured_stop_records_overshoot_and_resume_location():
    clock = FakeClock()
    deadline = ConstructionDeadline.from_budget(5.0, clock=clock)
    clock.value = 16.25
    stopped = deadline_stop(
        deadline,
        phase=ConstructionPhase.SHARD_CONSTRUCTION,
        reason="next shard cannot finish safely",
        completed_units=3,
        total_units=8,
        next_resumable_position="storage-000003",
        checkpoint_location="/cache/checkpoint",
        checkpoint_reusable=True,
        predicted_next_seconds=2.0,
    ).termination
    assert stopped.phase is ConstructionPhase.SHARD_CONSTRUCTION
    assert stopped.deadline_overshoot_seconds == pytest.approx(1.25)
    assert stopped.checkpoint_reusable
    assert stopped.next_resumable_position == "storage-000003"


def test_progress_schema_throttles_and_terminal_event_is_forced():
    clock = FakeClock()
    deadline = ConstructionDeadline.unlimited(clock=clock)
    events = []
    reporter = ConstructionProgressReporter(
        deadline, events.append, minimum_interval_seconds=2.0, clock=clock
    )
    reporter.emit(
        phase=ConstructionPhase.PLANNING,
        status="started",
        completed_units=0,
        total_units=4,
    )
    clock.value += 1.0
    reporter.emit(
        phase=ConstructionPhase.PLANNING,
        status="running",
        completed_units=1,
        total_units=4,
    )
    termination = deadline_stop(
        deadline,
        phase=ConstructionPhase.PLANNING,
        reason="bounded stop",
    ).termination
    reporter.terminal(termination)
    assert len(events) == 2
    assert events[0]["schema_version"] == 1
    assert events[0]["completed_units"] == 0
    assert events[1]["status"] == "deadline_stopped"
