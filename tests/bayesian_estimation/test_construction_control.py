from __future__ import annotations

import threading
import time

import pytest

from public_transportation.inference.construction_control import (
    ConstructionDeadline,
    ConstructionPhase,
    ConstructionProgressReporter,
    _finite_positive_samples,
    deadline_stop,
    estimate_completed_unit_eta,
    normalize_progress_event,
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


def test_eta_is_unavailable_until_completed_observations_accumulate():
    early = estimate_completed_unit_eta([0.5, 0.6], completed_units=2, total_units=10)
    assert early.predicted_remaining_seconds is None
    assert early.eta_confidence == "unavailable"
    assert early.eta_reason is not None

    estimate = estimate_completed_unit_eta(
        [0.5, 0.6, 0.55, 0.52, 0.58, 0.54, 0.56, 0.53],
        completed_units=8,
        total_units=20,
        parallelism=2,
    )
    assert estimate.predicted_remaining_seconds is not None
    assert estimate.predicted_remaining_seconds > 0.0
    assert estimate.eta_confidence == "high"
    assert estimate.estimated_completion_at_utc is not None
    assert estimate.estimated_completion_at_utc.endswith("Z")


def test_eta_marks_heterogeneous_units_conservatively():
    estimate = estimate_completed_unit_eta(
        [0.1, 0.2, 0.5, 2.0, 0.15, 0.3],
        completed_units=6,
        total_units=12,
    )
    assert estimate.predicted_remaining_seconds is not None
    assert estimate.eta_confidence in {"low", "medium"}


def test_eta_duration_samples_keep_only_a_bounded_recent_tail():
    samples = _finite_positive_samples(
        [1.0] * 10_000 + [2.0] * 32 + [float("nan"), -1.0, 0.0]
    )
    assert len(samples) == 32
    assert samples == [2.0] * 32


def test_reporting_sink_failure_does_not_fail_the_scientific_run():
    def failing_sink(event):
        raise OSError("log filesystem unavailable")

    reporter = ConstructionProgressReporter(
        ConstructionDeadline.unlimited(), failing_sink, minimum_interval_seconds=0.0
    )
    reporter.emit(
        phase=ConstructionPhase.PLANNING,
        status="running",
        completed_units=1,
        total_units=2,
    )
    assert reporter.reporting_failures == 1
    assert reporter.last_reporting_error == "OSError: log filesystem unavailable"
    assert reporter.last_event is not None
    assert reporter.last_event["completed_units"] == 1


def test_nonblocking_progress_does_not_wait_for_sink_io():
    started = threading.Event()
    release = threading.Event()
    events = []

    def blocking_sink(event):
        started.set()
        release.wait(timeout=2.0)
        events.append(event)

    reporter = ConstructionProgressReporter(
        ConstructionDeadline.unlimited(), blocking_sink, minimum_interval_seconds=0.0
    )
    started_at = time.perf_counter()
    reporter.emit_nonblocking(
        phase=ConstructionPhase.SUPPORT_DISCOVERY,
        status="running",
        completed_units=1,
        total_units=2,
    )
    assert time.perf_counter() - started_at < 0.5
    assert started.wait(timeout=1.0)
    release.set()
    reporter.flush()
    assert events and events[0]["completed_units"] == 1


def test_nonblocking_progress_queue_is_bounded_and_drops_when_full():
    started = threading.Event()
    release = threading.Event()

    def blocking_sink(_event):
        started.set()
        release.wait(timeout=2.0)

    reporter = ConstructionProgressReporter(
        ConstructionDeadline.unlimited(), blocking_sink, minimum_interval_seconds=0.0
    )
    reporter.emit_nonblocking(
        phase=ConstructionPhase.SUPPORT_DISCOVERY,
        status="running",
        force=True,
        completed_units=0,
        total_units=2,
    )
    assert started.wait(timeout=1.0)
    pending = reporter._sink_queue
    assert pending is not None
    assert pending.maxsize > 0
    for completed in range(pending.maxsize + 16):
        reporter.emit_nonblocking(
            phase=ConstructionPhase.SUPPORT_DISCOVERY,
            status="running",
            force=True,
            completed_units=completed + 1,
            total_units=pending.maxsize + 16,
        )
    assert reporter.reporting_failures > 0
    release.set()
    reporter.flush()


def test_normalize_progress_event_preserves_legacy_fields_and_adds_hierarchy():
    payload = normalize_progress_event(
        {
            "phase": "legacy_shards",
            "completed_shards": 2,
            "total_shards": 5,
            "estimated_remaining_seconds": 3.5,
        }
    )

    assert payload["completed_shards"] == 2
    assert payload["total_shards"] == 5
    assert payload["completed_units"] == 2
    assert payload["total_units"] == 5
    assert payload["predicted_remaining_seconds"] == 3.5
    assert payload["status"] == "running"
    assert payload["schema_version"] == 1
    assert payload["work_stack"] == (
        {
            "name": "legacy_shards",
            "completed_units": 2,
            "total_units": 5,
            "current_unit": None,
            "status": "running",
        },
    )


def test_opaque_heartbeat_reports_subphase_without_inventing_eta():
    deadline = ConstructionDeadline.unlimited()
    events: list[dict[str, object]] = []
    reporter = ConstructionProgressReporter(
        deadline, events.append, minimum_interval_seconds=0.01
    )
    with reporter.heartbeat_scope(
        current_unit="routing_factory",
        details={"routing_phase": "routing_factory"},
        interval_seconds=0.01,
    ):
        import time

        time.sleep(0.03)
    assert events
    assert all(event["phase"] == "routing_preparation" for event in events)
    assert any(event["routing_phase"] == "routing_factory" for event in events)
    assert all(event["predicted_remaining_seconds"] is None for event in events)
    assert all(event["eta_confidence"] == "unavailable" for event in events)
