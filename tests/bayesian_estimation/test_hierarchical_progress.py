from __future__ import annotations

import pytest

from public_transportation.inference.construction_control import (
    ConstructionDeadline,
    ConstructionPhase,
    ConstructionProgressReporter,
    ProgressWorkUnit,
    estimate_completed_unit_eta,
)


def test_nested_work_stack_and_job_phase_metadata_are_serialized():
    events: list[dict[str, object]] = []
    reporter = ConstructionProgressReporter(
        ConstructionDeadline.unlimited(), events.append, minimum_interval_seconds=0.0
    )
    reporter.set_phase_plan({"planning": 2.0, "routing_preparation": 4.0})
    with reporter.work_scope("planning", total_units=2) as outer:
        assert isinstance(outer, ProgressWorkUnit)
        reporter.push_work("worker-0", total_units=4, current_unit="group-000001")
        reporter.emit(
            phase=ConstructionPhase.PLANNING,
            status="running",
            completed_units=1,
            total_units=2,
            predicted_remaining_seconds=3.0,
            eta_confidence="medium",
            completed_weight=2.0,
            total_weight=8.0,
            active_units=("worker-0",),
            queued_units=2,
            active_workers=1,
            requested_workers=2,
            checkpoint_location="/tmp/checkpoint",
            checkpoint_reusable=True,
            next_resumable_position="group-000002",
        )
        reporter.pop_work()

    event = events[-1]
    assert event["phase_elapsed_seconds"] >= 0.0
    assert event["job_elapsed_seconds"] == event["elapsed_seconds"]
    assert event["work_stack"][0]["name"] == "planning"
    assert event["completed_weight"] == 2.0
    assert event["weighted_fraction"] == pytest.approx(0.25)
    assert event["predicted_job_remaining_seconds"] == pytest.approx(7.0)
    assert event["active_units"] == ["worker-0"]
    assert event["queued_units"] == 2
    assert event["checkpoint_reusable"] is True


def test_heartbeat_scope_reports_opaque_work_without_fabricating_eta():
    events: list[dict[str, object]] = []
    reporter = ConstructionProgressReporter(
        ConstructionDeadline.unlimited(), events.append, minimum_interval_seconds=0.0
    )
    with reporter.heartbeat_scope(
        current_unit="jax-compilation", phase=ConstructionPhase.PLANNING
    ):
        reporter.emit(
            phase=ConstructionPhase.PLANNING,
            status="running",
            current_unit="jax-compilation",
            eta_reason="operation does not expose completed-unit timing",
        )
    assert events[-1]["work_stack"][0]["name"] == "jax-compilation"
    assert events[-1]["predicted_remaining_seconds"] is None
    assert events[-1]["predicted_job_remaining_seconds"] is None


def test_eta_supports_weighted_work_and_uncertainty_interval():
    estimate = estimate_completed_unit_eta(
        [1.0, 1.1, 0.9, 1.2, 1.0],
        completed_units=4,
        total_units=10,
        completed_weight=30.0,
        total_weight=100.0,
        weight_durations=[2.0, 2.1, 1.9, 2.2, 2.0],
    )
    assert estimate.predicted_remaining_seconds is not None
    assert estimate.eta_lower_seconds is not None
    assert estimate.eta_upper_seconds is not None
    assert estimate.eta_lower_seconds <= estimate.predicted_remaining_seconds
    assert estimate.predicted_remaining_seconds <= estimate.eta_upper_seconds
    assert estimate.weighted_fraction == pytest.approx(0.3)
    assert estimate.throughput_weight_per_second is None


def test_eta_can_use_wall_clock_progress_for_parallel_units():
    estimate = estimate_completed_unit_eta(
        (),
        completed_units=20,
        total_units=100,
        parallelism=8,
        elapsed_seconds=40.0,
    )
    assert estimate.predicted_remaining_seconds == pytest.approx(160.0)
    assert estimate.eta_confidence == "low"
    assert estimate.throughput_units_per_second == pytest.approx(0.5)
