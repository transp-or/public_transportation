from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from public_transportation.inference.block_coordinate import (
    BlockConvergenceDiagnostics,
    BlockCoordinateFingerprints,
    BlockCoordinateMAPResult,
    BlockCoordinateState,
    BlockObjectiveComponents,
    BlockProgressEvent,
    DiagnosticValue,
    VALID_BLOCK_COORDINATE_STATUSES,
)


def identities() -> BlockCoordinateFingerprints:
    return BlockCoordinateFingerprints(
        scenario="s",
        assignment_inputs="a",
        od_layout="o",
        fixed_demand="f",
        measurements="m",
        prior="p",
        routing="r",
        partition="b",
        solver_semantics="v",
    )


def diagnostics() -> BlockConvergenceDiagnostics:
    return BlockConvergenceDiagnostics(
        latest_block_projected_gradient=DiagnosticValue(1.0, "exact", 0),
        estimated_global_projected_gradient=DiagnosticValue(2.0, "sampled", 0),
        exact_global_projected_gradient=DiagnosticValue(None, "unavailable"),
        maximum_block_flow_change=0.5,
        initialization_objective_improvement=3.0,
        current_sweep_objective_improvement=1.0,
        previous_sweep_objective_improvement=None,
    )


def state() -> BlockCoordinateState:
    return BlockCoordinateState(
        current_free_flow=[1.0, 2.0],
        best_free_flow=[1.0, 2.0],
        current_prediction=[4.0, 5.0, 6.0],
        fixed_measurement_offset=[0.0, 1.0, 0.0],
        current_objective=8.0,
        best_objective=8.0,
        current_components=BlockObjectiveComponents(data=6.0, prior=2.0),
        best_components=BlockObjectiveComponents(data=6.0, prior=2.0),
        sweep=0,
        schedule_position=1,
        accepted_updates=1,
        rejected_updates=0,
        elapsed_seconds=0.5,
        block_schedule=("a", "b"),
        random_state_json='{"seed":0}',
        diagnostics=diagnostics(),
        fingerprints=identities(),
    )


def test_state_owns_read_only_arrays_and_result_exposes_latest_and_best():
    instance = state()
    assert not instance.current_free_flow.flags.writeable
    assert not instance.current_prediction.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        instance.current_free_flow[0] = 9.0
    with pytest.raises(FrozenInstanceError):
        instance.sweep = 2

    result = BlockCoordinateMAPResult(
        status="stopped_by_update_budget",
        message="bounded run completed",
        state=instance,
        checkpoint_directory=Path("checkpoint"),
        resume_configuration_fingerprint="resume",
    )
    assert result.latest_free_flow is instance.current_free_flow
    assert result.best_free_flow is instance.best_free_flow


@pytest.mark.parametrize("status", VALID_BLOCK_COORDINATE_STATUSES)
def test_every_result_status_is_supported(status):
    result = BlockCoordinateMAPResult(
        status=status,
        message="status test",
        state=state(),
        checkpoint_directory="checkpoint",
        resume_configuration_fingerprint="resume",
    )
    assert result.status == status


def test_state_rejects_inconsistent_or_invalid_values():
    values = state()
    with pytest.raises(ValueError, match="equal shape"):
        BlockCoordinateState(
            **{
                name: getattr(values, name)
                for name in values.__dataclass_fields__
                if name != "best_free_flow"
            },
            best_free_flow=[1.0],
        )
    with pytest.raises(ValueError, match="components are inconsistent"):
        BlockCoordinateState(
            **{
                name: getattr(values, name)
                for name in values.__dataclass_fields__
                if name != "current_objective"
            },
            current_objective=9.0,
        )


def test_diagnostic_precision_semantics_are_explicit():
    assert DiagnosticValue(1.0, "exact", 2).kind == "exact"
    assert DiagnosticValue(1.0, "sampled", 2).kind == "sampled"
    assert DiagnosticValue(1.0, "stale", 1).kind == "stale"
    assert DiagnosticValue(None, "unavailable").kind == "unavailable"
    with pytest.raises(ValueError, match="cannot contain"):
        DiagnosticValue(1.0, "unavailable")
    with pytest.raises(ValueError, match="require a value"):
        DiagnosticValue(None, "exact")


def test_progress_event_contains_required_fields_and_round_trips_json():
    event = BlockProgressEvent(
        sweep=1,
        block_or_batch="block-a",
        blocks_completed_in_sweep=1,
        total_blocks=4,
        variables_visited=2,
        total_variables=8,
        elapsed_seconds=3.0,
        current_objective=10.0,
        best_objective=9.0,
        data_objective=8.0,
        prior_objective=2.0,
        latest_objective_improvement=1.0,
        latest_block_flow_change=0.5,
        latest_block_projected_gradient=0.25,
        estimated_global_projected_gradient=DiagnosticValue(0.5, "sampled", 1),
        exact_global_projected_gradient=DiagnosticValue(0.75, "stale", 0),
        last_exact_global_diagnostic_sweep=0,
        checkpoint_committed=True,
        estimated_remaining_sweep_seconds=9.0,
    )
    line = event.to_json_line()
    assert line.endswith("\n")
    assert '"kind":"sampled"' in line
    assert '"percentage_precision"' not in line
    payload = json.loads(line)
    assert payload["schema_version"] == 1
    assert payload["status"] == "running"
    assert payload["completed_units"] == 1
    assert payload["total_units"] == 4


def test_progress_rejects_invalid_coverage_and_objective_order():
    common = {
        "sweep": 0,
        "block_or_batch": "block",
        "blocks_completed_in_sweep": 1,
        "total_blocks": 1,
        "variables_visited": 1,
        "total_variables": 1,
        "elapsed_seconds": 0.0,
        "current_objective": 1.0,
        "best_objective": 1.0,
        "data_objective": 1.0,
        "prior_objective": 0.0,
        "latest_objective_improvement": 0.0,
        "latest_block_flow_change": 0.0,
        "latest_block_projected_gradient": 0.0,
        "estimated_global_projected_gradient": DiagnosticValue(None, "unavailable"),
        "exact_global_projected_gradient": DiagnosticValue(None, "unavailable"),
        "last_exact_global_diagnostic_sweep": None,
        "checkpoint_committed": False,
        "estimated_remaining_sweep_seconds": None,
    }
    with pytest.raises(ValueError, match="completed block"):
        BlockProgressEvent(**{**common, "blocks_completed_in_sweep": 2})
    with pytest.raises(ValueError, match="must not exceed"):
        BlockProgressEvent(**{**common, "best_objective": 2.0})
