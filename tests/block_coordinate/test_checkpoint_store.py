"""Crash-boundary tests for the durable checkpoint journal."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

import public_transportation.inference.block_coordinate.checkpoint_store as checkpoint_store

from public_transportation.inference.block_coordinate import (
    BlockConvergenceDiagnostics,
    BlockCoordinateFingerprints,
    BlockCoordinateState,
    BlockObjectiveComponents,
    BlockUpdateProposal,
    DiagnosticValue,
)
from public_transportation.inference.block_coordinate.checkpoint_store import (
    BlockCheckpointStore,
)


def _fingerprints(**overrides) -> BlockCoordinateFingerprints:
    values = {
        "scenario": "scenario",
        "assignment_inputs": "assignment",
        "od_layout": "layout",
        "fixed_demand": "fixed",
        "measurements": "measurements",
        "prior": "prior",
        "routing": "routing",
        "partition": "partition",
        "solver_semantics": "solver",
    }
    values.update(overrides)
    return BlockCoordinateFingerprints(**values)


def _diagnostics(improvement: float = 0.0) -> BlockConvergenceDiagnostics:
    unavailable = DiagnosticValue(None, "unavailable")
    return BlockConvergenceDiagnostics(
        latest_block_projected_gradient=unavailable,
        estimated_global_projected_gradient=unavailable,
        exact_global_projected_gradient=DiagnosticValue(2.0, "exact", 0),
        maximum_block_flow_change=improvement,
        initialization_objective_improvement=improvement,
        current_sweep_objective_improvement=improvement,
        previous_sweep_objective_improvement=None,
    )


def _state(identity: BlockCoordinateFingerprints) -> BlockCoordinateState:
    components = BlockObjectiveComponents(10.0, 1.0)
    return BlockCoordinateState(
        current_free_flow=[1.0, 2.0, 3.0],
        best_free_flow=[1.0, 2.0, 3.0],
        current_prediction=[4.0, 5.0],
        fixed_measurement_offset=[0.5, 0.25],
        current_objective=11.0,
        best_objective=11.0,
        current_components=components,
        best_components=components,
        sweep=0,
        schedule_position=0,
        accepted_updates=0,
        rejected_updates=0,
        elapsed_seconds=0.5,
        block_schedule=("first", "second"),
        random_state_json='{"state":1}',
        diagnostics=_diagnostics(),
        fingerprints=identity,
    )


def _accepted_state(initial: BlockCoordinateState) -> tuple[BlockUpdateProposal, BlockCoordinateState]:
    proposal = BlockUpdateProposal(
        block_id="first",
        block_fingerprint="block-fingerprint",
        free_column_indices=(0, 2),
        flow_before=[1.0, 3.0],
        flow_after=[1.5, 2.5],
        flow_delta=[0.5, -0.5],
        prediction_delta=[0.25, -0.1],
        trial_prediction=[4.25, 4.9],
    )
    components = BlockObjectiveComponents(8.5, 0.75)
    state = BlockCoordinateState(
        current_free_flow=[1.5, 2.0, 2.5],
        best_free_flow=[1.5, 2.0, 2.5],
        current_prediction=[4.25, 4.9],
        fixed_measurement_offset=initial.fixed_measurement_offset,
        current_objective=9.25,
        best_objective=9.25,
        current_components=components,
        best_components=components,
        sweep=0,
        schedule_position=1,
        accepted_updates=1,
        rejected_updates=0,
        elapsed_seconds=1.0,
        block_schedule=initial.block_schedule,
        random_state_json='{"state":2}',
        diagnostics=_diagnostics(1.75),
        fingerprints=initial.fingerprints,
    )
    return proposal, state


def _assert_state_equal(actual: BlockCoordinateState, expected: BlockCoordinateState) -> None:
    np.testing.assert_array_equal(actual.current_free_flow, expected.current_free_flow)
    np.testing.assert_array_equal(actual.best_free_flow, expected.best_free_flow)
    np.testing.assert_array_equal(actual.current_prediction, expected.current_prediction)
    assert actual.current_objective == expected.current_objective
    assert actual.best_objective == expected.best_objective
    assert actual.current_components == expected.current_components
    assert actual.best_components == expected.best_components
    assert actual.sweep == expected.sweep
    assert actual.schedule_position == expected.schedule_position
    assert actual.accepted_updates == expected.accepted_updates
    assert actual.rejected_updates == expected.rejected_updates
    assert actual.block_schedule == expected.block_schedule
    assert actual.random_state_json == expected.random_state_json
    assert actual.diagnostics == expected.diagnostics


def test_committed_journal_replays_exactly_once_and_compacts(tmp_path) -> None:
    identity = _fingerprints()
    initial = _state(identity)
    proposal, accepted = _accepted_state(initial)
    store = BlockCheckpointStore(tmp_path, identity)
    store.initialize(initial)
    store.append_accepted_update(
        proposal=proposal,
        objective_before=initial.current_objective,
        state_after=accepted,
        best_solution_updated=True,
    )

    recovered = BlockCheckpointStore(tmp_path, identity).load()
    _assert_state_equal(recovered, accepted)
    store.compact(accepted)
    assert not list(tmp_path.glob("journal-*"))
    compacted = BlockCheckpointStore(tmp_path, identity).load()
    _assert_state_equal(compacted, accepted)


def test_published_journal_without_commit_marker_is_ignored(tmp_path) -> None:
    identity = _fingerprints()
    initial = _state(identity)
    store = BlockCheckpointStore(tmp_path, identity)
    store.initialize(initial)
    (tmp_path / "journal-000000000001.json").write_text(
        json.dumps({"incomplete": True}), encoding="utf-8"
    )

    recovered = BlockCheckpointStore(tmp_path, identity).load()
    _assert_state_equal(recovered, initial)


def test_invalid_commit_marker_and_fingerprint_mismatch_are_rejected(tmp_path) -> None:
    identity = _fingerprints()
    initial = _state(identity)
    proposal, accepted = _accepted_state(initial)
    store = BlockCheckpointStore(tmp_path, identity)
    store.initialize(initial)
    store.append_accepted_update(
        proposal=proposal,
        objective_before=11.0,
        state_after=accepted,
        best_solution_updated=True,
    )
    marker = tmp_path / "journal-000000000001.commit"
    marker.write_text('{"sequence":1,"journal_sha256":"wrong"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="commit marker"):
        BlockCheckpointStore(tmp_path, identity).load()

    with pytest.raises(ValueError, match="fingerprints"):
        BlockCheckpointStore(tmp_path, _fingerprints(prior="changed")).load()


def test_failed_compact_publication_preserves_previous_checkpoint(tmp_path, monkeypatch) -> None:
    identity = _fingerprints()
    initial = _state(identity)
    store = BlockCheckpointStore(tmp_path, identity)
    store.initialize(initial)
    updated = replace(initial, elapsed_seconds=2.0)

    def fail_replace(_source, _target):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(
        "public_transportation.inference.block_coordinate.checkpoint_store.os.replace",
        fail_replace,
    )
    with pytest.raises(OSError, match="publication failure"):
        store.compact(updated)

    recovered = BlockCheckpointStore(tmp_path, identity).load()
    _assert_state_equal(recovered, initial)


def test_failure_before_journal_publication_preserves_previous_state(
    tmp_path, monkeypatch
) -> None:
    identity = _fingerprints()
    initial = _state(identity)
    proposal, accepted = _accepted_state(initial)
    store = BlockCheckpointStore(tmp_path, identity)
    store.initialize(initial)

    def fail_replace(_source, _target):
        raise OSError("synthetic journal publication failure")

    monkeypatch.setattr(
        "public_transportation.inference.block_coordinate.checkpoint_store.os.replace",
        fail_replace,
    )
    with pytest.raises(OSError, match="journal publication failure"):
        store.append_accepted_update(
            proposal=proposal,
            objective_before=11.0,
            state_after=accepted,
            best_solution_updated=True,
        )
    recovered = BlockCheckpointStore(tmp_path, identity).load()
    _assert_state_equal(recovered, initial)


def test_failure_after_journal_publication_before_commit_is_ignored(
    tmp_path, monkeypatch
) -> None:
    identity = _fingerprints()
    initial = _state(identity)
    proposal, accepted = _accepted_state(initial)
    store = BlockCheckpointStore(tmp_path, identity)
    store.initialize(initial)
    real_atomic_write = checkpoint_store._atomic_write

    def fail_commit(path, payload):
        if path.suffix == ".commit":
            raise OSError("synthetic commit failure")
        real_atomic_write(path, payload)

    monkeypatch.setattr(checkpoint_store, "_atomic_write", fail_commit)
    with pytest.raises(OSError, match="commit failure"):
        store.append_accepted_update(
            proposal=proposal,
            objective_before=11.0,
            state_after=accepted,
            best_solution_updated=True,
        )
    assert (tmp_path / "journal-000000000001.json").exists()
    assert not (tmp_path / "journal-000000000001.commit").exists()
    recovered = BlockCheckpointStore(tmp_path, identity).load()
    _assert_state_equal(recovered, initial)


def test_initialize_refuses_to_overwrite_existing_checkpoint(tmp_path) -> None:
    identity = _fingerprints()
    state = _state(identity)
    store = BlockCheckpointStore(tmp_path, identity)
    store.initialize(state)
    with pytest.raises(FileExistsError, match="resume"):
        BlockCheckpointStore(tmp_path, identity).initialize(state)
