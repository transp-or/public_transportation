"""Tests for exact immutable block-coordinate prediction algebra."""

from __future__ import annotations

import numpy as np
import pytest

from public_transportation.inference.block_coordinate import (
    ColumnSelectedLinearOperator,
    IncrementalLinearState,
    ODBlock,
    apply_incremental_update,
    block_data_gradient,
    initialize_incremental_state,
    propose_incremental_update,
    validate_incremental_prediction,
)
from public_transportation.inference.linear_operator import DenseLinearOperator


def _block(block_id: str, columns: tuple[int, ...]) -> ODBlock:
    return ODBlock(
        block_id=block_id,
        free_column_indices=columns,
        active_od_indices=columns,
        destination_group_indices=(0,),
        time_bin_ids=("morning",),
    )


def test_repeated_incremental_updates_equal_complete_recomputation() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.uniform(0.0, 1.0, size=(13, 8))
    operator = DenseLinearOperator(matrix)
    offset = rng.uniform(0.0, 3.0, size=13)
    initial = rng.uniform(0.0, 5.0, size=8)
    state = initialize_incremental_state(operator, initial, offset)
    original_flow = state.free_flow.copy()
    original_prediction = state.prediction.copy()

    for iteration, columns in enumerate(((0, 3, 7), (1, 2), (4, 5, 6), (0, 3, 7))):
        block = _block(f"block-{iteration}", columns)
        block_operator = ColumnSelectedLinearOperator(operator, columns)
        trial = rng.uniform(0.0, 5.0, size=len(columns))
        proposal = propose_incremental_update(
            state,
            block,
            block_operator,
            trial,
            lower_bounds=np.zeros(len(columns)),
            upper_bounds=np.full(len(columns), 5.0),
        )
        next_state = apply_incremental_update(state, proposal)
        np.testing.assert_allclose(
            next_state.prediction,
            matrix @ next_state.free_flow + offset,
            rtol=1.0e-14,
            atol=1.0e-14,
        )
        assert next_state.fixed_measurement_offset is not state.fixed_measurement_offset
        np.testing.assert_array_equal(next_state.fixed_measurement_offset, offset)
        state = next_state

    np.testing.assert_array_equal(original_flow, initial)
    np.testing.assert_array_equal(original_prediction, matrix @ initial + offset)
    assert not state.free_flow.flags.writeable
    assert not state.prediction.flags.writeable
    assert validate_incremental_prediction(state, operator).within_tolerance


def test_block_gradient_is_complete_gradient_slice() -> None:
    rng = np.random.default_rng(123)
    matrix = rng.normal(size=(10, 7))
    columns = (1, 4, 6)
    prediction = rng.normal(size=10)
    observations = rng.normal(size=10)
    weights = rng.uniform(0.2, 2.0, size=10)
    operator = DenseLinearOperator(matrix)
    block_operator = ColumnSelectedLinearOperator(operator, columns)

    block_gradient = block_data_gradient(
        block_operator, prediction, observations, weights
    )
    complete_gradient = matrix.T @ (weights * (prediction - observations))
    np.testing.assert_allclose(block_gradient, complete_gradient[list(columns)])


def test_proposal_rejects_bounds_and_stale_application() -> None:
    matrix = np.arange(20, dtype=float).reshape(4, 5)
    operator = DenseLinearOperator(matrix)
    state = initialize_incremental_state(operator, np.ones(5), np.zeros(4))
    block = _block("selected", (1, 3))
    block_operator = ColumnSelectedLinearOperator(operator, block.free_column_indices)

    with pytest.raises(ValueError, match="lower_bounds"):
        propose_incremental_update(
            state, block, block_operator, [0.5, 2.0], lower_bounds=[1.0, 1.0]
        )
    with pytest.raises(ValueError, match="upper_bounds"):
        propose_incremental_update(
            state, block, block_operator, [0.5, 2.0], upper_bounds=[1.0, 1.0]
        )

    unbounded = propose_incremental_update(
        state,
        block,
        block_operator,
        [0.5, 2.0],
        lower_bounds=[-np.inf, 0.0],
        upper_bounds=[np.inf, np.inf],
    )
    np.testing.assert_array_equal(unbounded.flow_after, [0.5, 2.0])

    proposal = propose_incremental_update(state, block, block_operator, [2.0, 3.0])
    changed_elsewhere = IncrementalLinearState(
        free_flow=[1.0, 9.0, 1.0, 1.0, 1.0],
        prediction=state.prediction,
        fixed_measurement_offset=state.fixed_measurement_offset,
    )
    with pytest.raises(ValueError, match="stale"):
        apply_incremental_update(changed_elsewhere, proposal)


def test_prediction_validation_detects_drift_and_handles_empty_measurements() -> None:
    operator = DenseLinearOperator(np.eye(3))
    drifted = IncrementalLinearState(
        free_flow=[1.0, 2.0, 3.0],
        prediction=[1.0, 2.01, 3.0],
        fixed_measurement_offset=np.zeros(3),
    )
    validation = validate_incremental_prediction(
        drifted, operator, absolute_tolerance=1.0e-6, relative_tolerance=1.0e-6
    )
    assert not validation.within_tolerance
    assert validation.max_absolute_error == pytest.approx(0.01)
    assert validation.relative_l2_error > 0.0

    empty_operator = DenseLinearOperator(np.empty((0, 2)))
    empty_state = initialize_incremental_state(empty_operator, [1.0, 2.0], [])
    empty_validation = validate_incremental_prediction(empty_state, empty_operator)
    assert empty_validation.within_tolerance
    assert empty_validation.max_absolute_error == 0.0
    assert empty_validation.relative_l2_error == 0.0
