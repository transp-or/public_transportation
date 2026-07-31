"""Tests for generic block-restricted linear operators."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from public_transportation.inference.block_coordinate import (
    BlockLinearOperatorProtocol,
    ColumnSelectedLinearOperator,
    DenseBlockLinearOperator,
    SparseBlockLinearOperator,
)
from public_transportation.inference.linear_operator import DenseLinearOperator


@pytest.mark.parametrize(
    "factory",
    [DenseBlockLinearOperator, lambda matrix: SparseBlockLinearOperator(sparse.csr_array(matrix))],
)
def test_dense_and_sparse_block_operators_match_matrix_algebra(factory) -> None:
    rng = np.random.default_rng(20260730)
    matrix = rng.normal(size=(11, 4))
    matrix[[1, 8], :] = 0.0
    operator = factory(matrix)
    local = rng.normal(size=4)
    measurement = rng.normal(size=11)

    assert isinstance(operator, BlockLinearOperatorProtocol)
    assert operator.shape == (11, 4)
    assert operator.num_measurements == 11
    assert operator.num_local_variables == 4
    assert operator.measurement_support_indices == (0, 2, 3, 4, 5, 6, 7, 9, 10)
    np.testing.assert_allclose(operator.matvec(local), matrix @ local)
    np.testing.assert_allclose(operator.rmatvec(measurement), matrix.T @ measurement)
    np.testing.assert_allclose(
        measurement @ operator.matvec(local), local @ operator.rmatvec(measurement)
    )


def test_column_selected_operator_matches_explicit_matrix_slice() -> None:
    rng = np.random.default_rng(91)
    matrix = rng.normal(size=(7, 9))
    columns = (0, 3, 8)
    complete = DenseLinearOperator(matrix)
    selected = ColumnSelectedLinearOperator(
        complete, columns, measurement_support_indices=(0, 2, 6)
    )
    local = rng.normal(size=len(columns))
    measurement = rng.normal(size=7)

    assert selected.measurement_support_indices == (0, 2, 6)
    np.testing.assert_allclose(selected.matvec(local), matrix[:, columns] @ local)
    np.testing.assert_allclose(
        selected.rmatvec(measurement), matrix[:, columns].T @ measurement
    )


@pytest.mark.parametrize("factory", [DenseBlockLinearOperator, SparseBlockLinearOperator])
def test_block_operators_support_zero_sized_dimensions(factory) -> None:
    no_variables = factory(np.empty((5, 0)))
    no_measurements = factory(np.empty((0, 3)))

    np.testing.assert_array_equal(no_variables.matvec(np.empty(0)), np.zeros(5))
    np.testing.assert_array_equal(no_variables.rmatvec(np.zeros(5)), np.empty(0))
    np.testing.assert_array_equal(no_measurements.matvec(np.zeros(3)), np.empty(0))
    np.testing.assert_array_equal(no_measurements.rmatvec(np.empty(0)), np.zeros(3))
    assert no_variables.measurement_support_indices == ()
    assert no_measurements.measurement_support_indices == ()


def test_block_operator_owns_immutable_storage_and_validates_inputs() -> None:
    source = np.eye(3)
    dense = DenseBlockLinearOperator(source)
    source[0, 0] = 7.0
    assert dense.matrix[0, 0] == 1.0
    assert not dense.matrix.flags.writeable

    with pytest.raises(ValueError, match="shape"):
        dense.matvec(np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        dense.rmatvec(np.array([0.0, np.nan, 0.0]))
    with pytest.raises(ValueError, match="unique and ascending"):
        ColumnSelectedLinearOperator(DenseLinearOperator(source), (2, 1))

