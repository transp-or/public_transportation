from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from public_transportation.inference.linear_operator import (
    DenseLinearOperator,
    LinearOperatorProtocol,
    SparseLinearOperator,
    as_linear_operator,
    as_sparse_linear_operator,
)


def test_dense_operator_exposes_shape_dtype_and_defensive_storage():
    source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    operator = DenseLinearOperator(source)
    source[0, 0] = 100.0

    assert isinstance(operator, LinearOperatorProtocol)
    assert operator.shape == (2, 2)
    assert operator.dtype == np.dtype(np.float32)
    assert operator.matrix[0, 0] == 1.0
    assert not operator.matrix.flags.writeable


def test_dense_operator_forward_and_transpose_match_numpy():
    matrix = np.array([[1.0, 2.0, 0.0], [0.5, 0.0, 3.0]])
    operator = DenseLinearOperator(matrix)
    x = np.array([2.0, -1.0, 4.0])
    v = np.array([0.25, -2.0])

    np.testing.assert_allclose(operator.matvec(x), matrix @ x)
    np.testing.assert_allclose(operator.rmatvec(v), matrix.T @ v)


def test_dense_operator_satisfies_adjoint_identity():
    rng = np.random.default_rng(9182)
    operator = DenseLinearOperator(rng.normal(size=(7, 5)))
    x = rng.normal(size=5)
    v = rng.normal(size=7)

    left = np.vdot(v, operator.matvec(x))
    right = np.vdot(x, operator.rmatvec(v))
    assert left == pytest.approx(right, rel=1e-13, abs=1e-13)


@pytest.mark.parametrize(
    ("method", "vector", "message"),
    [
        ("matvec", np.ones(3), "forward vector must have shape \\(2,\\)"),
        ("matvec", np.ones((2, 1)), "forward vector must have shape \\(2,\\)"),
        ("rmatvec", np.ones(3), "transpose vector must have shape \\(2,\\)"),
        ("rmatvec", np.ones((2, 1)), "transpose vector must have shape \\(2,\\)"),
        ("matvec", [1.0, np.nan], "forward vector must be finite"),
        ("rmatvec", [1.0, np.inf], "transpose vector must be finite"),
    ],
)
def test_dense_operator_rejects_invalid_vectors(method, vector, message):
    operator = DenseLinearOperator(np.eye(2))
    with pytest.raises(ValueError, match=message):
        getattr(operator, method)(vector)


@pytest.mark.parametrize(
    ("matrix", "error", "message"),
    [
        (np.ones(3), ValueError, "two-dimensional"),
        ([[1.0, np.nan]], ValueError, "must be finite"),
        ([[1.0 + 2.0j]], TypeError, "real numeric"),
        ([["not", "numeric"]], TypeError, "real numeric"),
    ],
)
def test_dense_operator_rejects_invalid_matrix(matrix, error, message):
    with pytest.raises(error, match=message):
        DenseLinearOperator(matrix)


def test_as_linear_operator_wraps_dense_and_preserves_existing_operator():
    dense = as_linear_operator([[1, 0], [0, 1]])
    assert isinstance(dense, DenseLinearOperator)
    assert dense.dtype == np.dtype(np.float64)
    assert as_linear_operator(dense) is dense


def test_sparse_operator_canonicalizes_csr_storage():
    matrix = sparse.coo_array(
        (
            np.array([2.0, -1.0, 1.0, 0.0, 4.0]),
            (np.array([1, 0, 0, 1, 0]), np.array([0, 1, 1, 1, 0])),
        ),
        shape=(2, 3),
    )
    operator = SparseLinearOperator(matrix)

    assert isinstance(operator.matrix, sparse.csr_array)
    assert operator.matrix.has_canonical_format
    assert operator.nonzero_entries == 2
    np.testing.assert_array_equal(
        operator.matrix.toarray(), np.array([[4.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    )
    assert not operator.matrix.data.flags.writeable
    assert not operator.matrix.indices.flags.writeable
    assert not operator.matrix.indptr.flags.writeable


def test_sparse_operator_matches_dense_forward_transpose_and_adjoint():
    rng = np.random.default_rng(2718)
    matrix = rng.normal(size=(8, 5))
    matrix[np.abs(matrix) < 0.8] = 0.0
    dense = DenseLinearOperator(matrix)
    sparse_operator = SparseLinearOperator(matrix)
    x = rng.normal(size=5)
    v = rng.normal(size=8)

    np.testing.assert_allclose(sparse_operator.matvec(x), dense.matvec(x))
    np.testing.assert_allclose(sparse_operator.rmatvec(v), dense.rmatvec(v))
    assert np.vdot(v, sparse_operator.matvec(x)) == pytest.approx(
        np.vdot(x, sparse_operator.rmatvec(v)), rel=1e-13, abs=1e-13
    )


def test_sparse_operator_reports_storage_metrics():
    operator = SparseLinearOperator(np.eye(4, dtype=np.float32))
    assert operator.nonzero_entries == 4
    assert operator.total_entries == 16
    assert operator.density == pytest.approx(0.25)
    assert operator.value_storage_bytes == operator.matrix.data.nbytes
    assert operator.index_storage_bytes == (
        operator.matrix.indices.nbytes + operator.matrix.indptr.nbytes
    )
    assert operator.stored_bytes == (
        operator.value_storage_bytes + operator.index_storage_bytes
    )
    assert operator.dense_equivalent_bytes == 16 * np.dtype(np.float32).itemsize
    assert isinstance(operator.transpose_matrix, sparse.csc_array)
    assert operator.solver_storage_bytes >= operator.stored_bytes


@pytest.mark.parametrize("method", ["matvec", "rmatvec"])
def test_sparse_operator_reuses_dense_vector_validation(method):
    operator = SparseLinearOperator(np.eye(2))
    with pytest.raises(ValueError, match="must have shape"):
        getattr(operator, method)(np.ones(3))
    with pytest.raises(ValueError, match="must be finite"):
        getattr(operator, method)([1.0, np.nan])


def test_as_sparse_linear_operator_converts_and_preserves():
    operator = as_sparse_linear_operator([[1, 0], [0, 1]])
    assert isinstance(operator, SparseLinearOperator)
    assert operator.dtype == np.dtype(np.float64)
    assert as_sparse_linear_operator(operator) is operator
