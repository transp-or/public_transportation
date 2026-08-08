from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse

from public_transportation.inference.reduced_od import (
    build_basis_response,
    build_reduced_response_operator,
)
from public_transportation.measurement import MeasurementType
from public_transportation.preprocessing.reduced_od import (
    MeasurementResponseArtifact,
    ResolvedMeasurement,
    ResponseCellKey,
    build_response_equivalence,
)


def _measurement(row: int) -> ResolvedMeasurement:
    return ResolvedMeasurement(
        row_index=row,
        method_id="synthetic",
        measurement_type=(
            MeasurementType.BOARDING if row != 1 else MeasurementType.ALIGHTING
        ),
        scenario_stop_id=f"S{row}",
        physical_stop_id=f"S{row}",
        seconds=100 + row,
        trip_id=f"T{row}",
        line_id=f"L{row}",
        observed_value=float(row),
    )


def _artifact() -> tuple[MeasurementResponseArtifact, np.ndarray]:
    dense = np.asarray(
        [
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [3.0, 0.0, 3.0],
        ]
    )
    rows, columns = np.nonzero(dense)
    values = dense[rows, columns]
    equivalence = build_response_equivalence(
        number_of_cells=3,
        measurement_index=rows,
        cell_index=columns,
        values=values,
    )
    artifact = MeasurementResponseArtifact(
        configuration_fingerprint="configuration",
        timetable_fingerprint="timetable",
        journey_choice_fingerprint="choices",
        measurement_fingerprint="measurements",
        free_cell_keys=(
            ResponseCellKey("A", "B", "P"),
            ResponseCellKey("A", "C", "P"),
            ResponseCellKey("A", "D", "P"),
        ),
        fixed_cell_keys=(),
        resolved_measurements=tuple(_measurement(row) for row in range(3)),
        observed_values=np.asarray([0.0, 1.0, 2.0]),
        measurement_index=rows,
        free_cell_index=columns,
        response_values=values,
        fixed_offset=np.asarray([1.0, 0.5, 2.0]),
        equivalence=equivalence,
    )
    return artifact, dense


def test_compressed_matvec_rmatvec_and_offset_match_dense_reference() -> None:
    artifact, dense = _artifact()
    operator = build_reduced_response_operator(artifact)
    demand = np.asarray([2.0, 3.0, 5.0])
    weights = np.asarray([0.5, -2.0, 1.5])

    np.testing.assert_allclose(operator.matvec(demand), dense @ demand)
    np.testing.assert_allclose(
        operator.predict(demand), artifact.fixed_offset + dense @ demand
    )
    np.testing.assert_allclose(operator.rmatvec(weights), dense.T @ weights)
    assert np.vdot(weights, operator.matvec(demand)) == pytest.approx(
        np.vdot(demand, operator.rmatvec(weights))
    )
    assert operator.shape == dense.shape
    assert operator.diagnostics.number_of_response_classes == 2
    assert operator.diagnostics.original_nnz == 5
    assert operator.diagnostics.compressed_nnz == 3
    assert operator.diagnostics.compression_ratio == pytest.approx(2 / 3)


def test_dense_basis_response_equals_explicit_b_phi() -> None:
    artifact, dense = _artifact()
    operator = build_reduced_response_operator(artifact)
    basis = np.asarray([[1.0, 0.0], [0.0, 2.0], [3.0, -1.0]])
    expected = dense @ basis

    result = build_basis_response(operator, basis, storage="dense")
    assert isinstance(result.matrix, np.ndarray)
    np.testing.assert_allclose(result.matrix, expected)
    np.testing.assert_allclose(
        result.operator.matvec(np.asarray([2.0, 4.0])),
        expected @ np.asarray([2.0, 4.0]),
    )
    np.testing.assert_allclose(result.operator.fixed_offset, artifact.fixed_offset)
    assert result.diagnostics.storage == "dense"
    assert result.diagnostics.number_of_basis_parameters == 2


def test_sparse_basis_and_sparse_h_equal_dense_reference() -> None:
    artifact, dense = _artifact()
    operator = build_reduced_response_operator(artifact)
    basis_dense = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]]
    )
    result = build_basis_response(
        operator, sparse.csr_matrix(basis_dense), storage="sparse"
    )

    assert sparse.isspmatrix_csr(result.matrix)
    np.testing.assert_allclose(result.matrix.toarray(), dense @ basis_dense)
    direction = np.asarray([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        result.operator.matvec(direction), (dense @ basis_dense) @ direction
    )
    assert result.diagnostics.storage == "sparse"
    assert result.diagnostics.basis_nnz == 3


def test_auto_storage_uses_result_density() -> None:
    artifact, _ = _artifact()
    operator = build_reduced_response_operator(artifact)
    basis = np.eye(3)
    sparse_result = build_basis_response(
        operator, basis, storage="auto", sparse_density_threshold=1.0
    )
    dense_result = build_basis_response(
        operator, basis, storage="auto", sparse_density_threshold=0.0
    )
    assert sparse_result.diagnostics.storage == "sparse"
    assert dense_result.diagnostics.storage == "dense"


def test_jax_products_and_gradients_match_numpy_adjoint() -> None:
    artifact, dense = _artifact()
    operator = build_reduced_response_operator(artifact)
    demand = jnp.asarray([2.0, 3.0, 5.0])
    weights = jnp.asarray([0.5, -2.0, 1.5])

    forward = jax.jit(operator.jax_matvec)(demand)
    reverse = jax.jit(operator.jax_rmatvec)(weights)
    gradient = jax.grad(
        lambda value: jnp.vdot(operator.jax_matvec(value), weights)
    )(demand)
    np.testing.assert_allclose(forward, dense @ np.asarray(demand), rtol=1e-6)
    np.testing.assert_allclose(reverse, dense.T @ np.asarray(weights), rtol=1e-6)
    np.testing.assert_allclose(gradient, reverse, rtol=1e-6)


def test_empty_free_system_and_zero_basis_are_well_defined() -> None:
    equivalence = build_response_equivalence(
        number_of_cells=0,
        measurement_index=np.asarray([], dtype=np.int64),
        cell_index=np.asarray([], dtype=np.int64),
        values=np.asarray([], dtype=float),
    )
    artifact = MeasurementResponseArtifact(
        configuration_fingerprint="configuration",
        timetable_fingerprint="timetable",
        journey_choice_fingerprint="choices",
        measurement_fingerprint="measurements",
        free_cell_keys=(),
        fixed_cell_keys=(ResponseCellKey("A", "B", "P"),),
        resolved_measurements=(_measurement(0), _measurement(1)),
        observed_values=np.asarray([0.0, 1.0]),
        measurement_index=np.asarray([], dtype=np.int64),
        free_cell_index=np.asarray([], dtype=np.int64),
        response_values=np.asarray([], dtype=float),
        fixed_offset=np.asarray([4.0, 5.0]),
        equivalence=equivalence,
    )
    operator = build_reduced_response_operator(artifact)
    np.testing.assert_array_equal(operator.matvec(np.asarray([])), [0.0, 0.0])
    np.testing.assert_array_equal(operator.predict(np.asarray([])), [4.0, 5.0])
    np.testing.assert_array_equal(operator.rmatvec(np.ones(2)), [])

    basis_result = build_basis_response(operator, np.zeros((0, 3)), storage="dense")
    np.testing.assert_array_equal(basis_result.matrix, np.zeros((2, 3)))
    np.testing.assert_array_equal(
        basis_result.operator.matvec(np.ones(3)), np.zeros(2)
    )


def test_invalid_shapes_nonfinite_basis_and_storage_are_rejected() -> None:
    artifact, _ = _artifact()
    operator = build_reduced_response_operator(artifact)
    with pytest.raises(ValueError, match="invalid shape"):
        operator.matvec(np.ones(2))
    with pytest.raises(ValueError, match="invalid shape"):
        operator.rmatvec(np.ones(2))
    with pytest.raises(ValueError, match="invalid shape"):
        build_basis_response(operator, np.ones((2, 2)))
    bad = np.ones((3, 2))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_basis_response(operator, bad)
    with pytest.raises(ValueError, match="storage"):
        build_basis_response(operator, np.ones((3, 2)), storage="invalid")  # type: ignore[arg-type]
