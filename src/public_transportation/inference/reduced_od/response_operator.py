"""Exact equivalence-compressed response operators and basis products."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse  # type: ignore[import-untyped]

from public_transportation.preprocessing.reduced_od.equivalence import (
    ResponseEquivalence,
    build_response_equivalence,
)
from public_transportation.preprocessing.reduced_od.response_atoms import (
    MeasurementResponseArtifact,
)
from public_transportation.inference.measurement_operator_protocol import (
    GravityMeasurementOperator,
)


BasisStorage = Literal["auto", "dense", "sparse"]


@dataclass(frozen=True, slots=True)
class GravityResponseOperatorAdapter:
    """Expose an existing fixed-routing operator to the generic demand kernel."""

    operator: GravityMeasurementOperator
    free_cell_index: np.ndarray | None = None

    def __post_init__(self) -> None:
        permutation = (
            np.arange(self.operator.num_free_od, dtype=np.int64)
            if self.free_cell_index is None
            else np.asarray(self.free_cell_index, dtype=np.int64)
        )
        if permutation.shape != (self.operator.num_free_od,) or not np.array_equal(
            np.sort(permutation), np.arange(self.operator.num_free_od)
        ):
            raise ValueError("free_cell_index must be a permutation of operator columns.")
        permutation = np.array(permutation, copy=True)
        permutation.setflags(write=False)
        object.__setattr__(self, "free_cell_index", permutation)

    @property
    def number_of_measurements(self) -> int:
        return self.operator.num_measurements

    @property
    def number_of_free_cells(self) -> int:
        return self.operator.num_free_od

    @property
    def fixed_offset(self) -> object:
        return self.operator.fixed_measurement_offset

    def jax_matvec(self, vector: jax.Array) -> jax.Array:
        legacy = jnp.zeros_like(vector).at[jnp.asarray(self.free_cell_index)].set(vector)
        return self.operator.jax_matvec(legacy)


def _immutable(value: object, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ReducedResponseDiagnostics:
    """Dimensions and storage retained by an exact reduced response."""

    number_of_measurements: int
    number_of_free_cells: int
    number_of_response_classes: int
    original_nnz: int
    compressed_nnz: int
    compression_ratio: float
    retained_bytes: int


@dataclass(frozen=True, slots=True)
class ReducedResponseOperator:
    """Linear ``B`` operator whose internal columns are response classes."""

    number_of_measurements: int
    number_of_free_cells: int
    measurement_index: np.ndarray
    response_class_index: np.ndarray
    response_values: np.ndarray
    class_by_free_cell: np.ndarray
    fixed_offset: np.ndarray
    original_nnz: int

    def __post_init__(self) -> None:
        if self.number_of_measurements < 0 or self.number_of_free_cells < 0:
            raise ValueError("operator dimensions must be non-negative.")
        rows = _immutable(
            self.measurement_index, np.dtype(np.int64), "measurement_index"
        )
        classes = _immutable(
            self.response_class_index,
            np.dtype(np.int64),
            "response_class_index",
        )
        values = _immutable(
            self.response_values, np.dtype(np.float64), "response_values"
        )
        class_by_cell = _immutable(
            self.class_by_free_cell,
            np.dtype(np.int64),
            "class_by_free_cell",
        )
        offset = _immutable(self.fixed_offset, np.dtype(np.float64), "fixed_offset")
        if not (rows.size == classes.size == values.size):
            raise ValueError("compressed sparse arrays must have equal length.")
        if class_by_cell.size != self.number_of_free_cells:
            raise ValueError("class_by_free_cell must classify every free cell.")
        number_of_classes = int(class_by_cell.max()) + 1 if class_by_cell.size else 0
        if class_by_cell.size and (
            np.any(class_by_cell < 0)
            or not np.array_equal(
                np.unique(class_by_cell), np.arange(number_of_classes)
            )
        ):
            raise ValueError("response classes must be contiguous and non-negative.")
        if offset.size != self.number_of_measurements:
            raise ValueError("fixed_offset must match measurement dimension.")
        if rows.size and (
            np.any(rows < 0)
            or np.any(rows >= self.number_of_measurements)
            or np.any(classes < 0)
            or np.any(classes >= number_of_classes)
        ):
            raise ValueError(
                "compressed sparse indices are outside the operator shape."
            )
        if not np.all(np.isfinite(values)) or np.any(values == 0.0):
            raise ValueError("operator values must be finite and nonzero.")
        if not np.all(np.isfinite(offset)):
            raise ValueError("fixed_offset must be finite.")
        if self.original_nnz < values.size:
            raise ValueError("original_nnz cannot be smaller than compressed nnz.")
        order = np.lexsort((classes, rows))
        if rows.size and not np.array_equal(order, np.arange(rows.size)):
            raise ValueError("compressed entries must be sorted by row and class.")
        object.__setattr__(self, "measurement_index", rows)
        object.__setattr__(self, "response_class_index", classes)
        object.__setattr__(self, "response_values", values)
        object.__setattr__(self, "class_by_free_cell", class_by_cell)
        object.__setattr__(self, "fixed_offset", offset)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.number_of_measurements, self.number_of_free_cells)

    @property
    def number_of_response_classes(self) -> int:
        if not self.class_by_free_cell.size:
            return 0
        return int(self.class_by_free_cell.max()) + 1

    @property
    def diagnostics(self) -> ReducedResponseDiagnostics:
        arrays = (
            self.measurement_index,
            self.response_class_index,
            self.response_values,
            self.class_by_free_cell,
            self.fixed_offset,
        )
        return ReducedResponseDiagnostics(
            number_of_measurements=self.number_of_measurements,
            number_of_free_cells=self.number_of_free_cells,
            number_of_response_classes=self.number_of_response_classes,
            original_nnz=self.original_nnz,
            compressed_nnz=int(self.response_values.size),
            compression_ratio=(
                self.number_of_response_classes / self.number_of_free_cells
                if self.number_of_free_cells
                else 1.0
            ),
            retained_bytes=sum(int(array.nbytes) for array in arrays),
        )

    def matvec(self, vector: object) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64)
        if value.shape != (self.number_of_free_cells,):
            raise ValueError("matvec input has an invalid shape.")
        class_values = np.zeros(self.number_of_response_classes, dtype=np.float64)
        np.add.at(class_values, self.class_by_free_cell, value)
        result = np.zeros(self.number_of_measurements, dtype=np.float64)
        np.add.at(
            result,
            self.measurement_index,
            self.response_values * class_values[self.response_class_index],
        )
        return result

    def predict(self, vector: object) -> np.ndarray:
        return self.fixed_offset + self.matvec(vector)

    def rmatvec(self, vector: object) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64)
        if value.shape != (self.number_of_measurements,):
            raise ValueError("rmatvec input has an invalid shape.")
        class_gradient = np.zeros(self.number_of_response_classes, dtype=np.float64)
        np.add.at(
            class_gradient,
            self.response_class_index,
            self.response_values * value[self.measurement_index],
        )
        return class_gradient[self.class_by_free_cell]

    def jax_matvec(self, vector: jax.Array) -> jax.Array:
        value = jnp.asarray(vector)
        if value.ndim != 1 or value.shape[0] != self.number_of_free_cells:
            raise ValueError("jax_matvec input has an invalid shape.")
        class_by_cell = jnp.asarray(self.class_by_free_cell)
        rows = jnp.asarray(self.measurement_index)
        classes = jnp.asarray(self.response_class_index)
        coefficients = jnp.asarray(self.response_values, dtype=value.dtype)
        class_values = (
            jnp.zeros((self.number_of_response_classes,), dtype=value.dtype)
            .at[class_by_cell]
            .add(value)
        )
        return (
            jnp.zeros((self.number_of_measurements,), dtype=value.dtype)
            .at[rows]
            .add(coefficients * class_values[classes])
        )

    def jax_rmatvec(self, vector: jax.Array) -> jax.Array:
        value = jnp.asarray(vector)
        if value.ndim != 1 or value.shape[0] != self.number_of_measurements:
            raise ValueError("jax_rmatvec input has an invalid shape.")
        rows = jnp.asarray(self.measurement_index)
        classes = jnp.asarray(self.response_class_index)
        coefficients = jnp.asarray(self.response_values, dtype=value.dtype)
        class_gradient = (
            jnp.zeros((self.number_of_response_classes,), dtype=value.dtype)
            .at[classes]
            .add(coefficients * value[rows])
        )
        return class_gradient[jnp.asarray(self.class_by_free_cell)]


def build_reduced_response_operator(
    artifact: MeasurementResponseArtifact,
) -> ReducedResponseOperator:
    """Compress a Phase-6 response by its validated exact column classes."""
    recomputed = build_response_equivalence(
        number_of_cells=artifact.number_of_free_cells,
        measurement_index=artifact.measurement_index,
        cell_index=artifact.free_cell_index,
        values=artifact.response_values,
    )
    if not all(
        np.array_equal(left, right)
        for left, right in (
            (recomputed.class_by_cell, artifact.equivalence.class_by_cell),
            (
                recomputed.representative_cell_indices,
                artifact.equivalence.representative_cell_indices,
            ),
            (recomputed.member_indptr, artifact.equivalence.member_indptr),
            (
                recomputed.member_cell_indices,
                artifact.equivalence.member_cell_indices,
            ),
        )
    ):
        raise ValueError("artifact response equivalence is not exact.")
    representative_to_class = {
        int(cell): class_index
        for class_index, cell in enumerate(
            artifact.equivalence.representative_cell_indices
        )
    }
    compressed = [
        (
            int(row),
            representative_to_class[int(cell)],
            float(value),
        )
        for row, cell, value in zip(
            artifact.measurement_index,
            artifact.free_cell_index,
            artifact.response_values,
            strict=True,
        )
        if int(cell) in representative_to_class
    ]
    compressed.sort()
    return ReducedResponseOperator(
        number_of_measurements=artifact.number_of_measurements,
        number_of_free_cells=artifact.number_of_free_cells,
        measurement_index=np.asarray([item[0] for item in compressed], dtype=np.int64),
        response_class_index=np.asarray(
            [item[1] for item in compressed], dtype=np.int64
        ),
        response_values=np.asarray([item[2] for item in compressed], dtype=np.float64),
        class_by_free_cell=artifact.equivalence.class_by_cell,
        fixed_offset=artifact.fixed_offset,
        original_nnz=artifact.nnz,
    )


@dataclass(frozen=True, slots=True)
class BasisResponseDiagnostics:
    """Construction and storage summary for ``H = B Phi``."""

    number_of_basis_parameters: int
    response_nnz: int
    basis_nnz: int
    result_nnz: int
    result_density: float
    storage: Literal["dense", "sparse"]
    retained_bytes: int


@dataclass(frozen=True, slots=True)
class BasisResponse:
    """Materialized compact ``H`` plus an operator sharing the fixed offset."""

    matrix: np.ndarray | sparse.csr_matrix
    operator: ReducedResponseOperator
    diagnostics: BasisResponseDiagnostics


def _class_basis_dense(
    operator: ReducedResponseOperator, basis: np.ndarray
) -> np.ndarray:
    result = np.zeros(
        (operator.number_of_response_classes, basis.shape[1]), dtype=np.float64
    )
    np.add.at(result, operator.class_by_free_cell, basis)
    return result


def _compressed_sparse_matrix(operator: ReducedResponseOperator) -> sparse.csr_matrix:
    return sparse.coo_matrix(
        (
            operator.response_values,
            (operator.measurement_index, operator.response_class_index),
        ),
        shape=(operator.number_of_measurements, operator.number_of_response_classes),
    ).tocsr()


def build_basis_response(
    operator: ReducedResponseOperator,
    basis: object,
    *,
    storage: BasisStorage = "auto",
    sparse_density_threshold: float = 0.2,
) -> BasisResponse:
    """Construct ``H=B Phi`` directly from compressed response atoms."""
    if storage not in {"auto", "dense", "sparse"}:
        raise ValueError("storage must be 'auto', 'dense', or 'sparse'.")
    if (
        not math.isfinite(sparse_density_threshold)
        or sparse_density_threshold < 0.0
        or sparse_density_threshold > 1.0
    ):
        raise ValueError("sparse_density_threshold must lie in [0, 1].")
    response = _compressed_sparse_matrix(operator)
    if sparse.issparse(basis):
        phi = sparse.csr_matrix(basis, dtype=np.float64)
        if phi.ndim != 2 or phi.shape[0] != operator.number_of_free_cells:
            raise ValueError("basis has an invalid shape.")
        if not np.all(np.isfinite(phi.data)):
            raise ValueError("basis must contain finite values.")
        aggregation = sparse.coo_matrix(
            (
                np.ones(operator.number_of_free_cells, dtype=np.float64),
                (
                    operator.class_by_free_cell,
                    np.arange(operator.number_of_free_cells),
                ),
            ),
            shape=(
                operator.number_of_response_classes,
                operator.number_of_free_cells,
            ),
        ).tocsr()
        class_basis_sparse = aggregation @ phi
        result_sparse = (response @ class_basis_sparse).tocsr()
        basis_nnz = int(phi.nnz)
    else:
        phi_dense = np.asarray(basis, dtype=np.float64)
        if phi_dense.ndim != 2 or phi_dense.shape[0] != operator.number_of_free_cells:
            raise ValueError("basis has an invalid shape.")
        if not np.all(np.isfinite(phi_dense)):
            raise ValueError("basis must contain finite values.")
        class_basis_dense = _class_basis_dense(operator, phi_dense)
        result_sparse = sparse.csr_matrix(response @ class_basis_dense)
        basis_nnz = int(np.count_nonzero(phi_dense))
    result_sparse.eliminate_zeros()
    rows, parameters = result_sparse.shape
    entries = rows * parameters
    density = result_sparse.nnz / entries if entries else 0.0
    selected_storage: Literal["dense", "sparse"]
    if storage == "auto":
        selected_storage = "sparse" if density <= sparse_density_threshold else "dense"
    else:
        selected_storage = storage
    if selected_storage == "dense":
        matrix: np.ndarray | sparse.csr_matrix = np.asarray(
            result_sparse.toarray(), dtype=np.float64
        )
        retained_bytes = int(matrix.nbytes)
    else:
        matrix = result_sparse
        retained_bytes = int(
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        )
    coo = result_sparse.tocoo()
    equivalence: ResponseEquivalence = build_response_equivalence(
        number_of_cells=parameters,
        measurement_index=np.asarray(coo.row, dtype=np.int64),
        cell_index=np.asarray(coo.col, dtype=np.int64),
        values=np.asarray(coo.data, dtype=np.float64),
    )
    basis_artifact_operator = _operator_from_coo(
        number_of_measurements=rows,
        number_of_cells=parameters,
        rows=np.asarray(coo.row, dtype=np.int64),
        columns=np.asarray(coo.col, dtype=np.int64),
        values=np.asarray(coo.data, dtype=np.float64),
        equivalence=equivalence,
        fixed_offset=operator.fixed_offset,
    )
    return BasisResponse(
        matrix=matrix,
        operator=basis_artifact_operator,
        diagnostics=BasisResponseDiagnostics(
            number_of_basis_parameters=parameters,
            response_nnz=int(operator.response_values.size),
            basis_nnz=basis_nnz,
            result_nnz=int(result_sparse.nnz),
            result_density=density,
            storage=selected_storage,
            retained_bytes=retained_bytes,
        ),
    )


def _operator_from_coo(
    *,
    number_of_measurements: int,
    number_of_cells: int,
    rows: np.ndarray,
    columns: np.ndarray,
    values: np.ndarray,
    equivalence: ResponseEquivalence,
    fixed_offset: np.ndarray,
) -> ReducedResponseOperator:
    representatives = {
        int(cell): class_index
        for class_index, cell in enumerate(equivalence.representative_cell_indices)
    }
    selected = [
        (int(row), representatives[int(column)], float(value))
        for row, column, value in zip(rows, columns, values, strict=True)
        if int(column) in representatives
    ]
    selected.sort()
    return ReducedResponseOperator(
        number_of_measurements=number_of_measurements,
        number_of_free_cells=number_of_cells,
        measurement_index=np.asarray([item[0] for item in selected], dtype=np.int64),
        response_class_index=np.asarray([item[1] for item in selected], dtype=np.int64),
        response_values=np.asarray([item[2] for item in selected], dtype=np.float64),
        class_by_free_cell=equivalence.class_by_cell,
        fixed_offset=fixed_offset,
        original_nnz=int(values.size),
    )


def build_reduced_response_operator_from_coo(
    *,
    number_of_measurements: int,
    number_of_free_cells: int,
    measurement_index: object,
    free_cell_index: object,
    response_values: object,
    fixed_offset: object | None = None,
) -> ReducedResponseOperator:
    """Build an exact reduced operator from canonicalizable COO response atoms."""
    if number_of_measurements < 0 or number_of_free_cells < 0:
        raise ValueError("operator dimensions must be non-negative.")
    rows = np.asarray(measurement_index, dtype=np.int64)
    columns = np.asarray(free_cell_index, dtype=np.int64)
    values = np.asarray(response_values, dtype=np.float64)
    if rows.ndim != 1 or columns.ndim != 1 or values.ndim != 1:
        raise ValueError("COO arrays must be one-dimensional.")
    if not (rows.size == columns.size == values.size):
        raise ValueError("COO arrays must have equal length.")
    if rows.size and (
        np.any(rows < 0)
        or np.any(rows >= number_of_measurements)
        or np.any(columns < 0)
        or np.any(columns >= number_of_free_cells)
    ):
        raise ValueError("COO indices are outside the declared shape.")
    if not np.all(np.isfinite(values)):
        raise ValueError("COO values must be finite.")
    canonical = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(number_of_measurements, number_of_free_cells),
    ).tocsr()
    canonical.eliminate_zeros()
    canonical_coo = canonical.tocoo()
    canonical_rows = np.asarray(canonical_coo.row, dtype=np.int64)
    canonical_columns = np.asarray(canonical_coo.col, dtype=np.int64)
    canonical_values = np.asarray(canonical_coo.data, dtype=np.float64)
    equivalence = build_response_equivalence(
        number_of_cells=number_of_free_cells,
        measurement_index=canonical_rows,
        cell_index=canonical_columns,
        values=canonical_values,
    )
    if fixed_offset is None:
        offset = np.zeros(number_of_measurements, dtype=np.float64)
    else:
        offset = np.asarray(fixed_offset, dtype=np.float64)
    return _operator_from_coo(
        number_of_measurements=number_of_measurements,
        number_of_cells=number_of_free_cells,
        rows=canonical_rows,
        columns=canonical_columns,
        values=canonical_values,
        equivalence=equivalence,
        fixed_offset=offset,
    )
