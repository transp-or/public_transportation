"""Generic linear operators restricted to one coordinate block."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from scipy import sparse

from public_transportation.inference.linear_operator import LinearOperatorProtocol

Array = np.ndarray


def _finite_vector(value: object, *, name: str, length: int) -> Array:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 1 or array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _support_tuple(value: object, *, num_measurements: int) -> tuple[int, ...]:
    try:
        support = tuple(int(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError("measurement support must contain integers.") from error
    if support != tuple(sorted(set(support))):
        raise ValueError("measurement support must be unique and ascending.")
    if any(item < 0 or item >= num_measurements for item in support):
        raise ValueError("measurement support contains an out-of-range index.")
    return support


@runtime_checkable
class BlockLinearOperatorProtocol(Protocol):
    """Measurement operator for only the variables in one OD block."""

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def dtype(self) -> np.dtype: ...

    @property
    def num_measurements(self) -> int: ...

    @property
    def num_local_variables(self) -> int: ...

    @property
    def measurement_support_indices(self) -> tuple[int, ...]: ...

    def matvec(self, local_vector: object) -> Array: ...

    def rmatvec(self, measurement_vector: object) -> Array: ...


@dataclass(frozen=True, slots=True)
class DenseBlockLinearOperator:
    """Immutable dense block operator used for reference calculations."""

    matrix: Array
    measurement_support_indices: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix)
        if matrix.dtype.kind not in "iuf":
            raise TypeError("matrix must contain real numeric values.")
        matrix = np.array(matrix, dtype=np.float64, copy=True)
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be two-dimensional, got {matrix.shape}.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must be finite.")
        support = tuple(np.flatnonzero(np.any(matrix != 0.0, axis=1)).tolist())
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "measurement_support_indices", support)

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    @property
    def dtype(self) -> np.dtype:
        return self.matrix.dtype

    @property
    def num_measurements(self) -> int:
        return self.shape[0]

    @property
    def num_local_variables(self) -> int:
        return self.shape[1]

    def matvec(self, local_vector: object) -> Array:
        vector = _finite_vector(
            local_vector, name="local vector", length=self.num_local_variables
        )
        return np.asarray(self.matrix @ vector)

    def rmatvec(self, measurement_vector: object) -> Array:
        vector = _finite_vector(
            measurement_vector,
            name="measurement vector",
            length=self.num_measurements,
        )
        return np.asarray(self.matrix.T @ vector)


@dataclass(frozen=True, slots=True)
class SparseBlockLinearOperator:
    """Immutable sparse block operator with persistent CSR and CSC storage."""

    matrix: sparse.spmatrix | sparse.sparray | Array
    transpose_matrix: sparse.csc_array = field(init=False, repr=False, compare=False)
    measurement_support_indices: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        try:
            matrix = sparse.csr_array(self.matrix, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "matrix must be convertible to a real numeric CSR array."
            ) from error
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be two-dimensional, got {matrix.shape}.")
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("matrix must be finite.")
        support = tuple(np.flatnonzero(np.diff(matrix.indptr) != 0).tolist())
        transpose = sparse.csc_array(matrix, copy=True)
        for array in (matrix.data, matrix.indices, matrix.indptr):
            array.setflags(write=False)
        for array in (transpose.data, transpose.indices, transpose.indptr):
            array.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "transpose_matrix", transpose)
        object.__setattr__(self, "measurement_support_indices", support)

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    @property
    def dtype(self) -> np.dtype:
        return self.matrix.dtype

    @property
    def num_measurements(self) -> int:
        return self.shape[0]

    @property
    def num_local_variables(self) -> int:
        return self.shape[1]

    def matvec(self, local_vector: object) -> Array:
        vector = _finite_vector(
            local_vector, name="local vector", length=self.num_local_variables
        )
        return np.asarray(self.matrix @ vector)

    def rmatvec(self, measurement_vector: object) -> Array:
        vector = _finite_vector(
            measurement_vector,
            name="measurement vector",
            length=self.num_measurements,
        )
        return np.asarray(self.transpose_matrix.T @ vector)


@dataclass(frozen=True, slots=True)
class SupportedRowsSparseBlockLinearOperator:
    """Sparse block stored only on supported rows with full-height products."""

    compact_matrix: sparse.spmatrix | sparse.sparray | Array
    full_num_measurements: int
    support_rows: tuple[int, ...]
    transpose_matrix: sparse.csc_array = field(init=False, repr=False, compare=False)
    measurement_support_indices: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        matrix = sparse.csr_array(self.compact_matrix, copy=True)
        if matrix.dtype.kind not in "iuf":
            raise TypeError("compact matrix must contain real numeric values.")
        if matrix.dtype.kind in "iu":
            matrix = matrix.astype(np.float64)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        rows = _support_tuple(
            self.support_rows, num_measurements=self.full_num_measurements
        )
        if matrix.shape[0] != len(rows):
            raise ValueError("compact matrix height must equal the support-row count.")
        if self.full_num_measurements <= 0:
            raise ValueError("full_num_measurements must be positive.")
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("compact matrix must be finite.")
        nonempty = np.diff(matrix.indptr) != 0
        realized_rows = tuple(np.asarray(rows)[nonempty].tolist())
        transpose = sparse.csc_array(matrix, copy=True)
        for storage in (matrix, transpose):
            for array in (storage.data, storage.indices, storage.indptr):
                array.setflags(write=False)
        object.__setattr__(self, "compact_matrix", matrix)
        object.__setattr__(self, "support_rows", rows)
        object.__setattr__(self, "transpose_matrix", transpose)
        object.__setattr__(self, "measurement_support_indices", realized_rows)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.full_num_measurements, self.compact_matrix.shape[1])

    @property
    def dtype(self) -> np.dtype:
        return self.compact_matrix.dtype

    @property
    def num_measurements(self) -> int:
        return self.full_num_measurements

    @property
    def num_local_variables(self) -> int:
        return self.compact_matrix.shape[1]

    @property
    def retained_bytes(self) -> int:
        return int(
            sum(
                array.nbytes
                for storage in (self.compact_matrix, self.transpose_matrix)
                for array in (storage.data, storage.indices, storage.indptr)
            )
            + np.asarray(self.support_rows, dtype=np.int64).nbytes
        )

    def matvec(self, local_vector: object) -> Array:
        vector = _finite_vector(
            local_vector, name="local vector", length=self.num_local_variables
        )
        result = np.zeros(self.full_num_measurements, dtype=self.dtype)
        result[np.asarray(self.support_rows, dtype=np.int64)] = (
            self.compact_matrix @ vector
        )
        return result

    def rmatvec(self, measurement_vector: object) -> Array:
        vector = _finite_vector(
            measurement_vector,
            name="measurement vector",
            length=self.full_num_measurements,
        )
        return np.asarray(
            self.transpose_matrix.T
            @ vector[np.asarray(self.support_rows, dtype=np.int64)]
        )


@dataclass(frozen=True, slots=True)
class ColumnSelectedLinearOperator:
    """Reference block adapter selecting columns from a complete operator.

    Forward products allocate a complete input vector.  This adapter is useful
    for correctness checks and small problems; scalable assignment-specific
    block construction belongs in a later phase.
    """

    operator: LinearOperatorProtocol
    column_indices: tuple[int, ...]
    measurement_support_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        rows, columns = self.operator.shape
        selected = tuple(int(item) for item in self.column_indices)
        if selected != tuple(sorted(set(selected))):
            raise ValueError("column_indices must be unique and ascending.")
        if any(item < 0 or item >= columns for item in selected):
            raise ValueError("column_indices contains an out-of-range index.")
        support = (
            tuple(range(rows))
            if self.measurement_support_indices is None
            else _support_tuple(self.measurement_support_indices, num_measurements=rows)
        )
        object.__setattr__(self, "column_indices", selected)
        object.__setattr__(self, "measurement_support_indices", support)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.operator.shape[0], len(self.column_indices))

    @property
    def dtype(self) -> np.dtype:
        return self.operator.dtype

    @property
    def num_measurements(self) -> int:
        return self.shape[0]

    @property
    def num_local_variables(self) -> int:
        return self.shape[1]

    def matvec(self, local_vector: object) -> Array:
        vector = _finite_vector(
            local_vector, name="local vector", length=self.num_local_variables
        )
        complete = np.zeros(self.operator.shape[1], dtype=self.dtype)
        complete[np.asarray(self.column_indices, dtype=np.intp)] = vector
        return np.asarray(self.operator.matvec(complete))

    def rmatvec(self, measurement_vector: object) -> Array:
        vector = _finite_vector(
            measurement_vector,
            name="measurement vector",
            length=self.num_measurements,
        )
        complete = np.asarray(self.operator.rmatvec(vector))
        return complete[np.asarray(self.column_indices, dtype=np.intp)]
