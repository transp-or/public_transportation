"""Solver-independent linear-operator contracts for OD estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from scipy import sparse

Array = np.ndarray


def _real_floating_array(value: object, *, name: str) -> Array:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")
    if array.dtype.kind in "iu":
        return array.astype(np.float64)
    return np.array(array, copy=True)


@runtime_checkable
class LinearOperatorProtocol(Protocol):
    """Minimal rectangular operator required by linear least-squares solvers."""

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def dtype(self) -> np.dtype: ...

    def matvec(self, vector: object) -> Array:
        """Return the forward product ``A @ vector``."""

    def rmatvec(self, vector: object) -> Array:
        """Return the transpose product ``A.T @ vector``."""


@dataclass(frozen=True, slots=True)
class DenseLinearOperator:
    """Immutable dense implementation of :class:`LinearOperatorProtocol`."""

    matrix: Array

    def __post_init__(self) -> None:
        matrix = _real_floating_array(self.matrix, name="matrix")
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be two-dimensional, got {matrix.shape}.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must be finite.")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    @property
    def dtype(self) -> np.dtype:
        return self.matrix.dtype

    def matvec(self, vector: object) -> Array:
        value = _real_floating_array(vector, name="forward vector")
        expected = (self.shape[1],)
        if value.ndim != 1 or value.shape != expected:
            raise ValueError(
                f"forward vector must have shape {expected}, got {value.shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("forward vector must be finite.")
        return np.asarray(self.matrix @ value)

    def rmatvec(self, vector: object) -> Array:
        value = _real_floating_array(vector, name="transpose vector")
        expected = (self.shape[0],)
        if value.ndim != 1 or value.shape != expected:
            raise ValueError(
                f"transpose vector must have shape {expected}, got {value.shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("transpose vector must be finite.")
        return np.asarray(self.matrix.T @ value)


@dataclass(frozen=True, slots=True)
class SparseLinearOperator:
    """Immutable CPU sparse operator with persistent CSR and CSC storage.

    CSR serves forward products and CSC serves transpose products.  Both formats
    are constructed once so iterative SciPy solvers never repeat conversion or
    transfer data through JAX.
    """

    matrix: sparse.spmatrix | sparse.sparray | Array
    transpose_matrix: sparse.csc_array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            matrix = sparse.csr_array(self.matrix, copy=True)
        except (TypeError, ValueError) as error:
            raise TypeError("matrix must be convertible to a real numeric CSR array.") from error
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be two-dimensional, got {matrix.shape}.")
        if matrix.dtype.kind not in "iuf":
            raise TypeError("matrix must contain real numeric values.")
        if matrix.dtype.kind in "iu":
            matrix = matrix.astype(np.float64)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if not np.all(np.isfinite(matrix.data)):
            raise ValueError("matrix must be finite.")
        matrix.data.setflags(write=False)
        matrix.indices.setflags(write=False)
        matrix.indptr.setflags(write=False)
        transpose = sparse.csc_array(matrix, copy=True)
        transpose.sum_duplicates()
        transpose.eliminate_zeros()
        transpose.sort_indices()
        transpose.data.setflags(write=False)
        transpose.indices.setflags(write=False)
        transpose.indptr.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "transpose_matrix", transpose)

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    @property
    def dtype(self) -> np.dtype:
        return self.matrix.dtype

    @property
    def nonzero_entries(self) -> int:
        return int(self.matrix.nnz)

    @property
    def total_entries(self) -> int:
        return int(self.shape[0] * self.shape[1])

    @property
    def density(self) -> float:
        total = self.total_entries
        return 0.0 if total == 0 else self.nonzero_entries / total

    @property
    def value_storage_bytes(self) -> int:
        return int(self.matrix.data.nbytes)

    @property
    def index_storage_bytes(self) -> int:
        return int(self.matrix.indices.nbytes + self.matrix.indptr.nbytes)

    @property
    def stored_bytes(self) -> int:
        return self.value_storage_bytes + self.index_storage_bytes

    @property
    def solver_storage_bytes(self) -> int:
        """Bytes held by the persistent CSR/CSC pair used by CPU solvers."""
        return self.stored_bytes + int(
            self.transpose_matrix.data.nbytes
            + self.transpose_matrix.indices.nbytes
            + self.transpose_matrix.indptr.nbytes
        )

    @property
    def dense_equivalent_bytes(self) -> int:
        return int(self.total_entries * self.dtype.itemsize)

    def matvec(self, vector: object) -> Array:
        value = _real_floating_array(vector, name="forward vector")
        expected = (self.shape[1],)
        if value.ndim != 1 or value.shape != expected:
            raise ValueError(
                f"forward vector must have shape {expected}, got {value.shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("forward vector must be finite.")
        return np.asarray(self.matrix @ value)

    def rmatvec(self, vector: object) -> Array:
        value = _real_floating_array(vector, name="transpose vector")
        expected = (self.shape[0],)
        if value.ndim != 1 or value.shape != expected:
            raise ValueError(
                f"transpose vector must have shape {expected}, got {value.shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("transpose vector must be finite.")
        return np.asarray(self.transpose_matrix.T @ value)


def as_linear_operator(value: object) -> LinearOperatorProtocol:
    """Normalize dense arrays or retain an existing protocol implementation."""
    if isinstance(value, LinearOperatorProtocol):
        return value
    return DenseLinearOperator(value)


def as_sparse_linear_operator(value: object) -> SparseLinearOperator:
    """Return a canonical CSR operator, preserving an existing instance."""
    if isinstance(value, SparseLinearOperator):
        return value
    return SparseLinearOperator(value)


def materialize_linear_operator(
    operator: LinearOperatorProtocol, *, max_entries: int = 10_000_000
) -> Array:
    """Materialize a small operator column by column for reference calculations."""
    if max_entries <= 0:
        raise ValueError("max_entries must be strictly positive.")
    total_entries = int(operator.shape[0] * operator.shape[1])
    if total_entries > max_entries:
        raise ValueError(
            f"operator has {total_entries} entries, exceeding the explicit "
            f"materialization limit {max_entries}."
        )
    matrix = np.empty(operator.shape, dtype=operator.dtype)
    basis = np.zeros(operator.shape[1], dtype=operator.dtype)
    for column in range(operator.shape[1]):
        basis[column] = 1.0
        matrix[:, column] = operator.matvec(basis)
        basis[column] = 0.0
    return matrix
