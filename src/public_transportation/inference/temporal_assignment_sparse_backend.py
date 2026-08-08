"""Optimized CSR/CSC execution for persisted temporal assignment blocks."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from .temporal_assignment_blocks import TemporalBlockAssignmentOperator


@dataclass(frozen=True, slots=True)
class TemporalSparseBackendMetrics:
    nonzero_entries: int
    csr_bytes: int
    csc_bytes: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class CSRCSCTemporalAssignmentOperator:
    """CPU sparse backend retaining forward-CSR and adjoint-CSC layouts."""

    source: TemporalBlockAssignmentOperator
    _csr: sparse.csr_array = field(init=False, repr=False)
    _csc: sparse.csc_array = field(init=False, repr=False)
    _jax_forward: object = field(init=False, repr=False)
    metrics: TemporalSparseBackendMetrics = field(init=False)

    def __post_init__(self) -> None:
        rows = np.concatenate(
            [block.row_indices for block in self.source.blocks], dtype=np.int32
        ) if self.source.blocks else np.empty(0, dtype=np.int32)
        columns = np.concatenate(
            [block.column_indices for block in self.source.blocks], dtype=np.int32
        ) if self.source.blocks else np.empty(0, dtype=np.int32)
        values = np.concatenate(
            [block.values for block in self.source.blocks]
        ) if self.source.blocks else np.empty(0, dtype=self.source.dtype)
        csr = sparse.coo_array(
            (values, (rows, columns)),
            shape=(self.number_of_measurements, self.number_of_demand_cells),
            dtype=self.dtype,
        ).tocsr()
        csr.sum_duplicates()
        csr.eliminate_zeros()
        csr.sort_indices()
        csc = sparse.csc_array(csr, copy=True)
        csc.sort_indices()
        for array in (csr.data, csr.indices, csr.indptr, csc.data, csc.indices, csc.indptr):
            array.setflags(write=False)
        object.__setattr__(self, "_csr", csr)
        object.__setattr__(self, "_csc", csc)
        csr_bytes = int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)
        csc_bytes = int(csc.data.nbytes + csc.indices.nbytes + csc.indptr.nbytes)
        object.__setattr__(
            self,
            "metrics",
            TemporalSparseBackendMetrics(
                nonzero_entries=int(csr.nnz),
                csr_bytes=csr_bytes,
                csc_bytes=csc_bytes,
                total_bytes=csr_bytes + csc_bytes,
            ),
        )
        forward_shape = jax.ShapeDtypeStruct(
            (self.number_of_measurements,), self.dtype
        )
        reverse_shape = jax.ShapeDtypeStruct(
            (self.number_of_demand_cells,), self.dtype
        )

        @jax.custom_vjp
        def forward(value: jax.Array) -> jax.Array:
            return jax.pure_callback(
                self._host_matvec, forward_shape, value, vmap_method="sequential"
            )

        def forward_rule(value: jax.Array):
            result = jax.pure_callback(
                self._host_matvec, forward_shape, value, vmap_method="sequential"
            )
            return result, None

        def reverse_rule(_, residual: jax.Array):
            return (
                jax.pure_callback(
                    self._host_rmatvec,
                    reverse_shape,
                    residual,
                    vmap_method="sequential",
                ),
            )

        forward.defvjp(forward_rule, reverse_rule)
        object.__setattr__(self, "_jax_forward", forward)

    @property
    def number_of_demand_cells(self) -> int:
        return self.source.number_of_demand_cells

    @property
    def number_of_measurements(self) -> int:
        return self.source.number_of_measurements

    @property
    def canonical_index_fingerprint(self) -> str:
        return self.source.canonical_index_fingerprint

    @property
    def artifact_fingerprint(self) -> str:
        return self.source.artifact_fingerprint

    @property
    def fixed_measurement_offset(self) -> np.ndarray:
        return self.source.fixed_measurement_offset

    @property
    def dtype(self) -> np.dtype:
        return self.source.dtype

    @property
    def representation(self) -> str:
        return "temporal_csr_csc"

    def _host_matvec(self, demand: object) -> np.ndarray:
        value = np.asarray(demand, dtype=self.dtype)
        if value.shape != (self.number_of_demand_cells,):
            raise ValueError("demand has an incompatible shape.")
        return np.asarray(self._csr @ value, dtype=self.dtype)

    def _host_rmatvec(self, residual: object) -> np.ndarray:
        value = np.asarray(residual, dtype=self.dtype)
        if value.shape != (self.number_of_measurements,):
            raise ValueError("residual has an incompatible shape.")
        return np.asarray(self._csc.T @ value, dtype=self.dtype)

    def matvec(self, demand: object) -> jax.Array:
        value = jnp.asarray(demand, dtype=self.dtype)
        if value.shape != (self.number_of_demand_cells,):
            raise ValueError("demand has an incompatible shape.")
        return self._jax_forward(value)

    def rmatvec(self, residual: object) -> jax.Array:
        value = jnp.asarray(residual, dtype=self.dtype)
        if value.shape != (self.number_of_measurements,):
            raise ValueError("residual has an incompatible shape.")
        output = jax.ShapeDtypeStruct((self.number_of_demand_cells,), self.dtype)
        return jax.pure_callback(
            self._host_rmatvec, output, value, vmap_method="sequential"
        )

    def jax_matvec(self, demand: object) -> jax.Array:
        return self.matvec(demand)

    def jax_rmatvec(self, residual: object) -> jax.Array:
        return self.rmatvec(residual)
