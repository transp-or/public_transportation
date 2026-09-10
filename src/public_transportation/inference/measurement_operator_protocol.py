"""Common bounded measurement-operator contract for gravity estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax


@dataclass(frozen=True, slots=True)
class GravityOperatorCapabilities:
    """Operational features that do not alter numerical operator identity."""

    progress: bool = False
    absolute_deadline: bool = False
    cancellation: bool = False
    resident_cache_diagnostics: bool = False
    batched_shards: bool = False
    concurrent_shards: bool = False
    matmat: bool = True


class GravityOperatorMetrics(Protocol):
    @property
    def stored_bytes(self) -> int: ...

    @property
    def peak_construction_bytes(self) -> int: ...


@runtime_checkable
class GravityMeasurementOperator(Protocol):
    """Numerical and operational products required by gravity estimation."""

    @property
    def num_free_od(self) -> int: ...

    @property
    def num_measurements(self) -> int: ...

    @property
    def compact_layout_fingerprint(self) -> str | None: ...

    @property
    def fixed_measurement_offset(self) -> object: ...

    @property
    def representation(self) -> str: ...

    @property
    def is_matrix_free(self) -> bool: ...

    @property
    def assignment_fingerprint(self) -> str: ...

    @property
    def graph_fingerprint(self) -> str: ...

    @property
    def mapping_fingerprint(self) -> str: ...

    @property
    def theta(self) -> float: ...

    @property
    def dtype(self) -> object: ...

    @property
    def metrics(self) -> GravityOperatorMetrics: ...

    @property
    def product_capabilities(self) -> GravityOperatorCapabilities: ...

    def jax_matvec(self, vector: jax.Array) -> jax.Array: ...

    def jax_rmatvec(self, vector: jax.Array) -> jax.Array: ...

    def jax_matmat(self, matrix: jax.Array) -> jax.Array: ...


@runtime_checkable
class GravityLinearMeasurementOperator(Protocol):
    """Minimal linear measurement contract for non-OD additive blocks.

    Unlike :class:`GravityMeasurementOperator`, this protocol deliberately
    carries no routing, OD-layout, or assignment provenance.  It is suitable
    for persisted companion matrices such as boundary-flow operators.
    """

    @property
    def num_rows(self) -> int: ...

    @property
    def num_columns(self) -> int: ...

    def jax_matvec(self, vector: jax.Array) -> jax.Array: ...

    def jax_rmatvec(self, vector: jax.Array) -> jax.Array: ...

    def jax_matmat(self, matrix: jax.Array) -> jax.Array: ...


@dataclass(frozen=True, slots=True)
class GravityMeasurementOperatorLinearAdapter:
    """Expose an existing OD operator through the generic linear protocol."""

    operator: GravityMeasurementOperator

    @property
    def num_rows(self) -> int:
        return int(self.operator.num_measurements)

    @property
    def num_columns(self) -> int:
        return int(self.operator.num_free_od)

    @property
    def fingerprint(self) -> str:
        value = getattr(self.operator, "artifact_fingerprint", None)
        if value is None:
            value = getattr(self.operator, "assignment_fingerprint", "")
        return str(value)

    def jax_matvec(self, vector: jax.Array) -> jax.Array:
        return self.operator.jax_matvec(vector)

    def jax_rmatvec(self, vector: jax.Array) -> jax.Array:
        return self.operator.jax_rmatvec(vector)

    def jax_matmat(self, matrix: jax.Array) -> jax.Array:
        return self.operator.jax_matmat(matrix)
