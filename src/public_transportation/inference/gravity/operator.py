"""Device-native measurement-operator contract used by gravity estimation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax


class GravityOperatorMetrics(Protocol):
    @property
    def stored_bytes(self) -> int: ...

    @property
    def peak_construction_bytes(self) -> int: ...


@runtime_checkable
class GravityMeasurementOperator(Protocol):
    """The bounded JAX products required by gravity objectives and gradients."""

    @property
    def num_free_od(self) -> int: ...

    @property
    def num_measurements(self) -> int: ...

    @property
    def compact_layout_fingerprint(self) -> str | None: ...

    @property
    def fixed_measurement_offset(self) -> object: ...

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

    def jax_matvec(self, vector: jax.Array) -> jax.Array: ...

    def jax_rmatvec(self, vector: jax.Array) -> jax.Array: ...

    def jax_matmat(self, matrix: jax.Array) -> jax.Array: ...
