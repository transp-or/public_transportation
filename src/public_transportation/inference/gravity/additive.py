"""Generic additive latent-flow blocks for gravity measurement models.

The classes in this module are intentionally independent of public-transport
routing.  A case adapter can therefore provide a boundary-flow matrix without
pretending that it is an OD assignment operator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.inference.block_coordinate._canonical import fingerprint
from public_transportation.inference.measurement_operator_protocol import (
    GravityLinearMeasurementOperator,
)


@dataclass(frozen=True, slots=True)
class GravityDenseLinearMeasurementOperator:
    """Small dense implementation of the generic linear operator contract."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix)
        if matrix.ndim != 2:
            raise ValueError("linear measurement matrix must be two-dimensional.")
        if matrix.dtype.kind not in "fiu" or not np.all(np.isfinite(matrix)):
            raise ValueError("linear measurement matrix must be finite and numeric.")
        matrix = np.asarray(matrix, dtype=np.result_type(matrix.dtype, np.float32))
        matrix = np.array(matrix, copy=True)
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @property
    def num_rows(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def num_columns(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def dtype(self) -> np.dtype:
        return self.matrix.dtype

    @property
    def support_mask(self) -> np.ndarray:
        """Rows with at least one nonzero coefficient."""
        return np.any(self.matrix != 0, axis=1)

    @property
    def fingerprint(self) -> str:
        return fingerprint(
            {
                "schema_version": 1,
                "matrix": self.matrix,
                "dtype": str(self.matrix.dtype),
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "dense",
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "dtype": str(self.matrix.dtype),
            "fingerprint": self.fingerprint,
        }

    def jax_matvec(self, vector: jax.Array) -> jax.Array:
        vector = jnp.asarray(vector)
        return jnp.asarray(self.matrix, dtype=vector.dtype) @ vector

    def jax_rmatvec(self, vector: jax.Array) -> jax.Array:
        vector = jnp.asarray(vector)
        return jnp.asarray(self.matrix, dtype=vector.dtype).T @ vector

    def jax_matmat(self, matrix: jax.Array) -> jax.Array:
        matrix = jnp.asarray(matrix)
        return jnp.asarray(self.matrix, dtype=matrix.dtype) @ matrix


@runtime_checkable
class GravityLatentFlowModel(Protocol):
    """Protocol implemented by models mapping raw parameters to flows."""

    @property
    def num_parameters(self) -> int: ...

    @property
    def num_flows(self) -> int: ...

    @property
    def parameter_names(self) -> tuple[str, ...]: ...

    def jax_flow(self, raw_parameters: jax.Array) -> jax.Array: ...

    def physical_parameters(self, raw_parameters: object) -> np.ndarray: ...

    def raw_from_physical(self, physical_parameters: object) -> np.ndarray: ...

    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class DirectNonnegativeFlowModel:
    """One positive raw parameter per latent flow."""

    num_flows: int
    prefix: str = "flow"
    positivity_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.num_flows <= 0:
            raise ValueError("num_flows must be positive.")
        if not self.prefix:
            raise ValueError("prefix must be non-empty.")
        if not np.isfinite(self.positivity_floor) or self.positivity_floor <= 0:
            raise ValueError("positivity_floor must be finite and positive.")

    @property
    def num_parameters(self) -> int:
        return self.num_flows

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"{self.prefix}[{index}]" for index in range(self.num_flows))

    def jax_flow(self, raw_parameters: jax.Array) -> jax.Array:
        raw = jnp.asarray(raw_parameters)
        if raw.ndim != 1 or raw.shape[0] != self.num_parameters:
            raise ValueError(
                f"flow parameters must have shape ({self.num_parameters},)."
            )
        return jax.nn.softplus(raw) + jnp.asarray(
            self.positivity_floor, dtype=raw.dtype
        )

    def physical_parameters(self, raw_parameters: object) -> np.ndarray:
        raw = np.asarray(raw_parameters, dtype=np.float64)
        if raw.shape != (self.num_parameters,) or not np.all(np.isfinite(raw)):
            raise ValueError("flow raw parameters have the wrong shape or values.")
        return np.logaddexp(0.0, raw) + self.positivity_floor

    def raw_from_physical(self, physical_parameters: object) -> np.ndarray:
        physical = np.asarray(physical_parameters, dtype=np.float64)
        if physical.shape != (self.num_flows,) or not np.all(np.isfinite(physical)):
            raise ValueError("flow physical parameters have the wrong shape or values.")
        shifted = physical - self.positivity_floor
        if np.any(shifted <= 0):
            raise ValueError("flow physical parameters must exceed positivity_floor.")
        return shifted + np.log(-np.expm1(-shifted))

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "direct_nonnegative",
            "num_flows": self.num_flows,
            "prefix": self.prefix,
            "positivity_floor": self.positivity_floor,
            "parameter_names": list(self.parameter_names),
        }


@dataclass(frozen=True, slots=True)
class LinearNonnegativeFlowModel:
    """Low-dimensional nonnegative flows generated from a fixed basis."""

    basis: np.ndarray
    prefix: str = "flow_factor"
    positivity_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis)
        if basis.ndim != 2 or basis.shape[0] <= 0 or basis.shape[1] <= 0:
            raise ValueError("basis must be a non-empty two-dimensional matrix.")
        if basis.dtype.kind not in "fiu" or not np.all(np.isfinite(basis)):
            raise ValueError("basis must be finite and numeric.")
        basis = np.asarray(basis, dtype=np.result_type(basis.dtype, np.float32))
        basis = np.array(basis, copy=True)
        basis.setflags(write=False)
        object.__setattr__(self, "basis", basis)
        if not self.prefix:
            raise ValueError("prefix must be non-empty.")
        if not np.isfinite(self.positivity_floor) or self.positivity_floor <= 0:
            raise ValueError("positivity_floor must be finite and positive.")

    @property
    def num_flows(self) -> int:
        return int(self.basis.shape[0])

    @property
    def num_parameters(self) -> int:
        return int(self.basis.shape[1])

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"{self.prefix}[{index}]" for index in range(self.num_parameters))

    def jax_flow(self, raw_parameters: jax.Array) -> jax.Array:
        raw = jnp.asarray(raw_parameters)
        if raw.ndim != 1 or raw.shape[0] != self.num_parameters:
            raise ValueError(
                f"flow parameters must have shape ({self.num_parameters},)."
            )
        basis = jnp.asarray(self.basis, dtype=raw.dtype)
        return jax.nn.softplus(basis @ raw) + jnp.asarray(
            self.positivity_floor, dtype=raw.dtype
        )

    def physical_parameters(self, raw_parameters: object) -> np.ndarray:
        raw = np.asarray(raw_parameters, dtype=np.float64)
        if raw.shape != (self.num_parameters,) or not np.all(np.isfinite(raw)):
            raise ValueError("flow raw parameters have the wrong shape or values.")
        return raw.copy()

    def raw_from_physical(self, physical_parameters: object) -> np.ndarray:
        raw = np.asarray(physical_parameters, dtype=np.float64)
        if raw.shape != (self.num_parameters,) or not np.all(np.isfinite(raw)):
            raise ValueError("flow physical parameters have the wrong shape or values.")
        return raw.copy()

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "linear_nonnegative",
            # Keep the specification directly JSON-serializable.  The same
            # array is converted back to a numeric matrix by
            # ``latent_flow_model_from_dict`` when a layout is restored.
            "basis": self.basis.tolist(),
            "basis_fingerprint": fingerprint(self.basis),
            "prefix": self.prefix,
            "positivity_floor": self.positivity_floor,
            "parameter_names": list(self.parameter_names),
        }


def latent_flow_model_from_dict(
    payload: Mapping[str, object],
) -> GravityLatentFlowModel:
    kind = str(payload.get("type", ""))
    if kind == "direct_nonnegative":
        return DirectNonnegativeFlowModel(
            num_flows=int(payload["num_flows"]),
            prefix=str(payload.get("prefix", "flow")),
            positivity_floor=float(payload.get("positivity_floor", 1.0e-6)),
        )
    if kind == "linear_nonnegative":
        basis = np.asarray(payload.get("basis"))
        model = LinearNonnegativeFlowModel(
            basis=basis,
            prefix=str(payload.get("prefix", "flow_factor")),
            positivity_floor=float(payload.get("positivity_floor", 1.0e-6)),
        )
        if payload.get("basis_fingerprint") != fingerprint(model.basis):
            raise ValueError("latent-flow basis fingerprint mismatch.")
        return model
    raise ValueError(f"unsupported latent-flow model type {kind!r}.")


@dataclass(frozen=True, slots=True)
class GravityAdditiveFlowBlock:
    """A named latent-flow vector and its linear measurement operator."""

    name: str
    operator: GravityLinearMeasurementOperator
    latent_flow_model: GravityLatentFlowModel
    regularization_strength: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("additive flow block name must be non-empty.")
        if not isinstance(self.operator, GravityLinearMeasurementOperator):
            raise TypeError("additive block operator does not implement the linear protocol.")
        if not isinstance(self.latent_flow_model, GravityLatentFlowModel):
            raise TypeError("latent_flow_model does not implement the flow protocol.")
        if self.operator.num_columns != self.latent_flow_model.num_flows:
            raise ValueError(
                "additive block operator columns must equal the latent-flow dimension."
            )
        if self.operator.num_rows <= 0:
            raise ValueError("additive block operator must have at least one row.")
        if not np.isfinite(self.regularization_strength) or self.regularization_strength < 0:
            raise ValueError("regularization_strength must be finite and non-negative.")

    @property
    def num_parameters(self) -> int:
        return self.latent_flow_model.num_parameters

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(f"{self.name}.{name}" for name in self.latent_flow_model.parameter_names)

    @property
    def operator_fingerprint(self) -> str:
        value = getattr(self.operator, "fingerprint", None)
        if value is None:
            value = fingerprint(
                {
                    "schema_version": 1,
                    "type": type(self.operator).__qualname__,
                    "num_rows": self.operator.num_rows,
                    "num_columns": self.operator.num_columns,
                }
            )
        return str(value)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def flow_from_raw(self, raw_parameters: object) -> jax.Array:
        return self.latent_flow_model.jax_flow(jnp.asarray(raw_parameters))

    def regularization(self, raw_parameters: object) -> jax.Array:
        raw = jnp.asarray(raw_parameters)
        return 0.5 * jnp.asarray(self.regularization_strength, dtype=raw.dtype) * jnp.sum(
            raw * raw
        )

    def to_dict(self) -> dict[str, object]:
        operator_payload = getattr(self.operator, "to_dict", None)
        if callable(operator_payload):
            operator_payload = operator_payload()
        else:
            operator_payload = {
                "type": type(self.operator).__qualname__,
                "num_rows": self.operator.num_rows,
                "num_columns": self.operator.num_columns,
                "fingerprint": self.operator_fingerprint,
            }
        return {
            "name": self.name,
            "operator": operator_payload,
            "operator_fingerprint": self.operator_fingerprint,
            "latent_flow_model": self.latent_flow_model.to_dict(),
            "regularization_strength": float(self.regularization_strength),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        operator: GravityLinearMeasurementOperator,
    ) -> "GravityAdditiveFlowBlock":
        model_payload = payload.get("latent_flow_model")
        if not isinstance(model_payload, Mapping):
            raise ValueError("additive block latent_flow_model is missing or invalid.")
        block = cls(
            name=str(payload["name"]),
            operator=operator,
            latent_flow_model=latent_flow_model_from_dict(model_payload),
            regularization_strength=float(payload.get("regularization_strength", 0.0)),
        )
        if payload.get("operator_fingerprint") != block.operator_fingerprint:
            raise ValueError("additive block operator fingerprint mismatch.")
        return block
