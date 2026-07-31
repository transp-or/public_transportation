"""Validated data contracts for fixed-routing linear OD estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import sparse

from .linear_operator import (
    DenseLinearOperator,
    LinearOperatorProtocol,
    SparseLinearOperator,
    as_linear_operator,
)

if TYPE_CHECKING:  # pragma: no cover
    from .fixed_routing_linear_backend import PreparedLinearMeasurementBackend
    from .fixed_routing_measurement_operator import FixedRoutingMeasurementOperator

Array = np.ndarray
RegularizationSelection = Literal["unspecified", "none", "configured"]


def _immutable_float_array(value: object, *, name: str) -> Array:
    """Return an owned, read-only real floating-point array."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")
    if array.dtype.kind in "iu":
        array = array.astype(np.float64)
    else:
        array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class FixedRoutingLinearProvenance:
    """Identity of the layout, routing, assignment, and measurement mapping."""

    od_layout_fingerprint: str
    assignment_fingerprint: str
    mapping_fingerprint: str
    routing_parameter: float

    def __post_init__(self) -> None:
        for field_name in (
            "od_layout_fingerprint",
            "assignment_fingerprint",
            "mapping_fingerprint",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be nonempty.")
        if not math.isfinite(self.routing_parameter) or self.routing_parameter <= 0.0:
            raise ValueError("routing_parameter must be finite and strictly positive.")


@dataclass(frozen=True, slots=True)
class LinearRegularizationBlock:
    """Declarative linear residual block ``sqrt(strength) * (L x - target)``."""

    name: str
    operator: LinearOperatorProtocol | object
    target: Array
    strength: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("regularization block name must be nonempty.")
        operator = as_linear_operator(self.operator)
        target = _immutable_float_array(self.target, name="target")
        if target.ndim != 1 or target.shape[0] != operator.shape[0]:
            raise ValueError(
                "regularization target must have shape "
                f"({operator.shape[0]},), got {target.shape}."
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("regularization target must be finite.")
        if not math.isfinite(self.strength) or self.strength < 0.0:
            raise ValueError("regularization strength must be finite and non-negative.")
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "target", target)


@dataclass(frozen=True, slots=True)
class FixedRoutingLinearProblem:
    """Complete solver-independent contract for a small dense linear problem.

    ``measurement_operator`` maps free physical OD demand to measurements.
    ``fixed_measurement_offset`` contains the complete contribution of positive
    fixed demand. All vector-valued fields use canonical OD or measurement
    ordering as identified by ``provenance``.

    The operator may be dense, sparse, or any validated implementation of the
    forward/transpose protocol without changing the mathematical contract.
    """

    measurement_operator: LinearOperatorProtocol | object
    fixed_measurement_offset: Array
    observations: Array
    observation_weights: Array
    prior_demand: Array
    lower_bounds: Array
    upper_bounds: Array
    provenance: FixedRoutingLinearProvenance
    regularization_selection: RegularizationSelection = "unspecified"
    regularization_blocks: tuple[LinearRegularizationBlock, ...] = ()
    variable_scales: Array | None = None
    free_od_indices: Array | None = None

    def __post_init__(self) -> None:
        operator = as_linear_operator(self.measurement_operator)
        offset = _immutable_float_array(
            self.fixed_measurement_offset, name="fixed_measurement_offset"
        )
        observations = _immutable_float_array(self.observations, name="observations")
        weights = _immutable_float_array(
            self.observation_weights, name="observation_weights"
        )
        prior = _immutable_float_array(self.prior_demand, name="prior_demand")
        lower = _immutable_float_array(self.lower_bounds, name="lower_bounds")
        upper = _immutable_float_array(self.upper_bounds, name="upper_bounds")

        if len(operator.shape) != 2:
            raise ValueError("measurement_operator shape must be two-dimensional.")
        num_measurements, num_free_od = operator.shape
        if num_measurements == 0:
            raise ValueError("at least one observation is required.")
        if num_free_od == 0:
            raise ValueError("at least one free OD variable is required.")
        if isinstance(operator, DenseLinearOperator) and np.any(operator.matrix < 0.0):
            raise ValueError("measurement_operator must be non-negative.")
        if isinstance(operator, SparseLinearOperator) and np.any(
            operator.matrix.data < 0.0
        ):
            raise ValueError("measurement_operator must be non-negative.")

        measurement_fields = {
            "fixed_measurement_offset": offset,
            "observations": observations,
            "observation_weights": weights,
        }
        for name, value in measurement_fields.items():
            if value.ndim != 1 or value.shape[0] != num_measurements:
                raise ValueError(
                    f"{name} must have shape ({num_measurements},), got {value.shape}."
                )

        od_fields = {
            "prior_demand": prior,
            "lower_bounds": lower,
            "upper_bounds": upper,
        }
        for name, value in od_fields.items():
            if value.ndim != 1 or value.shape[0] != num_free_od:
                raise ValueError(
                    f"{name} must have shape ({num_free_od},), got {value.shape}."
                )

        if not np.all(np.isfinite(offset)) or np.any(offset < 0.0):
            raise ValueError(
                "fixed_measurement_offset must be finite and non-negative."
            )
        if not np.all(np.isfinite(observations)) or np.any(observations < 0.0):
            raise ValueError("observations must be finite and non-negative.")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError(
                "observation_weights must be finite and strictly positive."
            )
        if not np.all(np.isfinite(prior)) or np.any(prior < 0.0):
            raise ValueError("prior_demand must be finite and non-negative.")
        if np.any(np.isnan(lower)) or np.any(np.isposinf(lower)):
            raise ValueError("lower_bounds may be finite or -inf, but not NaN or +inf.")
        if np.any(np.isnan(upper)) or np.any(np.isneginf(upper)):
            raise ValueError("upper_bounds may be finite or +inf, but not NaN or -inf.")
        if np.any(lower > upper):
            raise ValueError("lower_bounds must not exceed upper_bounds.")
        if np.any(prior < lower) or np.any(prior > upper):
            raise ValueError("prior_demand must satisfy the physical demand bounds.")

        if self.variable_scales is None:
            scales = np.ones((num_free_od,), dtype=operator.dtype)
            scales.setflags(write=False)
        else:
            scales = _immutable_float_array(
                self.variable_scales, name="variable_scales"
            )
            if scales.ndim != 1 or scales.shape[0] != num_free_od:
                raise ValueError(
                    f"variable_scales must have shape ({num_free_od},), "
                    f"got {scales.shape}."
                )
            if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
                raise ValueError(
                    "variable_scales must be finite and strictly positive."
                )

        if self.free_od_indices is None:
            free_od_indices = np.arange(num_free_od, dtype=np.int64)
        else:
            free_od_indices = np.asarray(self.free_od_indices)
            if free_od_indices.ndim != 1 or free_od_indices.shape != (num_free_od,):
                raise ValueError(
                    f"free_od_indices must have shape ({num_free_od},), "
                    f"got {free_od_indices.shape}."
                )
            if free_od_indices.dtype.kind not in "iu":
                raise TypeError("free_od_indices must contain integers.")
            free_od_indices = np.array(free_od_indices, dtype=np.int64, copy=True)
            if np.any(free_od_indices < 0):
                raise ValueError("free_od_indices must be non-negative.")
            if np.unique(free_od_indices).size != num_free_od:
                raise ValueError("free_od_indices must not contain duplicates.")
        free_od_indices.setflags(write=False)

        if self.regularization_selection not in {
            "unspecified",
            "none",
            "configured",
        }:
            raise ValueError(
                "regularization_selection must be 'unspecified', 'none', "
                "or 'configured'."
            )
        blocks = tuple(self.regularization_blocks)
        if self.regularization_selection == "configured" and not blocks:
            raise ValueError("configured regularization requires at least one block.")
        if self.regularization_selection != "configured" and blocks:
            raise ValueError(
                "regularization blocks require regularization_selection='configured'."
            )
        for block in blocks:
            if not isinstance(block, LinearRegularizationBlock):
                raise TypeError(
                    "regularization_blocks must contain LinearRegularizationBlock "
                    "instances."
                )
            if block.operator.shape[1] != num_free_od:
                raise ValueError(
                    f"regularization block {block.name!r} must have "
                    f"{num_free_od} columns, got {block.operator.shape[1]}."
                )
        names = [block.name for block in blocks]
        if len(names) != len(set(names)):
            raise ValueError("regularization block names must be unique.")

        object.__setattr__(self, "measurement_operator", operator)
        object.__setattr__(self, "fixed_measurement_offset", offset)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "observation_weights", weights)
        object.__setattr__(self, "prior_demand", prior)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)
        object.__setattr__(self, "variable_scales", scales)
        object.__setattr__(self, "free_od_indices", free_od_indices)
        object.__setattr__(self, "regularization_blocks", blocks)

    @property
    def num_measurements(self) -> int:
        return int(self.measurement_operator.shape[0])

    @property
    def num_free_od(self) -> int:
        return int(self.measurement_operator.shape[1])


def build_fixed_routing_linear_problem_from_operator(
    *,
    operator: FixedRoutingMeasurementOperator,
    observations: object,
    observation_weights: object,
    prior_demand: object,
    lower_bounds: object,
    upper_bounds: object,
    regularization_selection: RegularizationSelection = "unspecified",
    regularization_blocks: tuple[LinearRegularizationBlock, ...] = (),
    variable_scales: object | None = None,
    free_od_indices: object | None = None,
) -> FixedRoutingLinearProblem:
    """Build a linear problem from a validated dense or native sparse operator.

    Operator construction remains the responsibility of
    :func:`prepare_fixed_routing_measurement_operator`. This boundary transfers
    its matrix, fixed-demand offset, dimensions, and provenance into the new
    solver-independent problem without recomputing assignment. A BCOO artifact
    is converted coordinate-for-coordinate to canonical CSR storage; it is
    never materialized as a dense array.
    """
    if operator.representation not in {"dense", "bcoo"}:
        raise ValueError("fixed-routing operator representation must be dense or bcoo.")
    if operator.od_layout_fingerprint is None:
        raise ValueError(
            "operator must include an OD layout fingerprint for linear estimation."
        )
    expected_shape = (
        operator.num_measurements,
        operator.num_free_od,
    )
    if operator.matrix.shape != expected_shape:
        raise ValueError("operator matrix shape disagrees with its metadata.")
    if np.asarray(operator.fixed_measurement_offset).shape != (
        operator.num_measurements,
    ):
        raise ValueError(
            "fixed measurement offset shape disagrees with operator metadata."
        )

    if operator.representation == "dense":
        measurement_operator: object = np.asarray(operator.matrix)
    else:
        data = np.asarray(operator.matrix.data)
        indices = np.asarray(operator.matrix.indices)
        if indices.ndim != 2 or indices.shape != (data.size, 2):
            raise ValueError("BCOO data and index arrays have inconsistent shapes.")
        measurement_operator = SparseLinearOperator(
            sparse.coo_array(
                (data, (indices[:, 0], indices[:, 1])),
                shape=expected_shape,
            )
        )

    provenance = FixedRoutingLinearProvenance(
        od_layout_fingerprint=operator.od_layout_fingerprint,
        assignment_fingerprint=operator.assignment_fingerprint,
        mapping_fingerprint=operator.mapping_fingerprint,
        routing_parameter=operator.theta,
    )
    return FixedRoutingLinearProblem(
        measurement_operator=measurement_operator,
        fixed_measurement_offset=np.asarray(operator.fixed_measurement_offset),
        observations=observations,
        observation_weights=observation_weights,
        prior_demand=prior_demand,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        provenance=provenance,
        regularization_selection=regularization_selection,
        regularization_blocks=regularization_blocks,
        variable_scales=variable_scales,
        free_od_indices=free_od_indices,
    )


def build_fixed_routing_linear_problem_from_dense_operator(
    *,
    operator: FixedRoutingMeasurementOperator,
    observations: object,
    observation_weights: object,
    prior_demand: object,
    lower_bounds: object,
    upper_bounds: object,
    regularization_selection: RegularizationSelection = "unspecified",
    regularization_blocks: tuple[LinearRegularizationBlock, ...] = (),
    variable_scales: object | None = None,
    free_od_indices: object | None = None,
) -> FixedRoutingLinearProblem:
    """Compatibility wrapper requiring the historical dense representation."""
    if operator.representation != "dense":
        raise ValueError("fixed-routing linear problem requires a dense operator.")
    return build_fixed_routing_linear_problem_from_operator(
        operator=operator,
        observations=observations,
        observation_weights=observation_weights,
        prior_demand=prior_demand,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        regularization_selection=regularization_selection,
        regularization_blocks=regularization_blocks,
        variable_scales=variable_scales,
        free_od_indices=free_od_indices,
    )


def build_fixed_routing_linear_problem_from_backend(
    *,
    backend: PreparedLinearMeasurementBackend,
    observations: object,
    observation_weights: object,
    prior_demand: object,
    lower_bounds: object,
    upper_bounds: object,
    provenance: FixedRoutingLinearProvenance,
    regularization_selection: RegularizationSelection = "unspecified",
    regularization_blocks: tuple[LinearRegularizationBlock, ...] = (),
    variable_scales: object | None = None,
    free_od_indices: object | None = None,
) -> FixedRoutingLinearProblem:
    """Build a problem without rebuilding or converting a prepared backend."""
    return FixedRoutingLinearProblem(
        measurement_operator=backend.operator,
        fixed_measurement_offset=backend.fixed_measurement_offset,
        observations=observations,
        observation_weights=observation_weights,
        prior_demand=prior_demand,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        provenance=provenance,
        regularization_selection=regularization_selection,
        regularization_blocks=regularization_blocks,
        variable_scales=variable_scales,
        free_od_indices=free_od_indices,
    )
