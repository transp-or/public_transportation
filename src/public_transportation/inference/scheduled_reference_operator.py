"""Authoritative scheduled time-expanded assignment reference operator."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.measurement.likelihood_jax import (
    predict_measurements_from_link_flow,
)
from public_transportation.measurement.mapping import AggregationSpec

from .assignment_adapter import AssignmentInputs, assign_link_flow
from .assignment_contract import (
    AssignmentArtifactIdentity,
    AssignmentCompatibilityError,
    CanonicalAssignmentIndex,
    fixed_routing_route_choice_fingerprint,
)
from .fixed_routing_measurement_operator import (
    assignment_inputs_fingerprint,
    measurement_mapping_fingerprint,
)


@dataclass(frozen=True, slots=True)
class ScheduledTimeExpandedReferenceOperator:
    """Exact, deliberately unoptimized scheduled-assignment validation backend.

    Each product traverses the time-expanded assignment calculation.  This
    operator is therefore a numerical authority for small validation cases,
    not a production estimation backend.
    """

    inputs: AssignmentInputs
    spec: AggregationSpec
    canonical_index: CanonicalAssignmentIndex
    theta: float
    identity: AssignmentArtifactIdentity
    _free_active_indices: tuple[int, ...] = field(init=False, repr=False)
    _fixed_active_indices: tuple[int, ...] = field(init=False, repr=False)
    _fixed_active_values: tuple[float, ...] = field(init=False, repr=False)
    fixed_measurement_offset: jax.Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        theta = float(self.theta)
        if not np.isfinite(theta) or theta <= 0.0:
            raise ValueError("theta must be positive and finite.")
        object.__setattr__(self, "theta", theta)
        if (
            self.identity.canonical_index_fingerprint
            != self.canonical_index.artifact_fingerprint
        ):
            raise AssignmentCompatibilityError(
                "scheduled reference identity and canonical index differ."
            )
        assignment_fingerprint = assignment_inputs_fingerprint(self.inputs)
        if self.identity.network_fingerprint != assignment_fingerprint:
            raise AssignmentCompatibilityError(
                "scheduled reference network fingerprint is incompatible."
            )
        if self.identity.timetable_fingerprint != assignment_fingerprint:
            raise AssignmentCompatibilityError(
                "scheduled reference timetable fingerprint is incompatible."
            )
        if self.identity.route_choice_fingerprint != fixed_routing_route_choice_fingerprint(
            theta
        ):
            raise AssignmentCompatibilityError(
                "scheduled reference route-choice fingerprint is incompatible."
            )
        if (
            self.identity.measurement_mapping_fingerprint
            != measurement_mapping_fingerprint(self.spec)
        ):
            raise AssignmentCompatibilityError(
                "scheduled reference measurement mapping is incompatible."
            )
        dtype = np.dtype(self.inputs.base_link_cost.dtype)
        if dtype.name != np.dtype(self.identity.numeric_dtype).name:
            raise AssignmentCompatibilityError(
                "scheduled reference numeric dtype is incompatible."
            )
        if self.spec.num_measurements != self.canonical_index.number_of_measurements:
            raise AssignmentCompatibilityError(
                "scheduled reference measurement dimension is incompatible."
            )

        free_active_indices = []
        fixed_active_indices = []
        fixed_active_values = []
        active_index = 0
        for cell in self.canonical_index.demand_cells:
            if cell.role == "fixed_zero":
                continue
            if cell.role == "free":
                free_active_indices.append(active_index)
            else:
                fixed_active_indices.append(active_index)
                assert cell.fixed_value is not None
                fixed_active_values.append(cell.fixed_value)
            active_index += 1
        if active_index != int(self.inputs.od_origin_node.shape[0]):
            raise AssignmentCompatibilityError(
                "scheduled assignment inputs and canonical active-cell layout differ."
            )
        object.__setattr__(self, "_free_active_indices", tuple(free_active_indices))
        object.__setattr__(self, "_fixed_active_indices", tuple(fixed_active_indices))
        object.__setattr__(self, "_fixed_active_values", tuple(fixed_active_values))

        fixed_active = jnp.zeros((active_index,), dtype=dtype)
        if fixed_active_indices:
            fixed_active = fixed_active.at[jnp.asarray(fixed_active_indices)].set(
                jnp.asarray(fixed_active_values, dtype=dtype)
            )
        object.__setattr__(
            self,
            "fixed_measurement_offset",
            self._active_forward(fixed_active),
        )

    @property
    def number_of_demand_cells(self) -> int:
        return self.canonical_index.number_of_demand_cells

    @property
    def number_of_measurements(self) -> int:
        return self.canonical_index.number_of_measurements

    @property
    def num_free_od(self) -> int:
        return self.number_of_demand_cells

    @property
    def num_measurements(self) -> int:
        return self.number_of_measurements

    @property
    def canonical_index_fingerprint(self) -> str:
        return self.canonical_index.artifact_fingerprint

    @property
    def artifact_fingerprint(self) -> str:
        return self.identity.fingerprint

    @property
    def assignment_fingerprint(self) -> str:
        return assignment_inputs_fingerprint(self.inputs)

    @property
    def graph_fingerprint(self) -> str:
        return assignment_inputs_fingerprint(self.inputs)

    @property
    def mapping_fingerprint(self) -> str:
        return measurement_mapping_fingerprint(self.spec)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.inputs.base_link_cost.dtype)

    @property
    def representation(self) -> str:
        return "scheduled_time_expanded_reference"

    def _active_forward(self, active_demand: jax.Array) -> jax.Array:
        link_flow = assign_link_flow(
            inputs=self.inputs,
            f=active_demand,
            theta=jnp.asarray(self.theta, dtype=self.dtype),
        )
        return predict_measurements_from_link_flow(
            link_flow=link_flow,
            spec_num_measurements=self.spec.num_measurements,
            spec_measurement_index=jnp.asarray(
                self.spec.measurement_index, dtype=jnp.int32
            ),
            spec_link_index=jnp.asarray(self.spec.link_index, dtype=jnp.int32),
        )

    def _free_forward(self, demand: jax.Array) -> jax.Array:
        active = jnp.zeros((self.inputs.od_origin_node.shape[0],), dtype=self.dtype)
        if self._free_active_indices:
            active = active.at[jnp.asarray(self._free_active_indices)].set(demand)
        return self._active_forward(active)

    def matvec(self, demand: object) -> jax.Array:
        """Calculate the scheduled forward product for free demand cells."""
        value = jnp.asarray(demand, dtype=self.dtype)
        if value.shape != (self.number_of_demand_cells,):
            raise ValueError(
                "demand must have shape "
                f"({self.number_of_demand_cells},), got {value.shape}."
            )
        return self._free_forward(value)

    def rmatvec(self, residual: object) -> jax.Array:
        """Calculate the exact scheduled adjoint by reverse-mode differentiation."""
        value = jnp.asarray(residual, dtype=self.dtype)
        if value.shape != (self.number_of_measurements,):
            raise ValueError(
                "residual must have shape "
                f"({self.number_of_measurements},), got {value.shape}."
            )
        zero = jnp.zeros((self.number_of_demand_cells,), dtype=self.dtype)
        _, pullback = jax.vjp(self._free_forward, zero)
        return pullback(value)[0]

    def jax_matvec(self, demand: object) -> jax.Array:
        return self.matvec(demand)

    def jax_rmatvec(self, residual: object) -> jax.Array:
        return self.rmatvec(residual)


def build_scheduled_reference_artifact_identity(
    *,
    inputs: AssignmentInputs,
    spec: AggregationSpec,
    canonical_index: CanonicalAssignmentIndex,
    theta: float,
    temporal_discretization_fingerprint: str,
    departure_choice_fingerprint: str,
    feasibility_fingerprint: str,
    coefficient_policy_fingerprint: str,
) -> AssignmentArtifactIdentity:
    """Build the dependency identity for the scheduled validation backend."""
    assignment_fingerprint = assignment_inputs_fingerprint(inputs)
    return AssignmentArtifactIdentity(
        canonical_index_fingerprint=canonical_index.artifact_fingerprint,
        network_fingerprint=assignment_fingerprint,
        timetable_fingerprint=assignment_fingerprint,
        temporal_discretization_fingerprint=temporal_discretization_fingerprint,
        route_choice_fingerprint=fixed_routing_route_choice_fingerprint(theta),
        departure_choice_fingerprint=departure_choice_fingerprint,
        feasibility_fingerprint=feasibility_fingerprint,
        measurement_mapping_fingerprint=measurement_mapping_fingerprint(spec),
        coefficient_policy_fingerprint=coefficient_policy_fingerprint,
        numeric_dtype=np.dtype(inputs.base_link_cost.dtype).name,
    )
