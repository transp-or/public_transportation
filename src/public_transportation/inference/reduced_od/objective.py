"""JAX demand generation and likelihood for minimal reduced gravity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.measurement.likelihood_jax import (
    negbinom_logpmf_mu_r,
    poisson_logpmf,
)

from .features import ConditionalGravityFeatures
from .parameters import (
    MinimalGravityParameterLayout,
    transform_minimal_gravity_parameters,
)
from .response_operator import ReducedResponseOperator


class MinimalGravityDemand(NamedTuple):
    demand: jax.Array
    probabilities: jax.Array
    utilities: jax.Array
    productions: jax.Array
    group_sums: jax.Array
    destination_attractiveness: jax.Array


@dataclass(frozen=True, slots=True)
class MinimalGravityProblem:
    features: ConditionalGravityFeatures
    parameter_layout: MinimalGravityParameterLayout
    response_operator: ReducedResponseOperator
    observations: np.ndarray
    production_basis: np.ndarray | None = None
    production_basis_labels: tuple[str, ...] | None = None
    destination_attractiveness_basis: np.ndarray | None = None
    destination_attractiveness_basis_labels: tuple[str, ...] | None = None
    detection_rate: float = 1.0
    calibration_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.features.number_of_cells != self.response_operator.number_of_free_cells:
            raise ValueError("gravity features must align with response free cells.")
        observations = np.array(self.observations, dtype=np.float64, copy=True)
        if observations.shape != (self.response_operator.number_of_measurements,):
            raise ValueError("observations have an invalid measurement dimension.")
        if not np.all(np.isfinite(observations)) or np.any(observations < 0.0):
            raise ValueError("observations must be finite and non-negative.")
        observations.setflags(write=False)
        object.__setattr__(self, "observations", observations)
        if self.calibration_mask is not None:
            mask = np.array(self.calibration_mask, dtype=np.bool_, copy=True)
            if mask.shape != observations.shape or not np.any(mask):
                raise ValueError(
                    "calibration_mask must align and select at least one measurement."
                )
            mask.setflags(write=False)
            object.__setattr__(self, "calibration_mask", mask)
        specification = self.parameter_layout.specification
        if specification.production_mode == "provided":
            if self.production_basis is not None:
                raise ValueError("provided production mode does not accept a basis.")
            if self.production_basis_labels is not None:
                raise ValueError(
                    "provided production mode does not accept basis labels."
                )
        else:
            if self.production_basis is None:
                raise ValueError("estimated production mode requires a basis.")
            basis = np.array(self.production_basis, dtype=np.float64, copy=True)
            expected = (
                self.features.number_of_origin_time_groups,
                specification.production_basis_columns,
            )
            if basis.shape != expected or not np.all(np.isfinite(basis)):
                raise ValueError("production_basis has an invalid shape or values.")
            basis.setflags(write=False)
            object.__setattr__(self, "production_basis", basis)
            if self.production_basis_labels is not None:
                labels = tuple(str(value) for value in self.production_basis_labels)
                if (
                    len(labels) != specification.production_basis_columns
                    or any(not value for value in labels)
                    or len(set(labels)) != len(labels)
                ):
                    raise ValueError(
                        "production_basis_labels must be unique and align with columns."
                    )
                object.__setattr__(self, "production_basis_labels", labels)
        if specification.destination_attractiveness_mode == "provided":
            if self.destination_attractiveness_basis is not None:
                raise ValueError(
                    "provided destination attractiveness does not accept a basis."
                )
            if self.destination_attractiveness_basis_labels is not None:
                raise ValueError(
                    "provided destination attractiveness does not accept basis labels."
                )
        else:
            if self.destination_attractiveness_basis is None:
                raise ValueError(
                    "estimated destination attractiveness requires a basis."
                )
            basis = np.array(
                self.destination_attractiveness_basis, dtype=np.float64, copy=True
            )
            expected = (
                len(self.features.destination_ids),
                specification.destination_attractiveness_basis_columns,
            )
            if basis.shape != expected or not np.all(np.isfinite(basis)):
                raise ValueError(
                    "destination_attractiveness_basis has an invalid shape or values."
                )
            basis.setflags(write=False)
            object.__setattr__(self, "destination_attractiveness_basis", basis)
            if self.destination_attractiveness_basis_labels is not None:
                labels = tuple(
                    str(value) for value in self.destination_attractiveness_basis_labels
                )
                if (
                    len(labels)
                    != specification.destination_attractiveness_basis_columns
                    or any(not value for value in labels)
                    or len(set(labels)) != len(labels)
                ):
                    raise ValueError(
                        "destination-attractiveness basis labels must be unique and align with columns."
                    )
                object.__setattr__(
                    self, "destination_attractiveness_basis_labels", labels
                )
        if not np.isfinite(self.detection_rate) or self.detection_rate <= 0.0:
            raise ValueError("detection_rate must be finite and positive.")


class MinimalGravityEvaluation(NamedTuple):
    objective: jax.Array
    log_likelihood: jax.Array
    measurement_mean: jax.Array
    demand: jax.Array
    probabilities: jax.Array
    productions: jax.Array


def generate_minimal_gravity_demand(
    raw_parameters: object,
    *,
    problem: MinimalGravityProblem,
) -> MinimalGravityDemand:
    raw = jnp.asarray(raw_parameters)
    parameters = transform_minimal_gravity_parameters(
        raw, layout=problem.parameter_layout
    )
    features = problem.features
    groups = jnp.asarray(features.origin_time_group_index, dtype=jnp.int32)
    time = jnp.asarray(features.journey_time_seconds, dtype=raw.dtype)
    transfers = jnp.asarray(features.transfer_count, dtype=raw.dtype)
    attraction = jnp.asarray(features.destination_attractiveness, dtype=raw.dtype)
    if (
        problem.parameter_layout.specification.destination_attractiveness_mode
        == "estimated_basis"
    ):
        assert problem.destination_attractiveness_basis is not None
        destination_basis = jnp.asarray(
            problem.destination_attractiveness_basis, dtype=raw.dtype
        )
        destination_coefficients = parameters.destination_attractiveness_coefficients
        destination_effect = destination_basis @ destination_coefficients
        attraction = attraction * jnp.exp(destination_effect)
    utilities = (
        jnp.log(attraction)
        - parameters.beta_time
        * time
        / jnp.asarray(
            problem.parameter_layout.specification.journey_time_scale_seconds,
            dtype=raw.dtype,
        )
        - parameters.beta_transfer * transfers
    )
    groups_count = features.number_of_origin_time_groups
    maximum = jax.ops.segment_max(utilities, groups, num_segments=groups_count)
    weights = jnp.exp(utilities - maximum[groups])
    denominator = jax.ops.segment_sum(weights, groups, num_segments=groups_count)
    probabilities = weights / denominator[groups]
    baseline = jnp.asarray(features.baseline_productions, dtype=raw.dtype)
    if problem.parameter_layout.specification.production_mode == "provided":
        productions = baseline
    else:
        assert problem.production_basis is not None
        basis = jnp.asarray(problem.production_basis, dtype=raw.dtype)
        productions = baseline * jnp.exp(basis @ parameters.production_coefficients)
    demand = probabilities * productions[groups]
    group_sums = jax.ops.segment_sum(demand, groups, num_segments=groups_count)
    return MinimalGravityDemand(
        demand=demand,
        probabilities=probabilities,
        utilities=utilities,
        productions=productions,
        group_sums=group_sums,
        destination_attractiveness=attraction,
    )


def evaluate_minimal_gravity_objective(
    raw_parameters: object,
    *,
    problem: MinimalGravityProblem,
) -> MinimalGravityEvaluation:
    raw = jnp.asarray(raw_parameters)
    generated = generate_minimal_gravity_demand(raw, problem=problem)
    routed = problem.response_operator.jax_matvec(generated.demand)
    offset = jnp.asarray(
        problem.response_operator.fixed_offset, dtype=generated.demand.dtype
    )
    mean = jnp.asarray(problem.detection_rate, dtype=generated.demand.dtype) * (
        routed + offset
    )
    mean = jnp.maximum(
        mean,
        jnp.asarray(
            problem.parameter_layout.specification.mean_floor, dtype=mean.dtype
        ),
    )
    observations = jnp.asarray(problem.observations, dtype=mean.dtype)
    if problem.parameter_layout.specification.likelihood == "poisson":
        contributions = poisson_logpmf(observations, mean)
    else:
        parameters = transform_minimal_gravity_parameters(
            raw, layout=problem.parameter_layout
        )
        assert parameters.dispersion is not None
        contributions = negbinom_logpmf_mu_r(observations, mean, parameters.dispersion)
    if problem.calibration_mask is None:
        log_likelihood = jnp.sum(contributions)
    else:
        mask = jnp.asarray(problem.calibration_mask)
        log_likelihood = jnp.sum(jnp.where(mask, contributions, 0.0))
    return MinimalGravityEvaluation(
        objective=-log_likelihood,
        log_likelihood=log_likelihood,
        measurement_mean=mean,
        demand=generated.demand,
        probabilities=generated.probabilities,
        productions=generated.productions,
    )


def minimal_gravity_value_and_gradient(
    raw_parameters: object,
    *,
    problem: MinimalGravityProblem,
) -> tuple[MinimalGravityEvaluation, jax.Array]:
    raw = jnp.asarray(raw_parameters)
    evaluation = evaluate_minimal_gravity_objective(raw, problem=problem)
    gradient = jax.grad(
        lambda value: (
            evaluate_minimal_gravity_objective(value, problem=problem).objective
        )
    )(raw)
    return evaluation, gradient
