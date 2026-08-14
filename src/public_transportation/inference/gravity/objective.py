"""Phase-2 routing, likelihood, and gradient integration for gravity demand."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.measurement.likelihood_jax import (
    negbinom_logpmf_mu_r,
    poisson_logpmf,
)

from .demand import generate_gravity_demand
from .features import GravityFeatures
from .operator import GravityMeasurementOperator
from .parameters import GravityParameterLayout, validate_gravity_relaxation_features
from .specification import GravityEffectScope


class GravityLikelihood(str, Enum):
    POISSON = "poisson"
    NEGATIVE_BINOMIAL = "negative_binomial"


class GravityGradientStrategy(str, Enum):
    BATCHED_FORWARD = "batched_forward"
    ADJOINT = "adjoint"


@dataclass(frozen=True, slots=True)
class GravityObjectiveProblem:
    features: GravityFeatures
    parameter_layout: GravityParameterLayout
    operator: GravityMeasurementOperator
    observations: np.ndarray
    likelihood: GravityLikelihood = GravityLikelihood.NEGATIVE_BINOMIAL
    rho: float = 1.0
    calibration_mask: np.ndarray | None = None
    mean_floor: float = 1.0e-9

    def __post_init__(self) -> None:
        validate_gravity_relaxation_features(
            self.features, self.parameter_layout.specification
        )
        specification = self.parameter_layout.specification
        if specification.components and specification.likelihood.family != self.likelihood.value:
            raise ValueError(
                "objective likelihood does not match the declarative model specification."
            )
        if specification.components:
            mask_policy = specification.likelihood.calibration_mask
            if mask_policy == "explicit" and self.calibration_mask is None:
                raise ValueError(
                    "calibration-mask policy 'explicit' requires calibration_mask."
                )
            if mask_policy == "all_measurements" and self.calibration_mask is not None:
                supplied = np.asarray(self.calibration_mask, dtype=bool)
                if not np.all(supplied):
                    raise ValueError(
                        "calibration-mask policy 'all_measurements' cannot exclude rows."
                    )
        waiting = specification.component("waiting_time")
        if waiting.scope not in (
            GravityEffectScope.NONE,
            GravityEffectScope.FIXED,
        ) and self.features.initial_waiting_time is None:
            raise ValueError(
                "initial_waiting_time is required by the active waiting-time component."
            )
        if self.features.num_cells != self.operator.num_free_od:
            raise ValueError(
                "gravity cell count must equal the operator free-OD dimension."
            )
        if self.operator.compact_layout_fingerprint is not None and (
            self.features.od_layout_fingerprint
            != self.operator.compact_layout_fingerprint
        ):
            raise ValueError("gravity and operator compact-layout fingerprints differ.")
        observations = np.array(self.observations, copy=True)
        if (
            observations.ndim != 1
            or observations.shape[0] != self.operator.num_measurements
        ):
            raise ValueError("observations have the wrong measurement dimension.")
        if observations.dtype.kind not in "iuf" or not np.all(
            np.isfinite(observations)
        ):
            raise TypeError("observations must contain finite real values.")
        if np.any(observations < 0):
            raise ValueError("observations must be non-negative.")
        observations.setflags(write=False)
        object.__setattr__(self, "observations", observations)
        mask = (
            np.ones(observations.size, dtype=bool)
            if self.calibration_mask is None
            else np.array(self.calibration_mask, dtype=bool, copy=True)
        )
        if mask.shape != observations.shape:
            raise ValueError("calibration_mask must match observations.")
        if not np.any(mask):
            raise ValueError("calibration_mask must include at least one measurement.")
        mask.setflags(write=False)
        object.__setattr__(self, "calibration_mask", mask)
        if not np.isfinite(self.rho) or self.rho <= 0:
            raise ValueError("rho must be finite and positive.")
        if not np.isfinite(self.mean_floor) or self.mean_floor <= 0:
            raise ValueError("mean_floor must be finite and positive.")

    @property
    def calibration_measurements(self) -> int:
        mask = self.calibration_mask
        assert mask is not None
        return int(np.count_nonzero(mask))

    @property
    def excluded_measurements(self) -> int:
        return int(self.observations.size - self.calibration_measurements)


class GravityObjectiveEvaluation(NamedTuple):
    objective: jax.Array
    data_log_likelihood: jax.Array
    regularization: jax.Array
    measurement_mean: jax.Array
    demand: jax.Array
    calibration_measurements: jax.Array
    excluded_measurements: jax.Array


def predict_gravity_measurements(
    raw_parameters: object, *, problem: GravityObjectiveProblem
) -> tuple[jax.Array, jax.Array]:
    demand = generate_gravity_demand(
        raw_parameters,
        features=problem.features,
        parameter_layout=problem.parameter_layout,
    ).demand
    routed = problem.operator.jax_matvec(demand)
    offset = jnp.asarray(problem.operator.fixed_measurement_offset, dtype=demand.dtype)
    mean = jnp.asarray(problem.rho, dtype=demand.dtype) * (routed + offset)
    mean = jnp.maximum(mean, jnp.asarray(problem.mean_floor, dtype=demand.dtype))
    return mean, demand


def evaluate_gravity_objective(
    raw_parameters: object, *, problem: GravityObjectiveProblem
) -> GravityObjectiveEvaluation:
    raw = jnp.asarray(raw_parameters)
    mean, demand = predict_gravity_measurements(raw, problem=problem)
    return _evaluation_from_mean(raw, mean=mean, demand=demand, problem=problem)


def _evaluation_from_mean(
    raw: jax.Array,
    *,
    mean: jax.Array,
    demand: jax.Array,
    problem: GravityObjectiveProblem,
) -> GravityObjectiveEvaluation:
    observations = jnp.asarray(problem.observations, dtype=mean.dtype)
    mask = jnp.asarray(problem.calibration_mask)
    if problem.likelihood is GravityLikelihood.POISSON:
        contributions = poisson_logpmf(observations, mean)
    else:
        dispersion = problem.parameter_layout.transform(raw).dispersion
        contributions = negbinom_logpmf_mu_r(observations, mean, dispersion)
    data_log_likelihood = jnp.sum(jnp.where(mask, contributions, 0))
    regularization = problem.parameter_layout.regularization(raw)
    return GravityObjectiveEvaluation(
        objective=-data_log_likelihood + regularization,
        data_log_likelihood=data_log_likelihood,
        regularization=regularization,
        measurement_mean=mean,
        demand=demand,
        calibration_measurements=jnp.asarray(problem.calibration_measurements),
        excluded_measurements=jnp.asarray(problem.excluded_measurements),
    )


def _objective_scalar(raw: jax.Array, problem: GravityObjectiveProblem) -> jax.Array:
    return evaluate_gravity_objective(raw, problem=problem).objective


def gravity_value_and_gradient_batched_forward(
    raw_parameters: object, *, problem: GravityObjectiveProblem
) -> tuple[GravityObjectiveEvaluation, jax.Array]:
    """Small-parameter gradient using a batched demand Jacobian."""
    raw = jnp.asarray(raw_parameters)

    def demand_function(value: jax.Array) -> jax.Array:
        return generate_gravity_demand(
            value,
            features=problem.features,
            parameter_layout=problem.parameter_layout,
        ).demand

    demand = demand_function(raw)
    demand_jacobian = jax.jacfwd(demand_function)(raw)
    offset = jnp.asarray(problem.operator.fixed_measurement_offset, dtype=demand.dtype)
    rho = jnp.asarray(problem.rho, dtype=demand.dtype)
    mean_unfloored = rho * (problem.operator.jax_matvec(demand) + offset)
    mean = jnp.maximum(mean_unfloored, problem.mean_floor)
    active_mean = (mean_unfloored > problem.mean_floor).astype(mean.dtype)
    mean_jacobian = (
        active_mean[:, None]
        * rho
        * problem.operator.jax_matmat(demand_jacobian)
    )
    mean_gradient = jax.grad(lambda value: _objective_from_mean(value, raw, problem))(
        mean
    )
    direct_gradient = jax.grad(
        lambda parameters: _objective_from_mean(mean, parameters, problem)
    )(raw)
    gradient = mean_jacobian.T @ mean_gradient + direct_gradient
    return _evaluation_from_mean(
        raw, mean=mean, demand=demand, problem=problem
    ), gradient


def _objective_from_mean(
    mean: jax.Array, raw: jax.Array, problem: GravityObjectiveProblem
) -> jax.Array:
    observations = jnp.asarray(problem.observations, dtype=mean.dtype)
    mask = jnp.asarray(problem.calibration_mask)
    if problem.likelihood is GravityLikelihood.POISSON:
        contributions = poisson_logpmf(observations, mean)
    else:
        dispersion = problem.parameter_layout.transform(raw).dispersion
        contributions = negbinom_logpmf_mu_r(observations, mean, dispersion)
    return -jnp.sum(jnp.where(mask, contributions, 0)) + problem.parameter_layout.regularization(raw)


def gravity_value_and_gradient_adjoint(
    raw_parameters: object, *, problem: GravityObjectiveProblem
) -> tuple[GravityObjectiveEvaluation, jax.Array]:
    """Adjoint gradient using one routing transpose product and a demand VJP."""
    raw = jnp.asarray(raw_parameters)

    def demand_function(value: jax.Array) -> jax.Array:
        return generate_gravity_demand(
            value,
            features=problem.features,
            parameter_layout=problem.parameter_layout,
        ).demand

    demand, demand_pullback = jax.vjp(demand_function, raw)
    offset = jnp.asarray(problem.operator.fixed_measurement_offset, dtype=demand.dtype)
    rho = jnp.asarray(problem.rho, dtype=demand.dtype)
    mean_unfloored = rho * (problem.operator.jax_matvec(demand) + offset)
    mean = jnp.maximum(mean_unfloored, problem.mean_floor)
    mean_gradient = jax.grad(lambda value: _objective_from_mean(value, raw, problem))(
        mean
    )
    active_mean = (mean_unfloored > problem.mean_floor).astype(mean.dtype)
    demand_cotangent = rho * problem.operator.jax_rmatvec(
        active_mean * mean_gradient
    )
    demand_gradient = demand_pullback(demand_cotangent)[0]
    direct_gradient = jax.grad(
        lambda parameters: _objective_from_mean(mean, parameters, problem)
    )(raw)
    return _evaluation_from_mean(
        raw, mean=mean, demand=demand, problem=problem
    ), demand_gradient + direct_gradient


def gravity_value_and_gradient(
    raw_parameters: object,
    *,
    problem: GravityObjectiveProblem,
    strategy: GravityGradientStrategy,
) -> tuple[GravityObjectiveEvaluation, jax.Array]:
    if strategy is GravityGradientStrategy.BATCHED_FORWARD:
        return gravity_value_and_gradient_batched_forward(
            raw_parameters, problem=problem
        )
    if strategy is GravityGradientStrategy.ADJOINT:
        return gravity_value_and_gradient_adjoint(raw_parameters, problem=problem)
    raise ValueError(f"unsupported gravity gradient strategy {strategy!r}.")
