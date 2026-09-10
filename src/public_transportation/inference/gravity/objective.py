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
from .observation_model import GravityObservationModel
from .observations import GravityObservationBundle
from .parameters import (
    GravityJointParameterLayout,
    GravityParameterLayout,
    validate_gravity_relaxation_features,
)
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
    parameter_layout: GravityParameterLayout | GravityJointParameterLayout
    operator: GravityMeasurementOperator
    observations: np.ndarray
    likelihood: GravityLikelihood = GravityLikelihood.NEGATIVE_BINOMIAL
    rho: float = 1.0
    calibration_mask: np.ndarray | None = None
    mean_floor: float = 1.0e-9
    auxiliary_observations: GravityObservationBundle | None = None
    observation_model: GravityObservationModel | None = None

    def __post_init__(self) -> None:
        observations_bundle = (
            GravityObservationBundle.empty()
            if self.auxiliary_observations is None
            else self.auxiliary_observations
        )
        if not isinstance(observations_bundle, GravityObservationBundle):
            raise TypeError(
                "auxiliary_observations must be a GravityObservationBundle or None."
            )
        observations_bundle.validate(
            num_free_od=int(self.operator.num_free_od),
            dtype=np.dtype(getattr(self.operator, "dtype", np.float64)),
        )
        object.__setattr__(self, "auxiliary_observations", observations_bundle)
        if self.observation_model is not None:
            if not isinstance(self.observation_model, GravityObservationModel):
                raise TypeError(
                    "observation_model must be a GravityObservationModel or None."
                )
            self.observation_model.scale_vector(
                int(self.operator.num_measurements), dtype=np.float32
            )
        validate_gravity_relaxation_features(
            self.features, self.parameter_layout.specification
        )
        for block in getattr(self.parameter_layout, "additive_flow_blocks", ()):
            if block.operator.num_rows != self.operator.num_measurements:
                raise ValueError(
                    f"additive flow block {block.name!r} has {block.operator.num_rows} rows; "
                    "it must match the primary measurement dimension."
                )
        specification = self.parameter_layout.specification
        if (
            specification.components
            and specification.likelihood.family != self.likelihood.value
        ):
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
        if (
            waiting.scope
            not in (
                GravityEffectScope.NONE,
                GravityEffectScope.FIXED,
            )
            and self.features.initial_waiting_time is None
        ):
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
    count_log_likelihood: jax.Array = 0.0
    auxiliary_log_likelihood: jax.Array = 0.0
    auxiliary_channel_log_likelihoods: tuple[jax.Array, ...] = ()
    latent_measurement: jax.Array | None = None
    od_measurement: jax.Array | None = None
    additive_measurements: tuple[jax.Array, ...] = ()
    additive_flows: tuple[jax.Array, ...] = ()
    observation_scales: jax.Array | None = None


def _has_additive_measurement_blocks(problem: GravityObjectiveProblem) -> bool:
    return bool(getattr(problem.parameter_layout, "additive_flow_blocks", ()))


def _measurement_components(
    raw_parameters: object, *, problem: GravityObjectiveProblem
) -> tuple[
    jax.Array,
    jax.Array,
    tuple[jax.Array, ...],
    tuple[jax.Array, ...],
    jax.Array,
]:
    """Evaluate the opt-in additive measurement branch."""
    raw = jnp.asarray(raw_parameters)
    demand = generate_gravity_demand(
        raw,
        features=problem.features,
        parameter_layout=problem.parameter_layout,
    ).demand
    od_measurement = problem.operator.jax_matvec(demand)
    additive_measurements: list[jax.Array] = []
    additive_flows: list[jax.Array] = []
    for block in getattr(problem.parameter_layout, "additive_flow_blocks", ()):
        flow = block.flow_from_raw(
            problem.parameter_layout.block_raw(raw, block.name)
        )
        additive_flows.append(flow)
        additive_measurements.append(block.operator.jax_matvec(flow))
    latent = od_measurement
    for contribution in additive_measurements:
        latent = latent + contribution
    offset = jnp.asarray(problem.operator.fixed_measurement_offset, dtype=raw.dtype)
    latent = latent + offset
    if problem.observation_model is None:
        scales = jnp.full(
            (int(problem.operator.num_measurements),),
            jnp.asarray(problem.rho, dtype=raw.dtype),
        )
    else:
        scales = jnp.asarray(problem.rho, dtype=raw.dtype) * problem.observation_model.scale_vector(
            int(problem.operator.num_measurements), dtype=raw.dtype
        )
    return (
        demand,
        latent,
        tuple(additive_measurements),
        tuple(additive_flows),
        scales,
    )


def _apply_mean_floor(mean_unfloored: jax.Array, floor: float) -> jax.Array:
    """Apply the numerical mean floor with a zero derivative on floor rows."""
    floor_value = jnp.asarray(floor, dtype=mean_unfloored.dtype)
    # ``where`` keeps the floor branch constant when a row is at or below the
    # floor.  ``maximum`` has an implementation-defined subgradient at exact
    # equality (often one half), which would incorrectly let a floor-only row
    # contribute to the additive-flow gradient.
    return jnp.where(mean_unfloored > floor_value, mean_unfloored, floor_value)


def predict_gravity_measurements(
    raw_parameters: object, *, problem: GravityObjectiveProblem
) -> tuple[jax.Array, jax.Array]:
    if not _has_additive_measurement_blocks(problem) and problem.observation_model is None:
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
    demand, latent, _, _, scales = _measurement_components(
        raw_parameters, problem=problem
    )
    mean = _apply_mean_floor(scales * latent, problem.mean_floor)
    return mean, demand


def evaluate_gravity_objective(
    raw_parameters: object, *, problem: GravityObjectiveProblem
) -> GravityObjectiveEvaluation:
    raw = jnp.asarray(raw_parameters)
    if not _has_additive_measurement_blocks(problem) and problem.observation_model is None:
        mean, demand = predict_gravity_measurements(raw, problem=problem)
        return _evaluation_from_mean(raw, mean=mean, demand=demand, problem=problem)
    demand, latent, additive_measurements, additive_flows, scales = _measurement_components(
        raw, problem=problem
    )
    mean = _apply_mean_floor(scales * latent, problem.mean_floor)
    return _evaluation_from_mean(
        raw,
        mean=mean,
        demand=demand,
        problem=problem,
        latent_measurement=latent,
        od_measurement=problem.operator.jax_matvec(demand),
        additive_measurements=additive_measurements,
        additive_flows=additive_flows,
        observation_scales=scales,
    )


def _evaluation_from_mean(
    raw: jax.Array,
    *,
    mean: jax.Array,
    demand: jax.Array,
    problem: GravityObjectiveProblem,
    latent_measurement: jax.Array | None = None,
    od_measurement: jax.Array | None = None,
    additive_measurements: tuple[jax.Array, ...] = (),
    additive_flows: tuple[jax.Array, ...] = (),
    observation_scales: jax.Array | None = None,
) -> GravityObjectiveEvaluation:
    count_log_likelihood = _count_log_likelihood(mean, raw, problem)
    auxiliary_log_likelihood, auxiliary_channel_log_likelihoods = (
        _auxiliary_log_likelihood(raw, demand, problem)
    )
    data_log_likelihood = count_log_likelihood + auxiliary_log_likelihood
    regularization = problem.parameter_layout.regularization(raw)
    return GravityObjectiveEvaluation(
        objective=-data_log_likelihood + regularization,
        data_log_likelihood=data_log_likelihood,
        regularization=regularization,
        measurement_mean=mean,
        demand=demand,
        calibration_measurements=jnp.asarray(problem.calibration_measurements),
        excluded_measurements=jnp.asarray(problem.excluded_measurements),
        count_log_likelihood=count_log_likelihood,
        auxiliary_log_likelihood=auxiliary_log_likelihood,
        auxiliary_channel_log_likelihoods=auxiliary_channel_log_likelihoods,
        latent_measurement=latent_measurement,
        od_measurement=od_measurement,
        additive_measurements=additive_measurements,
        additive_flows=additive_flows,
        observation_scales=observation_scales,
    )


def _count_log_likelihood(
    mean: jax.Array, raw: jax.Array, problem: GravityObjectiveProblem
) -> jax.Array:
    observations = jnp.asarray(problem.observations, dtype=mean.dtype)
    mask = jnp.asarray(problem.calibration_mask)
    if problem.likelihood is GravityLikelihood.POISSON:
        contributions = poisson_logpmf(observations, mean)
    else:
        dispersion = problem.parameter_layout.transform(raw).dispersion
        contributions = negbinom_logpmf_mu_r(observations, mean, dispersion)
    return jnp.sum(jnp.where(mask, contributions, 0))


def _auxiliary_log_likelihood(
    raw: jax.Array,
    demand: jax.Array,
    problem: GravityObjectiveProblem,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    channels = problem.auxiliary_observations.channels
    if not channels:
        return jnp.asarray(0.0, dtype=demand.dtype), ()
    contributions = tuple(
        channel.log_likelihood(
            prediction=channel.predict(demand),
            raw_parameters=raw,
        )
        for channel in channels
    )
    return jnp.sum(jnp.stack(contributions)), contributions


def _objective_scalar(raw: jax.Array, problem: GravityObjectiveProblem) -> jax.Array:
    return evaluate_gravity_objective(raw, problem=problem).objective


def gravity_value_and_gradient_batched_forward(
    raw_parameters: object, *, problem: GravityObjectiveProblem
) -> tuple[GravityObjectiveEvaluation, jax.Array]:
    """Small-parameter gradient using a batched demand Jacobian."""
    if _has_additive_measurement_blocks(problem) or problem.observation_model is not None:
        return _gravity_value_and_gradient_additive(raw_parameters, problem=problem)
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
        active_mean[:, None] * rho * problem.operator.jax_matmat(demand_jacobian)
    )
    if problem.auxiliary_observations.enabled:
        mean_gradient = jax.grad(
            lambda value: _objective_from_mean_and_demand(value, raw, demand, problem)
        )(mean)
        demand_gradient = jax.grad(
            lambda value: _objective_from_mean_and_demand(mean, raw, value, problem)
        )(demand)
        direct_gradient = jax.grad(
            lambda parameters: _objective_from_mean_and_demand(
                mean, parameters, demand, problem
            )
        )(raw)
        gradient = (
            mean_jacobian.T @ mean_gradient
            + demand_jacobian.T @ demand_gradient
            + direct_gradient
        )
    else:
        mean_gradient = jax.grad(
            lambda value: _objective_from_mean(value, raw, problem)
        )(mean)
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
    return -_count_log_likelihood(
        mean, raw, problem
    ) + problem.parameter_layout.regularization(raw)


def _objective_from_mean_and_demand(
    mean: jax.Array,
    raw: jax.Array,
    demand: jax.Array,
    problem: GravityObjectiveProblem,
) -> jax.Array:
    auxiliary, _ = _auxiliary_log_likelihood(raw, demand, problem)
    return (
        -_count_log_likelihood(mean, raw, problem)
        - auxiliary
        + problem.parameter_layout.regularization(raw)
    )


def _gravity_value_and_gradient_additive(
    raw_parameters: object, *, problem: GravityObjectiveProblem
) -> tuple[GravityObjectiveEvaluation, jax.Array]:
    """Reference gradient for the opt-in additive-flow branch.

    The branch is intentionally isolated from the legacy kernels.  It uses
    JAX's reverse-mode differentiation through each declared linear operator,
    which is exact for dense and matrix-free operators implementing the
    generic protocol and provides a correctness baseline for later specialized
    batched/adjoint kernels.
    """
    raw = jnp.asarray(raw_parameters)
    evaluation = evaluate_gravity_objective(raw, problem=problem)
    gradient = jax.grad(
        lambda value: evaluate_gravity_objective(value, problem=problem).objective
    )(raw)
    return evaluation, gradient


def gravity_value_and_gradient_adjoint(
    raw_parameters: object, *, problem: GravityObjectiveProblem
) -> tuple[GravityObjectiveEvaluation, jax.Array]:
    """Adjoint gradient using one routing transpose product and a demand VJP."""
    if _has_additive_measurement_blocks(problem) or problem.observation_model is not None:
        return _gravity_value_and_gradient_additive(raw_parameters, problem=problem)
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
    if problem.auxiliary_observations.enabled:
        mean_gradient = jax.grad(
            lambda value: _objective_from_mean_and_demand(value, raw, demand, problem)
        )(mean)
        demand_gradient = jax.grad(
            lambda value: _objective_from_mean_and_demand(mean, raw, value, problem)
        )(demand)
        direct_gradient = jax.grad(
            lambda parameters: _objective_from_mean_and_demand(
                mean, parameters, demand, problem
            )
        )(raw)
    else:
        mean_gradient = jax.grad(
            lambda value: _objective_from_mean(value, raw, problem)
        )(mean)
        demand_gradient = jnp.zeros_like(demand)
        direct_gradient = jax.grad(
            lambda parameters: _objective_from_mean(mean, parameters, problem)
        )(raw)
    active_mean = (mean_unfloored > problem.mean_floor).astype(mean.dtype)
    demand_cotangent = (
        rho * problem.operator.jax_rmatvec(active_mean * mean_gradient)
        + demand_gradient
    )
    demand_gradient = demand_pullback(demand_cotangent)[0]
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
