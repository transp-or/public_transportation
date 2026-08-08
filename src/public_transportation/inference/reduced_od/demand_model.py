"""Unified JAX demand family assembled from resolved parameter blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .demand_parameters import DemandParameterLayout
from .demand_specification import DemandModelSpecification
from .features import ConditionalGravityFeatures
from .observation_models import resolve_observation_model
from .response_operator import GravityResponseOperatorAdapter, ReducedResponseOperator


def _immutable_index(value: object, size: int, upper: int, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.int64, copy=True)
    if result.shape != (size,) or np.any(result < 0) or np.any(result >= upper):
        raise ValueError(f"{name} must be a valid index vector of length {size}.")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class DemandModelProblem:
    features: ConditionalGravityFeatures
    response_operator: ReducedResponseOperator | GravityResponseOperatorAdapter
    observations: np.ndarray
    specification: DemandModelSpecification
    parameter_layout: DemandParameterLayout
    group_period_index: np.ndarray
    origin_group_index: np.ndarray
    cell_destination_group_index: np.ndarray
    zero_inflation_design: np.ndarray | None = None
    calibration_mask: np.ndarray | None = None
    mean_floor: float = 1.0e-9

    def __post_init__(self) -> None:
        d = self.parameter_layout.dimensions
        if (
            self.parameter_layout.specification_fingerprint
            != self.specification.fingerprint
        ):
            raise ValueError("parameter layout and model specification differ.")
        cells, groups = (
            self.features.number_of_cells,
            self.features.number_of_origin_time_groups,
        )
        if cells != self.response_operator.number_of_free_cells:
            raise ValueError(
                "features and response operator have different free-cell dimensions."
            )
        object.__setattr__(
            self,
            "group_period_index",
            _immutable_index(
                self.group_period_index, groups, d.periods, "group_period_index"
            ),
        )
        object.__setattr__(
            self,
            "origin_group_index",
            _immutable_index(
                self.origin_group_index, groups, d.origin_groups, "origin_group_index"
            ),
        )
        object.__setattr__(
            self,
            "cell_destination_group_index",
            _immutable_index(
                self.cell_destination_group_index,
                cells,
                d.destination_groups,
                "cell_destination_group_index",
            ),
        )
        observations = np.array(self.observations, dtype=np.float64, copy=True)
        if (
            observations.shape != (self.response_operator.number_of_measurements,)
            or np.any(observations < 0)
            or not np.all(np.isfinite(observations))
        ):
            raise ValueError(
                "observations must be finite nonnegative and measurement-aligned."
            )
        observations.setflags(write=False)
        object.__setattr__(self, "observations", observations)
        mask = (
            np.ones(observations.size, dtype=bool)
            if self.calibration_mask is None
            else np.array(self.calibration_mask, dtype=bool, copy=True)
        )
        if mask.shape != observations.shape or not np.any(mask):
            raise ValueError("calibration mask must select aligned observations.")
        mask.setflags(write=False)
        object.__setattr__(self, "calibration_mask", mask)
        if self.specification.observation.zero_inflation == "none":
            if self.zero_inflation_design is not None:
                raise ValueError(
                    "disabled zero inflation must not create a design matrix."
                )
        else:
            design = (
                np.ones((observations.size, 1))
                if self.specification.observation.zero_inflation == "intercept"
                and self.zero_inflation_design is None
                else np.asarray(self.zero_inflation_design)
            )
            if design.shape != (
                observations.size,
                d.zero_design_columns
                if self.specification.observation.zero_inflation != "intercept"
                else 1,
            ) or not np.all(np.isfinite(design)):
                raise ValueError(
                    "zero-inflation design has an invalid shape or values."
                )
            design = np.array(design, dtype=np.float64, copy=True)
            design.setflags(write=False)
            object.__setattr__(self, "zero_inflation_design", design)


class DemandModelEvaluation(NamedTuple):
    objective: jax.Array
    log_likelihood: jax.Array
    regularization: jax.Array
    measurement_mean: jax.Array
    demand: jax.Array
    productions: jax.Array
    destination_probabilities: jax.Array


def _block(
    raw: jax.Array, layout: DemandParameterLayout, name: str
) -> jax.Array | None:
    selected = [item for item in layout.blocks if item.name == name]
    if not selected:
        return None
    item = selected[0]
    value = raw[item.start : item.stop].reshape(item.shape)
    return jax.nn.softplus(value) + 1.0e-6 if item.transform == "softplus" else value


def _centered(value: jax.Array | None, size: int) -> jax.Array:
    if value is None:
        return jnp.zeros(size)
    flat = value.reshape(-1)
    return jnp.concatenate((flat, -jnp.sum(flat, keepdims=True)))


def evaluate_demand_model(
    raw_parameters: object, *, problem: DemandModelProblem
) -> DemandModelEvaluation:
    raw = jnp.asarray(raw_parameters)
    if raw.shape != (problem.parameter_layout.size,):
        raise ValueError(
            f"raw parameters must have shape ({problem.parameter_layout.size},)."
        )
    layout, spec, dims = (
        problem.parameter_layout,
        problem.specification,
        problem.parameter_layout.dimensions,
    )
    features = problem.features
    group_period = jnp.asarray(problem.group_period_index)
    origin_group = jnp.asarray(problem.origin_group_index)
    cell_group = jnp.asarray(features.origin_time_group_index)
    destination_group = jnp.asarray(problem.cell_destination_group_index)
    production_log = jnp.zeros(features.number_of_origin_time_groups, dtype=raw.dtype)
    intercept = _block(raw, layout, "production_intercept")
    if intercept is not None:
        production_log += intercept[0]
    period = _block(raw, layout, "production_period")
    if period is not None:
        production_log += _centered(period, dims.periods)[group_period]
    origins = _block(raw, layout, "production_origin_group")
    if origins is not None:
        production_log += _centered(origins, dims.origin_groups)[origin_group]
    origin_period = _block(raw, layout, "production_origin_period")
    if origin_period is not None:
        full = jnp.pad(origin_period, ((0, 1), (0, 1)))
        full = full.at[-1, :-1].set(-jnp.sum(full[:-1, :-1], axis=0))
        full = full.at[:-1, -1].set(-jnp.sum(full[:-1, :-1], axis=1))
        full = full.at[-1, -1].set(jnp.sum(full[:-1, :-1]))
        production_log += full[origin_group, group_period]
    baseline_productions = jnp.asarray(features.baseline_productions, dtype=raw.dtype)
    productions = baseline_productions * jnp.exp(production_log)

    utility = jnp.log(jnp.asarray(features.destination_attractiveness, dtype=raw.dtype))
    attraction_global = _block(raw, layout, "attraction_global")
    if attraction_global is not None:
        utility += attraction_global[0]
    attraction_period = _block(raw, layout, "attraction_period")
    if attraction_period is not None:
        utility += _centered(attraction_period, dims.periods)[group_period[cell_group]]
    attraction_group = _block(raw, layout, "attraction_destination_group")
    if attraction_group is not None:
        utility += _centered(attraction_group, dims.destination_groups)[
            destination_group
        ]
    attraction_gp = _block(raw, layout, "attraction_destination_period")
    if attraction_gp is not None:
        full = jnp.pad(attraction_gp, ((0, 1), (0, 1)))
        full = full.at[-1, :-1].set(-jnp.sum(full[:-1, :-1], axis=0))
        full = full.at[:-1, -1].set(-jnp.sum(full[:-1, :-1], axis=1))
        full = full.at[-1, -1].set(jnp.sum(full[:-1, :-1]))
        utility += full[destination_group, group_period[cell_group]]
    beta_time = _block(raw, layout, "impedance_time")
    if beta_time is not None:
        coefficient = (
            beta_time[group_period[cell_group]] if beta_time.size > 1 else beta_time[0]
        )
        utility -= (
            coefficient
            * jnp.asarray(features.journey_time_seconds, dtype=raw.dtype)
            / 1800.0
        )
    beta_transfer = _block(raw, layout, "impedance_transfer")
    if beta_transfer is not None:
        coefficient = (
            beta_transfer[group_period[cell_group]]
            if beta_transfer.size > 1
            else beta_transfer[0]
        )
        utility -= coefficient * jnp.asarray(features.transfer_count, dtype=raw.dtype)
    u = _block(raw, layout, "interaction_origin")
    v = _block(raw, layout, "interaction_destination")
    if u is not None and v is not None:
        u = u - jnp.mean(u, axis=0, keepdims=True)
        v = v - jnp.mean(v, axis=0, keepdims=True)
        utility += jnp.sum(u[origin_group[cell_group]] * v[destination_group], axis=1)
    maximum = jax.ops.segment_max(
        utility, cell_group, num_segments=features.number_of_origin_time_groups
    )
    weights = jnp.exp(utility - maximum[cell_group])
    denominator = jax.ops.segment_sum(
        weights, cell_group, num_segments=features.number_of_origin_time_groups
    )
    probabilities = weights / denominator[cell_group]
    demand = productions[cell_group] * probabilities
    mean = problem.response_operator.jax_matvec(demand) + jnp.asarray(
        problem.response_operator.fixed_offset, dtype=raw.dtype
    )
    mean = jnp.maximum(mean, problem.mean_floor)
    dispersion_block = _block(raw, layout, "observation_dispersion")
    dispersion = None if dispersion_block is None else dispersion_block[0]
    zero = _block(raw, layout, "zero_inflation")
    logits = (
        None
        if zero is None
        else jnp.asarray(problem.zero_inflation_design, dtype=raw.dtype) @ zero
    )
    model = resolve_observation_model(spec.observation.family)
    contributions = model.log_likelihood(
        problem.observations, mean, dispersion=dispersion, inflation_logits=logits
    )
    log_likelihood = jnp.sum(
        jnp.where(jnp.asarray(problem.calibration_mask), contributions, 0.0)
    )
    regularization = jnp.asarray(0.0, dtype=raw.dtype)
    for item in layout.blocks:
        if item.ridge:
            values = raw[item.start : item.stop]
            regularization += 0.5 * item.ridge * jnp.sum(values * values)
    return DemandModelEvaluation(
        -log_likelihood + regularization,
        log_likelihood,
        regularization,
        mean,
        demand,
        productions,
        probabilities,
    )


def demand_model_value_and_gradient(
    raw_parameters: object, *, problem: DemandModelProblem
) -> tuple[DemandModelEvaluation, jax.Array]:
    raw = jnp.asarray(raw_parameters)
    _, gradient = jax.value_and_grad(
        lambda value: evaluate_demand_model(value, problem=problem).objective
    )(raw)
    return evaluate_demand_model(raw, problem=problem), gradient
