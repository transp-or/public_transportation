"""Stable sparse grouped-softmax gravity demand."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .features import GravityFeatures
from .parameters import GravityParameterLayout, validate_gravity_relaxation_features


class GravityDemandResult(NamedTuple):
    demand: jax.Array
    probabilities: jax.Array
    utilities: jax.Array
    origin_time_sums: jax.Array


def gravity_demand_kernel(
    raw_parameters: object,
    *,
    journey_time: object,
    transfer_count: object,
    structural_feasible: object,
    origin_time_group_index: object,
    origin_time_totals: object,
    destination_attractiveness: object,
    journey_time_scale: float,
    positivity_floor: float = 1.0e-6,
) -> GravityDemandResult:
    """Generate canonical cell demand without a dense OD-time tensor."""
    raw = jnp.asarray(raw_parameters)
    if raw.ndim != 1 or raw.shape[0] != 3:
        raise ValueError("raw_parameters must have shape (3,).")
    positive = jax.nn.softplus(raw) + jnp.asarray(positivity_floor, dtype=raw.dtype)
    beta_time, beta_transfer = positive[:2]
    time = jnp.asarray(journey_time, dtype=raw.dtype)
    transfers = jnp.asarray(transfer_count, dtype=raw.dtype)
    feasible = jnp.asarray(structural_feasible, dtype=jnp.bool_)
    groups = jnp.asarray(origin_time_group_index, dtype=jnp.int32)
    totals = jnp.asarray(origin_time_totals, dtype=raw.dtype)
    attractiveness = jnp.asarray(destination_attractiveness, dtype=raw.dtype)
    utility = (
        -beta_time * time / jnp.asarray(journey_time_scale, dtype=raw.dtype)
        - beta_transfer * transfers
        + jnp.log(attractiveness)
    )
    masked_utility = jnp.where(feasible, utility, -jnp.inf)
    group_maximum = jax.ops.segment_max(
        masked_utility, groups, num_segments=totals.shape[0]
    )
    weights = jnp.where(feasible, jnp.exp(masked_utility - group_maximum[groups]), 0)
    denominators = jax.ops.segment_sum(weights, groups, num_segments=totals.shape[0])
    probabilities = jnp.where(feasible, weights / denominators[groups], 0)
    demand = probabilities * totals[groups]
    origin_time_sums = jax.ops.segment_sum(demand, groups, num_segments=totals.shape[0])
    return GravityDemandResult(demand, probabilities, masked_utility, origin_time_sums)


def generate_gravity_demand(
    raw_parameters: object,
    *,
    features: GravityFeatures,
    parameter_layout: GravityParameterLayout,
) -> GravityDemandResult:
    if parameter_layout.size == 3:
        return gravity_demand_kernel(
            raw_parameters,
            journey_time=features.journey_time,
            transfer_count=features.transfer_count,
            structural_feasible=features.structural_feasible,
            origin_time_group_index=features.origin_time_group_index,
            origin_time_totals=features.origin_time_totals,
            destination_attractiveness=features.destination_attractiveness,
            journey_time_scale=features.journey_time_scale,
            positivity_floor=parameter_layout.positivity_floor,
        )
    validate_gravity_relaxation_features(features, parameter_layout.specification)
    raw = jnp.asarray(raw_parameters)
    parameters = parameter_layout.transform(raw)
    groups = jnp.asarray(features.origin_time_group_index, dtype=jnp.int32)
    feasible = jnp.asarray(features.structural_feasible, dtype=jnp.bool_)
    time_coefficient = parameters.beta_time
    if features.time_period_index is not None and parameter_layout.specification.time_period_count:
        periods = jnp.asarray(features.time_period_index, dtype=jnp.int32)
        time_coefficient = time_coefficient * jnp.exp(
            parameter_layout.centered_effect(raw, "time_period")[periods]
        )
    utility = (
        -time_coefficient * jnp.asarray(features.journey_time, dtype=raw.dtype) / features.journey_time_scale
        - parameters.beta_transfer * jnp.asarray(features.transfer_count, dtype=raw.dtype)
        + jnp.log(jnp.asarray(features.destination_attractiveness, dtype=raw.dtype))
    )
    if features.destination_zone_index is not None and parameter_layout.specification.destination_zone_count:
        utility = utility + parameter_layout.centered_effect(raw, "destination_zone")[
            jnp.asarray(features.destination_zone_index, dtype=jnp.int32)
        ]
    masked = jnp.where(feasible, utility, -jnp.inf)
    maximum = jax.ops.segment_max(masked, groups, num_segments=features.num_origin_time_groups)
    weights = jnp.where(feasible, jnp.exp(masked - maximum[groups]), 0)
    denominator = jax.ops.segment_sum(weights, groups, num_segments=features.num_origin_time_groups)
    probabilities = jnp.where(feasible, weights / denominator[groups], 0)
    totals = jnp.asarray(features.origin_time_totals, dtype=raw.dtype)
    if features.origin_zone_index is not None and parameter_layout.specification.origin_zone_count:
        zones = jnp.asarray(features.origin_zone_index, dtype=jnp.int32)
        totals = totals * jnp.exp(
            parameter_layout.centered_effect(raw, "origin_zone")[zones][
                jnp.asarray([np.flatnonzero(features.origin_time_group_index == group)[0] for group in range(features.num_origin_time_groups)])
            ]
        )
    demand = probabilities * totals[groups]
    sums = jax.ops.segment_sum(demand, groups, num_segments=features.num_origin_time_groups)
    return GravityDemandResult(demand, probabilities, masked, sums)


def gravity_demand_numpy_reference(
    raw_parameters: object,
    *,
    features: GravityFeatures,
    parameter_layout: GravityParameterLayout,
) -> np.ndarray:
    """Testing-only NumPy reference with group-local stable softmax."""
    raw = np.asarray(raw_parameters, dtype=features.dtype)
    if raw.shape != (parameter_layout.size,):
        raise ValueError(f"raw_parameters must have shape ({parameter_layout.size},).")
    if parameter_layout.size != 3:
        return np.asarray(generate_gravity_demand(raw, features=features, parameter_layout=parameter_layout).demand)
    positive = np.logaddexp(np.asarray(0, dtype=raw.dtype), raw)
    positive += np.asarray(parameter_layout.positivity_floor, dtype=raw.dtype)
    utility = (
        -positive[0] * features.journey_time / features.journey_time_scale
        - positive[1] * features.transfer_count
        + np.log(features.destination_attractiveness)
    )
    demand = np.zeros(features.num_cells, dtype=features.dtype)
    for group in range(features.num_origin_time_groups):
        positions = np.flatnonzero(
            (features.origin_time_group_index == group) & features.structural_feasible
        )
        shifted = utility[positions] - np.max(utility[positions])
        weights = np.exp(shifted)
        demand[positions] = features.origin_time_totals[group] * weights / weights.sum()
    return demand
