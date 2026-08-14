"""Stable sparse grouped-softmax gravity demand."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .features import GravityFeatures
from .parameters import GravityParameterLayout, validate_gravity_relaxation_features
from .specification import (
    GravityConstraint,
    GravityEffectScope,
    GravityParameterization,
)


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
    if parameter_layout.specification == parameter_layout.specification.__class__():
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
    groups = jnp.asarray(features.origin_time_group_index, dtype=jnp.int32)
    feasible = jnp.asarray(features.structural_feasible, dtype=jnp.bool_)
    time_coefficient = parameter_layout.cell_effect(
        raw, "journey_time", features
    )
    explicit_component_names = {
        item.name for item in parameter_layout.specification.components
    }
    legacy_temporal_coefficient = (
        "temporal" not in explicit_component_names
        and parameter_layout.specification.temporal_basis_scope
        is GravityEffectScope.TIME_PERIOD
    )
    if legacy_temporal_coefficient:
        time_coefficient = time_coefficient * jnp.exp(
            parameter_layout.cell_effect(raw, "temporal", features)
        )
    transfer_coefficient = parameter_layout.cell_effect(raw, "transfer", features)
    utility = (
        -time_coefficient
        * jnp.asarray(features.journey_time, dtype=raw.dtype)
        / features.journey_time_scale
        - transfer_coefficient
        * jnp.asarray(features.transfer_count, dtype=raw.dtype)
        + jnp.log(jnp.asarray(features.destination_attractiveness, dtype=raw.dtype))
    )
    waiting = parameter_layout.specification.component("waiting_time")
    if waiting.scope not in (GravityEffectScope.NONE, GravityEffectScope.FIXED) or (
        waiting.fixed_value not in (None, 0.0)
    ):
        if features.initial_waiting_time is None:
            raise ValueError(
                "initial_waiting_time is required by the active waiting-time component."
            )
        utility = utility - parameter_layout.cell_effect(
            raw, "waiting_time", features
        ) * jnp.asarray(features.initial_waiting_time, dtype=raw.dtype)
    utility = utility + parameter_layout.cell_effect(
        raw, "destination_attractiveness", features
    )
    if not legacy_temporal_coefficient:
        utility = utility + parameter_layout.cell_effect(raw, "temporal", features)
    masked = jnp.where(feasible, utility, -jnp.inf)
    maximum = jax.ops.segment_max(masked, groups, num_segments=features.num_origin_time_groups)
    weights = jnp.where(feasible, jnp.exp(masked - maximum[groups]), 0)
    denominator = jax.ops.segment_sum(weights, groups, num_segments=features.num_origin_time_groups)
    probabilities = jnp.where(feasible, weights / denominator[groups], 0)
    log_multiplier = parameter_layout.production_group_log_multiplier(raw, features)
    totals = jnp.asarray(features.origin_time_totals, dtype=raw.dtype) * jnp.exp(
        log_multiplier
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
    validate_gravity_relaxation_features(features, parameter_layout.specification)

    def deviations(component_name: str) -> np.ndarray:
        block = parameter_layout.block(component_name)
        if block is None or block.group_count == 0:
            return np.empty(0, dtype=raw.dtype)
        values = raw[block.parameter_slice]
        if block.parameterization is GravityParameterization.POSITIVE:
            values = values[1:]
        if block.scope is GravityEffectScope.SMOOTH_BASIS:
            return values
        if block.constraint is GravityConstraint.SUM_ZERO:
            return np.concatenate((values, -np.sum(values, keepdims=True)))
        assert block.reference_category is not None
        return np.insert(values, block.reference_category, 0.0)

    def scalar_or_base(component_name: str) -> float:
        component = parameter_layout.specification.component(component_name)
        block = parameter_layout.block(component_name)
        if block is None:
            return float(0.0 if component.fixed_value is None else component.fixed_value)
        value = float(raw[block.parameter_slice.start])
        if block.parameterization is GravityParameterization.POSITIVE:
            return float(np.logaddexp(0.0, value) + parameter_layout.positivity_floor)
        return value

    def cell_effect(component_name: str) -> np.ndarray | float:
        component = parameter_layout.specification.component(component_name)
        block = parameter_layout.block(component_name)
        if block is None or component.scope in (
            GravityEffectScope.NONE,
            GravityEffectScope.FIXED,
        ):
            return scalar_or_base(component_name)
        if component.scope is GravityEffectScope.GLOBAL:
            return scalar_or_base(component_name)
        assert block.mapping is not None
        mapping = np.asarray(features.mapping(block.mapping))
        if component.scope is GravityEffectScope.SMOOTH_BASIS:
            return mapping @ raw[block.parameter_slice]
        effect = deviations(component_name)[mapping]
        if component.parameterization is GravityParameterization.POSITIVE:
            return scalar_or_base(component_name) * np.exp(effect)
        return effect

    time_coefficient = cell_effect("journey_time")
    explicit_component_names = {
        item.name for item in parameter_layout.specification.components
    }
    legacy_temporal_coefficient = (
        "temporal" not in explicit_component_names
        and parameter_layout.specification.temporal_basis_scope
        is GravityEffectScope.TIME_PERIOD
    )
    if legacy_temporal_coefficient:
        time_coefficient = time_coefficient * np.exp(cell_effect("temporal"))
    temporal_utility = 0.0 if legacy_temporal_coefficient else cell_effect("temporal")
    utility = (
        -time_coefficient
        * features.journey_time
        / features.journey_time_scale
        - cell_effect("transfer") * features.transfer_count
        + np.log(features.destination_attractiveness)
        + cell_effect("destination_attractiveness")
        + temporal_utility
    )
    waiting = parameter_layout.specification.component("waiting_time")
    if waiting.scope not in (GravityEffectScope.NONE, GravityEffectScope.FIXED) or (
        waiting.fixed_value not in (None, 0.0)
    ):
        if features.initial_waiting_time is None:
            raise ValueError(
                "initial_waiting_time is required by the active waiting-time component."
            )
        utility = utility - cell_effect("waiting_time") * features.initial_waiting_time
    production = parameter_layout.specification.component("production")
    if production.scope in (GravityEffectScope.NONE, GravityEffectScope.FIXED):
        log_multipliers = np.full(
            features.num_origin_time_groups, scalar_or_base("production")
        )
    elif production.scope is GravityEffectScope.GLOBAL:
        log_multipliers = np.full(
            features.num_origin_time_groups, scalar_or_base("production")
        )
    else:
        per_cell = np.asarray(cell_effect("production"))
        log_multipliers = np.zeros(features.num_origin_time_groups, dtype=raw.dtype)
        for group in range(features.num_origin_time_groups):
            log_multipliers[group] = per_cell[
                np.flatnonzero(features.origin_time_group_index == group)[0]
            ]
    demand = np.zeros(features.num_cells, dtype=features.dtype)
    for group in range(features.num_origin_time_groups):
        positions = np.flatnonzero(
            (features.origin_time_group_index == group) & features.structural_feasible
        )
        shifted = utility[positions] - np.max(utility[positions])
        weights = np.exp(shifted)
        demand[positions] = (
            features.origin_time_totals[group]
            * np.exp(log_multipliers[group])
            * weights
            / weights.sum()
        )
    return demand
