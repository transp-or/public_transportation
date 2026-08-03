"""Bounded operational preflight for gravity estimation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from time import perf_counter
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .demand import generate_gravity_demand
from .objective import (
    GravityGradientStrategy,
    GravityObjectiveProblem,
    gravity_value_and_gradient,
)


class GravityPreflightPhase(IntEnum):
    """Ordered safe boundaries at which a preflight may stop."""

    VALIDATION = 1
    FORWARD = 2
    REVERSE = 3
    OBJECTIVE_GRADIENT = 4
    PROJECTION = 5
    RECOMMENDATION = 6


@dataclass(frozen=True, slots=True)
class GravityPreflightRecommendation:
    gradient_strategy: GravityGradientStrategy
    operator_shards_per_batch: int
    resident_shard_limit: int
    expected_evaluation_seconds: float
    expected_peak_rss_bytes: int | None
    suggested_estimator_wall_time_seconds: float
    suggested_checkpoint_interval: int
    laptop_feasible: bool
    server_recommended: bool


@dataclass(frozen=True, slots=True)
class GravityPreflightResult:
    completed_phase: GravityPreflightPhase
    cache_shards_validated: int
    timings_seconds: Mapping[str, float]
    objective_values: Mapping[str, float]
    gradient_max_abs_difference: float | None
    peak_rss_bytes: int | None
    resident_routing_bytes: int
    recommendation: GravityPreflightRecommendation | None


def _metric(operator: object, name: str, default: object) -> object:
    return getattr(getattr(operator, "metrics", None), name, default)


def run_gravity_preflight(
    *,
    problem: GravityObjectiveProblem,
    raw_parameters: object,
    stop_after: GravityPreflightPhase = GravityPreflightPhase.RECOMMENDATION,
    projected_optimizer_iterations: int = 25,
) -> GravityPreflightResult:
    """Validate and time one bounded gravity evaluation, phase by phase.

    Each completed phase is a safe boundary. Operator products themselves obey
    any deadline or cancellation policy configured on the operator.
    """
    if projected_optimizer_iterations <= 0:
        raise ValueError("projected_optimizer_iterations must be positive.")
    operator = problem.operator
    raw = jnp.asarray(raw_parameters)
    timings: dict[str, float] = {}
    objective_values: dict[str, float] = {}
    cache_shards = 0

    started = perf_counter()
    validator = getattr(operator, "validate_routing_cache", None)
    if validator is not None:
        cache_shards = int(validator())
    if operator.num_free_od != problem.features.num_cells:
        raise ValueError("operator and gravity dimensions differ.")
    for name in ("assignment_fingerprint", "graph_fingerprint", "mapping_fingerprint"):
        if not getattr(operator, name, ""):
            raise ValueError(f"operator {name} is empty.")
    timings["validation"] = perf_counter() - started
    if stop_after <= GravityPreflightPhase.VALIDATION:
        return _result(stop_after, cache_shards, timings, objective_values, None, operator)

    demand = generate_gravity_demand(
        raw, features=problem.features, parameter_layout=problem.parameter_layout
    ).demand
    started = perf_counter()
    forward = operator.jax_matvec(demand)
    jax.block_until_ready(forward)
    timings["forward"] = perf_counter() - started
    if stop_after <= GravityPreflightPhase.FORWARD:
        return _result(stop_after, cache_shards, timings, objective_values, None, operator)

    started = perf_counter()
    reverse = operator.jax_rmatvec(jnp.ones(operator.num_measurements))
    jax.block_until_ready(reverse)
    timings["reverse"] = perf_counter() - started
    if stop_after <= GravityPreflightPhase.REVERSE:
        return _result(stop_after, cache_shards, timings, objective_values, None, operator)

    gradients: dict[GravityGradientStrategy, np.ndarray] = {}
    for strategy in (
        GravityGradientStrategy.ADJOINT,
        GravityGradientStrategy.BATCHED_FORWARD,
    ):
        started = perf_counter()
        evaluation, gradient = gravity_value_and_gradient(
            raw, problem=problem, strategy=strategy
        )
        jax.block_until_ready((evaluation, gradient))
        timings[f"{strategy.value}_cold"] = perf_counter() - started
        started = perf_counter()
        evaluation, gradient = gravity_value_and_gradient(
            raw, problem=problem, strategy=strategy
        )
        jax.block_until_ready((evaluation, gradient))
        timings[strategy.value] = perf_counter() - started
        objective_values[strategy.value] = float(evaluation.objective)
        gradients[strategy] = np.asarray(gradient)
    difference = float(
        np.max(
            np.abs(
                gradients[GravityGradientStrategy.ADJOINT]
                - gradients[GravityGradientStrategy.BATCHED_FORWARD]
            )
        )
    )
    if stop_after <= GravityPreflightPhase.OBJECTIVE_GRADIENT:
        return _result(
            stop_after, cache_shards, timings, objective_values, difference, operator
        )

    selected = min(
        (GravityGradientStrategy.ADJOINT, GravityGradientStrategy.BATCHED_FORWARD),
        key=lambda item: timings[item.value],
    )
    evaluation_seconds = timings[selected.value]
    timings["projected_10_iterations"] = 10.0 * evaluation_seconds
    timings["projected_25_iterations"] = 25.0 * evaluation_seconds
    timings["projected_50_iterations"] = 50.0 * evaluation_seconds
    timings["projected_100_iterations"] = 100.0 * evaluation_seconds
    if stop_after <= GravityPreflightPhase.PROJECTION:
        return _result(
            stop_after, cache_shards, timings, objective_values, difference, operator
        )

    peak_rss = _optional_int(_metric(operator, "peak_rss_bytes", None))
    laptop_feasible = evaluation_seconds <= 120.0 and (
        peak_rss is None or peak_rss <= 8 * 1024**3
    )
    recommendation = GravityPreflightRecommendation(
        gradient_strategy=selected,
        operator_shards_per_batch=int(
            getattr(operator, "operator_shards_per_batch", 1)
        ),
        resident_shard_limit=int(getattr(operator, "resident_shard_limit", 1)),
        expected_evaluation_seconds=evaluation_seconds,
        expected_peak_rss_bytes=peak_rss,
        suggested_estimator_wall_time_seconds=(
            1.5 * projected_optimizer_iterations * evaluation_seconds
        ),
        suggested_checkpoint_interval=1,
        laptop_feasible=laptop_feasible,
        server_recommended=not laptop_feasible,
    )
    return _result(
        GravityPreflightPhase.RECOMMENDATION,
        cache_shards,
        timings,
        objective_values,
        difference,
        operator,
        recommendation,
    )


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _result(
    phase: GravityPreflightPhase,
    cache_shards: int,
    timings: Mapping[str, float],
    objective_values: Mapping[str, float],
    difference: float | None,
    operator: object,
    recommendation: GravityPreflightRecommendation | None = None,
) -> GravityPreflightResult:
    return GravityPreflightResult(
        completed_phase=phase,
        cache_shards_validated=cache_shards,
        timings_seconds=dict(timings),
        objective_values=dict(objective_values),
        gradient_max_abs_difference=difference,
        peak_rss_bytes=_optional_int(_metric(operator, "peak_rss_bytes", None)),
        resident_routing_bytes=int(_metric(operator, "resident_routing_bytes", 0)),
        recommendation=recommendation,
    )
