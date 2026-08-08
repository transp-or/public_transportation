"""Deterministic fixed-step and adaptive sparse departure-time quadrature."""

from __future__ import annotations

import hashlib
import heapq
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np

from .artifacts import canonical_json
from .departure_sampling import (
    DepartureTimeSamplingConfig,
    DesiredDepartureSample,
    SamplingProgress,
    SparseWeightedResponse,
    _peak_rss_bytes,
    canonicalize_probability_mass,
    validate_sample_weights,
)
from .journey_choices import JourneyTimePeriod
from .response_atoms import ResponseCellKey


SparseResponseEvaluator = Callable[[float], SparseWeightedResponse | None]
CellSparseResponseEvaluator = Callable[
    [ResponseCellKey, float], SparseWeightedResponse | None
]
QUADRATURE_SCHEMA_VERSION = 4

ADAPTIVE_COMPARISON_MODES = {
    "assignment_response",
    "service_signature",
    "exact_service_identity",
    "measurement_support",
    "aggregate_response",
    "two_stage",
    "route_pattern_signature",
    "integral_response",
}


@dataclass(frozen=True, slots=True)
class WeightedSparseDepartureResponse:
    """One elapsed-time mass associated with one sparse response."""

    weight: float
    feasible: bool
    response: SparseWeightedResponse
    representative_seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("quadrature response weight must be nonnegative.")
        if not self.feasible and (
            self.response.measurement_indices or self.response.values
        ):
            raise ValueError("an infeasible quadrature response must be sparse zero.")
        if not self.representative_seconds or any(
            not math.isfinite(value) for value in self.representative_seconds
        ):
            raise ValueError("quadrature responses require finite representatives.")


@dataclass(frozen=True, slots=True)
class DepartureQuadratureDiagnostics:
    """Serializable heuristic quality and resource diagnostics."""

    strategy: str
    configuration_fingerprint: str
    quadrature_schema_version: int
    interval_seconds: float
    initial_subintervals: int
    routing_evaluations: int
    cache_hits: int
    accepted_subintervals: int
    refined_subintervals: int
    refinement_depth_counts: tuple[tuple[int, int], ...]
    unique_evaluated_times: int
    unique_responses: int
    merged_responses: int
    response_support_changes: int
    feasible_time_fraction: float
    infeasible_time_fraction: float
    total_quadrature_weight: float
    estimated_relative_response_error: float
    unresolved_interval_weight: float
    sample_cap_reached: bool
    minimum_resolution_reached_with_instability: bool
    maximum_depth_reached_with_instability: bool
    quadrature_converged: bool
    elapsed_seconds: float
    peak_rss_bytes: int
    fingerprint: str
    requested_comparison_mode: str = "assignment_response"
    effective_comparison_mode: str = "assignment_response"
    budget_scope: str = "origin_period_group"
    initial_subintervals_evaluated: int = 0
    evaluation_budget: int = 0
    reserved_baseline_evaluations: int = 0
    baseline_evaluations: int = 0
    refinement_evaluations: int = 0
    stable_interval_weight: float = 0.0
    quadrature_rule: str = "pointwise_support_comparison"
    absolute_response_tolerance: float = 0.0
    relative_response_tolerance: float = 0.0
    integration_scale_floor: float = 1.0e-12
    coarse_integral_norm: float = 0.0
    refined_integral_norm: float = 0.0
    estimated_absolute_integration_error: float = 0.0
    global_error_target: float = 0.0
    global_target_achieved: bool = False
    unresolved_estimated_error: float = 0.0
    support_additions: int = 0
    support_removals: int = 0
    final_interval_count: int = 0
    minimum_resolution_interval_count: int = 0
    largest_error_intervals: tuple[tuple[float, float, float], ...] = ()
    interval_duration_distribution: tuple[tuple[float, int], ...] = ()
    service_boundary_safeguard: str = "none"


@dataclass(frozen=True, slots=True)
class SparseIntervalContribution:
    """One already probability-weighted sparse interval integral."""

    left_seconds: float
    right_seconds: float
    quadrature_rule: str
    evaluated_times: tuple[float, ...]
    weighted_response: SparseWeightedResponse
    feasible_time_weight: float
    infeasible_time_weight: float
    interval_weight: float

    def __post_init__(self) -> None:
        if self.right_seconds <= self.left_seconds:
            raise ValueError("interval contribution must have positive duration.")
        if not math.isclose(
            self.feasible_time_weight + self.infeasible_time_weight,
            self.interval_weight,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("interval feasible and infeasible weights must conserve mass.")


@dataclass(frozen=True, slots=True)
class DepartureQuadratureResult:
    """Compressed numerical integration result for one OD-period cell."""

    cell_key: ResponseCellKey
    responses: tuple[WeightedSparseDepartureResponse, ...]
    diagnostics: DepartureQuadratureDiagnostics

    def __post_init__(self) -> None:
        total = sum(item.weight for item in self.responses)
        if not math.isclose(total, 1.0, abs_tol=1.0e-12, rel_tol=0.0):
            raise ValueError("quadrature response weights must sum to one.")

    @property
    def averaged_response(self) -> SparseWeightedResponse:
        accumulator: dict[int, float] = {}
        for item in self.responses:
            if not item.feasible:
                continue
            for index, value in zip(
                item.response.measurement_indices, item.response.values, strict=True
            ):
                accumulator[index] = accumulator.get(index, 0.0) + item.weight * value
        indices = tuple(sorted(accumulator))
        return SparseWeightedResponse(
            indices, tuple(accumulator[index] for index in indices)
        )


def _sampling_groups(
    *,
    origin_physical_stop_ids: Sequence[str],
    origin_period_groups: Sequence[tuple[str, str]],
    time_periods: Sequence[JourneyTimePeriod],
) -> tuple[tuple[tuple[str, str], ...], dict[str, JourneyTimePeriod]]:
    if origin_physical_stop_ids and origin_period_groups:
        raise ValueError("provide sparse groups or Cartesian origins, not both.")
    periods = tuple(time_periods)
    if periods != tuple(sorted(periods, key=lambda item: (item.start_seconds, item.period_id))):
        raise ValueError("time periods must be sorted.")
    by_id = {item.period_id: item for item in periods}
    if len(by_id) != len(periods):
        raise ValueError("time period identifiers must be unique.")
    groups = (
        tuple(sorted(origin_period_groups))
        if origin_period_groups
        else tuple(
            (origin, period.period_id)
            for origin in sorted(set(origin_physical_stop_ids))
            for period in periods
        )
    )
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("origin-period groups must be non-empty and unique.")
    if any(not origin or period not in by_id for origin, period in groups):
        raise ValueError("origin-period groups contain an invalid identifier.")
    return groups, by_id


def generate_fixed_time_step_samples(
    *,
    origin_physical_stop_ids: Sequence[str] = (),
    origin_period_groups: Sequence[tuple[str, str]] = (),
    time_periods: Sequence[JourneyTimePeriod],
    config: DepartureTimeSamplingConfig,
    progress: SamplingProgress | None = None,
) -> tuple[DesiredDepartureSample, ...]:
    """Partition periods into exact-duration bins and evaluate each midpoint."""
    groups, periods = _sampling_groups(
        origin_physical_stop_ids=origin_physical_stop_ids,
        origin_period_groups=origin_period_groups,
        time_periods=time_periods,
    )
    started = time.perf_counter()
    samples: list[DesiredDepartureSample] = []
    total_groups = len(groups)
    for completed, (origin, period_id) in enumerate(groups, start=1):
        period = periods[period_id]
        duration = float(period.end_seconds - period.start_seconds)
        step = config.step_for_period(period_id)
        left = float(period.start_seconds)
        index = 0
        while left < period.end_seconds:
            right = min(float(period.end_seconds), left + step)
            midpoint = 0.5 * (left + right)
            payload = [
                "fixed_time_step",
                QUADRATURE_SCHEMA_VERSION,
                origin,
                period_id,
                left,
                right,
            ]
            samples.append(
                DesiredDepartureSample(
                    hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
                    origin,
                    period_id,
                    midpoint,
                    (right - left) / duration,
                )
            )
            left = right
            index += 1
        if progress is not None:
            elapsed = time.perf_counter() - started
            progress(
                {
                    "phase": "departure_sampling",
                    "status": "completed" if completed == total_groups else "in_progress",
                    "strategy": "fixed_time_step",
                    "completed_origin_period_groups": completed,
                    "total_origin_period_groups": total_groups,
                    "completed_queries": len(samples),
                    "elapsed_seconds": elapsed,
                    "throughput_groups_per_second": completed / max(elapsed, 1.0e-12),
                    "estimated_remaining_seconds": elapsed * (total_groups - completed) / completed,
                    "peak_rss_bytes": _peak_rss_bytes(),
                }
            )
    validate_sample_weights(samples, tolerance=config.weight_tolerance)
    return tuple(samples)


def sparse_relative_response_error(
    left: SparseWeightedResponse | None,
    right: SparseWeightedResponse | None,
) -> tuple[float, bool]:
    """Return symmetric normalized L1 error and whether sparse support changed."""
    if left is None or right is None:
        return (0.0, False) if left is right else (1.0, True)
    support_changed = left.measurement_indices != right.measurement_indices
    left_values = dict(zip(left.measurement_indices, left.values, strict=True))
    right_values = dict(zip(right.measurement_indices, right.values, strict=True))
    support = set(left_values) | set(right_values)
    difference = sum(
        abs(left_values.get(index, 0.0) - right_values.get(index, 0.0))
        for index in support
    )
    scale = max(
        sum(abs(value) for value in left_values.values()),
        sum(abs(value) for value in right_values.values()),
        1.0e-15,
    )
    return difference / scale, support_changed


def _integrate_adaptive_departure_response_depth_first(
    *,
    cell_key: ResponseCellKey,
    start_seconds: float,
    end_seconds: float,
    evaluator: SparseResponseEvaluator,
    config: DepartureTimeSamplingConfig,
    progress: SamplingProgress | None = None,
) -> DepartureQuadratureResult:
    """Adaptively integrate one sparse response while preserving elapsed-time mass."""
    if config.strategy != "adaptive_service_aware":
        raise ValueError("adaptive integration requires adaptive_service_aware strategy.")
    if config.infeasible_policy not in {"preserve_mass", "retain_unserved_mass"}:
        raise ValueError("adaptive integration requires preserve_mass infeasibility policy.")
    if not math.isfinite(start_seconds) or not math.isfinite(end_seconds) or end_seconds <= start_seconds:
        raise ValueError("departure interval must be finite and have positive duration.")
    started = time.perf_counter()
    duration = end_seconds - start_seconds
    cache: dict[float, SparseWeightedResponse | None] = {}
    cache_hits = 0
    accepted: list[tuple[float, SparseWeightedResponse | None, float]] = []
    refined = 0
    support_changes = 0
    depth_counts: dict[int, int] = {}
    estimated_error_mass = 0.0
    unresolved_weight = 0.0
    cap_reached = False
    minimum_unstable = False
    depth_unstable = False

    def evaluate(seconds: float) -> SparseWeightedResponse | None:
        nonlocal cache_hits
        key = float(seconds)
        if key in cache:
            cache_hits += 1
            return cache[key]
        if len(cache) >= config.maximum_samples_per_cell:
            raise RuntimeError("sample_cap")
        value = evaluator(key)
        if value is not None and not isinstance(value, SparseWeightedResponse):
            raise TypeError("adaptive evaluator must return SparseWeightedResponse or None.")
        cache[key] = value
        return value

    def accept(
        left: float,
        right: float,
        depth: int,
        response: SparseWeightedResponse | None,
        *,
        error: float,
        unresolved: bool,
    ) -> None:
        nonlocal estimated_error_mass, unresolved_weight
        weight = (right - left) / duration
        accepted.append((weight, response, 0.5 * (left + right)))
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
        estimated_error_mass += weight * error
        if unresolved:
            unresolved_weight += weight

    def refine(left: float, right: float, depth: int, center: SparseWeightedResponse | None = None) -> None:
        nonlocal refined, support_changes, cap_reached, minimum_unstable, depth_unstable
        midpoint = 0.5 * (left + right)
        try:
            middle = evaluate(midpoint) if center is None else center
            first = evaluate(0.5 * (left + midpoint))
            third = evaluate(0.5 * (midpoint + right))
        except RuntimeError as error:
            if str(error) != "sample_cap":
                raise
            cap_reached = True
            middle = cache.get(midpoint)
            if midpoint not in cache:
                raise RuntimeError("sample cap is too small to evaluate interval midpoints.") from error
            accept(left, right, depth, middle, error=1.0, unresolved=True)
            return
        errors = [
            sparse_relative_response_error(first, middle),
            sparse_relative_response_error(middle, third),
            sparse_relative_response_error(first, third),
        ]
        support_changes += sum(int(changed) for _, changed in errors)
        local_error = max(value for value, _ in errors)
        unstable = local_error > config.response_tolerance
        maximum_depth = config.maximum_refinement_depth
        at_minimum = (right - left) <= config.minimum_interval_seconds
        at_depth = maximum_depth is not None and depth >= maximum_depth
        if not unstable:
            accept(left, right, depth, middle, error=local_error, unresolved=False)
        elif at_minimum or at_depth:
            minimum_unstable = minimum_unstable or at_minimum
            depth_unstable = depth_unstable or at_depth
            accept(left, right, depth, middle, error=local_error, unresolved=True)
        else:
            refined += 1
            refine(left, midpoint, depth + 1, first)
            refine(midpoint, right, depth + 1, third)

    initial = max(1, math.ceil(duration / config.initial_interval_seconds))
    if initial > config.maximum_samples_per_cell:
        raise ValueError(
            "maximum_samples_per_cell must cover every initial subinterval midpoint."
        )
    edges = np.linspace(start_seconds, end_seconds, initial + 1)
    initial_centers = [
        evaluate(0.5 * (float(edges[index]) + float(edges[index + 1])))
        for index in range(initial)
    ]
    for index in range(initial):
        refine(
            float(edges[index]),
            float(edges[index + 1]),
            0,
            initial_centers[index],
        )
        if progress is not None:
            elapsed = time.perf_counter() - started
            progress(
                {
                    "phase": "adaptive_departure_quadrature",
                    "status": "completed" if index + 1 == initial else "in_progress",
                    "cell_key": cell_key.tuple,
                    "completed_initial_subintervals": index + 1,
                    "total_initial_subintervals": initial,
                    "routing_evaluations": len(cache),
                    "cache_hits": cache_hits,
                    "accepted_subintervals": len(accepted),
                    "refined_subintervals": refined,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": elapsed * (initial - index - 1) / (index + 1),
                    "current_infeasible_fraction": sum(
                        weight
                        for weight, response, _ in accepted
                        if response is None
                    ),
                    "sample_cap_reached": cap_reached,
                    "unresolved_interval_weight": unresolved_weight,
                    "peak_rss_bytes": _peak_rss_bytes(),
                }
            )

    merged: dict[tuple[bool, tuple[int, ...], tuple[str, ...]], float] = {}
    response_by_key: dict[
        tuple[bool, tuple[int, ...], tuple[str, ...]], SparseWeightedResponse
    ] = {}
    representatives_by_key: dict[
        tuple[bool, tuple[int, ...], tuple[str, ...]], list[float]
    ] = {}
    zero = SparseWeightedResponse((), ())
    for weight, response, representative in accepted:
        feasible = response is not None
        material = zero if response is None else response
        key = (
            feasible,
            material.measurement_indices,
            tuple(float(value).hex() for value in material.values),
        )
        merged[key] = merged.get(key, 0.0) + weight
        response_by_key[key] = material
        representatives_by_key.setdefault(key, []).append(representative)
    responses = tuple(
        WeightedSparseDepartureResponse(
            merged[key],
            key[0],
            response_by_key[key],
            tuple(representatives_by_key[key]),
        )
        for key in sorted(merged)
    )
    total_weight = sum(item.weight for item in responses)
    infeasible = sum(item.weight for item in responses if not item.feasible)
    elapsed = time.perf_counter() - started
    diagnostic_payload = {
        "cell_key": cell_key.tuple,
        "config": asdict(config),
        "responses": [asdict(item) for item in responses],
        "unresolved_interval_weight": unresolved_weight,
        "schema": QUADRATURE_SCHEMA_VERSION,
    }
    fingerprint = hashlib.sha256(canonical_json(diagnostic_payload).encode()).hexdigest()
    diagnostics = DepartureQuadratureDiagnostics(
        strategy=config.strategy,
        configuration_fingerprint=config.fingerprint,
        quadrature_schema_version=QUADRATURE_SCHEMA_VERSION,
        interval_seconds=duration,
        initial_subintervals=initial,
        routing_evaluations=len(cache),
        cache_hits=cache_hits,
        accepted_subintervals=len(accepted),
        refined_subintervals=refined,
        refinement_depth_counts=tuple(sorted(depth_counts.items())),
        unique_evaluated_times=len(cache),
        unique_responses=len(responses),
        merged_responses=len(accepted) - len(responses),
        response_support_changes=support_changes,
        feasible_time_fraction=1.0 - infeasible,
        infeasible_time_fraction=infeasible,
        total_quadrature_weight=total_weight,
        estimated_relative_response_error=estimated_error_mass,
        unresolved_interval_weight=unresolved_weight,
        sample_cap_reached=cap_reached,
        minimum_resolution_reached_with_instability=minimum_unstable,
        maximum_depth_reached_with_instability=depth_unstable,
        quadrature_converged=unresolved_weight <= config.weight_tolerance,
        elapsed_seconds=elapsed,
        peak_rss_bytes=_peak_rss_bytes(),
        fingerprint=fingerprint,
    )
    return DepartureQuadratureResult(cell_key, responses, diagnostics)


def _weighted_sparse(
    response: SparseWeightedResponse | None, weight: float
) -> SparseWeightedResponse:
    if response is None:
        return SparseWeightedResponse((), ())
    return SparseWeightedResponse(
        response.measurement_indices,
        tuple(weight * value for value in response.values),
    )


def _sum_sparse(
    *responses: SparseWeightedResponse,
) -> SparseWeightedResponse:
    values: dict[int, float] = {}
    for response in responses:
        for index, value in zip(
            response.measurement_indices, response.values, strict=True
        ):
            values[index] = values.get(index, 0.0) + value
    indices = tuple(index for index in sorted(values) if values[index] != 0.0)
    return SparseWeightedResponse(indices, tuple(values[index] for index in indices))


def _sparse_l1(response: SparseWeightedResponse) -> float:
    return sum(abs(value) for value in response.values)


def _sparse_difference_l1(
    left: SparseWeightedResponse, right: SparseWeightedResponse
) -> tuple[float, int, int]:
    left_values = dict(zip(left.measurement_indices, left.values, strict=True))
    right_values = dict(zip(right.measurement_indices, right.values, strict=True))
    support = set(left_values) | set(right_values)
    difference = sum(
        abs(left_values.get(index, 0.0) - right_values.get(index, 0.0))
        for index in support
    )
    additions = len(set(right_values) - set(left_values))
    removals = len(set(left_values) - set(right_values))
    return difference, additions, removals


def add_sparse_interval_contributions(
    left: SparseIntervalContribution,
    right: SparseIntervalContribution,
) -> SparseIntervalContribution:
    """Add adjacent sparse interval integrals without densification."""
    if not math.isclose(left.right_seconds, right.left_seconds):
        raise ValueError("interval contributions must be adjacent.")
    return SparseIntervalContribution(
        left.left_seconds,
        right.right_seconds,
        f"sum({left.quadrature_rule},{right.quadrature_rule})",
        tuple(sorted(set(left.evaluated_times + right.evaluated_times))),
        _sum_sparse(left.weighted_response, right.weighted_response),
        left.feasible_time_weight + right.feasible_time_weight,
        left.infeasible_time_weight + right.infeasible_time_weight,
        left.interval_weight + right.interval_weight,
    )


@dataclass(frozen=True, slots=True)
class _IntegralCandidate:
    left: float
    right: float
    depth: int
    midpoint_response: SparseWeightedResponse | None
    left_midpoint_response: SparseWeightedResponse | None
    right_midpoint_response: SparseWeightedResponse | None
    coarse: SparseIntervalContribution
    refined: SparseIntervalContribution
    absolute_error: float
    support_additions: int
    support_removals: int


def _integral_initial_edges(
    *,
    start_seconds: float,
    end_seconds: float,
    initial_interval_seconds: float,
    maximum_base_intervals: int,
    maximum_intervals: int,
    service_boundary_seconds: Sequence[float],
) -> tuple[np.ndarray, int, tuple[float, ...]]:
    duration = end_seconds - start_seconds
    base_count = min(
        maximum_base_intervals,
        max(1, math.ceil(duration / initial_interval_seconds)),
    )
    base = list(np.linspace(start_seconds, end_seconds, base_count + 1))
    boundaries = sorted(
        {
            float(value)
            for value in service_boundary_seconds
            if start_seconds < float(value) < end_seconds
        }
    )
    available = max(0, maximum_intervals - base_count)
    selected: list[float] = []
    if boundaries and available:
        if len(boundaries) <= available:
            selected = boundaries
        else:
            positions = np.linspace(0, len(boundaries) - 1, available)
            selected = [boundaries[int(round(position))] for position in positions]
        base.extend(selected)
    edges = np.asarray(sorted(set(base)), dtype=float)
    omitted = tuple(value for value in boundaries if value not in set(selected))
    return edges, len(edges) - 1 - base_count, omitted


def _integrate_integral_adaptive_departure_response(
    *,
    cell_key: ResponseCellKey,
    start_seconds: float,
    end_seconds: float,
    evaluator: SparseResponseEvaluator,
    config: DepartureTimeSamplingConfig,
    progress: SamplingProgress | None,
    service_boundary_seconds: Sequence[float],
) -> DepartureQuadratureResult:
    """Embedded midpoint integration with deterministic global error control."""
    if config.infeasible_policy not in {"preserve_mass", "retain_unserved_mass"}:
        raise ValueError("integral adaptation requires preserve_mass infeasibility policy.")
    if end_seconds <= start_seconds:
        raise ValueError("departure interval must have positive duration.")
    started = time.perf_counter()
    duration = end_seconds - start_seconds
    budget = config.maximum_samples_per_cell
    if budget < 3:
        raise ValueError("integral adaptation requires at least three evaluations.")
    maximum_initial_intervals = budget // 4
    maximum_base_intervals = budget // 6
    if maximum_initial_intervals < 1:
        maximum_initial_intervals = 1
    if maximum_base_intervals < 1:
        maximum_base_intervals = 1
    edges, boundary_edges, omitted_boundaries = _integral_initial_edges(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        initial_interval_seconds=config.initial_interval_seconds,
        maximum_base_intervals=maximum_base_intervals,
        maximum_intervals=maximum_initial_intervals,
        service_boundary_seconds=(
            service_boundary_seconds if config.service_boundary_safeguard else ()
        ),
    )
    initial = len(edges) - 1
    reserved_baseline = 3 * initial
    cache: dict[float, SparseWeightedResponse | None] = {}
    cache_hits = 0
    total_support_additions = 0
    total_support_removals = 0
    evaluation_durations: deque[float] = deque(maxlen=10)

    def evaluate(seconds: float) -> SparseWeightedResponse | None:
        nonlocal cache_hits
        key = float(seconds)
        if key in cache:
            cache_hits += 1
            return cache[key]
        if len(cache) >= budget:
            raise RuntimeError("sample_cap")
        evaluation_started = time.perf_counter()
        response = evaluator(key)
        evaluation_durations.append(time.perf_counter() - evaluation_started)
        if response is not None and not isinstance(response, SparseWeightedResponse):
            raise TypeError("adaptive evaluator must return SparseWeightedResponse or None.")
        cache[key] = response
        return response

    def contribution(
        left: float,
        right: float,
        evaluated_times: tuple[float, ...],
        responses: tuple[SparseWeightedResponse | None, ...],
        rule: str,
    ) -> SparseIntervalContribution:
        interval_weight = (right - left) / duration
        sample_weight = interval_weight / len(responses)
        weighted = _sum_sparse(
            *(_weighted_sparse(response, sample_weight) for response in responses)
        )
        feasible = sample_weight * sum(response is not None for response in responses)
        return SparseIntervalContribution(
            left,
            right,
            rule,
            evaluated_times,
            weighted,
            feasible,
            interval_weight - feasible,
            interval_weight,
        )

    def characterize(
        left: float,
        right: float,
        depth: int,
        midpoint_response: SparseWeightedResponse | None = None,
    ) -> _IntegralCandidate:
        nonlocal total_support_additions, total_support_removals
        midpoint = 0.5 * (left + right)
        left_midpoint = 0.5 * (left + midpoint)
        right_midpoint = 0.5 * (midpoint + right)
        center = evaluate(midpoint) if midpoint_response is None else midpoint_response
        left_response = evaluate(left_midpoint)
        right_response = evaluate(right_midpoint)
        coarse = contribution(
            left, right, (midpoint,), (center,), "midpoint_coarse"
        )
        refined = contribution(
            left,
            right,
            (left_midpoint, right_midpoint),
            (left_response, right_response),
            "two_child_midpoints",
        )
        error, additions, removals = _sparse_difference_l1(
            coarse.weighted_response, refined.weighted_response
        )
        total_support_additions += additions
        total_support_removals += removals
        if not math.isclose(
            coarse.feasible_time_weight,
            refined.feasible_time_weight,
            abs_tol=config.weight_tolerance,
            rel_tol=0.0,
        ):
            error += abs(
                coarse.feasible_time_weight - refined.feasible_time_weight
            )
        if any(left < boundary < right for boundary in omitted_boundaries):
            # A bounded partition could not seed every timetable boundary.
            # Preserve uncertainty until refinement isolates each omitted edge.
            error = max(error, (right - left) / duration)
        return _IntegralCandidate(
            left,
            right,
            depth,
            center,
            left_response,
            right_response,
            coarse,
            refined,
            error,
            additions,
            removals,
        )

    leaves: dict[tuple[float, float], _IntegralCandidate] = {}
    queue: list[tuple[float, float, int, int, _IntegralCandidate]] = []
    sequence = 0
    minimum_count = 0
    depth_count = 0

    def refinable(candidate: _IntegralCandidate) -> bool:
        at_minimum = candidate.right - candidate.left <= config.minimum_interval_seconds
        at_depth = (
            config.maximum_refinement_depth is not None
            and candidate.depth >= config.maximum_refinement_depth
        )
        return not at_minimum and not at_depth

    def enqueue(candidate: _IntegralCandidate) -> None:
        nonlocal sequence, minimum_count, depth_count
        if refinable(candidate) and candidate.absolute_error > 0.0:
            heapq.heappush(
                queue,
                (
                    -candidate.absolute_error,
                    candidate.left,
                    candidate.depth,
                    sequence,
                    candidate,
                ),
            )
            sequence += 1
        elif candidate.absolute_error > 0.0:
            minimum_count += int(
                candidate.right - candidate.left <= config.minimum_interval_seconds
            )
            depth_count += int(
                config.maximum_refinement_depth is not None
                and candidate.depth >= config.maximum_refinement_depth
            )

    for index in range(initial):
        candidate = characterize(float(edges[index]), float(edges[index + 1]), 0)
        leaves[(candidate.left, candidate.right)] = candidate
        enqueue(candidate)

    initial_coarse_norm = _sparse_l1(
        _sum_sparse(*(item.coarse.weighted_response for item in leaves.values()))
    )
    refinements = 0
    sample_cap_reached = False

    def state() -> tuple[SparseWeightedResponse, float, float, float, bool]:
        estimate = _sum_sparse(
            *(item.refined.weighted_response for item in leaves.values())
        )
        norm = _sparse_l1(estimate)
        error = sum(item.absolute_error for item in leaves.values())
        target = config.absolute_response_tolerance + (
            config.effective_relative_response_tolerance
            * max(norm, config.integration_scale_floor)
        )
        return estimate, norm, error, target, error <= target

    def emit(status: str) -> None:
        if progress is None:
            return
        _, norm, error, target, achieved = state()
        elapsed = time.perf_counter() - started
        recent = float(np.mean(evaluation_durations)) if evaluation_durations else None
        progress(
            {
                "phase": "adaptive_departure_quadrature",
                "status": status,
                "cell_key": cell_key.tuple,
                "quadrature_rule": "embedded_midpoint_integral",
                "requested_comparison_mode": config.comparison_mode,
                "effective_comparison_mode": "integral_response",
                "routing_evaluations": len(cache),
                "evaluation_budget": budget,
                "initial_subintervals": initial,
                "final_interval_count": len(leaves),
                "refined_subintervals": refinements,
                "refined_integral_norm": norm,
                "estimated_absolute_integration_error": error,
                "estimated_relative_integration_error": error
                / max(norm, config.integration_scale_floor),
                "global_error_target": target,
                "global_target_achieved": achieved,
                "unresolved_interval_weight": sum(
                    item.refined.interval_weight
                    for item in leaves.values()
                    if item.absolute_error > 0.0
                ),
                "sample_cap_reached": sample_cap_reached,
                "elapsed_seconds": elapsed,
                "recent_routing_evaluation_seconds": recent,
                "estimated_remaining_seconds": (
                    None if recent is None else recent * max(0, budget - len(cache))
                ),
                "eta_confidence": "high" if achieved else "medium",
                "peak_rss_bytes": _peak_rss_bytes(),
            }
        )

    emit("in_progress")
    while True:
        _, _, _, _, achieved = state()
        if achieved:
            break
        while queue:
            _, _, _, _, candidate = heapq.heappop(queue)
            if leaves.get((candidate.left, candidate.right)) is candidate:
                break
        else:
            break
        if budget - len(cache) < 4:
            sample_cap_reached = True
            break
        midpoint = 0.5 * (candidate.left + candidate.right)
        left_child = characterize(
            candidate.left,
            midpoint,
            candidate.depth + 1,
            candidate.left_midpoint_response,
        )
        right_child = characterize(
            midpoint,
            candidate.right,
            candidate.depth + 1,
            candidate.right_midpoint_response,
        )
        del leaves[(candidate.left, candidate.right)]
        for child in (left_child, right_child):
            leaves[(child.left, child.right)] = child
            enqueue(child)
        refinements += 1
        emit("in_progress")

    estimate, refined_norm, total_error, target, target_achieved = state()
    unresolved_candidates = (
        ()
        if target_achieved
        else tuple(item for item in leaves.values() if item.absolute_error > 0.0)
    )
    unresolved_weight = sum(
        item.refined.interval_weight for item in unresolved_candidates
    )
    unresolved_error = sum(item.absolute_error for item in unresolved_candidates)

    samples: list[tuple[float, SparseWeightedResponse | None, float]] = []
    for candidate in sorted(leaves.values(), key=lambda item: item.left):
        half_weight = 0.5 * candidate.refined.interval_weight
        left_time, right_time = candidate.refined.evaluated_times
        samples.extend(
            (
                (half_weight, candidate.left_midpoint_response, left_time),
                (half_weight, candidate.right_midpoint_response, right_time),
            )
        )
    merged: dict[tuple[bool, tuple[int, ...], tuple[str, ...]], float] = {}
    response_by_key: dict[
        tuple[bool, tuple[int, ...], tuple[str, ...]], SparseWeightedResponse
    ] = {}
    representatives: dict[
        tuple[bool, tuple[int, ...], tuple[str, ...]], list[float]
    ] = {}
    zero = SparseWeightedResponse((), ())
    for weight, response, seconds in samples:
        material = zero if response is None else response
        key = (
            response is not None,
            material.measurement_indices,
            tuple(float(value).hex() for value in material.values),
        )
        merged[key] = merged.get(key, 0.0) + weight
        response_by_key[key] = material
        representatives.setdefault(key, []).append(seconds)
    responses = tuple(
        WeightedSparseDepartureResponse(
            merged[key], key[0], response_by_key[key], tuple(representatives[key])
        )
        for key in sorted(merged)
    )
    total_mass = canonicalize_probability_mass(
        sum(item.weight for item in responses),
        tolerance=config.weight_tolerance,
        name="integral quadrature weight",
    )
    if total_mass.applied:
        correction = total_mass.canonical_value - total_mass.raw_value
        last = responses[-1]
        responses = responses[:-1] + (
            WeightedSparseDepartureResponse(
                last.weight + correction,
                last.feasible,
                last.response,
                last.representative_seconds,
            ),
        )
    infeasible = sum(item.weight for item in responses if not item.feasible)
    durations: dict[float, int] = {}
    for item in leaves.values():
        interval_duration = item.right - item.left
        durations[interval_duration] = durations.get(interval_duration, 0) + 1
    largest = tuple(
        (item.left, item.right, item.absolute_error)
        for item in sorted(
            leaves.values(),
            key=lambda item: (-item.absolute_error, item.left, item.depth),
        )[:10]
    )
    additions = total_support_additions
    removals = total_support_removals
    elapsed = time.perf_counter() - started
    fingerprint = hashlib.sha256(
        canonical_json(
            {
                "cell_key": cell_key.tuple,
                "config": asdict(config),
                "quadrature_rule": "embedded_midpoint_integral",
                "responses": [asdict(item) for item in responses],
                "schema": QUADRATURE_SCHEMA_VERSION,
                "service_boundary_seconds": list(service_boundary_seconds),
            }
        ).encode()
    ).hexdigest()
    diagnostics = DepartureQuadratureDiagnostics(
        strategy=config.strategy,
        configuration_fingerprint=config.fingerprint,
        quadrature_schema_version=QUADRATURE_SCHEMA_VERSION,
        interval_seconds=duration,
        initial_subintervals=initial,
        routing_evaluations=len(cache),
        cache_hits=cache_hits,
        accepted_subintervals=len(leaves),
        refined_subintervals=refinements,
        refinement_depth_counts=tuple(
            sorted(
                {
                    depth: sum(item.depth == depth for item in leaves.values())
                    for depth in {item.depth for item in leaves.values()}
                }.items()
            )
        ),
        unique_evaluated_times=len(cache),
        unique_responses=len(responses),
        merged_responses=len(samples) - len(responses),
        response_support_changes=additions + removals,
        feasible_time_fraction=1.0 - infeasible,
        infeasible_time_fraction=infeasible,
        total_quadrature_weight=sum(item.weight for item in responses),
        estimated_relative_response_error=total_error
        / max(refined_norm, config.integration_scale_floor),
        unresolved_interval_weight=unresolved_weight,
        sample_cap_reached=sample_cap_reached,
        minimum_resolution_reached_with_instability=minimum_count > 0,
        maximum_depth_reached_with_instability=depth_count > 0,
        quadrature_converged=target_achieved,
        elapsed_seconds=elapsed,
        peak_rss_bytes=_peak_rss_bytes(),
        fingerprint=fingerprint,
        requested_comparison_mode=config.comparison_mode,
        effective_comparison_mode="integral_response",
        budget_scope="origin_period_group",
        initial_subintervals_evaluated=initial,
        evaluation_budget=budget,
        reserved_baseline_evaluations=reserved_baseline,
        baseline_evaluations=reserved_baseline,
        refinement_evaluations=len(cache) - reserved_baseline,
        stable_interval_weight=1.0 - unresolved_weight,
        quadrature_rule="embedded_midpoint_integral",
        absolute_response_tolerance=config.absolute_response_tolerance,
        relative_response_tolerance=config.effective_relative_response_tolerance,
        integration_scale_floor=config.integration_scale_floor,
        coarse_integral_norm=initial_coarse_norm,
        refined_integral_norm=refined_norm,
        estimated_absolute_integration_error=total_error,
        global_error_target=target,
        global_target_achieved=target_achieved,
        unresolved_estimated_error=unresolved_error,
        support_additions=additions,
        support_removals=removals,
        final_interval_count=len(leaves),
        minimum_resolution_interval_count=minimum_count,
        largest_error_intervals=largest,
        interval_duration_distribution=tuple(sorted(durations.items())),
        service_boundary_safeguard=(
            f"bounded_timetable_edges:{boundary_edges};"
            f"omitted_boundaries:{len(omitted_boundaries)}"
            if config.service_boundary_safeguard
            else "disabled"
        ),
    )
    emit("completed")
    return DepartureQuadratureResult(cell_key, responses, diagnostics)


@dataclass(frozen=True, slots=True)
class _RefinementCandidate:
    left: float
    right: float
    depth: int
    left_response: SparseWeightedResponse | None
    middle_response: SparseWeightedResponse | None
    right_response: SparseWeightedResponse | None
    error: float


def _integrate_pointwise_adaptive_departure_response(
    *,
    cell_key: ResponseCellKey,
    start_seconds: float,
    end_seconds: float,
    evaluator: SparseResponseEvaluator,
    config: DepartureTimeSamplingConfig,
    progress: SamplingProgress | None = None,
    effective_comparison_mode: str | None = None,
) -> DepartureQuadratureResult:
    """Integrate sparsely after whole-period baseline coverage and priority refinement."""
    if config.strategy != "adaptive_service_aware":
        raise ValueError("adaptive integration requires adaptive_service_aware strategy.")
    if config.infeasible_policy not in {"preserve_mass", "retain_unserved_mass"}:
        raise ValueError("adaptive integration requires preserve_mass infeasibility policy.")
    if not math.isfinite(start_seconds) or not math.isfinite(end_seconds) or end_seconds <= start_seconds:
        raise ValueError("departure interval must be finite and have positive duration.")
    effective_mode = effective_comparison_mode or config.comparison_mode
    if effective_mode not in ADAPTIVE_COMPARISON_MODES:
        raise ValueError("effective comparison mode is unsupported.")

    started = time.perf_counter()
    duration = end_seconds - start_seconds
    initial = max(1, math.ceil(duration / config.initial_interval_seconds))
    reserved_baseline = 2 * initial + 1
    budget = config.maximum_samples_per_cell
    if budget < reserved_baseline:
        raise ValueError(
            "maximum_samples_per_cell is an origin-period-group budget and must "
            f"be at least {reserved_baseline} for {initial} initial subintervals."
        )

    cache: dict[float, SparseWeightedResponse | None] = {}
    cache_hits = 0
    evaluation_durations: deque[float] = deque(maxlen=10)
    support_changes = 0
    refined = 0
    minimum_unstable = False
    depth_unstable = False
    accepted: list[
        tuple[float, SparseWeightedResponse | None, float, int, float, bool]
    ] = []
    queue: list[tuple[float, float, int, int, _RefinementCandidate]] = []
    sequence = 0
    budget_exhausted = False

    def evaluate(seconds: float) -> SparseWeightedResponse | None:
        nonlocal cache_hits
        key = float(seconds)
        if key in cache:
            cache_hits += 1
            return cache[key]
        if len(cache) >= budget:
            raise RuntimeError("sample_cap")
        evaluation_started = time.perf_counter()
        result = evaluator(key)
        evaluation_durations.append(time.perf_counter() - evaluation_started)
        if result is not None and not isinstance(result, SparseWeightedResponse):
            raise TypeError("adaptive evaluator must return SparseWeightedResponse or None.")
        cache[key] = result
        return result

    def characterize(
        left: float,
        right: float,
        depth: int,
        left_response: SparseWeightedResponse | None,
        middle_response: SparseWeightedResponse | None,
        right_response: SparseWeightedResponse | None,
    ) -> _RefinementCandidate:
        nonlocal support_changes
        comparisons = (
            sparse_relative_response_error(left_response, middle_response),
            sparse_relative_response_error(middle_response, right_response),
            sparse_relative_response_error(left_response, right_response),
        )
        support_changes += sum(int(changed) for _, changed in comparisons)
        return _RefinementCandidate(
            left,
            right,
            depth,
            left_response,
            middle_response,
            right_response,
            max(value for value, _ in comparisons),
        )

    def interval_weight(candidate: _RefinementCandidate) -> float:
        return (candidate.right - candidate.left) / duration

    def accept(candidate: _RefinementCandidate, *, unresolved: bool) -> None:
        accepted.append(
            (
                interval_weight(candidate),
                candidate.middle_response,
                0.5 * (candidate.left + candidate.right),
                candidate.depth,
                candidate.error,
                unresolved,
            )
        )

    def enqueue(candidate: _RefinementCandidate) -> None:
        nonlocal sequence
        priority = interval_weight(candidate) * candidate.error
        heapq.heappush(
            queue,
            (-priority, candidate.left, candidate.depth, sequence, candidate),
        )
        sequence += 1

    def classify(candidate: _RefinementCandidate) -> None:
        nonlocal minimum_unstable, depth_unstable
        if candidate.error <= config.response_tolerance:
            accept(candidate, unresolved=False)
            return
        at_minimum = (
            candidate.right - candidate.left <= config.minimum_interval_seconds
        )
        at_depth = (
            config.maximum_refinement_depth is not None
            and candidate.depth >= config.maximum_refinement_depth
        )
        if at_minimum or at_depth:
            minimum_unstable = minimum_unstable or at_minimum
            depth_unstable = depth_unstable or at_depth
            accept(candidate, unresolved=True)
        else:
            enqueue(candidate)

    def emit(status: str, *, completed_initial: int, eta_confidence: str) -> None:
        if progress is None:
            return
        elapsed = time.perf_counter() - started
        remaining_budget = budget - len(cache)
        estimated_evaluations = min(2 * len(queue), remaining_budget)
        recent = (
            float(np.mean(evaluation_durations)) if evaluation_durations else None
        )
        stable_weight = sum(item[0] for item in accepted if not item[5])
        unresolved_weight = sum(item[0] for item in accepted if item[5])
        progress(
            {
                "phase": "adaptive_departure_quadrature",
                "status": status,
                "cell_key": cell_key.tuple,
                "completed_initial_subintervals": completed_initial,
                "total_initial_subintervals": initial,
                "routing_evaluations": len(cache),
                "evaluation_budget": budget,
                "reserved_baseline_evaluations": reserved_baseline,
                "remaining_refinement_budget": remaining_budget,
                "baseline_evaluations": min(len(cache), reserved_baseline),
                "refinement_evaluations": max(0, len(cache) - reserved_baseline),
                "accepted_subintervals": len(accepted),
                "refined_subintervals": refined,
                "cache_hits": cache_hits,
                "stable_interval_weight": stable_weight,
                "unresolved_interval_weight": unresolved_weight,
                "sample_cap_reached": budget_exhausted
                or (remaining_budget < 2 and bool(queue)),
                "requested_comparison_mode": config.comparison_mode,
                "effective_comparison_mode": effective_mode,
                "elapsed_seconds": elapsed,
                "recent_routing_evaluation_seconds": recent,
                "estimated_remaining_routing_evaluations": estimated_evaluations,
                "estimated_remaining_seconds": (
                    None if recent is None else recent * estimated_evaluations
                ),
                "eta_confidence": eta_confidence,
                "peak_rss_bytes": _peak_rss_bytes(),
            }
        )

    edges = np.linspace(start_seconds, end_seconds, initial + 1)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    edge_responses = [evaluate(float(value)) for value in edges]
    midpoint_responses = [evaluate(float(value)) for value in midpoints]
    for index in range(initial):
        classify(
            characterize(
                float(edges[index]),
                float(edges[index + 1]),
                0,
                edge_responses[index],
                midpoint_responses[index],
                edge_responses[index + 1],
            )
        )
        emit("in_progress", completed_initial=index + 1, eta_confidence="low")

    while queue and budget - len(cache) >= 2:
        _, _, _, _, candidate = heapq.heappop(queue)
        midpoint = 0.5 * (candidate.left + candidate.right)
        first_midpoint = 0.5 * (candidate.left + midpoint)
        second_midpoint = 0.5 * (midpoint + candidate.right)
        first_response = evaluate(first_midpoint)
        second_response = evaluate(second_midpoint)
        refined += 1
        classify(
            characterize(
                candidate.left,
                midpoint,
                candidate.depth + 1,
                candidate.left_response,
                first_response,
                candidate.middle_response,
            )
        )
        classify(
            characterize(
                midpoint,
                candidate.right,
                candidate.depth + 1,
                candidate.middle_response,
                second_response,
                candidate.right_response,
            )
        )
        emit("in_progress", completed_initial=initial, eta_confidence="medium")

    sample_cap_reached = bool(queue)
    budget_exhausted = sample_cap_reached
    while queue:
        _, _, _, _, candidate = heapq.heappop(queue)
        accept(candidate, unresolved=True)

    merged: dict[tuple[bool, tuple[int, ...], tuple[str, ...]], float] = {}
    response_by_key: dict[
        tuple[bool, tuple[int, ...], tuple[str, ...]], SparseWeightedResponse
    ] = {}
    representatives_by_key: dict[
        tuple[bool, tuple[int, ...], tuple[str, ...]], list[float]
    ] = {}
    zero = SparseWeightedResponse((), ())
    depth_counts: dict[int, int] = {}
    estimated_error_mass = 0.0
    unresolved_raw = 0.0
    stable_raw = 0.0
    for weight, response, representative, depth, error, unresolved in accepted:
        feasible = response is not None
        material = zero if response is None else response
        key = (
            feasible,
            material.measurement_indices,
            tuple(float(value).hex() for value in material.values),
        )
        merged[key] = merged.get(key, 0.0) + weight
        response_by_key[key] = material
        representatives_by_key.setdefault(key, []).append(representative)
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
        estimated_error_mass += weight * error
        unresolved_raw += weight if unresolved else 0.0
        stable_raw += 0.0 if unresolved else weight
    unresolved_mass = canonicalize_probability_mass(
        unresolved_raw,
        tolerance=config.weight_tolerance,
        name="unresolved interval weight",
    ).canonical_value
    stable_mass = canonicalize_probability_mass(
        stable_raw,
        tolerance=config.weight_tolerance,
        name="stable interval weight",
    ).canonical_value
    responses = tuple(
        WeightedSparseDepartureResponse(
            merged[key],
            key[0],
            response_by_key[key],
            tuple(representatives_by_key[key]),
        )
        for key in sorted(merged)
    )
    total_mass = canonicalize_probability_mass(
        sum(item.weight for item in responses),
        tolerance=config.weight_tolerance,
        name="total quadrature weight",
    )
    if total_mass.applied:
        correction = total_mass.canonical_value - total_mass.raw_value
        last = responses[-1]
        responses = responses[:-1] + (
            WeightedSparseDepartureResponse(
                last.weight + correction,
                last.feasible,
                last.response,
                last.representative_seconds,
            ),
        )
    infeasible = sum(item.weight for item in responses if not item.feasible)
    elapsed = time.perf_counter() - started
    payload = {
        "cell_key": cell_key.tuple,
        "config": asdict(config),
        "effective_comparison_mode": effective_mode,
        "responses": [asdict(item) for item in responses],
        "schema": QUADRATURE_SCHEMA_VERSION,
        "unresolved_interval_weight": unresolved_mass,
    }
    fingerprint = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    diagnostics = DepartureQuadratureDiagnostics(
        strategy=config.strategy,
        configuration_fingerprint=config.fingerprint,
        quadrature_schema_version=QUADRATURE_SCHEMA_VERSION,
        interval_seconds=duration,
        initial_subintervals=initial,
        routing_evaluations=len(cache),
        cache_hits=cache_hits,
        accepted_subintervals=len(accepted),
        refined_subintervals=refined,
        refinement_depth_counts=tuple(sorted(depth_counts.items())),
        unique_evaluated_times=len(cache),
        unique_responses=len(responses),
        merged_responses=len(accepted) - len(responses),
        response_support_changes=support_changes,
        feasible_time_fraction=1.0 - infeasible,
        infeasible_time_fraction=infeasible,
        total_quadrature_weight=sum(item.weight for item in responses),
        estimated_relative_response_error=estimated_error_mass,
        unresolved_interval_weight=unresolved_mass,
        sample_cap_reached=sample_cap_reached,
        minimum_resolution_reached_with_instability=minimum_unstable,
        maximum_depth_reached_with_instability=depth_unstable,
        quadrature_converged=unresolved_mass <= config.weight_tolerance,
        elapsed_seconds=elapsed,
        peak_rss_bytes=_peak_rss_bytes(),
        fingerprint=fingerprint,
        requested_comparison_mode=config.comparison_mode,
        effective_comparison_mode=effective_mode,
        budget_scope="origin_period_group",
        initial_subintervals_evaluated=initial,
        evaluation_budget=budget,
        reserved_baseline_evaluations=reserved_baseline,
        baseline_evaluations=reserved_baseline,
        refinement_evaluations=len(cache) - reserved_baseline,
        stable_interval_weight=stable_mass,
    )
    emit("completed", completed_initial=initial, eta_confidence="high")
    return DepartureQuadratureResult(cell_key, responses, diagnostics)


def integrate_adaptive_departure_response(
    *,
    cell_key: ResponseCellKey,
    start_seconds: float,
    end_seconds: float,
    evaluator: SparseResponseEvaluator,
    config: DepartureTimeSamplingConfig,
    progress: SamplingProgress | None = None,
    effective_comparison_mode: str | None = None,
    service_boundary_seconds: Sequence[float] = (),
) -> DepartureQuadratureResult:
    """Integrate by embedded interval error or the compatible pointwise rule."""
    effective_mode = effective_comparison_mode or config.comparison_mode
    if effective_mode == "integral_response":
        return _integrate_integral_adaptive_departure_response(
            cell_key=cell_key,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            evaluator=evaluator,
            config=config,
            progress=progress,
            service_boundary_seconds=service_boundary_seconds,
        )
    return _integrate_pointwise_adaptive_departure_response(
        cell_key=cell_key,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        evaluator=evaluator,
        config=config,
        progress=progress,
        effective_comparison_mode=effective_mode,
    )


def integrate_adaptive_departure_responses(
    *,
    cells: Sequence[tuple[ResponseCellKey, float, float]],
    evaluator: CellSparseResponseEvaluator,
    config: DepartureTimeSamplingConfig,
    progress: SamplingProgress | None = None,
) -> tuple[DepartureQuadratureResult, ...]:
    """Run adaptive quadrature over cells with aggregate progress and ETA."""
    ordered = tuple(sorted(cells, key=lambda item: item[0]))
    if not ordered or len({item[0] for item in ordered}) != len(ordered):
        raise ValueError("adaptive cells must be non-empty and unique.")
    started = time.perf_counter()
    results: list[DepartureQuadratureResult] = []
    for completed, (cell, left, right) in enumerate(ordered, start=1):
        def evaluate_cell(seconds: float) -> SparseWeightedResponse | None:
            return evaluator(cell, seconds)

        result = integrate_adaptive_departure_response(
            cell_key=cell,
            start_seconds=left,
            end_seconds=right,
            evaluator=evaluate_cell,
            config=config,
        )
        results.append(result)
        if progress is not None:
            elapsed = time.perf_counter() - started
            evaluations = sum(item.diagnostics.routing_evaluations for item in results)
            progress(
                {
                    "phase": "adaptive_departure_quadrature_batch",
                    "status": "completed" if completed == len(ordered) else "in_progress",
                    "completed_cells": completed,
                    "total_cells": len(ordered),
                    "routing_evaluations": evaluations,
                    "mean_evaluations_per_cell": evaluations / completed,
                    "accepted_subintervals": sum(item.diagnostics.accepted_subintervals for item in results),
                    "refined_subintervals": sum(item.diagnostics.refined_subintervals for item in results),
                    "cache_hits": sum(item.diagnostics.cache_hits for item in results),
                    "elapsed_seconds": elapsed,
                    "throughput_cells_per_second": completed / max(elapsed, 1.0e-12),
                    "estimated_remaining_seconds": elapsed * (len(ordered) - completed) / completed,
                    "current_infeasible_fraction": float(np.mean([item.diagnostics.infeasible_time_fraction for item in results])),
                    "sample_cap_count": sum(item.diagnostics.sample_cap_reached for item in results),
                    "unresolved_count": sum(item.diagnostics.unresolved_interval_weight > 0 for item in results),
                    "peak_rss_bytes": _peak_rss_bytes(),
                }
            )
    return tuple(results)
