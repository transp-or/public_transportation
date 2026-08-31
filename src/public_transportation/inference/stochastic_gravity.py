"""Memory-bounded experimental gravity objective and gradient evaluation.

The sampling units are the persisted fixed-routing shards.  Selected shards are
processed twice, one at a time, and no prepared forward batch is retained for
the reverse pass.

Full-network validation dated 2026-08-05 confirmed bounded memory but found no
useful optimization-quality runtime--accuracy tradeoff for deterministic nested
uniform shard sampling.  Sub-100% effort is an experimental diagnostic
capability, not a validated drop-in replacement for exact gradients.  Callers
must validate gradient accuracy independently.  ``quality.status`` and the
dispersion indicators are diagnostics, not certified error bounds.  Effort
100 delegates to the established exact backend.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
import resource
import sys
from time import perf_counter
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .gravity.demand import generate_gravity_demand
from .gravity.objective import (
    GravityObjectiveEvaluation,
    GravityObjectiveProblem,
    _evaluation_from_mean,
    _objective_from_mean,
    gravity_value_and_gradient_adjoint,
)
from .construction_control import estimate_completed_unit_eta
from .parallel_gravity_anchor import ParallelGravityAnchor, _validate_anchor
from .sharded_fixed_routing import FixedRoutingShardDescriptor, fixed_routing_shard_path
from .sharded_matrix_free_operator import (
    ShardedMatrixFreeFixedRoutingMeasurementOperator,
)

QualityStatus = Literal["poor", "usable", "good"]
EvaluationStatus = Literal["complete", "interrupted"]


def _current_rss_bytes() -> int | None:
    """Return current RSS where the platform exposes it, otherwise peak RSS."""
    try:
        if sys.platform.startswith("linux"):
            fields = open("/proc/self/statm", encoding="ascii").read().split()  # noqa: SIM115
            return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (OSError, ValueError, IndexError):
        return None


@dataclass(frozen=True, slots=True)
class StochasticShardSelection:
    selected_shard_ids: tuple[int, ...]
    sampling_weights: tuple[float, ...]
    requested_effort_percent: float
    realized_effort_percent: float
    total_shards: int
    seed: int
    fingerprint: str


def select_stochastic_routing_shards(
    descriptors: tuple[FixedRoutingShardDescriptor, ...],
    *,
    effort_percent: float,
    seed: int,
) -> StochasticShardSelection:
    """Select a deterministic prefix of a seeded shard permutation.

    Prefix selection makes effort levels nested for a fixed seed.  Equal shard
    inclusion probabilities imply the Horvitz--Thompson expansion weight N / k.
    """
    if not descriptors:
        raise ValueError("at least one persisted routing shard is required.")
    if not np.isfinite(effort_percent) or not 0.0 < effort_percent <= 100.0:
        raise ValueError("effort_percent must be in (0, 100].")
    ids = np.asarray(sorted(item.shard_index for item in descriptors), dtype=np.int64)
    if np.unique(ids).size != ids.size:
        raise ValueError("persisted shard identities must be unique.")
    permutation = np.random.default_rng(seed).permutation(ids)
    selected_count = min(ids.size, max(1, math.ceil(ids.size * effort_percent / 100.0)))
    selected = tuple(int(value) for value in permutation[:selected_count])
    weight = float(ids.size / selected_count)
    payload = {
        "all_shard_ids": ids.tolist(),
        "selected_shard_ids": selected,
        "seed": int(seed),
        "requested_effort_percent": float(effort_percent),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return StochasticShardSelection(
        selected_shard_ids=selected,
        sampling_weights=(weight,) * selected_count,
        requested_effort_percent=float(effort_percent),
        realized_effort_percent=100.0 * selected_count / ids.size,
        total_shards=int(ids.size),
        seed=int(seed),
        fingerprint=fingerprint,
    )


@dataclass(frozen=True, slots=True)
class StochasticGravityConfig:
    effort_percent: float = 10.0
    seed: int = 20260722
    concurrency: int = 1
    rss_ceiling_bytes: int | None = None
    rss_safety_margin_bytes: int = 2 * 1024**3
    conservative_next_shard_bytes: int | None = None
    archive_expansion_factor: float = 4.0
    garbage_collect_every_shards: int = 0
    garbage_collect_rss_trigger_fraction: float = 0.90

    def __post_init__(self) -> None:
        if self.concurrency != 1:
            raise ValueError("the streaming backend currently requires concurrency=1.")
        if self.rss_ceiling_bytes is not None and self.rss_ceiling_bytes <= 0:
            raise ValueError("rss_ceiling_bytes must be positive when provided.")
        if self.rss_safety_margin_bytes < 0:
            raise ValueError("rss_safety_margin_bytes must be nonnegative.")
        if (
            self.conservative_next_shard_bytes is not None
            and self.conservative_next_shard_bytes <= 0
        ):
            raise ValueError("conservative_next_shard_bytes must be positive.")
        if not np.isfinite(self.archive_expansion_factor) or self.archive_expansion_factor < 1:
            raise ValueError("archive_expansion_factor must be finite and at least one.")
        if self.garbage_collect_every_shards < 0:
            raise ValueError("garbage_collect_every_shards must be nonnegative.")
        if not 0.0 < self.garbage_collect_rss_trigger_fraction <= 1.0:
            raise ValueError(
                "garbage_collect_rss_trigger_fraction must be in (0, 1]."
            )


@dataclass(frozen=True, slots=True)
class StochasticShardProgress:
    phase: Literal["forward", "reverse"]
    boundary: Literal["before", "after"]
    current_shard_id: int
    completed_shards: int
    total_selected_shards: int
    rss_bytes: int | None
    peak_rss_bytes: int | None
    estimated_next_shard_bytes: int
    completed_units: int | None = None
    total_units: int | None = None
    recent_unit_seconds: float | None = None
    predicted_remaining_seconds: float | None = None
    eta_confidence: str = "unavailable"
    eta_reason: str | None = None
    estimated_completion_at_utc: str | None = None
    eta_lower_seconds: float | None = None
    eta_upper_seconds: float | None = None
    throughput_units_per_second: float | None = None


@dataclass(frozen=True, slots=True)
class StochasticQualityDiagnostics:
    measurement_standard_error_indicator: float
    gradient_standard_error_indicator: float
    effective_sampled_support_fraction: float
    measurement_coverage_fraction: float
    maximum_shard_influence: float
    status: QualityStatus


@dataclass(frozen=True, slots=True)
class StochasticGravityResult:
    status: EvaluationStatus
    evaluation: GravityObjectiveEvaluation | None
    gradient: np.ndarray | None
    predicted_measurements: np.ndarray | None
    selection: StochasticShardSelection
    wall_time_seconds: float
    forward_seconds: float
    reverse_seconds: float
    rss_before_bytes: int | None
    peak_rss_bytes: int | None
    rss_after_bytes: int | None
    progress: tuple[StochasticShardProgress, ...]
    quality: StochasticQualityDiagnostics | None
    exact: bool
    interrupted_phase: str | None = None
    interrupted_before_shard_id: int | None = None
    completed_forward_shards: int = 0
    completed_reverse_shards: int = 0
    resumable: bool = False


def _estimated_shard_bytes(
    operator: ShardedMatrixFreeFixedRoutingMeasurementOperator,
    descriptor: FixedRoutingShardDescriptor,
    config: StochasticGravityConfig,
) -> int:
    if config.conservative_next_shard_bytes is not None:
        return config.conservative_next_shard_bytes
    try:
        archive = fixed_routing_shard_path(operator.routing, descriptor)
        return max(1, math.ceil(archive.stat().st_size * config.archive_expansion_factor))
    except OSError:
        groups = descriptor.num_groups
        links = operator.routing.num_links
        return max(1, groups * links * (operator.dtype.itemsize + np.dtype(bool).itemsize))


def _quality_status(measurement_se: float, gradient_se: float, coverage: float) -> QualityStatus:
    worst = max(measurement_se, gradient_se)
    if coverage >= 0.75 and worst <= 0.05:
        return "good"
    if coverage >= 0.25 and worst <= 0.20:
        return "usable"
    return "poor"


def _relative_total_standard_error(
    *, count: int, total_count: int, sum_norm_squared: float, total: np.ndarray
) -> float:
    if count <= 1 or count >= total_count:
        return 0.0
    centered = max(0.0, sum_norm_squared - float(np.vdot(total, total).real) / count)
    sample_variance_trace = centered / (count - 1)
    variance_trace = total_count**2 * (1.0 - count / total_count) * sample_variance_trace / count
    return float(math.sqrt(max(0.0, variance_trace)) / max(np.linalg.norm(total), 1.0e-12))


def stochastic_gravity_value_and_gradient(
    raw_parameters: object,
    *,
    problem: GravityObjectiveProblem,
    config: StochasticGravityConfig | None = None,
    anchor: ParallelGravityAnchor | None = None,
    progress_callback: Callable[[StochasticShardProgress], None] | None = None,
) -> StochasticGravityResult:
    """Evaluate a two-pass sampled gravity objective with bounded shard residency."""
    config = StochasticGravityConfig() if config is None else config
    operator = problem.operator
    if not isinstance(operator, ShardedMatrixFreeFixedRoutingMeasurementOperator):
        raise TypeError("stochastic gravity requires the persisted sharded operator.")
    descriptors = tuple(operator.routing.shard_partition)
    selection = select_stochastic_routing_shards(
        descriptors, effort_percent=config.effort_percent, seed=config.seed
    )
    started = perf_counter()
    rss_before = _current_rss_bytes()
    peak_rss = rss_before
    progress: list[StochasticShardProgress] = []

    if config.effort_percent == 100.0:
        evaluation, gradient = gravity_value_and_gradient_adjoint(
            raw_parameters, problem=problem
        )
        jax.block_until_ready((evaluation, gradient))
        rss_after = _current_rss_bytes()
        return StochasticGravityResult(
            status="complete", evaluation=evaluation, gradient=np.asarray(gradient),
            predicted_measurements=np.asarray(evaluation.measurement_mean), selection=selection,
            wall_time_seconds=perf_counter() - started, forward_seconds=0.0,
            reverse_seconds=0.0, rss_before_bytes=rss_before,
            peak_rss_bytes=rss_after if peak_rss is None else max(peak_rss, rss_after or 0),
            rss_after_bytes=rss_after, progress=(), quality=None, exact=True,
            completed_forward_shards=len(descriptors), completed_reverse_shards=len(descriptors),
        )

    raw = jnp.asarray(raw_parameters)
    if anchor is not None:
        _validate_anchor(anchor, problem=problem)
        if np.asarray(raw).shape != anchor.raw_parameters.shape:
            raise ValueError("raw parameter shape differs from the anchor.")
        if np.array_equal(np.asarray(raw), anchor.raw_parameters):
            evaluation = _evaluation_from_mean(
                raw, mean=jnp.asarray(anchor.measurement_mean),
                demand=jnp.asarray(anchor.demand), problem=problem,
            )
            return StochasticGravityResult(
                status="complete", evaluation=evaluation,
                gradient=np.asarray(anchor.gradient),
                predicted_measurements=np.asarray(anchor.measurement_mean),
                selection=selection, wall_time_seconds=perf_counter() - started,
                forward_seconds=0.0, reverse_seconds=0.0,
                rss_before_bytes=rss_before, peak_rss_bytes=rss_before,
                rss_after_bytes=_current_rss_bytes(), progress=(), quality=None,
                exact=False,
            )

    def demand_function(value):
        return generate_gravity_demand(
            value, features=problem.features, parameter_layout=problem.parameter_layout
        ).demand

    demand, demand_pullback = jax.vjp(demand_function, raw)
    demand_np = np.asarray(demand)
    routed = (
        np.asarray(anchor.routed_measurements, dtype=operator.dtype).copy()
        if anchor is not None
        else np.zeros(operator.num_measurements, dtype=operator.dtype)
    )
    routing_input = demand_np - anchor.demand if anchor is not None else demand_np
    by_id = {item.shard_index: item for item in descriptors}
    selected = [by_id[item] for item in selection.selected_shard_ids]
    weight = selection.sampling_weights[0]
    # Persisted shards normally share one full shape; padding the final short
    # shard to that shape avoids compiling a separate executable for the tail.
    padded_groups = max(item.num_groups for item in descriptors)
    contribution_norms: list[float] = []
    covered = np.zeros(operator.num_measurements, dtype=bool)
    phase_started_at: dict[str, float] = {}
    phase_durations: dict[str, deque[float]] = {
        "forward": deque(maxlen=32),
        "reverse": deque(maxlen=32),
    }

    def report(
        phase: str,
        boundary: Literal["before", "after"],
        descriptor: FixedRoutingShardDescriptor,
        completed: int,
    ) -> tuple[int | None, int]:
        nonlocal peak_rss
        rss = _current_rss_bytes()
        if rss is not None:
            peak_rss = rss if peak_rss is None else max(peak_rss, rss)
        estimated = _estimated_shard_bytes(operator, descriptor, config)
        now = perf_counter()
        if boundary == "before":
            phase_started_at[phase] = now
            duration = None
        else:
            started_at = phase_started_at.get(phase, now)
            duration = max(0.0, now - started_at)
            if duration > 0.0:
                phase_durations[phase].append(duration)
        eta = estimate_completed_unit_eta(
            phase_durations[phase],
            completed_units=completed,
            total_units=len(selected),
            elapsed_seconds=max(0.0, now - started),
        )
        item = StochasticShardProgress(
            phase=phase, boundary=boundary,
            current_shard_id=descriptor.shard_index,
            completed_shards=completed, total_selected_shards=len(selected),
            rss_bytes=rss, peak_rss_bytes=peak_rss,
            estimated_next_shard_bytes=estimated,
            completed_units=completed,
            total_units=len(selected),
            recent_unit_seconds=duration,
            predicted_remaining_seconds=eta.predicted_remaining_seconds,
            eta_confidence=eta.eta_confidence,
            eta_reason=eta.eta_reason,
            estimated_completion_at_utc=eta.estimated_completion_at_utc,
            eta_lower_seconds=eta.eta_lower_seconds,
            eta_upper_seconds=eta.eta_upper_seconds,
            throughput_units_per_second=eta.throughput_units_per_second,
        )
        progress.append(item)
        if progress_callback is not None:
            try:
                progress_callback(item)
            except Exception:
                # Progress is ancillary; retain the numerical result if the
                # user-provided sink fails.
                pass
        return rss, estimated

    def admit(
        phase: Literal["forward", "reverse"],
        descriptor: FixedRoutingShardDescriptor,
        completed: int,
    ) -> bool:
        rss, estimated = report(phase, "before", descriptor, completed)
        return not (
            config.rss_ceiling_bytes is not None
            and rss is not None
            and rss + estimated + config.rss_safety_margin_bytes
            > config.rss_ceiling_bytes
        )

    def interrupted(phase: str, descriptor: FixedRoutingShardDescriptor, forward_done: int, reverse_done: int):
        operator.evict_resident_shards()
        return StochasticGravityResult(
            status="interrupted", evaluation=None, gradient=None,
            predicted_measurements=None, selection=selection,
            wall_time_seconds=perf_counter() - started, forward_seconds=0.0,
            reverse_seconds=0.0, rss_before_bytes=rss_before,
            peak_rss_bytes=peak_rss, rss_after_bytes=_current_rss_bytes(),
            progress=tuple(progress), quality=None, exact=False,
            interrupted_phase=phase,
            interrupted_before_shard_id=descriptor.shard_index,
            completed_forward_shards=forward_done,
            completed_reverse_shards=reverse_done, resumable=True,
        )

    def collect_if_needed(completed: int) -> None:
        periodic = (
            config.garbage_collect_every_shards > 0
            and completed % config.garbage_collect_every_shards == 0
        )
        rss = _current_rss_bytes()
        pressure = (
            config.rss_ceiling_bytes is not None
            and rss is not None
            and rss
            >= config.rss_ceiling_bytes
            * config.garbage_collect_rss_trigger_fraction
        )
        if periodic or pressure:
            gc.collect()

    forward_started = perf_counter()
    for completed, descriptor in enumerate(selected):
        if not admit("forward", descriptor, completed):
            return interrupted("forward", descriptor, completed, 0)
        contribution = operator.partial_matvec(
            routing_input,
            destination_group_indices=descriptor.destination_group_indices,
            padded_groups=padded_groups,
            group_weights=(weight,) * descriptor.num_groups,
        )
        routed += contribution
        covered |= contribution != 0
        contribution_norms.append(float(np.vdot(contribution, contribution).real) / weight**2)
        del contribution
        operator.evict_resident_shards()
        collect_if_needed(completed + 1)
        report("forward", "after", descriptor, completed + 1)
    forward_seconds = perf_counter() - forward_started

    offset = jnp.asarray(operator.fixed_measurement_offset, dtype=demand.dtype)
    rho = jnp.asarray(problem.rho, dtype=demand.dtype)
    mean_unfloored = rho * (jnp.asarray(routed) + offset)
    mean = jnp.maximum(mean_unfloored, problem.mean_floor)
    mean_gradient = jax.grad(lambda value: _objective_from_mean(value, raw, problem))(mean)
    active_mean = (mean_unfloored > problem.mean_floor).astype(mean.dtype)
    measurement_cotangent = np.asarray(rho * active_mean * mean_gradient)
    direct_gradient = np.asarray(
        jax.grad(lambda parameters: _objective_from_mean(mean, parameters, problem))(raw)
    )
    accumulated_demand_cotangent = np.zeros(operator.num_free_od, dtype=operator.dtype)
    cotangent_norms: list[float] = []

    reverse_started = perf_counter()
    for completed, descriptor in enumerate(selected):
        if not admit("reverse", descriptor, completed):
            return interrupted("reverse", descriptor, len(selected), completed)
        demand_cotangent = operator.partial_rmatvec(
            measurement_cotangent,
            destination_group_indices=descriptor.destination_group_indices,
            padded_groups=padded_groups,
            group_weights=(weight,) * descriptor.num_groups,
        )
        accumulated_demand_cotangent += demand_cotangent
        cotangent_norms.append(
            float(np.vdot(demand_cotangent, demand_cotangent).real) / weight**2
        )
        del demand_cotangent
        operator.evict_resident_shards()
        collect_if_needed(completed + 1)
        report("reverse", "after", descriptor, completed + 1)
    gradient = direct_gradient + np.asarray(
        demand_pullback(jnp.asarray(accumulated_demand_cotangent))[0]
    )
    reverse_seconds = perf_counter() - reverse_started

    evaluation = _evaluation_from_mean(raw, mean=mean, demand=demand, problem=problem)
    jax.block_until_ready(evaluation)
    count = len(selected)
    measurement_se = _relative_total_standard_error(
        count=count, total_count=len(descriptors),
        sum_norm_squared=sum(contribution_norms), total=(routed if anchor is None else routed - anchor.routed_measurements) / weight,
    )
    gradient_se = _relative_total_standard_error(
        count=count, total_count=len(descriptors),
        sum_norm_squared=sum(cotangent_norms),
        total=accumulated_demand_cotangent / weight,
    )
    influences = [math.sqrt(value) for value in contribution_norms]
    max_influence = max(influences, default=0.0) / max(sum(influences), 1.0e-12)
    coverage = float(np.mean(covered))
    quality = StochasticQualityDiagnostics(
        measurement_standard_error_indicator=measurement_se,
        gradient_standard_error_indicator=gradient_se,
        effective_sampled_support_fraction=selection.realized_effort_percent / 100.0,
        measurement_coverage_fraction=coverage,
        maximum_shard_influence=max_influence,
        status=_quality_status(measurement_se, gradient_se, coverage),
    )
    rss_after = _current_rss_bytes()
    if rss_after is not None:
        peak_rss = rss_after if peak_rss is None else max(peak_rss, rss_after)
    return StochasticGravityResult(
        status="complete", evaluation=evaluation, gradient=gradient,
        predicted_measurements=np.asarray(mean), selection=selection,
        wall_time_seconds=perf_counter() - started,
        forward_seconds=forward_seconds, reverse_seconds=reverse_seconds,
        rss_before_bytes=rss_before, peak_rss_bytes=peak_rss,
        rss_after_bytes=rss_after, progress=tuple(progress), quality=quality,
        exact=False, completed_forward_shards=count, completed_reverse_shards=count,
    )
