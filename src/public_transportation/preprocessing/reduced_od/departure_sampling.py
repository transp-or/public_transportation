"""Supply-independent desired-departure sampling and weighted responses."""

from __future__ import annotations

import hashlib
import math
import resource
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from .artifacts import canonical_json
from .journey_choices import (
    JourneyAlternative,
    JourneyChoiceDiagnostics,
    JourneyChoiceResult,
    JourneyChoiceSet,
    JourneyTimePeriod,
)
from .response_atoms import ResponseCellKey


SamplingProgress = Callable[[Mapping[str, object]], None]
InfeasiblePolicy = Literal[
    "condition_on_feasible", "retain_unserved_mass", "preserve_mass", "reject_cell"
]
DepartureResponseClassification = Literal[
    "normal", "warning", "excluded_low_feasibility", "frozen_no_feasible_sample"
]
DepartureCellStatus = Literal["free", "fixed_zero", "fixed_positive"]


@dataclass(frozen=True, slots=True)
class ProbabilityMassCanonicalization:
    """Auditable tolerance-sized projection onto the probability interval."""

    raw_value: float
    canonical_value: float
    applied: bool
    delta: float


def canonicalize_probability_mass(
    value: float,
    *,
    tolerance: float,
    name: str = "probability mass",
) -> ProbabilityMassCanonicalization:
    """Canonicalize only roundoff-sized excursions outside ``[0, 1]``."""
    raw = float(value)
    if not math.isfinite(raw):
        raise ValueError(f"{name} must be finite.")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("probability mass tolerance must be finite and positive.")
    if raw < -tolerance or raw > 1.0 + tolerance:
        raise ValueError(f"{name} lies materially outside [0, 1]: {raw}.")
    canonical = 0.0 if raw < 0.0 else 1.0 if raw > 1.0 else raw
    return ProbabilityMassCanonicalization(
        raw, canonical, canonical != raw, canonical - raw
    )


class DepartureSampleInfeasibilityReason(str, Enum):
    """Controlled reasons for timetable or policy infeasibility."""

    NO_SERVICE_AFTER_DESIRED_TIME = "no_service_after_desired_time"
    NO_COMPLETE_JOURNEY_BEFORE_HORIZON = "no_complete_journey_before_horizon"
    INITIAL_WAIT_EXCEEDED = "initial_wait_exceeded"
    JOURNEY_DURATION_EXCEEDED = "journey_duration_exceeded"
    TRANSFER_LIMIT_EXCEEDED = "transfer_limit_exceeded"
    MISSING_TRANSFER_CONNECTION = "missing_transfer_connection"
    DIRECTED_PLATFORM_UNREACHABLE = "directed_platform_unreachable"
    OUTSIDE_TIMETABLE_HORIZON = "outside_timetable_horizon"
    NO_FEASIBLE_ALTERNATIVE = "no_feasible_alternative"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DepartureTimeSamplingConfig:
    strategy: Literal[
        "uniform_midpoint",
        "fixed_count",
        "fixed_time_step",
        "adaptive_service_aware",
    ] = "uniform_midpoint"
    samples_per_period: int | Mapping[str, int] = 12
    time_step_seconds: int | Mapping[str, int] = 300
    initial_interval_seconds: int = 900
    minimum_interval_seconds: int = 60
    response_tolerance: float = 1.0e-3
    absolute_response_tolerance: float = 0.0
    relative_response_tolerance: float | None = None
    integration_scale_floor: float = 1.0e-12
    service_boundary_safeguard: bool = True
    maximum_samples_per_cell: int = 128
    maximum_refinement_depth: int | None = None
    comparison_mode: Literal[
        "assignment_response",
        "service_signature",
        "exact_service_identity",
        "measurement_support",
        "aggregate_response",
        "two_stage",
        "route_pattern_signature",
        "integral_response",
    ] = "assignment_response"
    minimum_feasible_fraction: float = 0.5
    warning_feasible_fraction: float = 0.9
    infeasible_policy: InfeasiblePolicy = "condition_on_feasible"
    convergence_levels: tuple[int, ...] = (3, 6, 12)
    weight_tolerance: float = 1.0e-12
    progress_interval_groups: int = 25
    progress_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.strategy not in {
            "uniform_midpoint",
            "fixed_count",
            "fixed_time_step",
            "adaptive_service_aware",
        }:
            raise ValueError("unsupported desired-departure sampling strategy.")
        counts = (
            {"*": self.samples_per_period}
            if isinstance(self.samples_per_period, int)
            and not isinstance(self.samples_per_period, bool)
            else dict(self.samples_per_period)
            if isinstance(self.samples_per_period, Mapping)
            else None
        )
        if not counts or any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for name, value in counts.items()
        ):
            raise ValueError("samples_per_period must contain positive counts.")
        steps = (
            {"*": self.time_step_seconds}
            if isinstance(self.time_step_seconds, int)
            and not isinstance(self.time_step_seconds, bool)
            else dict(self.time_step_seconds)
            if isinstance(self.time_step_seconds, Mapping)
            else None
        )
        if not steps or any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for name, value in steps.items()
        ):
            raise ValueError("time_step_seconds must contain positive durations.")
        if self.initial_interval_seconds <= 0 or self.minimum_interval_seconds <= 0:
            raise ValueError("adaptive interval durations must be positive.")
        if self.minimum_interval_seconds > self.initial_interval_seconds:
            raise ValueError(
                "minimum_interval_seconds cannot exceed initial_interval_seconds."
            )
        if not math.isfinite(self.response_tolerance) or self.response_tolerance < 0:
            raise ValueError("response_tolerance must be finite and nonnegative.")
        if (
            not math.isfinite(self.absolute_response_tolerance)
            or self.absolute_response_tolerance < 0
        ):
            raise ValueError(
                "absolute_response_tolerance must be finite and nonnegative."
            )
        if self.relative_response_tolerance is not None and (
            not math.isfinite(self.relative_response_tolerance)
            or self.relative_response_tolerance < 0
        ):
            raise ValueError(
                "relative_response_tolerance must be finite and nonnegative."
            )
        if not math.isfinite(self.integration_scale_floor) or self.integration_scale_floor <= 0:
            raise ValueError("integration_scale_floor must be finite and positive.")
        if not isinstance(self.service_boundary_safeguard, bool):
            raise ValueError("service_boundary_safeguard must be boolean.")
        if self.maximum_samples_per_cell <= 0:
            raise ValueError("maximum_samples_per_cell must be positive.")
        if self.maximum_refinement_depth is not None and self.maximum_refinement_depth < 0:
            raise ValueError("maximum_refinement_depth must be nonnegative.")
        if self.comparison_mode not in {
            "assignment_response",
            "service_signature",
            "exact_service_identity",
            "measurement_support",
            "aggregate_response",
            "two_stage",
            "route_pattern_signature",
            "integral_response",
        }:
            raise ValueError("unsupported comparison_mode.")
        for value, name in (
            (self.minimum_feasible_fraction, "minimum_feasible_fraction"),
            (self.warning_feasible_fraction, "warning_feasible_fraction"),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if self.minimum_feasible_fraction > self.warning_feasible_fraction:
            raise ValueError("minimum feasibility cannot exceed warning feasibility.")
        if self.infeasible_policy not in {
            "condition_on_feasible",
            "retain_unserved_mass",
            "preserve_mass",
            "reject_cell",
        }:
            raise ValueError("unsupported infeasible_policy.")
        if (
            not self.convergence_levels
            or any(value <= 0 for value in self.convergence_levels)
            or tuple(sorted(set(self.convergence_levels))) != self.convergence_levels
        ):
            raise ValueError("convergence_levels must be positive and increasing.")
        if not math.isfinite(self.weight_tolerance) or self.weight_tolerance <= 0.0:
            raise ValueError("weight_tolerance must be finite and positive.")
        if self.progress_interval_groups <= 0 or self.progress_interval_seconds <= 0.0:
            raise ValueError("progress intervals must be positive.")

    def count_for_period(self, period_id: str) -> int:
        if isinstance(self.samples_per_period, int):
            return self.samples_per_period
        try:
            return int(self.samples_per_period[period_id])
        except KeyError as error:
            raise ValueError(
                f"no sample count configured for period {period_id!r}."
            ) from error

    def step_for_period(self, period_id: str) -> int:
        if isinstance(self.time_step_seconds, int):
            return self.time_step_seconds
        try:
            return int(self.time_step_seconds[period_id])
        except KeyError as error:
            raise ValueError(
                f"no time step configured for period {period_id!r}."
            ) from error

    @property
    def effective_relative_response_tolerance(self) -> float:
        """Integral relative tolerance with the legacy field as its alias."""
        return (
            self.response_tolerance
            if self.relative_response_tolerance is None
            else self.relative_response_tolerance
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class DesiredDepartureSample:
    sample_id: str
    origin_physical_stop_id: str
    time_period_id: str
    desired_departure_seconds: float
    original_weight: float

    def __post_init__(self) -> None:
        if (
            not self.sample_id
            or not self.origin_physical_stop_id
            or not self.time_period_id
        ):
            raise ValueError("desired-departure sample identities must be non-empty.")
        if not math.isfinite(self.desired_departure_seconds):
            raise ValueError("desired departure time must be finite.")
        if not math.isfinite(self.original_weight) or self.original_weight < 0.0:
            raise ValueError("original sample weight must be finite and nonnegative.")


@dataclass(frozen=True, slots=True)
class DesiredDepartureRoutingResult:
    sample: DesiredDepartureSample
    feasible: bool
    infeasibility_reason: DepartureSampleInfeasibilityReason | None
    alternatives: tuple[JourneyAlternative, ...]
    conditional_route_shares: tuple[float, ...]
    diagnostic_candidates: tuple[DepartureSampleInfeasibilityReason, ...] = ()

    def __post_init__(self) -> None:
        if len(self.alternatives) != len(self.conditional_route_shares):
            raise ValueError("route shares must align with journey alternatives.")
        if self.feasible:
            if not self.alternatives or self.infeasibility_reason is not None:
                raise ValueError(
                    "a feasible sample requires alternatives and no reason."
                )
            if any(
                not math.isfinite(value) or value < 0.0
                for value in self.conditional_route_shares
            ) or not math.isclose(
                sum(self.conditional_route_shares), 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("conditional route shares must sum to one.")
        elif self.alternatives or self.conditional_route_shares:
            raise ValueError("an infeasible sample cannot retain alternatives.")
        elif self.infeasibility_reason is None:
            raise ValueError("an infeasible sample requires a controlled reason.")


@dataclass(frozen=True, slots=True)
class SparseWeightedResponse:
    measurement_indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.measurement_indices) != len(self.values):
            raise ValueError("sparse response indices and values must align.")
        if tuple(sorted(set(self.measurement_indices))) != self.measurement_indices:
            raise ValueError("sparse response indices must be unique and sorted.")
        if any(index < 0 for index in self.measurement_indices) or any(
            not math.isfinite(value) for value in self.values
        ):
            raise ValueError("sparse response values are invalid.")


@dataclass(frozen=True, slots=True)
class AveragedDepartureResponse:
    cell_key: ResponseCellKey
    total_sample_count: int
    feasible_sample_count: int
    original_feasible_weight: float
    original_infeasible_weight: float
    conditional_sample_weights: tuple[float, ...]
    feasible_sample_ids: tuple[str, ...]
    averaged_response: SparseWeightedResponse
    classification: DepartureResponseClassification
    infeasibility_weight_by_reason: tuple[tuple[str, float], ...]
    distinct_alternative_ids: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SampledJourneyCellDiagnostics:
    cell_key: ResponseCellKey
    total_sample_count: int
    feasible_sample_count: int
    original_feasible_weight: float
    original_infeasible_weight: float
    classification: DepartureResponseClassification
    distinct_paths: int
    conditional_weight_concentration: float
    infeasibility_weight_by_reason: tuple[tuple[str, float], ...]
    cell_status: DepartureCellStatus
    timetable_feasible_fixed_zero: bool
    fixed_positive_assignment_failed: bool
    unexpected_status_change: bool = False
    raw_feasible_time_fraction: float | None = None
    canonical_feasible_time_fraction: float | None = None
    mass_canonicalization_applied: bool = False
    mass_canonicalization_delta: float = 0.0


@dataclass(frozen=True, slots=True)
class SampledJourneyChoiceResult:
    journey_choices: JourneyChoiceResult
    samples: tuple[DesiredDepartureSample, ...]
    cells: tuple[SampledJourneyCellDiagnostics, ...]
    sampling_fingerprint: str


def _path_identity(alternative: JourneyAlternative) -> str:
    payload = [
        [
            leg.trip_id,
            leg.board_physical_stop_id,
            leg.alight_physical_stop_id,
            leg.board_seconds,
            leg.alight_seconds,
        ]
        for leg in alternative.transit_legs
    ]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _sample_order(sample: DesiredDepartureSample) -> tuple[str, str, float, str]:
    return (
        sample.origin_physical_stop_id,
        sample.time_period_id,
        sample.desired_departure_seconds,
        sample.sample_id,
    )


def merge_sampled_journey_choices(
    *,
    samples: Sequence[DesiredDepartureSample],
    sample_choices: Sequence[JourneyChoiceResult],
    config: DepartureTimeSamplingConfig = DepartureTimeSamplingConfig(),
    candidate_cells: Sequence[ResponseCellKey] = (),
    cell_status: Mapping[ResponseCellKey, DepartureCellStatus] | None = None,
) -> SampledJourneyChoiceResult:
    """Merge sample-specific choices with ``P(sample) P(route | sample)``."""
    if len(samples) != len(sample_choices) or not samples:
        raise ValueError("samples and sample_choices must be non-empty and aligned.")
    validate_sample_weights(samples, tolerance=config.weight_tolerance)
    candidate = tuple(sorted(set(candidate_cells)))
    if cell_status is None:
        statuses: Mapping[ResponseCellKey, DepartureCellStatus] = {
            cell: "free" for cell in candidate
        }
    else:
        statuses = dict(cell_status)
        if set(statuses) != set(candidate):
            raise ValueError("cell_status must cover exactly the canonical candidates.")
    grouped_samples: dict[
        tuple[str, str], list[tuple[DesiredDepartureSample, JourneyChoiceResult]]
    ] = {}
    for sample, choices in sorted(
        zip(samples, sample_choices, strict=True),
        key=lambda item: _sample_order(item[0]),
    ):
        grouped_samples.setdefault(
            (sample.origin_physical_stop_id, sample.time_period_id), []
        ).append((sample, choices))

    merged_sets: list[JourneyChoiceSet] = []
    cell_diagnostics: list[SampledJourneyCellDiagnostics] = []
    for (origin, period), group in sorted(grouped_samples.items()):
        if candidate:
            destinations = sorted(
                {
                    cell.destination_physical_stop_id
                    for cell in candidate
                    if cell.origin_physical_stop_id == origin
                    and cell.origin_time_period_id == period
                }
            )
        else:
            # Backward-compatible unconstrained helper mode. Integrated
            # preparation always supplies the canonical candidates.
            destinations = sorted(
                {
                    choice.destination_physical_stop_id
                    for _, result in group
                    for choice in result.choice_sets
                    if choice.origin_physical_stop_id == origin
                }
            )
        for destination in destinations:
            raw_feasible_weight = 0.0
            path_weights: dict[str, float] = {}
            representatives: dict[str, JourneyAlternative] = {}
            feasible_count = 0
            for sample, result in group:
                matching = [
                    choice
                    for choice in result.choice_sets
                    if choice.origin_physical_stop_id == origin
                    and choice.destination_physical_stop_id == destination
                    and choice.demand_time_period_id == period
                ]
                if not matching:
                    continue
                if len(matching) != 1:
                    raise ValueError(
                        "sample choices contain duplicate canonical "
                        "origin-destination-demand-period cells."
                    )
                feasible_count += 1
                raw_feasible_weight += sample.original_weight
                choice = matching[0]
                for alternative, route_share in zip(
                    choice.alternatives, choice.initial_shares, strict=True
                ):
                    path_id = _path_identity(alternative)
                    representatives.setdefault(
                        path_id,
                        replace(
                            alternative,
                            desired_departure_time_period_id=period,
                        ),
                    )
                    path_weights[path_id] = (
                        path_weights.get(path_id, 0.0)
                        + sample.original_weight * route_share
                    )
            mass = canonicalize_probability_mass(
                raw_feasible_weight,
                tolerance=config.weight_tolerance,
                name=f"feasible time mass for {origin}-{destination}-{period}",
            )
            feasible_weight = mass.canonical_value
            infeasible_weight = 1.0 - feasible_weight
            if feasible_weight <= config.weight_tolerance:
                classification: DepartureResponseClassification = (
                    "frozen_no_feasible_sample"
                )
            elif feasible_weight < config.minimum_feasible_fraction:
                classification = "excluded_low_feasibility"
            elif feasible_weight < config.warning_feasible_fraction:
                classification = "warning"
            else:
                classification = "normal"
            normalized = (
                tuple(
                    (path_id, weight / sum(path_weights.values()))
                    for path_id, weight in sorted(path_weights.items())
                )
                if feasible_weight > config.weight_tolerance
                else ()
            )
            cell_key = ResponseCellKey(origin, destination, period)
            status = statuses.get(cell_key, "free")
            fixed_positive_failed = status == "fixed_positive" and (
                classification
                in {"excluded_low_feasibility", "frozen_no_feasible_sample"}
            )
            cell_diagnostics.append(
                SampledJourneyCellDiagnostics(
                    cell_key=cell_key,
                    total_sample_count=len(group),
                    feasible_sample_count=feasible_count,
                    original_feasible_weight=feasible_weight,
                    original_infeasible_weight=infeasible_weight,
                    classification=classification,
                    distinct_paths=len(normalized),
                    conditional_weight_concentration=max(
                        (weight for _, weight in normalized), default=0.0
                    ),
                    infeasibility_weight_by_reason=(
                        ()
                        if infeasible_weight <= config.weight_tolerance
                        else (
                            (
                                DepartureSampleInfeasibilityReason.NO_FEASIBLE_ALTERNATIVE.value,
                                infeasible_weight,
                            ),
                        )
                    ),
                    cell_status=status,
                    timetable_feasible_fixed_zero=(
                        status == "fixed_zero" and feasible_weight > 0.0
                    ),
                    fixed_positive_assignment_failed=fixed_positive_failed,
                    raw_feasible_time_fraction=raw_feasible_weight,
                    canonical_feasible_time_fraction=feasible_weight,
                    mass_canonicalization_applied=mass.applied,
                    mass_canonicalization_delta=mass.delta,
                )
            )
            if fixed_positive_failed:
                raise ValueError(
                    "fixed-positive cell cannot be assigned under the active "
                    f"sampling feasibility policy: {cell_key.tuple}."
                )
            retain_choice = status in {"free", "fixed_positive"} and classification in {
                "normal",
                "warning",
            }
            if retain_choice:
                merged_sets.append(
                    JourneyChoiceSet(
                        origin_physical_stop_id=origin,
                        destination_physical_stop_id=destination,
                        origin_time_period_id=period,
                        alternatives=tuple(
                            representatives[path_id] for path_id, _ in normalized
                        ),
                        initial_shares=tuple(weight for _, weight in normalized),
                        served_time_fraction=(
                            feasible_weight
                            if config.infeasible_policy
                            in {"retain_unserved_mass", "preserve_mass"}
                            else 1.0
                        ),
                    )
                )
    merged_sets.sort(
        key=lambda item: (
            item.origin_physical_stop_id,
            item.destination_physical_stop_id,
            item.origin_time_period_id,
        )
    )
    retained = sum(len(item.alternatives) for item in merged_sets)
    choices = JourneyChoiceResult(
        choice_sets=tuple(merged_sets),
        diagnostics=JourneyChoiceDiagnostics(
            feasible_destinations=len(merged_sets),
            candidate_alternatives=retained,
            retained_alternatives=retained,
            pruned_alternatives=0,
            choice_cells=len(merged_sets),
            maximum_candidates_in_cell=max(
                (len(item.alternatives) for item in merged_sets), default=0
            ),
            route_initialized_alternatives=0,
            estimated_payload_bytes=0,
        ),
    )
    fingerprint = hashlib.sha256(
        canonical_json(
            {
                "choices": choices.fingerprint,
                "config": config.fingerprint,
                "samples": [
                    asdict(sample) for sample in sorted(samples, key=_sample_order)
                ],
                "cells": [asdict(item) for item in cell_diagnostics],
            }
        ).encode("utf-8")
    ).hexdigest()
    return SampledJourneyChoiceResult(
        journey_choices=choices,
        samples=tuple(sorted(samples, key=_sample_order)),
        cells=tuple(
            sorted(
                cell_diagnostics,
                key=lambda item: (
                    item.cell_key.origin_physical_stop_id,
                    item.cell_key.destination_physical_stop_id,
                    item.cell_key.origin_time_period_id,
                ),
            )
        ),
        sampling_fingerprint=fingerprint,
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def generate_uniform_midpoint_samples(
    *,
    origin_physical_stop_ids: Sequence[str] = (),
    origin_period_groups: Sequence[tuple[str, str]] = (),
    time_periods: Sequence[JourneyTimePeriod],
    config: DepartureTimeSamplingConfig = DepartureTimeSamplingConfig(),
    progress: SamplingProgress | None = None,
) -> tuple[DesiredDepartureSample, ...]:
    """Generate deterministic desired times independently of the timetable."""
    if origin_physical_stop_ids and origin_period_groups:
        raise ValueError(
            "provide sparse origin_period_groups or Cartesian origins, not both."
        )
    origins = tuple(sorted(set(origin_physical_stop_ids)))
    periods = tuple(time_periods)
    if periods != tuple(
        sorted(periods, key=lambda item: (item.start_seconds, item.period_id))
    ):
        raise ValueError("time periods must be sorted.")
    if len({period.period_id for period in periods}) != len(periods):
        raise ValueError("time period identifiers must be unique.")
    for left, right in zip(periods, periods[1:]):
        if left.end_seconds > right.start_seconds:
            raise ValueError("time periods must not overlap.")
    period_by_id = {period.period_id: period for period in periods}
    if origin_period_groups:
        groups = tuple(sorted(origin_period_groups))
        if len(groups) != len(set(groups)):
            raise ValueError("origin-period groups must be unique.")
        for origin, period_id in groups:
            if not origin:
                raise ValueError("origin identifiers must be non-empty.")
            if period_id not in period_by_id:
                raise ValueError(f"unknown time period {period_id!r}.")
    else:
        groups = tuple(
            (origin, period.period_id) for origin in origins for period in periods
        )
    if not groups:
        raise ValueError("at least one origin-period group is required.")
    total_groups = len(groups)
    started = time.perf_counter()
    last_event = started
    if progress is not None:
        progress(
            {
                "phase": "departure_sampling",
                "status": "started",
                "completed_origin_period_groups": 0,
                "total_origin_period_groups": total_groups,
                "elapsed_seconds": 0.0,
                "peak_rss_bytes": _peak_rss_bytes(),
            }
        )
    samples: list[DesiredDepartureSample] = []
    completed = 0
    for origin, period_id in groups:
        period = period_by_id[period_id]
        count = config.count_for_period(period.period_id)
        duration = period.end_seconds - period.start_seconds
        weight = 1.0 / count
        for index in range(count):
            seconds = period.start_seconds + (index + 0.5) * duration / count
            identity = canonical_json([origin, period.period_id, index, count, seconds])
            samples.append(
                DesiredDepartureSample(
                    sample_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    origin_physical_stop_id=origin,
                    time_period_id=period.period_id,
                    desired_departure_seconds=seconds,
                    original_weight=weight,
                )
            )
        completed += 1
        now = time.perf_counter()
        if progress is not None and (
            completed == total_groups
            or completed % config.progress_interval_groups == 0
            or now - last_event >= config.progress_interval_seconds
        ):
            elapsed = now - started
            progress(
                {
                    "phase": "departure_sampling",
                    "status": "completed"
                    if completed == total_groups
                    else "in_progress",
                    "sampling_level": count,
                    "current_origin_physical_stop_id": origin,
                    "current_time_period_id": period_id,
                    "completed_origin_period_groups": completed,
                    "total_origin_period_groups": total_groups,
                    "completed_queries": len(samples),
                    "total_queries": sum(
                        config.count_for_period(group_period)
                        for _, group_period in groups
                    ),
                    "elapsed_seconds": elapsed,
                    "predicted_remaining_seconds": (
                        elapsed * (total_groups - completed) / completed
                    ),
                    "peak_rss_bytes": _peak_rss_bytes(),
                }
            )
            last_event = now
    validate_sample_weights(samples, tolerance=config.weight_tolerance)
    return tuple(samples)


def validate_sample_weights(
    samples: Sequence[DesiredDepartureSample], *, tolerance: float = 1.0e-12
) -> None:
    totals: dict[tuple[str, str], float] = {}
    for sample in samples:
        key = (sample.origin_physical_stop_id, sample.time_period_id)
        totals[key] = totals.get(key, 0.0) + sample.original_weight
    invalid = {
        key: value
        for key, value in totals.items()
        if not math.isclose(value, 1.0, abs_tol=tolerance, rel_tol=0.0)
    }
    if invalid:
        raise ValueError(f"desired-departure weights do not sum to one: {invalid}.")


def average_desired_departure_response(
    *,
    cell_key: ResponseCellKey,
    routing_results: Sequence[DesiredDepartureRoutingResult],
    alternative_responses: Mapping[str, Mapping[int, float]],
    config: DepartureTimeSamplingConfig = DepartureTimeSamplingConfig(),
) -> AveragedDepartureResponse:
    """Combine ``P(sample) P(route | sample)`` into one sparse cell response."""
    ordered = tuple(
        sorted(routing_results, key=lambda item: _sample_order(item.sample))
    )
    if not ordered:
        raise ValueError("routing_results must contain at least one sample.")
    validate_sample_weights(
        [item.sample for item in ordered], tolerance=config.weight_tolerance
    )
    for item in ordered:
        if (
            item.sample.origin_physical_stop_id != cell_key.origin_physical_stop_id
            or item.sample.time_period_id != cell_key.origin_time_period_id
        ):
            raise ValueError("routing sample does not match the response cell.")

    feasible: list[
        tuple[
            DesiredDepartureRoutingResult, tuple[tuple[JourneyAlternative, float], ...]
        ]
    ] = []
    reason_weights: dict[str, float] = {}
    for item in ordered:
        matching = tuple(
            (alternative, share)
            for alternative, share in zip(
                item.alternatives, item.conditional_route_shares, strict=True
            )
            if alternative.destination_physical_stop_id
            == cell_key.destination_physical_stop_id
        )
        if item.feasible and matching:
            matching_total = sum(share for _, share in matching)
            feasible.append(
                (
                    item,
                    tuple(
                        (alternative, share / matching_total)
                        for alternative, share in matching
                    ),
                )
            )
        else:
            reason = (
                item.infeasibility_reason
                if not item.feasible
                else DepartureSampleInfeasibilityReason.NO_FEASIBLE_ALTERNATIVE
            )
            assert reason is not None
            reason_weights[reason.value] = (
                reason_weights.get(reason.value, 0.0) + item.sample.original_weight
            )

    feasible_weight = sum(item.sample.original_weight for item, _ in feasible)
    infeasible_weight = max(0.0, 1.0 - feasible_weight)
    if feasible_weight <= config.weight_tolerance:
        classification: DepartureResponseClassification = "frozen_no_feasible_sample"
    elif feasible_weight < config.minimum_feasible_fraction:
        classification = "excluded_low_feasibility"
    elif feasible_weight < config.warning_feasible_fraction:
        classification = "warning"
    else:
        classification = "normal"
    if (
        config.infeasible_policy == "reject_cell"
        and infeasible_weight > config.weight_tolerance
    ):
        classification = "excluded_low_feasibility"

    if config.infeasible_policy == "condition_on_feasible" and feasible_weight > 0.0:
        normalization = feasible_weight
    else:
        normalization = 1.0
    conditional_weights = tuple(
        item.sample.original_weight / normalization for item, _ in feasible
    )
    accumulator: dict[int, float] = {}
    alternative_ids: set[str] = set()
    for conditional_weight, (_, alternatives) in zip(
        conditional_weights, feasible, strict=True
    ):
        for alternative, route_share in alternatives:
            alternative_ids.add(alternative.alternative_id)
            response = alternative_responses.get(alternative.alternative_id)
            if response is None:
                raise ValueError(
                    f"missing response for alternative {alternative.alternative_id}."
                )
            joint_weight = conditional_weight * route_share
            for measurement_index, value in sorted(response.items()):
                if measurement_index < 0 or not math.isfinite(value):
                    raise ValueError("alternative response contains an invalid value.")
                accumulator[measurement_index] = (
                    accumulator.get(measurement_index, 0.0) + joint_weight * value
                )
    indices = tuple(sorted(accumulator))
    return AveragedDepartureResponse(
        cell_key=cell_key,
        total_sample_count=len(ordered),
        feasible_sample_count=len(feasible),
        original_feasible_weight=feasible_weight,
        original_infeasible_weight=infeasible_weight,
        conditional_sample_weights=conditional_weights,
        feasible_sample_ids=tuple(item.sample.sample_id for item, _ in feasible),
        averaged_response=SparseWeightedResponse(
            measurement_indices=indices,
            values=tuple(accumulator[index] for index in indices),
        ),
        classification=classification,
        infeasibility_weight_by_reason=tuple(sorted(reason_weights.items())),
        distinct_alternative_ids=tuple(sorted(alternative_ids)),
    )


@dataclass(frozen=True, slots=True)
class DepartureSamplingEvaluation:
    level: int
    predicted_counts: np.ndarray
    journey_search_queries: int
    preprocessing_seconds: float
    phase_timings: Mapping[str, float] = field(default_factory=dict)
    peak_rss_bytes: int = 0
    artifact_bytes: int = 0
    response_nonzeros: int = 0
    response_classes: int = 0
    feasible_weight: float = 0.0
    infeasible_weight: float = 0.0
    classifications: Mapping[str, str] = field(default_factory=dict)
    operator_fingerprint: str = ""
    support_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        counts = np.array(self.predicted_counts, dtype=np.float64, copy=True)
        if counts.ndim != 1 or not np.all(np.isfinite(counts)):
            raise ValueError("predicted counts must be a finite vector.")
        counts.setflags(write=False)
        object.__setattr__(self, "predicted_counts", counts)


@dataclass(frozen=True, slots=True)
class DepartureSamplingPreflight:
    origin_period_groups: int
    distinct_origins: int | None
    number_of_periods: int | None
    cartesian_group_count: int | None
    avoided_cartesian_groups: int | None
    total_journey_queries: int
    hypothetical_cartesian_queries: int | None
    avoided_queries: int | None
    query_multiplier_from_single_sample: float
    estimated_temporary_bytes: int
    estimated_retained_bytes: int
    estimated_disk_bytes: int
    estimated_wall_seconds: float | None
    expected_wall_seconds_saved: float | None
    assumptions: Mapping[str, object]


def preflight_departure_sampling(
    *,
    origin_period_groups: int,
    samples_per_group: int,
    single_sample_retained_bytes: int = 0,
    single_sample_support_nonzeros: int = 0,
    observed_query_seconds: float | None = None,
    temporary_bytes_per_query: int = 0,
    distinct_origins: int | None = None,
    number_of_periods: int | None = None,
) -> DepartureSamplingPreflight:
    """Conservatively project sampling work before journey construction."""
    for value, name in (
        (origin_period_groups, "origin_period_groups"),
        (samples_per_group, "samples_per_group"),
        (single_sample_retained_bytes, "single_sample_retained_bytes"),
        (single_sample_support_nonzeros, "single_sample_support_nonzeros"),
        (temporary_bytes_per_query, "temporary_bytes_per_query"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if origin_period_groups == 0 or samples_per_group == 0:
        raise ValueError("origin-period groups and samples per group must be positive.")
    if observed_query_seconds is not None and (
        not math.isfinite(observed_query_seconds) or observed_query_seconds <= 0.0
    ):
        raise ValueError("observed_query_seconds must be finite and positive.")
    if (distinct_origins is None) != (number_of_periods is None):
        raise ValueError(
            "distinct_origins and number_of_periods must be supplied together."
        )
    if distinct_origins is not None and (
        isinstance(distinct_origins, bool)
        or not isinstance(distinct_origins, int)
        or distinct_origins <= 0
        or isinstance(number_of_periods, bool)
        or not isinstance(number_of_periods, int)
        or number_of_periods <= 0
    ):
        raise ValueError("Cartesian dimensions must be positive integers.")
    queries = origin_period_groups * samples_per_group
    cartesian_groups = (
        None
        if distinct_origins is None or number_of_periods is None
        else distinct_origins * number_of_periods
    )
    if cartesian_groups is not None and cartesian_groups < origin_period_groups:
        raise ValueError("sparse support cannot exceed its Cartesian envelope.")
    avoided_groups = (
        None if cartesian_groups is None else cartesian_groups - origin_period_groups
    )
    cartesian_queries = (
        None if cartesian_groups is None else cartesian_groups * samples_per_group
    )
    avoided_queries = (
        None if avoided_groups is None else avoided_groups * samples_per_group
    )
    # Retained sparse support need not grow linearly because identical paths are
    # merged. The linear estimate is deliberately conservative for admission.
    retained = single_sample_retained_bytes * samples_per_group
    support_bytes = single_sample_support_nonzeros * samples_per_group * 16
    retained = max(retained, support_bytes)
    return DepartureSamplingPreflight(
        origin_period_groups=origin_period_groups,
        distinct_origins=distinct_origins,
        number_of_periods=number_of_periods,
        cartesian_group_count=cartesian_groups,
        avoided_cartesian_groups=avoided_groups,
        total_journey_queries=queries,
        hypothetical_cartesian_queries=cartesian_queries,
        avoided_queries=avoided_queries,
        query_multiplier_from_single_sample=float(samples_per_group),
        estimated_temporary_bytes=temporary_bytes_per_query,
        estimated_retained_bytes=retained,
        estimated_disk_bytes=retained,
        estimated_wall_seconds=(
            None if observed_query_seconds is None else queries * observed_query_seconds
        ),
        expected_wall_seconds_saved=(
            None
            if observed_query_seconds is None or avoided_queries is None
            else avoided_queries * observed_query_seconds
        ),
        assumptions={
            "retained_support_growth": "conservative_linear_upper_projection",
            "sample_specific_objects_retained": "one_origin_period_batch",
            "identical_paths_merged": True,
        },
    )


@dataclass(frozen=True, slots=True)
class DepartureSamplingLevelChange:
    previous_level: int
    level: int
    relative_l1_change: float
    relative_l2_change: float
    maximum_absolute_change: float
    maximum_relative_group_change: float
    predicted_total_change: float
    zero_observation_mass_change: float | None
    classification_changes: tuple[str, ...]
    support_added: int
    support_removed: int


@dataclass(frozen=True, slots=True)
class DepartureSamplingConvergenceReport:
    levels: tuple[DepartureSamplingEvaluation, ...]
    changes: tuple[DepartureSamplingLevelChange, ...]
    stable: bool
    relative_change_tolerance: float


def compare_departure_sampling_levels(
    *,
    evaluator: Callable[[int], DepartureSamplingEvaluation],
    levels: Sequence[int] = (3, 6, 12),
    observations: object | None = None,
    denominator_floor: float = 1.0,
    relative_change_tolerance: float = 0.05,
    progress: SamplingProgress | None = None,
) -> DepartureSamplingConvergenceReport:
    """Compare preprocessing fidelities without fitting statistical parameters."""
    selected = tuple(levels)
    if not selected or tuple(sorted(set(selected))) != selected or selected[0] <= 0:
        raise ValueError("sampling levels must be positive and increasing.")
    if denominator_floor <= 0.0 or relative_change_tolerance < 0.0:
        raise ValueError("convergence thresholds are invalid.")
    observed = (
        None if observations is None else np.asarray(observations, dtype=np.float64)
    )
    started = time.perf_counter()
    evaluations: list[DepartureSamplingEvaluation] = []
    if progress is not None:
        progress(
            {
                "phase": "sampling_convergence",
                "status": "started",
                "completed_levels": 0,
                "total_levels": len(selected),
                "elapsed_seconds": 0.0,
            }
        )
    for index, level in enumerate(selected):
        try:
            evaluation = evaluator(level)
        except Exception as error:
            if progress is not None:
                progress(
                    {
                        "phase": "sampling_convergence",
                        "status": "failed",
                        "sampling_level": level,
                        "completed_levels": index,
                        "total_levels": len(selected),
                        "elapsed_seconds": time.perf_counter() - started,
                        "error": str(error),
                    }
                )
            raise
        if evaluation.level != level:
            raise ValueError("sampling evaluator returned the wrong level.")
        if (
            evaluations
            and evaluation.predicted_counts.shape
            != evaluations[0].predicted_counts.shape
        ):
            raise ValueError("sampling levels must retain observation ordering.")
        evaluations.append(evaluation)
        if progress is not None:
            elapsed = time.perf_counter() - started
            progress(
                {
                    "phase": "sampling_convergence",
                    "status": (
                        "completed"
                        if index + 1 == len(selected)
                        else "in_progress"
                    ),
                    "sampling_level": level,
                    "completed_levels": index + 1,
                    "total_levels": len(selected),
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": (
                        elapsed * (len(selected) - index - 1) / (index + 1)
                    ),
                    "eta_confidence": (
                        "complete" if index + 1 == len(selected) else "low"
                    ),
                    "peak_rss_bytes": _peak_rss_bytes(),
                }
            )
    changes: list[DepartureSamplingLevelChange] = []
    for previous, current in zip(evaluations, evaluations[1:]):
        difference = current.predicted_counts - previous.predicted_counts
        l1_denominator = max(
            denominator_floor, float(np.sum(np.abs(previous.predicted_counts)))
        )
        l2_denominator = max(
            denominator_floor, float(np.linalg.norm(previous.predicted_counts))
        )
        relative = np.abs(difference) / np.maximum(
            denominator_floor, np.abs(previous.predicted_counts)
        )
        zero_change = None
        if observed is not None:
            if observed.shape != current.predicted_counts.shape:
                raise ValueError("observations do not align with predicted counts.")
            mask = observed == 0.0
            zero_change = float(
                np.sum(current.predicted_counts[mask])
                - np.sum(previous.predicted_counts[mask])
            )
        keys = set(previous.classifications) | set(current.classifications)
        changes.append(
            DepartureSamplingLevelChange(
                previous_level=previous.level,
                level=current.level,
                relative_l1_change=float(np.sum(np.abs(difference)) / l1_denominator),
                relative_l2_change=float(np.linalg.norm(difference) / l2_denominator),
                maximum_absolute_change=float(np.max(np.abs(difference), initial=0.0)),
                maximum_relative_group_change=float(np.max(relative, initial=0.0)),
                predicted_total_change=float(np.sum(difference)),
                zero_observation_mass_change=zero_change,
                classification_changes=tuple(
                    sorted(
                        key
                        for key in keys
                        if previous.classifications.get(key)
                        != current.classifications.get(key)
                    )
                ),
                support_added=len(current.support_ids - previous.support_ids),
                support_removed=len(previous.support_ids - current.support_ids),
            )
        )
    stable = bool(changes) and all(
        change.relative_l1_change <= relative_change_tolerance
        and change.relative_l2_change <= relative_change_tolerance
        for change in changes
    )
    return DepartureSamplingConvergenceReport(
        tuple(evaluations), tuple(changes), stable, relative_change_tolerance
    )


@dataclass(frozen=True, slots=True)
class DepartureSamplingDiagnosticsReport:
    configuration: Mapping[str, object]
    network_totals: Mapping[str, object]
    per_period: tuple[Mapping[str, object], ...]
    per_origin: tuple[Mapping[str, object], ...]
    per_destination: tuple[Mapping[str, object], ...]
    cells: tuple[Mapping[str, object], ...]
    observation_alignment: Mapping[str, object] | None = None
    convergence: DepartureSamplingConvergenceReport | None = None
    accounting_valid: bool = True


def build_departure_sampling_diagnostics(
    *,
    sampled: SampledJourneyChoiceResult,
    config: DepartureTimeSamplingConfig,
    time_periods: Sequence[JourneyTimePeriod] = (),
    observations: object | None = None,
    predicted_counts: object | None = None,
    measurement_types: Sequence[str] | None = None,
    measurement_period_ids: Sequence[str] | None = None,
    vehicle_journey_ids: Sequence[str] | None = None,
    convergence: DepartureSamplingConvergenceReport | None = None,
    progress: SamplingProgress | None = None,
) -> DepartureSamplingDiagnosticsReport:
    """Build JSON-ready sampling, cell, and optional observation diagnostics."""
    started = time.perf_counter()
    if progress is not None:
        progress(
            {
                "phase": "departure_sampling_diagnostics",
                "status": "started",
                "completed_cells": 0,
                "total_cells": len(sampled.cells),
                "elapsed_seconds": 0.0,
            }
        )
    choice_by_cell = {
        ResponseCellKey(
            choice.origin_physical_stop_id,
            choice.destination_physical_stop_id,
            choice.origin_time_period_id,
        ): choice
        for choice in sampled.journey_choices.choice_sets
    }
    rows: list[Mapping[str, object]] = []
    for index, cell in enumerate(sampled.cells, start=1):
        choice = choice_by_cell.get(cell.cell_key)
        if choice is None:
            waits = journeys = transfers = np.asarray([], dtype=np.float64)
            distinct_lines: set[str] = set()
            distinct_trips: set[str] = set()
        else:
            shares = np.asarray(choice.initial_shares, dtype=np.float64)
            waits = np.asarray(
                [item.wait_seconds for item in choice.alternatives], dtype=np.float64
            )
            journeys = np.asarray(
                [item.travel_seconds for item in choice.alternatives], dtype=np.float64
            )
            transfers = np.asarray(
                [item.transfers for item in choice.alternatives], dtype=np.float64
            )
            waits = waits * shares
            journeys = journeys * shares
            transfers = transfers * shares
            distinct_lines = {
                pattern
                for item in choice.alternatives
                for pattern in item.route_pattern_ids
            }
            distinct_trips = {
                leg.trip_id for item in choice.alternatives for leg in item.transit_legs
            }
        rows.append(
            {
                "origin": cell.cell_key.origin_physical_stop_id,
                "destination": cell.cell_key.destination_physical_stop_id,
                "period": cell.cell_key.origin_time_period_id,
                "total_samples": cell.total_sample_count,
                "feasible_samples": cell.feasible_sample_count,
                "original_feasible_fraction": cell.original_feasible_weight,
                "original_infeasible_fraction": cell.original_infeasible_weight,
                "classification": cell.classification,
                "cell_status": cell.cell_status,
                "timetable_feasible_fixed_zero": cell.timetable_feasible_fixed_zero,
                "fixed_positive_assignment_failed": (
                    cell.fixed_positive_assignment_failed
                ),
                "unexpected_status_change": cell.unexpected_status_change,
                "infeasibility_reasons_and_weights": dict(
                    cell.infeasibility_weight_by_reason
                ),
                "conditional_weight_concentration": (
                    cell.conditional_weight_concentration
                ),
                "expected_waiting_seconds": float(np.sum(waits)),
                "expected_journey_seconds": float(np.sum(journeys)),
                "expected_transfers": float(np.sum(transfers)),
                "number_of_distinct_paths": cell.distinct_paths,
                "number_of_distinct_lines": len(distinct_lines),
                "number_of_distinct_vehicle_journeys": len(distinct_trips),
                "averaged_response_nonzeros": None,
            }
        )
        if progress is not None and (
            index == len(sampled.cells) or index % config.progress_interval_groups == 0
        ):
            elapsed = time.perf_counter() - started
            progress(
                {
                    "phase": "departure_sampling_diagnostics",
                    "status": (
                        "completed" if index == len(sampled.cells) else "in_progress"
                    ),
                    "completed_cells": index,
                    "total_cells": len(sampled.cells),
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": (
                        elapsed * (len(sampled.cells) - index) / index
                    ),
                    "eta_confidence": (
                        "complete" if index == len(sampled.cells) else (
                            "low" if index < 3 else "medium"
                        )
                    ),
                    "peak_rss_bytes": _peak_rss_bytes(),
                }
            )

    def grouped(field: str) -> tuple[Mapping[str, object], ...]:
        def feasible_fraction(row: Mapping[str, object]) -> float:
            value = row["original_feasible_fraction"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("cell feasible fraction must be numeric.")
            return float(value)

        values = sorted({str(row[field]) for row in rows})
        result: list[Mapping[str, object]] = []
        for value in values:
            selected = [row for row in rows if str(row[field]) == value]
            fractions = np.asarray([feasible_fraction(row) for row in selected])
            result.append(
                {
                    field: value,
                    "cells": len(selected),
                    "mean_feasible_fraction": float(np.mean(fractions)),
                    "minimum_feasible_fraction": float(np.min(fractions)),
                    "no_feasible_cells": sum(
                        row["classification"] == "frozen_no_feasible_sample"
                        for row in selected
                    ),
                    "low_feasibility_cells": sum(
                        row["classification"] == "excluded_low_feasibility"
                        for row in selected
                    ),
                    "warning_cells": sum(
                        row["classification"] == "warning" for row in selected
                    ),
                    "normal_cells": sum(
                        row["classification"] == "normal" for row in selected
                    ),
                }
            )
        return tuple(result)

    observation_alignment: Mapping[str, object] | None = None
    if observations is not None or predicted_counts is not None:
        if observations is None or predicted_counts is None:
            raise ValueError(
                "observations and predicted_counts must be supplied together."
            )
        observed = np.asarray(observations, dtype=np.float64)
        predicted = np.asarray(predicted_counts, dtype=np.float64)
        if (
            observed.ndim != 1
            or predicted.shape != observed.shape
            or not np.all(np.isfinite(observed))
            or not np.all(np.isfinite(predicted))
            or np.any(observed < 0.0)
            or np.any(predicted < 0.0)
        ):
            raise ValueError(
                "observations and predictions must be aligned nonnegative vectors."
            )
        residual = predicted - observed
        zero = observed == 0.0
        positive = ~zero
        types = tuple(measurement_types or ("unknown",) * observed.size)
        periods = tuple(measurement_period_ids or ("unknown",) * observed.size)
        journeys_ids = tuple(vehicle_journey_ids or ("unknown",) * observed.size)
        if any(
            len(values) != observed.size for values in (types, periods, journeys_ids)
        ):
            raise ValueError("observation metadata must align with observations.")

        def totals_for(kind: str) -> dict[str, float]:
            mask = np.asarray([value == kind for value in types])
            return {
                "observed": float(np.sum(observed[mask])),
                "predicted": float(np.sum(predicted[mask])),
            }

        largest = np.argsort(-np.abs(residual))[: min(20, observed.size)]
        journey_predictions: dict[str, float] = {}
        journey_observations: dict[str, float] = {}
        for journey, observed_value, predicted_value in zip(
            journeys_ids, observed, predicted, strict=True
        ):
            journey_predictions[journey] = journey_predictions.get(
                journey, 0.0
            ) + float(predicted_value)
            journey_observations[journey] = journey_observations.get(
                journey, 0.0
            ) + float(observed_value)
        total_prediction = float(np.sum(predicted))
        zero_prediction = float(np.sum(predicted[zero]))
        by_period = {
            period: {
                "observed": float(
                    np.sum(observed[np.asarray([value == period for value in periods])])
                ),
                "predicted": float(
                    np.sum(
                        predicted[np.asarray([value == period for value in periods])]
                    )
                ),
            }
            for period in sorted(set(periods))
        }
        observation_alignment = {
            "boarding_totals": totals_for("boarding"),
            "alighting_totals": totals_for("alighting"),
            "observed_total": float(np.sum(observed)),
            "predicted_total": total_prediction,
            "totals_by_period": by_period,
            "predicted_mass_on_zero_rows": zero_prediction,
            "fraction_prediction_on_zero_rows": (
                zero_prediction / total_prediction if total_prediction > 0.0 else 0.0
            ),
            "positive_row_predicted_total": float(np.sum(predicted[positive])),
            "rmse": float(np.sqrt(np.mean(residual * residual))),
            "mae": float(np.mean(np.abs(residual))),
            "maximum_predicted_measurement": float(np.max(predicted, initial=0.0)),
            "largest_absolute_residuals": [
                {
                    "row": int(row),
                    "observed": float(observed[row]),
                    "predicted": float(predicted[row]),
                    "residual": float(residual[row]),
                    "period": periods[row],
                    "vehicle_journey": journeys_ids[row],
                }
                for row in largest
            ],
            "maximum_vehicle_journey_prediction_fraction": (
                max(journey_predictions.values(), default=0.0) / total_prediction
                if total_prediction > 0.0
                else 0.0
            ),
            "positive_observation_journeys_with_negligible_prediction": sum(
                journey_observations[key] > 0.0
                and journey_predictions.get(key, 0.0) <= 1.0e-9
                for key in journey_observations
            ),
            "zero_observation_journeys_with_material_prediction": sum(
                journey_observations.get(key, 0.0) == 0.0 and value > 1.0
                for key, value in journey_predictions.items()
            ),
        }
    feasible_weight = sum(item.original_feasible_weight for item in sampled.cells)
    infeasible_weight = sum(item.original_infeasible_weight for item in sampled.cells)
    accounting_valid = all(
        math.isclose(
            item.original_feasible_weight + item.original_infeasible_weight,
            1.0,
            abs_tol=config.weight_tolerance,
            rel_tol=0.0,
        )
        for item in sampled.cells
    )
    return DepartureSamplingDiagnosticsReport(
        configuration={
            **asdict(config),
            "sampling_fingerprint": sampled.sampling_fingerprint,
            "exact_samples": [asdict(sample) for sample in sampled.samples],
            "periods": [asdict(period) for period in time_periods],
            "interpretation": "completed_public_transport_journeys",
        },
        network_totals={
            "origin_period_groups": len(
                {
                    (sample.origin_physical_stop_id, sample.time_period_id)
                    for sample in sampled.samples
                }
            ),
            "od_period_cells": len(sampled.cells),
            "canonical_free_cells": sum(
                item.cell_status == "free" for item in sampled.cells
            ),
            "canonical_fixed_zero_cells": sum(
                item.cell_status == "fixed_zero" for item in sampled.cells
            ),
            "canonical_fixed_positive_cells": sum(
                item.cell_status == "fixed_positive" for item in sampled.cells
            ),
            "timetable_feasible_fixed_zero_cells": sum(
                item.timetable_feasible_fixed_zero for item in sampled.cells
            ),
            "fixed_positive_assignment_failures": sum(
                item.fixed_positive_assignment_failed for item in sampled.cells
            ),
            "unexpected_status_changes": sum(
                item.unexpected_status_change for item in sampled.cells
            ),
            "total_desired_time_samples": len(sampled.samples),
            "feasible_sample_weight_across_cells": feasible_weight,
            "infeasible_sample_weight_across_cells": infeasible_weight,
            "mass_canonicalization_count": sum(
                item.mass_canonicalization_applied for item in sampled.cells
            ),
            "maximum_absolute_mass_canonicalization_delta": max(
                (
                    abs(item.mass_canonicalization_delta)
                    for item in sampled.cells
                ),
                default=0.0,
            ),
            "retained_cells": sum(
                item.classification in {"normal", "warning"} for item in sampled.cells
            ),
            "warning_cells": sum(
                item.classification == "warning" for item in sampled.cells
            ),
            "excluded_cells": sum(
                item.classification == "excluded_low_feasibility"
                for item in sampled.cells
            ),
            "frozen_cells": sum(
                item.classification == "frozen_no_feasible_sample"
                for item in sampled.cells
            ),
            "distinct_averaged_paths": sum(
                item.distinct_paths for item in sampled.cells
            ),
            "peak_rss_bytes": _peak_rss_bytes(),
            "diagnostic_seconds": time.perf_counter() - started,
        },
        per_period=grouped("period"),
        per_origin=grouped("origin"),
        per_destination=grouped("destination"),
        cells=tuple(rows),
        observation_alignment=observation_alignment,
        convergence=convergence,
        accounting_valid=accounting_valid,
    )


@dataclass(frozen=True, slots=True)
class DepartureSamplingRecommendation:
    code: str
    severity: Literal["info", "warning", "strong_warning", "blocking"]
    evidence: Mapping[str, object]
    suggestion: str
    expected_cost: Literal["low", "moderate", "high"]
    affected_cells: int


def recommend_departure_sampling_actions(
    report: DepartureSamplingDiagnosticsReport,
    *,
    unstable_fraction_threshold: float = 0.05,
    low_feasibility_fraction_threshold: float = 0.25,
) -> tuple[DepartureSamplingRecommendation, ...]:
    """Translate measured diagnostics into advisory, non-mutating actions."""

    def number(item: Mapping[str, object], key: str, default: float = 0.0) -> float:
        value = item.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"diagnostic field {key!r} must be numeric.")
        return float(value)

    recommendations: list[DepartureSamplingRecommendation] = []
    cells = report.cells
    unexpected = [
        item for item in cells if item.get("unexpected_status_change") is True
    ]
    if unexpected:
        recommendations.append(
            DepartureSamplingRecommendation(
                "unexpected_cell_status_change",
                "blocking",
                {"affected_cells": len(unexpected)},
                "Rebuild from the canonical free/fixed cell classification before estimation.",
                "low",
                len(unexpected),
            )
        )
    failed_fixed = [
        item for item in cells if item.get("fixed_positive_assignment_failed") is True
    ]
    if failed_fixed:
        recommendations.append(
            DepartureSamplingRecommendation(
                "fixed_positive_assignment_infeasible",
                "blocking",
                {"affected_cells": len(failed_fixed)},
                "Correct routing support or the fixed-positive specification before estimation.",
                "moderate",
                len(failed_fixed),
            )
        )
    feasible_fixed_zero = [
        item for item in cells if item.get("timetable_feasible_fixed_zero") is True
    ]
    if feasible_fixed_zero:
        recommendations.append(
            DepartureSamplingRecommendation(
                "timetable_feasible_fixed_zero",
                "info",
                {"affected_cells": len(feasible_fixed_zero)},
                "Keep these cells fixed at zero; timetable feasibility does not change their canonical status.",
                "low",
                len(feasible_fixed_zero),
            )
        )
    if not report.accounting_valid:
        recommendations.append(
            DepartureSamplingRecommendation(
                "stop_before_estimation",
                "blocking",
                {"accounting_valid": False},
                "Correct sample-weight or response accounting before estimation.",
                "moderate",
                len(cells),
            )
        )
    if report.convergence is not None and not report.convergence.stable:
        recommendations.append(
            DepartureSamplingRecommendation(
                "increase_sampling_resolution",
                "strong_warning",
                {
                    "changes": [asdict(item) for item in report.convergence.changes],
                    "threshold": report.convergence.relative_change_tolerance,
                },
                "Increase samples per period and evaluate the next convergence level.",
                "high",
                len(cells),
            )
        )
    low = [
        item
        for item in cells
        if number(item, "original_feasible_fraction", 1.0)
        < low_feasibility_fraction_threshold
    ]
    if low:
        recommendations.append(
            DepartureSamplingRecommendation(
                "exclude_or_freeze_low_feasibility",
                "strong_warning",
                {
                    "threshold": low_feasibility_fraction_threshold,
                    "affected_cells": len(low),
                },
                "Freeze zero-feasibility cells and review exclusion of uncertain low-feasibility cells.",
                "low",
                len(low),
            )
        )
    horizon = sum(number(item, "outside_horizon_weight") for item in cells)
    if horizon > unstable_fraction_threshold:
        recommendations.append(
            DepartureSamplingRecommendation(
                "extend_timetable_horizon",
                "warning",
                {
                    "outside_horizon_weight": horizon,
                    "threshold": unstable_fraction_threshold,
                },
                "Extend the loaded timetable so journeys starting in-period can complete.",
                "moderate",
                len(cells),
            )
        )
    zero_mass = (
        0.0
        if report.observation_alignment is None
        else number(report.observation_alignment, "fraction_prediction_on_zero_rows")
    )
    if zero_mass > unstable_fraction_threshold:
        recommendations.append(
            DepartureSamplingRecommendation(
                "aggregate_observations",
                "warning",
                {
                    "fraction_prediction_on_zero_rows": zero_mass,
                    "threshold": unstable_fraction_threshold,
                },
                "Review stop/line/period or stop/period count aggregation while retaining trip-level validation separately.",
                "low",
                len(cells),
            )
        )
    concentrated = [
        item for item in cells if number(item, "conditional_weight_concentration") > 0.9
    ]
    if concentrated:
        recommendations.append(
            DepartureSamplingRecommendation(
                "investigate_response_concentration",
                "warning",
                {"concentration_threshold": 0.9},
                "Increase sampling and review path merging and vehicle-trip identities.",
                "moderate",
                len(concentrated),
            )
        )
    abrupt = [
        item for item in cells if number(item, "abrupt_feasibility_changes") > 0.0
    ]
    if abrupt:
        recommendations.append(
            DepartureSamplingRecommendation(
                "split_broad_time_period",
                "warning",
                {"affected_cells": len(abrupt)},
                "Review evidence-based split times where adjacent desired-time samples change feasibility or service support.",
                "high",
                len(abrupt),
            )
        )
    trigger_specs = (
        (
            "review_waiting_time_constraint",
            "initial_wait_exceeded_weight",
            "Review waiting-time distributions and candidate thresholds without changing the active model.",
        ),
        (
            "review_journey_duration_constraint",
            "journey_duration_exceeded_weight",
            "Report recovery under candidate journey-duration limits before considering a change.",
        ),
        (
            "review_transfer_limit",
            "transfer_limit_exceeded_weight",
            "Report recovery from one additional transfer without changing the active limit.",
        ),
        (
            "add_or_review_footpaths",
            "missing_transfer_connection_weight",
            "Request and review caller-provided transfer footpaths at concentrated failure locations.",
        ),
    )
    for code, field_name, suggestion in trigger_specs:
        affected = [
            item
            for item in cells
            if number(item, field_name) > unstable_fraction_threshold
        ]
        if affected:
            recommendations.append(
                DepartureSamplingRecommendation(
                    code,
                    "warning",
                    {
                        "weight_threshold": unstable_fraction_threshold,
                        "affected_cells": len(affected),
                    },
                    suggestion,
                    "moderate",
                    len(affected),
                )
            )
    low_fraction = len(low) / max(1, len(cells))
    if low_fraction > 0.5:
        recommendations.append(
            DepartureSamplingRecommendation(
                "stop_before_estimation",
                "blocking",
                {
                    "insufficient_feasibility_cell_fraction": low_fraction,
                    "threshold": 0.5,
                },
                "Resolve widespread sampling infeasibility before fitting the statistical model.",
                "high",
                len(low),
            )
        )
    if (
        report.convergence is not None
        and report.convergence.stable
        and zero_mass > unstable_fraction_threshold
    ):
        recommendations.append(
            DepartureSamplingRecommendation(
                "review_nonuniform_departure_profile",
                "info",
                {
                    "sampling_stable": True,
                    "fraction_prediction_on_zero_rows": zero_mass,
                },
                "Consider caller-provided departure profiles or narrower periods; do not infer weights from service supply.",
                "high",
                len(cells),
            )
        )
    return tuple(recommendations)
