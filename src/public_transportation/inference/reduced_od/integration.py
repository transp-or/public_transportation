"""Stable persisted integration surface for the compact J0 workflow."""

from __future__ import annotations

import hashlib
import math
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence, cast

import numpy as np

from public_transportation.domain import Scenario
from public_transportation.measurement.schema import MeasurementTable
from public_transportation.preprocessing.reduced_od.artifact_store import (
    ReducedODArtifactStoreError,
    load_reduced_od_phase_artifact,
    save_reduced_od_phase_artifact,
)
from public_transportation.preprocessing.reduced_od.artifacts import (
    canonical_json,
)
from public_transportation.preprocessing.reduced_od.config import ReducedODConfig
from public_transportation.preprocessing.reduced_od.departure_sampling import (
    DepartureCellStatus,
    DepartureTimeSamplingConfig,
    DesiredDepartureSample,
    SampledJourneyCellDiagnostics,
    SparseWeightedResponse,
    generate_uniform_midpoint_samples,
    merge_sampled_journey_choices,
    validate_sample_weights,
)
from public_transportation.preprocessing.reduced_od.period_preflight import (
    preflight_reduced_od_time_periods,
)
from public_transportation.preprocessing.reduced_od.adaptive_departure_quadrature import (
    DepartureQuadratureDiagnostics,
    generate_fixed_time_step_samples,
    integrate_adaptive_departure_response,
)
from public_transportation.preprocessing.reduced_od.journey_choices import (
    JourneyChoiceDiagnostics,
    JourneyAlternative,
    JourneyChoicePolicy,
    JourneyChoiceResult,
    JourneyChoiceSet,
    JourneyTimePeriod,
    build_journey_choices,
)
from public_transportation.preprocessing.reduced_od.raptor import (
    Footpath,
    RaptorQuery,
    run_raptor_query,
)
from public_transportation.preprocessing.reduced_od.response_atoms import (
    MeasurementResponseArtifact,
    ResponseCellKey,
    build_measurement_response,
    _resolve_measurements,
)
from public_transportation.preprocessing.reduced_od.timetable_index import (
    TimetableIndex,
    prepare_reduced_od_timetable,
)

from .contracts import ReducedODModelContract
from .features import ConditionalGravityFeatures, build_conditional_gravity_features
from .objective import MinimalGravityProblem
from .parameters import MinimalGravityParameterLayout
from .response_operator import ReducedResponseOperator, build_reduced_response_operator
from .specification import MinimalGravitySpecification


ReducedODCachePolicy = Literal["read_only", "reuse_or_build", "rebuild"]
ProgressCallback = Callable[[Mapping[str, object]], None]

PHASES = (
    "configuration",
    "physical_stops",
    "service_periods_route_patterns",
    "timetable_index",
    "departure_time_samples",
    "journey_choices",
    "measurement_response",
    "response_equivalence",
    "production_inputs",
    "destination_attractiveness",
    "conditional_gravity_features",
    "reduced_response_operator",
    "problem_manifest",
)


def reduced_od_phase_configuration_fingerprint(
    configuration: ReducedODConfig, phase: str
) -> str:
    """Fingerprint only configuration fields that can affect one phase."""

    base: dict[str, object] = {"schema_version": configuration.schema_version}
    if phase in {
        "physical_stops",
        "service_periods_route_patterns",
        "timetable_index",
        "departure_time_samples",
        "journey_choices",
        "measurement_response",
        "response_equivalence",
        "reduced_response_operator",
    }:
        base["observations"] = {
            "service_day": configuration.observations.service_day,
            "analysis_start_seconds": (
                configuration.observations.analysis_start_seconds
            ),
            "analysis_end_seconds": configuration.observations.analysis_end_seconds,
            "after_midnight_convention": (
                configuration.observations.after_midnight_convention
            ),
        }
        base["stops"] = {
            "mapping_policy": configuration.stops.mapping_policy,
            "physical_stop_mapping_path": (
                None
                if configuration.stops.physical_stop_mapping_path is None
                else str(configuration.stops.physical_stop_mapping_path)
            ),
            "footpaths_path": (
                None
                if configuration.stops.footpaths_path is None
                else str(configuration.stops.footpaths_path)
            ),
        }
    if phase in {
        "journey_choices",
        "measurement_response",
        "response_equivalence",
        "reduced_response_operator",
    }:
        base["journeys"] = asdict(configuration.journeys)
    if phase in {
        "measurement_response",
        "response_equivalence",
        "reduced_response_operator",
    }:
        base["measurement_policy"] = asdict(configuration.observations)
        base["output_spatial_level"] = configuration.outputs.spatial_level
    if phase == "problem_manifest":
        base.update(
            {
                "model": asdict(configuration.model),
                "productions": {
                    "mode": configuration.productions.mode,
                    "semantics": configuration.productions.semantics,
                    "input_path": (
                        None
                        if configuration.productions.input_path is None
                        else str(configuration.productions.input_path)
                    ),
                    "basis": configuration.productions.basis,
                },
                "validation": asdict(configuration.validation),
            }
        )
    if phase == "configuration":
        return configuration.fingerprint
    return hashlib.sha256(canonical_json(base).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ReducedODPreparationInputs:
    """Typed policies not inferable from scenario counts."""

    departure_seconds_by_origin: Mapping[str, Sequence[int]]
    production_inputs: Mapping[tuple[str, str], float]
    destination_attractiveness: Mapping[tuple[str, str], float]
    physical_stop_mapping: Mapping[str, str] | None = None
    footpaths: tuple[Footpath, ...] = ()
    time_periods: tuple[JourneyTimePeriod, ...] = ()
    fixed_demand: Mapping[ResponseCellKey, float] | None = None
    route_pattern_initial_weights: Mapping[str, float] | None = None
    departure_time_sampling: DepartureTimeSamplingConfig | None = None
    departure_sampling_origin_period_groups: Sequence[tuple[str, str]] | None = None
    candidate_od_pairs: Sequence[tuple[str, str]] | None = None
    prior_demand: Mapping[object, float] | None = None


def _sampling_support(
    inputs: ReducedODPreparationInputs,
) -> tuple[tuple[tuple[str, str], ...], dict[ResponseCellKey, DepartureCellStatus]]:
    """Return exact sparse query support and immutable canonical cell statuses."""
    production_groups = set(inputs.production_inputs)
    explicit = inputs.departure_sampling_origin_period_groups
    groups = production_groups if explicit is None else set(explicit)
    if explicit is not None and len(groups) != len(tuple(explicit)):
        raise ValueError("departure sampling origin-period groups must be unique.")
    missing = sorted(production_groups - groups)
    if missing:
        raise ValueError(
            f"departure sampling support omits production groups: {missing}."
        )
    fixed = dict(inputs.fixed_demand or {})
    for key, value in fixed.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"fixed demand for {key.tuple!r} must be non-negative.")
        if value > 0.0:
            groups.add((key.origin_physical_stop_id, key.origin_time_period_id))
    if any(not origin or not period for origin, period in groups):
        raise ValueError("departure sampling groups require non-empty identifiers.")

    status: dict[ResponseCellKey, DepartureCellStatus] = {}
    for origin, period in sorted(groups):
        for destination, attraction_period in inputs.destination_attractiveness:
            if attraction_period != period:
                continue
            key = ResponseCellKey(origin, destination, period)
            if key in fixed:
                status[key] = "fixed_positive" if fixed[key] > 0.0 else "fixed_zero"
            elif (origin, period) in production_groups:
                status[key] = "free"
    for key, value in fixed.items():
        group = (key.origin_physical_stop_id, key.origin_time_period_id)
        if group in groups:
            status[key] = "fixed_positive" if value > 0.0 else "fixed_zero"
    return tuple(sorted(groups)), status


def _retained_productions(
    response: MeasurementResponseArtifact,
    productions: Mapping[tuple[str, str], float],
) -> tuple[dict[tuple[str, str], float], dict[str, object]]:
    retained = {
        (key.origin_physical_stop_id, key.origin_time_period_id)
        for key in response.free_cell_keys
    }
    declared = set(productions)
    missing = sorted(retained - declared)
    extra = sorted(declared - retained)
    if missing:
        raise ValueError(
            f"retained free origin-period groups lack production inputs: {missing}."
        )
    return (
        {key: float(productions[key]) for key in sorted(retained)},
        {
            "declared_groups": len(declared),
            "retained_free_groups": len(retained),
            "missing_groups": tuple(missing),
            "excluded_or_fixed_only_groups": tuple(extra),
        },
    )


def _departure_sampling_identity(inputs: ReducedODPreparationInputs) -> str:
    if inputs.departure_time_sampling is None:
        payload: object = {
            "mode": "legacy_representative_departure",
            "departures": {
                origin: sorted(set(int(value) for value in values))
                for origin, values in sorted(inputs.departure_seconds_by_origin.items())
            },
        }
    else:
        groups, statuses = _sampling_support(inputs)
        payload = {
            "mode": "desired_departure_sampling",
            "sampling_integration_schema": 6,
            "journey_period_semantics_version": 2,
            "config": asdict(inputs.departure_time_sampling),
            "origin_period_groups": groups,
            "periods": [asdict(period) for period in inputs.time_periods],
            "cell_status": [
                [*key.tuple, status] for key, status in sorted(statuses.items())
            ],
            "fixed_demand": [
                [*key.tuple, value]
                for key, value in sorted((inputs.fixed_demand or {}).items())
            ],
        }
    if isinstance(payload, dict):
        payload["candidate_od_pairs"] = (
            None
            if inputs.candidate_od_pairs is None
            else sorted([list(pair) for pair in inputs.candidate_od_pairs])
        )
        payload["prior_demand"] = (
            None
            if inputs.prior_demand is None
            else sorted(
                [
                    [str(key), float(value)]
                    for key, value in inputs.prior_demand.items()
                ],
                key=lambda item: item[0],
            )
        )
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _journey_service_signature(choices: JourneyChoiceResult) -> str:
    """Canonical service identity independent of query-specific alternative IDs."""
    payload = [
        [
            choice.destination_physical_stop_id,
            [
                [
                    [
                        leg.trip_id,
                        leg.board_physical_stop_id,
                        leg.alight_physical_stop_id,
                    ]
                    for leg in alternative.transit_legs
                ]
                for alternative in choice.alternatives
            ],
        ]
        for choice in choices.choice_sets
    ]
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _journey_route_pattern_signature(choices: JourneyChoiceResult) -> str:
    """Stable service shape without exact trip identities or departure times."""
    payload = [
        [
            choice.destination_physical_stop_id,
            [
                [
                    list(alternative.route_pattern_ids),
                    [
                        [leg.board_physical_stop_id, leg.alight_physical_stop_id]
                        for leg in alternative.transit_legs
                    ],
                ]
                for alternative in choice.alternatives
            ],
        ]
        for choice in choices.choice_sets
    ]
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _journey_measurement_comparison_response(
    choices: JourneyChoiceResult,
    *,
    measurement_lookup: Mapping[tuple[str, str, str, int], tuple[int, ...]],
    coordinate_index: dict[tuple[str, int], int],
    support_only: bool,
) -> SparseWeightedResponse:
    """Expected active-measurement response, separated by destination cell."""
    values: dict[int, float] = {}
    for choice in choices.choice_sets:
        for alternative, share in zip(
            choice.alternatives, choice.initial_shares, strict=True
        ):
            for event in alternative.events:
                leg = alternative.transit_legs[event.leg_index]
                kind = (
                    "boarding" if "boarding" in event.event_kind.value else "alighting"
                )
                key = (leg.trip_id, kind, event.physical_stop_id, event.seconds)
                for row in measurement_lookup.get(key, ()):
                    coordinate = coordinate_index.setdefault(
                        (choice.destination_physical_stop_id, row), len(coordinate_index)
                    )
                    values[coordinate] = 1.0 if support_only else values.get(coordinate, 0.0) + share
    ordered = tuple(sorted(values))
    return SparseWeightedResponse(ordered, tuple(values[index] for index in ordered))


@dataclass(frozen=True, slots=True)
class PreparedReducedODArtifacts:
    directory: Path
    paths: Mapping[str, Path]
    fingerprints: Mapping[str, str]
    dimensions: Mapping[str, int]
    retained_bytes: int
    estimated_in_memory_bytes: int
    phase_diagnostics: tuple[Mapping[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "directory": str(self.directory),
            "paths": {key: str(value) for key, value in self.paths.items()},
            "fingerprints": dict(self.fingerprints),
            "dimensions": dict(self.dimensions),
            "retained_bytes": self.retained_bytes,
            "estimated_in_memory_bytes": self.estimated_in_memory_bytes,
            "phase_diagnostics": [dict(item) for item in self.phase_diagnostics],
        }


@dataclass(frozen=True, slots=True)
class LoadedReducedODArtifacts:
    configuration: ReducedODConfig
    timetable: TimetableIndex
    departure_time_samples: object
    departure_sampling_cells: tuple[SampledJourneyCellDiagnostics, ...]
    journey_choices: JourneyChoiceResult
    measurement_response: MeasurementResponseArtifact
    production_inputs: Mapping[tuple[str, str], float]
    destination_attractiveness: Mapping[tuple[str, str], float]
    features: ConditionalGravityFeatures
    response_operator: ReducedResponseOperator
    fixed_demand: Mapping[ResponseCellKey, float]
    paths: Mapping[str, Path]
    fingerprints: Mapping[str, str]
    manifests: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class BuiltMinimalGravityProblem:
    problem: MinimalGravityProblem
    model_fingerprint: str
    artifact_fingerprints: Mapping[str, str]
    parameter_names: tuple[str, ...]
    raw_parameter_dimension: int
    transformed_parameter_dimension: int
    measurement_count: int
    canonical_cell_count: int
    free_cell_count: int
    fixed_zero_cell_count: int
    fixed_positive_cell_count: int
    origin_time_group_count: int
    response_nonzeros: int
    response_classes: int
    retained_artifact_bytes: int
    estimated_runtime_memory: int
    production_mode: str
    likelihood: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("problem")
        return payload


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if value > 10_000_000 else value * 1024


def _phase_path(directory: Path, phase: str) -> Path:
    return directory / phase


def _merge_choices(results: Sequence[JourneyChoiceResult]) -> JourneyChoiceResult:
    grouped: dict[tuple[str, str, str], dict[str, JourneyAlternative]] = {}
    for result in results:
        for choice in result.choice_sets:
            key = (
                choice.origin_physical_stop_id,
                choice.destination_physical_stop_id,
                choice.origin_time_period_id,
            )
            alternatives_for_key = grouped.setdefault(key, {})
            for alternative in choice.alternatives:
                alternatives_for_key.setdefault(alternative.alternative_id, alternative)
    choice_sets: list[JourneyChoiceSet] = []
    for key, alternatives_by_id in sorted(grouped.items()):
        alternatives = tuple(
            alternatives_by_id[item] for item in sorted(alternatives_by_id)
        )
        shares = tuple(np.full(len(alternatives), 1.0 / len(alternatives)))
        choice_sets.append(JourneyChoiceSet(*key, alternatives, shares))
    retained = sum(len(item.alternatives) for item in choice_sets)
    return JourneyChoiceResult(
        choice_sets=tuple(choice_sets),
        diagnostics=JourneyChoiceDiagnostics(
            feasible_destinations=len(choice_sets),
            candidate_alternatives=retained,
            retained_alternatives=retained,
            pruned_alternatives=0,
            choice_cells=len(choice_sets),
            maximum_candidates_in_cell=max(
                (len(item.alternatives) for item in choice_sets), default=0
            ),
            route_initialized_alternatives=0,
            estimated_payload_bytes=0,
        ),
    )


def prepare_reduced_od_artifacts(
    *,
    scenario: Scenario,
    measurements: MeasurementTable,
    configuration: ReducedODConfig,
    inputs: ReducedODPreparationInputs,
    output_directory: str | Path,
    cache_policy: ReducedODCachePolicy = "reuse_or_build",
    progress: ProgressCallback | None = None,
    journey_progress_interval_queries: int = 25,
    journey_progress_interval_seconds: float = 5.0,
) -> PreparedReducedODArtifacts:
    """Build and atomically persist every named J0 preprocessing phase.

    This path uses timetable RAPTOR and sparse response atoms only.  It never
    imports or constructs the detailed time-expanded assignment operator.
    """
    if cache_policy not in {"read_only", "reuse_or_build", "rebuild"}:
        raise ValueError("unsupported cache_policy.")
    if journey_progress_interval_queries <= 0:
        raise ValueError("journey_progress_interval_queries must be positive.")
    if journey_progress_interval_seconds <= 0.0:
        raise ValueError("journey_progress_interval_seconds must be positive.")
    period_preflight = preflight_reduced_od_time_periods(
        inputs.time_periods,
        sampling_config=inputs.departure_time_sampling,
    )
    blocking_period_issues = tuple(
        issue for issue in period_preflight.issues if issue.severity == "error"
    )
    if blocking_period_issues:
        details = "; ".join(issue.message for issue in blocking_period_issues)
        raise ValueError(f"reduced-OD time-period preflight failed: {details}")
    directory = Path(output_directory).resolve()
    reusable_timetable: TimetableIndex | None = None
    reusable_early_fingerprints: dict[str, str] = {}
    if cache_policy == "read_only":
        loaded = load_reduced_od_artifacts(
            configuration=configuration,
            artifact_directory=directory,
            expected_departure_sampling_fingerprint=_departure_sampling_identity(
                inputs
            ),
        )
        return _prepared_summary(loaded, ())
    if cache_policy == "reuse_or_build":
        try:
            loaded = load_reduced_od_artifacts(
                configuration=configuration,
                artifact_directory=directory,
                expected_departure_sampling_fingerprint=(
                    _departure_sampling_identity(inputs)
                ),
            )
        except ReducedODArtifactStoreError:
            try:
                early_paths = {
                    phase: _phase_path(directory, phase)
                    for phase in (
                        "physical_stops",
                        "service_periods_route_patterns",
                        "timetable_index",
                    )
                }
                physical, physical_manifest = load_reduced_od_phase_artifact(
                    early_paths["physical_stops"],
                    expected_phase="physical_stops",
                    expected_configuration_fingerprint=(
                        reduced_od_phase_configuration_fingerprint(
                            configuration, "physical_stops"
                        )
                    ),
                    expected_upstream_fingerprints={},
                )
                reusable_early_fingerprints["physical_stops"] = str(
                    physical_manifest["content_fingerprint"]
                )
                _, service_manifest = load_reduced_od_phase_artifact(
                    early_paths["service_periods_route_patterns"],
                    expected_phase="service_periods_route_patterns",
                    expected_configuration_fingerprint=(
                        reduced_od_phase_configuration_fingerprint(
                            configuration, "service_periods_route_patterns"
                        )
                    ),
                    expected_upstream_fingerprints={
                        "physical_stops": reusable_early_fingerprints["physical_stops"]
                    },
                )
                reusable_early_fingerprints["service_periods_route_patterns"] = str(
                    service_manifest["content_fingerprint"]
                )
                timetable_payload, timetable_manifest = load_reduced_od_phase_artifact(
                    early_paths["timetable_index"],
                    expected_phase="timetable_index",
                    expected_configuration_fingerprint=(
                        reduced_od_phase_configuration_fingerprint(
                            configuration, "timetable_index"
                        )
                    ),
                    expected_upstream_fingerprints={
                        "service_periods_route_patterns": (
                            reusable_early_fingerprints[
                                "service_periods_route_patterns"
                            ]
                        )
                    },
                )
                if not isinstance(timetable_payload, TimetableIndex):
                    raise ReducedODArtifactStoreError(
                        "persisted timetable_index has an invalid type."
                    )
                reusable_timetable = timetable_payload
                reusable_early_fingerprints["timetable_index"] = str(
                    timetable_manifest["content_fingerprint"]
                )
                del physical
            except ReducedODArtifactStoreError:
                reusable_timetable = None
                reusable_early_fingerprints.clear()
        else:
            production_changed = dict(loaded.production_inputs) != dict(
                inputs.production_inputs
            )
            attraction_changed = dict(loaded.destination_attractiveness) != dict(
                inputs.destination_attractiveness
            )
            rebuilt_phases: set[str] = set()
            if production_changed or attraction_changed:
                local_paths = {phase: _phase_path(directory, phase) for phase in PHASES}

                def republish(
                    phase: str,
                    payload: object,
                    upstream: Mapping[str, str],
                    dimensions: Mapping[str, int] = {},
                ) -> str:
                    rebuilt_phases.add(phase)
                    return save_reduced_od_phase_artifact(
                        local_paths[phase],
                        phase=phase,
                        payload=payload,
                        configuration_fingerprint=(
                            reduced_od_phase_configuration_fingerprint(
                                configuration, phase
                            )
                        ),
                        upstream_fingerprints=upstream,
                        dimensions=dimensions,
                    )

                updated_fingerprints = dict(loaded.fingerprints)
                if production_changed:
                    updated_fingerprints["production_inputs"] = republish(
                        "production_inputs",
                        dict(inputs.production_inputs),
                        {"journey_choices": updated_fingerprints["journey_choices"]},
                    )
                if attraction_changed:
                    updated_fingerprints["destination_attractiveness"] = republish(
                        "destination_attractiveness",
                        dict(inputs.destination_attractiveness),
                        {"journey_choices": updated_fingerprints["journey_choices"]},
                    )
                retained_productions, _ = _retained_productions(
                    loaded.measurement_response, inputs.production_inputs
                )
                features = build_conditional_gravity_features(
                    response=loaded.measurement_response,
                    journey_choices=loaded.journey_choices,
                    productions=retained_productions,
                    destination_attractiveness=dict(inputs.destination_attractiveness),
                )
                updated_fingerprints["conditional_gravity_features"] = republish(
                    "conditional_gravity_features",
                    features,
                    {
                        "measurement_response": updated_fingerprints[
                            "measurement_response"
                        ],
                        "production_inputs": updated_fingerprints["production_inputs"],
                        "destination_attractiveness": updated_fingerprints[
                            "destination_attractiveness"
                        ],
                    },
                    {
                        "origin_time_groups": features.number_of_origin_time_groups,
                        "free_cells": features.number_of_cells,
                    },
                )
                rebuilt_phases.add("problem_manifest")
                loaded = load_reduced_od_artifacts(
                    configuration=configuration,
                    artifact_directory=directory,
                    expected_departure_sampling_fingerprint=(
                        _departure_sampling_identity(inputs)
                    ),
                )
            items = tuple(
                {
                    "phase": phase,
                    "status": (
                        "rebuilt"
                        if phase in rebuilt_phases
                        else (
                            "rebound_model_identity"
                            if phase in {"configuration", "problem_manifest"}
                            else "reused"
                        )
                    ),
                    "seconds": 0.0,
                    "peak_rss_bytes": _peak_rss_bytes(),
                    "fingerprint": loaded.fingerprints[phase],
                }
                for phase in PHASES
            )
            if progress is not None:
                for item in items:
                    progress(item)
            return _prepared_summary(loaded, items)

    diagnostics: list[Mapping[str, object]] = []
    fingerprints: dict[str, str] = {}
    paths = {phase: _phase_path(directory, phase) for phase in PHASES}

    def publish(
        phase: str,
        payload: object,
        *,
        upstream: Mapping[str, str] = {},
        dimensions: Mapping[str, int] = {},
    ) -> str:
        started = time.perf_counter()
        fingerprint = save_reduced_od_phase_artifact(
            paths[phase],
            phase=phase,
            payload=payload,
            configuration_fingerprint=reduced_od_phase_configuration_fingerprint(
                configuration, phase
            ),
            upstream_fingerprints=upstream,
            dimensions=dimensions,
            semantic_conventions={
                "production_mode": configuration.productions.mode,
                "production_semantics": configuration.productions.semantics,
                "likelihood": configuration.model.likelihood,
                "time_bin_boundary": configuration.journeys.time_bin_membership,
                "output_geography": configuration.outputs.spatial_level,
                "journey_period_semantics": "desired-departure-v2",
            },
        )
        item = {
            "phase": phase,
            "status": "built",
            "seconds": time.perf_counter() - started,
            "peak_rss_bytes": _peak_rss_bytes(),
            "fingerprint": fingerprint,
        }
        diagnostics.append(item)
        if progress is not None:
            progress(item)
        fingerprints[phase] = fingerprint
        return fingerprint

    publish("configuration", configuration)
    if reusable_timetable is None:
        timetable = prepare_reduced_od_timetable(
            scenario,
            configuration_fingerprint=reduced_od_phase_configuration_fingerprint(
                configuration, "timetable_index"
            ),
            physical_stop_mapping=inputs.physical_stop_mapping,
            mapping_policy=configuration.stops.mapping_policy,
        )
        publish("physical_stops", timetable.physical_stops)
        publish(
            "service_periods_route_patterns",
            (timetable.service_periods, timetable.route_patterns),
            upstream={"physical_stops": fingerprints["physical_stops"]},
        )
        publish(
            "timetable_index",
            timetable,
            upstream={
                "service_periods_route_patterns": fingerprints[
                    "service_periods_route_patterns"
                ]
            },
            dimensions={
                "stops": len(timetable.stop_ids),
                "trips": len(timetable.trip_ids),
            },
        )
    else:
        timetable = reusable_timetable
        fingerprints.update(reusable_early_fingerprints)
        for phase in (
            "physical_stops",
            "service_periods_route_patterns",
            "timetable_index",
        ):
            item = {
                "phase": phase,
                "status": "reused",
                "seconds": 0.0,
                "peak_rss_bytes": _peak_rss_bytes(),
                "fingerprint": fingerprints[phase],
            }
            diagnostics.append(item)
            if progress is not None:
                progress(item)
    policy = JourneyChoicePolicy(
        maximum_alternatives_per_cell=configuration.journeys.maximum_alternatives_per_cell
    )
    choice_parts: list[JourneyChoiceResult] = []
    sampling_config = inputs.departure_time_sampling
    samples: tuple[DesiredDepartureSample, ...] = ()
    precomputed_sample_choices: dict[str, JourneyChoiceResult] = {}
    adaptive_diagnostics: list[DepartureQuadratureDiagnostics] = []
    queries: tuple[tuple[DesiredDepartureSample | None, str, int], ...]
    if sampling_config is None:
        queries = tuple(
            (None, origin, int(departure))
            for origin in sorted(inputs.departure_seconds_by_origin)
            for departure in sorted(set(inputs.departure_seconds_by_origin[origin]))
        )
        departure_sampling_payload: dict[str, object] = {
            "mode": "legacy_representative_departure",
            "fingerprint": _departure_sampling_identity(inputs),
            "queries": tuple((origin, departure) for _, origin, departure in queries),
            "warning": "one representative departure is low-fidelity for broad periods",
        }
    else:
        if not inputs.time_periods:
            raise ValueError("desired-departure sampling requires time_periods.")
        sampling_groups, cell_status = _sampling_support(inputs)
        if sampling_config.strategy in {"uniform_midpoint", "fixed_count"}:
            samples = generate_uniform_midpoint_samples(
                origin_period_groups=sampling_groups,
                time_periods=inputs.time_periods,
                config=sampling_config,
                progress=progress,
            )
        elif sampling_config.strategy == "fixed_time_step":
            samples = generate_fixed_time_step_samples(
                origin_period_groups=sampling_groups,
                time_periods=inputs.time_periods,
                config=sampling_config,
                progress=progress,
            )
        else:
            periods = {period.period_id: period for period in inputs.time_periods}
            adaptive_samples: list[DesiredDepartureSample] = []
            adaptive_started = time.perf_counter()
            requested_mode = sampling_config.comparison_mode
            effective_mode = (
                "integral_response"
                if requested_mode == "integral_response"
                else "exact_service_identity"
                if requested_mode in {"service_signature", "exact_service_identity"}
                else "route_pattern_signature"
                if requested_mode == "route_pattern_signature"
                else "measurement_support"
                if requested_mode == "measurement_support"
                else "aggregate_response"
            )
            resolved_for_comparison = _resolve_measurements(
                timetable, measurements
            )
            measurement_lookup: dict[
                tuple[str, str, str, int], tuple[int, ...]
            ] = {}
            mutable_measurement_lookup: dict[
                tuple[str, str, str, int], list[int]
            ] = {}
            for measurement in resolved_for_comparison:
                mutable_measurement_lookup.setdefault(
                    measurement.event_key, []
                ).append(measurement.row_index)
            measurement_lookup = {
                key: tuple(rows) for key, rows in mutable_measurement_lookup.items()
            }
            if progress is not None and requested_mode != effective_mode:
                progress(
                    {
                        "phase": "adaptive_departure_quadrature",
                        "status": "warning",
                        "requested_comparison_mode": requested_mode,
                        "effective_comparison_mode": effective_mode,
                        "warning": (
                            "The integrated preparation stage compares responses "
                            "visible to the active measurement system."
                        ),
                    }
                )
            for completed_group, (origin, period_id) in enumerate(
                sampling_groups, start=1
            ):
                period = periods[period_id]
                choice_by_time: dict[float, JourneyChoiceResult] = {}
                signature_index: dict[str, int] = {}
                coordinate_index: dict[tuple[str, int], int] = {}
                origin_physical_index = timetable.physical_stop_ids.index(origin)
                departure_values = timetable.array("departure_seconds")
                departure_physical = timetable.array(
                    "stop_time_physical_stop_index"
                )
                service_boundaries = tuple(
                    sorted(
                        {
                            boundary
                            for departure_value, physical_index in zip(
                                departure_values,
                                departure_physical,
                                strict=True,
                            )
                            if int(physical_index) == origin_physical_index
                            for boundary in (
                                float(departure_value),
                                float(departure_value)
                                - configuration.journeys.maximum_waiting_seconds,
                            )
                            if period.start_seconds
                            < boundary
                            < period.end_seconds
                        }
                    )
                )

                def evaluate_signature(seconds: float) -> SparseWeightedResponse | None:
                    departure = int(math.ceil(seconds))
                    query_choices = build_journey_choices(
                        timetable,
                        run_raptor_query(
                            timetable,
                            RaptorQuery(
                                origin,
                                departure,
                                configuration.journeys.maximum_transfers,
                                configuration.journeys.maximum_waiting_seconds,
                                configuration.journeys.maximum_journey_seconds,
                            ),
                            footpaths=inputs.footpaths,
                        ),
                        policy=policy,
                        time_periods=inputs.time_periods,
                        route_pattern_initial_weights=(
                            inputs.route_pattern_initial_weights
                        ),
                        desired_departure_time_period_id=period_id,
                    )
                    choice_by_time[seconds] = query_choices
                    if not query_choices.choice_sets:
                        return None
                    if effective_mode in {
                        "aggregate_response",
                        "measurement_support",
                        "integral_response",
                    }:
                        return _journey_measurement_comparison_response(
                            query_choices,
                            measurement_lookup=measurement_lookup,
                            coordinate_index=coordinate_index,
                            support_only=effective_mode == "measurement_support",
                        )
                    signature = (
                        _journey_route_pattern_signature(query_choices)
                        if effective_mode == "route_pattern_signature"
                        else _journey_service_signature(query_choices)
                    )
                    index = signature_index.setdefault(signature, len(signature_index))
                    return SparseWeightedResponse((index,), (1.0,))

                group_key = ResponseCellKey(origin, "__origin_period_group__", period_id)
                quadrature = integrate_adaptive_departure_response(
                    cell_key=group_key,
                    start_seconds=period.start_seconds,
                    end_seconds=period.end_seconds,
                    evaluator=evaluate_signature,
                    config=sampling_config,
                    progress=progress,
                    effective_comparison_mode=effective_mode,
                    service_boundary_seconds=service_boundaries,
                )
                adaptive_diagnostics.append(quadrature.diagnostics)
                for response_index, weighted in enumerate(quadrature.responses):
                    representative = weighted.representative_seconds[0]
                    payload = [
                        "adaptive_service_aware",
                        sampling_config.fingerprint,
                        origin,
                        period_id,
                        response_index,
                        representative,
                        weighted.weight,
                    ]
                    sample = DesiredDepartureSample(
                        hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
                        origin,
                        period_id,
                        representative,
                        weighted.weight,
                    )
                    adaptive_samples.append(sample)
                    precomputed_sample_choices[sample.sample_id] = (
                        choice_by_time[representative]
                        if weighted.feasible
                        else JourneyChoiceResult(
                            (),
                            JourneyChoiceDiagnostics(0, 0, 0, 0, 0, 0, 0, 0),
                        )
                    )
                if progress is not None:
                    elapsed = time.perf_counter() - adaptive_started
                    evaluations = sum(
                        item.routing_evaluations
                        for item in adaptive_diagnostics
                    )
                    remaining_estimate = (
                        elapsed
                        * (len(sampling_groups) - completed_group)
                        / completed_group
                    )
                    progress(
                        {
                            "phase": "adaptive_departure_quadrature_batch",
                            "status": (
                                "completed"
                                if completed_group == len(sampling_groups)
                                else "in_progress"
                            ),
                            "completed_origin_period_groups": completed_group,
                            "total_origin_period_groups": len(sampling_groups),
                            "routing_evaluations": evaluations,
                            "mean_evaluations_per_group": (
                                evaluations / completed_group
                            ),
                            "maximum_evaluations_per_group": max(
                                item.routing_evaluations
                                for item in adaptive_diagnostics
                            ),
                            "accepted_subintervals": sum(
                                item.accepted_subintervals
                                for item in adaptive_diagnostics
                            ),
                            "refined_subintervals": sum(
                                item.refined_subintervals
                                for item in adaptive_diagnostics
                            ),
                            "cache_hits": sum(
                                item.cache_hits
                                for item in adaptive_diagnostics
                            ),
                            "evaluation_budget": sum(
                                item.evaluation_budget
                                for item in adaptive_diagnostics
                            ),
                            "reserved_baseline_evaluations": sum(
                                item.reserved_baseline_evaluations
                                for item in adaptive_diagnostics
                            ),
                            "refinement_evaluations": sum(
                                item.refinement_evaluations
                                for item in adaptive_diagnostics
                            ),
                            "stable_interval_weight": sum(
                                item.stable_interval_weight
                                for item in adaptive_diagnostics
                            ),
                            "unresolved_interval_weight": sum(
                                item.unresolved_interval_weight
                                for item in adaptive_diagnostics
                            ),
                            "mean_stable_fraction": float(
                                np.mean(
                                    [
                                        item.stable_interval_weight
                                        for item in adaptive_diagnostics
                                    ]
                                )
                            ),
                            "mean_unresolved_fraction": float(
                                np.mean(
                                    [
                                        item.unresolved_interval_weight
                                        for item in adaptive_diagnostics
                                    ]
                                )
                            ),
                            "maximum_group_unresolved_fraction": max(
                                item.unresolved_interval_weight
                                for item in adaptive_diagnostics
                            ),
                            "fully_unresolved_group_count": sum(
                                item.unresolved_interval_weight
                                >= 1.0 - sampling_config.weight_tolerance
                                for item in adaptive_diagnostics
                            ),
                            "aggregate_estimated_absolute_integration_error": sum(
                                item.estimated_absolute_integration_error
                                for item in adaptive_diagnostics
                            ),
                            "mean_estimated_relative_integration_error": float(
                                np.mean(
                                    [
                                        item.estimated_relative_response_error
                                        for item in adaptive_diagnostics
                                    ]
                                )
                            ),
                            "maximum_estimated_relative_integration_error": max(
                                item.estimated_relative_response_error
                                for item in adaptive_diagnostics
                            ),
                            "groups_meeting_global_target": sum(
                                item.global_target_achieved
                                for item in adaptive_diagnostics
                            ),
                            "groups_stopped_by_budget": sum(
                                item.sample_cap_reached
                                for item in adaptive_diagnostics
                            ),
                            "requested_comparison_mode": (
                                sampling_config.comparison_mode
                            ),
                            "effective_comparison_mode": effective_mode,
                            "elapsed_seconds": elapsed,
                            "throughput_groups_per_second": (
                                completed_group / max(elapsed, 1.0e-12)
                            ),
                            "estimated_remaining_seconds": remaining_estimate,
                            "estimated_remaining_seconds_range": (
                                0.0
                                if remaining_estimate == 0.0
                                else 0.5 * remaining_estimate,
                                0.0
                                if remaining_estimate == 0.0
                                else 2.0 * remaining_estimate,
                            ),
                            "eta_confidence": (
                                "high"
                                if completed_group == len(sampling_groups)
                                else "low"
                                if completed_group < 3
                                else "medium"
                            ),
                            "current_infeasible_fraction": float(
                                np.mean(
                                    [
                                        item.infeasible_time_fraction
                                        for item in adaptive_diagnostics
                                    ]
                                )
                            ),
                            "sample_cap_count": sum(
                                item.sample_cap_reached
                                for item in adaptive_diagnostics
                            ),
                            "unresolved_group_count": sum(
                                item.unresolved_interval_weight > 0.0
                                for item in adaptive_diagnostics
                            ),
                            "peak_rss_bytes": _peak_rss_bytes(),
                        }
                    )
            samples = tuple(sorted(adaptive_samples, key=lambda item: (
                item.origin_physical_stop_id,
                item.time_period_id,
                item.desired_departure_seconds,
                item.sample_id,
            )))
            validate_sample_weights(
                samples, tolerance=sampling_config.weight_tolerance
            )
        queries = tuple(
            (
                sample,
                sample.origin_physical_stop_id,
                int(math.ceil(sample.desired_departure_seconds)),
            )
            for sample in samples
        )
        departure_sampling_payload = {
            "mode": "desired_departure_sampling",
            "fingerprint": _departure_sampling_identity(inputs),
            "config": sampling_config,
            "samples": samples,
            "origin_period_groups": sampling_groups,
            "adaptive_diagnostics": tuple(
                asdict(item) for item in adaptive_diagnostics
            ),
        }
    publish(
        "departure_time_samples",
        departure_sampling_payload,
        upstream={"timetable_index": fingerprints["timetable_index"]},
        dimensions={"desired_departure_samples": len(queries)},
    )
    journey_started = time.perf_counter()
    last_progress = journey_started
    recent_query_seconds: list[float] = []
    sampled_choice_sets: list[JourneyChoiceSet] = []
    sampled_cells: list[SampledJourneyCellDiagnostics] = []
    sampled_status_counts = {
        "free": 0,
        "fixed_zero": 0,
        "fixed_positive": 0,
        "retained": 0,
        "excluded": 0,
        "frozen": 0,
        "unexpected_status_changes": 0,
    }
    group_samples: list[DesiredDepartureSample] = []
    group_choices: list[JourneyChoiceResult] = []
    sampled_first_boarding_periods: dict[str, int] = {}
    sampled_later_boarding_count = 0
    sampled_multi_boarding_period_sets = 0
    sampled_cross_period_alternatives = 0
    sampled_maximum_cross_period_wait = 0
    sampled_legacy_period_sets = 0
    candidate_cells = set(cell_status) if sampling_config is not None else set()
    for query_index, (query_sample, origin, departure) in enumerate(queries, start=1):
        query_started = time.perf_counter()
        try:
            if query_sample is not None and query_sample.sample_id in precomputed_sample_choices:
                query_choices = precomputed_sample_choices[query_sample.sample_id]
            else:
                raptor = run_raptor_query(
                    timetable,
                    RaptorQuery(
                        origin,
                        departure,
                        configuration.journeys.maximum_transfers,
                        configuration.journeys.maximum_waiting_seconds,
                        configuration.journeys.maximum_journey_seconds,
                    ),
                    footpaths=inputs.footpaths,
                )
                query_choices = build_journey_choices(
                    timetable,
                    raptor,
                    policy=policy,
                    time_periods=inputs.time_periods,
                    route_pattern_initial_weights=(inputs.route_pattern_initial_weights),
                    desired_departure_time_period_id=(
                        query_sample.time_period_id
                        if query_sample is not None
                        else None
                    ),
                )
        except Exception as error:
            if progress is not None:
                progress(
                    {
                        "phase": "journey_choices",
                        "status": "failed",
                        "completed_queries": query_index - 1,
                        "total_queries": len(queries),
                        "current_origin": origin,
                        "current_departure": departure,
                        "elapsed_seconds": time.perf_counter() - journey_started,
                        "peak_rss_bytes": _peak_rss_bytes(),
                        "error": str(error),
                    }
                )
            raise
        if query_sample is None:
            choice_parts.append(query_choices)
        else:
            sample_has_later_boarding = False
            for choice in query_choices.choice_sets:
                boarding_periods = {
                    alternative.first_boarding_time_period_id
                    for alternative in choice.alternatives
                }
                sampled_multi_boarding_period_sets += int(len(boarding_periods) > 1)
                sampled_legacy_period_sets += int(
                    any(
                        alternative.desired_departure_time_period_id is None
                        for alternative in choice.alternatives
                    )
                )
                for alternative in choice.alternatives:
                    boarding_period = alternative.first_boarding_time_period_id
                    sampled_first_boarding_periods[boarding_period] = (
                        sampled_first_boarding_periods.get(boarding_period, 0) + 1
                    )
                    cross_period = boarding_period != query_sample.time_period_id
                    sample_has_later_boarding = sample_has_later_boarding or cross_period
                    sampled_cross_period_alternatives += int(cross_period)
                    if cross_period:
                        sampled_maximum_cross_period_wait = max(
                            sampled_maximum_cross_period_wait,
                            alternative.wait_seconds,
                        )
            sampled_later_boarding_count += int(sample_has_later_boarding)
            group_samples.append(query_sample)
            group_choices.append(query_choices)
            next_sample = (
                None if query_index == len(queries) else queries[query_index][0]
            )
            group_complete = next_sample is None or (
                next_sample.origin_physical_stop_id,
                next_sample.time_period_id,
            ) != (
                query_sample.origin_physical_stop_id,
                query_sample.time_period_id,
            )
            if group_complete:
                assert sampling_config is not None
                group_candidate_cells = tuple(
                    sorted(
                        cell
                        for cell in candidate_cells
                        if cell.origin_physical_stop_id
                        == query_sample.origin_physical_stop_id
                        and cell.origin_time_period_id == query_sample.time_period_id
                    )
                )
                merged_group = merge_sampled_journey_choices(
                    samples=group_samples,
                    sample_choices=group_choices,
                    config=sampling_config,
                    candidate_cells=group_candidate_cells,
                    cell_status={
                        cell: cell_status[cell] for cell in group_candidate_cells
                    },
                )
                sampled_choice_sets.extend(merged_group.journey_choices.choice_sets)
                sampled_cells.extend(merged_group.cells)
                for sampled_cell in merged_group.cells:
                    sampled_status_counts[sampled_cell.cell_status] += 1
                    if sampled_cell.cell_status == "free":
                        if sampled_cell.classification in {"normal", "warning"}:
                            sampled_status_counts["retained"] += 1
                        elif sampled_cell.classification == "excluded_low_feasibility":
                            sampled_status_counts["excluded"] += 1
                        else:
                            sampled_status_counts["frozen"] += 1
                    sampled_status_counts["unexpected_status_changes"] += int(
                        sampled_cell.unexpected_status_change
                    )
                group_samples.clear()
                group_choices.clear()
        now = time.perf_counter()
        recent_query_seconds.append(now - query_started)
        should_emit = (
            query_index == len(queries)
            or query_index % journey_progress_interval_queries == 0
            or now - last_progress >= journey_progress_interval_seconds
        )
        if progress is not None and should_emit:
            recent = float(np.mean(recent_query_seconds[-10:]))
            progress(
                {
                    "phase": "journey_choices",
                    "status": (
                        "completed" if query_index == len(queries) else "in_progress"
                    ),
                    "sampling_level": (
                        None
                        if sampling_config is None or query_sample is None
                        else sampling_config.count_for_period(
                            query_sample.time_period_id
                        )
                    ),
                    "completed_queries": query_index,
                    "total_queries": len(queries),
                    "current_origin": origin,
                    "current_departure": departure,
                    "current_period": (
                        None if query_sample is None else query_sample.time_period_id
                    ),
                    "sampled_merge_counts": dict(sampled_status_counts),
                    "elapsed_seconds": now - journey_started,
                    "recent_query_seconds": recent,
                    "predicted_remaining_seconds": (
                        (len(queries) - query_index) * recent
                    ),
                    "peak_rss_bytes": _peak_rss_bytes(),
                }
            )
            last_progress = now
    departure_sampling_cells: tuple[SampledJourneyCellDiagnostics, ...] = ()
    if sampling_config is None:
        choices = _merge_choices(choice_parts)
    else:
        sampled_choice_sets.sort(
            key=lambda item: (
                item.origin_physical_stop_id,
                item.destination_physical_stop_id,
                item.origin_time_period_id,
            )
        )
        retained = sum(len(item.alternatives) for item in sampled_choice_sets)
        choices = JourneyChoiceResult(
            choice_sets=tuple(sampled_choice_sets),
            diagnostics=JourneyChoiceDiagnostics(
                feasible_destinations=len(sampled_choice_sets),
                candidate_alternatives=retained,
                retained_alternatives=retained,
                pruned_alternatives=0,
                choice_cells=len(sampled_choice_sets),
                maximum_candidates_in_cell=max(
                    (len(item.alternatives) for item in sampled_choice_sets),
                    default=0,
                ),
                route_initialized_alternatives=0,
                estimated_payload_bytes=0,
            ),
        )
        departure_sampling_cells = tuple(
            sorted(
                sampled_cells,
                key=lambda item: (
                    item.cell_key.origin_physical_stop_id,
                    item.cell_key.destination_physical_stop_id,
                    item.cell_key.origin_time_period_id,
                ),
            )
        )
        averaged_fingerprint = hashlib.sha256(
            canonical_json(
                {
                    "choices": choices.fingerprint,
                    "sampling": _departure_sampling_identity(inputs),
                    "cells": [asdict(item) for item in departure_sampling_cells],
                }
            ).encode("utf-8")
        ).hexdigest()
        departure_sampling_payload = {
            **departure_sampling_payload,
            "cells": departure_sampling_cells,
            "averaged_journey_fingerprint": averaged_fingerprint,
            "cell_status_counts": {
                status: sum(
                    item.cell_status == status for item in departure_sampling_cells
                )
                for status in ("free", "fixed_zero", "fixed_positive")
            },
            "timetable_feasible_fixed_zero": sum(
                item.timetable_feasible_fixed_zero for item in departure_sampling_cells
            ),
            "fixed_positive_assignment_failed": sum(
                item.fixed_positive_assignment_failed
                for item in departure_sampling_cells
            ),
            "unexpected_status_changes": sum(
                item.unexpected_status_change for item in departure_sampling_cells
            ),
            "period_semantics_version": 2,
            "period_semantics_diagnostics": {
                "first_boarding_period_distribution": tuple(
                    sorted(sampled_first_boarding_periods.items())
                ),
                "samples_with_later_first_boarding": sampled_later_boarding_count,
                "multi_first_boarding_period_choice_sets": (
                    sampled_multi_boarding_period_sets
                ),
                "cross_period_alternatives": sampled_cross_period_alternatives,
                "maximum_cross_period_wait_seconds": (
                    sampled_maximum_cross_period_wait
                ),
                "legacy_period_semantics_choice_sets": sampled_legacy_period_sets,
            },
        }
        publish(
            "departure_time_samples",
            departure_sampling_payload,
            upstream={"timetable_index": fingerprints["timetable_index"]},
            dimensions={
                "desired_departure_samples": len(samples),
                "sampled_cells": len(departure_sampling_cells),
            },
        )
    publish(
        "journey_choices",
        choices,
        upstream={
            "timetable_index": fingerprints["timetable_index"],
            "departure_time_samples": fingerprints["departure_time_samples"],
        },
        dimensions={"choice_cells": len(choices.choice_sets)},
    )
    fixed_demand = dict(inputs.fixed_demand or {})
    response = build_measurement_response(
        timetable=timetable,
        journey_choices=choices,
        measurements=measurements,
        configuration_fingerprint=reduced_od_phase_configuration_fingerprint(
            configuration, "measurement_response"
        ),
        fixed_demand=fixed_demand,
    )
    publish(
        "measurement_response",
        (response, fixed_demand),
        upstream={"journey_choices": fingerprints["journey_choices"]},
        dimensions={
            "measurements": response.number_of_measurements,
            "free_cells": response.number_of_free_cells,
        },
    )
    publish(
        "response_equivalence",
        response.equivalence,
        upstream={"measurement_response": fingerprints["measurement_response"]},
        dimensions={"response_classes": response.equivalence.number_of_classes},
    )
    productions = dict(inputs.production_inputs)
    retained_productions, production_coverage = _retained_productions(
        response, productions
    )
    publish(
        "production_inputs",
        productions,
        upstream={"journey_choices": fingerprints["journey_choices"]},
        dimensions={
            "declared_origin_time_groups": len(productions),
            "retained_free_origin_time_groups": len(retained_productions),
        },
    )
    attractiveness = dict(inputs.destination_attractiveness)
    publish(
        "destination_attractiveness",
        attractiveness,
        upstream={"journey_choices": fingerprints["journey_choices"]},
    )
    features = build_conditional_gravity_features(
        response=response,
        journey_choices=choices,
        productions=retained_productions,
        destination_attractiveness=attractiveness,
    )
    publish(
        "conditional_gravity_features",
        features,
        upstream={
            "measurement_response": fingerprints["measurement_response"],
            "production_inputs": fingerprints["production_inputs"],
            "destination_attractiveness": fingerprints["destination_attractiveness"],
        },
        dimensions={
            "origin_time_groups": features.number_of_origin_time_groups,
            "free_cells": features.number_of_cells,
        },
    )
    operator = build_reduced_response_operator(response)
    publish(
        "reduced_response_operator",
        operator,
        upstream={
            "measurement_response": fingerprints["measurement_response"],
            "response_equivalence": fingerprints["response_equivalence"],
        },
        dimensions={
            "response_nnz": int(operator.response_values.size),
            "response_classes": operator.number_of_response_classes,
        },
    )
    manifest_payload = {
        "artifact_fingerprints": dict(fingerprints),
        "production_mode": configuration.productions.mode,
        "production_semantics": configuration.productions.semantics,
        "likelihood": configuration.model.likelihood,
        "production_coverage": production_coverage,
    }
    publish("problem_manifest", manifest_payload, upstream=dict(fingerprints))
    loaded = load_reduced_od_artifacts(
        configuration=configuration, artifact_directory=directory
    )
    return _prepared_summary(loaded, tuple(diagnostics))


def load_reduced_od_artifacts(
    *,
    configuration: ReducedODConfig,
    artifact_directory: str | Path,
    expected_departure_sampling_fingerprint: str | None = None,
) -> LoadedReducedODArtifacts:
    """Load all phases in dependency order, stopping at the first mismatch."""
    directory = Path(artifact_directory).resolve()
    payloads: dict[str, object] = {}
    manifests: dict[str, Mapping[str, object]] = {}
    fingerprints: dict[str, str] = {}
    upstream_by_phase = {
        "configuration": {},
        "physical_stops": {},
        "service_periods_route_patterns": lambda: {
            "physical_stops": fingerprints["physical_stops"]
        },
        "timetable_index": lambda: {
            "service_periods_route_patterns": fingerprints[
                "service_periods_route_patterns"
            ]
        },
        "departure_time_samples": lambda: {
            "timetable_index": fingerprints["timetable_index"]
        },
        "journey_choices": lambda: {
            "timetable_index": fingerprints["timetable_index"],
            "departure_time_samples": fingerprints["departure_time_samples"],
        },
        "measurement_response": lambda: {
            "journey_choices": fingerprints["journey_choices"]
        },
        "response_equivalence": lambda: {
            "measurement_response": fingerprints["measurement_response"]
        },
        "production_inputs": lambda: {
            "journey_choices": fingerprints["journey_choices"]
        },
        "destination_attractiveness": lambda: {
            "journey_choices": fingerprints["journey_choices"]
        },
        "conditional_gravity_features": lambda: {
            "measurement_response": fingerprints["measurement_response"],
            "production_inputs": fingerprints["production_inputs"],
            "destination_attractiveness": fingerprints["destination_attractiveness"],
        },
        "reduced_response_operator": lambda: {
            "measurement_response": fingerprints["measurement_response"],
            "response_equivalence": fingerprints["response_equivalence"],
        },
        "problem_manifest": lambda: dict(fingerprints),
    }
    paths = {phase: _phase_path(directory, phase) for phase in PHASES}
    payloads["configuration"] = configuration
    fingerprints["configuration"] = configuration.fingerprint
    for phase in PHASES:
        if phase in {"configuration", "problem_manifest"}:
            continue
        expected = upstream_by_phase[phase]
        if callable(expected):
            expected_upstream = cast(Mapping[str, str], expected())
        else:
            expected_upstream = cast(Mapping[str, str], expected)
        payload, manifest = load_reduced_od_phase_artifact(
            paths[phase],
            expected_phase=phase,
            expected_configuration_fingerprint=(
                reduced_od_phase_configuration_fingerprint(configuration, phase)
            ),
            expected_upstream_fingerprints=expected_upstream,
        )
        payloads[phase] = payload
        manifests[phase] = manifest
        fingerprints[phase] = str(manifest["content_fingerprint"])
    departure_sampling = cast(Mapping[str, object], payloads["departure_time_samples"])
    if (
        expected_departure_sampling_fingerprint is not None
        and departure_sampling.get("fingerprint")
        != expected_departure_sampling_fingerprint
    ):
        raise ReducedODArtifactStoreError(
            "departure_time_samples fingerprint is incompatible with requested sampling."
        )
    fingerprints["problem_manifest"] = hashlib.sha256(
        canonical_json(
            {
                "configuration": configuration.fingerprint,
                "upstream": dict(fingerprints),
            }
        ).encode("utf-8")
    ).hexdigest()
    response_payload = cast(tuple[object, object], payloads["measurement_response"])
    response = cast(MeasurementResponseArtifact, response_payload[0])
    fixed_demand = cast(Mapping[ResponseCellKey, float], response_payload[1])
    return LoadedReducedODArtifacts(
        configuration=configuration,
        timetable=payloads["timetable_index"],  # type: ignore[arg-type]
        departure_time_samples=departure_sampling,
        departure_sampling_cells=cast(
            tuple[SampledJourneyCellDiagnostics, ...],
            departure_sampling.get("cells", ()),
        ),
        journey_choices=payloads["journey_choices"],  # type: ignore[arg-type]
        measurement_response=response,
        production_inputs=payloads["production_inputs"],  # type: ignore[arg-type]
        destination_attractiveness=payloads["destination_attractiveness"],  # type: ignore[arg-type]
        features=payloads["conditional_gravity_features"],  # type: ignore[arg-type]
        response_operator=payloads["reduced_response_operator"],  # type: ignore[arg-type]
        fixed_demand=fixed_demand,
        paths=paths,
        fingerprints=fingerprints,
        manifests=manifests,
    )


def build_minimal_gravity_problem(
    *,
    artifacts: LoadedReducedODArtifacts,
    specification: MinimalGravitySpecification,
    production_basis: np.ndarray | None = None,
    production_basis_labels: tuple[str, ...] | None = None,
    destination_attractiveness_basis: np.ndarray | None = None,
    destination_attractiveness_basis_labels: tuple[str, ...] | None = None,
) -> BuiltMinimalGravityProblem:
    """Bind persisted compact artifacts into J0; no assignment/reconstruction."""
    configuration = artifacts.configuration
    if specification.likelihood != configuration.model.likelihood:
        raise ValueError(
            "specification likelihood conflicts with persisted configuration."
        )
    if specification.production_mode != configuration.productions.mode:
        raise ValueError(
            "specification production mode conflicts with persisted configuration."
        )
    layout = MinimalGravityParameterLayout(specification)
    problem = MinimalGravityProblem(
        features=artifacts.features,
        parameter_layout=layout,
        response_operator=artifacts.response_operator,
        observations=artifacts.measurement_response.observed_values,
        production_basis=production_basis,
        production_basis_labels=production_basis_labels,
        destination_attractiveness_basis=destination_attractiveness_basis,
        destination_attractiveness_basis_labels=destination_attractiveness_basis_labels,
    )
    parameter_names = ["beta_time", "beta_transfer"]
    if specification.likelihood == "negative_binomial":
        parameter_names.append("dispersion")
    parameter_names.extend(
        production_basis_labels
        if production_basis_labels is not None
        else tuple(
            f"production_coefficient_{index}"
            for index in range(specification.production_basis_columns)
        )
    )
    parameter_names.extend(
        destination_attractiveness_basis_labels
        if destination_attractiveness_basis_labels is not None
        else tuple(
            f"destination_attractiveness_coefficient_{index}"
            for index in range(specification.destination_attractiveness_basis_columns)
        )
    )
    model_contract = ReducedODModelContract(
        problem_fingerprint=hashlib.sha256(
            canonical_json(dict(artifacts.fingerprints)).encode("utf-8")
        ).hexdigest(),
        model_name="J0",
        production_mode=specification.production_mode,
        likelihood=specification.likelihood,
        estimated_parameters=tuple(sorted(parameter_names)),
    )
    fixed_values = np.asarray(tuple(artifacts.fixed_demand.values()), dtype=float)
    retained = sum(
        path.stat().st_size
        for directory in artifacts.paths.values()
        for path in directory.iterdir()
        if path.is_file()
    )
    runtime = (
        artifacts.features.origin_time_group_index.nbytes
        + artifacts.features.journey_time_seconds.nbytes
        + artifacts.features.transfer_count.nbytes
        + artifacts.features.destination_attractiveness.nbytes
        + artifacts.response_operator.diagnostics.retained_bytes
        + artifacts.measurement_response.observed_values.nbytes
    )
    return BuiltMinimalGravityProblem(
        problem=problem,
        model_fingerprint=model_contract.fingerprint,
        artifact_fingerprints=artifacts.fingerprints,
        parameter_names=tuple(parameter_names),
        raw_parameter_dimension=layout.size,
        transformed_parameter_dimension=layout.size,
        measurement_count=artifacts.measurement_response.number_of_measurements,
        canonical_cell_count=artifacts.features.number_of_cells + len(fixed_values),
        free_cell_count=artifacts.features.number_of_cells,
        fixed_zero_cell_count=int(np.count_nonzero(fixed_values == 0.0)),
        fixed_positive_cell_count=int(np.count_nonzero(fixed_values > 0.0)),
        origin_time_group_count=artifacts.features.number_of_origin_time_groups,
        response_nonzeros=artifacts.response_operator.original_nnz,
        response_classes=artifacts.response_operator.number_of_response_classes,
        retained_artifact_bytes=retained,
        estimated_runtime_memory=int(runtime),
        production_mode=specification.production_mode,
        likelihood=specification.likelihood,
    )


def _prepared_summary(
    loaded: LoadedReducedODArtifacts,
    diagnostics: tuple[Mapping[str, object], ...],
) -> PreparedReducedODArtifacts:
    response = loaded.measurement_response
    retained = (
        response.retained_bytes + loaded.response_operator.diagnostics.retained_bytes
    )
    dimensions = {
        "measurements": response.number_of_measurements,
        "free_cells": response.number_of_free_cells,
        "fixed_cells": len(response.fixed_cell_keys),
        "response_nnz_before": response.nnz,
        "response_nnz_after": int(loaded.response_operator.response_values.size),
        "response_classes": loaded.response_operator.number_of_response_classes,
        "origin_time_groups": loaded.features.number_of_origin_time_groups,
        "desired_departure_samples": len(
            cast(
                Sequence[object],
                cast(Mapping[str, object], loaded.departure_time_samples).get(
                    "samples", ()
                ),
            )
        ),
        "sampling_normal_cells": sum(
            item.classification == "normal" for item in loaded.departure_sampling_cells
        ),
        "sampling_warning_cells": sum(
            item.classification == "warning" for item in loaded.departure_sampling_cells
        ),
        "sampling_excluded_cells": sum(
            item.classification == "excluded_low_feasibility"
            for item in loaded.departure_sampling_cells
        ),
        "sampling_frozen_cells": sum(
            item.classification == "frozen_no_feasible_sample"
            for item in loaded.departure_sampling_cells
        ),
        "sampling_free_cells": sum(
            item.cell_status == "free" for item in loaded.departure_sampling_cells
        ),
        "sampling_fixed_zero_cells": sum(
            item.cell_status == "fixed_zero" for item in loaded.departure_sampling_cells
        ),
        "sampling_fixed_positive_cells": sum(
            item.cell_status == "fixed_positive"
            for item in loaded.departure_sampling_cells
        ),
        "sampling_feasible_fixed_zero_cells": sum(
            item.timetable_feasible_fixed_zero
            for item in loaded.departure_sampling_cells
        ),
        "sampling_unexpected_status_changes": sum(
            item.unexpected_status_change for item in loaded.departure_sampling_cells
        ),
    }
    return PreparedReducedODArtifacts(
        directory=next(iter(loaded.paths.values())).parent,
        paths=loaded.paths,
        fingerprints=loaded.fingerprints,
        dimensions=dimensions,
        retained_bytes=retained,
        estimated_in_memory_bytes=retained,
        phase_diagnostics=diagnostics,
    )


def preflight_reduced_od_j0(
    *,
    configuration: ReducedODConfig,
    artifact_directory: str | Path,
    specification: MinimalGravitySpecification | None = None,
    production_basis: np.ndarray | None = None,
    destination_attractiveness_basis: np.ndarray | None = None,
) -> dict[str, object]:
    """Read-only JSON-ready preflight that stops at the first invalid phase."""
    directory = Path(artifact_directory).resolve()
    try:
        artifacts = load_reduced_od_artifacts(
            configuration=configuration, artifact_directory=directory
        )
        selected = specification or MinimalGravitySpecification(
            likelihood=configuration.model.likelihood,
            production_mode=configuration.productions.mode,
            production_basis_columns=(
                0 if production_basis is None else production_basis.shape[1]
            ),
        )
        built = build_minimal_gravity_problem(
            artifacts=artifacts,
            specification=selected,
            production_basis=production_basis,
            destination_attractiveness_basis=destination_attractiveness_basis,
        )
    except (ReducedODArtifactStoreError, ValueError) as error:
        text = str(error)
        missing = next((phase for phase in PHASES if phase in text), None)
        return {
            "compatible": False,
            "artifact_directory": str(directory),
            "missing_or_incompatible_phase": missing,
            "error": text,
            "rebuild": "prepare_reduced_od_artifacts(..., cache_policy='reuse_or_build')",
        }
    return {
        "compatible": True,
        "artifact_directory": str(directory),
        "paths": {key: str(value) for key, value in artifacts.paths.items()},
        **built.to_dict(),
    }
