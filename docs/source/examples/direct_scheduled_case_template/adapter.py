"""Case-owned assembly of the current direct-scheduled public APIs.

This file is deliberately a small, explicit adapter rather than a hidden
framework.  A private case may replace the feature construction and model
specification, but should keep the identity checks and stage boundaries.
"""

from __future__ import annotations

import json
import hashlib
import tomllib
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

import jax
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario, read_fixed_demand_csv
from public_transportation.inference.assignment_adapter import (
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.assignment_contract import (
    CanonicalMeasurement,
    CanonicalTimeInterval,
    build_canonical_assignment_index,
)
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
    build_compact_od_assignment_layout,
)
from public_transportation.inference.direct_scheduled_temporal_builder import (
    DirectScheduledActivationResult,
    activate_direct_scheduled_temporal_operator,
)
from public_transportation.inference.fixed_routing_sharded_builder import (
    ShardedConstructionConfig,
)
from public_transportation.inference.support_discovery_profile import (
    SupportDiscoveryProfileRecorder,
    build_support_discovery_profile,
    write_support_discovery_profile,
)
from public_transportation.inference.gravity import (
    GravityGradientStrategy,
    GravityFeatures,
    GravityLikelihood,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    gravity_value_and_gradient,
)
from public_transportation.inference.gravity.specification import (
    GravityComponentSpecification,
    GravityConstraint,
    GravityEffectScope,
    GravityLikelihoodSpecification,
    GravityParameterization,
    GravityRegularization,
    GravityRegularizationType,
)
from public_transportation.inference.od_parameter_layout import (
    ODParameterLayout,
    build_od_parameter_layout,
)
from public_transportation.inference.scheduled_reference_operator import (
    build_scheduled_reference_artifact_identity,
)
from public_transportation.measurement import build_mapping_spec_strict, read_measurements_csv
from public_transportation.preprocessing import (
    build_canonical_timetable_index,
    expansion_contract_fingerprint,
    generate_candidate_od_pairs,
    fingerprint_scenario,
    materialize_prior_demand_from_checkpoint,
    run_candidate_od_time_expansion,
    ScheduledFeasibilityContract,
    TimetableFeasibilityIndex,
)


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value") and type(value).__module__ == "enum":
        return value.value
    return value


ProgressCallback = Callable[[Mapping[str, object]], None]


def _progress(
    callback: ProgressCallback | None,
    *,
    phase: str,
    status: str,
    current_unit: str,
    **fields: object,
) -> None:
    """Emit a phase event without inventing work or ETA information."""
    if callback is None:
        return
    callback(
        {
            "phase": phase,
            "status": status,
            "current_unit": current_unit,
            **fields,
        }
    )


@dataclass(frozen=True, slots=True)
class CaseSettings:
    root: Path
    scenario: Path
    scenario_demand: Path | None
    measurements: Path
    fixed_demand: Path
    results: Path
    package_revision: str
    theta: float
    rho: float
    expected_evaluations: int
    construction_time_budget_seconds: float | None
    safety_margin_seconds: float
    od_universe_source: str
    od_universe_level: str
    od_pairs_file: Path | None
    include_same_stop: bool
    active_service_only: bool
    connectivity_policy: str
    prior_source: str
    prior_value: float
    prior_semantics: str
    prior_file: Path | None
    maximum_transfers: int
    maximum_initial_wait_seconds: int
    maximum_journey_seconds: int
    maximum_waiting_seconds: int
    chunk_size_pairs: int
    progress_interval_seconds: float
    maximum_temporary_bytes: int
    model: dict[str, object]

    @classmethod
    def load(cls, root: str | Path) -> "CaseSettings":
        case_root = Path(root).expanduser().resolve()
        with (case_root / "config/case.toml").open("rb") as stream:
            raw = tomllib.load(stream)
        model_path = case_root / "config/model.toml"
        with model_path.open("rb") as stream:
            model = tomllib.load(stream)

        def path(name: str) -> Path:
            value = raw.get(name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"config/case.toml requires non-empty {name!r}.")
            result = (case_root / value).resolve()
            return result

        demand_value = raw.get("scenario_demand")
        scenario_demand = (
            None
            if demand_value is None
            else (case_root / str(demand_value)).resolve()
        )

        theta = float(raw.get("theta", 1.0))
        package_revision = str(raw.get("package_revision", "")).strip()
        if not package_revision or package_revision.startswith("REPLACE_"):
            raise ValueError("config/case.toml package_revision must be the exact installed commit.")
        rho = float(raw.get("rho", 1.0))
        if not np.isfinite(theta) or theta <= 0 or not np.isfinite(rho) or rho <= 0:
            raise ValueError("theta and rho must be finite and positive.")
        expected = int(raw.get("expected_evaluations", 100))
        if expected <= 0:
            raise ValueError("expected_evaluations must be positive.")
        budget = float(raw.get("construction_time_budget_seconds", 0.0))
        margin = float(raw.get("safety_margin_seconds", 0.0))
        if budget < 0 or margin < 0:
            raise ValueError("construction budget and safety margin cannot be negative.")
        required_policy_fields = (
            "od_universe_source",
            "od_universe_level",
            "include_same_stop",
            "active_service_only",
            "connectivity_policy",
            "prior_source",
            "prior_value",
            "prior_semantics",
            "maximum_transfers",
            "maximum_initial_wait_seconds",
            "maximum_journey_seconds",
            "maximum_waiting_seconds",
            "chunk_size_pairs",
            "progress_interval_seconds",
            "maximum_temporary_bytes",
        )
        missing_policy_fields = [field for field in required_policy_fields if field not in raw]
        if missing_policy_fields:
            raise ValueError(
                "config/case.toml must explicitly define prior/OD policy fields: "
                f"{missing_policy_fields}"
            )
        od_universe_source = str(raw["od_universe_source"])
        od_universe_level = str(raw["od_universe_level"])
        od_pairs_file_value = raw.get("od_pairs_file")
        od_pairs_file = (
            None
            if od_pairs_file_value in (None, "")
            else (case_root / str(od_pairs_file_value)).resolve()
        )
        connectivity_policy = str(raw["connectivity_policy"])
        prior_source = str(raw["prior_source"])
        prior_semantics = str(raw["prior_semantics"])
        prior_value = float(raw["prior_value"])
        prior_file_value = raw.get("prior_file")
        prior_file = (
            None
            if prior_file_value in (None, "")
            else (case_root / str(prior_file_value)).resolve()
        )
        maximum_transfers = int(raw["maximum_transfers"])
        maximum_initial_wait_seconds = int(raw["maximum_initial_wait_seconds"])
        maximum_journey_seconds = int(raw["maximum_journey_seconds"])
        maximum_waiting_seconds = int(raw["maximum_waiting_seconds"])
        chunk_size_pairs = int(raw["chunk_size_pairs"])
        progress_interval_seconds = float(raw["progress_interval_seconds"])
        maximum_temporary_bytes = int(raw["maximum_temporary_bytes"])
        if od_universe_source not in {"file", "network_ordered_pairs"}:
            raise ValueError("od_universe_source must be 'file' or 'network_ordered_pairs'.")
        if od_universe_source == "file" and od_pairs_file is None:
            raise ValueError("od_pairs_file is required when od_universe_source='file'.")
        if od_universe_level not in {"stop", "physical_stop"}:
            raise ValueError("od_universe_level must be 'stop' or 'physical_stop'.")
        if not isinstance(raw["include_same_stop"], bool) or not isinstance(
            raw["active_service_only"], bool
        ):
            raise ValueError("include_same_stop and active_service_only must be TOML booleans.")
        if connectivity_policy not in {"none", "directed_reachable"}:
            raise ValueError("connectivity_policy must be 'none' or 'directed_reachable'.")
        if prior_source not in {
            "all_ones",
            "external_file",
            "distance_decay",
            "travel_time_decay",
            "gravity_seed",
            "destination_attractiveness_seed",
        }:
            raise ValueError(f"unsupported prior_source {prior_source!r}.")
        if prior_source == "all_ones" and (not np.isfinite(prior_value) or prior_value <= 0.0):
            raise ValueError("prior_value must be finite and positive.")
        if maximum_transfers < 0 or maximum_initial_wait_seconds < 0:
            raise ValueError("transfer and initial-wait limits cannot be negative.")
        if maximum_journey_seconds <= 0 or maximum_waiting_seconds < 0:
            raise ValueError("journey time must be positive and waiting time non-negative.")
        if chunk_size_pairs <= 0 or progress_interval_seconds <= 0 or maximum_temporary_bytes <= 0:
            raise ValueError("expansion chunk, progress interval, and temporary-byte limit must be positive.")
        return cls(
            root=case_root,
            scenario=path("scenario"),
            scenario_demand=scenario_demand,
            measurements=path("measurements"),
            fixed_demand=path("fixed_demand"),
            results=path("results"),
            package_revision=package_revision,
            theta=theta,
            rho=rho,
            expected_evaluations=expected,
            construction_time_budget_seconds=None if budget == 0 else budget,
            safety_margin_seconds=margin,
            od_universe_source=od_universe_source,
            od_universe_level=od_universe_level,
            od_pairs_file=od_pairs_file,
            include_same_stop=bool(raw["include_same_stop"]),
            active_service_only=bool(raw["active_service_only"]),
            connectivity_policy=connectivity_policy,
            prior_source=prior_source,
            prior_value=prior_value,
            prior_semantics=prior_semantics,
            prior_file=prior_file,
            maximum_transfers=maximum_transfers,
            maximum_initial_wait_seconds=maximum_initial_wait_seconds,
            maximum_journey_seconds=maximum_journey_seconds,
            maximum_waiting_seconds=maximum_waiting_seconds,
            chunk_size_pairs=chunk_size_pairs,
            progress_interval_seconds=progress_interval_seconds,
            maximum_temporary_bytes=maximum_temporary_bytes,
            model=model,
        )


@dataclass(frozen=True, slots=True)
class CaseContext:
    settings: CaseSettings
    fixed_demand_path: Path
    fixed_demand_source: str
    fixed_demand_sha256: str
    scenario: Scenario
    timetable_index: Any
    assignment: Any
    id_manager: AssignmentIDManager
    measurements: Any
    mapping: Any
    parameter_layout: ODParameterLayout
    compact_layout: CompactODAssignmentLayout
    canonical_index: Any
    features: GravityFeatures
    identity: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_fixed_demand(settings: CaseSettings) -> tuple[Path, str, str]:
    """Resolve and fingerprint the fixed-demand file used by the context."""
    generated = (settings.results / "structural_zeros/fixed_demand.csv").resolve()
    if generated.is_file():
        path = generated
        source = "generated_structural_zeros"
    else:
        path = settings.fixed_demand.resolve()
        source = "case_config_fallback"
    return path, source, _sha256_file(path)


def _time_bin_fingerprint(scenario: Scenario) -> str:
    payload = [
        [
            str(item.bin_id),
            int(item.start.seconds_from_midnight),
            int(item.end.seconds_from_midnight),
        ]
        for item in scenario.time_bins
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
    """Persist a small stage audit without exposing a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _prior_expansion_audit(results_root: Path) -> dict[str, object]:
    path = results_root / "audit/prior_demand_generation.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read prior expansion audit {path}: {error}") from error
    return payload if isinstance(payload, dict) else {}


def bootstrap_prior_demand(
    root: str | Path,
    *,
    resume: bool = False,
    settings: CaseSettings | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Generate the scenario prior through a resumable expansion checkpoint.

    The final scenario-demand CSV is materialized only after the completed
    checkpoint, all chunk checksums, and all expansion fingerprints have been
    validated.  Large cases therefore retain durable progress without holding
    the full OD--time expansion in memory.
    """
    if settings is None:
        _progress(
            progress,
            phase="configuration_loading",
            status="started",
            current_unit="case.toml",
        )
        settings = CaseSettings.load(root)
        _progress(
            progress,
            phase="configuration_loading",
            status="completed",
            current_unit="case.toml",
        )
    else:
        _progress(
            progress,
            phase="configuration_loading",
            status="completed",
            current_unit="case.toml",
        )
    if settings.scenario_demand is None:
        raise ValueError("config/case.toml must define scenario_demand for prior generation.")

    _progress(
        progress,
        phase="scenario_loading",
        status="started",
        current_unit="scenario",
    )
    scenario = Scenario.from_folder(
        settings.scenario,
        strict=True,
        allow_missing_demand=True,
    )
    _progress(
        progress,
        phase="scenario_loading",
        status="completed",
        current_unit="scenario",
    )
    _progress(
        progress,
        phase="timetable_index_construction",
        status="started",
        current_unit="canonical_timetable_index",
    )
    timetable_index = build_canonical_timetable_index(scenario)
    _progress(
        progress,
        phase="timetable_index_construction",
        status="completed",
        current_unit="canonical_timetable_index",
    )
    _progress(
        progress,
        phase="candidate_universe_construction",
        status="started",
        current_unit="candidate_od_pairs",
    )
    universe = generate_candidate_od_pairs(
        scenario,
        source=settings.od_universe_source,
        level=settings.od_universe_level,
        include_same_stop=settings.include_same_stop,
        active_service_only=settings.active_service_only,
        connectivity_policy=settings.connectivity_policy,
        od_pairs_path=settings.od_pairs_file,
        timetable_index=timetable_index,
    )
    _progress(
        progress,
        phase="candidate_universe_construction",
        status="completed",
        current_unit="candidate_od_pairs",
        completed_units=universe.pair_count,
        total_units=universe.pair_count,
    )
    _progress(
        progress,
        phase="fingerprint_construction",
        status="started",
        current_unit="scenario_fingerprints",
    )
    scenario_fp = fingerprint_scenario(scenario)
    time_bins_fp = _time_bin_fingerprint(scenario)
    expansion_config: dict[str, object] = {
        "chunk_size_pairs": settings.chunk_size_pairs,
        "progress_interval_seconds": settings.progress_interval_seconds,
        "maximum_temporary_bytes": settings.maximum_temporary_bytes,
        "maximum_transfers": settings.maximum_transfers,
        "maximum_initial_wait_seconds": settings.maximum_initial_wait_seconds,
        "maximum_journey_seconds": settings.maximum_journey_seconds,
        "maximum_waiting_seconds": settings.maximum_waiting_seconds,
        "timetable_policy": "required",
        "package_revision": settings.package_revision,
        "approved_time_bins_fingerprint": time_bins_fp,
        "scenario_checksums": {
            "scenario_fingerprint": scenario_fp,
            "time_bins_fingerprint": time_bins_fp,
        },
    }
    feasibility_contract = ScheduledFeasibilityContract(
        maximum_transfers=settings.maximum_transfers,
        maximum_initial_wait_seconds=settings.maximum_initial_wait_seconds,
        maximum_journey_seconds=settings.maximum_journey_seconds,
        maximum_waiting_seconds=settings.maximum_waiting_seconds,
    )
    expansion_config["feasibility_contract_version"] = feasibility_contract.version
    expansion_config["feasibility_contract_fingerprint"] = feasibility_contract.fingerprint
    config_identity = dict(expansion_config)
    expansion_config["configuration_fingerprint"] = hashlib.sha256(
        json.dumps(config_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expansion_fp = expansion_contract_fingerprint(
        universe, scenario.time_bins, expansion_config
    )
    _progress(
        progress,
        phase="fingerprint_construction",
        status="completed",
        current_unit="expansion_fingerprint",
    )
    checkpoint = settings.results / "checkpoints/prior_demand" / expansion_fp

    if progress is not None:
        progress(
            {
                "phase": "candidate_universe",
                "status": "completed",
                "current_unit": "candidate_od_pairs",
                "completed_units": universe.pair_count,
                "total_units": universe.pair_count,
                "expansion_fingerprint": expansion_fp,
                "checkpoint_directory": str(checkpoint),
            }
        )
    expansion = run_candidate_od_time_expansion(
        universe,
        scenario.time_bins,
        scenario=scenario,
        configuration=expansion_config,
        checkpoint_directory=checkpoint,
        resume=resume,
        progress=progress,
        timetable_index=timetable_index,
    )
    materialized = materialize_prior_demand_from_checkpoint(
        expansion.checkpoint_directory,
        settings.scenario_demand,
        source=settings.prior_source,
        value=settings.prior_value,
        semantics=settings.prior_semantics,
        prior_file=settings.prior_file,
        scenario=scenario,
        package_revision=settings.package_revision,
        expansion_fingerprint=expansion.expansion_fingerprint,
        configuration_fingerprint=str(expansion_config["configuration_fingerprint"]),
        approved_time_bins=scenario.time_bins,
        approved_time_bins_fingerprint=time_bins_fp,
        scenario_fingerprint=scenario_fp,
        progress=progress,
    )
    audit = {
        "schema_version": 1,
        "package_revision": settings.package_revision,
        "prior_source": materialized.source,
        "prior_semantics": materialized.semantics,
        "prior_value": settings.prior_value,
        "od_universe": {
            "source": settings.od_universe_source,
            "level": settings.od_universe_level,
            "include_same_stop": settings.include_same_stop,
            "active_service_only": settings.active_service_only,
            "connectivity_policy": settings.connectivity_policy,
            "fingerprint": universe.fingerprint,
            "generated_pair_count": universe.pair_count + len(universe.exclusions),
            "retained_pair_count": universe.pair_count,
            "audit": universe.audit,
        },
        "time_bins": {
            "fingerprint": _time_bin_fingerprint(scenario),
            "count": len(scenario.time_bins),
        },
        "expansion": {
            "checkpoint_directory": str(expansion.checkpoint_directory),
            "configuration": expansion_config,
            "maximum_transfers": settings.maximum_transfers,
            "maximum_initial_wait_seconds": settings.maximum_initial_wait_seconds,
            "maximum_journey_seconds": settings.maximum_journey_seconds,
            "maximum_waiting_seconds": settings.maximum_waiting_seconds,
            "timetable_policy": "required",
            "fingerprint": expansion.expansion_fingerprint,
            "semantic_checksum": expansion.semantic_checksum,
            "retained_cell_count": expansion.retained_cells,
            "excluded_cell_count": expansion.excluded_cells,
            "total_chunks": expansion.total_chunks,
            "completed_chunks": expansion.completed_chunks,
            "checkpoint_reused": expansion.checkpoint_reused,
            "audit": {
                "status": expansion.status,
                "total_cells": expansion.total_cells,
                "next_chunk": expansion.next_chunk,
                "checkpoint_reused": expansion.checkpoint_reused,
            },
        },
        "prior_generation": materialized.audit,
        "output_file": str(settings.scenario_demand),
        "output_sha256": _sha256_file(settings.scenario_demand),
        "checkpoint_directory": str(checkpoint),
    }
    audit_path = settings.results / "audit/prior_demand_generation.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _canonical_index(scenario: Scenario, layout: ODParameterLayout, table: Any) -> Any:
    intervals = tuple(
        CanonicalTimeInterval(
            interval_id=item.bin_id,
            start_seconds=item.start.seconds_from_midnight,
            end_seconds=item.end.seconds_from_midnight,
        )
        for item in scenario.time_bins
    )
    latest = max(record.time.seconds_from_midnight for record in table.records)
    if latest >= intervals[-1].end_seconds:
        last = intervals[-1]
        intervals = intervals[:-1] + (
            CanonicalTimeInterval(
                interval_id=last.interval_id,
                start_seconds=last.start_seconds,
                end_seconds=latest + 1,
            ),
        )

    def interval_at(seconds: int) -> str:
        for interval in intervals:
            if interval.start_seconds <= seconds < interval.end_seconds:
                return interval.interval_id
        raise ValueError(f"measurement time {seconds} is outside scenario time bins.")

    measurements = tuple(
        CanonicalMeasurement(
            row_index=index,
            measurement_id="|".join(map(str, record.key())),
            event=record.measurement_type.value,
            location_id=record.stop_id,
            interval_id=interval_at(record.time.seconds_from_midnight),
        )
        for index, record in enumerate(table.records)
    )
    return build_canonical_assignment_index(
        parameter_layout=layout,
        time_intervals=intervals,
        measurements=measurements,
    )


def _features(
    scenario: Scenario,
    layout: ODParameterLayout,
    compact: CompactODAssignmentLayout,
    *,
    timetable_index: Any,
    feasibility_contract: ScheduledFeasibilityContract,
    results_root: Path,
    journey_time_scale: float,
) -> GravityFeatures:
    free_indices = np.asarray(layout.free_od_indices, dtype=np.int64)
    keys = [layout.od_keys[index] for index in free_indices]
    if scenario.timetable is None:
        raise ValueError("scheduled feature construction requires a scenario timetable")

    # Evaluate the same indexed timetable contract used by bootstrap-prior.  A
    # single origin/time slice is cached at a time; the complete OD--time
    # support is never materialized in memory.
    feasibility_index = TimetableFeasibilityIndex.from_scenario(
        scenario, timetable_index=timetable_index
    )
    periods = {
        str(item.bin_id): (
            str(item.bin_id),
            int(item.start.seconds_from_midnight),
            int(item.end.seconds_from_midnight),
        )
        for item in scenario.time_bins
    }
    cached_slice: tuple[str, str] | None = None
    cached_slice_metrics: Mapping[str, Any] = {}
    free_index_set = set(int(index) for index in free_indices)
    fixed_index_set = set(layout.fixed_od_indices)
    all_keys = tuple(layout.od_keys)
    free_metrics: list[Any] = []
    supported_cells = 0
    unsupported_free_count = 0
    unsupported_examples: list[tuple[str, str, str]] = []
    unsupported_fixed = 0
    counts_by_reason: dict[str, int] = {}
    counts_by_time_bin: dict[str, dict[str, object]] = {}
    support_cells_path = results_root / "audit/feasibility_support_cells.jsonl"
    support_cells_tmp = support_cells_path.with_name(
        f".{support_cells_path.name}.tmp"
    )
    support_cells_tmp.parent.mkdir(parents=True, exist_ok=True)

    def time_bin_counts(time_bin: str) -> dict[str, object]:
        return counts_by_time_bin.setdefault(
            time_bin,
            {
                "supported": 0,
                "unsupported_free": 0,
                "unsupported_fixed": 0,
                "reasons": {},
            },
        )

    def metric_for(
        origin: str, destination: str, time_bin: str
    ) -> Any | None:
        nonlocal cached_slice, cached_slice_metrics
        period = periods.get(time_bin)
        if period is None:
            return None
        slice_key = (origin, time_bin)
        if slice_key != cached_slice:
            cached_slice_metrics = feasibility_contract.path_metrics(
                feasibility_index, origin=origin, period=period
            )
            cached_slice = slice_key
        return cached_slice_metrics.get(destination)

    with support_cells_tmp.open("w", encoding="utf-8") as support_stream:
        support_order = sorted(
            range(len(all_keys)),
            key=lambda item: (
                str(all_keys[item][0]),
                str(all_keys[item][2]),
                str(all_keys[item][1]),
            ),
        )
        for index in support_order:
            key = all_keys[index]
            origin, destination, time_bin = (str(value) for value in key)
            counts = time_bin_counts(time_bin)
            metric = metric_for(origin, destination, time_bin)
            reason = None if metric is not None else (
                "unknown_time_bin"
                if time_bin not in periods
                else "no_feasible_path"
            )
            if reason is None:
                supported_cells += 1
                counts["supported"] += 1
            elif index in free_index_set:
                unsupported_free_count += 1
                if len(unsupported_examples) < 5:
                    unsupported_examples.append((origin, destination, time_bin))
                counts["unsupported_free"] += 1
                counts_by_reason[reason] = counts_by_reason.get(reason, 0) + 1
                reason_counts = counts["reasons"]
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                support_stream.write(
                    json.dumps(
                        {
                            "origin_stop_id": origin,
                            "destination_stop_id": destination,
                            "time_bin_id": time_bin,
                            "classification": "unsupported_free",
                            "reason": reason,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            elif index in fixed_index_set:
                unsupported_fixed += 1
                counts["unsupported_fixed"] += 1
                fixed_reason = "fixed_cell:" + reason
                counts_by_reason[fixed_reason] = counts_by_reason.get(fixed_reason, 0) + 1
                reason_counts = counts["reasons"]
                reason_counts[fixed_reason] = reason_counts.get(fixed_reason, 0) + 1
                support_stream.write(
                    json.dumps(
                        {
                            "origin_stop_id": origin,
                            "destination_stop_id": destination,
                            "time_bin_id": time_bin,
                            "classification": "fixed",
                            "reason": reason,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            else:  # pragma: no cover - ODParameterLayout validates the partition
                raise AssertionError(f"OD index {index} is neither free nor fixed")
    support_cells_tmp.replace(support_cells_path)

    prior_audit = _prior_expansion_audit(results_root)
    expansion_audit = prior_audit.get("expansion", {})
    if not isinstance(expansion_audit, dict):
        expansion_audit = {}
    total_generated = int(expansion_audit.get("total_cells", len(all_keys)))
    retained_cells = int(expansion_audit.get("retained_cell_count", len(all_keys)))
    bootstrap_only = max(retained_cells - len(all_keys), 0)
    feature_only = max(len(all_keys) - retained_cells, 0)
    if bootstrap_only:
        counts_by_reason["bootstrap_support_not_in_scenario_demand"] = bootstrap_only
    if feature_only:
        counts_by_reason["feature_support_not_in_bootstrap"] = feature_only

    support_audit: dict[str, object] = {
        "schema_version": 1,
        "status": (
            "completed"
            if not unsupported_free_count and not bootstrap_only and not feature_only
            else "failed"
        ),
        "support_domain": "all retained prior cells; feature vectors use the free subset",
        "contract": {
            "version": feasibility_contract.version,
            "fingerprint": feasibility_contract.fingerprint,
            "maximum_transfers": feasibility_contract.maximum_transfers,
            "maximum_initial_wait_seconds": feasibility_contract.maximum_initial_wait_seconds,
            "maximum_journey_seconds": feasibility_contract.maximum_journey_seconds,
            "maximum_waiting_seconds": feasibility_contract.maximum_waiting_seconds,
        },
        "scenario_fingerprint": fingerprint_scenario(scenario),
        "timetable_fingerprint": getattr(timetable_index, "fingerprint", None),
        "time_bins_fingerprint": _time_bin_fingerprint(scenario),
        "bootstrap_expansion_fingerprint": expansion_audit.get("fingerprint"),
        "bootstrap_package_revision": prior_audit.get("package_revision"),
        "od_universe_fingerprint": prior_audit.get("od_universe", {}).get("fingerprint")
        if isinstance(prior_audit.get("od_universe"), dict)
        else None,
        "bootstrap_configuration_fingerprint": (
            expansion_audit.get("configuration", {}).get("configuration_fingerprint")
            if isinstance(expansion_audit.get("configuration"), dict)
            else None
        ),
        "total_generated_prior_cells": total_generated,
        "retained_cells": retained_cells,
        "evaluated_retained_cells": len(all_keys),
        "supported_cells": supported_cells,
        "unsupported_retained_cells": unsupported_free_count + unsupported_fixed,
        "unsupported_free_cells": unsupported_free_count,
        "unsupported_fixed_cells": unsupported_fixed,
        "cells_present_only_in_bootstrap_support": bootstrap_only,
        "cells_present_only_in_feature_construction_support": feature_only,
        "od_layout_fingerprint": compact.fingerprint,
        "counts_by_reason": counts_by_reason,
        "counts_by_time_bin": counts_by_time_bin,
        "unsupported_cells_path": str(support_cells_path),
        "audit_path": str(results_root / "audit/feasibility_support.json"),
    }
    _write_json_atomically(results_root / "audit/feasibility_support.json", support_audit)
    if unsupported_free_count or bootstrap_only or feature_only:
        raise ValueError(
            "bootstrap-prior/check feasibility support mismatch; see "
            f"{results_root / 'audit/feasibility_support.json'}; "
            f"unsupported free cells={unsupported_free_count}, "
            f"bootstrap-only cells={bootstrap_only}, "
            f"feature-only cells={feature_only}, examples={unsupported_examples}"
        )

    # Build the feature arrays in the layout's canonical free-cell order. The
    # evaluator cache still contains only one origin/time slice at a time.
    cached_slice = None
    cached_slice_metrics = {}
    free_metrics: list[Any] = [None] * len(keys)
    free_positions = sorted(
        enumerate(free_indices.tolist()),
        key=lambda item: (
            str(all_keys[int(item[1])][0]),
            str(all_keys[int(item[1])][2]),
            str(all_keys[int(item[1])][1]),
        ),
    )
    for position, index in free_positions:
        key = all_keys[int(index)]
        metric = metric_for(str(key[0]), str(key[1]), str(key[2]))
        if metric is None:  # pragma: no cover - support pass already validated
            raise AssertionError(f"missing support metric for free cell {key!r}")
        free_metrics[position] = metric

    metrics = {
        key: metric for key, metric in zip(keys, free_metrics, strict=True)
    }
    origins = sorted({key[0] for key in keys})
    destinations = sorted({key[1] for key in keys})
    periods = sorted({key[2] for key in keys})
    origin_lookup = {value: index for index, value in enumerate(origins)}
    destination_lookup = {value: index for index, value in enumerate(destinations)}
    period_lookup = {value: index for index, value in enumerate(periods)}
    groups = sorted({(key[0], key[2]) for key in keys})
    group_lookup = {value: index for index, value in enumerate(groups)}
    baseline = np.asarray(layout.free_baseline_values, dtype=np.float32)
    totals = np.zeros(len(groups), dtype=np.float32)
    destination_totals: dict[tuple[str, str], float] = {}
    for key, value in zip(keys, baseline, strict=True):
        totals[group_lookup[(key[0], key[2])]] += value
        destination_totals[(key[1], key[2])] = destination_totals.get((key[1], key[2]), 0.0) + float(value)
    return GravityFeatures(
        canonical_od_index=free_indices,
        origin_index=np.asarray([origin_lookup[key[0]] for key in keys]),
        destination_index=np.asarray([destination_lookup[key[1]] for key in keys]),
        departure_time_index=np.asarray([period_lookup[key[2]] for key in keys]),
        origin_time_group_index=np.asarray([group_lookup[(key[0], key[2])] for key in keys]),
        journey_time=np.asarray([metrics[key].minimum_journey_seconds / 60.0 for key in keys], dtype=np.float32),
        transfer_count=np.asarray([metrics[key].minimum_transfers for key in keys], dtype=np.int64),
        structural_feasible=np.ones(len(keys), dtype=bool),
        origin_time_totals=totals,
        destination_attractiveness=np.asarray([destination_totals[(key[1], key[2])] for key in keys], dtype=np.float32),
        num_origins=len(origins),
        num_destinations=len(destinations),
        num_departure_times=len(periods),
        od_layout_fingerprint=compact.fingerprint,
        journey_time_scale=journey_time_scale,
        time_period_index=np.asarray([period_lookup[key[2]] for key in keys], dtype=np.int64),
    )


def load_context(
    root: str | Path,
    *,
    settings: CaseSettings | None = None,
    progress: ProgressCallback | None = None,
) -> CaseContext:
    """Build the case context while exposing coarse initialization phases."""
    if settings is None:
        _progress(
            progress,
            phase="configuration_loading",
            status="started",
            current_unit="case.toml",
        )
        settings = CaseSettings.load(root)
        _progress(
            progress,
            phase="configuration_loading",
            status="completed",
            current_unit="case.toml",
        )
    else:
        _progress(
            progress,
            phase="configuration_loading",
            status="completed",
            current_unit="case.toml",
        )

    _progress(progress, phase="scenario_loading", status="started", current_unit="scenario")
    scenario = Scenario.from_folder(
        settings.scenario,
        strict=True,
        demand_file=settings.scenario_demand,
    )
    _progress(progress, phase="scenario_loading", status="completed", current_unit="scenario")

    _progress(
        progress,
        phase="timetable_index_construction",
        status="started",
        current_unit="canonical_timetable_index",
    )
    timetable_index = build_canonical_timetable_index(scenario)
    _progress(
        progress,
        phase="timetable_index_construction",
        status="completed",
        current_unit="canonical_timetable_index",
    )

    _progress(progress, phase="fixed_demand_loading", status="started", current_unit="fixed_demand")
    fixed_path, fixed_source, fixed_sha256 = _resolve_fixed_demand(settings)
    fixed = read_fixed_demand_csv(fixed_path, scenario=scenario)
    _progress(
        progress,
        phase="fixed_demand_loading",
        status="completed",
        current_unit="fixed_demand",
        fixed_demand_path=str(fixed_path),
        fixed_demand_source=fixed_source,
        fixed_demand_sha256=fixed_sha256,
    )

    _progress(progress, phase="od_layout_construction", status="started", current_unit="od_layout")
    layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed)
    _progress(progress, phase="od_layout_construction", status="completed", current_unit="od_layout")
    _progress(progress, phase="compact_layout_construction", status="started", current_unit="compact_od_layout")
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    _progress(progress, phase="compact_layout_construction", status="completed", current_unit="compact_od_layout")

    _progress(progress, phase="assignment_preparation", status="started", current_unit="assignment")
    assignment = prepare_assignment(
        scenario=scenario,
        config=AssignmentConfig(),
        timetable_index=timetable_index,
    )
    inputs = build_assignment_inputs(artifacts=assignment, compact_layout=compact)
    _progress(progress, phase="assignment_preparation", status="completed", current_unit="assignment")
    _progress(progress, phase="assignment_id_construction", status="started", current_unit="assignment_ids")
    id_manager = AssignmentIDManager.build(scenario=scenario, graph=assignment.graph)
    _progress(progress, phase="assignment_id_construction", status="completed", current_unit="assignment_ids")

    _progress(progress, phase="measurement_mapping", status="started", current_unit="measurements")
    table = read_measurements_csv(settings.measurements)
    mapping = build_mapping_spec_strict(id_manager=id_manager, table=table)
    _progress(progress, phase="measurement_mapping", status="completed", current_unit="measurements")
    _progress(progress, phase="canonical_index_construction", status="started", current_unit="canonical_index")
    canonical = _canonical_index(scenario, layout, table)
    _progress(progress, phase="canonical_index_construction", status="completed", current_unit="canonical_index")

    _progress(progress, phase="identity_construction", status="started", current_unit="artifact_identity")
    identity = build_scheduled_reference_artifact_identity(
        inputs=inputs,
        spec=mapping.spec,
        canonical_index=canonical,
        theta=settings.theta,
        temporal_discretization_fingerprint="scenario-time-bins-v1",
        departure_choice_fingerprint="scenario-canonical-departure-bins-v1",
        feasibility_fingerprint="assignment-config-default-v1",
        coefficient_policy_fingerprint="exact-float32-v1",
    )
    _progress(progress, phase="identity_construction", status="completed", current_unit="artifact_identity")
    _progress(progress, phase="feature_construction", status="started", current_unit="gravity_features")
    feasibility_contract = ScheduledFeasibilityContract(
        maximum_transfers=settings.maximum_transfers,
        maximum_initial_wait_seconds=settings.maximum_initial_wait_seconds,
        maximum_journey_seconds=settings.maximum_journey_seconds,
        maximum_waiting_seconds=settings.maximum_waiting_seconds,
    )
    features = _features(
        scenario,
        layout,
        compact,
        timetable_index=timetable_index,
        feasibility_contract=feasibility_contract,
        results_root=settings.results,
        journey_time_scale=float(settings.model.get("journey_time_scale", 30.0)),
    )
    _progress(progress, phase="feature_construction", status="completed", current_unit="gravity_features")
    _progress(progress, phase="context_initialization", status="completed", current_unit="context")
    return CaseContext(
        settings=settings,
        fixed_demand_path=fixed_path,
        fixed_demand_source=fixed_source,
        fixed_demand_sha256=fixed_sha256,
        scenario=scenario,
        timetable_index=timetable_index,
        assignment=assignment,
        id_manager=id_manager,
        measurements=table,
        mapping=mapping,
        parameter_layout=layout,
        compact_layout=compact,
        canonical_index=canonical,
        features=features,
        identity=identity,
    )


def activate(
    context: CaseContext,
    *,
    progress: Callable[[object], None] | None = None,
) -> DirectScheduledActivationResult:
    settings = context.settings
    inputs = build_assignment_inputs(
        artifacts=context.assignment, compact_layout=context.compact_layout
    )
    config = ShardedConstructionConfig(
        od_chunk_size=int(settings.model.get("od_chunk_size", 16)),
        measurement_block_size=int(settings.model.get("measurement_block_size", 64)),
        worker_memory_budget_bytes=int(settings.model.get("worker_memory_budget_bytes", 256_000_000)),
        workers=int(
            settings.model.get(
                "shard_construction_workers",
                settings.model.get("workers", 1),
            )
        ),
        support_workers=int(
            settings.model.get(
                "support_workers",
                settings.model.get("workers", 1),
            )
        ),
        target_nonzeros_per_storage_shard=int(settings.model.get("target_nonzeros_per_storage_shard", 20_000)),
        maximum_nonzeros_per_storage_shard=int(settings.model.get("maximum_nonzeros_per_storage_shard", 100_000)),
        manifest_checkpoint_shards=int(settings.model.get("manifest_checkpoint_shards", 1)),
        progress_interval_seconds=float(
            settings.model.get("support_progress_interval_seconds", 1.0)
        ),
    )
    profile_enabled = bool(settings.model.get("profile_support_discovery", False))
    recorder = SupportDiscoveryProfileRecorder.create() if profile_enabled else None
    profile_started = perf_counter()
    activated = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=settings.expected_evaluations,
        construction_seconds=None,
        reference_evaluation_seconds=1.0,
        operator_evaluation_seconds=0.0,
        checkpoint_root=settings.results / "checkpoints",
        artifact_root=settings.results / "artifacts",
        inputs=inputs,
        routing_factory=lambda: prepare_fixed_routing(inputs=inputs, theta=settings.theta),
        theta=settings.theta,
        spec=context.mapping.spec,
        compact_layout=context.compact_layout,
        canonical_index=context.canonical_index,
        observations=np.asarray(context.mapping.y_obs),
        identity=context.identity,
        assignment_fingerprint=str(context.id_manager.fingerprint),
        od_layout_fingerprint=context.parameter_layout.fingerprint,
        config=config,
        progress=progress,
        time_budget_seconds=settings.construction_time_budget_seconds,
        safety_margin_seconds=settings.safety_margin_seconds,
        measurement_info=context.mapping.info,
        support_timing_callback=recorder,
    )
    if recorder is not None:
        profile = build_support_discovery_profile(
            recorder.records(),
            metadata={
                "package_revision": settings.package_revision,
                "assignment_fingerprint": str(context.id_manager.fingerprint),
                "od_layout_fingerprint": context.parameter_layout.fingerprint,
                "support_workers_requested": config.support_workers,
                "shard_construction_workers_requested": config.workers,
                "results_root": str(settings.results),
            },
            elapsed_seconds=perf_counter() - profile_started,
        )
        write_support_discovery_profile(
            settings.results / "audit/support_discovery_profile.json", profile
        )
    return activated


def gravity_problem(context: CaseContext, operator: object) -> tuple[GravityObjectiveProblem, GravityParameterLayout]:
    model = context.settings.model
    destination_count = context.features.num_destinations
    components = (
        GravityComponentSpecification(
            "production", GravityEffectScope.GLOBAL,
            GravityParameterization.LOG_MULTIPLIER,
            source="origin_time_totals",
        ),
        GravityComponentSpecification(
            "destination_attractiveness", GravityEffectScope.DESTINATION,
            GravityParameterization.ADDITIVE,
            grouping="destination_index", group_count=destination_count,
            constraint=GravityConstraint.SUM_ZERO,
            regularization=GravityRegularization(
                GravityRegularizationType.RIDGE,
                float(model.get("destination_ridge", 1.0)),
            ), source="feature_cache",
        ),
    )
    specification = GravityModelSpecification(
        components=components,
        likelihood=GravityLikelihoodSpecification(
            family=str(model.get("likelihood", "negative_binomial")),
            calibration_mask="supported_measurements",
        ),
        time=GravityModelSpecification().time,
    )
    parameters = GravityParameterLayout(specification)
    problem = GravityObjectiveProblem(
        features=context.features,
        parameter_layout=parameters,
        operator=operator,
        observations=np.asarray(context.mapping.y_obs),
        likelihood=GravityLikelihood(str(model.get("likelihood", "negative_binomial"))),
        rho=context.settings.rho,
    )
    return problem, parameters


def initial_raw_parameters(
    parameters: GravityParameterLayout,
    model: dict[str, object] | None = None,
) -> np.ndarray:
    """Return finite raw starts; named values are raw, not constrained values."""
    model = {} if model is None else model
    raw = np.zeros(parameters.size, dtype=np.float64)
    starts = {
        "beta_time": float(model.get("initial_beta_time", 0.0)),
        "beta_transfer": float(model.get("initial_beta_transfer", 0.0)),
        "dispersion": float(model.get("initial_log_dispersion", 0.0)),
        "production_scale": float(model.get("initial_production_log_multiplier", 0.0)),
    }
    for name, index in ((name, i) for i, name in enumerate(parameters.names)):
        if name in starts:
            raw[index] = starts[name]
    if not np.all(np.isfinite(raw)):
        raise ValueError("initial raw parameters must be finite.")
    return raw


def evaluate_once(problem: GravityObjectiveProblem, raw: np.ndarray) -> dict[str, object]:
    value, gradient = gravity_value_and_gradient(
        raw, problem=problem, strategy=GravityGradientStrategy.ADJOINT
    )
    value = jax.tree.map(lambda item: np.asarray(jax.block_until_ready(item)), value)
    gradient = np.asarray(jax.block_until_ready(gradient))
    return {
        "objective": float(value.objective),
        "gradient_inf_norm": float(np.max(np.abs(gradient))) if gradient.size else 0.0,
        "measurement_mean": np.asarray(value.measurement_mean),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
