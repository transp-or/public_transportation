"""Case-owned assembly of the current direct-scheduled public APIs.

This file is deliberately a small, explicit adapter rather than a hidden
framework.  A private case may replace the feature construction and model
specification, but should keep the identity checks and stage boundaries.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable

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
    ODTimeKey,
    build_canonical_timetable_index,
    build_structural_zero_topology,
    compute_od_path_metrics,
)
from public_transportation.preprocessing.structural_zeros.config import (
    StructuralZeroAssignmentConfig,
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
            model=model,
        )


@dataclass(frozen=True, slots=True)
class CaseContext:
    settings: CaseSettings
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
    journey_time_scale: float,
) -> GravityFeatures:
    free_indices = np.asarray(layout.free_od_indices, dtype=np.int64)
    keys = [layout.od_keys[index] for index in free_indices]
    topology = build_structural_zero_topology(
        scenario,
        StructuralZeroAssignmentConfig(),
        timetable_index=timetable_index,
    )
    records = compute_od_path_metrics(
        topology, keys=tuple(ODTimeKey(*key) for key in keys)
    )
    metrics = {record.key.tuple: record.metrics for record in records}
    unsupported = [key for key in keys if not metrics[key].feasible]
    if unsupported:
        raise ValueError(f"free demand contains unsupported scheduled cells: {unsupported[:5]}")
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
        journey_time=np.asarray([metrics[key].minimum_journey_time_minutes for key in keys], dtype=np.float32),
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


def load_context(root: str | Path) -> CaseContext:
    settings = CaseSettings.load(root)
    scenario = Scenario.from_folder(
        settings.scenario,
        strict=True,
        demand_file=settings.scenario_demand,
    )
    timetable_index = build_canonical_timetable_index(scenario)
    generated_fixed = settings.results / "structural_zeros/fixed_demand.csv"
    fixed_path = generated_fixed if generated_fixed.is_file() else settings.fixed_demand
    fixed = read_fixed_demand_csv(fixed_path, scenario=scenario)
    layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed)
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    assignment = prepare_assignment(
        scenario=scenario,
        config=AssignmentConfig(),
        timetable_index=timetable_index,
    )
    inputs = build_assignment_inputs(artifacts=assignment, compact_layout=compact)
    id_manager = AssignmentIDManager.build(scenario=scenario, graph=assignment.graph)
    table = read_measurements_csv(settings.measurements)
    mapping = build_mapping_spec_strict(id_manager=id_manager, table=table)
    canonical = _canonical_index(scenario, layout, table)
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
    features = _features(
        scenario,
        layout,
        compact,
        timetable_index=timetable_index,
        journey_time_scale=float(settings.model.get("journey_time_scale", 30.0)),
    )
    return CaseContext(
        settings, scenario, timetable_index, assignment, id_manager, table,
        mapping, layout, compact, canonical, features, identity
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
        target_nonzeros_per_storage_shard=int(settings.model.get("target_nonzeros_per_storage_shard", 20_000)),
        maximum_nonzeros_per_storage_shard=int(settings.model.get("maximum_nonzeros_per_storage_shard", 100_000)),
        manifest_checkpoint_shards=int(settings.model.get("manifest_checkpoint_shards", 1)),
    )
    return activate_direct_scheduled_temporal_operator(
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
    )


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
