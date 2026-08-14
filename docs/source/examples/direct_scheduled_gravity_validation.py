"""End-to-end validation of direct scheduled activation on a public example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

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
    build_compact_od_assignment_layout,
)
from public_transportation.inference.direct_scheduled_temporal_builder import (
    activate_direct_scheduled_temporal_operator,
)
from public_transportation.inference.construction_control import termination_payload
from public_transportation.inference.fixed_routing_measurement_operator import (
    load_or_prepare_fixed_routing_measurement_operator,
)
from public_transportation.inference.fixed_routing_sharded_builder import (
    ShardedConstructionConfig,
)
from public_transportation.inference.gravity import (
    GravityGradientStrategy,
    GravityLikelihood,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    gravity_value_and_gradient,
)
from public_transportation.inference.od_parameter_layout import (
    build_od_parameter_layout,
)
from public_transportation.inference.scheduled_reference_operator import (
    build_scheduled_reference_artifact_identity,
)
from public_transportation.measurement import (
    build_mapping_spec_strict,
    read_measurements_csv,
)

from docs.source.examples.simple_gravity_workflow import _features


def _canonical_index(scenario, layout, table):
    intervals = list(
        CanonicalTimeInterval(
            interval_id=item.bin_id,
            start_seconds=item.start.seconds_from_midnight,
            end_seconds=item.end.seconds_from_midnight,
        )
        for item in scenario.time_bins
    )
    latest_measurement = max(
        record.time.seconds_from_midnight for record in table.records
    )
    if latest_measurement >= intervals[-1].end_seconds:
        last = intervals[-1]
        intervals[-1] = CanonicalTimeInterval(
            interval_id=last.interval_id,
            start_seconds=last.start_seconds,
            end_seconds=latest_measurement + 1,
        )
    intervals_tuple = tuple(intervals)

    def interval_at(seconds: int) -> str:
        for interval in intervals_tuple:
            if interval.start_seconds <= seconds < interval.end_seconds:
                return interval.interval_id
        raise ValueError(f"measurement time {seconds} is outside canonical intervals.")

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
        time_intervals=intervals_tuple,
        measurements=measurements,
    )


def run_validation(
    *,
    example: Path,
    cache_directory: Path,
    time_budget_seconds: float | None = None,
    safety_margin_seconds: float = 0.0,
) -> dict[str, object]:
    """Build or reuse the temporal operator and compare one gravity evaluation."""
    data = example / "data"
    measurement_path = (
        example / "pre_processing/results/measurements_boarding_alighting.csv"
    )
    scenario = Scenario.from_folder(
        data, strict=True, demand_file=data / "prior_demand.csv"
    )
    layout = build_od_parameter_layout(
        scenario=scenario,
        fixed_demand=read_fixed_demand_csv(data / "fixed_demand.csv", scenario=scenario),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    features = _features(scenario, layout, compact)
    artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    id_manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)
    table = read_measurements_csv(measurement_path)
    mapped = build_mapping_spec_strict(id_manager=id_manager, table=table)
    canonical = _canonical_index(scenario, layout, table)
    theta = 1.0
    identity = build_scheduled_reference_artifact_identity(
        inputs=inputs,
        spec=mapped.spec,
        canonical_index=canonical,
        theta=theta,
        temporal_discretization_fingerprint="scenario-time-bins-v1",
        departure_choice_fingerprint="scenario-canonical-departure-bins-v1",
        feasibility_fingerprint="assignment-config-default-v1",
        coefficient_policy_fingerprint="exact-float32-v1",
    )
    progress_events: list[dict[str, object]] = []

    def progress(event: dict[str, object]) -> None:
        progress_events.append(dict(event))
        print("PROGRESS " + json.dumps(event, sort_keys=True), flush=True)

    started = perf_counter()
    activated = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=20,
        construction_seconds=None,
        reference_evaluation_seconds=1.94,
        operator_evaluation_seconds=0.0,
        checkpoint_root=cache_directory / "checkpoints",
        artifact_root=cache_directory / "artifacts",
        inputs=inputs,
        routing_factory=lambda: prepare_fixed_routing(inputs=inputs, theta=theta),
        theta=theta,
        spec=mapped.spec,
        compact_layout=compact,
        canonical_index=canonical,
        observations=np.asarray(mapped.y_obs),
        identity=identity,
        measurement_info=mapped.info,
        assignment_fingerprint=str(id_manager.fingerprint),
        od_layout_fingerprint=layout.fingerprint,
        config=ShardedConstructionConfig(
            od_chunk_size=16,
            measurement_block_size=64,
            worker_memory_budget_bytes=256_000_000,
            target_nonzeros_per_storage_shard=20_000,
            maximum_nonzeros_per_storage_shard=100_000,
            manifest_checkpoint_shards=1,
        ),
        progress=progress,
        time_budget_seconds=time_budget_seconds,
        safety_margin_seconds=safety_margin_seconds,
    )
    activation_seconds = perf_counter() - started
    if activated.operator is None:
        assert activated.termination is not None
        summary = {
            "schema_version": 1,
            "example": example.name,
            "cache_reused": False,
            "activation_seconds": activation_seconds,
            "construction_status": "deadline_stopped",
            "termination": termination_payload(activated.termination),
            "progress_event_count": len(progress_events),
            "construction_progress_event_count": sum(
                event.get("phase")
                not in {"cache_validation", "measurement_support_preflight"}
                for event in progress_events
            ),
        }
        print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
        return summary

    routing = prepare_fixed_routing(inputs=inputs, theta=theta)
    reference = load_or_prepare_fixed_routing_measurement_operator(
        cache_directory=cache_directory / "reference-bcoo",
        inputs=inputs,
        routing=routing,
        spec=mapped.spec,
        assignment_fingerprint=str(id_manager.fingerprint),
        compact_layout=compact,
        od_layout_fingerprint=layout.fingerprint,
        representation="bcoo",
        chunk_size=16,
    )
    parameter_layout = GravityParameterLayout(GravityModelSpecification())
    raw = parameter_layout.raw_from_physical((0.5, 1.0, 10.0))

    def evaluate(operator):
        problem = GravityObjectiveProblem(
            features=features,
            parameter_layout=parameter_layout,
            operator=operator,
            observations=np.asarray(mapped.y_obs),
            likelihood=GravityLikelihood.NEGATIVE_BINOMIAL,
        )
        result = gravity_value_and_gradient(
            raw, problem=problem, strategy=GravityGradientStrategy.ADJOINT
        )
        return jax.tree.map(lambda value: np.asarray(jax.block_until_ready(value)), result)

    temporal_value, temporal_gradient = evaluate(activated.operator)
    reference_value, reference_gradient = evaluate(reference)
    summary = {
        "schema_version": 1,
        "construction_status": "completed",
        "example": example.name,
        "cache_reused": activated.decision.cache_reused,
        "activation_reason": activated.decision.reason,
        "activation_seconds": activation_seconds,
        "construction_seconds": (
            None
            if activated.construction is None or activated.construction.source is None
            else activated.construction.source.total_seconds
        ),
        "progress_event_count": len(progress_events),
        "construction_progress_event_count": sum(
            event.get("phase")
            not in {"cache_validation", "measurement_support_preflight"}
            for event in progress_events
        ),
        "objective_absolute_difference": float(
            np.abs(temporal_value.objective - reference_value.objective)
        ),
        "gradient_maximum_absolute_difference": float(
            np.max(np.abs(temporal_gradient - reference_gradient))
        ),
        "stored_bytes": activated.operator.metrics.stored_bytes,
        "nonzero_entries": activated.operator.operator.diagnostics.nonzero_entries,
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    default_example = Path(__file__).resolve().parent / "simple_example_02"
    parser.add_argument("--example", type=Path, default=default_example)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--require-cache-reuse", action="store_true")
    parser.add_argument("--time-budget-seconds", type=float)
    parser.add_argument("--safety-margin-seconds", type=float, default=0.0)
    parser.add_argument("--allow-deadline-stop", action="store_true")
    arguments = parser.parse_args()
    summary = run_validation(
        example=arguments.example,
        cache_directory=arguments.cache_directory,
        time_budget_seconds=arguments.time_budget_seconds,
        safety_margin_seconds=arguments.safety_margin_seconds,
    )
    if summary["construction_status"] == "deadline_stopped":
        if arguments.allow_deadline_stop:
            return
        raise SystemExit("construction stopped cleanly at its deadline")
    if arguments.require_cache_reuse and not summary["cache_reused"]:
        raise SystemExit("expected a valid temporal artifact from an earlier process")
    if summary["objective_absolute_difference"] > 5.0e-3:
        raise SystemExit("objective equivalence tolerance exceeded")
    if summary["gradient_maximum_absolute_difference"] > 5.0e-3:
        raise SystemExit("gradient equivalence tolerance exceeded")


if __name__ == "__main__":
    main()
