from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from public_transportation.domain import (
    Metadata,
    ODDemand,
    Scenario,
    Stop,
    StopTime,
    TimeBin,
    TimeOfDay,
    Timetable,
    Trip,
)
from public_transportation.domain.line import Line
from public_transportation.inference.reduced_od import (
    MinimalGravitySpecification,
    ReducedODPreparationInputs,
    benchmark_minimal_gravity_objective,
    build_minimal_gravity_problem,
    load_reduced_od_artifacts,
    preflight_reduced_od_j0,
    prepare_reduced_od_artifacts,
)
from public_transportation.inference.reduced_od.integration import (
    _departure_sampling_identity,
    _sampling_support,
)
from public_transportation.preprocessing.reduced_od import DepartureTimeSamplingConfig
from public_transportation.measurement import (
    MeasurementRecord,
    MeasurementTable,
    MeasurementType,
)
from public_transportation.preprocessing.reduced_od import (
    Footpath,
    JourneyTimePeriod,
    ResponseCellKey,
    ReducedODArtifactStoreError,
    canonical_json,
    load_reduced_od_config,
)


def _scenario() -> Scenario:
    stops = [
        Stop(stop_id, stop_id, 46.0 + index * 0.001, 6.0)
        for index, stop_id in enumerate(("A", "B", "C", "D"))
    ]
    schedules = {
        "DIRECT": (("A", 29100), ("D", 30600)),
        "FIRST": (("A", 28800), ("B", 29400)),
        "SECOND": (("C", 29700), ("D", 30300)),
    }
    trips = [
        Trip("DIRECT", "LD", service_id="day", direction_id=0),
        Trip("FIRST", "L1", service_id="day", direction_id=0),
        Trip("SECOND", "L2", service_id="day", direction_id=0),
    ]
    stop_times = [
        StopTime(trip, stop, sequence, seconds, seconds)
        for trip, rows in schedules.items()
        for sequence, (stop, seconds) in enumerate(rows, start=1)
    ]
    return Scenario(
        metadata=Metadata(title="integration", created_at="2026-01-01T00:00:00"),
        stops=stops,
        lines=[Line("L1"), Line("L2"), Line("LD")],
        time_bins=[TimeBin("day", TimeOfDay(0), TimeOfDay(108000))],
        demand=ODDemand(records=[]),
        timetable=Timetable(trips=trips, stop_times=stop_times),
    )


def _measurements() -> MeasurementTable:
    rows = (
        (MeasurementType.BOARDING, "A", 28800, "FIRST", 10.0),
        (MeasurementType.ALIGHTING, "B", 29400, "FIRST", 9.0),
        (MeasurementType.BOARDING, "C", 29700, "SECOND", 8.0),
        (MeasurementType.ALIGHTING, "D", 30300, "SECOND", 7.0),
        (MeasurementType.BOARDING, "A", 29100, "DIRECT", 6.0),
        (MeasurementType.ALIGHTING, "D", 30600, "DIRECT", 5.0),
    )
    return MeasurementTable.from_records(
        MeasurementRecord("apc", kind, stop, TimeOfDay(seconds), value, trip_id=trip)
        for kind, stop, seconds, trip, value in rows
    )


def _configuration(tmp_path: Path):
    path = tmp_path / "reduced_od.toml"
    path.write_text(
        """schema_version = 2
[observations]
service_day = "2026-01-15"
analysis_start_seconds = 21600
analysis_end_seconds = 108000
after_midnight_convention = "service_day_extended"
apc_policy_identifier = "synthetic-v1"
sensor_coverage_policy = "complete"
sensor_outage_policy = "exclude"
unit = "timetable_event"
accepted_types = ["boarding", "alighting"]
missing_policy = "exclude"
duplicate_policy = "error"
ambiguous_event_policy = "error"
cleaning_stage = "external"
[journeys]
origin_semantics = "first_boarding"
destination_semantics = "final_alighting"
time_bin_membership = "half_open"
maximum_transfers = 1
maximum_waiting_seconds = 3600
maximum_journey_seconds = 7200
maximum_alternatives_per_cell = 4
transfer_footpath_policy = "declared-v1"
route_shares = "fixed_within_fit"
[productions]
mode = "provided"
semantics = "external_journey_productions"
input_path = "productions.csv"
[stops]
mapping_policy = "identity"
[outputs]
spatial_level = "physical_stop"
reconstruct_full_od = false
[model]
likelihood = "poisson"
[validation]
detailed_assignment = "explicit_only"
""",
        encoding="utf-8",
    )
    return load_reduced_od_config(path)


def _prepare(tmp_path: Path):
    config = _configuration(tmp_path)
    directory = tmp_path / "artifacts"
    prepared = prepare_reduced_od_artifacts(
        scenario=_scenario(),
        measurements=_measurements(),
        configuration=config,
        inputs=ReducedODPreparationInputs(
            departure_seconds_by_origin={"A": (28800,)},
            production_inputs={("A", "P"): 40.0},
            destination_attractiveness={
                ("B", "P"): 1.0,
                ("C", "P"): 1.0,
                ("D", "P"): 1.0,
            },
            footpaths=(Footpath("B", "C", 120),),
            time_periods=(JourneyTimePeriod("P", 0, 108000),),
        ),
        output_directory=directory,
        cache_policy="rebuild",
    )
    return config, directory, prepared


def test_end_to_end_persist_reload_preflight_and_compact_problem(
    tmp_path: Path,
) -> None:
    config, directory, prepared = _prepare(tmp_path)
    assert all((path / "manifest.json").is_file() for path in prepared.paths.values())
    loaded = load_reduced_od_artifacts(
        configuration=config, artifact_directory=directory
    )
    built = build_minimal_gravity_problem(
        artifacts=loaded,
        specification=MinimalGravitySpecification(),
    )
    assert built.raw_parameter_dimension == 2
    assert built.raw_parameter_dimension < built.free_cell_count
    report = preflight_reduced_od_j0(configuration=config, artifact_directory=directory)
    assert report["compatible"] is True
    json.dumps(report)
    benchmark = benchmark_minimal_gravity_objective(
        problem=built.problem,
        raw_parameters=np.zeros(2),
        warm_evaluations=2,
    )
    assert benchmark.finite
    assert benchmark.recompiled_after_value_change is not True


def test_missing_and_fingerprint_mismatch_fail_closed(tmp_path: Path) -> None:
    config, directory, _ = _prepare(tmp_path)
    (directory / "journey_choices" / "manifest.json").unlink()
    report = preflight_reduced_od_j0(configuration=config, artifact_directory=directory)
    assert report["compatible"] is False
    with pytest.raises(ReducedODArtifactStoreError, match="missing"):
        load_reduced_od_artifacts(configuration=config, artifact_directory=directory)


def test_array_tampering_is_rejected(tmp_path: Path) -> None:
    config, directory, prepared = _prepare(tmp_path)
    manifest_path = prepared.paths["reduced_response_operator"] / "manifest.json"
    document = json.loads(manifest_path.read_text())
    array_name = next(iter(document["array_descriptors"]))
    array_path = manifest_path.parent / array_name
    array = np.load(array_path, allow_pickle=False)
    changed = np.array(array, copy=True)
    changed.flat[0] += 1
    with array_path.open("wb") as stream:
        np.save(stream, changed, allow_pickle=False)
    with pytest.raises((ReducedODArtifactStoreError, ValueError, OSError)):
        load_reduced_od_artifacts(configuration=config, artifact_directory=directory)


def test_model_only_changes_reuse_preprocessing_phases(tmp_path: Path) -> None:
    config, directory, prepared = _prepare(tmp_path)
    changed = replace(
        config, model=replace(config.model, likelihood="negative_binomial")
    )
    reused = prepare_reduced_od_artifacts(
        scenario=_scenario(),
        measurements=_measurements(),
        configuration=changed,
        inputs=ReducedODPreparationInputs(
            departure_seconds_by_origin={"A": (28800,)},
            production_inputs={("A", "P"): 40.0},
            destination_attractiveness={
                ("B", "P"): 1.0,
                ("C", "P"): 1.0,
                ("D", "P"): 1.0,
            },
            footpaths=(Footpath("B", "C", 120),),
            time_periods=(JourneyTimePeriod("P", 0, 108000),),
        ),
        output_directory=directory,
        cache_policy="reuse_or_build",
    )

    statuses = {item["phase"]: item["status"] for item in reused.phase_diagnostics}
    for phase in (
        "physical_stops",
        "service_periods_route_patterns",
        "timetable_index",
        "journey_choices",
        "measurement_response",
        "response_equivalence",
        "reduced_response_operator",
    ):
        assert statuses[phase] == "reused"
        assert reused.fingerprints[phase] == prepared.fingerprints[phase]


def test_journey_query_progress_is_monotone_and_throttled(tmp_path: Path) -> None:
    config = _configuration(tmp_path)
    events = []
    prepare_reduced_od_artifacts(
        scenario=_scenario(),
        measurements=_measurements(),
        configuration=config,
        inputs=ReducedODPreparationInputs(
            departure_seconds_by_origin={"A": (28800, 28900, 29000)},
            production_inputs={("A", "P"): 40.0},
            destination_attractiveness={
                ("B", "P"): 1.0,
                ("C", "P"): 1.0,
                ("D", "P"): 1.0,
            },
            footpaths=(Footpath("B", "C", 120),),
            time_periods=(JourneyTimePeriod("P", 0, 108000),),
        ),
        output_directory=tmp_path / "progress-artifacts",
        cache_policy="rebuild",
        progress=events.append,
    )
    query_events = [
        item
        for item in events
        if item["phase"] == "journey_choices"
        and item.get("status") in {"in_progress", "completed"}
    ]
    assert query_events
    assert [item["completed_queries"] for item in query_events] == sorted(
        item["completed_queries"] for item in query_events
    )
    assert query_events[-1]["completed_queries"] == 3
    assert query_events[-1]["status"] == "completed"
    assert len(query_events) < 3
    assert all(item["total_queries"] == 3 for item in query_events)


def test_desired_departure_sampling_keeps_one_cell_and_invalidates_downstream(
    tmp_path: Path,
) -> None:
    config = _configuration(tmp_path)
    directory = tmp_path / "sampled-artifacts"

    def inputs(level: int) -> ReducedODPreparationInputs:
        return ReducedODPreparationInputs(
            departure_seconds_by_origin={"A": (28800,)},
            production_inputs={("A", "P"): 40.0},
            destination_attractiveness={
                ("B", "P"): 1.0,
                ("C", "P"): 1.0,
                ("D", "P"): 1.0,
            },
            footpaths=(Footpath("B", "C", 120),),
            time_periods=(JourneyTimePeriod("P", 28000, 31000),),
            departure_time_sampling=DepartureTimeSamplingConfig(
                samples_per_period=level,
                minimum_feasible_fraction=0.3,
            ),
        )

    first = prepare_reduced_od_artifacts(
        scenario=_scenario(),
        measurements=_measurements(),
        configuration=config,
        inputs=inputs(3),
        output_directory=directory,
        cache_policy="rebuild",
    )
    loaded = load_reduced_od_artifacts(
        configuration=config,
        artifact_directory=directory,
    )
    assert first.dimensions["desired_departure_samples"] == 3
    assert loaded.departure_sampling_cells
    assert all(
        choice.origin_time_period_id == "P"
        for choice in loaded.journey_choices.choice_sets
    )

    changed = prepare_reduced_od_artifacts(
        scenario=_scenario(),
        measurements=_measurements(),
        configuration=config,
        inputs=inputs(6),
        output_directory=directory,
        cache_policy="reuse_or_build",
    )
    statuses = {item["phase"]: item["status"] for item in changed.phase_diagnostics}
    assert statuses["timetable_index"] == "reused"
    assert statuses["departure_time_samples"] == "built"
    assert statuses["journey_choices"] == "built"
    assert changed.dimensions["desired_departure_samples"] == 6


@pytest.mark.parametrize(
    ("strategy", "comparison_mode", "expected_samples", "effective_mode"),
    [
        ("fixed_time_step", "assignment_response", 10, None),
        ("adaptive_service_aware", "assignment_response", None, "aggregate_response"),
        ("adaptive_service_aware", "integral_response", None, "integral_response"),
    ],
)
def test_fixed_step_and_adaptive_sampling_integrate_with_public_pipeline(
    tmp_path: Path,
    strategy: str,
    comparison_mode: str,
    expected_samples: int | None,
    effective_mode: str | None,
) -> None:
    config = _configuration(tmp_path)
    events: list[dict[str, object]] = []
    sampling = DepartureTimeSamplingConfig(
        strategy=strategy,  # type: ignore[arg-type]
        time_step_seconds=300,
        initial_interval_seconds=900,
        minimum_interval_seconds=60,
        maximum_samples_per_cell=64,
        infeasible_policy="preserve_mass",
        minimum_feasible_fraction=0.0,
        warning_feasible_fraction=0.0,
        comparison_mode=comparison_mode,  # type: ignore[arg-type]
    )
    output_directory = tmp_path / f"{strategy}-{comparison_mode}"
    prepared = prepare_reduced_od_artifacts(
        scenario=_scenario(),
        measurements=_measurements(),
        configuration=config,
        inputs=ReducedODPreparationInputs(
            departure_seconds_by_origin={"A": (28800,)},
            production_inputs={("A", "P"): 40.0},
            destination_attractiveness={
                ("B", "P"): 1.0,
                ("C", "P"): 1.0,
                ("D", "P"): 1.0,
            },
            footpaths=(Footpath("B", "C", 120),),
            time_periods=(JourneyTimePeriod("P", 28000, 31000),),
            departure_time_sampling=sampling,
        ),
        output_directory=output_directory,
        cache_policy="rebuild",
        progress=lambda event: events.append(dict(event)),
    )
    if expected_samples is not None:
        assert prepared.dimensions["desired_departure_samples"] == expected_samples
    else:
        assert 1 <= prepared.dimensions["desired_departure_samples"] <= 64
        assert any(
            event["phase"] == "adaptive_departure_quadrature" for event in events
        )
        aggregate = [
            event
            for event in events
            if event["phase"] == "adaptive_departure_quadrature_batch"
        ]
        assert aggregate[-1]["status"] == "completed"
        assert aggregate[-1]["completed_origin_period_groups"] == 1
        assert "estimated_remaining_seconds" in aggregate[-1]
        assert aggregate[-1]["requested_comparison_mode"] == comparison_mode
        assert aggregate[-1]["effective_comparison_mode"] == effective_mode
        assert aggregate[-1]["mean_unresolved_fraction"] <= 1.0
        assert aggregate[-1]["maximum_group_unresolved_fraction"] <= 1.0
        assert aggregate[-1]["mean_stable_fraction"] <= 1.0
        assert aggregate[-1]["maximum_evaluations_per_group"] >= 1
        if comparison_mode == "assignment_response":
            assert any(
                event.get("status") == "warning"
                and event.get("effective_comparison_mode") == effective_mode
                for event in events
            )
        else:
            assert any(
                event.get("quadrature_rule") == "embedded_midpoint_integral"
                for event in events
            )
    loaded = load_reduced_od_artifacts(
        configuration=config, artifact_directory=output_directory
    )
    assert loaded.response_operator.number_of_free_cells >= 1
    if comparison_mode == "integral_response":
        payload = loaded.departure_time_samples
        assert isinstance(payload, dict)
        diagnostics = payload["adaptive_diagnostics"][0]
        assert diagnostics["quadrature_rule"] == "embedded_midpoint_integral"
        assert diagnostics["effective_comparison_mode"] == "integral_response"
        assert diagnostics["global_error_target"] >= 0.0


def test_sampling_identity_tracks_sparse_support_and_fixed_status() -> None:
    common = dict(
        departure_seconds_by_origin={"A": (28800,), "B": (28800,)},
        production_inputs={("A", "P"): 40.0},
        destination_attractiveness={("C", "P"): 1.0},
        time_periods=(JourneyTimePeriod("P", 28000, 31000),),
        departure_time_sampling=DepartureTimeSamplingConfig(samples_per_period=1),
    )
    first = ReducedODPreparationInputs(
        **common,
        departure_sampling_origin_period_groups=(("A", "P"),),
        fixed_demand={ResponseCellKey("A", "C", "P"): 0.0},
    )
    second = ReducedODPreparationInputs(
        **common,
        departure_sampling_origin_period_groups=(("A", "P"), ("B", "P")),
        fixed_demand={ResponseCellKey("A", "C", "P"): 0.0},
    )
    changed_fixed = ReducedODPreparationInputs(
        **common,
        departure_sampling_origin_period_groups=(("A", "P"),),
        fixed_demand={ResponseCellKey("A", "C", "P"): 2.0},
    )
    groups, statuses = _sampling_support(first)
    assert groups == (("A", "P"),)
    assert statuses == {ResponseCellKey("A", "C", "P"): "fixed_zero"}
    assert _departure_sampling_identity(first) != _departure_sampling_identity(second)
    assert _departure_sampling_identity(first) != _departure_sampling_identity(
        changed_fixed
    )


def test_sampling_identity_tracks_adaptive_budget_tolerance_and_mode() -> None:
    common = dict(
        departure_seconds_by_origin={"A": (28800,)},
        production_inputs={("A", "P"): 40.0},
        destination_attractiveness={("C", "P"): 1.0},
        time_periods=(JourneyTimePeriod("P", 28000, 31000),),
    )

    def identity(**changes: object) -> str:
        values: dict[str, object] = {
            "strategy": "adaptive_service_aware",
            "infeasible_policy": "preserve_mass",
            "maximum_samples_per_cell": 128,
            "response_tolerance": 1.0e-3,
            "comparison_mode": "assignment_response",
        }
        values.update(changes)
        inputs = ReducedODPreparationInputs(
            **common,
            departure_time_sampling=DepartureTimeSamplingConfig(  # type: ignore[arg-type]
                **values
            ),
        )
        return _departure_sampling_identity(inputs)

    baseline = identity()
    assert baseline != identity(maximum_samples_per_cell=256)
    assert baseline != identity(response_tolerance=2.0e-3)
    assert baseline != identity(comparison_mode="service_signature")
    integral = identity(comparison_mode="integral_response")
    assert baseline != integral
    assert integral != identity(
        comparison_mode="integral_response",
        absolute_response_tolerance=1.0e-3,
    )
    assert integral != identity(
        comparison_mode="integral_response",
        relative_response_tolerance=2.0e-2,
    )


def test_reuse_or_build_rebuilds_legacy_sampling_diagnostics(tmp_path: Path) -> None:
    config = _configuration(tmp_path)
    directory = tmp_path / "legacy-sampling"
    inputs = ReducedODPreparationInputs(
        departure_seconds_by_origin={"A": (28800,)},
        production_inputs={("A", "P"): 40.0},
        destination_attractiveness={
            ("B", "P"): 1.0,
            ("C", "P"): 1.0,
            ("D", "P"): 1.0,
        },
        footpaths=(Footpath("B", "C", 120),),
        time_periods=(JourneyTimePeriod("P", 28000, 31000),),
        departure_time_sampling=DepartureTimeSamplingConfig(samples_per_period=1),
    )
    prepared = prepare_reduced_od_artifacts(
        scenario=_scenario(),
        measurements=_measurements(),
        configuration=config,
        inputs=inputs,
        output_directory=directory,
        cache_policy="rebuild",
    )
    manifest_path = prepared.paths["departure_time_samples"] / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))

    def remove_new_field(value: object) -> bool:
        if isinstance(value, list):
            return any(remove_new_field(item) for item in value)
        if not isinstance(value, dict):
            return False
        if value.get("__dataclass__", "").endswith(":SampledJourneyCellDiagnostics"):
            value["fields"].pop("cell_status")
            return True
        return any(remove_new_field(item) for item in value.values())

    assert remove_new_field(document["payload"])
    document.pop("content_fingerprint")
    document["content_fingerprint"] = hashlib.sha256(
        canonical_json(document).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(canonical_json(document), encoding="utf-8")

    rebuilt = prepare_reduced_od_artifacts(
        scenario=_scenario(),
        measurements=_measurements(),
        configuration=config,
        inputs=inputs,
        output_directory=directory,
        cache_policy="reuse_or_build",
    )
    statuses = {item["phase"]: item["status"] for item in rebuilt.phase_diagnostics}
    assert statuses["timetable_index"] == "reused"
    assert statuses["departure_time_samples"] == "built"
    assert load_reduced_od_artifacts(
        configuration=config,
        artifact_directory=directory,
        expected_departure_sampling_fingerprint=_departure_sampling_identity(inputs),
    ).departure_sampling_cells


def test_production_and_journey_changes_rebuild_only_dependencies(
    tmp_path: Path,
) -> None:
    config, directory, _ = _prepare(tmp_path)
    common = dict(
        departure_seconds_by_origin={"A": (28800,)},
        destination_attractiveness={
            ("B", "P"): 1.0,
            ("C", "P"): 1.0,
            ("D", "P"): 1.0,
        },
        footpaths=(Footpath("B", "C", 120),),
        time_periods=(JourneyTimePeriod("P", 0, 108000),),
    )
    production_change = prepare_reduced_od_artifacts(
        scenario=_scenario(),
        measurements=_measurements(),
        configuration=config,
        inputs=ReducedODPreparationInputs(
            production_inputs={("A", "P"): 41.0}, **common
        ),
        output_directory=directory,
        cache_policy="reuse_or_build",
    )
    statuses = {
        item["phase"]: item["status"] for item in production_change.phase_diagnostics
    }
    assert statuses["production_inputs"] == "rebuilt"
    assert statuses["conditional_gravity_features"] == "rebuilt"
    assert statuses["journey_choices"] == "reused"
    assert statuses["reduced_response_operator"] == "reused"

    journey_change_config = replace(
        config,
        journeys=replace(config.journeys, maximum_transfers=0),
    )
    journey_change = prepare_reduced_od_artifacts(
        scenario=_scenario(),
        measurements=_measurements(),
        configuration=journey_change_config,
        inputs=ReducedODPreparationInputs(
            production_inputs={("A", "P"): 41.0}, **common
        ),
        output_directory=directory,
        cache_policy="reuse_or_build",
    )
    statuses = {
        item["phase"]: item["status"] for item in journey_change.phase_diagnostics
    }
    assert statuses["physical_stops"] == "reused"
    assert statuses["timetable_index"] == "reused"
    assert statuses["journey_choices"] == "built"
    assert statuses["measurement_response"] == "built"
