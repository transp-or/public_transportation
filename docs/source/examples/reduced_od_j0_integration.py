"""Complete public synthetic reduced-OD J0 integration example."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from public_transportation.domain import (
    Metadata, ODDemand, Scenario, Stop, StopTime, TimeBin, TimeOfDay, Timetable, Trip,
)
from public_transportation.domain.line import Line
from public_transportation.inference.reduced_od import (
    GaussianRawParameterPrior, JourneyODTimeKey, MinimalGravitySpecification,
    ReducedODAdequacyConfig, ReducedODFitConfig, ReducedODHoldoutConfig,
    ReducedODNamedRawParameterBounds,
    ReducedODMeasurementMetadata, ReducedODPreparationInputs, ReducedODProblemContract,
    benchmark_minimal_gravity_objective, build_minimal_gravity_problem,
    build_reduced_od_holdout_split, default_minimal_gravity_raw_parameters,
    compare_reduced_od_likelihoods,
    diagnose_reduced_od_adequacy, estimate_minimal_gravity,
    evaluate_minimal_gravity_objective, load_reduced_od_artifacts,
    load_reduced_od_checkpoint, preflight_reduced_od_j0,
    prepare_reduced_od_artifacts, recommend_reduced_od_relaxations,
    reconstruct_full_od, run_reduced_od_prior_sensitivity,
    validate_reduced_od_holdout,
)
from public_transportation.measurement import MeasurementRecord, MeasurementTable, MeasurementType
from public_transportation.preprocessing.reduced_od import JourneyTimePeriod, load_reduced_od_config


def synthetic_inputs() -> tuple[Scenario, MeasurementTable]:
    stops = [Stop(key, key, 46.0 + index * 0.001, 6.0) for index, key in enumerate("ABCD")]
    schedules = {
        "AB": (("A", 28800), ("B", 29400)),
        "AC": (("A", 28920), ("C", 29820)),
        "AD": (("A", 29040), ("D", 30540)),
    }
    trips = [Trip(key, key, service_id="day", direction_id=0) for key in schedules]
    stop_times = [
        StopTime(trip, stop, sequence, seconds, seconds)
        for trip, rows in schedules.items()
        for sequence, (stop, seconds) in enumerate(rows, start=1)
    ]
    scenario = Scenario(
        metadata=Metadata(title="Reduced OD J0", created_at="2026-01-01T00:00:00"),
        stops=stops,
        lines=[Line(key) for key in schedules],
        time_bins=[TimeBin("P", TimeOfDay(0), TimeOfDay(108000))],
        demand=ODDemand(records=[]),
        timetable=Timetable(trips=trips, stop_times=stop_times),
    )
    records = []
    for index, (trip, rows) in enumerate(schedules.items(), start=1):
        records.extend((
            MeasurementRecord("synthetic", MeasurementType.BOARDING, "A", TimeOfDay(rows[0][1]), 20.0 + index, trip_id=trip),
            MeasurementRecord("synthetic", MeasurementType.ALIGHTING, rows[1][0], TimeOfDay(rows[1][1]), 19.0 + index, trip_id=trip),
        ))
    return scenario, MeasurementTable.from_records(records)


def write_configuration(path: Path) -> None:
    path.write_text("""schema_version = 2
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
maximum_transfers = 0
maximum_waiting_seconds = 3600
maximum_journey_seconds = 7200
maximum_alternatives_per_cell = 4
transfer_footpath_policy = "none-v1"
route_shares = "fixed_within_fit"
[productions]
mode = "estimated_basis"
semantics = "estimated_production_basis"
basis = "origin_period"
[stops]
mapping_policy = "identity"
[outputs]
spatial_level = "physical_stop"
reconstruct_full_od = false
[model]
likelihood = "poisson"
[validation]
detailed_assignment = "explicit_only"
""", encoding="utf-8")


def main(output_directory: Path | None = None) -> dict[str, object]:
    root = output_directory or Path(tempfile.mkdtemp(prefix="reduced-od-j0-"))
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "reduced_od.toml"
    write_configuration(config_path)
    configuration = load_reduced_od_config(config_path)
    scenario, measurements = synthetic_inputs()
    artifact_directory = root / "artifacts"
    prepared = prepare_reduced_od_artifacts(
        scenario=scenario,
        measurements=measurements,
        configuration=configuration,
        inputs=ReducedODPreparationInputs(
            departure_seconds_by_origin={"A": (28800,)},
            production_inputs={("A", "P"): 60.0},
            destination_attractiveness={(key, "P"): 1.0 for key in "BCD"},
            time_periods=(JourneyTimePeriod("P", 0, 108000),),
        ),
        output_directory=artifact_directory,
        cache_policy="reuse_or_build",
    )
    loaded = load_reduced_od_artifacts(configuration=configuration, artifact_directory=artifact_directory)
    production_basis = np.ones((loaded.features.number_of_origin_time_groups, 1))
    specification = MinimalGravitySpecification(
        likelihood="poisson",
        production_mode="estimated_basis",
        production_basis_columns=1,
    )
    preflight = preflight_reduced_od_j0(
        configuration=configuration,
        artifact_directory=artifact_directory,
        specification=specification,
        production_basis=production_basis,
    )
    built = build_minimal_gravity_problem(
        artifacts=loaded,
        specification=specification,
        production_basis=production_basis,
        production_basis_labels=("global_log_scale",),
    )
    initial = default_minimal_gravity_raw_parameters(built.problem.parameter_layout)
    timing = benchmark_minimal_gravity_objective(problem=built.problem, raw_parameters=initial)

    checkpoint_path = root / "j0-ml.checkpoint.json"
    ml = estimate_minimal_gravity(
        problem=built.problem, initial_raw_parameters=initial,
        model_fingerprint=built.model_fingerprint, checkpoint_path=checkpoint_path,
    )
    load_reduced_od_checkpoint(checkpoint_path, expected_manifest=ml.manifest)
    flat_map = estimate_minimal_gravity(
        problem=built.problem, initial_raw_parameters=initial,
        model_fingerprint=built.model_fingerprint,
        config=ReducedODFitConfig(method="map"),
        prior=GaussianRawParameterPrior(np.zeros(initial.size), np.full(initial.size, np.inf)),
    )
    informative_map = estimate_minimal_gravity(
        problem=built.problem, initial_raw_parameters=initial,
        model_fingerprint=built.model_fingerprint,
        config=ReducedODFitConfig(
            method="map", gradient_tolerance=1.0e-5, function_tolerance=1.0e-8
        ),
        prior=GaussianRawParameterPrior(np.zeros(initial.size), np.full(initial.size, 2.0)),
    )
    likelihood_comparison = compare_reduced_od_likelihoods(
        problem=built.problem,
        initial_raw_parameters={
            "poisson": initial,
            "negative_binomial": np.insert(initial, 2, 3.0),
        },
        artifact_fingerprint=loaded.fingerprints["reduced_response_operator"],
        fit_configs={
            "poisson": ReducedODFitConfig(
                named_raw_parameter_bounds=ReducedODNamedRawParameterBounds(
                    {
                        "beta_time": (-10.0, 10.0),
                        "beta_transfer": (-10.0, 10.0),
                        "global_log_scale": (-3.0, 3.0),
                    }
                )
            ),
            "negative_binomial": ReducedODFitConfig(
                named_raw_parameter_bounds=ReducedODNamedRawParameterBounds(
                    {
                        "beta_time": (-10.0, 10.0),
                        "beta_transfer": (-10.0, 10.0),
                        "dispersion": (-5.0, 60.0),
                        "global_log_scale": (-3.0, 3.0),
                    }
                )
            ),
        },
    )
    prior_sensitivity = run_reduced_od_prior_sensitivity(
        problem=built.problem,
        initial_raw_parameters=initial,
        model_fingerprint=built.model_fingerprint,
        scenarios={
            "weak": GaussianRawParameterPrior(
                np.zeros(initial.size), np.full(initial.size, 100.0)
            ),
            "moderate": GaussianRawParameterPrior(
                np.zeros(initial.size), np.full(initial.size, 2.0)
            ),
        },
        fit_config=ReducedODFitConfig(),
    )
    resolved = loaded.measurement_response.resolved_measurements
    metadata = ReducedODMeasurementMetadata(
        number_of_measurements=len(resolved),
        measurement_type=np.asarray([item.measurement_type.value for item in resolved]),
        line=np.asarray([item.line_id for item in resolved]),
        stop=np.asarray([item.physical_stop_id for item in resolved]),
        time_period=np.asarray(["P"] * len(resolved)),
        vehicle_journey=np.asarray([item.trip_id for item in resolved]),
        destination_zone=np.asarray([item.physical_stop_id for item in resolved]),
    )
    adequacy = diagnose_reduced_od_adequacy(fit=ml, problem=built.problem, metadata=metadata)
    split = build_reduced_od_holdout_split(
        metadata=metadata,
        measurement_identity=loaded.measurement_response.measurement_fingerprint,
        config=ReducedODHoldoutConfig(unit="vehicle_journey", fraction=1 / 3, seed=7),
    )
    holdout = validate_reduced_od_holdout(
        problem=built.problem, initial_raw_parameters=ml.raw_parameters,
        model_fingerprint=built.model_fingerprint, metadata=metadata, split=split,
    )
    advice = recommend_reduced_od_relaxations(
        adequacy=adequacy, metadata=metadata, config=ReducedODAdequacyConfig()
    )
    free_keys = loaded.measurement_response.free_cell_keys
    fixed_items = tuple(sorted(loaded.fixed_demand.items()))
    all_keys = tuple(sorted(
        [JourneyODTimeKey(*key.tuple) for key in free_keys]
        + [JourneyODTimeKey(*key.tuple) for key, _ in fixed_items]
    ))
    free_external = {JourneyODTimeKey(*key.tuple) for key in free_keys}
    free_indices = np.asarray(
        [index for index, key in enumerate(all_keys) if key in free_external],
        dtype=np.int64,
    )
    fixed_indices = np.asarray(
        [index for index, key in enumerate(all_keys) if key not in free_external],
        dtype=np.int64,
    )
    fixed_by_key = {JourneyODTimeKey(*key.tuple): value for key, value in fixed_items}
    contract = ReducedODProblemContract(
        configuration_fingerprint=configuration.fingerprint,
        timetable_artifact_fingerprint=loaded.fingerprints["timetable_index"],
        response_artifact_fingerprint=loaded.fingerprints["measurement_response"],
        od_keys=all_keys, free_od_indices=free_indices, fixed_od_indices=fixed_indices,
        fixed_od_values=np.asarray(
            [fixed_by_key[all_keys[index]] for index in fixed_indices],
            dtype=np.float64,
        ),
    )
    fitted_demand = evaluate_minimal_gravity_objective(ml.raw_parameters, problem=built.problem).demand
    reconstructed = reconstruct_full_od(contract=contract, free_cell_keys=free_keys, free_demand=fitted_demand)
    report = {
        "output_directory": str(root), "prepared": prepared.to_dict(),
        "preflight": preflight, "benchmark": timing.to_dict(),
        "ml_status": ml.status, "flat_map_matches_ml": bool(np.array_equal(flat_map.raw_parameters, ml.raw_parameters)),
        "informative_map_status": informative_map.status,
        "numerically_converged": ml.success,
        "optimizer_success": ml.optimizer_success,
        "production_total": (
            None if ml.production is None else ml.production.fitted_total
        ),
        "likelihood_comparison": [
            {
                "likelihood": item.likelihood,
                "numerically_converged": item.fit.success,
            }
            for item in likelihood_comparison.entries
        ],
        "prior_sensitivity": [
            {
                "scenario": item.scenario,
                "distance_from_ml": item.distance_from_ml,
            }
            for item in prior_sensitivity.scenarios
        ],
        "adequacy_rmse": adequacy.rmse, "holdout_rmse": holdout.holdout.rmse,
        "recommendations": [item.stage for item in advice.candidates],
        "reconstructed_cells": len(reconstructed.rows),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
