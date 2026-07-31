"""Compare fixed-routing linear least-squares choices for simple example 02."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario, read_fixed_demand_csv
from public_transportation.inference.assignment_adapter import (
    assign_link_flow_fixed_routing,
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.fixed_routing_linear_dense_solver import (
    DenseReferenceResult,
)
from public_transportation.inference.fixed_routing_linear_objective import (
    predict_linear_measurements,
)
from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    build_fixed_routing_linear_problem_from_operator,
)
from public_transportation.inference.fixed_routing_linear_quality import (
    LinearEstimateQuality,
)
from public_transportation.inference.fixed_routing_linear_recommendation import (
    RegularizationRecommendation,
    recommend_linear_regularization,
)
from public_transportation.inference.fixed_routing_linear_trf_solver import (
    TRFLSMRConfig,
    TRFLSMRResult,
)
from public_transportation.inference.fixed_routing_linear_workflow import (
    FixedRoutingLinearEstimationConfig,
    run_fixed_routing_linear_estimation,
)
from public_transportation.inference.fixed_routing_linear_validation import (
    ForwardEquivalenceValidation,
    NoiseFreeRecoveryValidation,
    validate_fixed_routing_forward_equivalence,
    validate_noise_free_linear_recovery,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    load_or_prepare_fixed_routing_measurement_operator,
    prepare_fixed_routing_measurement_operator,
)
from public_transportation.inference.od_parameter_layout import (
    build_od_parameter_layout,
)
from public_transportation.measurement import (
    build_mapping_spec_strict,
    read_measurements_csv,
)
from public_transportation.measurement.likelihood_jax import (
    predict_measurements_from_link_flow,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREPROCESSING_RESULTS = ROOT / "pre_processing" / "results"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "linear_fixed_routing"
DEFAULT_OPERATOR_CACHE = Path(__file__).resolve().parent / "results" / "operator_cache"
DEFAULT_VALIDATION_REPORT = (
    Path(__file__).resolve().parent / "results" / "linear_model_validation.json"
)
NETWORK_FILES = (
    "metadata.json",
    "stops.csv",
    "lines.csv",
    "trips.csv",
    "stop_times.csv",
    "time_bins.csv",
)
FIXED_ROUTING_PARAMETER = 1.0
REGULARIZATION_STRENGTH = 1.0
CONFIGURATION_NAMES = ("none", "ridge_to_prior", "scaled_ridge_to_prior")


@dataclass(frozen=True, slots=True)
class LinearConfigurationRun:
    name: str
    problem: FixedRoutingLinearProblem
    dense_result: DenseReferenceResult
    iterative_result: TRFLSMRResult
    quality: LinearEstimateQuality


@dataclass(frozen=True, slots=True)
class LinearExample02Run:
    base_problem: FixedRoutingLinearProblem
    recommendation: RegularizationRecommendation
    configurations: tuple[LinearConfigurationRun, ...]
    operator_validation_max_abs_difference: float
    operator_cache_hit: bool
    output_path: Path


@dataclass(frozen=True, slots=True)
class LinearExample02ValidationRun:
    forward: ForwardEquivalenceValidation
    recovery: NoiseFreeRecoveryValidation
    report: dict[str, object]
    report_path: Path


def _prepare_scenario() -> tuple[tempfile.TemporaryDirectory[str], Scenario]:
    temporary = tempfile.TemporaryDirectory(prefix="pt-linear-example-02-")
    directory = Path(temporary.name)
    for name in NETWORK_FILES:
        shutil.copy2(DATA / name, directory / name)
    shutil.copy2(PREPROCESSING_RESULTS / "demand.csv", directory / "demand.csv")
    return temporary, Scenario.from_folder(directory, strict=True)


def _active_demand(compact_layout, free_demand: np.ndarray) -> jnp.ndarray:
    active = jnp.zeros((compact_layout.num_active,), dtype=jnp.float32)
    active = active.at[jnp.asarray(compact_layout.free_compact_indices)].set(
        jnp.asarray(free_demand, dtype=jnp.float32)
    )
    active = active.at[jnp.asarray(compact_layout.fixed_compact_indices)].set(
        jnp.asarray(compact_layout.fixed_compact_values, dtype=jnp.float32)
    )
    return active


def _true_full_demand(od_layout) -> np.ndarray:
    with (DATA / "true_demand.csv").open(newline="", encoding="utf-8") as stream:
        values = {
            (row["origin_stop_id"], row["dest_stop_id"], row["time_bin_id"]): float(
                row["flow"]
            )
            for row in csv.DictReader(stream)
        }
    missing = [key for key in od_layout.od_keys if key not in values]
    if missing:
        raise ValueError(f"true_demand.csv is missing OD keys: {missing!r}")
    return np.asarray([values[key] for key in od_layout.od_keys], dtype=np.float64)


def prepare_linear_problem(
    *,
    chunk_size: int = 16,
    operator_cache_directory: Path | None = DEFAULT_OPERATOR_CACHE,
) -> tuple[FixedRoutingLinearProblem, float, bool]:
    """Prepare and independently validate the example's common dense operator."""
    temporary, scenario = _prepare_scenario()
    try:
        fixed_demand = read_fixed_demand_csv(
            DATA / "fixed_demand.csv", scenario=scenario
        )
        od_layout = build_od_parameter_layout(
            scenario=scenario, fixed_demand=fixed_demand
        )
        compact_layout = build_compact_od_assignment_layout(parameter_layout=od_layout)
        artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
        id_manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)
        mapping = build_mapping_spec_strict(
            id_manager=id_manager,
            table=read_measurements_csv(
                PREPROCESSING_RESULTS / "measurements_boarding_alighting.csv"
            ),
            include_link_lists_for_report=False,
        )
        inputs = build_assignment_inputs(
            artifacts=artifacts, compact_layout=compact_layout
        )
        routing = prepare_fixed_routing(inputs=inputs, theta=FIXED_ROUTING_PARAMETER)
        operator_kwargs = dict(
            inputs=inputs,
            routing=routing,
            spec=mapping.spec,
            assignment_fingerprint=str(id_manager.fingerprint),
            compact_layout=compact_layout,
            od_layout_fingerprint=od_layout.fingerprint,
            representation="bcoo",
            chunk_size=chunk_size,
        )
        operator = (
            prepare_fixed_routing_measurement_operator(**operator_kwargs)
            if operator_cache_directory is None
            else load_or_prepare_fixed_routing_measurement_operator(
                cache_directory=operator_cache_directory,
                **operator_kwargs,
            )
        )
        prior = np.asarray(od_layout.free_baseline_values, dtype=np.float64)
        problem = build_fixed_routing_linear_problem_from_operator(
            operator=operator,
            observations=np.asarray(mapping.y_obs),
            observation_weights=np.ones(mapping.spec.num_measurements),
            prior_demand=prior,
            lower_bounds=np.zeros(od_layout.num_free),
            upper_bounds=np.full(od_layout.num_free, np.inf),
            variable_scales=np.maximum(prior, 1.0),
            regularization_selection="unspecified",
            free_od_indices=np.asarray(od_layout.free_od_indices),
        )

        link_flow = assign_link_flow_fixed_routing(
            inputs=inputs,
            routing=routing,
            f=_active_demand(compact_layout, prior),
        )
        assignment_prediction = np.asarray(
            predict_measurements_from_link_flow(
                link_flow,
                spec_num_measurements=mapping.spec.num_measurements,
                spec_measurement_index=jnp.asarray(mapping.spec.measurement_index),
                spec_link_index=jnp.asarray(mapping.spec.link_index),
            )
        )
        linear_prediction = predict_linear_measurements(problem, prior)
        difference = float(
            np.max(np.abs(linear_prediction - assignment_prediction), initial=0.0)
        )
        if not np.allclose(
            linear_prediction, assignment_prediction, rtol=4.0e-5, atol=4.0e-5
        ):
            raise RuntimeError(
                "Dense linear operator does not match fixed-routing assignment; "
                f"maximum absolute difference is {difference:.6g}."
            )
        return problem, difference, operator.metrics.cache_hit
    finally:
        temporary.cleanup()


def run_linear_model_validation(
    *,
    report_path: Path = DEFAULT_VALIDATION_REPORT,
    chunk_size: int = 16,
    operator_cache_directory: Path | None = DEFAULT_OPERATOR_CACHE,
    random_seed: int = 20260729,
) -> LinearExample02ValidationRun:
    """Audit fixture consistency, the forward map, and noise-free recovery."""
    problem, _, _ = prepare_linear_problem(
        chunk_size=chunk_size,
        operator_cache_directory=operator_cache_directory,
    )
    temporary, scenario = _prepare_scenario()
    try:
        fixed = read_fixed_demand_csv(DATA / "fixed_demand.csv", scenario=scenario)
        layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed)
        compact = build_compact_od_assignment_layout(parameter_layout=layout)
        artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
        id_manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)
        mapping = build_mapping_spec_strict(
            id_manager=id_manager,
            table=read_measurements_csv(
                PREPROCESSING_RESULTS / "measurements_boarding_alighting.csv"
            ),
            include_link_lists_for_report=False,
        )
        inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
        routing = prepare_fixed_routing(inputs=inputs, theta=FIXED_ROUTING_PARAMETER)

        def assignment_prediction(free_demand: np.ndarray) -> np.ndarray:
            link_flow = assign_link_flow_fixed_routing(
                inputs=inputs,
                routing=routing,
                f=_active_demand(compact, free_demand),
            )
            return np.asarray(
                predict_measurements_from_link_flow(
                    link_flow,
                    spec_num_measurements=mapping.spec.num_measurements,
                    spec_measurement_index=jnp.asarray(mapping.spec.measurement_index),
                    spec_link_index=jnp.asarray(mapping.spec.link_index),
                )
            )

        full_truth = _true_full_demand(layout)
        free_truth = full_truth[np.asarray(layout.free_od_indices)]
        fixed_indices = np.asarray(layout.fixed_od_indices)
        fixed_values = np.asarray(layout.fixed_od_values)
        fixed_truth = full_truth[fixed_indices]
        rng = np.random.default_rng(random_seed)
        forward = validate_fixed_routing_forward_equivalence(
            problem,
            {
                "zero": np.zeros(problem.num_free_od),
                "prior": problem.prior_demand,
                "true": free_truth,
                "seeded_random": rng.uniform(0.0, 2.0)
                * np.maximum(problem.prior_demand, 1.0),
            },
            assignment_prediction,
        )
        recovery = validate_noise_free_linear_recovery(problem, free_truth)
        actual_truth_residual = predict_linear_measurements(problem, free_truth) - (
            problem.observations
        )
        report: dict[str, object] = {
            "schema_version": 1,
            "example": "simple_example_02",
            "routing_parameter": FIXED_ROUTING_PARAMETER,
            "fixed_constraints_match_generating_truth": bool(
                np.allclose(fixed_values, fixed_truth)
            ),
            "fixed_od_constraints": [
                {
                    "od_index": int(index),
                    "od_key": list(layout.od_keys[int(index)]),
                    "fixed_value": float(value),
                    "generating_true_value": float(true_value),
                }
                for index, value, true_value in zip(
                    fixed_indices, fixed_values, fixed_truth, strict=True
                )
            ],
            "forward_equivalence": {
                "passed": forward.passed,
                "worst_max_abs_difference": forward.worst_max_abs_difference,
                "cases": [
                    {
                        "name": case.name,
                        "max_abs_difference": case.max_abs_difference,
                    }
                    for case in forward.cases
                ],
            },
            "noise_free_recovery": {
                "solver_success": recovery.result.success,
                "measurement_rank": recovery.measurement_rank,
                "measurement_nullity": recovery.measurement_nullity,
                "measurement_residual_inf_norm": (
                    recovery.measurement_residual_inf_norm
                ),
                "estimation_error_norm": recovery.estimation_error_norm,
                "identifiable_error_norm": recovery.identifiable_error_norm,
                "null_space_error_norm": recovery.null_space_error_norm,
            },
            "actual_observations_at_true_demand": {
                "residual_inf_norm": float(
                    np.max(np.abs(actual_truth_residual), initial=0.0)
                ),
                "residual_rmse": float(
                    np.sqrt(np.mean(np.square(actual_truth_residual)))
                ),
            },
        }
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return LinearExample02ValidationRun(
            forward=forward,
            recovery=recovery,
            report=report,
            report_path=report_path,
        )
    finally:
        temporary.cleanup()


def _solve_configuration(
    base: FixedRoutingLinearProblem,
    *,
    name: str,
    strength: float,
    output_path: Path,
) -> LinearConfigurationRun:
    workflow = run_fixed_routing_linear_estimation(
        base,
        config=FixedRoutingLinearEstimationConfig(
            regularization=name,
            regularization_strength=None if name == "none" else strength,
            trf_lsmr=TRFLSMRConfig(
                tolerance=1.0e-9,
                lsmr_tolerance=1.0e-11,
                active_tolerance=1.0e-6,
            ),
            quality_active_tolerance=1.0e-6,
            verification_relative_tolerance=3.0e-4,
            verification_absolute_tolerance=3.0e-4,
        ),
        output_path=output_path,
    )
    return LinearConfigurationRun(
        name=name,
        problem=workflow.problem,
        dense_result=workflow.dense_reference,
        iterative_result=workflow.iterative_result,
        quality=workflow.quality,
    )


def run_linear_fixed_routing(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    regularization_strength: float = REGULARIZATION_STRENGTH,
    chunk_size: int = 16,
    operator_cache_directory: Path | None = DEFAULT_OPERATOR_CACHE,
) -> LinearExample02Run:
    """Run all three explicit regularization configurations."""
    base, validation_difference, cache_hit = prepare_linear_problem(
        chunk_size=chunk_size,
        operator_cache_directory=operator_cache_directory,
    )
    recommendation = recommend_linear_regularization(base)
    output_path = Path(output_path)
    configurations = tuple(
        _solve_configuration(
            base,
            name=name,
            strength=regularization_strength,
            output_path=output_path / f"{name}.npz",
        )
        for name in CONFIGURATION_NAMES
    )
    run = LinearExample02Run(
        base_problem=base,
        recommendation=recommendation,
        configurations=configurations,
        operator_validation_max_abs_difference=validation_difference,
        operator_cache_hit=cache_hit,
        output_path=output_path,
    )
    return run


def main() -> None:
    run = run_linear_fixed_routing()
    validation = run_linear_model_validation()
    print("Fixed-routing linear least squares — simple example 02")
    print(f"regularization recommendation: {run.recommendation.status}")
    print(f"operator cache hit: {run.operator_cache_hit}")
    print()
    print(
        "configuration              objective   data residual   active bounds   data df"
    )
    for configuration in run.configurations:
        result = configuration.iterative_result
        quality = configuration.quality
        active = int(
            np.count_nonzero(result.kkt.lower_active | result.kkt.upper_active)
        )
        print(
            f"{configuration.name:27s} "
            f"{result.evaluation.objective:10.4f} "
            f"{np.linalg.norm(result.evaluation.data_fit.raw_residual):15.4f} "
            f"{active:13d} "
            f"{quality.effective_data_degrees_of_freedom:9.3f}"
        )
    print()
    print(f"saved results: {run.output_path}")
    print(f"saved validation report: {validation.report_path}")


if __name__ == "__main__":
    main()
