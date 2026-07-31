"""Run fixed-routing linear least squares for simple example 01.

The script constructs and validates the dense measurement operator, reports a
nonbinding regularization recommendation, explicitly selects no regularization
for the full-rank noise-free example, and compares dense BVLS with sparse
TRF/LSMR. A separate report compares the supported regularization choices.
"""

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
)
from public_transportation.inference.fixed_routing_linear_trf_solver import (
    TRFLSMRConfig,
    TRFLSMRResult,
)
from public_transportation.inference.fixed_routing_linear_validation import (
    ForwardEquivalenceValidation,
    NoiseFreeRecoveryValidation,
    validate_fixed_routing_forward_equivalence,
    validate_noise_free_linear_recovery,
)
from public_transportation.inference.fixed_routing_linear_workflow import (
    FixedRoutingLinearEstimationConfig,
    FixedRoutingLinearEstimationRun,
    RegularizationChoice,
    run_fixed_routing_linear_estimation,
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
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "results" / "linear_fixed_routing_results.npz"
)
DEFAULT_OPERATOR_CACHE = Path(__file__).resolve().parent / "results" / "operator_cache"
DEFAULT_VALIDATION_REPORT = (
    Path(__file__).resolve().parent / "results" / "linear_model_validation.json"
)
DEFAULT_ACCURACY_REPORT = (
    Path(__file__).resolve().parent / "results" / "linear_accuracy_comparison.json"
)
DEFAULT_NONLINEAR_RESULTS = (
    Path(__file__).resolve().parent / "results" / "compare_vi_ml_od_theta_results.npz"
)
NETWORK_FILES = (
    "metadata.json",
    "stops.csv",
    "lines.csv",
    "trips.csv",
    "stop_times.csv",
    "time_bins.csv",
)
FIXED_ROUTING_PARAMETER = 5.0
REGULARIZATION_STRENGTH = 1.0


@dataclass(frozen=True, slots=True)
class LinearExample01Run:
    problem: FixedRoutingLinearProblem
    recommendation: RegularizationRecommendation
    dense_result: DenseReferenceResult
    iterative_result: TRFLSMRResult
    quality: LinearEstimateQuality
    operator_validation_max_abs_difference: float
    operator_cache_hit: bool
    output_path: Path


@dataclass(frozen=True, slots=True)
class LinearExample01ValidationRun:
    """Controlled forward and noise-free validation for the linear model."""

    forward: ForwardEquivalenceValidation
    recovery: NoiseFreeRecoveryValidation
    report_path: Path


@dataclass(frozen=True, slots=True)
class LinearExample01AccuracyRun:
    """Fits and metrics explaining accuracy on the actual observations."""

    linear_runs: tuple[tuple[str, FixedRoutingLinearEstimationRun], ...]
    report: dict[str, object]
    report_path: Path


def _prepare_scenario() -> tuple[tempfile.TemporaryDirectory[str], Scenario]:
    temporary = tempfile.TemporaryDirectory(prefix="pt-linear-example-01-")
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


def _true_free_demand(od_layout) -> np.ndarray:
    full = _true_full_demand(od_layout)
    return full[np.asarray(od_layout.free_od_indices)]


def _write_validation_report(
    path: Path,
    *,
    forward: ForwardEquivalenceValidation,
    recovery: NoiseFreeRecoveryValidation,
) -> None:
    report = {
        "schema_version": 1,
        "example": "simple_example_01",
        "routing_parameter": recovery.problem.provenance.routing_parameter,
        "forward_equivalence": {
            "passed": forward.passed,
            "absolute_tolerance": forward.absolute_tolerance,
            "relative_tolerance": forward.relative_tolerance,
            "worst_max_abs_difference": forward.worst_max_abs_difference,
            "cases": [
                {
                    "name": case.name,
                    "max_abs_difference": case.max_abs_difference,
                    "rms_difference": case.rms_difference,
                }
                for case in forward.cases
            ],
        },
        "noise_free_recovery": {
            "solver_success": recovery.result.success,
            "measurement_rank": recovery.measurement_rank,
            "measurement_nullity": recovery.measurement_nullity,
            "rank_tolerance": recovery.rank_tolerance,
            "measurement_residual_inf_norm": recovery.measurement_residual_inf_norm,
            "estimation_error_norm": recovery.estimation_error_norm,
            "identifiable_error_norm": recovery.identifiable_error_norm,
            "null_space_error_norm": recovery.null_space_error_norm,
            "projected_gradient_inf_norm": (
                recovery.result.kkt.projected_gradient_inf_norm
            ),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _accuracy_metrics(
    problem: FixedRoutingLinearProblem,
    estimate: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float]:
    prediction = predict_linear_measurements(problem, estimate)
    demand_error = np.asarray(estimate) - truth
    measurement_error = prediction - problem.observations
    return {
        "demand_mae": float(np.mean(np.abs(demand_error))),
        "demand_rmse": float(np.sqrt(np.mean(np.square(demand_error)))),
        "demand_max_abs_error": float(np.max(np.abs(demand_error), initial=0.0)),
        "measurement_mae": float(np.mean(np.abs(measurement_error))),
        "measurement_rmse": float(np.sqrt(np.mean(np.square(measurement_error)))),
        "measurement_max_abs_error": float(
            np.max(np.abs(measurement_error), initial=0.0)
        ),
        "least_squares_data_objective": float(
            0.5
            * np.vdot(
                np.sqrt(problem.observation_weights) * measurement_error,
                np.sqrt(problem.observation_weights) * measurement_error,
            )
        ),
        "distance_to_prior": float(np.linalg.norm(estimate - problem.prior_demand)),
    }


def prepare_linear_problem(
    *,
    chunk_size: int = 16,
    operator_cache_directory: Path | None = DEFAULT_OPERATOR_CACHE,
) -> tuple[FixedRoutingLinearProblem, float, bool]:
    """Prepare and independently validate the dense small-example problem."""
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
) -> LinearExample01ValidationRun:
    """Validate the linear map and noise-free recovery independently."""
    problem, _, _ = prepare_linear_problem(
        chunk_size=chunk_size,
        operator_cache_directory=operator_cache_directory,
    )
    temporary, scenario = _prepare_scenario()
    try:
        fixed_demand = read_fixed_demand_csv(
            DATA / "fixed_demand.csv", scenario=scenario
        )
        od_layout = build_od_parameter_layout(
            scenario=scenario, fixed_demand=fixed_demand
        )
        if od_layout.fingerprint != problem.provenance.od_layout_fingerprint:
            raise RuntimeError("validation OD layout differs from the linear problem.")
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

        def assignment_prediction(free_demand: np.ndarray) -> np.ndarray:
            link_flow = assign_link_flow_fixed_routing(
                inputs=inputs,
                routing=routing,
                f=_active_demand(compact_layout, free_demand),
            )
            return np.asarray(
                predict_measurements_from_link_flow(
                    link_flow,
                    spec_num_measurements=mapping.spec.num_measurements,
                    spec_measurement_index=jnp.asarray(
                        mapping.spec.measurement_index
                    ),
                    spec_link_index=jnp.asarray(mapping.spec.link_index),
                )
            )

        truth = _true_free_demand(od_layout)
        rng = np.random.default_rng(random_seed)
        random_demand = rng.uniform(0.0, 2.0) * np.maximum(
            problem.prior_demand, 1.0
        )
        forward = validate_fixed_routing_forward_equivalence(
            problem,
            {
                "zero": np.zeros(problem.num_free_od),
                "prior": problem.prior_demand,
                "true": truth,
                "seeded_random": random_demand,
            },
            assignment_prediction,
        )
        recovery = validate_noise_free_linear_recovery(problem, truth)
        report_path = Path(report_path)
        _write_validation_report(
            report_path,
            forward=forward,
            recovery=recovery,
        )
        return LinearExample01ValidationRun(
            forward=forward,
            recovery=recovery,
            report_path=report_path,
        )
    finally:
        temporary.cleanup()


def run_actual_observation_accuracy_comparison(
    *,
    report_path: Path = DEFAULT_ACCURACY_REPORT,
    nonlinear_results_path: Path | None = DEFAULT_NONLINEAR_RESULTS,
    regularization_strength: float = REGULARIZATION_STRENGTH,
    chunk_size: int = 16,
    operator_cache_directory: Path | None = DEFAULT_OPERATOR_CACHE,
) -> LinearExample01AccuracyRun:
    """Explain demand accuracy under explicit objective choices."""
    problem, _, _ = prepare_linear_problem(
        chunk_size=chunk_size,
        operator_cache_directory=operator_cache_directory,
    )
    temporary, scenario = _prepare_scenario()
    try:
        fixed_demand = read_fixed_demand_csv(
            DATA / "fixed_demand.csv", scenario=scenario
        )
        od_layout = build_od_parameter_layout(
            scenario=scenario, fixed_demand=fixed_demand
        )
        full_truth = _true_full_demand(od_layout)
        truth = full_truth[np.asarray(od_layout.free_od_indices)]
        fixed_indices = np.asarray(od_layout.fixed_od_indices)
        fixed_truth = full_truth[fixed_indices]
        fixed_values = np.asarray(od_layout.fixed_od_values)
    finally:
        temporary.cleanup()

    configurations = (
        ("none", None),
        ("ridge_to_prior", regularization_strength),
        ("scaled_ridge_to_prior", regularization_strength),
    )
    linear_runs: list[tuple[str, FixedRoutingLinearEstimationRun]] = []
    methods: dict[str, object] = {}
    for name, strength in configurations:
        run = run_fixed_routing_linear_estimation(
            problem,
            config=FixedRoutingLinearEstimationConfig(
                regularization=name,
                regularization_strength=strength,
                trf_lsmr=TRFLSMRConfig(
                    tolerance=1.0e-10,
                    lsmr_tolerance=1.0e-12,
                    active_tolerance=1.0e-7,
                ),
                quality_active_tolerance=1.0e-7,
            ),
        )
        linear_runs.append((name, run))
        metrics = _accuracy_metrics(problem, run.iterative_result.demand, truth)
        metrics.update(
            {
                "solver_success": run.iterative_result.success,
                "projected_gradient_inf_norm": (
                    run.iterative_result.kkt.projected_gradient_inf_norm
                ),
                "total_objective": run.iterative_result.evaluation.objective,
            }
        )
        methods[f"linear_{name}"] = metrics

    truth_metrics = _accuracy_metrics(problem, truth, truth)
    methods["true_demand"] = truth_metrics
    fixed_constraints_match = bool(np.allclose(fixed_values, fixed_truth))

    nonlinear_archive_compatible = False
    if nonlinear_results_path is not None and Path(nonlinear_results_path).exists():
        with np.load(nonlinear_results_path, allow_pickle=False) as archive:
            full_estimate = np.asarray(archive["ml_f_hat"], dtype=np.float64)
            nonlinear_free = full_estimate[np.asarray(problem.free_od_indices)]
            nonlinear_metrics = {
                "demand_mae": float(np.mean(np.abs(full_estimate - full_truth))),
                "demand_rmse": float(
                    np.sqrt(np.mean(np.square(full_estimate - full_truth)))
                ),
                "demand_max_abs_error": float(
                    np.max(np.abs(full_estimate - full_truth), initial=0.0)
                ),
                "current_free_od_demand_rmse": float(
                    np.sqrt(np.mean(np.square(nonlinear_free - truth)))
                ),
            }
            nonlinear_archive_compatible = (
                "od_layout_fingerprint" in archive.files
                and str(archive["od_layout_fingerprint"])
                == problem.provenance.od_layout_fingerprint
            )
            nonlinear_metrics.update(
                {
                    "compatible_with_current_od_layout": nonlinear_archive_compatible,
                    "solver_success": bool(archive["ml_success"]),
                    "optimizer_message": str(archive["ml_message"]),
                    "negative_binomial_log_likelihood": float(
                        archive["ml_log_likelihood"]
                    ),
                    "negative_binomial_objective": float(
                        archive["ml_objective_value"]
                    ),
                    "runtime_seconds": float(archive["ml_time_s"]),
                    "prior_weight": float(archive["ml_prior_weight"]),
                    "fixed_theta": float(archive["fixed_theta"]),
                }
            )
        methods["nonlinear_negative_binomial_ml"] = nonlinear_metrics

    report: dict[str, object] = {
        "schema_version": 1,
        "example": "simple_example_01",
        "measurement_generation": {
            "source": "deterministic fixed-routing assignment of true demand",
            "added_observation_noise": False,
            "routing_parameter": problem.provenance.routing_parameter,
        },
        "problem": {
            "num_measurements": problem.num_measurements,
            "num_free_od": problem.num_free_od,
            "regularization_strength_for_regularized_runs": regularization_strength,
            "fixed_od_constraints": [
                {
                    "od_index": int(index),
                    "od_key": list(od_layout.od_keys[int(index)]),
                    "fixed_value": float(fixed_value),
                    "generating_true_value": float(true_value),
                    "difference": float(fixed_value - true_value),
                }
                for index, fixed_value, true_value in zip(
                    fixed_indices, fixed_values, fixed_truth, strict=True
                )
            ],
            "fixed_constraints_match_generating_truth": fixed_constraints_match,
        },
        "methods": methods,
        "conclusion": (
            (
                "The fixed-OD constraints match the generating demand. The "
                "unregularized linear model therefore provides the correctness "
                "reference; regularization intentionally trades fit for proximity "
                "to the prior."
            )
            if fixed_constraints_match
            else (
                "The observations are deterministic, but the current fixed-OD "
                "constraints do not match the demand used to generate them. The "
                "constrained model therefore cannot reproduce the generating model."
            )
        ),
    }
    if nonlinear_results_path is not None and Path(nonlinear_results_path).exists():
        report["nonlinear_comparison"] = {
            "archive_path": str(nonlinear_results_path),
            "compatible_with_current_od_layout": nonlinear_archive_compatible,
            "interpretation": (
                "The saved nonlinear archive predates the current OD-layout metadata "
                "and is not a controlled comparison with the current fixed-demand "
                "model. Its demand error is reported for historical context only."
            ),
        }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return LinearExample01AccuracyRun(
        linear_runs=tuple(linear_runs),
        report=report,
        report_path=report_path,
    )


def run_linear_fixed_routing(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    regularization: RegularizationChoice = "none",
    regularization_strength: float = REGULARIZATION_STRENGTH,
    chunk_size: int = 16,
    operator_cache_directory: Path | None = DEFAULT_OPERATOR_CACHE,
) -> LinearExample01Run:
    """Run dense and sparse solvers and persist their common diagnostics."""
    problem, validation_difference, cache_hit = prepare_linear_problem(
        chunk_size=chunk_size,
        operator_cache_directory=operator_cache_directory,
    )
    workflow = run_fixed_routing_linear_estimation(
        problem,
        config=FixedRoutingLinearEstimationConfig(
            regularization=regularization,
            regularization_strength=(
                None if regularization == "none" else regularization_strength
            ),
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
    return LinearExample01Run(
        problem=workflow.problem,
        recommendation=workflow.recommendation,
        dense_result=workflow.dense_reference,
        iterative_result=workflow.iterative_result,
        quality=workflow.quality,
        operator_validation_max_abs_difference=validation_difference,
        operator_cache_hit=cache_hit,
        output_path=Path(output_path),
    )


def main() -> None:
    result = run_linear_fixed_routing()
    validation = run_linear_model_validation()
    accuracy = run_actual_observation_accuracy_comparison()
    print("Fixed-routing linear least squares — simple example 01")
    print(f"regularization recommendation: {result.recommendation.status}")
    print("explicit selection: none")
    print(f"objective: {result.iterative_result.evaluation.objective:.6g}")
    print(
        "projected-gradient KKT residual: "
        f"{result.iterative_result.kkt.projected_gradient_inf_norm:.6g}"
    )
    print(f"measurement rank: {result.quality.measurement_rank}")
    print(f"measurement nullity: {result.quality.measurement_nullity}")
    print(f"operator cache hit: {result.operator_cache_hit}")
    print(f"saved results: {result.output_path}")
    print(f"forward validation passed: {validation.forward.passed}")
    print(
        "noise-free identifiable error norm: "
        f"{validation.recovery.identifiable_error_norm:.6g}"
    )
    print(f"saved validation report: {validation.report_path}")
    print(f"saved accuracy comparison: {accuracy.report_path}")


if __name__ == "__main__":
    main()
