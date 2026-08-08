from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import jax

from public_transportation.inference.reduced_od import (
    ConditionalGravityFeatures,
    DetailedAssignmentOutput,
    GaussianRawParameterPrior,
    JourneyODTimeKey,
    MinimalGravityParameterLayout,
    MinimalGravityProblem,
    MinimalGravitySpecification,
    ReducedODFitConfig,
    ReducedODNamedRawParameterBounds,
    ReducedODNumericalConfig,
    ReducedODRawParameterBounds,
    ReducedODModelContract,
    ReducedODProblemContract,
    build_reduced_response_operator_from_coo,
    compare_reduced_od_likelihoods,
    default_minimal_gravity_raw_parameters,
    estimate_minimal_gravity,
    evaluate_minimal_gravity_objective,
    load_reduced_od_checkpoint,
    reconstruct_full_od,
    run_reduced_od_prior_sensitivity,
    save_reduced_od_fit_result,
    validate_with_detailed_assignment,
)
from public_transportation.preprocessing.reduced_od import ResponseCellKey


@pytest.fixture(autouse=True)
def _enable_required_precision():
    with jax.enable_x64():
        yield


def _problem() -> tuple[MinimalGravityProblem, np.ndarray, str]:
    keys = (
        ResponseCellKey("A", "B", "P"),
        ResponseCellKey("A", "C", "P"),
        ResponseCellKey("A", "D", "P"),
    )
    features = ConditionalGravityFeatures(
        cell_keys=keys,
        origin_time_group_index=np.asarray([0, 0, 0]),
        destination_index=np.asarray([0, 1, 2]),
        journey_time_seconds=np.asarray([300.0, 900.0, 1800.0]),
        transfer_count=np.asarray([0.0, 2.0, 1.0]),
        destination_attractiveness=np.asarray([1.0, 2.0, 0.7]),
        baseline_productions=np.asarray([120.0]),
        origin_time_group_keys=(("A", "P"),),
        destination_ids=("B", "C", "D"),
    )
    operator = build_reduced_response_operator_from_coo(
        number_of_measurements=3,
        number_of_free_cells=3,
        measurement_index=np.arange(3),
        free_cell_index=np.arange(3),
        response_values=np.ones(3),
        fixed_offset=np.asarray([1.0, 2.0, 3.0]),
    )
    specification = MinimalGravitySpecification()
    layout = MinimalGravityParameterLayout(specification)
    truth = default_minimal_gravity_raw_parameters(
        layout, beta_time=0.8, beta_transfer=1.2
    )
    provisional = MinimalGravityProblem(
        features=features,
        parameter_layout=layout,
        response_operator=operator,
        observations=np.ones(3),
    )
    observations = np.asarray(
        evaluate_minimal_gravity_objective(
            truth, problem=provisional
        ).measurement_mean
    )
    problem = MinimalGravityProblem(
        features=features,
        parameter_layout=layout,
        response_operator=operator,
        observations=observations,
    )
    contract = ReducedODModelContract(
        problem_fingerprint="problem",
        model_name="J0",
        production_mode="provided",
        likelihood="poisson",
        estimated_parameters=("beta_time", "beta_transfer"),
    )
    return problem, truth, contract.fingerprint


def _estimated_production_problem() -> MinimalGravityProblem:
    base, _, _ = _problem()
    features = ConditionalGravityFeatures(
        cell_keys=(
            ResponseCellKey("A", "B", "P0"),
            ResponseCellKey("A", "B", "P1"),
            ResponseCellKey("A", "C", "P0"),
            ResponseCellKey("A", "C", "P1"),
        ),
        origin_time_group_index=np.asarray([0, 1, 0, 1]),
        destination_index=np.asarray([0, 0, 1, 1]),
        journey_time_seconds=np.asarray([300.0, 360.0, 900.0, 960.0]),
        transfer_count=np.asarray([0.0, 0.0, 1.0, 1.0]),
        destination_attractiveness=np.asarray([1.0, 2.0, 1.0, 2.0]),
        baseline_productions=np.asarray([60.0, 70.0]),
        origin_time_group_keys=(("A", "P0"), ("A", "P1")),
        destination_ids=("B", "C"),
    )
    operator = build_reduced_response_operator_from_coo(
        number_of_measurements=4,
        number_of_free_cells=4,
        measurement_index=np.arange(4),
        free_cell_index=np.arange(4),
        response_values=np.ones(4),
        fixed_offset=np.zeros(4),
    )
    specification = replace(
        base.parameter_layout.specification,
        production_mode="estimated_basis",
        production_basis_columns=2,
    )
    return MinimalGravityProblem(
        features=features,
        parameter_layout=MinimalGravityParameterLayout(specification),
        response_operator=operator,
        observations=np.asarray([35.0, 25.0, 40.0, 30.0]),
        production_basis=np.eye(2),
        production_basis_labels=(
            "global_log_scale",
            "t1_log_scale_relative_to_t0",
        ),
    )


def test_float64_required_rejects_disabled_x64_before_optimization() -> None:
    problem, _, fingerprint = _problem()
    with jax.enable_x64(False):
        with pytest.raises(RuntimeError, match="JAX_ENABLE_X64=true"):
            estimate_minimal_gravity(
                problem=problem,
                initial_raw_parameters=np.zeros(2),
                model_fingerprint=fingerprint,
            )


def test_precision_diagnostics_report_compiled_float64() -> None:
    problem, _, fingerprint = _problem()
    result = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.zeros(2),
        model_fingerprint=fingerprint,
    )

    assert result.precision is not None
    assert result.precision.x64_enabled
    assert result.precision.actual_jax_input_dtype == "float64"
    assert result.precision.compiled_objective_dtype == "float64"
    assert result.precision.compiled_gradient_dtype == "float64"
    assert result.precision.tolerance_resolvable


def test_unresolved_float32_tolerance_can_warn_instead_of_fail() -> None:
    problem, _, fingerprint = _problem()
    config = ReducedODFitConfig(
        numerical=ReducedODNumericalConfig(
            precision="allow_float32",
            requested_dtype="float32",
            reject_unresolved_function_tolerance=False,
        )
    )
    with jax.enable_x64(False):
        result = estimate_minimal_gravity(
            problem=problem,
            initial_raw_parameters=np.zeros(2),
            model_fingerprint=fingerprint,
            config=config,
        )

    assert result.precision is not None
    assert not result.precision.tolerance_resolvable
    assert result.precision.warnings
    assert result.convergence is not None
    assert not result.convergence.numerically_converged


def test_optimizer_success_with_large_gradient_is_not_numerical_convergence(
    monkeypatch,
) -> None:
    problem, _, fingerprint = _problem()

    def fake_minimize(fun, x0, **kwargs):
        objective, _ = fun(np.asarray(x0))
        return SimpleNamespace(
            x=np.asarray(x0),
            fun=objective,
            nit=0,
            success=True,
            status=0,
            message="relative function reduction",
        )

    monkeypatch.setattr(
        "public_transportation.inference.reduced_od.estimator.minimize",
        fake_minimize,
    )
    result = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.asarray([10.0, -10.0]),
        model_fingerprint=fingerprint,
    )

    assert result.optimizer_success
    assert not result.success
    assert result.convergence is not None
    assert not result.convergence.stationary
    assert result.convergence.optimizer_status == 0
    assert result.convergence.optimizer_message == "relative function reduction"


def test_raw_bounds_are_validated_passed_and_reported(monkeypatch) -> None:
    problem, _, fingerprint = _problem()
    captured = {}

    def fake_minimize(fun, x0, **kwargs):
        captured["bounds"] = kwargs["bounds"]
        objective, _ = fun(np.asarray(x0))
        return SimpleNamespace(
            x=np.asarray(x0),
            fun=objective,
            nit=0,
            success=False,
            status=1,
            message="bounded test",
        )

    monkeypatch.setattr(
        "public_transportation.inference.reduced_od.estimator.minimize",
        fake_minimize,
    )
    bounds = ReducedODRawParameterBounds(
        lower=np.asarray([-1.0, -2.0]), upper=np.asarray([1.0, 2.0])
    )
    result = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.asarray([-1.0, 0.0]),
        model_fingerprint=fingerprint,
        config=ReducedODFitConfig(raw_parameter_bounds=bounds),
    )

    assert captured["bounds"] == ((-1.0, 1.0), (-2.0, 2.0))
    assert result.active_bound_parameters == ("beta_time",)
    with pytest.raises(ValueError, match="below"):
        ReducedODRawParameterBounds(
            lower=np.asarray([0.0, 1.0]), upper=np.asarray([0.0, 2.0])
        )


def test_estimated_production_basis_rank_and_summary() -> None:
    provided, _, fingerprint = _problem()
    specification = MinimalGravitySpecification(
        production_mode="estimated_basis", production_basis_columns=1
    )
    problem = replace(
        provided,
        parameter_layout=MinimalGravityParameterLayout(specification),
        production_basis=np.ones((1, 1)),
        production_basis_labels=("global_log_scale",),
    )
    result = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.zeros(3),
        model_fingerprint=fingerprint,
    )

    assert result.production is not None
    assert result.production.baseline_total == 120.0
    assert result.production.fitted_total > 0.0
    assert result.production.minimum_multiplier == pytest.approx(
        result.production.maximum_multiplier
    )
    assert result.transformed_parameters is not None
    assert result.transformed_parameters.parameter_names[-1] == "global_log_scale"

    collinear_specification = replace(specification, production_basis_columns=2)
    collinear = replace(
        provided,
        parameter_layout=MinimalGravityParameterLayout(collinear_specification),
        production_basis=np.asarray([[1.0, 2.0]]),
        production_basis_labels=("one", "two"),
    )
    with pytest.raises(ValueError, match="full column rank"):
        estimate_minimal_gravity(
            problem=collinear,
            initial_raw_parameters=np.zeros(4),
            model_fingerprint=fingerprint,
        )


def test_likelihood_comparison_and_prior_sensitivity_reuse_problem() -> None:
    problem, _, fingerprint = _problem()
    comparison = compare_reduced_od_likelihoods(
        problem=problem,
        initial_raw_parameters={
            "poisson": np.zeros(2),
            "negative_binomial": np.asarray([0.0, 0.0, 3.0]),
        },
        artifact_fingerprint="same-upstream-artifacts",
        fit_config=ReducedODFitConfig(maximum_iterations=50),
    )

    assert [entry.likelihood for entry in comparison.entries] == [
        "poisson",
        "negative_binomial",
    ]
    assert {
        entry.artifact_fingerprint for entry in comparison.entries
    } == {"same-upstream-artifacts"}
    assert comparison.entries[0].fit.transformed_parameters is not None
    assert comparison.entries[0].fit.transformed_parameters.dispersion is None
    assert comparison.entries[1].fit.transformed_parameters is not None
    assert comparison.entries[1].fit.transformed_parameters.dispersion is not None

    sensitivity_events: list[dict[str, object]] = []
    sensitivity = run_reduced_od_prior_sensitivity(
        problem=problem,
        initial_raw_parameters=np.zeros(2),
        model_fingerprint=fingerprint,
        scenarios={
            "moderate": GaussianRawParameterPrior(
                mean=np.zeros(2), scale=np.full(2, 2.0)
            ),
            "weak": GaussianRawParameterPrior(
                mean=np.zeros(2), scale=np.full(2, 100.0)
            ),
        },
        fit_config=ReducedODFitConfig(maximum_iterations=50),
        progress=lambda event: sensitivity_events.append(dict(event)),
    )
    assert [entry.scenario for entry in sensitivity.scenarios] == [
        "moderate",
        "weak",
    ]
    assert all(entry.warm_start_parent == "ml" for entry in sensitivity.scenarios)
    assert all(entry.fit.map_diagnostics is not None for entry in sensitivity.scenarios)
    assert {
        event.get("scenario")
        for event in sensitivity_events
        if event.get("scenario") is not None
    } == {"moderate", "weak"}
    assert all(
        event["completed_scenarios"] <= event["total_scenarios"]
        for event in sensitivity_events
    )


def test_likelihood_specific_named_bounds_and_progress_are_resolved() -> None:
    problem = _estimated_production_problem()
    common = {
        "beta_time": (-10.0, 10.0),
        "beta_transfer": (-10.0, 10.0),
        "global_log_scale": (-3.0, 3.0),
        "t1_log_scale_relative_to_t0": (-3.0, 3.0),
    }
    events: list[dict[str, object]] = []
    comparison = compare_reduced_od_likelihoods(
        problem=problem,
        initial_raw_parameters={
            "poisson": np.zeros(4),
            "negative_binomial": np.asarray([0.0, 0.0, 3.0, 0.0, 0.0]),
        },
        artifact_fingerprint="shared-prepared-artifact",
        fit_configs={
            "poisson": ReducedODFitConfig(
                maximum_iterations=5,
                named_raw_parameter_bounds=ReducedODNamedRawParameterBounds(common),
            ),
            "negative_binomial": ReducedODFitConfig(
                maximum_iterations=5,
                named_raw_parameter_bounds=ReducedODNamedRawParameterBounds(
                    {**common, "dispersion": (-5.0, 60.0)}
                ),
            ),
        },
        progress=lambda event: events.append(dict(event)),
    )

    poisson, negative_binomial = comparison.entries
    assert poisson.parameter_names == (
        "beta_time",
        "beta_transfer",
        "global_log_scale",
        "t1_log_scale_relative_to_t0",
    )
    assert negative_binomial.parameter_names[2] == "dispersion"
    assert poisson.initial_raw_parameters.shape == (4,)
    assert negative_binomial.initial_raw_parameters.shape == (5,)
    np.testing.assert_array_equal(
        poisson.resolved_lower_bounds[:2], negative_binomial.resolved_lower_bounds[:2]
    )
    assert {entry.artifact_fingerprint for entry in comparison.entries} == {
        "shared-prepared-artifact"
    }
    assert poisson.model_fingerprint != negative_binomial.model_fingerprint
    assert any(event.get("likelihood") == "poisson" for event in events)
    assert any(event.get("likelihood") == "negative_binomial" for event in events)
    for event in events:
        json.dumps(event)
        assert not any(isinstance(value, np.ndarray) for value in event.values())


def test_likelihood_comparison_rejects_bounds_before_compilation() -> None:
    problem = _estimated_production_problem()
    initial = {
        "poisson": np.zeros(4),
        "negative_binomial": np.zeros(5),
    }
    with pytest.raises(ValueError, match="keys must be exactly"):
        compare_reduced_od_likelihoods(
            problem=problem,
            initial_raw_parameters=initial,
            artifact_fingerprint="artifact",
            fit_configs={"poisson": ReducedODFitConfig()},
        )
    with pytest.raises(ValueError, match="unknown raw parameter"):
        compare_reduced_od_likelihoods(
            problem=problem,
            initial_raw_parameters=initial,
            artifact_fingerprint="artifact",
            fit_configs={
                likelihood: ReducedODFitConfig(
                    named_raw_parameter_bounds=ReducedODNamedRawParameterBounds(
                        {"not_a_parameter": (-1.0, 1.0)}, require_complete=False
                    )
                )
                for likelihood in ("poisson", "negative_binomial")
            },
        )
    with pytest.raises(ValueError, match="missing required"):
        ReducedODNamedRawParameterBounds(
            {"beta_time": (-1.0, 1.0)}
        ).resolve(("beta_time", "beta_transfer"))
    with pytest.raises(ValueError, match="outside its bounds"):
        compare_reduced_od_likelihoods(
            problem=problem,
            initial_raw_parameters={
                "poisson": np.asarray([20.0, 0.0, 0.0, 0.0]),
                "negative_binomial": np.zeros(5),
            },
            artifact_fingerprint="artifact",
            fit_configs={
                "poisson": ReducedODFitConfig(
                    raw_parameter_bounds=ReducedODRawParameterBounds(
                        np.full(4, -1.0), np.full(4, 1.0)
                    )
                ),
                "negative_binomial": ReducedODFitConfig(),
            },
        )


def test_singular_bounded_comparison_config_is_rejected() -> None:
    problem, _, _ = _problem()
    bounded = ReducedODFitConfig(
        raw_parameter_bounds=ReducedODRawParameterBounds(
            np.full(2, -1.0), np.full(2, 1.0)
        )
    )
    with pytest.raises(ValueError, match="singular fit_config"):
        compare_reduced_od_likelihoods(
            problem=problem,
            initial_raw_parameters={
                "poisson": np.zeros(2),
                "negative_binomial": np.zeros(3),
            },
            artifact_fingerprint="artifact",
            fit_config=bounded,
        )
    with pytest.raises(ValueError, match="not both"):
        compare_reduced_od_likelihoods(
            problem=problem,
            initial_raw_parameters={
                "poisson": np.zeros(2),
                "negative_binomial": np.zeros(3),
            },
            artifact_fingerprint="artifact",
            fit_config=ReducedODFitConfig(),
            fit_configs={
                "poisson": ReducedODFitConfig(),
                "negative_binomial": ReducedODFitConfig(),
            },
        )


def test_ml_recovers_truth_and_reports_timings() -> None:
    problem, truth, fingerprint = _problem()
    result = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.asarray([-0.5, -0.5]),
        model_fingerprint=fingerprint,
        config=ReducedODFitConfig(maximum_iterations=200),
    )
    assert result.status == "complete"
    assert result.success
    np.testing.assert_allclose(result.raw_parameters, truth, rtol=2e-3, atol=2e-3)
    assert result.compile_seconds >= 0.0
    assert result.optimization_seconds >= 0.0
    assert result.evaluations > 0


def test_map_with_flat_prior_is_exactly_ml() -> None:
    problem, _, fingerprint = _problem()
    initial = np.asarray([-0.25, -0.25])
    common = dict(
        problem=problem,
        initial_raw_parameters=initial,
        model_fingerprint=fingerprint,
    )
    ml = estimate_minimal_gravity(**common)
    map_result = estimate_minimal_gravity(
        **common,
        config=ReducedODFitConfig(method="map"),
        prior=GaussianRawParameterPrior(
            mean=np.zeros(2), scale=np.full(2, np.inf)
        ),
    )
    np.testing.assert_array_equal(map_result.raw_parameters, ml.raw_parameters)
    assert map_result.objective == ml.objective
    assert map_result.log_prior == 0.0


def test_informative_map_prior_changes_the_estimate() -> None:
    problem, _, fingerprint = _problem()
    ml = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.zeros(2),
        model_fingerprint=fingerprint,
    )
    mapped = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.zeros(2),
        model_fingerprint=fingerprint,
        config=ReducedODFitConfig(method="map"),
        prior=GaussianRawParameterPrior(
            mean=np.asarray([-2.0, -2.0]), scale=np.asarray([0.1, 0.1])
        ),
    )
    assert np.linalg.norm(mapped.raw_parameters - ml.raw_parameters) > 0.1
    assert mapped.log_prior < 0.0


def test_checkpoint_resume_and_incompatible_fingerprint(tmp_path: Path) -> None:
    problem, _, fingerprint = _problem()
    checkpoint_path = tmp_path / "fit.checkpoint.json"
    first = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.zeros(2),
        model_fingerprint=fingerprint,
        config=ReducedODFitConfig(checkpoint_every_iterations=1),
        checkpoint_path=checkpoint_path,
    )
    checkpoint = load_reduced_od_checkpoint(
        checkpoint_path, expected_manifest=first.manifest
    )
    assert checkpoint.iteration == first.iterations
    resumed = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.ones(2) * 99.0,
        model_fingerprint=fingerprint,
        checkpoint_path=checkpoint_path,
        resume=True,
    )
    assert resumed.resumed_from_iteration == first.iterations
    np.testing.assert_allclose(resumed.raw_parameters, first.raw_parameters, atol=1e-5)
    with pytest.raises(ValueError, match="incompatible"):
        estimate_minimal_gravity(
            problem=problem,
            initial_raw_parameters=np.zeros(2),
            model_fingerprint="different",
            checkpoint_path=checkpoint_path,
            resume=True,
        )


def test_deadline_saves_checkpoint_without_claiming_completion(tmp_path: Path) -> None:
    problem, _, fingerprint = _problem()
    ticks = iter(np.arange(100, dtype=float) * 0.01)
    checkpoint_path = tmp_path / "deadline.json"
    result = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.asarray([-3.0, 3.0]),
        model_fingerprint=fingerprint,
        config=ReducedODFitConfig(
            deadline_seconds=0.001, checkpoint_every_iterations=100
        ),
        checkpoint_path=checkpoint_path,
        clock=lambda: float(next(ticks)),
    )
    assert result.status == "deadline"
    assert not result.success
    assert checkpoint_path.is_file()


def test_result_persistence_records_noncomplete_status(tmp_path: Path) -> None:
    problem, _, fingerprint = _problem()
    fitted = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.zeros(2),
        model_fingerprint=fingerprint,
    )
    path = tmp_path / "result.json"
    save_reduced_od_fit_result(
        path,
        manifest=fitted.manifest,
        status="deadline",
        raw_parameters=fitted.raw_parameters,
        summary={"iterations": fitted.iterations},
    )
    document = json.loads(path.read_text())
    assert document["status"] == "deadline"
    assert document["summary"]["iterations"] == fitted.iterations


def _contract() -> ReducedODProblemContract:
    return ReducedODProblemContract(
        configuration_fingerprint="config",
        timetable_artifact_fingerprint="timetable",
        response_artifact_fingerprint="response",
        od_keys=(
            JourneyODTimeKey("A", "B", "P"),
            JourneyODTimeKey("A", "C", "P"),
            JourneyODTimeKey("A", "D", "P"),
            JourneyODTimeKey("X", "Y", "P"),
        ),
        free_od_indices=np.asarray([0, 2]),
        fixed_od_indices=np.asarray([1, 3]),
        fixed_od_values=np.asarray([7.0, 0.0]),
    )


def test_reconstruction_preserves_canonical_fixed_and_structural_zero_cells() -> None:
    reconstructed = reconstruct_full_od(
        contract=_contract(),
        free_cell_keys=(
            ResponseCellKey("A", "B", "P"),
            ResponseCellKey("A", "D", "P"),
        ),
        free_demand=np.asarray([11.0, 13.0]),
    )
    np.testing.assert_array_equal(reconstructed.demand, [11.0, 7.0, 13.0, 0.0])
    np.testing.assert_array_equal(reconstructed.estimated, [True, False, True, False])
    assert reconstructed.rows[1].demand == 7.0
    assert not reconstructed.rows[3].estimated


def test_reconstruction_rejects_noncanonical_free_order() -> None:
    with pytest.raises(ValueError, match="free OD order"):
        reconstruct_full_od(
            contract=_contract(),
            free_cell_keys=(
                ResponseCellKey("A", "D", "P"),
                ResponseCellKey("A", "B", "P"),
            ),
            free_demand=np.asarray([13.0, 11.0]),
        )


def test_detailed_assignment_runs_once_and_quantifies_discrepancy() -> None:
    reconstructed = reconstruct_full_od(
        contract=_contract(),
        free_cell_keys=(
            ResponseCellKey("A", "B", "P"),
            ResponseCellKey("A", "D", "P"),
        ),
        free_demand=np.asarray([11.0, 13.0]),
    )
    calls: list[np.ndarray] = []

    def assignment(value):
        calls.append(value.demand)
        return DetailedAssignmentOutput(
            measurement_prediction=np.asarray([10.0, 18.0]),
            transfer_audit={"transfer_boardings": 3.0},
        )

    result = validate_with_detailed_assignment(
        reconstructed_od=reconstructed,
        compact_prediction=np.asarray([9.0, 20.0]),
        run_detailed_assignment=assignment,
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(result.difference, [1.0, -2.0])
    assert result.maximum_absolute_error == 2.0
    assert result.root_mean_square_error == pytest.approx(np.sqrt(2.5))
    assert result.transfer_audit["transfer_boardings"] == 3.0


def test_estimator_never_invokes_reconstruction_or_assignment(monkeypatch) -> None:
    problem, _, fingerprint = _problem()
    forbidden = SimpleNamespace(calls=0)

    def fail(*args, **kwargs):
        forbidden.calls += 1
        raise AssertionError("post-fit operation reached from optimizer")

    monkeypatch.setattr(
        "public_transportation.inference.reduced_od.reconstruction.reconstruct_full_od",
        fail,
    )
    result = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.asarray([-0.5, -0.5]),
        model_fingerprint=fingerprint,
    )
    assert result.status == "complete"
    assert forbidden.calls == 0
