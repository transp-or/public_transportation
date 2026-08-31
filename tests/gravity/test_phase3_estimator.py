from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import public_transportation.inference.gravity.estimator as estimator_module

from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    FixedRoutingMeasurementOperator,
    MeasurementOperatorMetrics,
)
from public_transportation.inference.gravity import (
    GravityEstimatorConfig,
    GravityBiogemePilotResult,
    GravityEstimatorProgress,
    GravityExecutionPolicy,
    GravityFeatures,
    GravityLikelihood,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    GravityPreflightPhase,
    GravityJSONLProgressSink,
    build_gravity_run_manifest,
    estimate_gravity_model,
    predict_gravity_measurements,
    run_gravity_preflight,
    scaled_gradient_inf_norm,
    run_biogeme_tr_bfgs_pilot,
    write_gravity_run_manifest,
)


def setup_problem(dtype=np.float64):
    cells = 9
    numeric_dtype = np.dtype(dtype)
    layout = CompactODAssignmentLayout(
        num_od_total=cells,
        active_full_indices=tuple(range(cells)),
        removed_zero_full_indices=(),
        full_to_compact=tuple(range(cells)),
        free_full_indices=tuple(range(cells)),
        free_compact_indices=tuple(range(cells)),
        free_baseline_values=tuple(1.0 for _ in range(cells)),
        fixed_compact_indices=(),
        fixed_compact_values=(),
    )
    features = GravityFeatures(
        canonical_od_index=np.arange(cells),
        origin_index=np.repeat(np.arange(3), 3),
        destination_index=np.tile(np.arange(3), 3),
        departure_time_index=np.repeat(np.arange(3), 3),
        origin_time_group_index=np.repeat(np.arange(3), 3),
        journey_time=np.asarray(
            (2, 8, 15, 4, 12, 7, 20, 5, 10), dtype=numeric_dtype
        ),
        transfer_count=np.asarray((0, 1, 2, 2, 0, 1, 1, 2, 0)),
        structural_feasible=np.ones(cells, dtype=bool),
        origin_time_totals=np.asarray((100.0, 80.0, 120.0)),
        destination_attractiveness=np.asarray((1, 2, 1, 1, 1, 3, 2, 1, 1), dtype=float),
        num_origins=3,
        num_destinations=3,
        num_departure_times=3,
        od_layout_fingerprint=layout.fingerprint,
        journey_time_scale=10.0,
    )
    matrix = jnp.eye(cells, dtype=numeric_dtype)
    operator = FixedRoutingMeasurementOperator(
        matrix,
        jnp.zeros(cells, dtype=numeric_dtype),
        "dense",
        cells,
        cells,
        cells,
        layout.fingerprint,
        layout.fingerprint,
        "assignment",
        "graph",
        "mapping",
        1.0,
        str(numeric_dtype),
        MeasurementOperatorMetrics(
            0, matrix.nbytes, matrix.nbytes, 0, cells, cells**2, 1 / cells, cells
        ),
    )
    parameter_layout = GravityParameterLayout(GravityModelSpecification())
    seed = GravityObjectiveProblem(
        features,
        parameter_layout,
        operator,
        np.ones(cells),
        GravityLikelihood.POISSON,
    )
    true_raw = parameter_layout.raw_from_physical((0.7, 1.2, 5.0))
    observations = np.asarray(predict_gravity_measurements(true_raw, problem=seed)[0])
    problem = GravityObjectiveProblem(
        features,
        parameter_layout,
        operator,
        observations,
        GravityLikelihood.POISSON,
    )
    return problem, layout, true_raw


def test_optimizer_maxls_is_configurable_and_validated():
    config = GravityEstimatorConfig(optimizer_maxls=100)
    assert config.optimizer_maxls == 100

    with pytest.raises(ValueError, match="optimizer_maxls must be positive"):
        GravityEstimatorConfig(optimizer_maxls=0)


def test_optimizer_maxls_is_passed_to_lbfgsb(monkeypatch):
    with jax.enable_x64():
        problem, layout, _ = setup_problem()
        captured: dict[str, object] = {}

        def fake_minimize(fun, x0, *, method, jac, callback, options):
            captured.update(
                method=method,
                jac=jac,
                options=options,
            )
            fun(np.asarray(x0, dtype=float))
            return SimpleNamespace(
                x=np.asarray(x0, dtype=float),
                success=True,
                message="stub optimizer",
            )

        monkeypatch.setattr(estimator_module, "minimize", fake_minimize)
        result = estimate_gravity_model(
            problem=problem,
            compact_layout=layout,
            initial_raw_parameters=np.zeros(3),
            config=GravityEstimatorConfig(
                maximum_iterations=4,
                optimizer_maxls=100,
                scaled_gradient_tolerance=1.0,
            ),
            execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
        )

    assert result.success
    assert captured["method"] == "L-BFGS-B"
    assert captured["jac"] is True
    options = captured["options"]
    assert isinstance(options, dict)
    assert options["maxls"] == 100
    assert options["maxiter"] == 4


def test_dennis_schnabel_scaled_gradient_uses_scalar_and_parameter_scales():
    scalar = scaled_gradient_inf_norm(
        parameters=np.asarray((2.0, -0.5)),
        gradient=np.asarray((3.0, 4.0)),
        objective=5.0,
        typical_objective_scale=1.0,
        typical_parameter_scales=2.0,
    )
    assert scalar == pytest.approx(1.6)

    per_parameter = scaled_gradient_inf_norm(
        parameters=np.asarray((2.0, -0.5)),
        gradient=np.asarray((3.0, 4.0)),
        objective=5.0,
        typical_objective_scale=1.0,
        typical_parameter_scales=(1.0, 10.0),
    )
    assert per_parameter == pytest.approx(8.0)

    zero_dimensional = scaled_gradient_inf_norm(
        parameters=np.asarray((2.0,)),
        gradient=np.asarray((3.0,)),
        objective=5.0,
        typical_parameter_scales=np.asarray(2.0),
    )
    assert zero_dimensional == pytest.approx(1.2)


def test_dennis_schnabel_scaled_gradient_uses_typf_near_zero_objective():
    value = scaled_gradient_inf_norm(
        parameters=np.asarray((3.0,)),
        gradient=np.asarray((2.0,)),
        objective=0.0,
        typical_objective_scale=4.0,
        typical_parameter_scales=(5.0,),
    )
    assert value == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("scaled_gradient_tolerance", 0.0),
        ("scaled_gradient_tolerance", np.inf),
        ("typical_objective_scale", 0.0),
        ("typical_objective_scale", np.nan),
        ("typical_parameter_scales", 0.0),
        ("typical_parameter_scales", (1.0, np.inf)),
    ),
)
def test_scaled_gradient_configuration_rejects_invalid_scales(field, value):
    with pytest.raises(ValueError, match=field):
        GravityEstimatorConfig(**{field: value})


def test_estimator_rejects_typx_vector_with_wrong_parameter_count():
    with jax.enable_x64():
        problem, layout, _ = setup_problem()
        with pytest.raises(ValueError, match="one value per parameter"):
            estimate_gravity_model(
                problem=problem,
                compact_layout=layout,
                initial_raw_parameters=np.zeros(3),
                config=GravityEstimatorConfig(
                    maximum_iterations=1,
                    typical_parameter_scales=(1.0, 2.0),
                ),
                execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
            )


def _run_with_patched_scaled_gradient(monkeypatch, value):
    with jax.enable_x64():
        problem, layout, _ = setup_problem()
        monkeypatch.setattr(
            estimator_module,
            "scaled_gradient_inf_norm",
            lambda *args, **kwargs: value,
        )
        def fake_minimize(fun, x0, *, method, jac, callback, options):
            fun(np.asarray(x0, dtype=float))
            return SimpleNamespace(
                x=np.asarray(x0, dtype=float),
                success=True,
                message="stub optimizer success",
            )

        monkeypatch.setattr(estimator_module, "minimize", fake_minimize)
        return estimate_gravity_model(
            problem=problem,
            compact_layout=layout,
            initial_raw_parameters=np.zeros(3),
            config=GravityEstimatorConfig(
                maximum_iterations=4,
                scaled_gradient_tolerance=1.0,
            ),
            execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
        )


def test_scipy_success_is_rejected_when_scaled_gradient_is_too_large(monkeypatch):
    result = _run_with_patched_scaled_gradient(monkeypatch, value=2.0)
    assert result.status == "iteration_limit"
    assert not result.success
    assert result.message
    assert result.scaled_gradient_inf_norm == pytest.approx(2.0)


def test_scipy_success_is_accepted_when_scaled_gradient_meets_tolerance(monkeypatch):
    result = _run_with_patched_scaled_gradient(monkeypatch, value=0.5)
    assert result.status == "converged"
    assert result.success
    assert result.scaled_gradient_inf_norm == pytest.approx(0.5)


def test_minimal_estimator_recovers_impedance_and_complete_od_layout(tmp_path):
    with jax.enable_x64():
        problem, layout, true_raw = setup_problem()
        result = estimate_gravity_model(
            problem=problem,
            compact_layout=layout,
            initial_raw_parameters=np.zeros(3),
            config=GravityEstimatorConfig(maximum_iterations=100),
            execution=GravityExecutionPolicy(
                gradient_strategy="batched_forward",
                checkpoint_path=tmp_path / "gravity.json",
                jax_compilation_cache_directory=tmp_path / "jax-cache",
            ),
        )
        assert result.success
        np.testing.assert_allclose(result.raw_parameters[:2], true_raw[:2], atol=2e-4)
        assert result.full_od_demand.shape == (layout.num_od_total,)
        np.testing.assert_allclose(
            result.free_od_demand.reshape(3, 3).sum(axis=1),
            problem.features.origin_time_totals,
        )
        np.testing.assert_allclose(result.predicted_measurements, result.free_od_demand)
        assert not result.full_od_demand.flags.writeable
        assert result.strategy_selection.persistent_compilation_cache_enabled
        assert result.strategy_selection.persistent_compilation_cache_directory
        assert (tmp_path / "jax-cache").is_dir()


def test_estimator_records_scaled_gradient_and_precision_diagnostics():
    with jax.enable_x64():
        problem, layout, _ = setup_problem(dtype=np.float32)
        events: list[GravityEstimatorProgress] = []
        result = estimate_gravity_model(
            problem=problem,
            compact_layout=layout,
            initial_raw_parameters=np.zeros(3),
            config=GravityEstimatorConfig(
                maximum_iterations=20,
                scaled_gradient_tolerance=1.0e-4,
                typical_objective_scale=10.0,
                typical_parameter_scales=(1.0, 2.0, 3.0),
            ),
            execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
            progress=events.append,
        )

    assert result.gradient_inf_norm is not None
    assert result.scaled_gradient_inf_norm is not None
    assert result.gradient_inf_norm == pytest.approx(
        np.max(np.abs(result.gradient))
    )
    assert result.objective_dtype == "float32"
    assert result.gradient_dtype == "float32"
    assert result.objective_spacing == pytest.approx(
        float(np.spacing(np.float32(result.objective)))
    )
    assert result.initial_objective is not None
    assert result.typical_objective_scale_provenance
    assert result.typical_parameter_scales_provenance
    assert result.typical_objective_scale_selection
    assert result.objective_tolerance_below_precision is True
    assert result.objective_reduction is None or np.isfinite(
        result.objective_reduction
    )
    assert events
    final_event = events[-1]
    assert final_event.status == result.status
    assert final_event.termination_message == result.message
    assert final_event.scaled_gradient_inf_norm == pytest.approx(
        result.scaled_gradient_inf_norm
    )
    assert final_event.typical_objective_scale == 10.0
    assert final_event.typical_parameter_scales == (1.0, 2.0, 3.0)


def test_estimator_reports_reduction_between_accepted_iterates(monkeypatch):
    with jax.enable_x64():
        problem, layout, _ = setup_problem()
        accepted_objectives: list[float] = []

        def fake_minimize(fun, x0, *, method, jac, callback, options):
            first = np.asarray(x0, dtype=float)
            objective, _ = fun(first)
            accepted_objectives.append(float(objective))
            callback(first)
            second = first.copy()
            second[0] = 0.25
            objective, _ = fun(second)
            accepted_objectives.append(float(objective))
            callback(second)
            return SimpleNamespace(
                x=second,
                success=True,
                message="stub optimizer success",
            )

        monkeypatch.setattr(estimator_module, "minimize", fake_minimize)
        result = estimate_gravity_model(
            problem=problem,
            compact_layout=layout,
            initial_raw_parameters=np.zeros(3),
            config=GravityEstimatorConfig(
                maximum_iterations=4,
                scaled_gradient_tolerance=1.0e6,
            ),
            execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
        )

    assert result.objective_reduction == pytest.approx(
        accepted_objectives[-2] - accepted_objectives[-1]
    )


def test_float32_and_float64_diagnostic_pilot_is_comparable():
    with jax.enable_x64():
        results = {}
        for dtype in (np.float32, np.float64):
            problem, layout, _ = setup_problem(dtype=dtype)
            results[dtype] = estimate_gravity_model(
                problem=problem,
                compact_layout=layout,
                initial_raw_parameters=np.zeros(3),
                config=GravityEstimatorConfig(maximum_iterations=30),
                execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
            )

    float32_result = results[np.float32]
    float64_result = results[np.float64]
    assert float32_result.objective_dtype == "float32"
    assert float64_result.objective_dtype == "float64"
    assert float32_result.gradient_dtype == "float32"
    assert float64_result.gradient_dtype == "float64"
    assert np.isfinite(float32_result.scaled_gradient_inf_norm)
    assert np.isfinite(float64_result.scaled_gradient_inf_norm)
    np.testing.assert_allclose(
        float32_result.objective,
        float64_result.objective,
        rtol=2.0e-4,
        atol=2.0e-4,
    )
    np.testing.assert_allclose(
        float32_result.raw_parameters,
        float64_result.raw_parameters,
        rtol=2.0e-3,
        atol=2.0e-3,
    )
    assert float32_result.message
    assert float64_result.message


def test_optional_biogeme_tr_bfgs_pilot_uses_same_convergence_audit(monkeypatch):
    optimization_module = ModuleType("biogeme.optimization")

    def fake_tr_bfgs(function, initial, bounds, variable_names, parameters):
        assert parameters["maxiter"] == 12
        assert parameters["tolerance"] == pytest.approx(1.0e-6)
        assert parameters["objective_tolerance"] == pytest.approx(1.0e-9)
        candidate = np.asarray(initial, dtype=float).copy()
        candidate[0] = 0.5
        function.set_variables(candidate)
        function.f_g()
        return SimpleNamespace(
            solution=candidate,
            messages={
                "Cause of termination": "stub trust-region convergence",
                "Number of iterations": 2,
            },
            convergence=True,
        )

    optimization_module.bfgs_trust_region_for_biogeme = fake_tr_bfgs
    biogeme_module = ModuleType("biogeme")
    biogeme_module.optimization = optimization_module
    monkeypatch.setitem(sys.modules, "biogeme", biogeme_module)
    monkeypatch.setitem(sys.modules, "biogeme.optimization", optimization_module)

    progress: list[GravityEstimatorProgress] = []
    result = run_biogeme_tr_bfgs_pilot(
        objective_and_gradient=lambda value: (
            np.sum((value - 1.0) ** 2, dtype=np.float32),
            np.asarray(2.0 * (value - 1.0), dtype=np.float32),
        ),
        initial_raw_parameters=np.zeros(2),
        config=GravityEstimatorConfig(
            maximum_iterations=12,
            scaled_gradient_tolerance=10.0,
            typical_objective_scale=2.0,
            typical_parameter_scales=(1.0, 2.0),
        ),
        progress=progress.append,
    )

    assert isinstance(result, GravityBiogemePilotResult)
    assert result.optimizer == "TR-BFGS"
    assert result.success
    assert result.status == "converged"
    assert result.objective_dtype == "float32"
    assert result.gradient_dtype == "float32"
    assert result.scaled_gradient_inf_norm == pytest.approx(2.0)
    assert result.message == "stub trust-region convergence"
    assert result.initial_objective == pytest.approx(2.0)
    assert result.typical_objective_scale_provenance
    assert result.typical_parameter_scales_provenance
    assert progress and progress[-1].scaled_gradient_inf_norm == pytest.approx(2.0)


def test_resume_reaches_same_solution_as_uninterrupted(tmp_path):
    with jax.enable_x64():
        problem, layout, _ = setup_problem()
        common = dict(
            problem=problem, compact_layout=layout, initial_raw_parameters=np.zeros(3)
        )
        uninterrupted = estimate_gravity_model(
            **common,
            config=GravityEstimatorConfig(maximum_iterations=80),
            execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
        )
        checkpoint = tmp_path / "resume.json"
        estimate_gravity_model(
            **common,
            config=GravityEstimatorConfig(maximum_iterations=2),
            execution=GravityExecutionPolicy(
                gradient_strategy="adjoint", checkpoint_path=checkpoint
            ),
        )
        resumed = estimate_gravity_model(
            **common,
            config=GravityEstimatorConfig(maximum_iterations=80),
            execution=GravityExecutionPolicy(
                gradient_strategy="adjoint", checkpoint_path=checkpoint
            ),
            resume=True,
        )
        np.testing.assert_allclose(
            resumed.raw_parameters, uninterrupted.raw_parameters, atol=2e-5
        )
        np.testing.assert_allclose(
            resumed.predicted_measurements,
            uninterrupted.predicted_measurements,
            atol=1e-4,
        )
        assert resumed.resumed


def test_deadline_preserves_valid_initial_checkpoint(tmp_path):
    with jax.enable_x64():
        problem, layout, _ = setup_problem()

        class Clock:
            value = 0.0

            def __call__(self):
                self.value += 1.0
                return self.value

        checkpoint = tmp_path / "deadline.json"
        result = estimate_gravity_model(
            problem=problem,
            compact_layout=layout,
            initial_raw_parameters=np.zeros(3),
            execution=GravityExecutionPolicy(
                gradient_strategy="batched_forward",
                checkpoint_path=checkpoint,
                wall_time_seconds=0.5,
            ),
            clock=Clock(),
        )
        assert result.status == "stopped_by_time_budget"
        payload = json.loads(checkpoint.read_text())
        assert payload["iterations"] == 0
        assert payload["model_fingerprint"] == result.model_fingerprint


def test_deadline_after_valid_iteration_resumes_to_uninterrupted_result(tmp_path):
    with jax.enable_x64():
        problem, layout, _ = setup_problem()

        class Clock:
            value = 0.0

            def __call__(self):
                self.value += 1.0
                return self.value

        common = dict(
            problem=problem,
            compact_layout=layout,
            initial_raw_parameters=np.zeros(3),
            config=GravityEstimatorConfig(maximum_iterations=80),
        )
        uninterrupted = estimate_gravity_model(
            **common,
            execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
        )
        checkpoint = tmp_path / "deadline-resume.json"
        interrupted = estimate_gravity_model(
            **common,
            execution=GravityExecutionPolicy(
                gradient_strategy="adjoint",
                checkpoint_path=checkpoint,
                wall_time_seconds=15.0,
            ),
            clock=Clock(),
        )
        payload = json.loads(checkpoint.read_text())
        assert interrupted.status == "stopped_by_time_budget"
        assert interrupted.iterations >= 1
        assert payload["iterations"] == interrupted.iterations
        assert interrupted.deadline_phase in {
            "before objective-and-gradient evaluation",
            "after a completed optimizer iteration",
        }

        resumed = estimate_gravity_model(
            **common,
            execution=GravityExecutionPolicy(
                gradient_strategy="adjoint", checkpoint_path=checkpoint
            ),
            resume=True,
        )
        np.testing.assert_allclose(
            resumed.raw_parameters, uninterrupted.raw_parameters, atol=2e-5
        )
        np.testing.assert_allclose(
            resumed.predicted_measurements,
            uninterrupted.predicted_measurements,
            atol=1e-4,
        )


def test_auto_strategy_reports_bounded_preflight_and_is_deterministic():
    with jax.enable_x64():
        problem, layout, _ = setup_problem()
        results = [
            estimate_gravity_model(
                problem=problem,
                compact_layout=layout,
                initial_raw_parameters=np.zeros(3),
                config=GravityEstimatorConfig(maximum_iterations=20),
                execution=GravityExecutionPolicy(gradient_strategy="auto"),
            )
            for _ in range(2)
        ]
        for result in results:
            assert result.strategy_selection.requested == "auto"
            assert len(result.strategy_selection.candidates) == 2
            assert all(
                item.tracing_seconds >= 0
                and item.lowering_seconds >= 0
                and item.compilation_seconds >= 0
                and item.first_execution_seconds >= 0
                and item.warm_execution_seconds >= 0
                for item in result.strategy_selection.candidates
            )
        np.testing.assert_allclose(
            results[0].predicted_measurements,
            results[1].predicted_measurements,
            atol=1e-10,
        )


def test_public_preflight_stops_at_boundaries_and_recommends_measured_strategy():
    with jax.enable_x64():
        problem, _, _ = setup_problem()
        raw = np.zeros(3)
        forward_only = run_gravity_preflight(
            problem=problem,
            raw_parameters=raw,
            stop_after=GravityPreflightPhase.FORWARD,
        )
        assert forward_only.completed_phase is GravityPreflightPhase.FORWARD
        assert set(forward_only.timings_seconds) == {"validation", "forward"}
        assert forward_only.recommendation is None

        complete = run_gravity_preflight(problem=problem, raw_parameters=raw)
        assert complete.completed_phase is GravityPreflightPhase.RECOMMENDATION
        assert complete.gradient_max_abs_difference == pytest.approx(0.0, abs=1e-9)
        assert complete.recommendation is not None
        selected = complete.recommendation.gradient_strategy.value
        assert complete.recommendation.expected_evaluation_seconds == pytest.approx(
            complete.timings_seconds[selected]
        )
        assert complete.recommendation.suggested_checkpoint_interval == 1


def test_run_manifest_and_progress_log_are_durable_and_serializable(tmp_path):
    with jax.enable_x64():
        problem, layout, _ = setup_problem()
        config = GravityEstimatorConfig(maximum_iterations=7, optimizer_maxls=100)
        result = estimate_gravity_model(
            problem=problem,
            compact_layout=layout,
            initial_raw_parameters=np.zeros(3),
            config=config,
            execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
        )
    execution = GravityExecutionPolicy(
        gradient_strategy="adjoint",
        checkpoint_path=tmp_path / "checkpoint.json",
        jax_compilation_cache_directory=tmp_path / "jax-cache",
    )
    manifest = build_gravity_run_manifest(
        problem=problem,
        compact_layout=layout,
        estimator_config=config,
        execution=execution,
        repository_revision="test-revision",
        result=result,
        extra={"purpose": "unit-test"},
    )
    manifest_path = tmp_path / "run-manifest.json"
    write_gravity_run_manifest(manifest_path, manifest)
    loaded = json.loads(manifest_path.read_text())
    assert loaded["repository_revision"] == "test-revision"
    assert loaded["model_fingerprint"] == manifest["model_fingerprint"]
    assert loaded["estimator_config"]["optimizer_maxls"] == 100
    diagnostics = loaded["convergence_diagnostics"]
    assert diagnostics["objective"] == pytest.approx(result.objective)
    assert diagnostics["initial_objective"] == pytest.approx(
        result.initial_objective
    )
    assert diagnostics["gradient_inf_norm"] == pytest.approx(
        result.gradient_inf_norm
    )
    assert diagnostics["scaled_gradient_inf_norm"] == pytest.approx(
        result.scaled_gradient_inf_norm
    )
    assert diagnostics["termination_message"] == result.message
    assert diagnostics["objective_dtype"] == result.objective_dtype
    assert diagnostics["typical_parameter_scales"] == [1.0, 1.0, 1.0]
    assert diagnostics["typical_objective_scale"] == pytest.approx(
        config.typical_objective_scale
    )
    assert diagnostics["typical_objective_scale_provenance"] == (
        result.typical_objective_scale_provenance
    )
    assert diagnostics["typical_parameter_scales_provenance"] == (
        result.typical_parameter_scales_provenance
    )
    assert diagnostics["scaled_gradient_tolerance"] == pytest.approx(
        config.scaled_gradient_tolerance
    )
    assert diagnostics["objective_spacing"] == pytest.approx(
        result.objective_spacing
    )
    assert diagnostics["objective_tolerance_below_precision"] is False
    assert loaded["execution"]["checkpoint_path"] == str(
        execution.checkpoint_path
    )
    assert loaded["operator"]["num_free_od"] == problem.operator.num_free_od

    progress_path = tmp_path / "progress.jsonl"
    sink = GravityJSONLProgressSink(
        progress_path, durable=True, context={"run_id": "test-run"}
    )
    sink(
        GravityEstimatorProgress(
            iteration=1,
            objective=2.0,
            gradient_inf_norm=0.5,
            elapsed_seconds=3.0,
            checkpoint_written=True,
        )
    )
    sink(
        GravityEstimatorProgress(
            iteration=2,
            objective=1.0,
            gradient_inf_norm=0.25,
            elapsed_seconds=4.0,
            checkpoint_written=True,
        )
    )
    records = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert [record["event"]["iteration"] for record in records] == [1, 2]
    assert all(record["run_id"] == "test-run" for record in records)
    assert records[0]["event"]["schema_version"] == 1
    assert records[0]["event"]["completed_units"] == 1
    assert records[0]["event"]["work_stack"][0]["name"] == "optimizer_iterations"


def test_negative_binomial_synthetic_fit_recovers_all_minimal_parameters():
    with jax.enable_x64():
        base, layout, true_raw = setup_problem()
        mean = np.asarray(predict_gravity_measurements(true_raw, problem=base)[0])
        repetitions = 120
        routing_matrix = jnp.tile(jnp.eye(mean.size), (repetitions, 1))
        true_dispersion = 5.0
        probability = true_dispersion / (true_dispersion + mean)
        rng = np.random.default_rng(731)
        observations = np.concatenate(
            [
                rng.negative_binomial(true_dispersion, probability)
                for _ in range(repetitions)
            ]
        )
        repeated_operator = replace(
            base.operator,
            matrix=routing_matrix,
            fixed_measurement_offset=jnp.zeros(observations.size),
            num_measurements=observations.size,
            metrics=replace(
                base.operator.metrics,
                dense_bytes=int(routing_matrix.nbytes),
                stored_bytes=int(routing_matrix.nbytes),
                total_entries=int(routing_matrix.size),
            ),
        )
        nb_problem = replace(
            base,
            operator=repeated_operator,
            observations=observations,
            likelihood=GravityLikelihood.NEGATIVE_BINOMIAL,
            calibration_mask=None,
        )
        result = estimate_gravity_model(
            problem=nb_problem,
            compact_layout=layout,
            initial_raw_parameters=np.zeros(3),
            config=GravityEstimatorConfig(maximum_iterations=100),
            execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
        )
        assert result.success
        np.testing.assert_allclose(result.physical_parameters[:2], (0.7, 1.2), atol=0.2)
        assert result.physical_parameters[2] == pytest.approx(true_dispersion, rel=0.35)


def test_resume_rejects_changed_model_fingerprint(tmp_path):
    with jax.enable_x64():
        problem, layout, _ = setup_problem()
        checkpoint = tmp_path / "fingerprint.json"
        estimate_gravity_model(
            problem=problem,
            compact_layout=layout,
            initial_raw_parameters=np.zeros(3),
            config=GravityEstimatorConfig(maximum_iterations=1),
            execution=GravityExecutionPolicy(
                gradient_strategy="adjoint", checkpoint_path=checkpoint
            ),
        )
        changed = replace(problem, observations=problem.observations + 1)
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            estimate_gravity_model(
                problem=changed,
                compact_layout=layout,
                initial_raw_parameters=np.zeros(3),
                execution=GravityExecutionPolicy(
                    gradient_strategy="adjoint", checkpoint_path=checkpoint
                ),
                resume=True,
            )
