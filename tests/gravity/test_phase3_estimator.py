from __future__ import annotations

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    FixedRoutingMeasurementOperator,
    MeasurementOperatorMetrics,
)
from public_transportation.inference.gravity import (
    GravityEstimatorConfig,
    GravityExecutionPolicy,
    GravityFeatures,
    GravityLikelihood,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    estimate_gravity_model,
    predict_gravity_measurements,
)


def setup_problem():
    cells = 9
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
        journey_time=np.asarray((2, 8, 15, 4, 12, 7, 20, 5, 10), dtype=np.float64),
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
    matrix = jnp.eye(cells, dtype=jnp.float64)
    operator = FixedRoutingMeasurementOperator(
        matrix,
        jnp.zeros(cells, dtype=jnp.float64),
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
        "float64",
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
