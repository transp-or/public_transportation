from __future__ import annotations

import json
from dataclasses import replace

import jax
import numpy as np
from jax.experimental import sparse as jsparse

from public_transportation.inference.gravity import (
    GravityEffectScope,
    GravityEstimatorConfig,
    GravityExecutionPolicy,
    GravityGradientStrategy,
    GravityLikelihood,
    GravityParameterLayout,
    add_gravity_relaxation,
    estimate_gravity_model,
    gravity_value_and_gradient,
    warm_start_gravity_parameters,
)
from tests.gravity.test_phase6_recommendations import recommendation_case


def relaxed_problems():
    parent_problem, compact, parent_result = recommendation_case(
        destination_effect=0.4
    )
    specification, _ = add_gravity_relaxation(
        parent_problem.parameter_layout.specification,
        features=parent_problem.features,
        scope=GravityEffectScope.DESTINATION_ZONE,
    )
    layout = GravityParameterLayout(specification)
    raw = warm_start_gravity_parameters(
        parent_problem.parameter_layout, layout, parent_result.raw_parameters
    )
    dense = replace(parent_problem, parameter_layout=layout)
    sparse_matrix = jsparse.BCOO.fromdense(parent_problem.operator.matrix)
    sparse_operator = replace(
        parent_problem.operator,
        matrix=sparse_matrix,
        representation="bcoo",
        metrics=replace(
            parent_problem.operator.metrics,
            stored_bytes=int(sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes),
        ),
    )
    sparse = replace(dense, operator=sparse_operator)
    return dense, sparse, compact, raw


def test_relaxed_dense_and_bcoo_objectives_and_gradients_agree():
    with jax.enable_x64():
        dense, sparse, _, raw = relaxed_problems()
        for strategy in GravityGradientStrategy:
            dense_evaluation, dense_gradient = gravity_value_and_gradient(
                raw, problem=dense, strategy=strategy
            )
            sparse_evaluation, sparse_gradient = gravity_value_and_gradient(
                raw, problem=sparse, strategy=strategy
            )
            np.testing.assert_allclose(
                sparse_evaluation.measurement_mean,
                dense_evaluation.measurement_mean,
                rtol=1e-11,
                atol=1e-11,
            )
            np.testing.assert_allclose(
                sparse_gradient, dense_gradient, rtol=1e-10, atol=1e-10
            )


def test_relaxed_checkpoint_resume_matches_uninterrupted(tmp_path):
    with jax.enable_x64():
        dense, _, compact, raw = relaxed_problems()
        dense = replace(dense, likelihood=GravityLikelihood.POISSON)
        common: dict[str, object] = {
            "problem": dense,
            "compact_layout": compact,
            "initial_raw_parameters": raw,
        }
        uninterrupted = estimate_gravity_model(
            **common,
            config=GravityEstimatorConfig(maximum_iterations=50),
            execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
        )
        checkpoint = tmp_path / "relaxed-resume.json"
        estimate_gravity_model(
            **common,
            config=GravityEstimatorConfig(maximum_iterations=2),
            execution=GravityExecutionPolicy(
                gradient_strategy="adjoint", checkpoint_path=checkpoint
            ),
        )
        resumed = estimate_gravity_model(
            **common,
            config=GravityEstimatorConfig(maximum_iterations=50),
            execution=GravityExecutionPolicy(
                gradient_strategy="adjoint", checkpoint_path=checkpoint
            ),
            resume=True,
        )
        assert resumed.resumed
        np.testing.assert_allclose(
            resumed.raw_parameters, uninterrupted.raw_parameters, atol=2e-5
        )
        np.testing.assert_allclose(
            resumed.predicted_measurements,
            uninterrupted.predicted_measurements,
            atol=2e-4,
        )


def test_relaxed_bcoo_estimator_returns_immutable_complete_result():
    with jax.enable_x64():
        _, sparse, compact, raw = relaxed_problems()
        result = estimate_gravity_model(
            problem=sparse,
            compact_layout=compact,
            initial_raw_parameters=raw,
            config=GravityEstimatorConfig(maximum_iterations=2),
            execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
        )
        assert result.full_od_demand.shape == (compact.num_od_total,)
        assert result.physical_parameters.size == sparse.parameter_layout.size
        assert not result.full_od_demand.flags.writeable
        assert not result.raw_parameters.flags.writeable


def test_relaxed_deadline_preserves_checkpoint_and_reports_cache(tmp_path):
    with jax.enable_x64():
        dense, _, compact, raw = relaxed_problems()

        class Clock:
            value = 0.0

            def __call__(self):
                self.value += 1.0
                return self.value

        checkpoint = tmp_path / "relaxed-deadline.json"
        cache = tmp_path / "jax-cache"
        result = estimate_gravity_model(
            problem=dense,
            compact_layout=compact,
            initial_raw_parameters=raw,
            execution=GravityExecutionPolicy(
                gradient_strategy="adjoint",
                wall_time_seconds=0.5,
                checkpoint_path=checkpoint,
                jax_compilation_cache_directory=cache,
            ),
            clock=Clock(),
        )
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert result.status == "stopped_by_time_budget"
        assert payload["iterations"] == 0
        assert payload["raw_parameters"] == raw.tolist()
        assert result.strategy_selection.persistent_compilation_cache_enabled
        assert result.strategy_selection.persistent_compilation_cache_directory == str(
            cache.resolve()
        )
