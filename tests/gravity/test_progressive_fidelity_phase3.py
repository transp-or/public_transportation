from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.inference.gravity import (
    GravityFidelityContext,
    GravityFidelityRequest,
    GravityFidelityShard,
    GravityLikelihood,
    gravity_fidelity_problem_identity,
    gravity_value_and_gradient_progressive,
)
from tests.gravity.test_phase2_objective import problem
from tests.gravity.test_progressive_fidelity_phase2 import MatrixShard, split_context


def test_quality_reports_estimated_objective_count_and_gradient_errors():
    with jax.enable_x64():
        item = problem()
        context = split_context(item, count=16)
        result = gravity_value_and_gradient_progressive(
            np.asarray((0.2, -0.1, 1.0)),
            problem=item,
            fidelity=GravityFidelityRequest(
                effort_percent=40, seed=17, quality_groups=4
            ),
            context=context,
        )
        quality = result.quality
        assert not quality.exact
        assert 0.0 <= quality.quality_score <= 1.0
        assert quality.reliability in {"low", "medium", "high"}
        assert quality.objective_standard_error is not None
        assert quality.objective_standard_error >= 0.0
        assert quality.objective_relative_standard_error is not None
        assert quality.gradient_error_norm_estimate is not None
        assert quality.gradient_error_norm_estimate >= 0.0
        assert quality.gradient_relative_error_estimate is not None
        assert quality.predicted_count_relative_error_estimate is not None
        assert quality.gradient_cosine_lower_estimate is not None
        assert result.fidelity.quality_groups_completed == 4


def test_too_few_selected_shards_reports_insufficient_sample():
    with jax.enable_x64():
        item = problem(likelihood=GravityLikelihood.POISSON)
        matrix = jnp.asarray(item.operator.matrix)
        context = GravityFidelityContext(
            gravity_fidelity_problem_identity(item),
            (GravityFidelityShard("only", int(np.count_nonzero(matrix)), matrix.nbytes),),
            (MatrixShard("only", matrix),),
        )
        result = gravity_value_and_gradient_progressive(
            np.zeros(3),
            problem=item,
            fidelity=GravityFidelityRequest(effort_percent=10, quality_groups=4),
            context=context,
        )
        assert result.quality.reliability == "insufficient_sample"
        assert result.quality.quality_score == 0.0
        assert result.quality.objective_standard_error is None
        assert result.quality.gradient_relative_error_estimate is None
        assert any("Fewer than two" in warning for warning in result.quality.warnings)


def test_missing_measurements_reduce_coverage_and_quality():
    with jax.enable_x64():
        item = problem(likelihood=GravityLikelihood.POISSON)
        matrix = np.asarray(item.operator.matrix)
        products = []
        metadata = []
        for row in range(matrix.shape[0]):
            shard_matrix = np.zeros_like(matrix)
            shard_matrix[row] = matrix[row]
            identifier = f"row-{row}"
            products.append(MatrixShard(identifier, jnp.asarray(shard_matrix)))
            metadata.append(
                GravityFidelityShard(
                    identifier,
                    max(1, int(np.count_nonzero(shard_matrix))),
                    shard_matrix.nbytes,
                )
            )
        context = GravityFidelityContext(
            gravity_fidelity_problem_identity(item), tuple(metadata), tuple(products)
        )
        result = gravity_value_and_gradient_progressive(
            np.asarray((0.1, 0.2, 0.8)),
            problem=item,
            fidelity=GravityFidelityRequest(effort_percent=1, seed=3),
            context=context,
        )
        assert result.quality.measurement_coverage_fraction == pytest.approx(1.0 / 3.0)
        assert result.quality.quality_score == 0.0
        assert any("does not cover" in warning for warning in result.quality.warnings)


def test_higher_effort_reduces_median_actual_error_across_seeds():
    with jax.enable_x64():
        item = problem()
        context = split_context(item, count=20)
        raw = np.asarray((0.2, -0.1, 1.0))
        exact = gravity_value_and_gradient_progressive(
            raw,
            problem=item,
            fidelity=GravityFidelityRequest(effort_percent=100),
            context=context,
        )
        errors: dict[int, list[tuple[float, float]]] = {20: [], 75: []}
        for effort in errors:
            for seed in range(30):
                approximate = gravity_value_and_gradient_progressive(
                    raw,
                    problem=item,
                    fidelity=GravityFidelityRequest(
                        effort_percent=effort, seed=seed, quality_groups=4
                    ),
                    context=context,
                )
                objective_error = abs(
                    float(approximate.evaluation.objective)
                    - float(exact.evaluation.objective)
                )
                gradient_error = float(
                    np.linalg.norm(
                        np.asarray(approximate.gradient) - np.asarray(exact.gradient)
                    )
                )
                errors[effort].append((objective_error, gradient_error))
        assert np.median(np.asarray(errors[75])[:, 0]) <= np.median(
            np.asarray(errors[20])[:, 0]
        )
        assert np.median(np.asarray(errors[75])[:, 1]) <= np.median(
            np.asarray(errors[20])[:, 1]
        )
