from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from public_transportation.inference.fixed_routing_linear_dense_solver import (
    solve_dense_reference,
)
from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.fixed_routing_linear_quality import (
    analyze_linear_estimate_quality,
)
from public_transportation.inference.fixed_routing_linear_regularization import (
    ridge_to_prior,
)
from public_transportation.inference.fixed_routing_linear_scalable_quality import (
    ScalableQualityConfig,
    analyze_linear_estimate_quality_scalable,
)


def _problem(matrix, *, ridge_strength=1.0):
    matrix = np.asarray(matrix, dtype=float)
    columns = matrix.shape[1]
    prior = np.ones(columns)
    return FixedRoutingLinearProblem(
        measurement_operator=matrix,
        fixed_measurement_offset=np.zeros(matrix.shape[0]),
        observations=matrix @ prior,
        observation_weights=np.ones(matrix.shape[0]),
        prior_demand=prior,
        lower_bounds=np.full(columns, -np.inf),
        upper_bounds=np.full(columns, np.inf),
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        regularization_selection="configured",
        regularization_blocks=(ridge_to_prior(prior, strength=ridge_strength),),
    )


def test_diagonal_resolution_estimates_are_exact_for_every_probe():
    problem = _problem(np.diag([1.0, 2.0, 4.0]), ridge_strength=2.0)
    solved = solve_dense_reference(problem)
    exact = analyze_linear_estimate_quality(problem, solved.demand)
    approximate = analyze_linear_estimate_quality_scalable(
        problem,
        solved.demand,
        config=ScalableQualityConfig(
            resolution_samples=4,
            smallest_singular_values=2,
        ),
    )

    assert approximate.spectral_converged
    assert approximate.spectral_message == "converged"
    assert approximate.resolution_converged_samples == 4
    assert approximate.resolution_failed_samples == 0
    np.testing.assert_allclose(
        approximate.data_resolution_score_estimate,
        exact.data_resolution_score,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(approximate.data_resolution_standard_error, 0.0)
    assert approximate.effective_data_degrees_of_freedom_estimate == pytest.approx(
        exact.effective_data_degrees_of_freedom
    )
    assert approximate.effective_data_degrees_of_freedom_standard_error == 0.0
    assert approximate.largest_singular_value_estimate == pytest.approx(4.0)
    assert approximate.condition_estimate == pytest.approx(4.0)


def test_sampled_smallest_spectrum_detects_rank_deficiency():
    problem = _problem([[1.0, 1.0], [2.0, 2.0]], ridge_strength=0.5)
    solved = solve_dense_reference(problem)
    quality = analyze_linear_estimate_quality_scalable(
        problem,
        solved.demand,
        config=ScalableQualityConfig(
            resolution_samples=8,
            smallest_singular_values=1,
            rank_relative_tolerance=1e-7,
        ),
    )

    assert quality.spectral_converged
    assert quality.estimated_nullity_lower_bound == 1
    assert quality.estimated_rank_upper_bound == 1
    assert quality.condition_estimate == np.inf
    assert quality.smallest_singular_value_estimates[0] <= quality.rank_tolerance


def test_unregularized_sampled_null_space_is_not_mislabeled_as_prior_reliance():
    configured = _problem([[1.0, 1.0], [2.0, 2.0]], ridge_strength=0.5)
    problem = replace(
        configured,
        regularization_selection="none",
        regularization_blocks=(),
    )
    solved = solve_dense_reference(problem)
    quality = analyze_linear_estimate_quality_scalable(
        problem,
        solved.demand,
        config=ScalableQualityConfig(
            resolution_samples=4,
            smallest_singular_values=1,
            rank_relative_tolerance=1e-7,
        ),
    )

    assert quality.estimated_nullity_lower_bound == 1
    assert quality.resolution_converged_samples == 0
    assert quality.resolution_failed_samples == 4
    assert np.all(np.isnan(quality.data_resolution_score_estimate))
    assert np.all(np.isnan(quality.regularization_reliance_score_estimate))
    assert quality.classifications == ("weakly_identified", "weakly_identified")


def test_randomized_resolution_is_reproducible_and_reports_sampling_error():
    problem = _problem([[1.0, 0.4], [0.2, 1.0], [1.0, 1.0]], ridge_strength=1.0)
    solved = solve_dense_reference(problem)
    config = ScalableQualityConfig(resolution_samples=64, random_seed=42)
    first = analyze_linear_estimate_quality_scalable(
        problem, solved.demand, config=config
    )
    second = analyze_linear_estimate_quality_scalable(
        problem, solved.demand, config=config
    )
    exact = analyze_linear_estimate_quality(problem, solved.demand)

    np.testing.assert_array_equal(
        first.data_resolution_score_estimate,
        second.data_resolution_score_estimate,
    )
    np.testing.assert_array_equal(
        first.data_resolution_standard_error,
        second.data_resolution_standard_error,
    )
    assert np.all(first.data_resolution_standard_error >= 0.0)
    assert first.random_seed == 42
    assert (
        abs(
            first.effective_data_degrees_of_freedom_estimate
            - exact.effective_data_degrees_of_freedom
        )
        <= 4.0 * first.effective_data_degrees_of_freedom_standard_error + 1.0e-10
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"smallest_singular_values": 0},
        {"resolution_samples": 0},
        {"rank_relative_tolerance": 0.0},
        {"linear_solve_relative_tolerance": 0.0},
        {"linear_solve_max_iterations": 0},
        {"spectral_max_iterations": 0},
        {"regularization_dominated_threshold": 0.9, "data_informed_threshold": 0.1},
        {"classification_standard_error_multiplier": -1.0},
    ],
)
def test_scalable_quality_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ScalableQualityConfig(**kwargs)
