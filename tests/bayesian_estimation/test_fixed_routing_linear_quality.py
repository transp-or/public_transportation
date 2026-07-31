from __future__ import annotations

import math

import numpy as np
import pytest

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


def problem(
    matrix,
    *,
    observations=None,
    lower=None,
    upper=None,
    prior=None,
    strength=None,
    selection=None,
) -> FixedRoutingLinearProblem:
    matrix = np.asarray(matrix, dtype=float)
    rows, columns = matrix.shape
    prior = np.ones(columns) if prior is None else np.asarray(prior, dtype=float)
    blocks = () if strength is None else (ridge_to_prior(prior, strength=strength),)
    if selection is None:
        selection = "configured" if blocks else "none"
    return FixedRoutingLinearProblem(
        measurement_operator=matrix,
        fixed_measurement_offset=np.zeros(rows),
        observations=(
            np.ones(rows) if observations is None else np.asarray(observations)
        ),
        observation_weights=np.ones(rows),
        prior_demand=prior,
        lower_bounds=(np.full(columns, -np.inf) if lower is None else np.asarray(lower)),
        upper_bounds=(np.full(columns, np.inf) if upper is None else np.asarray(upper)),
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        regularization_selection=selection,
        regularization_blocks=blocks,
    )


def test_fully_observed_identity_is_entirely_data_informed():
    instance = problem(np.eye(3))
    quality = analyze_linear_estimate_quality(instance, [1.0, 1.0, 1.0])

    assert quality.measurement_rank == 3
    assert quality.measurement_nullity == 0
    assert quality.combined_rank == 3
    assert quality.combined_nullity == 0
    assert quality.measurement_condition_estimate == pytest.approx(1.0)
    np.testing.assert_allclose(quality.data_resolution, np.eye(3))
    np.testing.assert_allclose(quality.regularization_resolution, np.zeros((3, 3)))
    np.testing.assert_allclose(quality.data_resolution_score, np.ones(3))
    np.testing.assert_allclose(quality.regularization_reliance_score, np.zeros(3))
    assert quality.effective_data_degrees_of_freedom == pytest.approx(3.0)
    np.testing.assert_allclose(quality.data_mode_fractions, np.ones(3))
    assert quality.classifications == ("data_informed",) * 3


def test_unobserved_variable_is_weakly_identified():
    instance = problem([[1.0, 0.0], [2.0, 0.0]])
    quality = analyze_linear_estimate_quality(instance, [1.0, 1.0])

    assert quality.measurement_rank == 1
    assert quality.measurement_nullity == 1
    assert quality.combined_nullity == 1
    assert math.isinf(quality.measurement_condition_estimate)
    np.testing.assert_allclose(quality.null_space_participation, [0.0, 1.0])
    assert quality.classifications == ("data_informed", "weakly_identified")


def test_duplicate_columns_expose_unidentifiable_contrast():
    instance = problem([[1.0, 1.0], [2.0, 2.0]])
    quality = analyze_linear_estimate_quality(instance, [0.5, 0.5])

    assert quality.measurement_rank == 1
    assert quality.combined_nullity == 1
    np.testing.assert_allclose(
        quality.null_space_participation, [0.5, 0.5], atol=1e-12
    )
    assert quality.classifications == ("weakly_identified", "weakly_identified")


def test_strong_and_weak_ridge_change_data_prior_resolution():
    strong = analyze_linear_estimate_quality(
        problem(np.eye(2), strength=99.0), [1.0, 1.0]
    )
    weak = analyze_linear_estimate_quality(
        problem(np.eye(2), strength=0.01), [1.0, 1.0]
    )

    np.testing.assert_allclose(strong.data_resolution_score, [0.01, 0.01])
    np.testing.assert_allclose(strong.regularization_reliance_score, [0.99, 0.99])
    assert strong.classifications == ("regularization_dominated",) * 2
    np.testing.assert_allclose(weak.data_resolution_score, [1 / 1.01] * 2)
    assert weak.classifications == ("data_informed",) * 2
    assert strong.resolution_closure_inf_norm <= 1e-12
    np.testing.assert_allclose(
        strong.data_resolution + strong.regularization_resolution,
        np.eye(2),
        atol=1e-12,
    )


def test_bound_activity_takes_precedence_over_resolution_classification():
    instance = problem(
        np.eye(3),
        observations=[0.0, 5.0, 2.0],
        lower=[0.0, 0.0, 2.0],
        upper=[10.0, 4.0, 2.0],
        prior=[1.0, 2.0, 2.0],
    )
    quality = analyze_linear_estimate_quality(instance, [0.0, 4.0, 2.0])

    assert quality.classifications == (
        "lower_bound_active",
        "upper_bound_active",
        "fixed_by_bounds",
    )
    assert quality.free_indices.size == 0
    assert np.all(np.isnan(quality.data_resolution_score))


def test_unspecified_regularization_still_allows_preselection_analysis():
    instance = problem([[1.0, 0.0], [0.0, 0.0]], selection="unspecified")
    quality = analyze_linear_estimate_quality(instance, [1.0, 1.0])
    assert quality.measurement_rank == 1
    assert quality.combined_nullity == 1


def test_data_modes_report_data_fractions_between_zero_and_one():
    instance = problem([[1.0, 0.5], [0.0, 2.0]], strength=2.0)
    quality = analyze_linear_estimate_quality(instance, [1.0, 1.0])
    assert np.all(quality.data_mode_fractions >= 0.0)
    assert np.all(quality.data_mode_fractions <= 1.0)
    assert np.all(np.diff(quality.data_mode_fractions) <= 0.0)
    assert quality.data_modes.shape == (2, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data_informed_threshold": -0.1},
        {"regularization_dominated_threshold": 0.9, "data_informed_threshold": 0.8},
        {"null_participation_tolerance": -1.0},
    ],
)
def test_quality_rejects_invalid_thresholds(kwargs):
    with pytest.raises(ValueError):
        analyze_linear_estimate_quality(problem(np.eye(2)), [1.0, 1.0], **kwargs)
