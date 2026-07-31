from __future__ import annotations

import numpy as np
import pytest

from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.fixed_routing_linear_recommendation import (
    RegularizationSelectionRequiredError,
    recommend_linear_regularization,
    require_explicit_regularization_selection,
)


def problem(
    matrix,
    *,
    observations=None,
    prior=None,
    lower=None,
    upper=None,
    selection="unspecified",
) -> FixedRoutingLinearProblem:
    matrix = np.asarray(matrix, dtype=float)
    rows, columns = matrix.shape
    return FixedRoutingLinearProblem(
        measurement_operator=matrix,
        fixed_measurement_offset=np.zeros(rows),
        observations=(
            np.ones(rows) if observations is None else np.asarray(observations)
        ),
        observation_weights=np.ones(rows),
        prior_demand=(np.ones(columns) if prior is None else np.asarray(prior)),
        lower_bounds=(np.full(columns, -np.inf) if lower is None else np.asarray(lower)),
        upper_bounds=(np.full(columns, np.inf) if upper is None else np.asarray(upper)),
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        regularization_selection=selection,
    )


def option(recommendation, name):
    return next(item for item in recommendation.options if item.name == name)


def test_full_rank_well_conditioned_problem_recommends_explicit_none():
    instance = problem(np.eye(3))
    recommendation = recommend_linear_regularization(instance)

    assert recommendation.status == "none_is_reasonable"
    assert recommendation.recommended_options == ("none",)
    assert option(recommendation, "none").requires_strength is False
    assert recommendation.automatic_selection_applied is False
    assert instance.regularization_selection == "unspecified"


def test_rank_deficiency_requires_prior_regularization_for_uniqueness():
    recommendation = recommend_linear_regularization(
        problem([[1.0, 1.0], [2.0, 2.0]])
    )
    assert recommendation.status == "regularization_required_for_uniqueness"
    assert recommendation.quality.measurement_nullity == 1
    assert recommendation.recommended_options == ("ridge_to_prior",)
    assert option(recommendation, "ridge_to_prior").requires_strength
    assert not option(recommendation, "none").recommended


def test_ill_conditioning_recommends_regularization_before_numerical_rank_loss():
    recommendation = recommend_linear_regularization(
        problem([[1.0, 0.0], [0.0, 1.0e-10]])
    )
    assert recommendation.quality.measurement_nullity == 0
    assert recommendation.status == "regularization_recommended"
    assert recommendation.recommended_options == ("ridge_to_prior",)


def test_heterogeneous_prior_recommends_scaled_ridge():
    recommendation = recommend_linear_regularization(
        problem(np.eye(3), prior=[0.0, 0.01, 100.0])
    )
    assert recommendation.status == "scaling_recommended"
    assert recommendation.recommended_options == ("scaled_ridge_to_prior",)


def test_rank_deficiency_and_heterogeneous_prior_prefer_scaled_ridge():
    recommendation = recommend_linear_regularization(
        problem([[1.0, 1.0], [2.0, 2.0]], prior=[0.01, 100.0])
    )
    assert recommendation.status == "regularization_required_for_uniqueness"
    assert recommendation.recommended_options == ("scaled_ridge_to_prior",)


def test_preselection_analysis_does_not_remove_zero_prior_cell_at_lower_bound():
    recommendation = recommend_linear_regularization(
        problem(
            [[0.0, 1.0]],
            prior=[0.0, 1.0],
            lower=[0.0, 0.0],
            upper=[np.inf, np.inf],
        )
    )
    assert recommendation.quality.free_indices.size == 2
    assert recommendation.quality.measurement_nullity == 1
    assert recommendation.status == "regularization_required_for_uniqueness"


def test_post_estimation_bound_dominance_is_reported_separately():
    instance = problem(
        np.eye(3),
        observations=[0.0, 1.0, 2.0],
        prior=[1.0, 1.0, 1.0],
        lower=[0.0, 0.0, 0.0],
        upper=[1.0, 1.0, 5.0],
    )
    recommendation = recommend_linear_regularization(
        instance, demand=[0.0, 1.0, 2.0]
    )
    assert recommendation.status == "bounds_are_dominant"
    assert "bounds" in recommendation.reasons[0]
    assert recommendation.automatic_selection_applied is False


def test_missing_selection_raises_exception_carrying_recommendation():
    instance = problem([[1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(RegularizationSelectionRequiredError) as captured:
        require_explicit_regularization_selection(instance)
    assert captured.value.recommendation.status == (
        "regularization_required_for_uniqueness"
    )
    assert "ridge_to_prior" in str(captured.value)
    assert instance.regularization_selection == "unspecified"


@pytest.mark.parametrize("selection", ["none"])
def test_explicit_selection_passes_through_unchanged(selection):
    instance = problem(np.eye(2), selection=selection)
    assert require_explicit_regularization_selection(instance) is instance


@pytest.mark.parametrize(
    "kwargs",
    [
        {"condition_threshold": 1.0},
        {"scale_ratio_threshold": 1.0},
        {"bound_dominance_fraction": -0.1},
        {"bound_dominance_fraction": 1.1},
    ],
)
def test_recommendation_rejects_invalid_thresholds(kwargs):
    with pytest.raises(ValueError):
        recommend_linear_regularization(problem(np.eye(2)), **kwargs)
