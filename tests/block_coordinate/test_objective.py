"""Tests for exact conditional objectives and supported priors."""

from __future__ import annotations

import numpy as np
import pytest

from public_transportation.inference.block_coordinate import (
    ColumnSelectedLinearOperator,
    ODBlock,
    UnsupportedConditionalPriorError,
    build_conditional_block_objective,
    initialize_incremental_state,
    prepare_separable_quadratic_prior,
    projected_gradient,
)
from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
    LinearRegularizationBlock,
)
from public_transportation.inference.fixed_routing_linear_regularization import (
    evaluate_linear_least_squares,
    ridge_to_prior,
    scaled_ridge_to_prior,
)


def _problem(*, regularization: str, blocks=()) -> FixedRoutingLinearProblem:
    return FixedRoutingLinearProblem(
        measurement_operator=np.array(
            [[1.0, 0.2, 0.0, 0.5], [0.0, 1.5, 0.3, 0.0], [0.4, 0.0, 0.8, 1.0]]
        ),
        fixed_measurement_offset=np.array([0.5, 1.0, 0.25]),
        observations=np.array([4.0, 3.0, 5.0]),
        observation_weights=np.array([1.0, 2.0, 0.5]),
        prior_demand=np.array([1.0, 2.0, 3.0, 4.0]),
        lower_bounds=np.array([0.0, 0.5, 0.0, 1.0]),
        upper_bounds=np.array([10.0, 8.0, np.inf, 9.0]),
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        regularization_selection=regularization,
        regularization_blocks=tuple(blocks),
    )


def _block(columns: tuple[int, ...]) -> ODBlock:
    return ODBlock(
        block_id="test-block",
        free_column_indices=columns,
        active_od_indices=columns,
        destination_group_indices=(0,),
        time_bin_ids=("t0",),
    )


@pytest.mark.parametrize("kind", ["none", "ridge", "scaled"])
def test_conditional_evaluation_equals_complete_objective_and_gradient(kind: str) -> None:
    prior_demand = np.array([1.0, 2.0, 3.0, 4.0])
    if kind == "none":
        problem = _problem(regularization="none")
    elif kind == "ridge":
        problem = _problem(
            regularization="configured",
            blocks=(ridge_to_prior(prior_demand, strength=1.7),),
        )
    else:
        problem = _problem(
            regularization="configured",
            blocks=(
                scaled_ridge_to_prior(
                    prior_demand, [1.0, 2.0, 0.5, 4.0], strength=2.3
                ),
            ),
        )
    current = np.array([2.0, 1.5, 4.0, 3.0])
    state = initialize_incremental_state(
        problem.measurement_operator, current, problem.fixed_measurement_offset
    )
    columns = (1, 3)
    block = _block(columns)
    block_operator = ColumnSelectedLinearOperator(
        problem.measurement_operator, columns
    )
    prior = prepare_separable_quadratic_prior(problem)
    conditional = build_conditional_block_objective(
        problem, prior, state, block, block_operator
    )
    trial = np.array([2.25, 5.5])
    complete_trial = current.copy()
    complete_trial[list(columns)] = trial

    actual = conditional.evaluate(trial)
    expected = evaluate_linear_least_squares(problem, complete_trial)
    np.testing.assert_allclose(actual.prediction, expected.data_fit.prediction)
    assert actual.components.data == pytest.approx(expected.data_fit.objective)
    assert actual.components.prior == pytest.approx(
        sum(item.objective for item in expected.regularization), abs=1.0e-12
    )
    assert actual.objective == pytest.approx(expected.objective)
    np.testing.assert_allclose(actual.gradient, expected.gradient[list(columns)])


def test_conditional_gradient_matches_central_finite_difference() -> None:
    prior = np.array([1.0, 2.0, 3.0, 4.0])
    problem = _problem(
        regularization="configured",
        blocks=(scaled_ridge_to_prior(prior, [2.0, 3.0, 4.0, 5.0], strength=3.0),),
    )
    flow = np.array([2.0, 2.5, 3.5, 4.5])
    state = initialize_incremental_state(
        problem.measurement_operator, flow, problem.fixed_measurement_offset
    )
    block = _block((0, 2))
    conditional = build_conditional_block_objective(
        problem,
        prepare_separable_quadratic_prior(problem),
        state,
        block,
        ColumnSelectedLinearOperator(problem.measurement_operator, (0, 2)),
    )
    local = np.array([1.7, 2.8])
    step = 1.0e-6
    finite_difference = np.empty(2)
    for index in range(2):
        direction = np.zeros(2)
        direction[index] = step
        finite_difference[index] = (
            conditional.evaluate(local + direction).objective
            - conditional.evaluate(local - direction).objective
        ) / (2.0 * step)
    np.testing.assert_allclose(
        conditional.evaluate(local).gradient,
        finite_difference,
        rtol=2.0e-9,
        atol=2.0e-9,
    )


def test_coupled_prior_is_rejected_but_zero_strength_is_harmless() -> None:
    coupled = LinearRegularizationBlock(
        "difference", np.array([[1.0, -1.0, 0.0, 0.0]]), [0.0], 2.0
    )
    with pytest.raises(UnsupportedConditionalPriorError, match="couples multiple"):
        prepare_separable_quadratic_prior(
            _problem(regularization="configured", blocks=(coupled,))
        )

    inactive = LinearRegularizationBlock(
        "inactive-difference", np.array([[1.0, -1.0, 0.0, 0.0]]), [0.0], 0.0
    )
    prepared = prepare_separable_quadratic_prior(
        _problem(regularization="configured", blocks=(inactive,))
    )
    np.testing.assert_array_equal(prepared.quadratic, np.zeros(4))
    np.testing.assert_array_equal(prepared.linear, np.zeros(4))
    assert prepared.objective([2.0, 3.0, 4.0, 5.0]) == 0.0


def test_unspecified_regularization_is_rejected() -> None:
    with pytest.raises(ValueError, match="explicitly select 'none'"):
        prepare_separable_quadratic_prior(_problem(regularization="unspecified"))


def test_projected_gradient_respects_active_bounds() -> None:
    flow = np.array([0.0, 0.0, 5.0, 5.0, 2.0])
    gradient = np.array([3.0, -2.0, -4.0, 1.5, 0.75])
    lower = np.zeros(5)
    upper = np.array([5.0, 5.0, 5.0, 5.0, np.inf])
    np.testing.assert_array_equal(
        projected_gradient(flow, gradient, lower, upper),
        np.array([0.0, -2.0, 0.0, 1.5, 0.75]),
    )

    with pytest.raises(ValueError, match="violates"):
        projected_gradient([-0.1], [1.0], [0.0], [2.0])

