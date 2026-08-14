from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize
from types import SimpleNamespace

from public_transportation.inference.reduced_od import (
    ConditionalGravityFeatures,
    MinimalGravityParameterLayout,
    MinimalGravityProblem,
    MinimalGravitySpecification,
    build_reduced_response_operator_from_coo,
    build_conditional_gravity_features,
    default_minimal_gravity_raw_parameters,
    evaluate_minimal_gravity_objective,
    generate_minimal_gravity_demand,
    initialize_minimal_gravity_from_entropy,
    minimal_gravity_value_and_gradient,
    transform_minimal_gravity_parameters,
)
from public_transportation.preprocessing.reduced_od import ResponseCellKey


def _features() -> ConditionalGravityFeatures:
    return ConditionalGravityFeatures(
        cell_keys=(
            ResponseCellKey("A", "B", "P"),
            ResponseCellKey("A", "C", "P"),
            ResponseCellKey("A", "D", "P"),
        ),
        origin_time_group_index=np.asarray([0, 0, 0]),
        destination_index=np.asarray([0, 1, 2]),
        journey_time_seconds=np.asarray([600.0, 1200.0, 2100.0]),
        transfer_count=np.asarray([0.0, 2.0, 1.0]),
        destination_attractiveness=np.asarray([1.0, 1.5, 0.8]),
        baseline_productions=np.asarray([100.0]),
        origin_time_group_keys=(("A", "P"),),
        destination_ids=("B", "C", "D"),
    )


def _operator(offset: np.ndarray | None = None):
    return build_reduced_response_operator_from_coo(
        number_of_measurements=3,
        number_of_free_cells=3,
        measurement_index=np.asarray([0, 1, 2]),
        free_cell_index=np.asarray([0, 1, 2]),
        response_values=np.ones(3),
        fixed_offset=np.zeros(3) if offset is None else offset,
    )


def _problem(
    specification: MinimalGravitySpecification,
    observations: np.ndarray,
    *,
    production_basis: np.ndarray | None = None,
    offset: np.ndarray | None = None,
) -> MinimalGravityProblem:
    return MinimalGravityProblem(
        features=_features(),
        parameter_layout=MinimalGravityParameterLayout(specification),
        response_operator=_operator(offset),
        observations=observations,
        production_basis=production_basis,
    )


def test_grouped_softmax_normalizes_and_preserves_provided_production() -> None:
    specification = MinimalGravitySpecification()
    layout = MinimalGravityParameterLayout(specification)
    raw = default_minimal_gravity_raw_parameters(layout)
    generated = generate_minimal_gravity_demand(
        raw, problem=_problem(specification, np.ones(3))
    )
    np.testing.assert_allclose(np.asarray(generated.probabilities).sum(), 1.0)
    np.testing.assert_allclose(np.asarray(generated.demand).sum(), 100.0)
    np.testing.assert_allclose(np.asarray(generated.group_sums), [100.0])
    assert layout.size == 2


def test_estimated_destination_attractiveness_has_constrained_parameter_block() -> None:
    specification = MinimalGravitySpecification(
        destination_attractiveness_mode="estimated_basis",
        destination_attractiveness_basis_columns=2,
    )
    layout = MinimalGravityParameterLayout(specification)
    assert layout.size == 4
    assert layout.destination_attractiveness_slice == slice(2, 4)
    names = layout.raw_parameter_names(
        destination_attractiveness_basis_labels=("destination.deviation[0]", "destination.deviation[1]")
    )
    assert names[-2:] == ("destination.deviation[0]", "destination.deviation[1]")
    basis = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=np.float64
    )
    problem = MinimalGravityProblem(
        features=_features(),
        parameter_layout=layout,
        response_operator=_operator(),
        observations=np.ones(3),
        destination_attractiveness_basis=basis,
    )
    raw = default_minimal_gravity_raw_parameters(layout)
    raw[layout.destination_attractiveness_slice] = np.asarray([0.4, -0.2])
    generated = generate_minimal_gravity_demand(raw, problem=problem)
    assert np.all(np.isfinite(np.asarray(generated.demand)))
    assert not np.allclose(np.asarray(generated.probabilities), [1 / 3] * 3)


def test_feature_builder_aligns_free_cells_and_fixed_choice_summaries() -> None:
    keys = (
        ResponseCellKey("A", "B", "P"),
        ResponseCellKey("A", "C", "P"),
    )
    choice_sets = tuple(
        SimpleNamespace(
            origin_physical_stop_id=key.origin_physical_stop_id,
            destination_physical_stop_id=key.destination_physical_stop_id,
            origin_time_period_id=key.origin_time_period_id,
            alternatives=(
                SimpleNamespace(travel_seconds=600 + index * 300, transfers=index),
            ),
            initial_shares=(1.0,),
        )
        for index, key in enumerate(keys)
    )
    built = build_conditional_gravity_features(
        response=SimpleNamespace(free_cell_keys=keys),  # type: ignore[arg-type]
        journey_choices=SimpleNamespace(choice_sets=choice_sets),  # type: ignore[arg-type]
        productions={("A", "P"): 42.0},
        destination_attractiveness={("B", "P"): 1.0, ("C", "P"): 2.0},
    )
    assert built.cell_keys == keys
    np.testing.assert_array_equal(built.origin_time_group_index, [0, 0])
    np.testing.assert_allclose(built.journey_time_seconds, [600.0, 900.0])
    np.testing.assert_allclose(built.transfer_count, [0.0, 1.0])
    np.testing.assert_allclose(built.baseline_productions, [42.0])


def test_extreme_utilities_remain_finite() -> None:
    specification = MinimalGravitySpecification()
    problem = _problem(specification, np.ones(3))
    for raw in (np.asarray([100.0, -100.0]), np.asarray([-100.0, 100.0])):
        generated = generate_minimal_gravity_demand(raw, problem=problem)
        assert np.all(np.isfinite(np.asarray(generated.probabilities)))
        assert np.asarray(generated.probabilities).sum() == pytest.approx(1.0)


def test_poisson_objective_uses_fixed_offset_and_exact_jax_gradient() -> None:
    specification = MinimalGravitySpecification(likelihood="poisson")
    layout = MinimalGravityParameterLayout(specification)
    raw = default_minimal_gravity_raw_parameters(
        layout, beta_time=0.8, beta_transfer=1.3
    )
    offset = np.asarray([2.0, 3.0, 4.0])
    provisional = _problem(specification, np.ones(3), offset=offset)
    mean = np.asarray(
        evaluate_minimal_gravity_objective(raw, problem=provisional).measurement_mean
    )
    problem = _problem(specification, mean, offset=offset)
    evaluated_raw = raw + np.asarray([0.2, -0.1])
    evaluation, gradient = minimal_gravity_value_and_gradient(
        evaluated_raw, problem=problem
    )
    assert np.all(np.isfinite(np.asarray(gradient)))
    np.testing.assert_allclose(
        np.asarray(evaluation.measurement_mean),
        np.asarray(evaluation.demand) + offset,
        rtol=1e-6,
    )
    finite_difference = np.empty(layout.size)
    step = 1.0e-2
    for index in range(layout.size):
        direction = np.zeros(layout.size)
        direction[index] = step
        plus = float(
            evaluate_minimal_gravity_objective(
                evaluated_raw + direction, problem=problem
            ).objective
        )
        minus = float(
            evaluate_minimal_gravity_objective(
                evaluated_raw - direction, problem=problem
            ).objective
        )
        finite_difference[index] = (plus - minus) / (2 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-2, atol=2e-3)


def test_negative_binomial_and_estimated_production_basis() -> None:
    specification = MinimalGravitySpecification(
        likelihood="negative_binomial",
        production_mode="estimated_basis",
        production_basis_columns=1,
    )
    layout = MinimalGravityParameterLayout(specification)
    basis = np.ones((1, 1))
    raw = default_minimal_gravity_raw_parameters(layout, dispersion=20.0)
    raw[layout.production_slice] = np.log(1.5)
    problem = _problem(
        specification, np.asarray([20.0, 30.0, 40.0]), production_basis=basis
    )
    evaluation, gradient = minimal_gravity_value_and_gradient(raw, problem=problem)
    transformed = transform_minimal_gravity_parameters(raw, layout=layout)
    assert transformed.dispersion is not None
    assert float(transformed.dispersion) == pytest.approx(20.0, rel=1e-5)
    np.testing.assert_allclose(evaluation.productions, [150.0], rtol=1e-6)
    assert layout.size == 4
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_entropy_plan_initializes_gravity_coefficients() -> None:
    specification = MinimalGravitySpecification()
    layout = MinimalGravityParameterLayout(specification)
    true_raw = default_minimal_gravity_raw_parameters(
        layout, beta_time=0.7, beta_transfer=1.4
    )
    problem = _problem(specification, np.ones(3))
    flow = np.asarray(generate_minimal_gravity_demand(true_raw, problem=problem).demand)
    initialized = initialize_minimal_gravity_from_entropy(
        features=problem.features,
        entropy_cell_flow=flow,
        layout=layout,
    )
    transformed = transform_minimal_gravity_parameters(initialized, layout=layout)
    assert float(transformed.beta_time) == pytest.approx(0.7, rel=2e-4)
    assert float(transformed.beta_transfer) == pytest.approx(1.4, rel=2e-4)


def test_synthetic_poisson_truth_is_recovered_with_two_parameters() -> None:
    specification = MinimalGravitySpecification()
    layout = MinimalGravityParameterLayout(specification)
    true_raw = default_minimal_gravity_raw_parameters(
        layout, beta_time=0.9, beta_transfer=1.1
    )
    provisional = _problem(specification, np.ones(3))
    observations = np.asarray(
        evaluate_minimal_gravity_objective(
            true_raw, problem=provisional
        ).measurement_mean
    )
    problem = _problem(specification, observations)

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        evaluation, gradient = minimal_gravity_value_and_gradient(value, problem=problem)
        return float(evaluation.objective), np.asarray(gradient, dtype=float)

    result = minimize(
        objective,
        np.asarray([-0.5, -0.5]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8},
    )
    fitted = transform_minimal_gravity_parameters(result.x, layout=layout)
    assert float(fitted.beta_time) == pytest.approx(0.9, rel=2e-3)
    assert float(fitted.beta_transfer) == pytest.approx(1.1, rel=2e-3)
    assert np.linalg.norm(result.jac) < 1.0e-4
    assert layout.size == 2 < problem.features.number_of_cells


def test_jitted_value_and_gradient_are_finite() -> None:
    specification = MinimalGravitySpecification()
    layout = MinimalGravityParameterLayout(specification)
    problem = _problem(specification, np.asarray([20.0, 30.0, 50.0]))
    raw = jnp.asarray(default_minimal_gravity_raw_parameters(layout))
    value, gradient = jax.jit(
        jax.value_and_grad(
            lambda parameters: evaluate_minimal_gravity_objective(
                parameters, problem=problem
            ).objective
        )
    )(raw)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
