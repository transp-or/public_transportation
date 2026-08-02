from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse as jsparse
from scipy.special import gammaln, xlogy

from public_transportation.inference.fixed_routing_measurement_operator import (
    FixedRoutingMeasurementOperator,
    MeasurementOperatorMetrics,
)
from public_transportation.inference.gravity import (
    GravityFeatures,
    GravityGradientStrategy,
    GravityLikelihood,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    evaluate_gravity_objective,
    gravity_value_and_gradient,
    predict_gravity_measurements,
)


def features(dtype=np.float64) -> GravityFeatures:
    return GravityFeatures(
        canonical_od_index=np.arange(4),
        origin_index=np.asarray((0, 0, 1, 1)),
        destination_index=np.asarray((0, 1, 0, 1)),
        departure_time_index=np.asarray((0, 0, 0, 0)),
        origin_time_group_index=np.asarray((0, 0, 1, 1)),
        journey_time=np.asarray((5.0, 15.0, 12.0, 7.0), dtype=dtype),
        transfer_count=np.asarray((0, 1, 2, 0)),
        structural_feasible=np.asarray((True, False, True, True)),
        origin_time_totals=np.asarray((20.0, 30.0), dtype=dtype),
        destination_attractiveness=np.asarray((1.0, 3.0, 2.0, 1.0), dtype=dtype),
        num_origins=2,
        num_destinations=2,
        num_departure_times=1,
        od_layout_fingerprint="gravity-layout",
        journey_time_scale=10.0,
    )


def operator(*, sparse=False, cache_hit=False) -> FixedRoutingMeasurementOperator:
    dense = jnp.asarray(
        ((1.0, 10.0, 0.5, 0.0), (0.0, 2.0, 1.0, 1.5), (0.2, 0.0, 0.0, 1.0)),
        dtype=jnp.float64,
    )
    matrix = jsparse.BCOO.fromdense(dense) if sparse else dense
    return FixedRoutingMeasurementOperator(
        matrix=matrix,
        fixed_measurement_offset=jnp.asarray((2.0, 0.0, 1.0), dtype=dense.dtype),
        representation="bcoo" if sparse else "dense",
        num_active_od=4,
        num_free_od=4,
        num_measurements=3,
        od_layout_fingerprint="gravity-layout",
        compact_layout_fingerprint="gravity-layout",
        assignment_fingerprint="assignment",
        graph_fingerprint="graph",
        mapping_fingerprint="mapping",
        theta=1.0,
        dtype="float64",
        metrics=MeasurementOperatorMetrics(
            construction_seconds=0.0,
            dense_bytes=int(dense.nbytes),
            stored_bytes=int(dense.nbytes),
            peak_construction_bytes=0,
            nonzero_entries=8,
            total_entries=12,
            density=8 / 12,
            chunk_size=4,
            cache_hit=cache_hit,
        ),
    )


def problem(
    *,
    likelihood=GravityLikelihood.NEGATIVE_BINOMIAL,
    sparse=False,
    cache_hit=False,
    observations=(19.0, 31.0, 9.0),
) -> GravityObjectiveProblem:
    return GravityObjectiveProblem(
        features=features(),
        parameter_layout=GravityParameterLayout(GravityModelSpecification()),
        operator=operator(sparse=sparse, cache_hit=cache_hit),
        observations=np.asarray(observations),
        likelihood=likelihood,
    )


def finite_difference(raw, item, step=1.0e-5):
    result = np.empty_like(raw)
    for index in range(raw.size):
        delta = np.zeros_like(raw)
        delta[index] = step
        plus = evaluate_gravity_objective(raw + delta, problem=item).objective
        minus = evaluate_gravity_objective(raw - delta, problem=item).objective
        result[index] = (float(plus) - float(minus)) / (2 * step)
    return result


@pytest.mark.parametrize(
    "likelihood", (GravityLikelihood.POISSON, GravityLikelihood.NEGATIVE_BINOMIAL)
)
def test_batched_forward_adjoint_autodiff_and_finite_difference_agree(likelihood):
    with jax.enable_x64():
        item = problem(likelihood=likelihood)
        raw = np.asarray((-0.2, 0.4, 1.5))
        forward_evaluation, forward = gravity_value_and_gradient(
            raw, problem=item, strategy=GravityGradientStrategy.BATCHED_FORWARD
        )
        adjoint_evaluation, adjoint = gravity_value_and_gradient(
            raw, problem=item, strategy=GravityGradientStrategy.ADJOINT
        )
        automatic = jax.grad(
            lambda value: evaluate_gravity_objective(value, problem=item).objective
        )(jnp.asarray(raw))
        numerical = finite_difference(raw, item)
        np.testing.assert_allclose(forward, adjoint, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(forward, automatic, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(forward, numerical, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            forward_evaluation.measurement_mean, adjoint_evaluation.measurement_mean
        )


def test_poisson_and_negative_binomial_match_reference_formulas():
    with jax.enable_x64():
        raw = np.asarray((0.1, -0.3, 0.8))
        poisson_problem = problem(likelihood=GravityLikelihood.POISSON)
        poisson = evaluate_gravity_objective(raw, problem=poisson_problem)
        mu = np.asarray(poisson.measurement_mean)
        y = poisson_problem.observations
        expected_poisson = np.sum(xlogy(y, mu) - mu - gammaln(y + 1))
        assert float(poisson.data_log_likelihood) == pytest.approx(expected_poisson)

        nb_problem = problem(likelihood=GravityLikelihood.NEGATIVE_BINOMIAL)
        nb = evaluate_gravity_objective(raw, problem=nb_problem)
        r = float(nb_problem.parameter_layout.transform(raw).dispersion)
        expected_nb = np.sum(
            gammaln(y + r)
            - gammaln(r)
            - gammaln(y + 1)
            - r * np.log1p(mu / r)
            + xlogy(y, mu)
            - xlogy(y, r + mu)
        )
        assert float(nb.data_log_likelihood) == pytest.approx(expected_nb)
        assert float(nb.objective) == pytest.approx(-expected_nb)
        assert float(nb.regularization) == 0.0


def test_negative_binomial_approaches_poisson_for_large_dispersion():
    with jax.enable_x64():
        item = problem()
        layout = item.parameter_layout
        raw = layout.raw_from_physical((0.5, 0.5, 1.0e8))
        nb = evaluate_gravity_objective(raw, problem=item)
        poisson = evaluate_gravity_objective(
            raw, problem=replace(item, likelihood=GravityLikelihood.POISSON)
        )
        assert float(nb.data_log_likelihood) == pytest.approx(
            float(poisson.data_log_likelihood), rel=2e-6
        )


@pytest.mark.parametrize("likelihood", list(GravityLikelihood))
def test_zero_counts_have_finite_objective_and_gradient(likelihood):
    with jax.enable_x64():
        item = problem(likelihood=likelihood, observations=(0.0, 0.0, 0.0))
        evaluation, gradient = gravity_value_and_gradient(
            np.zeros(3), problem=item, strategy=GravityGradientStrategy.ADJOINT
        )
        assert np.isfinite(float(evaluation.objective))
        assert np.all(np.isfinite(np.asarray(gradient)))


def test_structural_zero_and_fixed_positive_offset_are_preserved():
    with jax.enable_x64():
        item = problem(likelihood=GravityLikelihood.POISSON)
        mean, demand = predict_gravity_measurements(np.zeros(3), problem=item)
        demand_np = np.asarray(demand)
        assert demand_np[1] == 0.0
        expected = np.asarray(item.operator.matrix) @ demand_np + np.asarray(
            item.operator.fixed_measurement_offset
        )
        np.testing.assert_allclose(mean, expected)


def test_dense_bcoo_and_cache_status_do_not_change_results():
    with jax.enable_x64():
        raw = np.asarray((0.2, -0.1, 1.0))
        dense = evaluate_gravity_objective(raw, problem=problem())
        sparse = evaluate_gravity_objective(raw, problem=problem(sparse=True))
        cached = evaluate_gravity_objective(raw, problem=problem(cache_hit=True))
        np.testing.assert_allclose(dense.measurement_mean, sparse.measurement_mean)
        np.testing.assert_allclose(dense.measurement_mean, cached.measurement_mean)
        assert float(dense.objective) == pytest.approx(float(sparse.objective))


def test_calibration_mask_decomposes_used_and_excluded_measurements():
    with jax.enable_x64():
        item = replace(problem(), calibration_mask=np.asarray((True, False, True)))
        evaluation = evaluate_gravity_objective(np.zeros(3), problem=item)
        assert int(evaluation.calibration_measurements) == 2
        assert int(evaluation.excluded_measurements) == 1


def test_objective_and_direct_gradient_compile_with_problem_as_fixed_context():
    with jax.enable_x64():
        item = problem()
        compiled = jax.jit(
            jax.value_and_grad(
                lambda raw: evaluate_gravity_objective(raw, problem=item).objective
            )
        )
        value, gradient = compiled(jnp.zeros(3, dtype=jnp.float64))
        value.block_until_ready()
        assert np.isfinite(float(value))
        assert np.all(np.isfinite(np.asarray(gradient)))


class _RecordingOperator:
    """Protocol-only operator proving gravity never requires `.matrix`."""

    def __init__(self, base: FixedRoutingMeasurementOperator):
        self.base = base
        self.num_free_od = base.num_free_od
        self.num_measurements = base.num_measurements
        self.compact_layout_fingerprint = base.compact_layout_fingerprint
        self.fixed_measurement_offset = base.fixed_measurement_offset
        self.is_matrix_free = False
        self.forward_calls = 0
        self.transpose_calls = 0
        self.matmat_calls = 0

    def jax_matvec(self, vector):
        self.forward_calls += 1
        return self.base.jax_matvec(vector)

    def jax_rmatvec(self, vector):
        self.transpose_calls += 1
        return self.base.jax_rmatvec(vector)

    def jax_matmat(self, matrix):
        self.matmat_calls += 1
        return self.base.jax_matmat(matrix)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    (
        (GravityGradientStrategy.BATCHED_FORWARD, (1, 0, 1)),
        (GravityGradientStrategy.ADJOINT, (1, 1, 0)),
    ),
)
def test_value_and_gradient_routes_once_with_protocol_only_operator(
    strategy, expected
):
    with jax.enable_x64():
        original = problem()
        recording = _RecordingOperator(original.operator)
        item = replace(original, operator=recording)
        evaluation, gradient = gravity_value_and_gradient(
            np.asarray((0.2, -0.1, 1.0)), problem=item, strategy=strategy
        )
        assert np.isfinite(float(evaluation.objective))
        assert np.all(np.isfinite(np.asarray(gradient)))
        assert (
            recording.forward_calls,
            recording.transpose_calls,
            recording.matmat_calls,
        ) == expected
