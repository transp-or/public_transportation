from __future__ import annotations

from dataclasses import dataclass

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
    plan_gravity_fidelity,
)
from tests.gravity.test_phase2_objective import problem


@dataclass(frozen=True)
class MatrixShard:
    shard_id: str
    matrix: jax.Array

    def jax_matvec(self, vector):
        return self.matrix @ vector

    def jax_rmatvec(self, vector):
        return self.matrix.T @ vector


def split_context(item, *, count=8):
    matrix = np.asarray(item.operator.matrix)
    products = []
    metadata = []
    for index in range(count):
        shard_matrix = np.zeros_like(matrix)
        shard_matrix[:, index % matrix.shape[1]] = (
            matrix[:, index % matrix.shape[1]] / (count / matrix.shape[1])
        )
        identifier = f"shard-{index}"
        products.append(MatrixShard(identifier, jnp.asarray(shard_matrix)))
        metadata.append(
            GravityFidelityShard(
                identifier,
                support_entries=max(1, int(np.count_nonzero(shard_matrix))),
                routing_bytes=shard_matrix.nbytes,
                stratum="even" if index % 2 == 0 else "odd",
            )
        )
    np.testing.assert_allclose(sum(np.asarray(item.matrix) for item in products), matrix)
    return GravityFidelityContext(
        gravity_fidelity_problem_identity(item), tuple(metadata), tuple(products)
    )


def finite_difference(raw, item, request, context, step=1.0e-5):
    result = np.empty_like(raw)
    for index in range(raw.size):
        delta = np.zeros_like(raw)
        delta[index] = step
        plus = gravity_value_and_gradient_progressive(
            raw + delta, problem=item, fidelity=request, context=context
        ).evaluation.objective
        minus = gravity_value_and_gradient_progressive(
            raw - delta, problem=item, fidelity=request, context=context
        ).evaluation.objective
        result[index] = (float(plus) - float(minus)) / (2.0 * step)
    return result


@pytest.mark.parametrize(
    "likelihood", (GravityLikelihood.POISSON, GravityLikelihood.NEGATIVE_BINOMIAL)
)
def test_approximate_gradient_matches_same_approximate_objective(likelihood):
    with jax.enable_x64():
        item = problem(likelihood=likelihood)
        context = split_context(item)
        request = GravityFidelityRequest(effort_percent=35, seed=11)
        raw = np.asarray((0.2, -0.1, 1.0))
        result = gravity_value_and_gradient_progressive(
            raw, problem=item, fidelity=request, context=context
        )
        numerical = finite_difference(raw, item, request, context)
        np.testing.assert_allclose(result.gradient, numerical, rtol=3e-5, atol=3e-5)
        assert not result.fidelity.exact
        assert result.fidelity.forward_seconds is not None
        assert result.fidelity.reverse_seconds is not None
        assert result.quality.estimator == (
            "stratified_replicate_groups_linearized_gradient"
        )
        assert result.quality.objective_standard_error is not None


def test_sampled_forward_reverse_are_an_adjoint_pair():
    with jax.enable_x64():
        item = problem(likelihood=GravityLikelihood.POISSON)
        context = split_context(item)
        request = GravityFidelityRequest(effort_percent=40, seed=5)
        plan = plan_gravity_fidelity(request, context=context)
        x = jnp.asarray((1.0, 2.0, 3.0, 4.0))
        y = jnp.asarray((0.5, -1.0, 2.0))
        forward = jnp.zeros(3)
        reverse = jnp.zeros(4)
        for identifier, weight in zip(
            plan.selected_shard_ids, plan.expansion_weights, strict=True
        ):
            product = context.product_by_id(identifier)
            forward += weight * product.jax_matvec(x)
            reverse += weight * product.jax_rmatvec(y)
        assert float(jnp.vdot(y, forward)) == pytest.approx(
            float(jnp.vdot(reverse, x)), rel=1e-12, abs=1e-12
        )


def test_forward_uses_ht_weights_and_fixed_offset_exactly_once():
    with jax.enable_x64():
        item = problem(likelihood=GravityLikelihood.POISSON)
        context = split_context(item)
        request = GravityFidelityRequest(effort_percent=25, seed=19)
        raw = np.asarray((0.1, 0.2, 0.8))
        result = gravity_value_and_gradient_progressive(
            raw, problem=item, fidelity=request, context=context
        )
        plan = plan_gravity_fidelity(request, context=context)
        routed = np.zeros(item.operator.num_measurements)
        demand = np.asarray(result.evaluation.demand)
        for identifier, weight in zip(
            plan.selected_shard_ids, plan.expansion_weights, strict=True
        ):
            routed += weight * np.asarray(
                context.product_by_id(identifier).jax_matvec(demand)
            )
        expected = item.rho * (
            routed + np.asarray(item.operator.fixed_measurement_offset)
        )
        expected = np.maximum(expected, item.mean_floor)
        np.testing.assert_allclose(result.evaluation.measurement_mean, expected)


def test_repeated_sampled_evaluation_is_deterministic():
    with jax.enable_x64():
        item = problem()
        context = split_context(item)
        request = GravityFidelityRequest(effort_percent=30, seed=123)
        raw = np.asarray((-0.2, 0.4, 1.2))
        first = gravity_value_and_gradient_progressive(
            raw, problem=item, fidelity=request, context=context
        )
        second = gravity_value_and_gradient_progressive(
            raw, problem=item, fidelity=request, context=context
        )
        assert first.fidelity.selection_identity == second.fidelity.selection_identity
        np.testing.assert_array_equal(first.evaluation.measurement_mean, second.evaluation.measurement_mean)
        np.testing.assert_array_equal(first.gradient, second.gradient)


def test_ht_expansion_recovers_unequal_shard_total_across_seeds():
    metadata = tuple(
        GravityFidelityShard(
            f"s{index}",
            support_entries=support,
            routing_bytes=10 * support,
        )
        for index, support in enumerate((1, 3, 10, 25, 60))
    )
    context = GravityFidelityContext("unequal", metadata)
    contributions = {
        "s0": 2.0,
        "s1": 7.0,
        "s2": 11.0,
        "s3": 31.0,
        "s4": 101.0,
    }
    estimates = []
    for seed in range(2000):
        plan = plan_gravity_fidelity(
            GravityFidelityRequest(effort_percent=30, seed=seed), context=context
        )
        estimates.append(
            sum(
                contributions[identifier] * weight
                for identifier, weight in zip(
                    plan.selected_shard_ids, plan.expansion_weights, strict=True
                )
            )
        )
    assert np.mean(estimates) == pytest.approx(sum(contributions.values()), rel=0.04)
