from __future__ import annotations

import numpy as np
import pytest

from public_transportation.inference.gravity import (
    GravityFidelityContext,
    GravityFidelityRequest,
    GravityFidelityShard,
    gravity_fidelity_problem_identity,
    gravity_value_and_gradient_adjoint,
    gravity_value_and_gradient_progressive,
    plan_gravity_fidelity,
)
from tests.gravity.test_phase2_objective import problem


def context(identity: str = "problem") -> GravityFidelityContext:
    return GravityFidelityContext(
        problem_identity=identity,
        shards=(
            GravityFidelityShard("a", 5, 50, "morning"),
            GravityFidelityShard("b", 35, 350, "morning"),
            GravityFidelityShard("c", 10, 100, "afternoon"),
            GravityFidelityShard("d", 50, 500, "afternoon"),
        ),
    )


@pytest.mark.parametrize("effort", (0.0, 0.999, 100.001, np.nan, np.inf, -np.inf))
def test_request_rejects_invalid_effort(effort):
    with pytest.raises(ValueError, match="effort_percent"):
        GravityFidelityRequest(effort_percent=effort)


@pytest.mark.parametrize("seed", (-1, 1.5, True))
def test_request_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        GravityFidelityRequest(seed=seed)


@pytest.mark.parametrize("groups", (0, -1, 1.5, True))
def test_request_rejects_invalid_quality_groups(groups):
    with pytest.raises(ValueError, match="quality_groups"):
        GravityFidelityRequest(quality_groups=groups)


def test_context_rejects_duplicate_shards_and_invalid_work():
    with pytest.raises(ValueError, match="unique"):
        GravityFidelityContext(
            "problem",
            (GravityFidelityShard("a", 1, 1), GravityFidelityShard("a", 2, 2)),
        )
    with pytest.raises(ValueError, match="support_entries"):
        GravityFidelityShard("a", 0, 1)
    with pytest.raises(ValueError, match="routing_bytes"):
        GravityFidelityShard("a", 1, -1)


def test_selection_is_deterministic_and_seeded():
    request = GravityFidelityRequest(effort_percent=25, seed=7)
    first = plan_gravity_fidelity(request, context=context())
    second = plan_gravity_fidelity(request, context=context())
    changed = plan_gravity_fidelity(
        GravityFidelityRequest(effort_percent=25, seed=8), context=context()
    )
    assert first == second
    assert first.selection_identity == second.selection_identity
    assert first.ordered_shard_ids != changed.ordered_shard_ids


def test_fidelity_subsets_are_nested_and_work_is_nondecreasing():
    plans = [
        plan_gravity_fidelity(
            GravityFidelityRequest(effort_percent=effort, seed=42),
            context=context(),
        )
        for effort in (1, 10, 25, 50, 75, 100)
    ]
    for lower, higher in zip(plans, plans[1:]):
        assert set(lower.selected_shard_ids) <= set(higher.selected_shard_ids)
        assert lower.selected_support_entries <= higher.selected_support_entries
        assert lower.effective_effort_percent <= higher.effective_effort_percent
    assert plans[-1].selected_support_entries == plans[-1].total_support_entries
    assert plans[-1].selected_shard_ids == plans[-1].ordered_shard_ids
    assert plans[-1].exact


def test_effort_targets_support_not_shard_count():
    many = GravityFidelityContext(
        "problem",
        tuple(
            GravityFidelityShard(
                f"s{index}", index + 1, 10 * (index + 1), "one-stratum"
            )
            for index in range(100)
        ),
    )
    plans = [
        plan_gravity_fidelity(
            GravityFidelityRequest(effort_percent=25, seed=seed), context=many
        )
        for seed in range(40)
    ]
    mean_effective = np.mean([plan.effective_effort_percent for plan in plans])
    assert mean_effective == pytest.approx(25.0, abs=4.0)
    assert all(
        plan.selected_routing_bytes == 10 * plan.selected_support_entries
        for plan in plans
    )
    expected_probability = 0.25 + 0.75**100 / 100
    assert all(
        probability == pytest.approx(expected_probability)
        for probability in plans[0].inclusion_probabilities
    )
    assert all(
        weight == pytest.approx(1.0 / expected_probability)
        for weight in plans[0].expansion_weights
    )


def test_effort_100_matches_existing_exact_adjoint_contract():
    item = problem()
    raw = np.asarray((0.2, -0.1, 1.0))
    identity = gravity_fidelity_problem_identity(item)
    fidelity_context = GravityFidelityContext(
        identity,
        (
            GravityFidelityShard("s0", 3, 30, "morning"),
            GravityFidelityShard("s1", 5, 50, "afternoon"),
        ),
    )
    expected_evaluation, expected_gradient = gravity_value_and_gradient_adjoint(
        raw, problem=item
    )
    result = gravity_value_and_gradient_progressive(
        raw,
        problem=item,
        fidelity=GravityFidelityRequest(effort_percent=100, seed=9),
        context=fidelity_context,
    )
    np.testing.assert_allclose(result.evaluation.objective, expected_evaluation.objective)
    np.testing.assert_allclose(result.evaluation.measurement_mean, expected_evaluation.measurement_mean)
    np.testing.assert_allclose(result.gradient, expected_gradient)
    assert result.fidelity.exact
    assert result.fidelity.selected_shard_count == 2
    assert result.fidelity.effective_effort_percent == 100.0
    assert result.quality.exact
    assert result.quality.quality_score == 1.0
    assert result.quality.objective_standard_error == 0.0
    assert result.quality.gradient_error_norm_estimate == 0.0
    assert result.quality.measurement_coverage_fraction == 1.0


def test_progressive_rejects_wrong_context_and_requires_low_effort_products():
    item = problem()
    with pytest.raises(ValueError, match="identities differ"):
        gravity_value_and_gradient_progressive(
            np.zeros(3),
            problem=item,
            fidelity=GravityFidelityRequest(),
            context=context("wrong"),
        )
    matching = GravityFidelityContext(
        gravity_fidelity_problem_identity(item),
        (GravityFidelityShard("s0", 1, 1),),
    )
    with pytest.raises(ValueError, match="requires additive shard products"):
        gravity_value_and_gradient_progressive(
            np.zeros(3),
            problem=item,
            fidelity=GravityFidelityRequest(effort_percent=50),
            context=matching,
        )
