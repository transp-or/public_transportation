from __future__ import annotations

import jax
import numpy as np
import pytest
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

from public_transportation.inference.gravity import (
    GravityEffectScope,
    GravityFeatures,
    GravityModelSpecification,
    GravityParameterLayout,
    add_gravity_relaxation,
    generate_gravity_demand,
    remove_gravity_relaxation,
    validate_gravity_relaxation_features,
    warm_start_gravity_parameters,
)


def features() -> GravityFeatures:
    return GravityFeatures(
        canonical_od_index=np.arange(8),
        origin_index=np.asarray((0, 0, 0, 0, 1, 1, 1, 1)),
        destination_index=np.asarray((0, 1, 0, 1, 0, 1, 0, 1)),
        departure_time_index=np.asarray((0, 0, 1, 1, 0, 0, 1, 1)),
        origin_time_group_index=np.asarray((0, 0, 1, 1, 2, 2, 3, 3)),
        journey_time=np.asarray((5, 15, 7, 12, 20, 8, 18, 6), dtype=np.float64),
        transfer_count=np.asarray((0, 1, 0, 1, 1, 0, 1, 0)),
        structural_feasible=np.ones(8, dtype=bool),
        origin_time_totals=np.asarray((20, 30, 40, 50), dtype=np.float64),
        destination_attractiveness=np.ones(8, dtype=np.float64),
        num_origins=2,
        num_destinations=2,
        num_departure_times=2,
        od_layout_fingerprint="phase-5",
        journey_time_scale=10,
        origin_zone_index=np.asarray((0, 0, 0, 0, 1, 1, 1, 1)),
        destination_zone_index=np.asarray((0, 1, 0, 1, 0, 1, 0, 1)),
        time_period_index=np.asarray((0, 0, 1, 1, 0, 0, 1, 1)),
    )


@pytest.mark.parametrize(
    "scope",
    (
        GravityEffectScope.DESTINATION_ZONE,
        GravityEffectScope.TIME_PERIOD,
        GravityEffectScope.ORIGIN_ZONE,
    ),
)
def test_atomic_child_zero_warm_start_exactly_reproduces_parent(scope):
    item = features()
    parent_specification = GravityModelSpecification()
    child_specification, info = add_gravity_relaxation(
        parent_specification, features=item, scope=scope, ridge=2.5
    )
    parent = GravityParameterLayout(parent_specification)
    child = GravityParameterLayout(child_specification)
    raw = np.asarray((-0.4, 0.2, 1.1))
    warm = warm_start_gravity_parameters(parent, child, raw)
    assert info.added_parameter_count == 1
    assert child.size == parent.size + info.added_parameter_count
    assert info.description and info.execution_impact
    np.testing.assert_array_equal(
        generate_gravity_demand(raw, features=item, parameter_layout=parent).demand,
        generate_gravity_demand(warm, features=item, parameter_layout=child).demand,
    )
    assert float(child.regularization(warm)) == 0.0
    assert remove_gravity_relaxation(child_specification, scope) == parent_specification


@pytest.mark.parametrize(
    ("scope", "block"),
    (
        (GravityEffectScope.DESTINATION_ZONE, "destination_zone"),
        (GravityEffectScope.TIME_PERIOD, "time_period"),
        (GravityEffectScope.ORIGIN_ZONE, "origin_zone"),
    ),
)
def test_centering_is_exact_and_ridge_is_zero_only_at_parent(scope, block):
    item = features()
    specification, _ = add_gravity_relaxation(
        GravityModelSpecification(), features=item, scope=scope, ridge=3.0
    )
    layout = GravityParameterLayout(specification)
    raw = np.asarray((-0.4, 0.2, 1.1, 0.35))
    effect = np.asarray(layout.centered_effect(raw, block))
    assert effect.sum() == pytest.approx(0.0, abs=1e-15)
    assert float(layout.regularization(raw)) == pytest.approx(
        0.5 * 3.0 * np.dot(effect, effect)
    )


def test_destination_zone_effect_is_recovered_from_synthetic_demand():
    item = features()
    specification, _ = add_gravity_relaxation(
        GravityModelSpecification(),
        features=item,
        scope=GravityEffectScope.DESTINATION_ZONE,
        ridge=0.0,
    )
    layout = GravityParameterLayout(specification)
    truth = np.asarray((-0.4, 0.2, 1.1, 0.45))
    target = np.asarray(
        generate_gravity_demand(truth, features=item, parameter_layout=layout).demand
    )

    def loss(effect: float) -> float:
        candidate = truth.copy()
        candidate[-1] = effect
        demand = np.asarray(
            generate_gravity_demand(candidate, features=item, parameter_layout=layout).demand
        )
        return float(np.sum((demand - target) ** 2))

    recovered = minimize_scalar(loss, bounds=(-1.0, 1.0), method="bounded")
    assert recovered.x == pytest.approx(truth[-1], abs=2e-5)


def test_applicability_rejects_missing_noncontiguous_and_inconsistent_maps():
    item = features()
    specification, _ = add_gravity_relaxation(
        GravityModelSpecification(), features=item, scope=GravityEffectScope.TIME_PERIOD
    )
    bad = GravityFeatures.from_dict(item.to_dict())
    object.__setattr__(bad, "time_period_index", np.asarray((0, 1, 1, 1, 0, 0, 1, 1)))
    with pytest.raises(ValueError, match="constant within"):
        validate_gravity_relaxation_features(bad, specification)
    with pytest.raises(ValueError, match="required"):
        add_gravity_relaxation(
            GravityModelSpecification(),
            features=GravityFeatures.from_dict({**item.to_dict(), "destination_zone_index": None}),
            scope=GravityEffectScope.DESTINATION_ZONE,
        )


def test_relaxed_demand_is_jittable_and_parameter_values_are_dynamic():
    item = features()
    specification, _ = add_gravity_relaxation(
        GravityModelSpecification(), features=item, scope=GravityEffectScope.ORIGIN_ZONE
    )
    layout = GravityParameterLayout(specification)
    function = jax.jit(lambda raw: generate_gravity_demand(raw, features=item, parameter_layout=layout).demand)
    first = function(np.zeros(layout.size))
    second = function(np.asarray((0.0, 0.0, 0.0, 0.3)))
    assert not np.array_equal(np.asarray(first), np.asarray(second))
