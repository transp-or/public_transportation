from __future__ import annotations

import jax
import numpy as np
import pytest

from public_transportation.inference.gravity import (
    GravityComponentSpecification,
    GravityConstraint,
    GravityDeviationSpecification,
    GravityEffectScope,
    GravityFeatures,
    GravityModelSpecification,
    GravityParameterLayout,
    GravityParameterization,
    GravityRegularization,
    GravityRegularizationType,
    generate_gravity_demand,
    gravity_demand_numpy_reference,
    gravity_model_specification_from_mapping,
)


def _features() -> GravityFeatures:
    return GravityFeatures(
        canonical_od_index=np.arange(8),
        origin_index=np.asarray((0, 0, 0, 0, 1, 1, 1, 1)),
        destination_index=np.asarray((0, 1, 0, 1, 0, 1, 0, 1)),
        departure_time_index=np.asarray((0, 0, 1, 1, 0, 0, 1, 1)),
        origin_time_group_index=np.asarray((0, 0, 1, 1, 2, 2, 3, 3)),
        journey_time=np.asarray((5, 10, 8, 12, 7, 11, 6, 9), dtype=np.float64),
        transfer_count=np.asarray((0, 1, 1, 0, 0, 1, 1, 0)),
        structural_feasible=np.ones(8, dtype=bool),
        origin_time_totals=np.asarray((20, 30, 40, 50), dtype=np.float64),
        destination_attractiveness=np.ones(8, dtype=np.float64),
        num_origins=2,
        num_destinations=2,
        num_departure_times=2,
        od_layout_fingerprint="production-time-test",
        time_period_index=np.asarray((0, 0, 0, 0, 1, 1, 1, 1)),
    )


def _spec(strength: float = 4.0) -> GravityModelSpecification:
    return GravityModelSpecification(
        components=(
            GravityComponentSpecification(
                name="production",
                scope=GravityEffectScope.GLOBAL,
                parameterization=GravityParameterization.LOG_MULTIPLIER,
                source="origin_time_totals",
                deviation=GravityDeviationSpecification(
                    scope=GravityEffectScope.TIME_PERIOD,
                    grouping="time_period_index",
                    group_count=2,
                    constraint=GravityConstraint.SUM_ZERO,
                    regularization=GravityRegularization(
                        GravityRegularizationType.RIDGE, strength
                    ),
                ),
            ),
        )
    )


def test_zero_deviation_reproduces_legacy_global_production_exactly():
    item = _features()
    legacy_layout = GravityParameterLayout(
        GravityModelSpecification(estimate_global_production_correction=True)
    )
    new_layout = GravityParameterLayout(_spec())
    assert new_layout.names == (
        "beta_time",
        "beta_transfer",
        "dispersion",
        "production_scale",
        "production_time_deviation[0]",
    )
    legacy_raw = np.asarray((-0.2, 0.1, 0.4, 0.0))
    new_raw = np.asarray((*legacy_raw, 0.0))
    np.testing.assert_array_equal(
        np.asarray(
            generate_gravity_demand(
                legacy_raw, features=item, parameter_layout=legacy_layout
            ).demand
        ),
        np.asarray(
            generate_gravity_demand(
                new_raw, features=item, parameter_layout=new_layout
            ).demand
        ),
    )


def test_time_deviations_change_only_the_intended_origin_time_totals():
    item = _features()
    layout = GravityParameterLayout(_spec())
    raw = np.asarray((0.0, 0.0, 0.0, 0.0, 0.3))
    with jax.enable_x64():
        result = generate_gravity_demand(raw, features=item, parameter_layout=layout)
        expected = item.origin_time_totals * np.exp((0.3, 0.3, -0.3, -0.3))
        np.testing.assert_allclose(np.asarray(result.origin_time_sums), expected)
        np.testing.assert_allclose(
            np.asarray(result.demand),
            gravity_demand_numpy_reference(
                raw, features=item, parameter_layout=layout
            ),
        )


def test_deviation_centering_and_ridge_gradient_are_on_physical_deviations():
    layout = GravityParameterLayout(_spec(strength=4.0))
    raw = np.asarray((0.0, 0.0, 0.0, 0.0, 0.3))
    deviations = np.asarray(
        layout.constrained_deviations(raw, "production_time_deviation")
    )
    np.testing.assert_allclose(deviations, (0.3, -0.3))
    assert np.sum(deviations) == 0.0
    assert float(layout.regularization(raw)) == pytest.approx(0.36)
    with jax.enable_x64():
        gradient = jax.grad(lambda value: layout.regularization(value))(raw)
        finite_difference = np.zeros_like(raw)
        step = 1.0e-5
        for index in range(raw.size):
            plus = raw.copy()
            minus = raw.copy()
            plus[index] += step
            minus[index] -= step
            finite_difference[index] = (
                float(layout.regularization(plus))
                - float(layout.regularization(minus))
            ) / (2 * step)
        np.testing.assert_allclose(
            np.asarray(gradient), finite_difference, rtol=1e-5, atol=1e-5
        )
        assert np.asarray(gradient)[-1] != 0


def test_deviation_specification_round_trip_and_fingerprint_change():
    specification = _spec()
    restored = GravityModelSpecification.from_dict(specification.to_dict())
    assert restored == specification
    assert restored.fingerprint == specification.fingerprint
    assert restored.fingerprint != GravityModelSpecification(
        estimate_global_production_correction=True
    ).fingerprint


def test_configuration_mapping_resolves_time_deviation_groups():
    specification = gravity_model_specification_from_mapping(
        {
            "schema_version": 1,
            "production": {
                "correction": {
                    "scope": "global",
                    "deviation": {
                        "scope": "time_period",
                        "grouping": "time_period_index",
                        "group_count": 2,
                        "constraint": "sum_zero",
                        "regularization": {"type": "ridge", "strength": 2.0},
                    },
                }
            },
        },
        features=_features(),
    )
    assert specification.component("production").deviation is not None
    assert GravityParameterLayout(specification).names[-1] == (
        "production_time_deviation[0]"
    )
