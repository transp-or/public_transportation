from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import pytest

from public_transportation.inference.gravity import (
    GravityEffectScope,
    GravityFeatures,
    GravityModelSpecification,
    GravityParameterLayout,
    generate_gravity_demand,
    gravity_demand_kernel,
    gravity_demand_numpy_reference,
)
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)


def features(
    *,
    dtype=np.float64,
    feasible=(True, True, False, True, True),
    journey_time=(10.0, 20.0, 30.0, 5.0, 15.0),
    transfers=(0, 1, 2, 1000, 0),
) -> GravityFeatures:
    return GravityFeatures(
        canonical_od_index=np.arange(5),
        origin_index=np.asarray((0, 0, 0, 1, 1)),
        destination_index=np.asarray((0, 1, 2, 0, 1)),
        departure_time_index=np.asarray((0, 0, 0, 1, 1)),
        origin_time_group_index=np.asarray((0, 0, 0, 1, 1)),
        journey_time=np.asarray(journey_time, dtype=dtype),
        transfer_count=np.asarray(transfers),
        structural_feasible=np.asarray(feasible),
        origin_time_totals=np.asarray((30.0, 70.0), dtype=dtype),
        destination_attractiveness=np.asarray((1.0, 2.0, 1.0, 0.5, 3.0), dtype=dtype),
        num_origins=2,
        num_destinations=3,
        num_departure_times=2,
        od_layout_fingerprint="layout-1",
        journey_time_scale=10.0,
    )


def test_one_origin_one_destination_conserves_exactly():
    item = GravityFeatures(
        canonical_od_index=np.asarray((4,)),
        origin_index=np.asarray((0,)),
        destination_index=np.asarray((0,)),
        departure_time_index=np.asarray((0,)),
        origin_time_group_index=np.asarray((0,)),
        journey_time=np.asarray((123.0,), dtype=np.float32),
        transfer_count=np.asarray((50,)),
        structural_feasible=np.asarray((True,)),
        origin_time_totals=np.asarray((17.0,), dtype=np.float32),
        destination_attractiveness=np.asarray((1.0,), dtype=np.float32),
        num_origins=1,
        num_destinations=1,
        num_departure_times=1,
        od_layout_fingerprint="one-cell",
    )
    layout = GravityParameterLayout(GravityModelSpecification())
    result = generate_gravity_demand(
        np.zeros(3, np.float32), features=item, parameter_layout=layout
    )
    np.testing.assert_array_equal(
        np.asarray(result.demand), np.asarray((17.0,), np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(result.probabilities), np.ones(1, np.float32)
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_jax_matches_numpy_preserves_zeros_and_group_totals(dtype):
    with jax.enable_x64():
        item = features(dtype=dtype)
        layout = GravityParameterLayout(GravityModelSpecification())
        raw = np.asarray((-2.0, 0.5, 1.0), dtype=dtype)
        result = generate_gravity_demand(raw, features=item, parameter_layout=layout)
        reference = gravity_demand_numpy_reference(
            raw, features=item, parameter_layout=layout
        )
        tolerance = 2e-6 if dtype is np.float32 else 1e-12
        np.testing.assert_allclose(np.asarray(result.demand), reference, rtol=tolerance)
        np.testing.assert_allclose(
            np.asarray(result.origin_time_sums), item.origin_time_totals, rtol=tolerance
        )
        assert np.asarray(result.demand)[2] == 0.0
        assert np.asarray(result.probabilities)[2] == 0.0
        assert np.asarray(result.demand).dtype == dtype


def test_extreme_utilities_and_large_transfers_remain_finite():
    item = features(journey_time=(0.0, 1.0e12, 2.0e12, 1.0, 1.0e12))
    layout = GravityParameterLayout(GravityModelSpecification())
    result = generate_gravity_demand(
        np.asarray((1000.0, 1000.0, 0.0)), features=item, parameter_layout=layout
    )
    assert np.all(np.isfinite(np.asarray(result.demand)))
    assert np.all(np.isfinite(np.asarray(result.probabilities)))
    np.testing.assert_allclose(np.asarray(result.origin_time_sums), (30.0, 70.0))


def test_kernel_is_jittable_and_parameter_values_are_dynamic():
    item = features(dtype=np.float32)
    kernel = jax.jit(gravity_demand_kernel)
    arguments = {
        "journey_time": item.journey_time,
        "transfer_count": item.transfer_count,
        "structural_feasible": item.structural_feasible,
        "origin_time_group_index": item.origin_time_group_index,
        "origin_time_totals": item.origin_time_totals,
        "destination_attractiveness": item.destination_attractiveness,
        "journey_time_scale": item.journey_time_scale,
    }
    first = kernel(np.zeros(3, np.float32), **arguments)
    second = kernel(np.ones(3, np.float32), **arguments)
    assert first.demand.shape == (item.num_cells,)
    assert not np.array_equal(np.asarray(first.demand), np.asarray(second.demand))


def test_rejects_origin_time_group_with_no_feasible_destination():
    with pytest.raises(ValueError, match="at least one feasible"):
        features(feasible=(True, True, False, False, False))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"journey_time": np.asarray((1.0, np.nan, 2.0, 3.0, 4.0))}, "finite"),
        ({"transfer_count": np.asarray((0, 0, -1, 0, 0))}, "non-negative"),
        ({"destination_index": np.asarray((0, 1, 3, 0, 1))}, "out-of-bounds"),
        ({"destination_attractiveness": np.ones(4)}, "must contain 5"),
    ],
)
def test_feature_validation(change, message):
    values = {
        field: getattr(features(), field)
        for field in GravityFeatures.__dataclass_fields__
        if field != "journey_time_scale"
    }
    values.update(change)
    with pytest.raises((TypeError, ValueError), match=message):
        GravityFeatures(**values)


def test_arrays_are_immutable_and_fingerprint_is_deterministic():
    first = features()
    second = features()
    assert first.fingerprint == second.fingerprint
    assert not first.journey_time.flags.writeable
    with pytest.raises(ValueError):
        first.journey_time[0] = 99.0
    changed = features(journey_time=(11.0, 20.0, 30.0, 5.0, 15.0))
    assert changed.fingerprint != first.fingerprint
    restored = GravityFeatures.from_dict(first.to_dict())
    assert restored.fingerprint == first.fingerprint
    assert restored.dtype == first.dtype


def test_optional_waiting_and_group_indices_are_validated_and_serialized():
    item = replace(
        features(),
        initial_waiting_time=np.arange(5, dtype=np.float32),
        origin_zone_index=np.asarray((0, 0, 0, 1, 1)),
        destination_zone_index=np.asarray((0, 1, 2, 0, 1)),
        time_period_index=np.asarray((0, 0, 0, 1, 1)),
    )
    restored = GravityFeatures.from_dict(item.to_dict())
    assert restored.fingerprint == item.fingerprint
    assert restored.initial_waiting_time is not None
    assert not restored.initial_waiting_time.flags.writeable
    with pytest.raises(ValueError, match="initial_waiting_time"):
        replace(item, initial_waiting_time=np.asarray((0.0, -1.0, 0.0, 0.0, 0.0)))


def test_compact_layout_identity_and_order_are_validated():
    template = CompactODAssignmentLayout(
        num_od_total=5,
        active_full_indices=(0, 1, 2, 3, 4),
        removed_zero_full_indices=(),
        full_to_compact=(0, 1, 2, 3, 4),
        free_full_indices=(0, 1, 2, 3, 4),
        free_compact_indices=(0, 1, 2, 3, 4),
        free_baseline_values=(1.0, 1.0, 1.0, 1.0, 1.0),
        fixed_compact_indices=(),
        fixed_compact_values=(),
    )
    item = replace(features(), od_layout_fingerprint=template.fingerprint)
    item.validate_compact_layout(template)
    with pytest.raises(ValueError, match="fingerprints differ"):
        features().validate_compact_layout(template)
    reordered = replace(item, canonical_od_index=np.asarray((1, 0, 2, 3, 4)))
    with pytest.raises(ValueError, match="free-cell order"):
        reordered.validate_compact_layout(template)


def test_minimal_specification_serialization_and_future_scope_rejection():
    specification = GravityModelSpecification()
    assert specification.parameter_count == 3
    assert specification.parameter_names == (
        "beta_time",
        "beta_transfer",
        "dispersion",
    )
    assert specification.canonical_json == GravityModelSpecification().canonical_json
    assert specification.fingerprint == GravityModelSpecification().fingerprint
    restored = GravityModelSpecification.from_dict(specification.to_dict())
    assert restored == specification
    with pytest.raises(ValueError, match="destination_zone_count"):
        replace(
            specification,
            destination_attractiveness_scope=GravityEffectScope.DESTINATION_ZONE,
        )


def test_parameter_layout_sign_transform_and_round_trip():
    layout = GravityParameterLayout(GravityModelSpecification())
    raw = layout.raw_from_physical((0.25, 1.5, 20.0))
    transformed = layout.transform(raw)
    np.testing.assert_allclose(np.asarray(transformed), (0.25, 1.5, 20.0))
    assert np.all(np.asarray(layout.transform((-1.0e6, -100.0, -20.0))) > 0)
    assert layout.names == ("beta_time", "beta_transfer", "dispersion")
    assert layout.slices["beta_transfer"] == slice(1, 2)
    assert (
        layout.fingerprint
        == GravityParameterLayout(GravityModelSpecification()).fingerprint
    )
