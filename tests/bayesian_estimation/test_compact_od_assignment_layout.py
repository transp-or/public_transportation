from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
    build_compact_od_assignment_layout,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout


def _parameter_layout(
    *,
    free=(0, 2, 5),
    fixed=((1, 0.0), (3, 12.0), (4, 0.0)),
    num_od_total=6,
):
    fixed_indices = tuple(index for index, _ in fixed)
    fixed_values = tuple(value for _, value in fixed)
    return ODParameterLayout(
        num_od_total=num_od_total,
        od_keys=tuple((f"O{i}", f"D{i}", "t0") for i in range(num_od_total)),
        free_od_indices=tuple(free),
        fixed_od_indices=fixed_indices,
        fixed_od_values=fixed_values,
        free_baseline_values=tuple(float(10 * (i + 1)) for i in range(len(free))),
        fixed_zero_indices=tuple(index for index, value in fixed if value == 0.0),
        fixed_positive_indices=tuple(index for index, value in fixed if value > 0.0),
    )


def test_mixed_layout_removes_only_structurally_frozen_zero_cells():
    compact = build_compact_od_assignment_layout(parameter_layout=_parameter_layout())

    assert compact.active_full_indices == (0, 2, 3, 5)
    assert compact.removed_zero_full_indices == (1, 4)
    assert compact.full_to_compact == (0, -1, 1, 2, -1, 3)
    assert compact.free_full_indices == (0, 2, 5)
    assert compact.free_compact_indices == (0, 1, 3)
    assert compact.fixed_compact_indices == (2,)
    assert compact.fixed_compact_values == (12.0,)
    assert compact.num_active == 4
    assert compact.num_free == 3
    assert compact.num_fixed_positive == 1
    assert compact.num_removed_zero == 2


def test_assemble_compact_demand_has_no_full_od_coordinates():
    compact = build_compact_od_assignment_layout(parameter_layout=_parameter_layout())
    demand = compact.assemble_compact_jax(jnp.log(jnp.asarray([2.0, 0.5, 3.0])))

    assert demand.shape == (compact.num_free + compact.num_fixed_positive,)
    assert np.allclose(demand, [20.0, 10.0, 12.0, 90.0])
    assert demand.shape != (compact.num_od_total,)

    jaxpr = jax.make_jaxpr(compact.assemble_compact_jax)(jnp.zeros((compact.num_free,)))
    produced_shapes = {
        tuple(variable.aval.shape)
        for equation in jaxpr.jaxpr.eqns
        for variable in equation.outvars
        if hasattr(variable, "aval") and hasattr(variable.aval, "shape")
    }
    assert (compact.num_active,) in produced_shapes
    assert (compact.num_od_total,) not in produced_shapes


def test_full_reconstruction_is_reserved_for_reporting_and_supports_draws():
    compact = build_compact_od_assignment_layout(parameter_layout=_parameter_layout())
    full = compact.reconstruct_full_numpy(np.log([[2.0, 0.5, 3.0], [1.0, 1.0, 1.0]]))

    assert full.shape == (2, 6)
    assert np.allclose(full[0], [20.0, 0.0, 10.0, 12.0, 0.0, 90.0])
    assert np.allclose(full[1], [10.0, 0.0, 20.0, 12.0, 0.0, 30.0])


def test_all_free_layout_preserves_scenario_order():
    compact = build_compact_od_assignment_layout(
        parameter_layout=_parameter_layout(free=(0, 1, 2), fixed=(), num_od_total=3)
    )
    assert compact.active_full_indices == (0, 1, 2)
    assert compact.full_to_compact == (0, 1, 2)
    assert compact.free_compact_indices == (0, 1, 2)
    assert compact.fixed_compact_indices == ()
    assert compact.removed_zero_full_indices == ()


def test_all_frozen_zero_layout_has_empty_assignment_vector():
    compact = build_compact_od_assignment_layout(
        parameter_layout=_parameter_layout(
            free=(), fixed=((0, 0.0), (1, 0.0), (2, 0.0)), num_od_total=3
        )
    )
    assert compact.active_full_indices == ()
    assert compact.full_to_compact == (-1, -1, -1)
    assert compact.assemble_compact_jax(jnp.empty((0,))).shape == (0,)
    assert np.array_equal(compact.reconstruct_full_numpy(np.empty((0,))), np.zeros(3))


def test_all_frozen_positive_layout_contains_constants_but_no_parameters():
    compact = build_compact_od_assignment_layout(
        parameter_layout=_parameter_layout(
            free=(), fixed=((0, 4.0), (1, 5.0), (2, 6.0)), num_od_total=3
        )
    )
    assert compact.num_free == 0
    assert compact.num_active == 3
    assert np.array_equal(compact.assemble_compact_jax(jnp.empty((0,))), [4.0, 5.0, 6.0])


def test_fingerprint_is_stable_and_changes_with_assignment_contract():
    first = build_compact_od_assignment_layout(parameter_layout=_parameter_layout())
    second = build_compact_od_assignment_layout(parameter_layout=_parameter_layout())
    changed = build_compact_od_assignment_layout(
        parameter_layout=_parameter_layout(fixed=((1, 0.0), (3, 13.0), (4, 0.0)))
    )
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != changed.fingerprint


def test_layout_is_immutable():
    compact = build_compact_od_assignment_layout(parameter_layout=_parameter_layout())
    with pytest.raises(FrozenInstanceError):
        compact.num_od_total = 99


@pytest.mark.parametrize(
    "mutation",
    [
        {"active_full_indices": (0, 2, 5, 3)},
        {"removed_zero_full_indices": (1,)},
        {"full_to_compact": (0, -1, 1, 2, -1, 2)},
        {"free_compact_indices": (0, 1, 2)},
        {"fixed_compact_values": (0.0,)},
    ],
)
def test_manual_layout_rejects_inconsistent_contract(mutation):
    compact = build_compact_od_assignment_layout(parameter_layout=_parameter_layout())
    with pytest.raises(ValueError):
        replace(compact, **mutation)


def test_public_type_can_be_constructed_only_with_a_complete_valid_partition():
    assert CompactODAssignmentLayout is not None
