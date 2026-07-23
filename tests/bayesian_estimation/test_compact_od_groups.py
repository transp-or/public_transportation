from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace

from public_transportation.assignment.build_od_groups import ODGroups
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.assignment_adapter import AssignmentInputs, assign_link_flow
from public_transportation.inference.compact_od_groups import compact_od_groups
from public_transportation.inference.od_parameter_layout import ODParameterLayout


def _groups() -> ODGroups:
    # Full cells 0..5 have destinations [10, 20, 10, 30, 20, 30].
    return ODGroups(
        num_od=6,
        od_origin_node=jnp.asarray([100, 101, 102, 103, 104, 105]),
        od_dest_node=jnp.asarray([10, 20, 10, 30, 20, 30]),
        group_start=jnp.asarray([0, 2, 4, 6]),
        group_dest_node=jnp.asarray([10, 20, 30]),
        group_od_index=jnp.asarray([0, 2, 1, 4, 3, 5]),
        group_od_index_padded=jnp.asarray([[0, 2], [1, 4], [3, 5]]),
        group_od_mask=jnp.ones((3, 2), dtype=bool),
        group_link_mask=jnp.asarray(
            [[True, False, False], [False, True, False], [False, False, True]]
        ),
    )


def _layout(*, free, fixed) -> ODParameterLayout:
    fixed_indices = tuple(index for index, _ in fixed)
    fixed_values = tuple(value for _, value in fixed)
    return ODParameterLayout(
        num_od_total=6,
        od_keys=tuple((f"O{i}", f"D{i}", "t0") for i in range(6)),
        free_od_indices=tuple(free),
        fixed_od_indices=fixed_indices,
        fixed_od_values=fixed_values,
        free_baseline_values=tuple(float(i + 1) for i in range(len(free))),
        fixed_zero_indices=tuple(index for index, value in fixed if value == 0.0),
        fixed_positive_indices=tuple(index for index, value in fixed if value > 0.0),
    )


def _compact(*, free, fixed) -> ODGroups:
    layout = build_compact_od_assignment_layout(
        parameter_layout=_layout(free=free, fixed=fixed)
    )
    return compact_od_groups(od_groups=_groups(), layout=layout)


def test_compacts_indices_and_preserves_origin_destination_alignment():
    groups = _compact(free=(0, 2, 5), fixed=((1, 0.0), (3, 12.0), (4, 0.0)))

    # Active full indices [0, 2, 3, 5] become compact indices [0, 1, 2, 3].
    assert groups.num_od == 4
    assert np.array_equal(groups.od_origin_node, [100, 102, 103, 105])
    assert np.array_equal(groups.od_dest_node, [10, 10, 30, 30])
    assert np.array_equal(groups.group_dest_node, [10, 30])
    assert np.array_equal(groups.group_start, [0, 2, 4])
    assert np.array_equal(groups.group_od_index, [0, 1, 2, 3])
    assert np.array_equal(groups.group_od_index_padded, [[0, 1], [2, 3]])
    assert np.array_equal(groups.group_od_mask, [[True, True], [True, True]])
    assert np.array_equal(
        groups.group_link_mask,
        [[True, False, False], [False, False, True]],
    )


def test_destination_with_only_frozen_zero_cells_is_removed():
    groups = _compact(
        free=(0, 2, 3, 5),
        fixed=((1, 0.0), (4, 0.0)),
    )
    assert np.array_equal(groups.group_dest_node, [10, 30])
    assert 20 not in np.asarray(groups.od_dest_node)


def test_every_active_compact_index_appears_exactly_once():
    groups = _compact(free=(0, 3, 5), fixed=((1, 7.0), (2, 0.0), (4, 8.0)))
    selected = np.asarray(groups.group_od_index_padded)[np.asarray(groups.group_od_mask)]
    assert np.array_equal(np.sort(selected), np.arange(groups.num_od))


def test_no_removal_reproduces_original_group_contract():
    groups = _compact(free=(0, 1, 2, 3, 4, 5), fixed=())
    original = _groups()
    for field in (
        "od_origin_node",
        "od_dest_node",
        "group_start",
        "group_dest_node",
        "group_od_index",
        "group_od_index_padded",
        "group_od_mask",
        "group_link_mask",
    ):
        assert np.array_equal(getattr(groups, field), getattr(original, field)), field


def test_all_frozen_zero_produces_valid_empty_groups():
    groups = _compact(
        free=(),
        fixed=tuple((index, 0.0) for index in range(6)),
    )
    assert groups.num_od == 0
    assert groups.od_origin_node.shape == (0,)
    assert groups.od_dest_node.shape == (0,)
    assert np.array_equal(groups.group_start, [0])
    assert groups.group_dest_node.shape == (0,)
    assert groups.group_od_index.shape == (0,)
    assert groups.group_od_index_padded.shape == (0, 0)
    assert groups.group_od_mask.shape == (0, 0)
    assert groups.group_link_mask.shape == (0, 3)


def test_padding_is_rebuilt_for_unequal_surviving_group_sizes():
    groups = _compact(free=(0, 1, 2), fixed=((3, 0.0), (4, 0.0), (5, 9.0)))
    assert np.array_equal(groups.group_dest_node, [10, 20, 30])
    assert groups.group_od_index_padded.shape == (3, 2)
    assert np.array_equal(groups.group_od_mask, [[True, True], [True, False], [True, False]])
    selected = np.asarray(groups.group_od_index_padded)[np.asarray(groups.group_od_mask)]
    assert np.array_equal(np.sort(selected), np.arange(4))


def test_size_mismatch_is_rejected():
    layout = build_compact_od_assignment_layout(
        parameter_layout=_layout(
            free=(0, 1, 2, 3, 4, 5),
            fixed=(),
        )
    )
    bad_groups = _groups()
    object.__setattr__(bad_groups, "num_od", 5)
    with pytest.raises(ValueError, match="size mismatch"):
        compact_od_groups(od_groups=bad_groups, layout=layout)


def test_empty_compact_assignment_returns_zero_without_calling_core(monkeypatch):
    def fail_if_called(**_):
        raise AssertionError("assignment core must not run for empty compact groups")

    monkeypatch.setattr(
        "public_transportation.inference.assignment_adapter._assign_core",
        fail_if_called,
    )
    inputs = AssignmentInputs(
        graph=SimpleNamespace(num_links=3),
        base_link_cost=jnp.ones((3,)),
        group_dest_node=jnp.empty((0,), dtype=jnp.int32),
        group_link_mask=jnp.empty((0, 3), dtype=bool),
        od_origin_node=jnp.empty((0,), dtype=jnp.int32),
        group_od_index_padded=jnp.empty((0, 0), dtype=jnp.int32),
        group_od_mask=jnp.empty((0, 0), dtype=bool),
    )
    actual = assign_link_flow(inputs=inputs, f=jnp.empty((0,)), theta=jnp.asarray(2.0))
    assert np.array_equal(actual, np.zeros(3))


def test_assignment_adapter_rejects_noncompact_demand_shape():
    inputs = AssignmentInputs(
        graph=SimpleNamespace(num_links=1),
        base_link_cost=jnp.ones((1,)),
        group_dest_node=jnp.asarray([2]),
        group_link_mask=jnp.ones((1, 1), dtype=bool),
        od_origin_node=jnp.asarray([0, 1]),
        group_od_index_padded=jnp.asarray([[0, 1]]),
        group_od_mask=jnp.ones((1, 2), dtype=bool),
    )
    with pytest.raises(ValueError, match=r"f must have shape \(2,\)"):
        assign_link_flow(inputs=inputs, f=jnp.ones((3,)), theta=jnp.asarray(2.0))
