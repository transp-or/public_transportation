from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import public_transportation.inference.fixed_routing_group as group_module
from public_transportation.inference.fixed_routing_group import (
    assemble_single_group_demand,
    build_single_free_group_assignment,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout


def _artifacts():
    groups = SimpleNamespace(
        num_od=5,
        od_origin_node=jnp.asarray([10, 11, 12, 13, 14], dtype=jnp.int32),
        group_dest_node=jnp.asarray([20, 21], dtype=jnp.int32),
        group_od_index_padded=jnp.asarray([[0, 2, 4], [1, 3, 0]], dtype=jnp.int32),
        group_od_mask=jnp.asarray(
            [[True, True, True], [True, True, False]], dtype=bool
        ),
        group_link_mask=jnp.asarray(
            [[True, False, True], [False, True, True]], dtype=bool
        ),
    )
    return SimpleNamespace(graph=SimpleNamespace(num_links=3), od_groups=groups)


def _layout(*, positive=False):
    return ODParameterLayout(
        num_od_total=5,
        od_keys=(("a", "x", 0), ("b", "y", 0), ("c", "x", 0), ("d", "y", 0), ("e", "x", 0)),
        free_od_indices=((0, 2, 3) if positive else (0, 2, 3, 4)),
        fixed_od_indices=((1, 4) if positive else (1,)),
        fixed_od_values=((0.0, 7.0) if positive else (0.0,)),
        free_baseline_values=((1.0, 2.0, 3.0) if positive else (1.0, 2.0, 3.0, 4.0)),
        fixed_zero_indices=(1,),
        fixed_positive_indices=((4,) if positive else ()),
    )


def test_single_group_view_contains_only_selected_free_cells(monkeypatch):
    monkeypatch.setattr(
        group_module,
        "build_base_link_cost",
        lambda **_: jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
    )
    selected = build_single_free_group_assignment(
        artifacts=_artifacts(), layout=_layout(), group_index=0
    )

    assert selected.inputs.group_dest_node.shape == (1,)
    assert selected.inputs.group_link_mask.shape == (1, 3)
    assert selected.inputs.group_od_index_padded.shape == (1, 3)
    np.testing.assert_array_equal(selected.full_od_indices, [0, 2, 4])
    np.testing.assert_array_equal(selected.free_parameter_indices, [0, 1, 3])
    np.testing.assert_array_equal(selected.inputs.od_origin_node, [10, 12, 14])
    np.testing.assert_allclose(selected.baseline_demand, [1.0, 2.0, 4.0])


def test_single_group_view_preserves_positive_frozen_offset(monkeypatch):
    monkeypatch.setattr(
        group_module,
        "build_base_link_cost",
        lambda **_: jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
    )
    selected = build_single_free_group_assignment(
        artifacts=_artifacts(), layout=_layout(positive=True), group_index=0
    )

    np.testing.assert_array_equal(selected.full_od_indices, [0, 2, 4])
    np.testing.assert_array_equal(selected.free_parameter_indices, [0, 1])
    np.testing.assert_array_equal(selected.free_local_indices, [0, 1])
    np.testing.assert_array_equal(selected.fixed_positive_local_indices, [2])
    np.testing.assert_allclose(
        assemble_single_group_demand(
            group=selected, free_demand=jnp.asarray([5.0, 6.0])
        ),
        [5.0, 6.0, 7.0],
    )


def test_single_group_view_rejects_invalid_group(monkeypatch):
    monkeypatch.setattr(
        group_module,
        "build_base_link_cost",
        lambda **_: jnp.zeros((3,), dtype=jnp.float32),
    )
    with pytest.raises(IndexError, match="group_index"):
        build_single_free_group_assignment(
            artifacts=_artifacts(), layout=_layout(), group_index=2
        )
