from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout
from public_transportation.inference.runtime_profile import (
    build_od_assignment_runtime_profile,
)


def _layout() -> ODParameterLayout:
    return ODParameterLayout(
        num_od_total=6,
        od_keys=tuple((str(i), str(i), "t") for i in range(6)),
        free_od_indices=(0, 2, 5),
        fixed_od_indices=(1, 3, 4),
        fixed_od_values=(0.0, 12.0, 0.0),
        free_baseline_values=(10.0, 20.0, 30.0),
        fixed_zero_indices=(1, 4),
        fixed_positive_indices=(3,),
    )


def test_runtime_profile_reports_compactness_and_destination_removal():
    layout = _layout()
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    profile = build_od_assignment_runtime_profile(
        num_od_total=6,
        parameter_layout=layout,
        compact_layout=compact,
        artifacts=SimpleNamespace(
            od_groups=SimpleNamespace(group_dest_node=jnp.arange(3))
        ),
        assignment_inputs=SimpleNamespace(
            od_origin_node=jnp.arange(4),
            group_dest_node=jnp.arange(2),
        ),
    )

    assert profile.num_od_total == 6
    assert profile.num_free_od == 3
    assert profile.num_fixed_od == 3
    assert profile.num_fixed_zero_od == 2
    assert profile.num_fixed_positive_od == 1
    assert profile.assignment_active_od == 4
    assert profile.original_destination_groups == 3
    assert profile.active_destination_groups == 2
    assert profile.removed_destination_groups == 1
    assert profile.od_layout_fingerprint == layout.fingerprint
    assert profile.compact_layout_fingerprint == compact.fingerprint
    assert profile.as_dict()["assignment_active_od"] == 4
    assert profile.format_lines() == (
        "OD cells: 6 total, 3 free, 2 frozen zero, 1 frozen positive",
        "Assignment OD vector: 4 active cells",
        "Destination groups: 2/3 active (1 removed)",
    )


def test_all_free_runtime_profile_uses_full_assignment_counts():
    profile = build_od_assignment_runtime_profile(
        num_od_total=4,
        parameter_layout=None,
        compact_layout=None,
        artifacts=SimpleNamespace(
            od_groups=SimpleNamespace(group_dest_node=jnp.arange(2))
        ),
        assignment_inputs=SimpleNamespace(
            od_origin_node=jnp.arange(4),
            group_dest_node=jnp.arange(2),
        ),
    )
    assert profile.num_free_od == 4
    assert profile.num_fixed_od == 0
    assert profile.assignment_active_od == 4
    assert profile.removed_destination_groups == 0
    assert profile.od_layout_fingerprint is None
    assert profile.compact_layout_fingerprint is None


def test_runtime_profile_rejects_assignment_size_mismatch():
    layout = _layout()
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    with pytest.raises(ValueError, match="Compact assignment size mismatch"):
        build_od_assignment_runtime_profile(
            num_od_total=6,
            parameter_layout=layout,
            compact_layout=compact,
            artifacts=SimpleNamespace(
                od_groups=SimpleNamespace(group_dest_node=jnp.arange(3))
            ),
            assignment_inputs=SimpleNamespace(
                od_origin_node=jnp.arange(5),
                group_dest_node=jnp.arange(2),
            ),
        )
