"""Bounded single-destination views for matrix-free fixed-routing studies."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from public_transportation.inference.assignment_adapter import (
    AssignmentInputs,
    build_base_link_cost,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout


@dataclass(frozen=True, slots=True)
class SingleGroupAssignment:
    """One destination group containing free and positive-frozen coordinates."""

    inputs: AssignmentInputs
    original_group_index: int
    full_od_indices: np.ndarray
    free_parameter_indices: np.ndarray
    free_local_indices: np.ndarray
    fixed_positive_local_indices: np.ndarray
    fixed_positive_values: jnp.ndarray
    baseline_demand: jnp.ndarray


def assemble_single_group_demand(
    *, group: SingleGroupAssignment, free_demand: jnp.ndarray
) -> jnp.ndarray:
    """Assemble local demand while preserving positive-frozen offsets."""
    free = jnp.asarray(free_demand, dtype=group.baseline_demand.dtype)
    if free.shape != group.baseline_demand.shape:
        raise ValueError(
            f"free_demand must have shape {group.baseline_demand.shape}, "
            f"got {free.shape}."
        )
    demand = jnp.zeros((group.full_od_indices.size,), dtype=free.dtype)
    demand = demand.at[group.free_local_indices].set(free)
    return demand.at[group.fixed_positive_local_indices].set(
        group.fixed_positive_values
    )


def build_single_free_group_assignment(
    *,
    artifacts: object,
    layout: ODParameterLayout,
    group_index: int,
) -> SingleGroupAssignment:
    """Create a one-group assignment view without copying all destination masks.

    Frozen-zero cells are removed. Positive-frozen cells remain in the local
    group with their fixed values so their measurement contribution is exact.
    """
    od_groups = artifacts.od_groups
    num_groups = int(od_groups.group_dest_node.shape[0])
    if group_index < 0 or group_index >= num_groups:
        raise IndexError(
            f"group_index must be in [0, {num_groups}), got {group_index}."
        )
    if int(od_groups.num_od) != layout.num_od_total:
        raise ValueError("OD layout and assignment group dimensions do not match.")

    padded = np.asarray(od_groups.group_od_index_padded[group_index])
    padded_mask = np.asarray(od_groups.group_od_mask[group_index], dtype=bool)
    group_full = padded[padded_mask].astype(np.int64, copy=False)
    free_position = np.full(layout.num_od_total, -1, dtype=np.int64)
    free_position[np.asarray(layout.free_od_indices, dtype=np.int64)] = np.arange(
        layout.num_free, dtype=np.int64
    )
    positions = free_position[group_full]
    fixed_values = np.zeros(layout.num_od_total, dtype=np.float32)
    fixed_values[np.asarray(layout.fixed_od_indices, dtype=np.int64)] = np.asarray(
        layout.fixed_od_values, dtype=np.float32
    )
    positive = fixed_values[group_full] > 0.0
    selected = (positions >= 0) | positive
    full_indices = group_full[selected]
    selected_positions = positions[selected]
    free_local = np.flatnonzero(selected_positions >= 0).astype(np.int64)
    fixed_local = np.flatnonzero(selected_positions < 0).astype(np.int64)
    parameter_indices = selected_positions[free_local]
    if parameter_indices.size == 0:
        raise ValueError(f"Destination group {group_index} has no free OD cells.")

    origins = np.asarray(od_groups.od_origin_node)[full_indices]
    baseline = np.asarray(layout.free_baseline_values, dtype=np.float32)[
        parameter_indices
    ]
    num_selected = int(full_indices.size)
    inputs = AssignmentInputs(
        graph=artifacts.graph,
        base_link_cost=build_base_link_cost(artifacts=artifacts, dtype=jnp.float32),
        group_dest_node=jnp.asarray(
            np.asarray(od_groups.group_dest_node)[group_index : group_index + 1],
            dtype=jnp.int32,
        ),
        group_link_mask=jnp.asarray(
            np.asarray(od_groups.group_link_mask[group_index : group_index + 1]),
            dtype=bool,
        ),
        od_origin_node=jnp.asarray(origins, dtype=jnp.int32),
        group_od_index_padded=jnp.arange(
            num_selected, dtype=jnp.int32
        ).reshape((1, num_selected)),
        group_od_mask=jnp.ones((1, num_selected), dtype=bool),
    )
    return SingleGroupAssignment(
        inputs=inputs,
        original_group_index=group_index,
        full_od_indices=np.array(full_indices, dtype=np.int64, copy=True),
        free_parameter_indices=np.array(
            parameter_indices, dtype=np.int64, copy=True
        ),
        free_local_indices=np.array(free_local, dtype=np.int64, copy=True),
        fixed_positive_local_indices=np.array(fixed_local, dtype=np.int64, copy=True),
        fixed_positive_values=jnp.asarray(
            fixed_values[full_indices[fixed_local]], dtype=jnp.float32
        ),
        baseline_demand=jnp.asarray(baseline, dtype=jnp.float32),
    )
