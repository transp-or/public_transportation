"""Pure compaction of assignment OD groups for inference."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from public_transportation.assignment.build_od_groups import ODGroups
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)


def compact_od_groups(
    *,
    od_groups: ODGroups,
    layout: CompactODAssignmentLayout,
) -> ODGroups:
    """Return OD groups indexed by the compact assignment demand vector.

    The transformation is structural and runs once outside JAX tracing. It
    preserves the original destination ordering and corresponding link masks,
    while dropping destination groups with no active OD cells.
    """
    if int(od_groups.num_od) != layout.num_od_total:
        raise ValueError(
            "OD group/layout size mismatch: "
            f"od_groups.num_od={int(od_groups.num_od)}, "
            f"layout.num_od_total={layout.num_od_total}."
        )

    origin_full = np.asarray(od_groups.od_origin_node)
    destination_full = np.asarray(od_groups.od_dest_node)
    if origin_full.shape != (layout.num_od_total,):
        raise ValueError(
            f"od_origin_node must have shape ({layout.num_od_total},), "
            f"got {origin_full.shape}."
        )
    if destination_full.shape != (layout.num_od_total,):
        raise ValueError(
            f"od_dest_node must have shape ({layout.num_od_total},), "
            f"got {destination_full.shape}."
        )

    original_group_destinations = np.asarray(od_groups.group_dest_node)
    original_group_masks = np.asarray(od_groups.group_link_mask, dtype=bool)
    if original_group_destinations.ndim != 1:
        raise ValueError("group_dest_node must be one-dimensional.")
    if original_group_masks.ndim != 2 or original_group_masks.shape[0] != len(
        original_group_destinations
    ):
        raise ValueError(
            "group_link_mask must have shape (num_groups, num_links) consistent "
            "with group_dest_node."
        )
    if len(set(original_group_destinations.tolist())) != len(original_group_destinations):
        raise ValueError("group_dest_node must contain unique destinations.")

    mask_by_destination = {
        int(destination): original_group_masks[group_index]
        for group_index, destination in enumerate(original_group_destinations)
    }
    active = np.asarray(layout.active_full_indices, dtype=np.int64)
    compact_origins = origin_full[active]
    compact_destinations = destination_full[active]
    num_active = layout.num_active
    num_links = int(original_group_masks.shape[1])

    if num_active == 0:
        return ODGroups(
            num_od=0,
            od_origin_node=jnp.empty((0,), dtype=jnp.int32),
            od_dest_node=jnp.empty((0,), dtype=jnp.int32),
            group_start=jnp.asarray([0], dtype=jnp.int32),
            group_dest_node=jnp.empty((0,), dtype=jnp.int32),
            group_od_index=jnp.empty((0,), dtype=jnp.int32),
            group_od_index_padded=jnp.empty((0, 0), dtype=jnp.int32),
            group_od_mask=jnp.empty((0, 0), dtype=bool),
            group_link_mask=jnp.empty((0, num_links), dtype=bool),
        )

    order = np.argsort(compact_destinations, kind="mergesort")
    destinations_sorted = compact_destinations[order]
    change = np.ones(num_active, dtype=bool)
    change[1:] = destinations_sorted[1:] != destinations_sorted[:-1]
    group_starts = np.flatnonzero(change)
    group_start = np.append(group_starts, num_active)
    group_destinations = destinations_sorted[group_starts]
    num_groups = len(group_destinations)

    missing_destinations = [
        int(destination)
        for destination in group_destinations
        if int(destination) not in mask_by_destination
    ]
    if missing_destinations:
        raise ValueError(
            "Active OD cells refer to destinations absent from the original groups: "
            f"{missing_destinations}."
        )

    group_sizes = np.diff(group_start)
    max_group_size = int(group_sizes.max())
    padded = np.zeros((num_groups, max_group_size), dtype=np.int32)
    padded_mask = np.zeros((num_groups, max_group_size), dtype=bool)
    for group_index, (start, end) in enumerate(
        zip(group_start[:-1], group_start[1:], strict=True)
    ):
        size = int(end - start)
        padded[group_index, :size] = order[start:end]
        padded_mask[group_index, :size] = True

    compact_group_masks = np.stack(
        [mask_by_destination[int(destination)] for destination in group_destinations]
    )
    return ODGroups(
        num_od=num_active,
        od_origin_node=jnp.asarray(compact_origins, dtype=jnp.int32),
        od_dest_node=jnp.asarray(compact_destinations, dtype=jnp.int32),
        group_start=jnp.asarray(group_start, dtype=jnp.int32),
        group_dest_node=jnp.asarray(group_destinations, dtype=jnp.int32),
        group_od_index=jnp.asarray(order, dtype=jnp.int32),
        group_od_index_padded=jnp.asarray(padded, dtype=jnp.int32),
        group_od_mask=jnp.asarray(padded_mask, dtype=bool),
        group_link_mask=jnp.asarray(compact_group_masks, dtype=bool),
    )
