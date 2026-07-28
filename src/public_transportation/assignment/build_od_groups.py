"""Build OD groupings for the assignment.

OD records are grouped by destination (centroid-out) so the assignment can run one
backward value function per destination group.

Centroid-in nodes are duplicated per time bin: one node per (stop, time_bin).
Access-link costs depend only on link endpoints, so grouping is destination-only.

Output arrays are aligned with `scenario.demand.records` iteration order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

from .jax_graph_types import JaxGraph

from .graph_sentinels import (
    NODE_KIND_CENTROID_IN,
    NODE_KIND_CENTROID_OUT,
    LINK_TYPE_EGRESS,
)

if TYPE_CHECKING:  # pragma: no cover
    from public_transportation.domain import Scenario


Array = jnp.ndarray


@dataclass(frozen=True, slots=True)
class ODGroups:
    """Grouping of OD records for batched evaluation.

    All arrays refer to OD-record indices in the canonical order used to build
    the demand vector.

    Attributes
    ----------
    num_od : int
        Total number of OD records.

    od_origin_node : Array[int] shape (num_od,)
        Origin centroid-in node index for each OD record.

    od_dest_node : Array[int] shape (num_od,)
        Destination centroid-out node index for each OD record.

    group_start : Array[int] shape (num_groups + 1,)
        CSR-style pointers into `group_od_index`.

    group_dest_node : Array[int] shape (num_groups,)
        Destination centroid-out node for each group.

    group_od_index : Array[int] shape (num_od,)
        Concatenation of OD indices, grouped by destination.

    group_link_mask : Array[bool] shape (num_groups, num_links)
        Boolean mask indicating which links are enabled for each destination group.
        This implements destination-gated egress by disabling egress links not ending at the
        group's destination centroid-out node.
    """

    num_od: int
    od_origin_node: Array
    od_dest_node: Array
    group_start: Array
    group_dest_node: Array
    group_od_index: Array

    # JIT-friendly padded representation of OD indices per group:
    # group_od_index_padded[g, k] gives an OD index (valid only if group_od_mask[g, k] is True).
    group_od_index_padded: Array
    group_od_mask: Array

    group_link_mask: Array

    def make_initial_node_flow(
        self, group_index: int, od_values: Array, num_nodes: int
    ) -> Array:
        """Construct the node inflow vector y0 for one OD group (JIT-safe).

        Uses padded per-group OD indices to avoid traced Python slicing / int(...) conversions.
        """
        od_idx_padded = self.group_od_index_padded[group_index]  # (max_group_size,)
        m = self.group_od_mask[group_index]  # (max_group_size,) bool

        # Replace padded entries by a safe index (0). Mask them out afterwards.
        od_idx_safe = jnp.where(
            m, od_idx_padded, jnp.asarray(0, dtype=od_idx_padded.dtype)
        )

        origin_nodes = self.od_origin_node[od_idx_safe]
        flows = od_values[od_idx_safe] * m.astype(od_values.dtype)

        y0 = jnp.zeros((num_nodes,), dtype=od_values.dtype)
        y0 = y0.at[origin_nodes].add(flows)
        return y0

    def enabled_link_mask(self, group_index: int, num_links: int) -> Array:
        """Return the enabled-link mask for a destination group.

        :param group_index: Group id in [0, num_groups).
        :param num_links: Total number of links in the graph (used for validation).
        :return: Boolean mask of shape (num_links,).
        """
        m = self.group_link_mask[group_index]
        m = jnp.asarray(m, dtype=bool)
        if m.shape != (int(num_links),):
            raise ValueError(
                f"group_link_mask[{group_index}] has shape {m.shape}, expected {(int(num_links),)}."
            )
        return m

# --------------------------------------------------------------------------------------
# JAX pytree registration
# --------------------------------------------------------------------------------------

def _odgroups_flatten(g: "ODGroups"):
    children = (
        g.od_origin_node,
        g.od_dest_node,
        g.group_start,
        g.group_dest_node,
        g.group_od_index,
        g.group_od_index_padded,
        g.group_od_mask,
        g.group_link_mask,
    )
    aux = (int(g.num_od),)
    return children, aux

def _odgroups_unflatten(aux, children):
    (num_od,) = aux
    (
        od_origin_node,
        od_dest_node,
        group_start,
        group_dest_node,
        group_od_index,
        group_od_index_padded,
        group_od_mask,
        group_link_mask,
    ) = children
    return ODGroups(
        num_od=int(num_od),
        od_origin_node=od_origin_node,
        od_dest_node=od_dest_node,
        group_start=group_start,
        group_dest_node=group_dest_node,
        group_od_index=group_od_index,
        group_od_index_padded=group_od_index_padded,
        group_od_mask=group_od_mask,
        group_link_mask=group_link_mask,
    )

jax.tree_util.register_pytree_node(ODGroups, _odgroups_flatten, _odgroups_unflatten)


def build_od_groups(
    scenario: "Scenario",
    *,
    graph: JaxGraph,
    profile: dict[str, float] | None = None,
) -> ODGroups:
    """Build OD record arrays, groupings, and per-group link masks.

    Assumptions about the domain structures
    ---------------------------------------
    - `scenario.demand.records` is an iterable of demand records.
    - Each demand record has:
        - origin_stop_id
        - dest_stop_id
        - time_bin_index (preferred) or time_bin_id
        - demand (not used here; values are passed later as parameters)

    - `scenario.time_bins` is a sequence of time bins with fields:
        - bin_id (optional)
        - start: TimeOfDay (seconds_from_midnight)
        - end: TimeOfDay (seconds_from_midnight)

    Output alignment
    ----------------
    The returned OD ordering is exactly the iteration order of `scenario.demand.records`.
    The inference code should provide an OD vector aligned with this ordering.

    :param scenario: Domain scenario with demand and time bins.
    :param graph: Built time-expanded graph.
    :return: ODGroups with JAX arrays.
    """
    from time import perf_counter

    def record(name: str, started: float) -> None:
        if profile is not None:
            profile[name] = perf_counter() - started

    started = perf_counter()
    if scenario.demand is None:
        raise ValueError("Scenario has no demand.")
    if scenario.time_bins is None or len(scenario.time_bins) == 0:
        raise ValueError("Scenario has no time bins.")

    if not graph.node_stop_id:
        raise ValueError(
            "Graph is missing `node_stop_id` metadata. "
            "Ensure build_time_expanded stores stop ids in JaxGraph(node_stop_id=...)."
        )
    record("od_input_validation", started)


    # ---------------------------------------------------------------
    # Precompute bin_index_by_id for time-bin lookup
    # ---------------------------------------------------------------
    started = perf_counter()
    bin_index_by_id: dict[str, int] = {}
    for idx, tb in enumerate(scenario.time_bins):
        tb_id = getattr(tb, "bin_id", None)
        if tb_id is not None:
            bin_index_by_id[str(tb_id)] = int(idx)

    # ---------------------------------------------------------------
    # Derive centroid node indices from the built graph.
    # New design: centroid-in nodes are NOT time-tagged.
    # The builder creates exactly one centroid-in node per (stop_id, time_bin_index)
    # as the first block of nodes, in deterministic order:
    #   for sid in sorted(stop_ids):
    #       for t_idx in range(num_time_bins):
    #           create centroid-in
    # Therefore, the centroid-in node index is:
    #   idx = stop_pos * num_time_bins + t_idx
    # where stop_pos is the index of stop_id in `graph.node_stop_id`.
    # ---------------------------------------------------------------
    num_time_bins = len(scenario.time_bins)
    if num_time_bins <= 0:
        raise ValueError("Scenario has no time bins.")

    stop_pos_by_id = {str(sid): i for i, sid in enumerate(graph.node_stop_id)}
    num_stops = len(graph.node_stop_id)
    # Read immutable graph metadata once. Repeated ``int(jax_array[i])`` calls
    # each synchronize a scalar device result and dominated large preparations.
    node_kind = np.asarray(graph.node_kind)
    node_stop_index = np.asarray(graph.node_stop_index)

    expected_num_centroid_in = num_stops * num_time_bins
    if int(graph.num_nodes) < expected_num_centroid_in:
        raise ValueError(
            "Graph has fewer nodes than expected for centroid-in block. "
            f"Expected at least {expected_num_centroid_in} nodes (|stops|={num_stops}, |time_bins|={num_time_bins}), "
            f"got num_nodes={int(graph.num_nodes)}."
        )

    centroid_in_index: dict[tuple[str, int], int] = {}
    # Validate the centroid-in block and fill the mapping.
    for sid, spos in stop_pos_by_id.items():
        for t_idx in range(num_time_bins):
            node = int(spos * num_time_bins + t_idx)
            if int(node_kind[node]) != NODE_KIND_CENTROID_IN:
                raise ValueError(
                    "Graph centroid-in block is inconsistent with expected ordering. "
                    f"Node {node} for stop_id={sid}, time_bin_index={t_idx} has node_kind={int(node_kind[node])}."
                )
            if int(node_stop_index[node]) != int(spos):
                raise ValueError(
                    "Graph centroid-in block has inconsistent node_stop_index. "
                    f"Node {node} for stop_id={sid}, time_bin_index={t_idx} has node_stop_index={int(node_stop_index[node])}, "
                    f"expected {spos}."
                )
            centroid_in_index[(sid, int(t_idx))] = node

    # Centroid-out nodes are not time-tagged either; we derive their indices by scanning nodes.
    centroid_out_index: dict[str, int] = {}
    for node in range(int(graph.num_nodes)):
        if int(node_kind[node]) != NODE_KIND_CENTROID_OUT:
            continue
        s_idx = int(node_stop_index[node])
        if s_idx < 0 or s_idx >= len(graph.node_stop_id):
            raise ValueError(f"Invalid node_stop_index for node {node}: {s_idx}")
        sid = str(graph.node_stop_id[s_idx])
        centroid_out_index[sid] = int(node)

    if len(centroid_out_index) == 0:
        raise ValueError(
            "Could not derive centroid-out node indices from the graph. "
            "Check that build_time_expanded created centroid-out nodes and populated node_kind/node_stop_index."
        )

    # ---------------------------------------------------------------
    # Read OD records
    # ---------------------------------------------------------------
    records = list(scenario.demand.records)
    num_od = len(records)
    if num_od == 0:
        raise ValueError("Demand has zero records.")

    od_origin_node = np.empty(num_od, dtype=int)
    od_dest_node = np.empty(num_od, dtype=int)

    for k, r in enumerate(records):
        o = str(r.origin_stop_id)
        d = str(r.dest_stop_id)

        tb_index = getattr(r, "time_bin_index", None)
        tb_id = getattr(r, "time_bin_id", None)
        if tb_index is None and tb_id is None:
            raise ValueError("Demand record missing time_bin_index/time_bin_id.")

        if tb_index is not None:
            tb_index_int = int(tb_index)
            if tb_index_int < 0 or tb_index_int >= len(scenario.time_bins):
                raise ValueError(f"Unknown time bin index in demand: {tb_index_int}")
        else:
            tb_id_str = str(tb_id)
            if tb_id_str not in bin_index_by_id:
                raise ValueError(f"Unknown time bin id in demand: {tb_id_str}")
            tb_index_int = int(bin_index_by_id[tb_id_str])

        if d not in centroid_out_index:
            raise ValueError(f"Unknown dest_stop_id in demand: {d}")

        key = (o, tb_index_int)
        if key not in centroid_in_index:
            raise ValueError(
                f"Unknown origin/time-bin combination in demand: origin_stop_id={o}, time_bin_index={tb_index_int}. "
                "Ensure the time-expanded graph contains a centroid-in node for this (stop, time bin)."
            )

        od_origin_node[k] = int(centroid_in_index[key])
        od_dest_node[k] = int(centroid_out_index[d])
    record("od_and_destination_indexing", started)

    # ---------------------------------------------------------------
    # Group by destination (centroid-out) only.
    # ---------------------------------------------------------------
    started = perf_counter()
    keys = od_dest_node
    order = np.argsort(keys, kind="mergesort")  # stable, deterministic
    keys_sorted = keys[order]

    change = np.ones(num_od, dtype=bool)
    change[1:] = keys_sorted[1:] != keys_sorted[:-1]
    group_starts = np.nonzero(change)[0]
    num_groups = int(group_starts.shape[0])

    group_start = np.empty(num_groups + 1, dtype=int)
    group_start[:-1] = group_starts
    group_start[-1] = num_od

    group_dest_node = np.empty(num_groups, dtype=int)
    for g in range(num_groups):
        i0 = group_start[g]
        group_dest_node[g] = int(keys_sorted[i0])

    group_od_index = order.astype(int)
    record("destination_grouping_and_stable_sort", started)

    # ---------------------------------------------------------------
    # JIT-friendly padded OD indices per group
    # ---------------------------------------------------------------
    started = perf_counter()
    group_sizes = np.diff(group_start)  # (num_groups,)
    max_group_size = int(group_sizes.max()) if num_groups > 0 else 0

    group_od_index_padded = np.zeros((num_groups, max_group_size), dtype=int)
    group_od_mask = np.zeros((num_groups, max_group_size), dtype=bool)

    for g in range(num_groups):
        i0 = int(group_start[g])
        i1 = int(group_start[g + 1])
        n = int(i1 - i0)
        if n <= 0:
            continue
        idx = group_od_index[i0:i1]
        group_od_index_padded[g, :n] = idx
        group_od_mask[g, :n] = True
    record("od_padded_index_and_mask_construction", started)

    # ---------------------------------------------------------------
    # Per-group enabled-link masks implementing destination-gated egress.
    # link_type codes: 0 ride, 1 transfer, 2 access, 3 egress, 4 dwell/continue.
    # ---------------------------------------------------------------
    started = perf_counter()
    link_type = np.asarray(graph.link_type)
    head = np.asarray(graph.head)

    is_egress = link_type == LINK_TYPE_EGRESS
    group_link_mask = (~is_egress[None, :]) | (
        head[None, :] == group_dest_node[:, None]
    )
    record("destination_link_mask_construction", started)

    # Sanity: group destinations must be centroid-out nodes.
    for g in range(num_groups):
        dnode = int(group_dest_node[g])
        if int(node_kind[dnode]) != NODE_KIND_CENTROID_OUT:
            raise ValueError(
                f"group_dest_node[{g}]={dnode} is not a centroid-out node (node_kind={int(node_kind[dnode])})."
            )

    started = perf_counter()
    result = ODGroups(
        num_od=num_od,
        od_origin_node=jnp.asarray(od_origin_node),
        od_dest_node=jnp.asarray(od_dest_node),
        group_start=jnp.asarray(group_start),
        group_dest_node=jnp.asarray(group_dest_node),
        group_od_index=jnp.asarray(group_od_index),
        group_od_index_padded=jnp.asarray(group_od_index_padded),
        group_od_mask=jnp.asarray(group_od_mask),
        group_link_mask=jnp.asarray(group_link_mask),
    )
    jax.block_until_ready(result)
    record("od_numpy_to_jax_device_transfer_and_synchronization", started)
    return result
