"""High-level assignment interface (JAX): OD demand -> link flows.

Pipeline:
- Build a time-expanded DAG from a Scenario.
- Build ODGroups (group by destination; centroid-in nodes are duplicated per time bin).
- Precompute group-independent link costs.
- For each destination group, run Dial-style DP to obtain link flows.

Contracts:
- Scenario provides timetable, stops, time_bins, and demand.
- ODGroups provides:
    - group_dest_node: (num_groups,) destination centroid-out node indices
    - group_link_mask: (num_groups, num_links) boolean enabled-link mask per group
    - od_origin_node: (num_od,) origin centroid-in node per OD record
    - group_od_index_padded: (num_groups, max_group_size) OD indices per group (padded)
    - group_od_mask: (num_groups, max_group_size) mask for valid OD indices
- link_costs(graph, cost_parts, config) returns (num_links,) generalized costs in minutes.

Note: the OD grouping is an internal performance detail; scripts should not handle it.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from public_transportation.domain.scenario import Scenario

from .build_od_groups import ODGroups, build_od_groups
from .build_time_expanded import build_jax_graph as build_time_expanded_graph
from .config import AssignmentConfig

from .costs import CostParts, precompute_base_costs, link_costs

from .dial_dp import run_dial_for_destination
from .jax_graph_types import JaxGraph
from .graph_sentinels import NODE_KIND_CENTROID_OUT, NODE_KIND_EVENT_ARR, LINK_TYPE_EGRESS, LINK_TYPE_ACCESS

Array = jnp.ndarray

BIG_M_COST = 1.0e6

# --------------------------------------------------------------------------------------
# JAX helpers (jit/scan-friendly core)
# --------------------------------------------------------------------------------------

def _dest_absorption_masks(
    *,
    graph: JaxGraph,
    dest_node: Array,
) -> tuple[Array, Array]:
    """Compute masks for destination absorption.

    Returns:
        keep_if_dest_arrival: links that are allowed to leave destination ARR nodes
        disallowed_dest_outgoing: all other outgoing links from destination ARR nodes

    Pure array logic; JIT-safe.
    """
    dest_stop_idx = graph.node_stop_index[dest_node]
    is_dest_arrival_node = (
        (graph.node_kind == NODE_KIND_EVENT_ARR)
        & (graph.node_stop_index == dest_stop_idx)
    )
    tail_is_dest_arrival = is_dest_arrival_node[graph.tail]
    keep_if_dest_arrival = (
        (graph.link_type == LINK_TYPE_EGRESS)
        & (graph.head == dest_node)
    )
    disallowed_dest_outgoing = tail_is_dest_arrival & (~keep_if_dest_arrival)
    return keep_if_dest_arrival, disallowed_dest_outgoing


def _initial_node_flow_for_group(
    *,
    num_nodes: int,
    od_values: Array,
    od_origin_node: Array,
    group_od_index_padded: Array,
    group_od_mask: Array,
    group_index: Array,
) -> Array:
    """JIT-safe construction of y0 for one destination group.

    This avoids Python slicing / int() coercions by relying on padded group indices.

    Expected shapes:
      - group_od_index_padded: (num_groups, max_group_size) int
      - group_od_mask:        (num_groups, max_group_size) bool
      - od_origin_node:       (num_od,) int
      - od_values:            (num_od,) float

    Returns:
      - y0: (num_nodes,) with OD flows injected at centroid-in origin nodes.
    """
    od_idx = group_od_index_padded[group_index]
    m = group_od_mask[group_index]

    # Make indices safe for masked entries.
    od_idx_safe = jnp.where(m, od_idx, jnp.zeros_like(od_idx))

    origin_nodes = od_origin_node[od_idx_safe]
    flows = od_values[od_idx_safe]

    origin_nodes_safe = jnp.where(m, origin_nodes, jnp.zeros_like(origin_nodes))
    flows_safe = jnp.where(m, flows, jnp.zeros_like(flows))

    y0 = jnp.zeros((num_nodes,), dtype=od_values.dtype)
    y0 = y0.at[origin_nodes_safe].add(flows_safe)
    return y0


@partial(jax.jit, static_argnames=("return_group_link_flows",))
def _assign_core(
    *,
    graph: JaxGraph,
    od_values: Array,
    base_link_cost: Array,
    theta: float,
    # ODGroups fields passed explicitly (avoid passing a Python object to jit)
    group_dest_node: Array,
    group_link_mask: Array,
    od_origin_node: Array,
    group_od_index_padded: Array,
    group_od_mask: Array,
    return_group_link_flows: bool,
) -> tuple[Array, Array | None]:
    """JIT-friendly assignment core.

    Returns:
      - total_link_flow: (num_links,)
      - group_link_flow: (num_groups, num_links) if requested else None

    All validation must remain outside this function.
    """
    num_groups = group_dest_node.shape[0]
    num_links = graph.num_links

    def step(total, g):
        dest_node = group_dest_node[g]
        enabled_link_mask = jnp.asarray(group_link_mask[g], dtype=bool)

        # Destination absorption masks
        _, disallowed = _dest_absorption_masks(graph=graph, dest_node=dest_node)
        enabled_link_mask = enabled_link_mask & (~disallowed)

        # Costs: base costs with BIG-M on disallowed links
        c = jnp.where(
            disallowed,
            jnp.asarray(BIG_M_COST, dtype=base_link_cost.dtype),
            base_link_cost,
        )

        y0 = _initial_node_flow_for_group(
            num_nodes=graph.num_nodes,
            od_values=od_values,
            od_origin_node=od_origin_node,
            group_od_index_padded=group_od_index_padded,
            group_od_mask=group_od_mask,
            group_index=g,
        )

        dial_res = run_dial_for_destination(
            graph=graph,
            link_cost=c,
            dest_node=dest_node,
            theta=theta,
            initial_node_flow=y0,
            enabled_link_mask=enabled_link_mask,
        )

        total = total + dial_res.link_flow
        return total, dial_res.link_flow

    init_total = jnp.zeros((num_links,), dtype=od_values.dtype)
    total_flow, per_group = lax.scan(step, init_total, jnp.arange(num_groups))

    if return_group_link_flows:
        return total_flow, per_group
    return total_flow, None

@dataclass(frozen=True, slots=True)
class AssignmentArtifacts:
    """Precomputed artifacts for repeated assignment evaluations.

    :param graph: Time-expanded graph in JAX arrays.
    :param od_groups: OD grouping structure used to inject demand per group.
    :param cost_parts: Precomputed group-independent cost components.
    :param config: Assignment configuration used to build these artifacts.
    """

    graph: JaxGraph
    od_groups: ODGroups
    cost_parts: CostParts
    config: AssignmentConfig


@dataclass(frozen=True, slots=True)
class AssignmentResult:
    """Output of one assignment evaluation.

    :param theta: Dispersion parameter used (minutes).
    :param link_flow: Total flow on each link (summed across all groups), shape (num_links,).
    :param link_cost: Base per-link generalized costs in minutes, shape (num_links,).
        Important: this is the group-independent base cost vector (ride/transfer/access penalties).
        Per-destination absorption safeguards (BIG_M on disallowed links) are applied inside the
        group loop and are not meaningful to report as a single global vector.
    :param group_link_flow: Optional per-group link flows, shape (num_groups, num_links).
        This is None by default to save memory.
    """

    theta: float
    link_flow: Array
    link_cost: Array
    group_link_flow: Array | None


def prepare_assignment(
    scenario: Scenario,
    config: AssignmentConfig,
) -> AssignmentArtifacts:
    """Build and validate everything that is invariant across OD-demand evaluations."""

    config.validate()

    if scenario.timetable is None:
        raise ValueError("Scenario.timetable is required for assignment.")

    # Build JAX graph representation (immutable arrays)
    graph = build_time_expanded_graph(scenario=scenario, config=config)

    # Build OD groups (group by destination; centroid-in nodes are duplicated per time bin).
    # NOTE: build_od_groups must accept the `graph` keyword.
    od_groups = build_od_groups(scenario=scenario, graph=graph)

    # Precompute base costs (ride/transfer)
    cost_parts = precompute_base_costs(graph, config)

    return AssignmentArtifacts(
        graph=graph,
        od_groups=od_groups,
        cost_parts=cost_parts,
        config=config,
    )


def _theta_value(theta: float | None, config: AssignmentConfig) -> float:
    """Resolve theta to use."""
    th = float(config.theta_default if theta is None else theta)
    if th <= 0.0:
        raise ValueError("theta must be positive.")
    th = max(th, float(config.theta_min))
    return th


def assign(
    od_values: Array,
    artifacts: AssignmentArtifacts,
    *,
    theta: float | None = None,
    return_group_link_flows: bool = False,
) -> AssignmentResult:
    """Evaluate the assignment model for a given OD-demand vector."""
    graph = artifacts.graph
    od_groups = artifacts.od_groups
    cost_parts = artifacts.cost_parts
    config = artifacts.config

    th = _theta_value(theta, config)

    od_values = jnp.asarray(od_values)
    if od_values.ndim != 1:
        raise ValueError(f"od_values must be a 1D array, got shape {od_values.shape}.")

    # ------------------------------------------------------------------
    # Base generalized link costs (group-independent)
    # ------------------------------------------------------------------
    base_link_cost = link_costs(
        graph=graph,
        cost_parts=cost_parts,
        config=config,
    )
    base_link_cost = jnp.asarray(base_link_cost, dtype=od_values.dtype)
    if base_link_cost.shape != (int(graph.num_links),):
        raise ValueError(
            f"link_costs returned shape {base_link_cost.shape}, expected {(int(graph.num_links),)}."
        )

    # Validate ODGroups fields (outside JIT)
    try:
        group_dest_node = od_groups.group_dest_node
    except AttributeError as e:
        raise ValueError("ODGroups is missing required field `group_dest_node` for assignment.") from e

    try:
        group_link_mask = od_groups.group_link_mask
    except AttributeError as e:
        raise ValueError("ODGroups is missing required field `group_link_mask` (num_groups, num_links).") from e

    try:
        od_origin_node = od_groups.od_origin_node
    except AttributeError as e:
        raise ValueError("ODGroups is missing required field `od_origin_node` (num_od,).") from e

    try:
        group_od_index_padded = od_groups.group_od_index_padded
    except AttributeError as e:
        raise ValueError(
            "ODGroups is missing required field `group_od_index_padded` (num_groups, max_group_size)."
        ) from e

    try:
        group_od_mask = od_groups.group_od_mask
    except AttributeError as e:
        raise ValueError(
            "ODGroups is missing required field `group_od_mask` (num_groups, max_group_size)."
        ) from e

    if group_dest_node.ndim != 1:
        raise ValueError(f"ODGroups.group_dest_node must be 1D, got shape {group_dest_node.shape}.")

    if int(jnp.min(group_dest_node)) < 0 or int(jnp.max(group_dest_node)) >= int(graph.num_nodes):
        raise ValueError("ODGroups.group_dest_node contains invalid node indices.")

    num_groups = int(group_dest_node.shape[0])
    if group_link_mask.shape != (num_groups, int(graph.num_links)):
        raise ValueError(
            "ODGroups.group_link_mask has wrong shape: "
            f"got {group_link_mask.shape}, expected {(num_groups, int(graph.num_links))}."
        )

    if od_origin_node.ndim != 1:
        raise ValueError(f"ODGroups.od_origin_node must be 1D, got shape {od_origin_node.shape}.")

    if group_od_index_padded.ndim != 2 or group_od_mask.ndim != 2:
        raise ValueError(
            "ODGroups.group_od_index_padded and ODGroups.group_od_mask must be 2D arrays. "
            f"Got shapes {group_od_index_padded.shape} and {group_od_mask.shape}."
        )

    if group_od_index_padded.shape != group_od_mask.shape:
        raise ValueError(
            "ODGroups.group_od_index_padded and ODGroups.group_od_mask must have the same shape. "
            f"Got {group_od_index_padded.shape} vs {group_od_mask.shape}."
        )

    if group_od_index_padded.shape[0] != num_groups:
        raise ValueError(
            "ODGroups.group_od_index_padded first dimension must match num_groups. "
            f"Got {group_od_index_padded.shape[0]} vs num_groups={num_groups}."
        )

    # Validate destination node kinds (outside JIT)
    kinds = jnp.asarray(graph.node_kind)[jnp.asarray(group_dest_node, dtype=jnp.int32)]
    if not bool(jnp.all(kinds == NODE_KIND_CENTROID_OUT)):
        bad = jnp.where(kinds != NODE_KIND_CENTROID_OUT, size=1, fill_value=-1)[0]
        bad = int(bad[0]) if int(bad[0]) >= 0 else -1
        if bad >= 0:
            dn = int(jnp.asarray(group_dest_node)[bad])
            raise ValueError(
                f"group_dest_node[{bad}]={dn} must be a destination centroid-out node "
                f"(expected node_kind=NODE_KIND_CENTROID_OUT, got node_kind={int(graph.node_kind[dn])}). "
                "This indicates OD grouping is inconsistent with the current graph; "
                "rebuild ODGroups using build_od_groups(..., graph=graph)."
            )

    # ------------------------------------------------------------------
    # JIT-friendly core: scan over groups
    # ------------------------------------------------------------------
    total_link_flow, group_link_flow = _assign_core(
        graph=graph,
        od_values=od_values,
        base_link_cost=base_link_cost,
        theta=th,
        group_dest_node=jnp.asarray(group_dest_node),
        group_link_mask=jnp.asarray(group_link_mask),
        od_origin_node=jnp.asarray(od_origin_node),
        group_od_index_padded=jnp.asarray(group_od_index_padded),
        group_od_mask=jnp.asarray(group_od_mask),
        return_group_link_flows=return_group_link_flows,
    )

    return AssignmentResult(
        theta=th,
        link_flow=total_link_flow,
        link_cost=base_link_cost,
        group_link_flow=group_link_flow,
    )

def assign_from_scenario(
    scenario: Scenario,
    od_values: Array,
    config: AssignmentConfig,
    *,
    theta: float | None = None,
    return_group_link_flows: bool = False,
) -> AssignmentResult:
    """Convenience wrapper: prepare artifacts and assign in one call."""

    artifacts = prepare_assignment(scenario, config)
    return assign(
        od_values,
        artifacts,
        theta=theta,
        return_group_link_flows=return_group_link_flows,
    )
