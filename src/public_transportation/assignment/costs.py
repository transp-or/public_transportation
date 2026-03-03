"""
Cost computation for the time-expanded public transportation assignment model.

This module computes *generalized costs* per link for the Dial-style (logit)
dynamic programming assignment.

Key points
----------
- All costs are expressed in **minutes**.
- Only OD demand (and optionally the dispersion parameter theta) is estimated.
  All cost coefficients are fixed and provided by the user via AssignmentConfig.
- Link costs depend on link type:

  1) Ride links (event_dep -> event_arr): cost = in-vehicle travel time (minutes)
  2) Transfer links (event_arr -> event_dep, inter-line): cost = beta_transfer * waiting_time (minutes)
  3) Access links (centroid-in -> event_dep): cost = beta_early * max(0, a - tau) + beta_late * max(0, tau - b)
     where [a, b] is the centroid-in node's own departure interval (time bin) and tau is the departure-event time.
     Since centroid-in nodes are duplicated per time bin, costs are group-independent.
  4) Egress links (event_arr -> centroid-out): cost = 0
  5) Dwell/continue links (event_arr -> event_dep, same trip at same stop): cost = dwell time (minutes)

Important
---------
Access-link costs depend on the *departure-time interval bounds* [a, b] of the centroid-in node.
With centroid-in nodes duplicated per time bin, access penalties are group-independent.

Capacity penalties (future extension)
-------------------------------------
The first implementation does not include capacity penalties. The API includes
a hook for future use (reference flows and capacities), but by default it returns
zero penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .config import AssignmentConfig
from .jax_graph_types import JaxGraph
from .graph_sentinels import (
    LINK_TYPE_RIDE,
    LINK_TYPE_TRANSFER,
    LINK_TYPE_ACCESS,
    LINK_TYPE_EGRESS,
    LINK_TYPE_DWELL,
)



Array = jnp.ndarray


@dataclass(frozen=True, slots=True)
class CostParts:
    """
    Precomputed group-independent cost components.

    :param base_cost: Base generalized cost for each link, without access penalties,
        shape (num_links,). For access links, base_cost is 0.
    :param is_access: Boolean mask for access links, shape (num_links,).
    :param is_ride: Boolean mask for ride links, shape (num_links,).
    :param is_transfer: Boolean mask for transfer links, shape (num_links,).
    :param is_egress: Boolean mask for egress links, shape (num_links,).
    :param is_dwell: Boolean mask for dwell/continue links, shape (num_links,).
    """
    base_cost: Array
    is_access: Array
    is_ride: Array
    is_transfer: Array
    is_egress: Array
    is_dwell: Array


def precompute_base_costs(graph: JaxGraph, config: AssignmentConfig) -> CostParts:
    """
    Precompute costs that do not depend on OD desired departure intervals.

    - Ride: travel_time
    - Transfer: beta_transfer * travel_time
    - Access: 0 here (schedule-deviation penalty handled later per group)
    - Egress: 0
    - Dwell: dwell_time (minutes)

    :param graph: Time-expanded JAX graph.
    :param config: Assignment configuration (coefficients).
    :return: CostParts with base costs and masks.
    """
    config.validate()

    lt = graph.link_type
    is_ride = lt == LINK_TYPE_RIDE
    is_transfer = lt == LINK_TYPE_TRANSFER
    is_access = lt == LINK_TYPE_ACCESS
    is_egress = lt == LINK_TYPE_EGRESS
    is_dwell = lt == LINK_TYPE_DWELL

    # travel_time is in minutes (float)
    # - ride links: in-vehicle travel time
    # - transfer links: waiting time weighted by beta_transfer
    # - dwell links: dwell time (same units as travel_time)
    base_cost = jnp.where(is_ride, graph.travel_time, 0.0)
    base_cost = jnp.where(is_transfer, config.beta_transfer * graph.travel_time, base_cost)
    base_cost = jnp.where(is_dwell, graph.travel_time, base_cost)
    # access links: keep at 0.0 (penalty added later, per OD-group)
    # egress links: keep at 0.0 (can be made configurable later)

    return CostParts(
        base_cost=base_cost,
        is_access=is_access,
        is_ride=is_ride,
        is_transfer=is_transfer,
        is_egress=is_egress,
        is_dwell=is_dwell,
    )



def _access_penalty(
    *,
    graph: JaxGraph,
    cost_parts: CostParts,
    config: AssignmentConfig,
) -> Array:
    """
    Compute access-link penalties using the centroid-in node's own interval.
    This uses graph.node_bin_start_min and graph.node_bin_end_min (in minutes) for the tail node of each access link.
    All non-access links receive 0 penalty.
    """
    # Check required fields
    if not (hasattr(graph, "node_bin_start_min") and hasattr(graph, "node_bin_end_min")):
        raise RuntimeError(
            "Graph is missing node_bin_start_min/node_bin_end_min. "
            "These are required for group-independent access penalties."
        )
    access_link_idx = jnp.where(cost_parts.is_access, size=graph.num_links, fill_value=-1)[0]
    access_link_idx = access_link_idx[access_link_idx >= 0]
    pen_all = jnp.zeros((graph.num_links,), dtype=graph.node_time.dtype)
    def _compute_and_scatter(pen_vec: Array) -> Array:
        tau = graph.node_time[graph.head[access_link_idx]]  # minutes at head node for access links
        a = graph.node_bin_start_min[graph.tail[access_link_idx]]
        b = graph.node_bin_end_min[graph.tail[access_link_idx]]
        early = jnp.maximum(0.0, a - tau)
        late = jnp.maximum(0.0, tau - b)
        pen = config.beta_early * early + config.beta_late * late
        return pen_vec.at[access_link_idx].set(pen)
    pen_all = jax.lax.cond(
        access_link_idx.size > 0,
        _compute_and_scatter,
        lambda p: p,
        pen_all,
    )
    return pen_all



def link_costs(
    *,
    graph: JaxGraph,
    cost_parts: CostParts,
    config: AssignmentConfig,
) -> Array:
    """
    Compute effective link costs (group-independent).

    Returns generalized costs in minutes, including:
    - base costs (ride/transfer/dwell)
    - access schedule-deviation penalties using the centroid-in node's interval
    """
    config.validate()
    access_pen = _access_penalty(
        graph=graph,
        cost_parts=cost_parts,
        config=config,
    )
    return cost_parts.base_cost + access_pen


def link_costs_for_group(
    *,
    graph: JaxGraph,
    cost_parts: CostParts,
    config: AssignmentConfig,
    od_groups: Any,
    group_index: int,
) -> Array:
    """
    [DEPRECATED] Compute effective link costs for one OD group.
    This function is deprecated now that centroid-in nodes are duplicated per time bin.
    Group-dependent costs are no longer needed; use `link_costs()` instead.
    """
    # Ignore od_groups and group_index, call group-independent version.
    return link_costs(graph=graph, cost_parts=cost_parts, config=config)


# Backward compatibility shim for interval-based API
def link_costs_for_interval(
    *,
    graph: JaxGraph,
    cost_parts: CostParts,
    config: AssignmentConfig,
    bin_start_min: float,
    bin_end_min: float,
) -> Array:
    """
    Legacy helper: prefer `link_costs()` (group-independent).
    If the graph exposes node_bin_start_min/node_bin_end_min, interval is encoded on centroid-in nodes.
    Otherwise, falls back to legacy computation with explicit interval.
    """
    config.validate()
    if hasattr(graph, "node_bin_start_min") and hasattr(graph, "node_bin_end_min"):
        # Interval is encoded per centroid-in node; use group-independent costs
        return link_costs(graph=graph, cost_parts=cost_parts, config=config)
    # Legacy fallback: compute access penalties for the provided interval
    access_link_idx = jnp.where(cost_parts.is_access, size=graph.num_links, fill_value=-1)[0]
    access_link_idx = access_link_idx[access_link_idx >= 0]
    pen_all = jnp.zeros((graph.num_links,), dtype=graph.node_time.dtype)
    def _compute_and_scatter(pen_vec: Array) -> Array:
        tau = graph.node_time[graph.head[access_link_idx]]  # minutes at head node for access links
        a = float(bin_start_min)
        b = float(bin_end_min)
        early = jnp.maximum(0.0, a - tau)
        late = jnp.maximum(0.0, tau - b)
        pen = config.beta_early * early + config.beta_late * late
        return pen_vec.at[access_link_idx].set(pen)
    pen_all = jax.lax.cond(
        access_link_idx.size > 0,
        _compute_and_scatter,
        lambda p: p,
        pen_all,
    )
    access_pen = pen_all
    return cost_parts.base_cost + access_pen


def typical_cost_scale_from_assignment(
    *,
    graph: JaxGraph,
    link_flow: Array,
    link_cost: Array,
    demand_total: float | None = None,
    eps: float = 1.0e-12,
) -> float:
    """Compute a single typical generalized-cost scale (minutes) from an assignment output.

    Purpose
    -------
    This helper is intended to provide an order-of-magnitude scale for choosing a prior
    on the logit dispersion parameter ``theta``.

    Rationale
    ---------
    The logit routing model uses ``exp(-C/\theta)``. Therefore, the magnitude of ``theta``
    should be comparable to typical *differences* in generalized costs along reasonable
    paths. A practical, robust proxy is the *average generalized cost experienced per
    passenger* under a reference demand (e.g., a prior OD matrix).

    Definition
    ----------
    Let ``x_\ell`` be the predicted flow on link ``\ell`` and ``c_\ell`` its generalized
    cost (minutes). We define

        scale = (\sum_\ell x_\ell c_\ell) / D,

    where ``D`` is the total number of passengers.

    By default, ``D`` is inferred from the total flow on ACCESS links (each passenger
    boards exactly once in the current assignment model). Alternatively, the caller may
    pass ``demand_total`` (e.g., ``sum(od_values)``).

    Notes
    -----
    - This function is meant to be called *once* outside the inner inference loop.
      It is not jitted and may perform device-to-host transfers when returning a Python
      float.
    - If the model later introduces opt-out or multi-boarding semantics, the default
      inference of ``D`` from ACCESS links must be revisited; in that case pass
      ``demand_total`` explicitly.

    Parameters
    ----------
    graph:
        The time-expanded graph (used only to identify ACCESS links).
    link_flow:
        Predicted link flows, shape ``(num_links,)``.
    link_cost:
        Per-link generalized costs in minutes, shape ``(num_links,)``.
        This should be the same cost vector used by the assignment for the given run.
    demand_total:
        Optional total demand (passengers). If provided, it overrides the default
        ACCESS-flow-based inference.
    eps:
        Small positive constant to avoid division by zero.

    Returns
    -------
    float
        Typical generalized-cost scale in minutes.

    Raises
    ------
    ValueError
        If the inferred or provided total demand is non-positive.
    """
    lf = jnp.asarray(link_flow)
    lc = jnp.asarray(link_cost)

    if lf.ndim != 1 or lc.ndim != 1:
        raise ValueError(f"link_flow and link_cost must be 1D arrays, got {lf.shape} and {lc.shape}.")
    if lf.shape[0] != int(graph.num_links) or lc.shape[0] != int(graph.num_links):
        raise ValueError(
            "link_flow/link_cost length must match graph.num_links: "
            f"got {lf.shape[0]} and {lc.shape[0]} vs {int(graph.num_links)}."
        )

    # Total generalized cost experienced (minutes * passengers)
    total_cost = jnp.sum(lf * lc)

    # Total passengers.
    if demand_total is None:
        # In the current model each passenger contributes exactly one ACCESS link.
        is_access = jnp.asarray(graph.link_type) == LINK_TYPE_ACCESS
        total_demand = jnp.sum(jnp.where(is_access, lf, jnp.asarray(0.0, dtype=lf.dtype)))
    else:
        total_demand = jnp.asarray(float(demand_total), dtype=lf.dtype)

    # Guard against degenerate cases.
    total_demand = jnp.maximum(total_demand, jnp.asarray(0.0, dtype=lf.dtype))

    # Convert to Python float for convenience.
    scale = total_cost / jnp.maximum(total_demand, jnp.asarray(eps, dtype=lf.dtype))
    scale_f = float(jax.device_get(scale))

    if scale_f <= 0.0:
        raise ValueError(
            "Typical cost scale is non-positive. "
            "Check that link flows are positive and that demand_total is correctly specified."
        )

    return scale_f


@jax.jit
def stable_logit_transition_logits(
    *,
    link_cost: Array,
    v_head: Array,
    theta: float,
) -> Array:
    """
    Compute logits for outgoing links in the Bellman recursion.

    For a link (i -> j) the contribution is:
        -(cost_link + V(j)) / theta

    :param link_cost: Cost per link, shape (num_links,).
    :param v_head: Value function at head nodes, shape (num_nodes,).
    :param theta: Dispersion parameter (minutes).
    :return: Logits per link, shape (num_links,).
    """
    return -(link_cost + v_head) / theta