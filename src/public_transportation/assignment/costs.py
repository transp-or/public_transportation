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

  1) Ride links:
       cost = in-vehicle travel time (minutes)

  2) Transfer links:
       cost = beta_transfer * transfer_time (minutes)

  3) Access links (centroid -> event):
       cost = beta_early * early_minutes + beta_late * late_minutes,
       where early/late are computed relative to the desired departure interval [a_t, b_t]
       for the corresponding OD group.

Important
---------
Access-link costs depend on the *desired departure interval*.
Therefore they are computed **per OD-group** (destination + time-bin) because
within a group all OD records share the same time window.

Ride/transfer costs are group-independent and can be precomputed once.

Capacity penalties (future extension)
-------------------------------------
The first implementation does not include capacity penalties. The API includes
a hook for future use (reference flows and capacities), but by default it returns
zero penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .config import AssignmentConfig
from .jax_graph_types import JaxGraph

if TYPE_CHECKING:  # pragma: no cover
    from .build_od_groups import ODGroups


Array = jnp.ndarray


# Link-type codes (must match build_time_expanded.py)
LINK_RIDE = 0
LINK_TRANSFER = 1
LINK_ACCESS = 2


@dataclass(frozen=True, slots=True)
class CostParts:
    """
    Precomputed group-independent cost components.

    :param base_cost: Base generalized cost for each link, without access penalties,
        shape (num_links,). For access links, base_cost is 0.
    :param is_access: Boolean mask for access links, shape (num_links,).
    :param is_ride: Boolean mask for ride links, shape (num_links,).
    :param is_transfer: Boolean mask for transfer links, shape (num_links,).
    """
    base_cost: Array
    is_access: Array
    is_ride: Array
    is_transfer: Array


def precompute_base_costs(graph: JaxGraph, config: AssignmentConfig) -> CostParts:
    """
    Precompute costs that do not depend on OD desired departure intervals.

    - Ride: travel_time
    - Transfer: beta_transfer * travel_time
    - Access: 0 here (handled later per group)

    :param graph: Time-expanded JAX graph.
    :param config: Assignment configuration (coefficients).
    :return: CostParts with base costs and masks.
    """
    config.validate()

    lt = graph.link_type
    is_ride = lt == LINK_RIDE
    is_transfer = lt == LINK_TRANSFER
    is_access = lt == LINK_ACCESS

    # travel_time is in minutes (float)
    base_cost = jnp.where(is_ride, graph.travel_time, 0.0)
    base_cost = jnp.where(is_transfer, config.beta_transfer * graph.travel_time, base_cost)
    # access links: keep at 0.0 (penalty added later)

    return CostParts(
        base_cost=base_cost,
        is_access=is_access,
        is_ride=is_ride,
        is_transfer=is_transfer,
    )


def _access_penalty_for_group(
    *,
    graph: JaxGraph,
    cost_parts: CostParts,
    config: AssignmentConfig,
    a_min: float,
    b_min: float,
) -> Array:
    """
    Compute access-link penalties for a specific desired departure interval [a_min, b_min].

    For an access link from centroid -> event node at time tau (minutes):
      early = max(0, a_min - tau)
      late  = max(0, tau - b_min)
      penalty = beta_early * early + beta_late * late

    All non-access links receive 0 penalty.

    :param graph: JAX graph.
    :param cost_parts: Precomputed masks.
    :param config: Configuration.
    :param a_min: Interval lower bound in minutes.
    :param b_min: Interval upper bound in minutes.
    :return: Penalty per link, shape (num_links,).
    """
    # Access link head is an event node with a meaningful node_time (minutes)
    # graph.node_time is currently stored in the builder only indirectly via topological order.
    # We reconstruct event time by using head node time from a cached node_time array.
    #
    # In the first implementation, JaxGraph does NOT yet carry node_time. For access penalties,
    # we require it. The builder should provide it; until then, we raise a clear error.
    if getattr(graph, "node_time", None) is None:
        raise ValueError(
            "JaxGraph is missing `node_time` (minutes). "
            "Update build_time_expanded.py / jax_graph_types.py to store node_time "
            "so access-link schedule-deviation penalties can be computed."
        )

    tau = graph.node_time[graph.head]  # minutes at event node (for access links)
    early = jnp.maximum(0.0, a_min - tau)
    late = jnp.maximum(0.0, tau - b_min)
    pen = config.beta_early * early + config.beta_late * late

    return jnp.where(cost_parts.is_access, pen, 0.0)


def link_costs_for_group(
    *,
    graph: JaxGraph,
    cost_parts: CostParts,
    config: AssignmentConfig,
    a_min: float,
    b_min: float,
    theta: float,
) -> Array:
    """
    Compute effective link costs for one OD group (destination + time-bin).

    The returned costs are generalized costs in minutes, including:
    - base costs (ride/transfer)
    - access schedule-deviation penalties for the group's desired departure interval

    Capacity penalties are not included in the first implementation.

    :param graph: Time-expanded JAX graph.
    :param cost_parts: Precomputed base costs and masks.
    :param config: Assignment configuration.
    :param a_min: Desired departure interval lower bound (minutes).
    :param b_min: Desired departure interval upper bound (minutes).
    :param theta: Dispersion parameter of the route-choice logit (minutes). Not used directly
        in costs, but validated here for numerical stability (theta > 0).
    :return: Effective link costs, shape (num_links,).
    """
    config.validate()
    if theta <= 0.0:
        raise ValueError("theta must be positive.")

    access_pen = _access_penalty_for_group(
        graph=graph,
        cost_parts=cost_parts,
        config=config,
        a_min=a_min,
        b_min=b_min,
    )

    return cost_parts.base_cost + access_pen


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