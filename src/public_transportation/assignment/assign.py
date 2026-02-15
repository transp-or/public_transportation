"""
High-level assignment interface: OD demand -> link flows (JAX).

This module glues together:

- building the time-expanded graph (from the domain Scenario),
- grouping OD records (to enable efficient evaluation),
- computing link costs (including access schedule-deviation penalties),
- running Dial-style backward/forward DP to obtain link flows.

Scope of the first implementation
---------------------------------
- Deterministic, differentiable Dial-style assignment on a time-expanded DAG.
- Parameters to be estimated: OD demand vector (and optionally theta).
- All other coefficients (transfer penalty, early/late penalties, etc.) are fixed
  and provided by the user via AssignmentConfig.
- Opt-out alternative is NOT implemented in the first version.
- Capacity penalties are NOT implemented in the first version.

Performance notes
-----------------
This implementation is correct and modular, and already JAX-friendly.
However, it uses a Python loop over OD groups. This is usually fine because
the number of groups (destination x time-bin) is modest compared to the number
of OD records. If needed later, we can vmap/scan over groups for more speed.

Contracts (IMPORTANT)
---------------------
- The Scenario must contain: stops, time_bins, demand, timetable (trips + stop_times).
- build_time_expanded(...) must produce a JaxGraph with:
    - node_time (minutes)  (needed for access penalties)
    - CSR adjacency: out_start (num_nodes+1) and out_links (num_links) (needed for fast DP)
    - tail/head/link_type/travel_time arrays
- build_od_groups(...) must produce ODGroups describing how to:
    - map each OD-demand record to a group,
    - inject demand into centroid nodes for each group,
    - identify the destination node for the group.
"""

from __future__ import annotations

from dataclasses import dataclass


import jax
import jax.numpy as jnp

from public_transportation.domain.scenario import Scenario

from .config import AssignmentConfig
from .jax_graph_types import JaxGraph
from .build_time_expanded import build_jax_graph as build_time_expanded_graph
from .build_od_groups import ODGroups, build_od_groups
from .costs import precompute_base_costs, link_costs_for_group, CostParts
from .dial_dp import run_dial_for_destination

Array = jnp.ndarray


@dataclass(frozen=True, slots=True)
class AssignmentArtifacts:
    """
    Precomputed artifacts for repeated assignment evaluations.

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
    """
    Output of one assignment evaluation.

    :param theta: Dispersion parameter used (minutes).
    :param link_flow: Total flow on each link (summed across all groups), shape (num_links,).
    :param group_link_flow: Optional per-group link flows, shape (num_groups, num_links).
        This is None by default to save memory.
    """
    theta: float
    link_flow: Array
    group_link_flow: Array | None


def prepare_assignment(
    scenario: Scenario,
    config: AssignmentConfig,
) -> AssignmentArtifacts:
    """
    Build and validate everything that is invariant across OD-demand evaluations.

    Typical usage:
        artifacts = prepare_assignment(scenario, config)
        res = assign(od_vector, artifacts)

    :param scenario: Domain Scenario.
    :param config: AssignmentConfig (fixed coefficients).
    :return: AssignmentArtifacts.
    """
    config.validate()

    if scenario.timetable is None:
        raise ValueError("Scenario.timetable is required for assignment.")

    # Build JAX graph representation (immutable arrays)
    graph = build_time_expanded_graph(scenario=scenario, config=config)

    # Build OD groups (destination x time-bin, etc.)
    od_groups = build_od_groups(scenario=scenario, graph=graph)

    # Precompute base costs (ride/transfer) and masks
    cost_parts = precompute_base_costs(graph, config)

    return AssignmentArtifacts(
        graph=graph,
        od_groups=od_groups,
        cost_parts=cost_parts,
        config=config,
    )


def _theta_value(
    theta: float | None,
    config: AssignmentConfig,
) -> float:
    """
    Resolve theta to use.

    :param theta: Optional theta passed by the user.
    :param config: AssignmentConfig (default theta and bounds).
    :return: Theta value to use.
    """
    th = float(config.theta_default if theta is None else theta)
    if th <= 0.0:
        raise ValueError("theta must be positive.")
    theta_min = getattr(config, "theta_min", None)
    if theta_min is not None:
        th = max(th, float(theta_min))
    return th


def _build_initial_node_flow_for_group(
    *,
    graph: JaxGraph,
    od_groups: ODGroups,
    group_index: int,
    od_values: Array,
) -> Array:
    """
    Construct the node inflow vector y0 for one group.

    This function delegates almost all logic to ODGroups, because the mapping
    depends on how you decide to represent OD demand and groups.

    Expected ODGroups API
    ---------------------
    ODGroups should provide a method:

        make_initial_node_flow(group_index: int, od_values: Array, num_nodes: int) -> Array

    returning a vector y0 of shape (num_nodes,) with demand injected at centroid nodes.

    :param graph: JaxGraph.
    :param od_groups: ODGroups.
    :param group_index: Group id in [0, num_groups).
    :param od_values: OD-demand vector (JAX), interpretation defined by ODGroups.
    :return: Initial node flow vector y0, shape (num_nodes,).
    """
    if not hasattr(od_groups, "make_initial_node_flow"):
        raise ValueError(
            "ODGroups is missing method `make_initial_node_flow(group_index, od_values, num_nodes)`."
        )
    return od_groups.make_initial_node_flow(
        group_index=group_index,
        od_values=od_values,
        num_nodes=graph.num_nodes,
    )


def assign(
    od_values: Array,
    artifacts: AssignmentArtifacts,
    *,
    theta: float | None = None,
    return_group_link_flows: bool = False,
) -> AssignmentResult:
    """
    Evaluate the assignment model for a given OD-demand vector.

    :param od_values: OD-demand parameters as a JAX array.
        The exact layout is defined by ODGroups. The simplest convention is
        to store one value per demand record in scenario.demand, in the same order.
    :param artifacts: Precomputed AssignmentArtifacts from prepare_assignment(...).
    :param theta: Optional logit dispersion parameter (minutes). If None,
        uses artifacts.config.theta_default.
    :param return_group_link_flows: If True, also return per-group link flows
        (can be large; useful for debugging).
    :return: AssignmentResult with total link flows.
    """
    graph = artifacts.graph
    od_groups = artifacts.od_groups
    cost_parts = artifacts.cost_parts
    config = artifacts.config

    th = _theta_value(theta, config)

    num_links = graph.num_links
    num_groups = od_groups.num_groups

    total_link_flow = jnp.zeros((num_links,), dtype=jnp.asarray(od_values).dtype)

    group_link_flow = (
        jnp.zeros((num_groups, num_links), dtype=total_link_flow.dtype)
        if return_group_link_flows
        else None
    )

    # NOTE: Python loop over groups. Usually OK because num_groups is modest.
    # If we need to push further, we can vmap/scan once ODGroups exposes
    # fully vectorized group injection arrays.
    for g in range(int(num_groups)):
        # Group metadata
        if not hasattr(od_groups, "dest_node"):
            raise ValueError("ODGroups must provide `dest_node` array of shape (num_groups,).")
        if not hasattr(od_groups, "a_min") or not hasattr(od_groups, "b_min"):
            raise ValueError("ODGroups must provide `a_min` and `b_min` arrays of shape (num_groups,).")

        dest_node = int(od_groups.dest_node[g])
        a_min = float(od_groups.a_min[g])
        b_min = float(od_groups.b_min[g])

        # Costs for this group (access penalties depend on [a_min, b_min])
        c = link_costs_for_group(
            graph=graph,
            cost_parts=cost_parts,
            config=config,
            a_min=a_min,
            b_min=b_min,
            theta=th,
        )

        # Initial node flow for this group
        y0 = _build_initial_node_flow_for_group(
            graph=graph,
            od_groups=od_groups,
            group_index=g,
            od_values=od_values,
        )

        # Dial backward/forward pass for this destination
        dial_res = run_dial_for_destination(
            graph=graph,
            link_cost=c,
            dest_node=dest_node,
            theta=th,
            initial_node_flow=y0,
        )

        total_link_flow = total_link_flow + dial_res.link_flow
        if group_link_flow is not None:
            group_link_flow = group_link_flow.at[g].set(dial_res.link_flow)

    return AssignmentResult(
        theta=th,
        link_flow=total_link_flow,
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
    """
    Convenience wrapper: prepare artifacts and assign in one call.

    This is handy for small tests, but for calibration/inference you should
    call prepare_assignment(...) once and reuse the artifacts.

    :param scenario: Domain Scenario.
    :param od_values: OD-demand vector.
    :param config: AssignmentConfig.
    :param theta: Optional theta override.
    :param return_group_link_flows: Whether to return per-group flows.
    :return: AssignmentResult.
    """
    artifacts = prepare_assignment(scenario, config)
    return assign(
        od_values,
        artifacts,
        theta=theta,
        return_group_link_flows=return_group_link_flows,
    )