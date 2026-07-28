# src/public_transportation/inference/assignment_adapter.py
"""
Assignment adapter for inference (JAX-facing, user-hidden).

Goal
----
Hide all assignment-internal details (ODGroups arrays, base link cost computation,
and the private `_assign_core` call) behind a small, stable interface for the
inference forward model.

The main script should NOT import `_assign_core`, `link_costs`, or touch ODGroups.
Instead it should pass `artifacts` to the inference pipeline, which will call:

    inputs = build_assignment_inputs(artifacts)
    link_flow = assign_link_flow(inputs=inputs, f=f, theta=theta)

Design constraints
------------------
- JAX-traceable: the call path used inside VI must avoid Python-side data-dependent control flow.
- Keep graph/static objects out of traced arrays as much as possible.
- One responsibility per function.

Notes on JAX PyTrees
--------------------
`AssignmentInputs` is registered as a PyTree:
- children: the JAX arrays used by `_assign_core`
- aux/static: the graph object (and any other static, non-array state)

This lets JAX trace through computations while treating `graph` as static.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.assignment.assign import (
    _assign_core,
    _assign_fixed_routing_core,
    _assign_fixed_routing_custom_adjoint_core,
    _routing_inputs_for_destination,
)
from public_transportation.assignment.costs import link_costs
from public_transportation.assignment.dial_dp import prepare_destination_routing
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)
from public_transportation.inference.compact_od_groups import compact_od_groups


Array = jnp.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class _GraphIdentity:
    """Hashable identity wrapper for graph objects used as JAX PyTree metadata."""

    graph: Any

    def __hash__(self) -> int:
        return id(self.graph)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _GraphIdentity) and self.graph is other.graph


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class AssignmentInputs:
    """JAX-friendly inputs required to evaluate assignment link flows.

    Fields
    ------
    graph:
        Static assignment graph object (treated as auxiliary/static in PyTree).
        This is typically the object produced by `prepare_assignment(...).graph`.

    base_link_cost:
        Precomputed base link costs, shape (num_links,).

    group_dest_node, group_link_mask, od_origin_node, group_od_index_padded, group_od_mask:
        ODGroups arrays extracted from `artifacts.od_groups` and converted to JAX arrays.

    Notes
    -----
    - Keep this object immutable and build it once per scenario/artifacts.
    - Use inside inference; do not expose to users.
    """

    # Static (non-JAX) object
    graph: Any

    # JAX arrays
    base_link_cost: Array

    group_dest_node: Array
    group_link_mask: Array
    od_origin_node: Array
    group_od_index_padded: Array
    group_od_mask: Array

    def tree_flatten(self):
        children = (
            self.base_link_cost,
            self.group_dest_node,
            self.group_link_mask,
            self.od_origin_node,
            self.group_od_index_padded,
            self.group_od_mask,
        )
        aux = (_GraphIdentity(self.graph),)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (graph_identity,) = aux
        (
            base_link_cost,
            group_dest_node,
            group_link_mask,
            od_origin_node,
            group_od_index_padded,
            group_od_mask,
        ) = children
        return cls(
            graph=graph_identity.graph,
            base_link_cost=base_link_cost,
            group_dest_node=group_dest_node,
            group_link_mask=group_link_mask,
            od_origin_node=od_origin_node,
            group_od_index_padded=group_od_index_padded,
            group_od_mask=group_od_mask,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class FixedRoutingInputs:
    """Prepared routing probabilities for every active destination group.

    This immutable cache contains only demand-independent state needed by the
    future fixed-routing loading path. ``source_group_link_mask`` preserves the
    original mask so compatibility with ``AssignmentInputs`` can be checked
    before the cache is used.
    """

    theta: Array
    graph: Any
    source_base_link_cost: Array
    group_dest_node: Array
    source_group_link_mask: Array
    effective_group_link_mask: Array
    group_link_probability: Array
    num_nodes: int
    num_links: int

    def tree_flatten(self):
        children = (
            self.theta,
            self.source_base_link_cost,
            self.group_dest_node,
            self.source_group_link_mask,
            self.effective_group_link_mask,
            self.group_link_probability,
        )
        return children, (_GraphIdentity(self.graph), self.num_nodes, self.num_links)

    @classmethod
    def tree_unflatten(cls, aux, children):
        graph_identity, num_nodes, num_links = aux
        (
            theta,
            source_base_link_cost,
            group_dest_node,
            source_group_link_mask,
            effective_group_link_mask,
            group_link_probability,
        ) = children
        return cls(
            theta=theta,
            graph=graph_identity.graph,
            source_base_link_cost=source_base_link_cost,
            group_dest_node=group_dest_node,
            source_group_link_mask=source_group_link_mask,
            effective_group_link_mask=effective_group_link_mask,
            group_link_probability=group_link_probability,
            num_nodes=num_nodes,
            num_links=num_links,
        )
def build_base_link_cost(*, artifacts: Any, dtype: Any = jnp.float32) -> Array:
    """Compute base link costs once from assignment artifacts.

    Responsibility
    --------------
    Only computes the base link cost vector from (graph, cost_parts, config).

    This is kept separate so you can test/replace it independently.
    """
    return jnp.asarray(
        link_costs(
            graph=artifacts.graph,
            cost_parts=artifacts.cost_parts,
            config=artifacts.config,
        ),
        dtype=dtype,
    )


def build_assignment_inputs(
    *,
    artifacts: Any,
    compact_layout: CompactODAssignmentLayout | None = None,
) -> AssignmentInputs:
    """Extract and convert all assignment inputs needed for inference.

    Responsibility
    --------------
    - precompute base_link_cost
    - extract ODGroups arrays
    - convert all arrays to JAX arrays with stable dtypes
    - store `graph` as static state

    Parameters
    ----------
    artifacts:
        Result of `prepare_assignment(...)` (opaque bundle).

    Returns
    -------
    AssignmentInputs
        JAX-ready input bundle for `assign_link_flow`.
    """
    base_link_cost = build_base_link_cost(artifacts=artifacts, dtype=jnp.float32)

    odg = artifacts.od_groups
    if compact_layout is not None:
        odg = compact_od_groups(od_groups=odg, layout=compact_layout)
    # dtypes: indices int32, masks bool
    group_dest_node = jnp.asarray(odg.group_dest_node, dtype=jnp.int32)
    group_link_mask = jnp.asarray(odg.group_link_mask, dtype=bool)
    od_origin_node = jnp.asarray(odg.od_origin_node, dtype=jnp.int32)
    group_od_index_padded = jnp.asarray(odg.group_od_index_padded, dtype=jnp.int32)
    group_od_mask = jnp.asarray(odg.group_od_mask, dtype=bool)

    return AssignmentInputs(
        graph=artifacts.graph,
        base_link_cost=base_link_cost,
        group_dest_node=group_dest_node,
        group_link_mask=group_link_mask,
        od_origin_node=od_origin_node,
        group_od_index_padded=group_od_index_padded,
        group_od_mask=group_od_mask,
    )


@jax.jit
def _prepare_fixed_routing_core(
    *,
    inputs: AssignmentInputs,
    theta: Array,
) -> tuple[Array, Array]:
    """JAX core preparing effective masks and probabilities for all groups."""
    num_groups = inputs.group_dest_node.shape[0]

    def step(_, group_index):
        dest_node = inputs.group_dest_node[group_index]
        enabled, cost = _routing_inputs_for_destination(
            graph=inputs.graph,
            base_link_cost=inputs.base_link_cost,
            group_link_mask=inputs.group_link_mask[group_index],
            dest_node=dest_node,
        )
        routing = prepare_destination_routing(
            graph=inputs.graph,
            link_cost=cost,
            enabled_link_mask=enabled,
            dest_node=dest_node,
            theta=theta,
        )
        return None, (enabled, routing.link_prob)

    _, prepared = jax.lax.scan(step, None, jnp.arange(num_groups, dtype=jnp.int32))
    return prepared


def prepare_fixed_routing(
    *,
    inputs: AssignmentInputs,
    theta: float,
) -> FixedRoutingInputs:
    """Prepare reusable routing for fixed positive dispersion.

    The returned cache is not yet selected by inference; Phase 3 introduces
    the corresponding demand-loading path.
    """
    theta_value = float(theta)
    if not np.isfinite(theta_value) or theta_value <= 0.0:
        raise ValueError("theta must be positive and finite.")

    num_groups = int(inputs.group_dest_node.shape[0])
    num_links = int(inputs.graph.num_links)
    dtype = inputs.base_link_cost.dtype
    theta_array = jnp.asarray(theta_value, dtype=dtype).reshape(())
    if num_groups == 0:
        effective_masks = jnp.empty((0, num_links), dtype=bool)
        probabilities = jnp.empty((0, num_links), dtype=dtype)
    else:
        effective_masks, probabilities = _prepare_fixed_routing_core(
            inputs=inputs,
            theta=theta_array,
        )

    return FixedRoutingInputs(
        theta=theta_array,
        graph=inputs.graph,
        source_base_link_cost=inputs.base_link_cost,
        group_dest_node=inputs.group_dest_node,
        source_group_link_mask=inputs.group_link_mask,
        effective_group_link_mask=effective_masks,
        group_link_probability=probabilities,
        num_nodes=int(inputs.graph.num_nodes),
        num_links=num_links,
    )


def validate_fixed_routing_compatibility(
    *,
    inputs: AssignmentInputs,
    routing: FixedRoutingInputs,
) -> None:
    """Reject a cache prepared from different routing-sensitive inputs."""
    if routing.graph is not inputs.graph:
        raise ValueError("Fixed routing was prepared for a different graph.")
    if routing.num_nodes != int(inputs.graph.num_nodes) or routing.num_links != int(
        inputs.graph.num_links
    ):
        raise ValueError("Fixed routing graph dimensions do not match assignment inputs.")
    num_groups = int(inputs.group_dest_node.shape[0])
    expected_group_link_shape = (num_groups, int(inputs.graph.num_links))
    if routing.effective_group_link_mask.shape != expected_group_link_shape:
        raise ValueError(
            "Fixed routing effective masks must have shape "
            f"{expected_group_link_shape}, got {routing.effective_group_link_mask.shape}."
        )
    if routing.group_link_probability.shape != expected_group_link_shape:
        raise ValueError(
            "Fixed routing probabilities must have shape "
            f"{expected_group_link_shape}, got {routing.group_link_probability.shape}."
        )
    comparisons = (
        ("base link costs", routing.source_base_link_cost, inputs.base_link_cost),
        ("destination groups", routing.group_dest_node, inputs.group_dest_node),
        ("group link masks", routing.source_group_link_mask, inputs.group_link_mask),
    )
    for label, cached, current in comparisons:
        if not np.array_equal(np.asarray(cached), np.asarray(current)):
            raise ValueError(f"Fixed routing {label} do not match assignment inputs.")


def assign_link_flow(*, inputs: AssignmentInputs, f: Array, theta: Array) -> Array:
    """Compute link flows from assignment demand and dispersion ``theta``.

    Responsibility
    --------------
    Single responsibility: call the assignment core and return `link_flow`.

    Parameters
    ----------
    inputs:
        AssignmentInputs built by `build_assignment_inputs(...)`.
    f:
        Demand aligned with ``inputs.od_origin_node``. This is either the full
        OD vector or the compact vector of free and positive-frozen cells. The
        historical argument name ``f`` is retained for API compatibility.
    theta:
        Positive scalar dispersion parameter (minutes), shape ().

    Returns
    -------
    link_flow:
        Vector of link flows, shape (num_links,).

    Notes
    -----
    - We keep `return_group_link_flows=False` for inference.
    - The second return value from `_assign_core` (group flows) is ignored.
    """
    demand = jnp.asarray(f)
    expected_shape = inputs.od_origin_node.shape
    if demand.ndim != 1 or demand.shape != expected_shape:
        raise ValueError(f"f must have shape {expected_shape}, got {demand.shape}.")
    if inputs.group_dest_node.shape[0] == 0:
        return jnp.zeros((inputs.graph.num_links,), dtype=demand.dtype)

    link_flow, _ = _assign_core(
        graph=inputs.graph,
        od_values=demand,
        base_link_cost=inputs.base_link_cost,
        theta=jnp.asarray(theta).reshape(()),
        group_dest_node=inputs.group_dest_node,
        group_link_mask=inputs.group_link_mask,
        od_origin_node=inputs.od_origin_node,
        group_od_index_padded=inputs.group_od_index_padded,
        group_od_mask=inputs.group_od_mask,
        return_group_link_flows=False,
    )
    return link_flow


def assign_link_flow_fixed_routing(
    *,
    inputs: AssignmentInputs,
    routing: FixedRoutingInputs,
    f: Array,
) -> Array:
    """Compute link flows using routing prepared for fixed dispersion."""
    validate_fixed_routing_compatibility(inputs=inputs, routing=routing)
    demand = jnp.asarray(f)
    expected_shape = inputs.od_origin_node.shape
    if demand.ndim != 1 or demand.shape != expected_shape:
        raise ValueError(f"f must have shape {expected_shape}, got {demand.shape}.")
    if inputs.group_dest_node.shape[0] == 0:
        return jnp.zeros((inputs.graph.num_links,), dtype=demand.dtype)

    return _assign_fixed_routing_core(
        graph=inputs.graph,
        od_values=demand,
        effective_group_link_mask=routing.effective_group_link_mask,
        group_link_probability=routing.group_link_probability,
        od_origin_node=inputs.od_origin_node,
        group_od_index_padded=inputs.group_od_index_padded,
        group_od_mask=inputs.group_od_mask,
    )


def assign_link_flow_fixed_routing_custom_adjoint(
    *, inputs: AssignmentInputs, routing: FixedRoutingInputs, f: Array
) -> Array:
    """Load fixed routing with an explicit node-sized demand adjoint."""
    validate_fixed_routing_compatibility(inputs=inputs, routing=routing)
    demand = jnp.asarray(f)
    expected_shape = inputs.od_origin_node.shape
    if demand.ndim != 1 or demand.shape != expected_shape:
        raise ValueError(f"f must have shape {expected_shape}, got {demand.shape}.")
    if inputs.group_dest_node.shape[0] == 0:
        return jnp.zeros((inputs.graph.num_links,), dtype=demand.dtype)
    return _assign_fixed_routing_custom_adjoint_core(
        graph=inputs.graph,
        od_values=demand,
        effective_group_link_mask=routing.effective_group_link_mask,
        group_link_probability=routing.group_link_probability,
        od_origin_node=inputs.od_origin_node,
        group_od_index_padded=inputs.group_od_index_padded,
        group_od_mask=inputs.group_od_mask,
    )
