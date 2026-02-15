"""
JAX-compatible graph data structures for the differentiable assignment model.

This module defines *static*, array-based representations of the time-expanded
public transport network. These structures are designed to be:

- compatible with JAX (jax.numpy arrays only),
- immutable during assignment,
- efficient for jit-compilation and vectorized evaluation.

The graph is link-indexed. All computations during assignment are performed
using arrays indexed by links or nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp


Array = jnp.ndarray


# ============================================================================
# Core graph structure
# ============================================================================

@dataclass(frozen=True, slots=True)
class JaxGraph:
    """
    Static JAX-compatible representation of the time-expanded network.

    Attributes
    ----------
    num_nodes : int
        Total number of nodes (centroids + event nodes).

    num_links : int
        Total number of directed links.

    tail : Array[int] shape (num_links,)
        Tail node index of each link.

    head : Array[int] shape (num_links,)
        Head node index of each link.

    topo_order : Array[int] shape (num_nodes,)
        Nodes sorted in topological order (time increasing).

    topo_order_rev : Array[int] shape (num_nodes,)
        Reverse topological order (for backward dynamic programming).

    out_start : Array[int] shape (num_nodes + 1,)
        CSR pointer to outgoing links.

    out_links : Array[int] shape (num_links,)
        Concatenation of outgoing link indices.

    link_type : Array[int] shape (num_links,)
        Integer code describing link type:
            0 = ride
            1 = transfer
            2 = access

    travel_time : Array[float] shape (num_links,)
        Physical time on the link (minutes).
        Used to compute generalized cost.

    capacity : Array[float] shape (num_links,)
        Capacity of ride links.
        For non-ride links, set to +inf.
    """

    num_nodes: int
    num_links: int

    tail: Array
    head: Array

    topo_order: Array
    topo_order_rev: Array

    out_start: Array
    out_links: Array

    link_type: Array
    travel_time: Array
    capacity: Array


# ============================================================================
# OD demand representation (JAX side)
# ============================================================================

@dataclass(frozen=True, slots=True)
class JaxOD:
    """
    OD demand representation compatible with JAX.

    This is the *parameter vector* of the inference procedure.

    Attributes
    ----------
    origin_node : Array[int] shape (num_od,)
        Index of centroid node where demand is injected.

    dest_node : Array[int] shape (num_od,)
        Destination centroid node.

    desired_a : Array[float] shape (num_od,)
        Lower bound of desired departure interval.

    desired_b : Array[float] shape (num_od,)
        Upper bound of desired departure interval.
    """

    origin_node: Array
    dest_node: Array
    desired_a: Array
    desired_b: Array


# ============================================================================
# Reference flows for capacity penalty
# ============================================================================

@dataclass(frozen=True, slots=True)
class ReferenceFlows:
    """
    Reference flows used for lagged capacity penalties.

    These flows remain FIXED during one evaluation of the assignment.

    Attributes
    ----------
    flow : Array[float] shape (num_links,)
        Reference flow per link.
    """

    flow: Array