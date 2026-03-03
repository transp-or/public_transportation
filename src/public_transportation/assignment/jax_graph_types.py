"""JAX-compatible graph data structures for the differentiable assignment model.

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

import jax
import jax.numpy as jnp

# (No unused constants from .graph_sentinels are imported here.)

Array = jnp.ndarray


# ============================================================================
# Core graph structure
# ============================================================================


@dataclass(frozen=True, slots=True)
class JaxGraph:
    r"""Static JAX-compatible representation of the time-expanded network.

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
        Nodes sorted in topological order. The builder enforces a strict
        lexicographic order on (node_time, node_kind).
        Under the centroid policy used by the builder:
        - centroid-in nodes have node_time = CENTROID_IN_TIME_MIN and appear first,
        - event nodes are ordered by their scheduled times,
        - centroid-out nodes have node_time = CENTROID_OUT_TIME_MIN and appear last.
        The time-bin index is not encoded in node_time/node_time_s; it is represented by the centroid-in node identity and OD grouping.

    topo_order_rev : Array[int] shape (num_nodes,)
        Reverse topological order (for backward dynamic programming).

    node_time : Array[float] shape (num_nodes,)
        Time stamp of each node in minutes.
        - centroid-in nodes: CENTROID_IN_TIME_MIN (conceptual -\infty), never time-tagged.
        - event-arrival nodes: scheduled arrival time in minutes.
        - event-departure nodes: scheduled departure time in minutes.
        - centroid-out nodes: CENTROID_OUT_TIME_MIN (conceptual +\infty) so they
          appear after all event nodes.

    node_stop_index : Array[int] shape (num_nodes,)
        Index of the associated stop for each node (centroids and events),
        following the builder's sorted stop id convention.

    node_time_s : Array[int] shape (num_nodes,)
        Node time stamp in seconds-from-midnight.
        - event-arrival nodes: scheduled arrival time.
        - event-departure nodes: scheduled departure time.
        - centroid nodes (in/out): set to a sentinel value and must not be interpreted as a time-bin index.
          (Event nodes keep their scheduled seconds-from-midnight.)

    node_kind : Array[int] shape (num_nodes,)
        NODE_KIND_CENTROID_IN = centroid_in, NODE_KIND_EVENT_ARR = event_arr, NODE_KIND_EVENT_DEP = event_dep, NODE_KIND_CENTROID_OUT = centroid_out.

    node_trip_index : Array[int] shape (num_nodes,)
        Trip index associated with each node (event nodes only; centroids = -1).

    node_time_bin_index : Array[int] shape (num_nodes,)
        Time-bin index associated with each node.
        - centroid-in nodes: the time-bin index they represent.
        - all other nodes: -1.

    # Optional: only centroid-in nodes carry valid departure-interval bounds (minutes);
    # all other nodes use NaN. Kept optional for backward compatibility.
    node_bin_start_min: Array | None = None
    node_bin_end_min: Array | None = None

    out_start : Array[int] shape (num_nodes + 1,)
        CSR pointer to outgoing links.

    out_links_csr : Array[int] shape (num_links,)
        Concatenation of outgoing link indices in CSR format.

    out_links : Array[int] shape (num_nodes, max_out_degree)
        Padded adjacency list of outgoing links.

    out_mask : Array[bool] shape (num_nodes, max_out_degree)
        Mask indicating valid entries in `out_links`.

    link_type : Array[int] shape (num_links,)
        Integer code describing link type:
            - LINK_TYPE_RIDE = ride (event_dep -> event_arr)
            - LINK_TYPE_TRANSFER = transfer (event_arr -> event_dep, inter-line)
            - LINK_TYPE_ACCESS = access (centroid_in -> event_dep)
            - LINK_TYPE_EGRESS = egress (event_arr -> centroid_out)
            - LINK_TYPE_DWELL = dwell / continue (event_arr -> event_dep, same trip at same stop)

    travel_time : Array[float] shape (num_links,)
        Physical time on the link (minutes). Used to compute generalized cost.

    capacity : Array[float] shape (num_links,)
        Capacity of ride links. For non-ride links, set to +inf.

    link_trip_index : Array[int] shape (num_links,)
        Trip index for ride links (index into `scenario.timetable.trips` in the builder);
        for non-ride links set to -1.
        May also be set for access and egress links if desired by the builder,
        but remains -1 for pure transfer and dwell links unless the builder chooses otherwise.

    node_stop_id : tuple[str, ...]
        Python-side stop ids corresponding to `node_stop_index` mapping.
        Optional metadata for human-readable printing.

    node_stop_name : tuple[str, ...]
        Python-side stop names corresponding to `node_stop_id`/`node_stop_index` mapping.
        Optional metadata for human-readable printing.

    trip_id : tuple[str, ...]
        Python-side trip ids corresponding to `link_trip_index` mapping.
        Optional metadata for human-readable printing.

    trip_line_ref : tuple[str, ...]
        Line reference for each trip index (aligned with `trip_id`).
        Line identifiers are expected to be provided (builder enforces non-empty IDs).
    """

    num_nodes: int
    num_links: int

    tail: Array
    head: Array

    topo_order: Array
    topo_order_rev: Array
    node_time: Array
    node_stop_index: Array
    node_time_s: Array
    node_kind: Array
    node_trip_index: Array

    out_start: Array
    out_links_csr: Array
    out_links: Array
    out_mask: Array

    link_type: Array
    travel_time: Array
    capacity: Array
    link_trip_index: Array

    # Optional: only centroid-in nodes carry a valid time-bin index; all other nodes use -1.
    # Kept optional for backward compatibility with graphs built before this field existed.
    node_time_bin_index: Array | None = None

    # Optional: only centroid-in nodes carry valid departure-interval bounds (minutes);
    # all other nodes use NaN. Kept optional for backward compatibility.
    node_bin_start_min: Array | None = None
    node_bin_end_min: Array | None = None

    node_stop_id: tuple[str, ...] = ()
    node_stop_name: tuple[str, ...] = ()
    trip_id: tuple[str, ...] = ()
    trip_line_ref: tuple[str, ...] = ()


# ============================================================================
# OD demand representation (JAX side)
# ============================================================================


@dataclass(frozen=True, slots=True)
class JaxOD:
    r"""OD demand representation compatible with JAX.

    This is the *parameter vector* of the inference procedure.

    Attributes
    ----------
    origin_node : Array[int] shape (num_od,)
        Index of centroid-in node where demand is injected.

    dest_node : Array[int] shape (num_od,)
        Destination centroid-out node.

    desired_time : Array[float] shape (num_od,)
        Desired departure time \(\tau^\star_{q,t}\) in minutes. Used to compute access schedule-deviation penalties; independent of centroid node timestamps.
    """

    origin_node: Array
    dest_node: Array
    desired_time: Array


# ============================================================================
# Reference flows for capacity penalty
# ============================================================================


@dataclass(frozen=True, slots=True)
class ReferenceFlows:
    """Reference flows used for lagged capacity penalties.

    These flows remain FIXED during one evaluation of the assignment.

    Attributes
    ----------
    flow : Array[float] shape (num_links,)
        Reference flow per link.
    """

    flow: Array


# ==========================================================================
# JAX pytree registration
# ==========================================================================


def _jaxgraph_flatten(g: JaxGraph):
    tbi = g.node_time_bin_index
    if tbi is None:
        tbi = jnp.asarray([], dtype=jnp.int32)
    bstart = g.node_bin_start_min
    if bstart is None:
        bstart = jnp.asarray([], dtype=jnp.float32)
    bend = g.node_bin_end_min
    if bend is None:
        bend = jnp.asarray([], dtype=jnp.float32)
    children = (
        g.tail,
        g.head,
        g.topo_order,
        g.topo_order_rev,
        g.node_time,
        g.node_stop_index,
        g.node_time_s,
        g.node_kind,
        g.node_trip_index,
        tbi,
        bstart,
        bend,
        g.out_start,
        g.out_links_csr,
        g.out_links,
        g.out_mask,
        g.link_type,
        g.travel_time,
        g.capacity,
        g.link_trip_index,
    )
    has_tbi = g.node_time_bin_index is not None
    has_bstart = g.node_bin_start_min is not None
    has_bend = g.node_bin_end_min is not None
    aux = (
        g.num_nodes,
        g.num_links,
        g.node_stop_id,
        g.node_stop_name,
        g.trip_id,
        g.trip_line_ref,
        has_tbi,
        has_bstart,
        has_bend,
    )
    return children, aux


def _jaxgraph_unflatten(aux, children):
    (num_nodes, num_links, node_stop_id, node_stop_name, trip_id, trip_line_ref, has_tbi, has_bstart, has_bend) = aux
    (
        tail,
        head,
        topo_order,
        topo_order_rev,
        node_time,
        node_stop_index,
        node_time_s,
        node_kind,
        node_trip_index,
        node_time_bin_index,
        node_bin_start_min,
        node_bin_end_min,
        out_start,
        out_links_csr,
        out_links,
        out_mask,
        link_type,
        travel_time,
        capacity,
        link_trip_index,
    ) = children
    if not has_tbi:
        node_time_bin_index = None
    if not has_bstart:
        node_bin_start_min = None
    if not has_bend:
        node_bin_end_min = None
    return JaxGraph(
        num_nodes=int(num_nodes),
        num_links=int(num_links),
        tail=tail,
        head=head,
        topo_order=topo_order,
        topo_order_rev=topo_order_rev,
        node_time=node_time,
        node_stop_index=node_stop_index,
        node_time_s=node_time_s,
        node_kind=node_kind,
        node_trip_index=node_trip_index,
        node_time_bin_index=node_time_bin_index,
        node_bin_start_min=node_bin_start_min,
        node_bin_end_min=node_bin_end_min,
        out_start=out_start,
        out_links_csr=out_links_csr,
        out_links=out_links,
        out_mask=out_mask,
        link_type=link_type,
        travel_time=travel_time,
        capacity=capacity,
        link_trip_index=link_trip_index,
        node_stop_id=node_stop_id,
        node_stop_name=node_stop_name,
        trip_id=trip_id,
        trip_line_ref=trip_line_ref,
    )


def _jaxod_flatten(od: JaxOD):
    children = (od.origin_node, od.dest_node, od.desired_time)
    aux = None
    return children, aux


def _jaxod_unflatten(aux, children):
    (origin_node, dest_node, desired_time) = children
    return JaxOD(
        origin_node=origin_node,
        dest_node=dest_node,
        desired_time=desired_time,
    )


def _refflows_flatten(rf: ReferenceFlows):
    children = (rf.flow,)
    aux = None
    return children, aux


def _refflows_unflatten(aux, children):
    (flow,) = children
    return ReferenceFlows(flow=flow)


jax.tree_util.register_pytree_node(JaxGraph, _jaxgraph_flatten, _jaxgraph_unflatten)
jax.tree_util.register_pytree_node(JaxOD, _jaxod_flatten, _jaxod_unflatten)
jax.tree_util.register_pytree_node(ReferenceFlows, _refflows_flatten, _refflows_unflatten)