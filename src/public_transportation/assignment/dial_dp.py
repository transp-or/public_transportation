"""
Differentiable Dial-style dynamic programming and loading (JAX).

This module implements the two core steps for a *fixed* time-expanded DAG:

1) Backward pass (value function):
   For a given destination node d, compute V(i) via a logsumexp Bellman recursion:
       V(d) = 0
       V(i) = -theta * logsumexp_{(i->j)} ( -(c_ij + V(j))/theta )

2) Forward pass (loading):
   Given OD demand injected at centroid nodes (for a fixed destination + time-bin group),
   propagate flow through the network using transition probabilities derived from V.

Design constraints
------------------
- We target **fast JIT compilation** in JAX.
- Variable out-degrees must be handled without Python loops during evaluation.

Graph requirements (IMPORTANT)
------------------------------
This module expects the JaxGraph to contain a *padded* adjacency representation:

- graph.out_links : int32 array, shape (num_nodes, max_out_degree)
    For each node i, the row contains link indices of outgoing links, padded with -1.

- graph.out_mask : bool array, shape (num_nodes, max_out_degree)
    True for valid outgoing links, False for padding positions.

These arrays are produced by the builder (build_time_expanded.py).
If they are missing, this module raises a clear error.

Notes
-----
- All costs are in minutes.
- The only parameters estimated are OD demand values (and optionally theta).
- Capacity penalties are not included in the first implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .jax_graph_types import JaxGraph

Array = jnp.ndarray


@dataclass(frozen=True, slots=True)
class DialResult:
    """
    Output container for one Dial-style evaluation for a single destination group.

    :param dest_node: Destination node index.
    :param theta: Dispersion parameter used (minutes).
    :param value: Value function V, shape (num_nodes,).
    :param link_prob: Transition probability per link, shape (num_links,).
    :param node_flow: Total flow reaching each node, shape (num_nodes,).
    :param link_flow: Flow on each link, shape (num_links,).
    """
    dest_node: int
    theta: float
    value: Array
    link_prob: Array
    node_flow: Array
    link_flow: Array


def _require_padded_adjacency(graph: JaxGraph) -> None:
    """
    Check that the graph has the padded adjacency representation required for JIT.

    :param graph: JaxGraph instance.
    :raises ValueError: if required fields are missing.
    """
    if getattr(graph, "out_links", None) is None or getattr(graph, "out_mask", None) is None:
        raise ValueError(
            "JaxGraph must provide padded adjacency arrays `out_links` and `out_mask` "
            "(shape (num_nodes, max_out_degree)) for JIT-friendly evaluation."
        )


@jax.jit
def compute_value_function(
    graph: JaxGraph,
    link_cost: Array,
    dest_node: int,
    theta: float,
) -> Array:
    """
    Compute the value function V for a fixed destination node using backward DP.

    The recursion is evaluated in reverse topological order.

    :param graph: JAX time-expanded graph (DAG).
    :param link_cost: Effective generalized cost per link (minutes), shape (num_links,).
    :param dest_node: Destination node index.
    :param theta: Logit dispersion parameter (minutes), must be > 0.
    :return: Value function V, shape (num_nodes,).
    """
    _require_padded_adjacency(graph)
    if theta <= 0.0:
        raise ValueError("theta must be positive.")

    num_nodes = graph.num_nodes
    topo = graph.topo_order  # shape (num_nodes,)

    # Initialize with +inf everywhere, then set V(dest)=0.
    v0 = jnp.full((num_nodes,), jnp.inf, dtype=link_cost.dtype)
    v0 = v0.at[dest_node].set(0.0)

    out_links = graph.out_links  # (num_nodes, max_out)
    out_mask = graph.out_mask    # (num_nodes, max_out)
    head = graph.head            # (num_links,)

    def body(v: Array, idx: Array) -> Array:
        node = topo[idx]

        # Keep destination fixed at 0.
        def compute_for_node(v_in: Array) -> Array:
            links = out_links[node]            # (max_out,)
            mask = out_mask[node]              # (max_out,)
            # For padding links == -1, substitute 0 to keep gathers in-bounds;
            # they'll be masked out.
            safe_links = jnp.where(mask, links, 0)

            v_head = v_in[head[safe_links]]    # (max_out,)
            c = link_cost[safe_links]          # (max_out,)

            logits = -(c + v_head) / theta
            logits = jnp.where(mask, logits, -jnp.inf)

            vi = -theta * logsumexp(logits)
            return v_in.at[node].set(vi)

        v_out = jax.lax.cond(node == dest_node, lambda x: x, compute_for_node, v)
        return v_out

    # Reverse indices over topo: from num_nodes-1 down to 0
    idxs = jnp.arange(num_nodes - 1, -1, -1, dtype=jnp.int32)
    v = jax.lax.fori_loop(0, num_nodes, lambda k, vv: body(vv, idxs[k]), v0)

    return v


@jax.jit
def compute_link_probabilities(
    graph: JaxGraph,
    link_cost: Array,
    value: Array,
    theta: float,
) -> Array:
    """
    Compute transition probabilities for each link given V.

    For each tail node i, and each outgoing link (i->j):
        P(i->j) = exp( -(c_ij + V(j))/theta ) / sum_{(i->k)} exp( -(c_ik + V(k))/theta )

    :param graph: JAX graph.
    :param link_cost: Link costs, shape (num_links,).
    :param value: Value function V, shape (num_nodes,).
    :param theta: Dispersion parameter (minutes), must be > 0.
    :return: Probability per link, shape (num_links,).
    """
    _require_padded_adjacency(graph)
    if theta <= 0.0:
        raise ValueError("theta must be positive.")

    num_links = graph.num_links
    num_nodes = graph.num_nodes

    # Compute per-link "utility" u = -(c + V(head))/theta
    u = -(link_cost + value[graph.head]) / theta  # shape (num_links,)

    # We need log-denominator per node (tail), using padded adjacency + logsumexp.
    out_links = graph.out_links
    out_mask = graph.out_mask

    def denom_for_node(i: Array) -> Array:
        links = out_links[i]
        mask = out_mask[i]
        safe_links = jnp.where(mask, links, 0)
        ui = u[safe_links]
        ui = jnp.where(mask, ui, -jnp.inf)
        return logsumexp(ui)

    log_denom = jax.vmap(denom_for_node)(jnp.arange(num_nodes, dtype=jnp.int32))  # (num_nodes,)

    # P(link) = exp(u_link - log_denom[tail])
    p = jnp.exp(u - log_denom[graph.tail])

    # Numerical cleanup: ensure padding never contributes; also clamp tiny negatives due to roundoff.
    p = jnp.maximum(p, 0.0)

    # It is possible that some nodes have zero out-degree (log_denom=-inf), leading to nan.
    # Those links do not exist; set any non-finite to 0.
    p = jnp.where(jnp.isfinite(p), p, 0.0)

    # Shape check at compile-time
    p = p.reshape((num_links,))

    return p


@jax.jit
def load_flows(
    graph: JaxGraph,
    link_prob: Array,
    initial_node_flow: Array,
) -> tuple[Array, Array]:
    """
    Forward loading pass to compute node and link flows.

    Given:
    - link_prob P_ell for each link,
    - initial_node_flow y0 on nodes (typically demand injected at centroid nodes),

    We propagate in topological order:
      x_ell = y_tail * P_ell
      y_head += x_ell

    :param graph: JAX graph.
    :param link_prob: Link transition probabilities, shape (num_links,).
    :param initial_node_flow: Initial node inflows, shape (num_nodes,).
    :return: (node_flow, link_flow)
        - node_flow: total flow reaching each node, shape (num_nodes,)
        - link_flow: flow on each link, shape (num_links,)
    """
    _require_padded_adjacency(graph)

    num_nodes = graph.num_nodes
    num_links = graph.num_links

    topo = graph.topo_order
    out_links = graph.out_links
    out_mask = graph.out_mask
    head = graph.head

    y0 = initial_node_flow.reshape((num_nodes,))
    x0 = jnp.zeros((num_links,), dtype=y0.dtype)

    def step(carry: tuple[Array, Array], k: Array) -> tuple[tuple[Array, Array], None]:
        y, x = carry
        i = topo[k]

        links = out_links[i]             # (max_out,)
        mask = out_mask[i]               # (max_out,)
        safe_links = jnp.where(mask, links, 0)

        # outgoing link flows for node i
        yi = y[i]
        p = link_prob[safe_links]
        flow_links = yi * p
        flow_links = jnp.where(mask, flow_links, 0.0)

        # accumulate link flows
        x = x.at[safe_links].add(flow_links)

        # accumulate to heads
        heads = head[safe_links]
        y = y.at[heads].add(flow_links)

        return (y, x), None

    (y, x), _ = jax.lax.scan(step, (y0, x0), jnp.arange(num_nodes, dtype=jnp.int32))
    return y, x


def run_dial_for_destination(
    *,
    graph: JaxGraph,
    link_cost: Array,
    dest_node: int,
    theta: float,
    initial_node_flow: Array,
) -> DialResult:
    """
    Convenience wrapper: compute V, probabilities, and flows for one destination group.

    :param graph: JAX graph.
    :param link_cost: Link costs (minutes), shape (num_links,).
    :param dest_node: Destination node index.
    :param theta: Dispersion parameter (minutes), must be > 0.
    :param initial_node_flow: Node inflows (demand injected at origins), shape (num_nodes,).
    :return: DialResult.
    """
    v = compute_value_function(graph, link_cost, dest_node, theta)
    p = compute_link_probabilities(graph, link_cost, v, theta)
    node_flow, link_flow = load_flows(graph, p, initial_node_flow)
    return DialResult(
        dest_node=int(dest_node),
        theta=float(theta),
        value=v,
        link_prob=p,
        node_flow=node_flow,
        link_flow=link_flow,
    )