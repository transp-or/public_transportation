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
- Capacity effects can be represented via precomputed effective link costs; this module is agnostic to how costs are formed.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .jax_graph_types import JaxGraph

Array = jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class DialResult:
    """Output container for one Dial-style evaluation for a single destination group.

    Notes
    -----
    - This container is a JAX pytree so it can be returned from jitted code.
    - `dest_node` and `theta` are stored as JAX scalar arrays (not Python scalars)
      to avoid recompilation and to remain compatible with `lax.scan` / `vmap`.

    Fields
    ------
    dest_node : Array scalar (int32)
        Destination node index.
    theta : Array scalar (float)
        Dispersion parameter used (minutes).
    value : Array shape (num_nodes,)
        Value function V.
    link_prob : Array shape (num_links,)
        Transition probability per link.
    node_flow : Array shape (num_nodes,)
        Total flow reaching each node.
    link_flow : Array shape (num_links,)
        Flow on each link.
    """

    dest_node: Array
    theta: Array
    value: Array
    link_prob: Array
    node_flow: Array
    link_flow: Array

    # --- pytree protocol ---
    def tree_flatten(self):
        children = (self.dest_node, self.theta, self.value, self.link_prob, self.node_flow, self.link_flow)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        dest_node, theta, value, link_prob, node_flow, link_flow = children
        return cls(
            dest_node=dest_node,
            theta=theta,
            value=value,
            link_prob=link_prob,
            node_flow=node_flow,
            link_flow=link_flow,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class DestinationRouting:
    """Demand-independent routing state for one destination.

    The value function and link probabilities depend on the graph, costs,
    destination, enabled links, and dispersion, but not on OD demand. Keeping
    them in a separate PyTree allows fixed-dispersion inference to prepare this
    state once and reuse it for multiple flow-loading calls.
    """

    dest_node: Array
    theta: Array
    enabled_link_mask: Array
    value: Array
    link_prob: Array

    def tree_flatten(self):
        children = (
            self.dest_node,
            self.theta,
            self.enabled_link_mask,
            self.value,
            self.link_prob,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        dest_node, theta, enabled_link_mask, value, link_prob = children
        return cls(
            dest_node=dest_node,
            theta=theta,
            enabled_link_mask=enabled_link_mask,
            value=value,
            link_prob=link_prob,
        )


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
    enabled_link_mask: Array,
    dest_node: int,
    theta: float,
) -> Array:
    """
    Compute the value function V for a fixed destination node using backward DP.

    The recursion is evaluated in reverse topological order.

    :param graph: JAX time-expanded graph (DAG).
    :param link_cost: Effective generalized cost per link (minutes), shape (num_links,).
    :param enabled_link_mask: Boolean mask per link, shape (num_links,), indicating enabled links for the group.
    :param dest_node: Destination node index.
    :param theta: Logit dispersion parameter (minutes), must be > 0.
    :return: Value function V, shape (num_nodes,).
    """
    _require_padded_adjacency(graph)
    # Defensive: ensure boolean mask dtype (helps if callers pass 0/1 arrays).
    enabled_link_mask = enabled_link_mask.astype(jnp.bool_)
    num_nodes = graph.num_nodes
    topo = graph.topo_order  # shape (num_nodes,)
    # Numerical safety: protect against extremely small theta during AD/initialization.
    theta = jnp.maximum(jnp.asarray(theta, dtype=link_cost.dtype), jnp.asarray(1e-3, dtype=link_cost.dtype))

    # Costs are in minutes; a sentinel of 1e6 is already effectively unreachable and avoids huge-magnitude intermediates.
    large = jnp.asarray(1e6, dtype=link_cost.dtype)
    v0 = jnp.full((num_nodes,), large, dtype=link_cost.dtype)
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

            # Combine adjacency mask with per-group enabled-link mask.
            enabled = enabled_link_mask[safe_links]
            mask2 = jnp.logical_and(mask, enabled)

            v_head = v_in[head[safe_links]]    # (max_out,)
            c = link_cost[safe_links]          # (max_out,)

            logits = -(c + v_head) / theta
            logits = jnp.where(mask2, logits, -jnp.inf)

            # If a node has zero enabled outgoing links, logsumexp = -inf.
            # IMPORTANT for AD stability: `jnp.where` evaluates both branches, so it would still
            # compute (-theta * -inf) = +inf as an intermediate, which can poison gradients.
            # Use lax.cond so the non-taken branch is not evaluated.
            lse = logsumexp(logits)

            def _finite(_: Array) -> Array:
                return -theta * lse

            def _nonfinite(_: Array) -> Array:
                return large

            vi = jax.lax.cond(jnp.isfinite(lse), _finite, _nonfinite, operand=lse)
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
    enabled_link_mask: Array,
    value: Array,
    theta: float,
) -> Array:
    """
    Compute transition probabilities for each link given V.

    For each tail node i, and each outgoing link (i->j):
        P(i->j) = exp( -(c_ij + V(j))/theta ) / sum_{(i->k)} exp( -(c_ik + V(k))/theta )

    :param graph: JAX graph.
    :param link_cost: Link costs, shape (num_links,).
    :param enabled_link_mask: Boolean mask per link, shape (num_links,), indicating enabled links for the group.
    :param value: Value function V, shape (num_nodes,).
    :param theta: Dispersion parameter (minutes), must be > 0.
    :return: Probability per link, shape (num_links,).
    """
    _require_padded_adjacency(graph)
    # Defensive: ensure boolean mask dtype (helps if callers pass 0/1 arrays).
    enabled_link_mask = enabled_link_mask.astype(jnp.bool_)
    num_links = graph.num_links
    num_nodes = graph.num_nodes
    # Numerical safety: protect against extremely small theta during AD/initialization.
    theta = jnp.maximum(jnp.asarray(theta, dtype=link_cost.dtype), jnp.asarray(1e-3, dtype=link_cost.dtype))

    # Compute per-link "utility" u = -(c + V(head))/theta
    u = -(link_cost + value[graph.head]) / theta  # shape (num_links,)

    # We need log-denominator per node (tail). Use a scatter-based stable logsumexp over *all links*
    # to avoid any mismatch between `graph.out_links` and `graph.tail` that can otherwise yield
    # positive `logit` and exp overflow under AD.

    # Mask disabled links early.
    u = jnp.where(enabled_link_mask, u, -jnp.inf)
    u = jnp.nan_to_num(u, neginf=-1e6, posinf=1e6)
    u = jnp.clip(u, -1e3, 1e3)

    # Stable per-tail logsumexp(u) using max-shift.
    m = jnp.full((num_nodes,), -jnp.inf, dtype=u.dtype)
    m = m.at[graph.tail].max(u)

    # sum exp(u - m_tail); for u=-inf, exp(-inf)=0 so safe.
    s = jnp.zeros((num_nodes,), dtype=u.dtype)
    s = s.at[graph.tail].add(jnp.exp(u - m[graph.tail]))

    # log_denom(tail) = m_tail + log(s_tail); if no enabled outgoing links, keep -inf.
    log_denom = jnp.where(s > 0.0, m + jnp.log(s), -jnp.inf)

    # P(link) = exp(u_link - log_denom[tail]) for enabled links and finite denominators.
    # IMPORTANT for autodiff stability:
    # Do NOT compute exp(u - (-inf)) which yields +inf intermediates and can lead to NaN gradients
    # even if later masked out with `where`. Instead, mask in log-space before exponentiating.
    log_d = log_denom[graph.tail]

    # IMPORTANT: avoid computing (-inf) - (-inf) which produces NaN intermediates.
    logit = jnp.where(jnp.isfinite(log_d), u - log_d, -jnp.inf)

    # By construction logit should be <= 0 (since log_d is logsumexp over the same set),
    # but clamp for safety to prevent any exp overflow from numerical/masking edge cases.
    logit = jnp.minimum(logit, jnp.asarray(0.0, dtype=logit.dtype))

    # Force disabled links to probability 0 by masking in log-space.
    logit = jnp.where(enabled_link_mask, logit, -jnp.inf)

    logit = jnp.nan_to_num(logit, neginf=-jnp.inf, posinf=0.0)
    logit = jnp.clip(logit, -80.0, 0.0)  # exp(logit) in [exp(-80), 1]
    p_raw = jnp.exp(logit)


    # Numerical cleanup: clamp tiny negatives due to roundoff.
    p = jnp.maximum(p_raw, 0.0)

    # Any remaining non-finite values are set to 0.
    p = jnp.where(jnp.isfinite(p), p, 0.0)

    # Shape check at compile-time
    p = p.reshape((num_links,))

    return p


@jax.jit
def load_flows(
    graph: JaxGraph,
    link_prob: Array,
    enabled_link_mask: Array,
    initial_node_flow: Array,
) -> tuple[Array, Array]:
    """
    Forward loading pass to compute node and link flows.

    Given:
    - link_prob P_ell for each link,
    - enabled_link_mask: Boolean mask per link indicating enabled links for the group,
    - initial_node_flow y0 on nodes (typically demand injected at centroid nodes),

    We propagate in topological order:
      x_ell = y_tail * P_ell
      y_head += x_ell

    :param graph: JAX graph.
    :param link_prob: Link transition probabilities, shape (num_links,).
    :param enabled_link_mask: Boolean mask per link, shape (num_links,), indicating enabled links for the group.
    :param initial_node_flow: Initial node inflows, shape (num_nodes,).
    :return: (node_flow, link_flow)
        - node_flow: total flow reaching each node, shape (num_nodes,)
        - link_flow: flow on each link, shape (num_links,)
    """
    _require_padded_adjacency(graph)
    # Defensive: ensure boolean mask dtype (helps if callers pass 0/1 arrays).
    enabled_link_mask = enabled_link_mask.astype(jnp.bool_)

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

        enabled = enabled_link_mask[safe_links]
        mask2 = jnp.logical_and(mask, enabled)

        # outgoing link flows for node i
        yi = y[i]
        p = link_prob[safe_links]
        flow_links = yi * p
        flow_links = jnp.where(mask2, flow_links, 0.0)

        # accumulate link flows
        x = x.at[safe_links].add(flow_links)

        # accumulate to heads
        heads = head[safe_links]
        y = y.at[heads].add(flow_links)

        return (y, x), None

    (y, x), _ = jax.lax.scan(step, (y0, x0), jnp.arange(num_nodes, dtype=jnp.int32))
    return y, x


def _fixed_loading_initial_flow_adjoint(
    graph: JaxGraph,
    link_prob: Array,
    enabled_link_mask: Array,
    link_flow_cotangent: Array,
) -> Array:
    """Apply the exact fixed-routing adjoint to a link-flow cotangent."""
    enabled_link_mask = enabled_link_mask.astype(jnp.bool_)
    node_adjoint = jnp.zeros((graph.num_nodes,), dtype=link_flow_cotangent.dtype)

    def step(adjoint: Array, node_index: Array) -> tuple[Array, None]:
        node = graph.topo_order_rev[node_index]
        links = graph.out_links[node]
        mask = graph.out_mask[node]
        safe_links = jnp.where(mask, links, 0)
        active = jnp.logical_and(mask, enabled_link_mask[safe_links])
        heads = graph.head[safe_links]
        contribution = link_prob[safe_links] * (
            link_flow_cotangent[safe_links] + adjoint[heads]
        )
        value = jnp.sum(jnp.where(active, contribution, 0.0))
        return adjoint.at[node].add(value), None

    node_adjoint, _ = jax.lax.scan(
        step,
        node_adjoint,
        jnp.arange(graph.num_nodes, dtype=jnp.int32),
    )
    return node_adjoint


@jax.custom_vjp
def load_link_flow_fixed_custom_adjoint(
    graph: JaxGraph,
    link_prob: Array,
    enabled_link_mask: Array,
    initial_node_flow: Array,
) -> Array:
    """Load link flow with an explicit fixed-routing demand adjoint."""
    return load_flows(
        graph, link_prob, enabled_link_mask, initial_node_flow
    )[1]


def _load_link_flow_fixed_fwd(
    graph: JaxGraph,
    link_prob: Array,
    enabled_link_mask: Array,
    initial_node_flow: Array,
) -> tuple[Array, tuple[JaxGraph, Array, Array]]:
    link_flow = load_flows(
        graph, link_prob, enabled_link_mask, initial_node_flow
    )[1]
    return link_flow, (graph, link_prob, enabled_link_mask)


def _load_link_flow_fixed_bwd(
    residual: tuple[JaxGraph, Array, Array], link_flow_cotangent: Array
) -> tuple[None, None, None, Array]:
    graph, link_prob, enabled_link_mask = residual
    initial_adjoint = _fixed_loading_initial_flow_adjoint(
        graph, link_prob, enabled_link_mask, link_flow_cotangent
    )
    return None, None, None, initial_adjoint


load_link_flow_fixed_custom_adjoint.defvjp(
    _load_link_flow_fixed_fwd, _load_link_flow_fixed_bwd
)


def prepare_destination_routing(
    *,
    graph: JaxGraph,
    link_cost: Array,
    enabled_link_mask: Array,
    dest_node: int | Array,
    theta: float | Array,
) -> DestinationRouting:
    """Compute the demand-independent routing state for one destination."""
    _require_padded_adjacency(graph)
    if isinstance(theta, (int, float)) and float(theta) <= 0.0:
        raise ValueError("theta must be positive.")

    link_cost_j = jnp.asarray(link_cost)
    enabled_j = jnp.asarray(enabled_link_mask, dtype=jnp.bool_)
    dest_node_j = jnp.asarray(dest_node, dtype=jnp.int32)
    theta_j = jnp.asarray(theta, dtype=link_cost_j.dtype)
    value = compute_value_function(
        graph,
        link_cost_j,
        enabled_j,
        dest_node_j,
        theta_j,
    )
    link_prob = compute_link_probabilities(
        graph,
        link_cost_j,
        enabled_j,
        value,
        theta_j,
    )
    return DestinationRouting(
        dest_node=dest_node_j,
        theta=theta_j,
        enabled_link_mask=enabled_j,
        value=value,
        link_prob=link_prob,
    )


def load_destination_flows(
    *,
    graph: JaxGraph,
    routing: DestinationRouting,
    initial_node_flow: Array,
) -> tuple[Array, Array]:
    """Load demand using routing state prepared independently of demand."""
    return load_flows(
        graph,
        routing.link_prob,
        routing.enabled_link_mask,
        initial_node_flow,
    )


def run_dial_for_destination(
    *,
    graph: JaxGraph,
    link_cost: Array,
    enabled_link_mask: Array,
    dest_node: int | Array,
    theta: float | Array,
    initial_node_flow: Array,
) -> DialResult:
    """Convenience wrapper: compute V, probabilities, and flows for one destination group.

    This wrapper is safe to call from jitted code:
    - `dest_node` is treated as a dynamic argument (JAX int scalar).
    - `theta` is treated as a dynamic argument (JAX float scalar).

    Notes on validation
    -------------------
    - When called in eager mode with Python scalars, we validate that `theta > 0`.
    - When called under JIT with traced values, we cannot raise Python exceptions;
      callers must ensure `theta` is positive.

    Parameters
    ----------
    graph : JaxGraph
    link_cost : Array, shape (num_links,)
    enabled_link_mask : Array, shape (num_links,)
    dest_node : int or Array scalar
    theta : float or Array scalar
    initial_node_flow : Array, shape (num_nodes,)

    Returns
    -------
    DialResult
    """
    _require_padded_adjacency(graph)

    # Eager-mode validation only (do not attempt Python comparisons on traced values).
    if isinstance(theta, (int, float)) and float(theta) <= 0.0:
        raise ValueError("theta must be positive.")

    routing = prepare_destination_routing(
        graph=graph,
        link_cost=link_cost,
        enabled_link_mask=enabled_link_mask,
        dest_node=dest_node,
        theta=theta,
    )
    node_flow, link_flow = load_destination_flows(
        graph=graph,
        routing=routing,
        initial_node_flow=initial_node_flow,
    )

    return DialResult(
        dest_node=routing.dest_node,
        theta=routing.theta,
        value=routing.value,
        link_prob=routing.link_prob,
        node_flow=node_flow,
        link_flow=link_flow,
    )
