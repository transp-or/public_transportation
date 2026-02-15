"""
Build a JAX-compatible time-expanded graph from a domain-layer Scenario.

This module converts the flexible, user-facing domain representation
(stops, timetable, demand) into a static, array-based representation that
is suitable for fast JAX evaluation.

Design principles
-----------------
- Build once, evaluate many times (inside VI / gradients).
- All graph connectivity is stored as immutable arrays.
- The time-expanded graph is acyclic by construction (time increases along links).
- Link costs are NOT fully finalized here: this builder stores base physical
  durations (minutes) and link types. Generalized costs are computed later
  from these features and the user configuration.

Current scope (first implementation)
------------------------------------
- Centroid nodes: one per stop.
- Event nodes: one per departure event in stop_times (departure time at stop).
- Ride links: between consecutive stops within a trip: (stop_i, dep_i) -> (stop_{i+1}, dep_{i+1})
  using travel time computed from departure times.
  (We keep this simple and consistent with acyclicity: departure times strictly increase.)
- Transfer links: between consecutive events at the same stop (sorted by time).
- Access links: from centroid(stop) -> each event(stop, time), with zero physical duration.
  (Early/late penalties are computed later using desired time window and config.)

Notes
-----
- This code assumes the domain timetable provides departure times (seconds from midnight)
  and stop sequence. It will fail fast if inconsistencies are detected.
- All times are converted to minutes (float).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, TYPE_CHECKING

import numpy as np
import jax.numpy as jnp

from .config import AssignmentConfig
from .jax_graph_types import JaxGraph

if TYPE_CHECKING:  # pragma: no cover
    from public_transportation.domain import Scenario
    from public_transportation.domain import StopTime, Trip


# -----------------------------
# Helpers
# -----------------------------

def _to_minutes(seconds: float | int) -> float:
    """
    Convert seconds to minutes.

    :param seconds: Time in seconds.
    :return: Time in minutes.
    """
    return float(seconds) / 60.0


def _require(cond: bool, msg: str) -> None:
    """
    Raise a ValueError if condition is false.

    :param cond: Condition to check.
    :param msg: Error message.
    :raises ValueError: if cond is False.
    """
    if not cond:
        raise ValueError(msg)


def _sorted_stop_times_for_trip(stop_times: Iterable[Any]) -> list[Any]:
    """
    Sort stop_times of a trip by stop_sequence, then by departure_time.

    :param stop_times: StopTime-like objects with stop_sequence and departure_time.
    :return: Sorted list.
    """
    sts = list(stop_times)
    sts.sort(key=lambda st: (int(getattr(st, "stop_sequence")), float(getattr(st, "departure_time"))))
    return sts


# -----------------------------
# Core builder
# -----------------------------

@dataclass(slots=True)
class _NodeIndex:
    """
    Internal node indexing helpers.

    :param centroid_index: Mapping stop_id -> centroid node index.
    :param event_index: Mapping (stop_id, dep_time_seconds) -> event node index.
    :param node_time_min: Array of node times (minutes) used to define topological order.
    """
    centroid_index: dict[str, int]
    event_index: dict[tuple[str, int], int]
    node_time_min: np.ndarray


def _build_nodes(scenario: "Scenario") -> _NodeIndex:
    """
    Build centroid and event nodes.

    Centroid nodes: one per stop (time = -inf surrogate).
    Event nodes: one per (stop_id, departure_time_seconds) present in stop_times.

    :param scenario: Domain scenario.
    :return: Node indexing structure.
    """
    _require(scenario.timetable is not None, "Scenario has no timetable.")

    # Stops: accept dict or list-like containers
    if isinstance(scenario.stops, dict):
        stop_ids = list(scenario.stops.keys())
    else:
        stop_ids = [getattr(s, "stop_id", getattr(s, "id")) for s in scenario.stops]

    stop_ids_sorted = sorted(stop_ids)

    centroid_index: dict[str, int] = {}
    node_time_min: list[float] = []

    # Centroids first
    for k, sid in enumerate(stop_ids_sorted):
        centroid_index[sid] = k
        node_time_min.append(-1e12)  # "minus infinity" surrogate

    next_node = len(stop_ids_sorted)

    # Event nodes
    event_index: dict[tuple[str, int], int] = {}

    for st in scenario.timetable.stop_times:
        stop_id = str(getattr(st, "stop_id"))
        dep_sec = int(getattr(st, "departure_time"))
        key = (stop_id, dep_sec)
        if key not in event_index:
            event_index[key] = next_node
            next_node += 1
            node_time_min.append(_to_minutes(dep_sec))

    return _NodeIndex(
        centroid_index=centroid_index,
        event_index=event_index,
        node_time_min=np.asarray(node_time_min, dtype=float),
    )


def _build_links(
    scenario: "Scenario",
    nodes: _NodeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build link arrays (tail, head, link_type, travel_time_min, capacity).

    Link type codes:
      0 = ride
      1 = transfer
      2 = access

    :param scenario: Domain scenario.
    :param nodes: Node index structure.
    :return: (tail, head, link_type, travel_time_min, capacity)
    """
    _require(scenario.timetable is not None, "Scenario has no timetable.")

    tail: list[int] = []
    head: list[int] = []
    link_type: list[int] = []
    travel_time_min: list[float] = []
    capacity: list[float] = []

    # --- Access links: centroid(stop) -> event(stop, dep_time) with zero physical time
    for (stop_id, dep_sec), ev_node in nodes.event_index.items():
        c_node = nodes.centroid_index.get(stop_id)
        _require(c_node is not None, f"Unknown stop_id in stop_times: {stop_id}")
        tail.append(int(c_node))
        head.append(int(ev_node))
        link_type.append(2)
        travel_time_min.append(0.0)
        capacity.append(np.inf)

    # --- Ride links: between consecutive stop_times within each trip
    # Group stop_times by trip_id
    st_by_trip: dict[str, list[Any]] = {}
    for st in scenario.timetable.stop_times:
        trip_id = str(getattr(st, "trip_id"))
        st_by_trip.setdefault(trip_id, []).append(st)

    # Map trip_id -> capacity if present (optional)
    cap_by_trip: dict[str, float] = {}
    for tr in scenario.timetable.trips:
        tid = str(getattr(tr, "trip_id"))
        cap = getattr(tr, "capacity", None)
        cap_by_trip[tid] = float(cap) if cap is not None else np.inf

    for trip_id, sts in st_by_trip.items():
        sts_sorted = _sorted_stop_times_for_trip(sts)

        for a, b in zip(sts_sorted[:-1], sts_sorted[1:]):
            stop_a = str(getattr(a, "stop_id"))
            stop_b = str(getattr(b, "stop_id"))
            dep_a = int(getattr(a, "departure_time"))
            dep_b = int(getattr(b, "departure_time"))

            # Acyclicity + positive duration check (strictly increasing)
            _require(
                dep_b > dep_a,
                f"Non-increasing departure times in trip {trip_id}: {stop_a}@{dep_a} -> {stop_b}@{dep_b}",
            )

            na = nodes.event_index[(stop_a, dep_a)]
            nb = nodes.event_index[(stop_b, dep_b)]

            tail.append(int(na))
            head.append(int(nb))
            link_type.append(0)
            travel_time_min.append(_to_minutes(dep_b - dep_a))
            capacity.append(float(cap_by_trip.get(trip_id, np.inf)))

    # --- Transfer links: connect consecutive events at same stop
    # For each stop, sort events by time and connect in sequence
    events_by_stop: dict[str, list[int]] = {}
    times_by_stop: dict[str, list[int]] = {}
    for (stop_id, dep_sec), ev_node in nodes.event_index.items():
        events_by_stop.setdefault(stop_id, []).append(int(ev_node))
        times_by_stop.setdefault(stop_id, []).append(int(dep_sec))

    for stop_id, ev_nodes in events_by_stop.items():
        times = times_by_stop[stop_id]
        order = np.argsort(np.asarray(times, dtype=int))
        ev_sorted = [ev_nodes[i] for i in order]
        t_sorted = [times[i] for i in order]

        for (n1, t1), (n2, t2) in zip(zip(ev_sorted[:-1], t_sorted[:-1]), zip(ev_sorted[1:], t_sorted[1:])):
            if t2 == t1:
                continue  # skip zero-time transfers
            # Transfer always forward in time
            _require(t2 > t1, "Internal error: transfer times not sorted increasing.")
            tail.append(int(n1))
            head.append(int(n2))
            link_type.append(1)
            travel_time_min.append(_to_minutes(t2 - t1))
            capacity.append(np.inf)

    return (
        np.asarray(tail, dtype=int),
        np.asarray(head, dtype=int),
        np.asarray(link_type, dtype=int),
        np.asarray(travel_time_min, dtype=float),
        np.asarray(capacity, dtype=float),
    )


def _build_csr_outgoing(num_nodes: int, tail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build CSR-like outgoing adjacency from tail indices.

    :param num_nodes: Number of nodes.
    :param tail: Tail node index for each link, shape (num_links,).
    :return: (out_start, out_links) arrays.
    """
    num_links = int(tail.shape[0])
    order = np.argsort(tail, kind="mergesort")
    out_links = order.astype(int)

    # Count outgoing degree
    deg = np.zeros(num_nodes, dtype=int)
    np.add.at(deg, tail, 1)

    out_start = np.zeros(num_nodes + 1, dtype=int)
    out_start[1:] = np.cumsum(deg)

    return out_start, out_links


def build_jax_graph(
    scenario: "Scenario",
    *,
    config: AssignmentConfig,
) -> JaxGraph:
    """
    Build a static, JAX-compatible time-expanded graph from a validated Scenario.

    :param scenario: Domain Scenario containing stops, timetable, etc.
    :param config: Assignment configuration (validated here for convenience).
    :return: JaxGraph with immutable arrays.
    """
    config.validate()

    nodes = _build_nodes(scenario)
    tail, head, link_type, travel_time_min, capacity = _build_links(scenario, nodes)

    num_nodes = int(nodes.node_time_min.shape[0])
    num_links = int(tail.shape[0])

    _require(num_links > 0, "Built graph has no links. Check timetable/stop_times inputs.")
    _require(num_nodes > 0, "Built graph has no nodes.")

    # Topological order derived from node times (centroids first due to -inf surrogate)
    topo_order = np.argsort(nodes.node_time_min, kind="mergesort").astype(int)
    topo_order_rev = topo_order[::-1].copy()

    # CSR outgoing adjacency
    out_start, out_links = _build_csr_outgoing(num_nodes, tail)

    # Convert to jax arrays
    return JaxGraph(
        num_nodes=num_nodes,
        num_links=num_links,
        tail=jnp.asarray(tail),
        head=jnp.asarray(head),
        topo_order=jnp.asarray(topo_order),
        topo_order_rev=jnp.asarray(topo_order_rev),
        out_start=jnp.asarray(out_start),
        out_links=jnp.asarray(out_links),
        link_type=jnp.asarray(link_type),
        travel_time=jnp.asarray(travel_time_min),
        capacity=jnp.asarray(capacity),
    )


