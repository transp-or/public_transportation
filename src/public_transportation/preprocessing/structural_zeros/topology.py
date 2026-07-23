"""Build the immutable scheduled topology used by structural-zero analysis."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from public_transportation.assignment.build_time_expanded import build_jax_graph
from public_transportation.assignment.config import AssignmentConfig
from public_transportation.assignment.graph_sentinels import (
    LINK_TYPE_ACCESS,
    LINK_TYPE_TRANSFER,
    NODE_KIND_CENTROID_IN,
    NODE_KIND_CENTROID_OUT,
)
from public_transportation.assignment.jax_graph_types import JaxGraph

from .config import StructuralZeroAssignmentConfig

if TYPE_CHECKING:
    from public_transportation.domain.scenario import Scenario


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...


@dataclass(frozen=True, slots=True)
class StructuralZeroTopology:
    """Indexed view of the assignment graph with no demand-dependent state.

    Centroid arrays use the deterministic Cartesian ordering
    ``stop_ids × time_bin_ids``. Thus the centroid-in node for stop index ``s``
    and time-bin index ``t`` is at ``centroid_in_node[s][t]``.
    """

    graph: JaxGraph
    assignment_config: StructuralZeroAssignmentConfig
    stop_ids: tuple[str, ...]
    time_bin_ids: tuple[str, ...]
    centroid_in_node: tuple[tuple[int, ...], ...]
    centroid_out_node: tuple[int, ...]
    outgoing_links: tuple[tuple[int, ...], ...]
    incoming_links: tuple[tuple[int, ...], ...]
    access_links: tuple[int, ...]
    transfer_links: tuple[int, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if len(self.stop_ids) != len(self.centroid_in_node):
            raise ValueError("centroid_in_node must contain one row per stop.")
        if len(self.stop_ids) != len(self.centroid_out_node):
            raise ValueError("centroid_out_node must contain one node per stop.")
        if any(len(row) != len(self.time_bin_ids) for row in self.centroid_in_node):
            raise ValueError("Each centroid-in row must contain one node per time bin.")
        if len(self.outgoing_links) != self.graph.num_nodes:
            raise ValueError("outgoing_links must contain one row per graph node.")
        if len(self.incoming_links) != self.graph.num_nodes:
            raise ValueError("incoming_links must contain one row per graph node.")
        if not self.fingerprint:
            raise ValueError("fingerprint must be non-empty.")

    def origin_node(self, stop_id: str, time_bin_id: str) -> int:
        """Return the centroid-in node for an OD/time origin."""
        try:
            stop_index = self.stop_ids.index(str(stop_id))
        except ValueError as error:
            raise KeyError(f"Unknown stop_id: {stop_id!r}") from error
        try:
            bin_index = self.time_bin_ids.index(str(time_bin_id))
        except ValueError as error:
            raise KeyError(f"Unknown time_bin_id: {time_bin_id!r}") from error
        return self.centroid_in_node[stop_index][bin_index]

    def destination_node(self, stop_id: str) -> int:
        """Return the centroid-out node for a destination stop."""
        try:
            stop_index = self.stop_ids.index(str(stop_id))
        except ValueError as error:
            raise KeyError(f"Unknown stop_id: {stop_id!r}") from error
        return self.centroid_out_node[stop_index]


def build_structural_zero_topology(
    scenario: Scenario,
    config: StructuralZeroAssignmentConfig,
) -> StructuralZeroTopology:
    """Build the same feasible scheduled graph used by assignment.

    Only graph-construction settings are copied. Cost coefficients, theta, and
    demand are irrelevant to topological feasibility and remain at their
    assignment defaults.
    """
    if scenario.timetable is None:
        raise ValueError("Scenario.timetable is required for structural-zero analysis.")

    assignment_config = AssignmentConfig(
        max_access_deviation_min=config.max_access_deviation_minutes,
        max_transfer_wait_min=config.max_transfer_wait_minutes,
        min_dwell_s=config.minimum_dwell_seconds,
    )
    assignment_config.validate()
    graph = build_jax_graph(scenario=scenario, config=assignment_config)

    stop_ids = tuple(str(value) for value in graph.node_stop_id)
    if not stop_ids or len(stop_ids) != len(set(stop_ids)):
        raise ValueError("The assignment graph must contain unique stop identifiers.")
    time_bin_ids = _time_bin_ids(scenario)

    node_kind = np.asarray(graph.node_kind, dtype=np.int64)
    node_stop = np.asarray(graph.node_stop_index, dtype=np.int64)
    node_bin = np.asarray(graph.node_time_bin_index, dtype=np.int64)
    tail = np.asarray(graph.tail, dtype=np.int64)
    head = np.asarray(graph.head, dtype=np.int64)
    link_type = np.asarray(graph.link_type, dtype=np.int64)

    _validate_graph_arrays(graph, tail=tail, head=head)
    centroid_in = _centroid_in_nodes(
        graph,
        node_kind=node_kind,
        node_stop=node_stop,
        node_bin=node_bin,
        num_stops=len(stop_ids),
        num_bins=len(time_bin_ids),
    )
    centroid_out = _centroid_out_nodes(
        graph,
        node_kind=node_kind,
        node_stop=node_stop,
        num_stops=len(stop_ids),
    )
    outgoing, incoming = _adjacency(graph.num_nodes, tail=tail, head=head)

    fingerprint = _graph_fingerprint(
        graph,
        stop_ids=stop_ids,
        time_bin_ids=time_bin_ids,
        config=config,
    )
    return StructuralZeroTopology(
        graph=graph,
        assignment_config=config,
        stop_ids=stop_ids,
        time_bin_ids=time_bin_ids,
        centroid_in_node=centroid_in,
        centroid_out_node=centroid_out,
        outgoing_links=outgoing,
        incoming_links=incoming,
        access_links=tuple(
            int(i) for i in np.flatnonzero(link_type == LINK_TYPE_ACCESS)
        ),
        transfer_links=tuple(
            int(i) for i in np.flatnonzero(link_type == LINK_TYPE_TRANSFER)
        ),
        fingerprint=fingerprint,
    )


def _time_bin_ids(scenario: Scenario) -> tuple[str, ...]:
    values: list[str] = []
    for index, time_bin in enumerate(scenario.time_bins):
        value = getattr(time_bin, "bin_id", None)
        if value is None or not str(value).strip():
            raise ValueError(f"Scenario time bin {index} has no non-empty bin_id.")
        values.append(str(value))
    result = tuple(values)
    if not result:
        raise ValueError("Scenario must contain at least one time bin.")
    if len(result) != len(set(result)):
        raise ValueError("Scenario time-bin identifiers must be unique.")
    return result


def _validate_graph_arrays(
    graph: JaxGraph, *, tail: np.ndarray, head: np.ndarray
) -> None:
    if tail.shape != (graph.num_links,) or head.shape != (graph.num_links,):
        raise ValueError("Assignment graph link arrays have inconsistent shapes.")
    if np.any(tail < 0) or np.any(tail >= graph.num_nodes):
        raise ValueError("Assignment graph contains an invalid link tail.")
    if np.any(head < 0) or np.any(head >= graph.num_nodes):
        raise ValueError("Assignment graph contains an invalid link head.")

    order = np.asarray(graph.topo_order, dtype=np.int64)
    if order.shape != (graph.num_nodes,) or set(order.tolist()) != set(
        range(graph.num_nodes)
    ):
        raise ValueError("Assignment graph has an invalid topological order.")
    rank = np.empty(graph.num_nodes, dtype=np.int64)
    rank[order] = np.arange(graph.num_nodes)
    if np.any(rank[tail] >= rank[head]):
        raise ValueError("Assignment graph is not a directed acyclic graph.")


def _centroid_in_nodes(
    graph: JaxGraph,
    *,
    node_kind: np.ndarray,
    node_stop: np.ndarray,
    node_bin: np.ndarray,
    num_stops: int,
    num_bins: int,
) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for stop_index in range(num_stops):
        row: list[int] = []
        for bin_index in range(num_bins):
            matches = np.flatnonzero(
                (node_kind == NODE_KIND_CENTROID_IN)
                & (node_stop == stop_index)
                & (node_bin == bin_index)
            )
            if matches.size != 1:
                raise ValueError(
                    "Expected exactly one centroid-in node for "
                    f"stop index {stop_index}, time-bin index {bin_index}; "
                    f"found {matches.size}."
                )
            row.append(int(matches[0]))
        rows.append(tuple(row))
    return tuple(rows)


def _centroid_out_nodes(
    graph: JaxGraph,
    *,
    node_kind: np.ndarray,
    node_stop: np.ndarray,
    num_stops: int,
) -> tuple[int, ...]:
    result: list[int] = []
    for stop_index in range(num_stops):
        matches = np.flatnonzero(
            (node_kind == NODE_KIND_CENTROID_OUT) & (node_stop == stop_index)
        )
        if matches.size != 1:
            raise ValueError(
                "Expected exactly one centroid-out node for "
                f"stop index {stop_index}; found {matches.size}."
            )
        result.append(int(matches[0]))
    return tuple(result)


def _adjacency(
    num_nodes: int, *, tail: np.ndarray, head: np.ndarray
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    outgoing: list[list[int]] = [[] for _ in range(num_nodes)]
    incoming: list[list[int]] = [[] for _ in range(num_nodes)]
    for link, (tail_node, head_node) in enumerate(zip(tail, head, strict=True)):
        outgoing[int(tail_node)].append(link)
        incoming[int(head_node)].append(link)
    return (
        tuple(tuple(row) for row in outgoing),
        tuple(tuple(row) for row in incoming),
    )


def _graph_fingerprint(
    graph: JaxGraph,
    *,
    stop_ids: tuple[str, ...],
    time_bin_ids: tuple[str, ...],
    config: StructuralZeroAssignmentConfig,
) -> str:
    digest = hashlib.sha256()
    for text in (*stop_ids, *time_bin_ids):
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    for value in (
        config.max_access_deviation_minutes,
        config.max_transfer_wait_minutes,
        config.minimum_dwell_seconds,
    ):
        digest.update(repr(value).encode("ascii"))
        digest.update(b"\0")
    for array in (
        graph.tail,
        graph.head,
        graph.topo_order,
        graph.node_time_s,
        graph.node_stop_index,
        graph.node_kind,
        graph.node_trip_index,
        graph.node_time_bin_index,
        graph.link_type,
        graph.travel_time,
        graph.link_trip_index,
    ):
        _update_array_digest(digest, np.asarray(array))
    for text in (*graph.trip_id, *graph.trip_line_ref):
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _update_array_digest(digest: _Digest, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
