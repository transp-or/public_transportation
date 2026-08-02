"""Compute scheduled path metrics for every OD/time cell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from public_transportation.assignment.graph_sentinels import (
    LINK_TYPE_ACCESS,
    LINK_TYPE_EGRESS,
    LINK_TYPE_TRANSFER,
    NODE_KIND_EVENT_ARR,
)

from .progress import ProgressEmitter, StructuralZeroProgress
from .topology import StructuralZeroTopology
from .types import ODPathMetricRecord, ODPathMetrics, ODTimeKey


@dataclass(frozen=True, slots=True)
class _DestinationProfile:
    reachable: np.ndarray
    minimum_transfers: np.ndarray
    earliest_destination_arrival_s: np.ndarray


def compute_od_path_metrics(
    topology: StructuralZeroTopology,
    *,
    keys: tuple[ODTimeKey, ...] | None = None,
    progress: Callable[[StructuralZeroProgress], None] | None = None,
) -> tuple[ODPathMetricRecord, ...]:
    """Compute metrics for candidate keys or the full Cartesian product.

    The dynamic program runs once per destination, not once per OD/time cell.
    Its complexity is ``O(|stops| * (|nodes| + |links|) + |OD/time cells| *
    mean access degree)``.

    ``minimum_initial_wait_minutes`` is measured from the start of the time bin
    to the boarding departure. A departure before the bin start has zero wait.
    ``minimum_journey_time_minutes`` is measured from boarding departure to the
    final arrival at the destination. Each reported minimum may be attained by
    a different feasible path.
    """
    graph = topology.graph
    node_time_s = np.asarray(graph.node_time_s, dtype=np.int64)
    link_type = np.asarray(graph.link_type, dtype=np.int64)
    head = np.asarray(graph.head, dtype=np.int64)
    bin_start_min = np.asarray(graph.node_bin_start_min, dtype=float)

    profiles_list: list[_DestinationProfile] = []
    profile_progress = ProgressEmitter(
        progress,
        phase="destination_profiles",
        total=len(topology.stop_ids),
    )
    profile_progress.start()
    for destination_index in range(len(topology.stop_ids)):
        profiles_list.append(
            _profile_for_destination(
                topology, destination_stop_index=destination_index
            )
        )
        profile_progress.update(destination_index + 1)
    profiles = tuple(profiles_list)

    candidates = _candidate_keys(topology, keys)
    stop_index = {stop_id: index for index, stop_id in enumerate(topology.stop_ids)}
    bin_index = {
        time_bin_id: index for index, time_bin_id in enumerate(topology.time_bin_ids)
    }
    records: list[ODPathMetricRecord] = []
    for key in candidates:
        origin_index = stop_index[key.origin_stop_id]
        destination_index = stop_index[key.dest_stop_id]
        time_bin_index = bin_index[key.time_bin_id]
        profile = profiles[destination_index]
        origin_node = topology.centroid_in_node[origin_index][time_bin_index]
        feasible_access = tuple(
            link
            for link in topology.outgoing_links[origin_node]
            if link_type[link] == LINK_TYPE_ACCESS and profile.reachable[head[link]]
        )
        if not feasible_access:
            metrics = ODPathMetrics.unreachable()
        else:
            departure_nodes = head[np.asarray(feasible_access, dtype=np.int64)]
            departure_s = node_time_s[departure_nodes]
            arrival_s = profile.earliest_destination_arrival_s[departure_nodes]
            journey_s = arrival_s - departure_s
            if np.any(journey_s < 0):
                raise ValueError(
                    "Scheduled topology produced an arrival before its departure."
                )
            start_min = float(bin_start_min[origin_node])
            waits_min = np.maximum(0.0, departure_s / 60.0 - start_min)
            metrics = ODPathMetrics(
                feasible=True,
                minimum_transfers=int(
                    np.min(profile.minimum_transfers[departure_nodes])
                ),
                minimum_initial_wait_minutes=float(np.min(waits_min)),
                minimum_journey_time_minutes=float(np.min(journey_s) / 60.0),
                feasible_departure_count=len(feasible_access),
                earliest_arrival_seconds=int(np.min(arrival_s)),
            )
        records.append(ODPathMetricRecord(key=key, metrics=metrics))

    return tuple(sorted(records, key=lambda record: record.key))


def _candidate_keys(
    topology: StructuralZeroTopology,
    keys: tuple[ODTimeKey, ...] | None,
) -> tuple[ODTimeKey, ...]:
    if keys is None:
        return tuple(
            ODTimeKey(origin, destination, time_bin)
            for origin in topology.stop_ids
            for destination in topology.stop_ids
            for time_bin in topology.time_bin_ids
        )
    if not isinstance(keys, tuple):
        raise TypeError("keys must be a tuple of ODTimeKey values or None.")
    if any(not isinstance(key, ODTimeKey) for key in keys):
        raise TypeError("keys must contain ODTimeKey values.")
    if len(keys) != len(set(keys)):
        raise ValueError("Candidate OD/time keys must be unique.")
    stop_ids = set(topology.stop_ids)
    time_bin_ids = set(topology.time_bin_ids)
    for key in keys:
        if key.origin_stop_id not in stop_ids or key.dest_stop_id not in stop_ids:
            raise ValueError(
                f"Candidate key references an unknown stop: {key.tuple!r}."
            )
        if key.time_bin_id not in time_bin_ids:
            raise ValueError(
                f"Candidate key references an unknown time bin: {key.tuple!r}."
            )
    return tuple(sorted(keys))


def _profile_for_destination(
    topology: StructuralZeroTopology,
    *,
    destination_stop_index: int,
) -> _DestinationProfile:
    graph = topology.graph
    num_nodes = graph.num_nodes
    destination_node = topology.centroid_out_node[destination_stop_index]
    node_kind = np.asarray(graph.node_kind, dtype=np.int64)
    node_stop = np.asarray(graph.node_stop_index, dtype=np.int64)
    node_time_s = np.asarray(graph.node_time_s, dtype=np.int64)
    head = np.asarray(graph.head, dtype=np.int64)
    link_type = np.asarray(graph.link_type, dtype=np.int64)
    reverse_order = np.asarray(graph.topo_order_rev, dtype=np.int64)

    reachable = np.zeros(num_nodes, dtype=bool)
    minimum_transfers = np.full(num_nodes, np.iinfo(np.int64).max, dtype=np.int64)
    earliest_arrival = np.full(num_nodes, np.iinfo(np.int64).max, dtype=np.int64)
    reachable[destination_node] = True
    minimum_transfers[destination_node] = 0

    for node in reverse_order:
        node_index = int(node)
        if node_index == destination_node:
            continue
        is_destination_arrival = (
            node_kind[node_index] == NODE_KIND_EVENT_ARR
            and node_stop[node_index] == destination_stop_index
        )
        for link in topology.outgoing_links[node_index]:
            next_node = int(head[link])
            if is_destination_arrival and not (
                link_type[link] == LINK_TYPE_EGRESS and next_node == destination_node
            ):
                continue
            if not reachable[next_node]:
                continue
            reachable[node_index] = True
            transfer_increment = int(link_type[link] == LINK_TYPE_TRANSFER)
            minimum_transfers[node_index] = min(
                minimum_transfers[node_index],
                transfer_increment + minimum_transfers[next_node],
            )
            if link_type[link] == LINK_TYPE_EGRESS and next_node == destination_node:
                candidate_arrival = int(node_time_s[node_index])
            else:
                candidate_arrival = int(earliest_arrival[next_node])
            earliest_arrival[node_index] = min(
                earliest_arrival[node_index], candidate_arrival
            )

    return _DestinationProfile(
        reachable=reachable,
        minimum_transfers=minimum_transfers,
        earliest_destination_arrival_s=earliest_arrival,
    )
