from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.assignment.graph_sentinels import (
    LINK_TYPE_ACCESS,
    LINK_TYPE_EGRESS,
    NODE_KIND_EVENT_ARR,
    NODE_KIND_EVENT_DEP,
)


@dataclass(frozen=True, slots=True)
class AssignmentMappingIndex:
    """Precomputed lookup structures derived from AssignmentIDManager."""
    stop_index_by_id: dict[str, int]
    trip_index_by_id: dict[str, int]
    trip_indices_by_line_id: dict[str, list[int]]
    event_node_index: dict[tuple[int, int, int, int], int]  # (kind, stop_i, trip_i, time_s) -> node_id
    boarding_link_start: np.ndarray
    boarding_link_index: np.ndarray
    alighting_link_start: np.ndarray
    alighting_link_index: np.ndarray


def _event_link_csr(
    *, num_nodes: int, event_node: np.ndarray, link_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Create stable node-to-link CSR arrays in one linear-size preparation."""
    nodes = np.asarray(event_node, dtype=np.int64)
    links = np.asarray(link_index, dtype=np.int32)
    order = np.argsort(nodes, kind="stable")
    sorted_nodes = nodes[order]
    sorted_links = np.ascontiguousarray(links[order])
    counts = np.bincount(sorted_nodes, minlength=num_nodes)
    start = np.empty((num_nodes + 1,), dtype=np.int64)
    start[0] = 0
    np.cumsum(counts, out=start[1:])
    return start, sorted_links


def build_assignment_mapping_index(idm: AssignmentIDManager) -> AssignmentMappingIndex:
    stop_index_by_id = {str(sid): int(i) for i, sid in enumerate(idm.stop_id)}
    trip_index_by_id = {str(tid): int(i) for i, tid in enumerate(idm.trip_id)}

    trip_indices_by_line_id: dict[str, list[int]] = {}
    for ti, lr in enumerate(idm.trip_line_ref):
        trip_indices_by_line_id.setdefault(str(lr), []).append(int(ti))
    for lid in trip_indices_by_line_id:
        trip_indices_by_line_id[lid].sort()

    # Build strict event-node index:
    # key = (node_kind, stop_index, trip_index, time_s) -> unique node_id
    idx: dict[tuple[int, int, int, int], int] = {}

    nk = idm.node_kind
    ns = idm.node_stop_index
    try:
        nt = idm.node_trip_index
    except AttributeError as e:
        raise ValueError(
            "AssignmentIDManager is missing node_trip_index. "
            "It must expose `node_trip_index` aligned with graph nodes so event nodes can be matched by trip."
        ) from e
    time_s = idm.node_time_s

    for node_id in range(int(idm.num_nodes)):
        kind = int(nk[node_id])
        if kind not in (int(NODE_KIND_EVENT_DEP), int(NODE_KIND_EVENT_ARR)):
            continue
        key = (kind, int(ns[node_id]), int(nt[node_id]), int(time_s[node_id]))
        if key in idx:
            raise ValueError(
                "Ambiguous graph: multiple event nodes share the same (kind, stop, trip, time) key: "
                f"{key}. Node ids: {idx[key]} and {node_id}"
            )
        idx[key] = int(node_id)

    access_links = np.flatnonzero(
        idm.link_type == int(LINK_TYPE_ACCESS)
    ).astype(np.int32)
    boarding_start, boarding_links = _event_link_csr(
        num_nodes=int(idm.num_nodes),
        event_node=idm.link_head[access_links],
        link_index=access_links,
    )
    egress_links = np.flatnonzero(
        idm.link_type == int(LINK_TYPE_EGRESS)
    ).astype(np.int32)
    alighting_start, alighting_links = _event_link_csr(
        num_nodes=int(idm.num_nodes),
        event_node=idm.link_tail[egress_links],
        link_index=egress_links,
    )

    return AssignmentMappingIndex(
        stop_index_by_id=stop_index_by_id,
        trip_index_by_id=trip_index_by_id,
        trip_indices_by_line_id=trip_indices_by_line_id,
        event_node_index=idx,
        boarding_link_start=boarding_start,
        boarding_link_index=boarding_links,
        alighting_link_start=alighting_start,
        alighting_link_index=alighting_links,
    )


def indexed_links_for_boarding(
    index: AssignmentMappingIndex, dep_node: int
) -> np.ndarray:
    """Return access links entering a departure node from the prepared CSR."""
    start = int(index.boarding_link_start[dep_node])
    end = int(index.boarding_link_start[dep_node + 1])
    return index.boarding_link_index[start:end]


def indexed_links_for_alighting(
    index: AssignmentMappingIndex, arr_node: int
) -> np.ndarray:
    """Return egress links leaving an arrival node from the prepared CSR."""
    start = int(index.alighting_link_start[arr_node])
    end = int(index.alighting_link_start[arr_node + 1])
    return index.alighting_link_index[start:end]


def links_for_boarding(idm: AssignmentIDManager, dep_node: int) -> np.ndarray:
    """Boarding := ACCESS links entering the departure event node."""
    head = idm.link_head
    ltype = idm.link_type
    return np.where((ltype == int(LINK_TYPE_ACCESS)) & (head == int(dep_node)))[0]


def links_for_alighting(idm: AssignmentIDManager, arr_node: int) -> np.ndarray:
    """Alighting := EGRESS links leaving the arrival event node."""
    tail = idm.link_tail
    ltype = idm.link_type
    return np.where((ltype == int(LINK_TYPE_EGRESS)) & (tail == int(arr_node)))[0]
