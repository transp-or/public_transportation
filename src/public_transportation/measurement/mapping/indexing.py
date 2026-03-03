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

    return AssignmentMappingIndex(
        stop_index_by_id=stop_index_by_id,
        trip_index_by_id=trip_index_by_id,
        trip_indices_by_line_id=trip_indices_by_line_id,
        event_node_index=idx,
    )


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