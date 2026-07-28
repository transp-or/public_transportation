from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from public_transportation.assignment.graph_sentinels import (
    LINK_TYPE_ACCESS,
    LINK_TYPE_EGRESS,
    NODE_KIND_EVENT_ARR,
    NODE_KIND_EVENT_DEP,
)
from public_transportation.measurement.mapping.indexing import (
    build_assignment_mapping_index,
    indexed_links_for_alighting,
    indexed_links_for_boarding,
    links_for_alighting,
    links_for_boarding,
)


def _id_manager():
    return SimpleNamespace(
        num_nodes=6,
        num_links=7,
        stop_id=("a", "b"),
        trip_id=("t",),
        trip_line_ref=("l",),
        node_kind=np.asarray(
            [0, NODE_KIND_EVENT_DEP, NODE_KIND_EVENT_ARR, NODE_KIND_EVENT_DEP, 0, 0]
        ),
        node_stop_index=np.asarray([-1, 0, 0, 1, -1, -1]),
        node_trip_index=np.asarray([-1, 0, 0, 0, -1, -1]),
        node_time_s=np.asarray([-1, 100, 110, 120, -1, -1]),
        link_type=np.asarray(
            [LINK_TYPE_ACCESS, 0, LINK_TYPE_EGRESS, LINK_TYPE_ACCESS, 1, LINK_TYPE_ACCESS, LINK_TYPE_EGRESS]
        ),
        link_head=np.asarray([1, 2, 5, 3, 4, 1, 5]),
        link_tail=np.asarray([0, 1, 2, 0, 3, 4, 2]),
    )


def test_event_link_csr_matches_legacy_full_link_scan():
    id_manager = _id_manager()
    index = build_assignment_mapping_index(id_manager)

    for node in range(id_manager.num_nodes):
        np.testing.assert_array_equal(
            indexed_links_for_boarding(index, node),
            links_for_boarding(id_manager, node),
        )
        np.testing.assert_array_equal(
            indexed_links_for_alighting(index, node),
            links_for_alighting(id_manager, node),
        )

    np.testing.assert_array_equal(indexed_links_for_boarding(index, 1), [0, 5])
    np.testing.assert_array_equal(indexed_links_for_alighting(index, 2), [2, 6])
