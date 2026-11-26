from __future__ import annotations

from types import SimpleNamespace
import threading
from pathlib import Path

import numpy as np
import pytest

from public_transportation.assignment.graph_sentinels import (
    LINK_TYPE_ACCESS,
    LINK_TYPE_EGRESS,
    LINK_TYPE_RIDE,
    NODE_KIND_CENTROID_IN,
    NODE_KIND_CENTROID_OUT,
    NODE_KIND_EVENT_ARR,
    NODE_KIND_EVENT_DEP,
)
from public_transportation.inference.assignment_contract import (
    CanonicalTimeInterval,
    fixed_routing_route_choice_fingerprint,
)
from public_transportation.inference.assignment_adapter import FixedRoutingInputs
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingShard,
    FixedRoutingShardCacheProvenance,
    FixedRoutingShardDescriptor,
    ShardedFixedRoutingInputs,
    save_fixed_routing_shard,
)
from public_transportation.inference.gravity.route_path_query import (
    RouteChoicePathQuery,
)
from public_transportation.inference.gravity.viewer_query_service import (
    GravityViewerQueryService,
)


def _graph() -> SimpleNamespace:
    return SimpleNamespace(
        tail=np.asarray([0, 0, 1, 2, 3, 4]),
        head=np.asarray([1, 2, 3, 4, 5, 5]),
        node_kind=np.asarray(
            [
                NODE_KIND_CENTROID_IN,
                NODE_KIND_EVENT_DEP,
                NODE_KIND_EVENT_DEP,
                NODE_KIND_EVENT_ARR,
                NODE_KIND_EVENT_ARR,
                NODE_KIND_CENTROID_OUT,
            ]
        ),
        node_stop_index=np.asarray([0, 0, 0, 1, 1, 1]),
        node_time_bin_index=np.asarray([0, -1, -1, -1, -1, -1]),
        node_time_s=np.asarray([-1, 28_800, 28_920, 29_400, 29_520, -1]),
        node_trip_index=np.asarray([-1, 0, 1, 0, 1, -1]),
        link_type=np.asarray(
            [
                LINK_TYPE_ACCESS,
                LINK_TYPE_ACCESS,
                LINK_TYPE_RIDE,
                LINK_TYPE_RIDE,
                LINK_TYPE_EGRESS,
                LINK_TYPE_EGRESS,
            ]
        ),
        link_trip_index=np.asarray([-1, -1, 0, 1, -1, -1]),
        topo_order=np.arange(6),
        node_stop_id=("A", "B"),
        node_stop_name=("Origin", "Destination"),
        trip_id=("trip-a", "trip-b"),
        trip_line_ref=("line-a", "line-b"),
        out_start=np.asarray([0, 2, 3, 4, 5, 6, 6]),
        out_links_csr=np.arange(6),
    )


def _query(*, max_paths: int = 200) -> RouteChoicePathQuery:
    graph = _graph()
    probabilities = np.asarray([0.6, 0.4, 1.0, 1.0, 1.0, 1.0])
    routing = FixedRoutingInputs(
        theta=np.asarray(1.0),
        graph=graph,
        source_base_link_cost=np.ones(6),
        group_dest_node=np.asarray([5]),
        source_group_link_mask=np.ones((1, 6), dtype=bool),
        effective_group_link_mask=np.ones((1, 6), dtype=bool),
        group_link_probability=probabilities[np.newaxis, :],
        num_nodes=6,
        num_links=6,
    )
    query = RouteChoicePathQuery.from_fixed_routing(
        routing=routing,
        time_intervals=(CanonicalTimeInterval("bin-0", 28_800, 32_400),),
        mapping_artifact_fingerprint="mapping-identity",
        fit_fingerprint="fit-identity",
        max_paths=max_paths,
    )
    return query


def test_route_path_query_lists_complete_paths_above_threshold_in_probability_order() -> (
    None
):
    result = _query().query(
        origin_stop_id="A",
        destination_stop_id="B",
        departure_interval_id="bin-0",
        minimum_probability=0.3,
    )

    assert [row["probability"] for row in result["paths"]] == pytest.approx([0.6, 0.4])
    assert result["total_path_probability"] == pytest.approx(1.0)
    assert result["listed_probability_mass"] == pytest.approx(1.0)
    assert result["unlisted_probability_mass"] == pytest.approx(0.0)
    assert [row["stops"] for row in result["paths"]] == [["A", "B"], ["A", "B"]]
    assert result["paths"][0]["legs"][0]["line_id"] == "line-a"
    assert result["paths"][1]["legs"][0]["line_id"] == "line-b"
    assert result["fit_fingerprint"] == "fit-identity"
    assert result["mapping_artifact_fingerprint"] == "mapping-identity"


def test_route_path_query_threshold_and_path_count_bound_are_explicit() -> None:
    query = _query(max_paths=1)
    result = query.query(
        origin_stop_id="A",
        destination_stop_id="B",
        departure_interval_id="bin-0",
        minimum_probability=0.3,
    )
    assert [row["probability"] for row in result["paths"]] == pytest.approx([0.6])
    assert result["truncated"] is True

    no_paths = query.query(
        origin_stop_id="A",
        destination_stop_id="B",
        departure_interval_id="bin-0",
        minimum_probability=0.7,
    )
    assert no_paths["paths"] == []
    assert no_paths["total_path_probability"] == pytest.approx(1.0)


def test_route_path_query_rejects_unknown_endpoint_interval_and_threshold() -> None:
    query = _query()
    with pytest.raises(ValueError, match="different stops"):
        query.query(
            origin_stop_id="A",
            destination_stop_id="A",
            departure_interval_id="bin-0",
        )
    with pytest.raises(ValueError, match="selected stop or interval"):
        query.query(
            origin_stop_id="missing",
            destination_stop_id="B",
            departure_interval_id="bin-0",
        )
    with pytest.raises(ValueError, match="minimum_probability"):
        query.query(
            origin_stop_id="A",
            destination_stop_id="B",
            departure_interval_id="bin-0",
            minimum_probability=0,
        )


def test_route_path_query_identity_validation_binds_fit_mapping_and_schedule() -> None:
    query = _query()
    query.validate_identity(
        mapping_artifact_fingerprint="mapping-identity",
        fit_fingerprint="fit-identity",
        route_choice_fingerprint=fixed_routing_route_choice_fingerprint(1.0),
        time_intervals=(CanonicalTimeInterval("bin-0", 28_800, 32_400),),
    )
    with pytest.raises(ValueError, match="mapping identity"):
        query.validate_identity(
            mapping_artifact_fingerprint="other-mapping",
            fit_fingerprint="fit-identity",
            route_choice_fingerprint=fixed_routing_route_choice_fingerprint(1.0),
            time_intervals=(CanonicalTimeInterval("bin-0", 28_800, 32_400),),
        )


def test_local_service_returns_bounded_route_path_page() -> None:
    query = _query(max_paths=1)
    service = object.__new__(GravityViewerQueryService)
    service.route_path_query = query
    service._query_lock = threading.Lock()
    service._query_request_lock = threading.Lock()
    service._query_requests = {}
    service._cancelled_query_ids = set()

    result = service.query_route_paths(
        origin_stop_id="A",
        destination_stop_id="B",
        departure_interval_id="bin-0",
        minimum_probability=0.3,
        request_id="route-request",
    )
    assert result["request_id"] == "route-request"
    assert len(result["paths"]) == 1
    assert result["truncated"] is True
    assert service._query_requests == {}


def test_route_path_query_reads_a_validated_sharded_destination_group(
    tmp_path: Path,
) -> None:
    descriptor = FixedRoutingShardDescriptor(0, 0, 1)
    probabilities = np.asarray([0.6, 0.4, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    routing = ShardedFixedRoutingInputs(
        theta=1.0,
        graph=_graph(),
        graph_fingerprint="graph-fingerprint",
        base_link_cost_fingerprint="base-fingerprint",
        source_group_link_mask_fingerprint="mask-fingerprint",
        assignment_fingerprint="assignment-fingerprint",
        destination_group_identifiers=(5,),
        num_nodes=6,
        num_links=6,
        probability_dtype="float32",
        shard_partition=(descriptor,),
        provenance=FixedRoutingShardCacheProvenance(
            preparation_fingerprint="preparation-fingerprint",
            cache_directory=tmp_path,
        ),
    )
    save_fixed_routing_shard(
        routing=routing,
        shard=FixedRoutingShard(
            descriptor=descriptor,
            effective_group_link_mask=np.ones((1, 6), dtype=bool),
            group_link_probability=probabilities[np.newaxis, :],
        ),
    )

    query = RouteChoicePathQuery.from_fixed_routing(
        routing=routing,
        time_intervals=(CanonicalTimeInterval("bin-0", 28_800, 32_400),),
        mapping_artifact_fingerprint="mapping-identity",
        fit_fingerprint="fit-identity",
    )
    result = query.query(
        origin_stop_id="A",
        destination_stop_id="B",
        departure_interval_id="bin-0",
        minimum_probability=0.5,
    )
    assert [row["probability"] for row in result["paths"]] == pytest.approx([0.6])
