from __future__ import annotations

import csv

import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.assignment.jax_graph_types import JaxGraph
from public_transportation.inference.assignment_adapter import AssignmentInputs
from public_transportation.inference.gravity import (
    ODSectionCell,
    RideLinkSectionMetadata,
    export_od_section_marginals,
    plan_od_section_marginal_export,
)
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingShard,
    FixedRoutingShardDescriptor,
    build_sharded_fixed_routing_inputs,
    save_fixed_routing_shard,
)


def _problem(tmp_path):
    graph = JaxGraph(
        num_nodes=6,
        num_links=6,
        tail=jnp.asarray([0, 0, 1, 3, 2, 4]),
        head=jnp.asarray([1, 3, 2, 4, 5, 5]),
        topo_order=jnp.asarray([0, 1, 3, 2, 4, 5]),
        topo_order_rev=jnp.asarray([5, 4, 2, 3, 1, 0]),
        node_time=jnp.arange(6, dtype=jnp.float32),
        node_stop_index=jnp.asarray([0, 1, 2, 1, 2, 2]),
        node_time_s=jnp.arange(6),
        node_kind=jnp.asarray([0, 1, 1, 1, 1, 3]),
        node_trip_index=jnp.asarray([-1, 0, 0, 1, 1, -1]),
        out_start=jnp.asarray([0, 2, 3, 4, 5, 6, 6]),
        out_links_csr=jnp.asarray([0, 1, 2, 4, 3, 5]),
        out_links=jnp.asarray([[0, 1], [2, 0], [4, 0], [3, 0], [5, 0], [0, 0]]),
        out_mask=jnp.asarray([[1, 1], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0]], dtype=bool),
        link_type=jnp.asarray([2, 2, 0, 0, 3, 3]),
        travel_time=jnp.ones(6),
        capacity=jnp.full(6, jnp.inf),
        link_trip_index=jnp.asarray([-1, -1, 0, 1, -1, -1]),
        node_stop_id=("O", "A", "B"),
        trip_id=("T1", "T2"),
        trip_line_ref=("L", "L"),
    )
    inputs = AssignmentInputs(
        graph=graph,
        base_link_cost=jnp.zeros(6),
        group_dest_node=jnp.asarray([5]),
        group_link_mask=jnp.ones((1, 6), dtype=bool),
        od_origin_node=jnp.asarray([0, 0]),
        group_od_index_padded=jnp.asarray([[0, 1]]),
        group_od_mask=jnp.asarray([[True, True]]),
    )
    descriptor = FixedRoutingShardDescriptor(0, 0, 1)
    routing = build_sharded_fixed_routing_inputs(
        inputs=inputs,
        theta=1.0,
        shard_partition=(descriptor,),
        cache_directory=tmp_path / "routing",
    )
    save_fixed_routing_shard(
        routing=routing,
        shard=FixedRoutingShard(
            descriptor=descriptor,
            effective_group_link_mask=np.ones((1, 6), dtype=bool),
            group_link_probability=np.asarray([[0.6, 0.4, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
        ),
    )
    metadata = {
        2: RideLinkSectionMetadata("L", "T1", "A", "B", pattern_id="P", direction="0"),
        3: RideLinkSectionMetadata("L", "T2", "A", "B", pattern_id="P", direction="0"),
    }
    cells = (
        ODSectionCell("O", "B", "am"),
        ODSectionCell("O", "B", "pm"),
    )
    return inputs, routing, metadata, cells


def test_expected_od_section_marginals_match_hand_calculation(tmp_path):
    inputs, routing, metadata, cells = _problem(tmp_path)
    result = export_od_section_marginals(
        assignment_inputs=inputs,
        routing=routing,
        fitted_demand=np.asarray([10.0, 5.0]),
        od_cells=cells,
        output_path=tmp_path / "flows.csv",
        checkpoint_directory=tmp_path / "checkpoint",
        fit_assignment_fingerprint="fit-123",
        ride_link_metadata=metadata,
    )
    assert result["rows_written"] == 4
    with (tmp_path / "flows.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["assigned_demand"]) for row in rows] == pytest.approx([6.0, 4.0, 3.0, 2.0], rel=1e-6)
    assert {row["flow_semantics"] for row in rows} == {"service_marginal"}
    assert {row["boarding_stop_id"] for row in rows} == {"A"}
    assert {row["alighting_stop_id"] for row in rows} == {"B"}
    assert {row["assignment_fingerprint"] for row in rows} == {"fit-123"}
    assert sum(float(row["assigned_demand"]) for row in rows if row["trip_id"] == "T1") == pytest.approx(9.0, rel=1e-6)
    assert sum(float(row["assigned_demand"]) for row in rows if row["trip_id"] == "T2") == pytest.approx(6.0, rel=1e-6)


def test_export_rejects_incomplete_routing_shard(tmp_path):
    inputs, routing, metadata, cells = _problem(tmp_path)
    (tmp_path / "routing" / "routing-shard-000000.npz").unlink()
    with pytest.raises(Exception, match="routing shard"):
        export_od_section_marginals(
            assignment_inputs=inputs,
            routing=routing,
            fitted_demand=np.asarray([1.0, 1.0]),
            od_cells=cells,
            output_path=tmp_path / "flows.csv",
            fit_assignment_fingerprint="fit-123",
            ride_link_metadata=metadata,
        )


def test_plan_reports_work_units_and_pilot_estimate(tmp_path):
    inputs, routing, metadata, cells = _problem(tmp_path)
    plan = plan_od_section_marginal_export(
        assignment_inputs=inputs,
        routing=routing,
        fitted_demand=np.asarray([10.0, 5.0]),
        od_cells=cells,
        fit_assignment_fingerprint="fit-123",
        ride_link_metadata=metadata,
        pilot_groups=1,
    )
    assert plan.num_destination_groups == 1
    assert plan.potential_rows == 4
    assert plan.pilot_rows == 4
    assert plan.pilot_seconds is not None
    assert plan.estimated_full_seconds is not None


def test_export_resumes_from_verified_group_unit(tmp_path):
    inputs, routing, metadata, cells = _problem(tmp_path)
    with pytest.raises(RuntimeError, match="stop"):
        export_od_section_marginals(
            assignment_inputs=inputs,
            routing=routing,
            fitted_demand=np.asarray([10.0, 5.0]),
            od_cells=cells,
            output_path=tmp_path / "flows.csv",
            checkpoint_directory=tmp_path / "checkpoint",
            fit_assignment_fingerprint="fit-123",
            ride_link_metadata=metadata,
            progress=lambda event: (_ for _ in ()).throw(RuntimeError("stop"))
            if event["group"] == 0
            else None,
        )
    # The unit was durably published before the progress callback ran.
    assert (tmp_path / "checkpoint" / "units" / "group-00000000.csv").exists()
    result = export_od_section_marginals(
        assignment_inputs=inputs,
        routing=routing,
        fitted_demand=np.asarray([10.0, 5.0]),
        od_cells=cells,
        output_path=tmp_path / "flows.csv",
        checkpoint_directory=tmp_path / "checkpoint",
        fit_assignment_fingerprint="fit-123",
        ride_link_metadata=metadata,
    )
    assert result["rows_written"] == 4
