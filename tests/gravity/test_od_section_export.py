from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.assignment.jax_graph_types import JaxGraph
from public_transportation.inference.assignment_adapter import AssignmentInputs
from public_transportation.inference.gravity import (
    ODSectionCell,
    ODSectionPlanCheckpointError,
    ODSectionPlanInterrupted,
    RideLinkSectionMetadata,
    export_od_section_marginals,
    plan_od_section_marginal_export,
)
from public_transportation.inference.gravity.od_section_export import _PlanCheckpointLock
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingShard,
    FixedRoutingShardDescriptor,
    build_sharded_fixed_routing_inputs,
    save_fixed_routing_shard,
)


def _sha256_file_for_test(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _multi_group_problem(tmp_path):
    inputs, _, metadata, _ = _problem(tmp_path)
    inputs = replace(
        inputs,
        group_dest_node=jnp.asarray([5, 5, 5]),
        group_link_mask=jnp.ones((3, 6), dtype=bool),
        od_origin_node=jnp.asarray([0, 0, 0]),
        group_od_index_padded=jnp.asarray([[0], [1], [2]]),
        group_od_mask=jnp.ones((3, 1), dtype=bool),
    )
    descriptors = tuple(FixedRoutingShardDescriptor(index, index, index + 1) for index in range(3))
    routing = build_sharded_fixed_routing_inputs(
        inputs=inputs,
        theta=1.0,
        shard_partition=descriptors,
        cache_directory=tmp_path / "multi-routing",
    )
    for descriptor in descriptors:
        save_fixed_routing_shard(
            routing=routing,
            shard=FixedRoutingShard(
                descriptor=descriptor,
                effective_group_link_mask=np.ones((1, 6), dtype=bool),
                group_link_probability=np.asarray([[0.6, 0.4, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
            ),
        )
    cells = tuple(ODSectionCell("O", "B", slot) for slot in ("am", "mid", "pm"))
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


def test_plan_checkpoint_resumes_without_recomputing_completed_groups(tmp_path):
    inputs, routing, metadata, cells = _multi_group_problem(tmp_path)
    checkpoint = tmp_path / "plan-checkpoint"
    progress: list[dict[str, object]] = []

    def interrupt(event):
        progress.append(dict(event))
        if event.get("status") == "running" and event.get("current_group") == 0:
            raise ODSectionPlanInterrupted("test interruption")

    with pytest.raises(ODSectionPlanInterrupted):
        plan_od_section_marginal_export(
            assignment_inputs=inputs,
            routing=routing,
            fitted_demand=np.asarray([10.0, 5.0, 7.0]),
            od_cells=cells,
            fit_assignment_fingerprint="fit-123",
            ride_link_metadata=metadata,
            pilot_groups=3,
            checkpoint_directory=checkpoint,
            progress=interrupt,
        )
    checkpoint_payload = json.loads((checkpoint / "plan_checkpoint.json").read_text())
    assert checkpoint_payload["status"] == "interrupted"
    assert checkpoint_payload["completed_groups"] == [0]

    resumed_events: list[dict[str, object]] = []
    resumed = plan_od_section_marginal_export(
        assignment_inputs=inputs,
        routing=routing,
        fitted_demand=np.asarray([10.0, 5.0, 7.0]),
        od_cells=cells,
        fit_assignment_fingerprint="fit-123",
        ride_link_metadata=metadata,
        pilot_groups=3,
        checkpoint_directory=checkpoint,
        resume=True,
        progress=lambda event: resumed_events.append(dict(event)),
    )
    assert resumed.pilot_rows == 6
    assert [event["current_group"] for event in resumed_events if event["status"] == "running"] == [1, 2]

    uninterrupted = plan_od_section_marginal_export(
        assignment_inputs=inputs,
        routing=routing,
        fitted_demand=np.asarray([10.0, 5.0, 7.0]),
        od_cells=cells,
        fit_assignment_fingerprint="fit-123",
        ride_link_metadata=metadata,
        pilot_groups=3,
    )
    assert resumed.pilot_rows == uninterrupted.pilot_rows
    assert resumed.effective_pilot_groups == uninterrupted.effective_pilot_groups


def test_completed_plan_checkpoint_is_reused_and_identity_is_bound(tmp_path):
    inputs, routing, metadata, cells = _multi_group_problem(tmp_path)
    checkpoint = tmp_path / "plan-checkpoint"
    plan_od_section_marginal_export(
        assignment_inputs=inputs,
        routing=routing,
        fitted_demand=np.asarray([10.0, 5.0, 7.0]),
        od_cells=cells,
        fit_assignment_fingerprint="fit-123",
        ride_link_metadata=metadata,
        pilot_groups=3,
        checkpoint_directory=checkpoint,
    )
    reused = plan_od_section_marginal_export(
        assignment_inputs=inputs,
        routing=routing,
        fitted_demand=np.asarray([10.0, 5.0, 7.0]),
        od_cells=cells,
        fit_assignment_fingerprint="fit-123",
        ride_link_metadata=metadata,
        pilot_groups=3,
        checkpoint_directory=checkpoint,
        resume=True,
    )
    assert reused.pilot_rows == 6
    with pytest.raises(ValueError, match="identity differs"):
        plan_od_section_marginal_export(
            assignment_inputs=inputs,
            routing=routing,
            fitted_demand=np.asarray([10.0, 5.0, 7.0]),
            od_cells=cells,
            fit_assignment_fingerprint="different-fit",
            ride_link_metadata=metadata,
            pilot_groups=3,
            checkpoint_directory=checkpoint,
            resume=True,
        )


def test_corrupt_checkpoint_and_concurrent_writer_fail_closed(tmp_path):
    inputs, routing, metadata, cells = _multi_group_problem(tmp_path)
    checkpoint = tmp_path / "plan-checkpoint"
    plan_od_section_marginal_export(
        assignment_inputs=inputs,
        routing=routing,
        fitted_demand=np.asarray([10.0, 5.0, 7.0]),
        od_cells=cells,
        fit_assignment_fingerprint="fit-123",
        ride_link_metadata=metadata,
        pilot_groups=3,
        checkpoint_directory=checkpoint,
    )
    group_path = checkpoint / "groups" / "group-00000000.json"
    group_path.write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(ODSectionPlanCheckpointError, match="missing or corrupt"):
        plan_od_section_marginal_export(
            assignment_inputs=inputs,
            routing=routing,
            fitted_demand=np.asarray([10.0, 5.0, 7.0]),
            od_cells=cells,
            fit_assignment_fingerprint="fit-123",
            ride_link_metadata=metadata,
            pilot_groups=3,
            checkpoint_directory=checkpoint,
            resume=True,
        )

    # A kernel-held lock prevents a second plan writer from entering.
    lock_checkpoint = tmp_path / "locked-checkpoint"
    with _PlanCheckpointLock(lock_checkpoint):
        with pytest.raises(RuntimeError, match="already owns"):
            plan_od_section_marginal_export(
                assignment_inputs=inputs,
                routing=routing,
                fitted_demand=np.asarray([10.0, 5.0, 7.0]),
                od_cells=cells,
                fit_assignment_fingerprint="fit-123",
                ride_link_metadata=metadata,
                pilot_groups=3,
                checkpoint_directory=lock_checkpoint,
            )


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
            if event["group"] == 0 and event["status"] == "chunk_checkpointed"
            else None,
        )
    # The first bounded chunk was durably published before the progress callback ran.
    assert (tmp_path / "checkpoint" / "units" / "group-00000000" / "chunk-00000000.csv").exists()
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


def test_within_group_chunks_report_eta_and_preserve_final_checksum(tmp_path):
    inputs, routing, metadata, cells = _problem(tmp_path)
    events: list[dict[str, object]] = []
    export_od_section_marginals(
        assignment_inputs=inputs,
        routing=routing,
        fitted_demand=np.asarray([10.0, 5.0]),
        od_cells=cells,
        output_path=tmp_path / "complete.csv",
        checkpoint_directory=tmp_path / "complete-checkpoint",
        fit_assignment_fingerprint="fit-123",
        ride_link_metadata=metadata,
        od_chunk_size=1,
        progress_interval_seconds=0.0,
        progress=lambda event: events.append(dict(event)),
    )
    durable = [event for event in events if event["status"] == "chunk_checkpointed"]
    assert durable
    assert durable[0]["estimated_remaining_seconds"] is not None
    assert durable[0]["eta_confidence"] == "low"
    assert durable[0]["completed_group_od_cells"] == 1

    interrupted_checkpoint = tmp_path / "interrupted-checkpoint"

    def stop_after_first_chunk(event):
        if event["status"] == "chunk_checkpointed" and event["chunk_index"] == 0:
            raise ODSectionPlanInterrupted("stop after first chunk")

    with pytest.raises(ODSectionPlanInterrupted):
        export_od_section_marginals(
            assignment_inputs=inputs,
            routing=routing,
            fitted_demand=np.asarray([10.0, 5.0]),
            od_cells=cells,
            output_path=tmp_path / "resumed.csv",
            checkpoint_directory=interrupted_checkpoint,
            fit_assignment_fingerprint="fit-123",
            ride_link_metadata=metadata,
            od_chunk_size=1,
            progress=stop_after_first_chunk,
        )
    resumed = export_od_section_marginals(
        assignment_inputs=inputs,
        routing=routing,
        fitted_demand=np.asarray([10.0, 5.0]),
        od_cells=cells,
        output_path=tmp_path / "resumed.csv",
        checkpoint_directory=interrupted_checkpoint,
        fit_assignment_fingerprint="fit-123",
        ride_link_metadata=metadata,
        od_chunk_size=1,
        progress=lambda _event: None,
    )
    assert resumed["output_sha256"] == _sha256_file_for_test(tmp_path / "complete.csv")


def test_legacy_group_checkpoint_is_migrated_safely(tmp_path):
    inputs, routing, metadata, cells = _problem(tmp_path)
    source = export_od_section_marginals(
        assignment_inputs=inputs,
        routing=routing,
        fitted_demand=np.asarray([10.0, 5.0]),
        od_cells=cells,
        output_path=tmp_path / "source.csv",
        checkpoint_directory=tmp_path / "source-checkpoint",
        fit_assignment_fingerprint="fit-123",
        ride_link_metadata=metadata,
    )
    legacy = tmp_path / "legacy-checkpoint"
    legacy_units = legacy / "units"
    legacy_units.mkdir(parents=True)
    legacy_group = legacy_units / "group-00000000.csv"
    shutil.copyfile(tmp_path / "source.csv", legacy_group)
    current_state = json.loads((tmp_path / "source-checkpoint" / "manifest.json").read_text())
    old_provenance = {
        key: value
        for key, value in current_state["provenance"].items()
        if key in {
            "schema_version",
            "routing_preparation_fingerprint",
            "fit_assignment_fingerprint",
            "fitted_demand_fingerprint",
            "routing_cache_directory",
            "routing_cache_identity",
            "num_od",
            "num_destination_groups",
            "num_ride_links",
        }
    }
    legacy_state = {
        "status": "completed",
        "completed_units": {
            "0": {
                "path": str(legacy_group),
                "rows": int(source["rows_written"]),
                "sha256": _sha256_file_for_test(legacy_group),
            }
        },
        "rows_written": int(source["rows_written"]),
        **old_provenance,
    }
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "manifest.json").write_text(json.dumps(legacy_state), encoding="utf-8")
    migrated = export_od_section_marginals(
        assignment_inputs=inputs,
        routing=routing,
        fitted_demand=np.asarray([10.0, 5.0]),
        od_cells=cells,
        output_path=tmp_path / "migrated.csv",
        checkpoint_directory=legacy,
        fit_assignment_fingerprint="fit-123",
        ride_link_metadata=metadata,
    )
    assert migrated["output_sha256"] == _sha256_file_for_test(tmp_path / "source.csv")
    migrated_state = json.loads((legacy / "manifest.json").read_text())
    assert migrated_state["migration"] == "verified_group_units_to_chunked_checkpoint"
