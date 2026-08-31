from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import public_transportation.inference.sharded_fixed_routing as sharded_module

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import (
    _routing_inputs_for_destination,
    prepare_assignment,
)
from public_transportation.assignment.dial_dp import prepare_destination_routing
from public_transportation.domain import Scenario
from public_transportation.inference.assignment_adapter import (
    FixedRoutingPreparationDiagnostics,
    FixedRoutingInputs,
    assign_link_flow,
    assign_link_flow_fixed_routing,
    assign_link_flow_fixed_routing_custom_adjoint,
    build_assignment_inputs,
    prepare_fixed_routing,
    validate_fixed_routing_compatibility,
)
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
    fixed_routing_shard_path,
    load_fixed_routing_shard,
    plan_fixed_routing_shards,
    prepare_fixed_routing_sharded,
    recommend_fixed_routing_workers,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "docs/source/examples/simple_example_02"
NETWORK_FILES = (
    "metadata.json",
    "stops.csv",
    "lines.csv",
    "trips.csv",
    "stop_times.csv",
    "time_bins.csv",
)


@pytest.fixture(scope="module")
def assignment_inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fixed-routing-inputs")
    for name in NETWORK_FILES:
        shutil.copy2(EXAMPLE / "data" / name, directory / name)
    shutil.copy2(EXAMPLE / "pre_processing/results/demand.csv", directory / "demand.csv")
    scenario = Scenario.from_folder(directory, strict=True)
    artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
    return build_assignment_inputs(artifacts=artifacts)


def test_prepare_fixed_routing_shapes_values_and_pytree(assignment_inputs):
    prepared = prepare_fixed_routing(inputs=assignment_inputs, theta=1.0)

    num_groups = int(assignment_inputs.group_dest_node.shape[0])
    num_links = assignment_inputs.graph.num_links
    assert isinstance(prepared, FixedRoutingInputs)
    assert prepared.graph is assignment_inputs.graph
    assert prepared.group_link_probability.shape == (num_groups, num_links)
    assert prepared.effective_group_link_mask.shape == (num_groups, num_links)
    assert np.array_equal(prepared.group_dest_node, assignment_inputs.group_dest_node)
    assert np.array_equal(
        prepared.source_group_link_mask,
        assignment_inputs.group_link_mask,
    )
    assert np.array_equal(prepared.source_base_link_cost, assignment_inputs.base_link_cost)
    probability = np.asarray(prepared.group_link_probability)
    effective = np.asarray(prepared.effective_group_link_mask)
    assert np.all(np.isfinite(probability))
    assert np.all(probability >= 0.0)
    assert np.all(probability[~effective] <= np.exp(-80.0))

    children, treedef = jax.tree_util.tree_flatten(prepared)
    rebuilt = jax.tree_util.tree_unflatten(treedef, children)
    assert isinstance(rebuilt, FixedRoutingInputs)
    assert np.array_equal(rebuilt.group_link_probability, probability)
    validate_fixed_routing_compatibility(inputs=assignment_inputs, routing=rebuilt)


def test_prepare_fixed_routing_reports_synchronized_profile(assignment_inputs):
    reports: list[FixedRoutingPreparationDiagnostics] = []

    prepared = prepare_fixed_routing(
        inputs=assignment_inputs,
        theta=1.0,
        diagnostics_callback=reports.append,
    )

    assert len(reports) == 1
    report = reports[0]
    expected_shape = (
        int(assignment_inputs.group_dest_node.shape[0]),
        assignment_inputs.graph.num_links,
    )
    assert report.profiling_enabled
    assert report.effective_mask_shape == expected_shape
    assert report.probability_shape == expected_shape
    assert report.observed_retained_bytes == (
        prepared.effective_group_link_mask.nbytes
        + prepared.group_link_probability.nbytes
    )
    assert report.estimated_retained_bytes == report.observed_retained_bytes
    assert report.tracing_seconds >= 0.0
    assert report.lowering_seconds >= 0.0
    assert report.compilation_seconds >= 0.0
    assert report.first_execution_seconds >= 0.0
    assert report.synchronization_seconds >= 0.0
    assert report.total_elapsed_seconds >= 0.0
    assert report.backend == jax.default_backend()
    assert report.devices
    assert report.captured_constant_bytes is None
    assert not report.deadline_exceeded
    assert report.deadline_phase is None


def test_prepare_fixed_routing_reports_expired_deadline_before_tracing(
    assignment_inputs,
):
    reports: list[FixedRoutingPreparationDiagnostics] = []

    with pytest.raises(TimeoutError, match="during tracing"):
        prepare_fixed_routing(
            inputs=assignment_inputs,
            theta=1.0,
            diagnostics_callback=reports.append,
            absolute_deadline=0.0,
        )

    assert len(reports) == 1
    assert reports[0].deadline_exceeded
    assert reports[0].deadline_phase == "tracing"
    assert not reports[0].indivisible_operation_overshoot


def test_sharded_preparation_matches_complete_and_resumes_from_cache(
    assignment_inputs, tmp_path
):
    complete = prepare_fixed_routing(inputs=assignment_inputs, theta=1.0)
    config = FixedRoutingPreparationConfig(
        maximum_groups_per_shard=2,
        cache_directory=tmp_path / "cache",
        checkpoint_directory=tmp_path / "checkpoint",
    )
    events = []

    first = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=config,
        progress=events.append,
    )
    shards = [
        load_fixed_routing_shard(routing=first.routing, descriptor=descriptor)
        for descriptor in first.plan.descriptors
    ]
    effective = np.concatenate(
        [shard.effective_group_link_mask for shard in shards], axis=0
    )
    probability = np.concatenate(
        [shard.group_link_probability for shard in shards], axis=0
    )

    assert first.status == "completed"
    assert first.compilation_count == 1
    np.testing.assert_array_equal(effective, complete.effective_group_link_mask)
    np.testing.assert_allclose(
        probability,
        complete.group_link_probability,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert [event.completed_shards for event in events] == sorted(
        event.completed_shards for event in events
    )

    resumed = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=config,
    )
    assert resumed.status == "completed"
    assert resumed.cache_hits == len(first.plan.descriptors)
    assert resumed.cache_misses == 0
    assert resumed.compilation_count == 0

    damaged = fixed_routing_shard_path(
        first.routing, first.plan.descriptors[0]
    )
    damaged.write_bytes(b"corrupt routing batch")
    repaired = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=config,
    )
    assert repaired.status == "completed"
    assert repaired.reconstructed_shards == 1
    assert repaired.cache_hits == len(first.plan.descriptors) - 1
    assert list(damaged.parent.glob(f"{damaged.name}.invalid-*"))


def test_sharded_detailed_profile_has_synchronized_warm_phase_diagnostics(
    assignment_inputs, tmp_path
):
    result = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=2,
            detailed_profiling=True,
            cache_directory=tmp_path / "cache",
            checkpoint_directory=tmp_path / "checkpoint",
        ),
    )

    assert len(result.shard_diagnostics) == result.routing.num_shards
    diagnostic = result.shard_diagnostics[-1]
    assert diagnostic.total_shard_seconds > 0.0
    assert diagnostic.device_synchronization_seconds >= 0.0
    assert diagnostic.graph_nodes_traversed == (
        result.plan.groups_per_full_shard * result.routing.num_nodes
    )
    assert diagnostic.graph_links_traversed == (
        result.plan.groups_per_full_shard * result.routing.num_links
    )
    assert 0.0 <= diagnostic.enabled_link_fraction <= 1.0
    assert 0.0 <= diagnostic.probability_density <= 1.0
    assert diagnostic.retained_bytes > 0


def test_sharded_preparation_stops_cleanly_before_first_shard(
    assignment_inputs, tmp_path
):
    result = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            cache_directory=tmp_path / "cache",
            checkpoint_directory=tmp_path / "checkpoint",
        ),
        absolute_deadline=0.0,
    )

    assert result.status == "deadline_reached"
    assert result.completed_shards == 0
    assert result.deadline_phase == "before shard"
    assert not result.indivisible_operation_overshoot
    manifest = tmp_path / "checkpoint" / "manifest.json"
    assert manifest.exists()


def test_sharded_preparation_rejects_incompatible_manifest(
    assignment_inputs, tmp_path
):
    config = FixedRoutingPreparationConfig(
        maximum_groups_per_shard=2,
        cache_directory=tmp_path / "cache",
        checkpoint_directory=tmp_path / "checkpoint",
    )
    prepare_fixed_routing_sharded(
        inputs=assignment_inputs, theta=1.0, config=config
    )

    with pytest.raises(ValueError, match="manifest identity mismatch"):
        prepare_fixed_routing_sharded(
            inputs=assignment_inputs, theta=2.0, config=config
        )

    refreshed = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=2.0,
        config=config,
        cache_policy="refresh",
    )
    assert refreshed.status == "completed"


def test_sharded_preparation_two_workers_matches_serial_and_reuses_cache(
    assignment_inputs, tmp_path
):
    serial_config = FixedRoutingPreparationConfig(
        maximum_groups_per_shard=2,
        cache_directory=tmp_path / "serial-cache",
        checkpoint_directory=tmp_path / "serial-checkpoint",
    )
    parallel_config = FixedRoutingPreparationConfig(
        maximum_groups_per_shard=2,
        construction_workers=2,
        cache_directory=tmp_path / "parallel-cache",
        checkpoint_directory=tmp_path / "parallel-checkpoint",
    )
    serial = prepare_fixed_routing_sharded(
        inputs=assignment_inputs, theta=1.0, config=serial_config
    )
    events = []
    parallel = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=parallel_config,
        progress=events.append,
    )

    assert parallel.status == "completed"
    assert parallel.compilation_count == 1
    persisted_indices = [
        event.shard_index for event in events if event.phase == "shard_persisted"
    ]
    assert persisted_indices == sorted(persisted_indices)
    assert all(event.admitted_worker_count == 2 for event in events)
    assert parallel.routing.provenance.preparation_fingerprint == (
        serial.routing.provenance.preparation_fingerprint
    )
    for descriptor in serial.plan.descriptors:
        serial_shard = load_fixed_routing_shard(
            routing=serial.routing, descriptor=descriptor
        )
        parallel_shard = load_fixed_routing_shard(
            routing=parallel.routing, descriptor=descriptor
        )
        np.testing.assert_array_equal(
            parallel_shard.effective_group_link_mask,
            serial_shard.effective_group_link_mask,
        )
        np.testing.assert_allclose(
            parallel_shard.group_link_probability,
            serial_shard.group_link_probability,
            rtol=1.0e-6,
            atol=1.0e-7,
        )

    reload = prepare_fixed_routing_sharded(
        inputs=assignment_inputs, theta=1.0, config=parallel_config
    )
    assert reload.cache_hits == parallel.routing.num_shards
    assert reload.cache_misses == 0
    assert reload.compilation_count == 0


def test_parallel_detailed_profile_is_complete_ordered_and_cache_safe(
    assignment_inputs, tmp_path, monkeypatch
):
    original = sharded_module.save_fixed_routing_shard

    def delay_first(*, routing, shard, durable=True):
        # Force completion order away from canonical order without a polling
        # loop. The coordinator must still publish diagnostics canonically.
        if shard.descriptor.shard_index == 0:
            import time

            time.sleep(0.02)
        return original(routing=routing, shard=shard, durable=durable)

    monkeypatch.setattr(sharded_module, "save_fixed_routing_shard", delay_first)
    events = []
    config = FixedRoutingPreparationConfig(
        maximum_groups_per_shard=2,
        construction_workers=2,
        detailed_profiling=True,
        progress_interval_groups=1,
        cache_directory=tmp_path / "cache",
        checkpoint_directory=tmp_path / "checkpoint",
    )
    result = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=config,
        progress=events.append,
    )

    diagnostics = result.shard_diagnostics
    assert len(diagnostics) == result.routing.num_shards
    assert [item.shard_index for item in diagnostics] == list(
        range(result.routing.num_shards)
    )
    timing_names = (
        "host_destination_preparation_seconds",
        "host_mask_preparation_seconds",
        "argument_transfer_seconds",
        "kernel_execution_seconds",
        "device_synchronization_seconds",
        "host_transfer_seconds",
        "output_slicing_seconds",
        "validation_seconds",
        "shard_persistence_seconds",
        "manifest_persistence_seconds",
        "cleanup_seconds",
    )
    for diagnostic in diagnostics:
        phases = sum(getattr(diagnostic, name) for name in timing_names)
        assert all(getattr(diagnostic, name) >= 0.0 for name in timing_names)
        assert diagnostic.total_shard_seconds >= phases - 1.0e-6
        assert diagnostic.total_shard_seconds <= phases + 0.25
        assert diagnostic.input_shapes
        assert diagnostic.input_dtypes
        assert diagnostic.output_shapes
        assert diagnostic.output_dtypes
        assert diagnostic.retained_bytes > 0
        assert diagnostic.estimated_temporary_bytes > 0
        expected_enabled_fraction = (
            diagnostic.enabled_links / diagnostic.graph_links_traversed
            if diagnostic.graph_links_traversed
            else 0.0
        )
        retained_domain = diagnostic.num_groups * result.routing.num_links
        expected_probability_density = (
            diagnostic.probability_nonzeros / retained_domain
            if retained_domain
            else 0.0
        )
        assert diagnostic.enabled_link_fraction == pytest.approx(
            expected_enabled_fraction
        )
        assert diagnostic.probability_density == pytest.approx(
            expected_probability_density
        )

    planning = [event for event in events if event.phase == "planning_cache_scan"]
    dispatch = [event for event in events if event.phase == "dispatch"]
    persisted = [event for event in events if event.phase == "shard_persisted"]
    assert planning
    assert planning[0].cache_hits == 0
    assert planning[0].cache_misses == result.routing.num_shards
    assert planning[0].queued_shards == result.routing.num_shards
    assert planning[0].admitted_worker_count == 2
    assert dispatch
    assert all(event.current_shard_indices for event in dispatch)
    assert dispatch[0].current_shard_indices == (0, 1)
    assert dispatch[0].queued_shards == result.routing.num_shards - 2
    assert any(
        0 in event.current_shard_indices and 2 in event.current_shard_indices
        for event in dispatch[1:]
    )
    for event in events:
        assert event.current_shard_indices == tuple(
            sorted(event.current_shard_indices)
        )
        assert event.active_workers == len(event.current_shard_indices)
        assert event.remaining_shards == (
            event.total_shards - event.completed_shards
        )
        assert (
            event.completed_shards
            + event.active_workers
            + event.buffered_shards
            + event.queued_shards
            + event.failed_shards
            == event.total_shards
        )
    assert any(event.buffered_shards > 0 for event in events)
    assert persisted
    assert [event.shard_index for event in persisted] == sorted(
        event.shard_index for event in persisted
    )
    # The progress contract now includes one planning event and explicit
    # tracing/lowering/compilation phase observations before shard events.
    assert len(events) <= 3 * result.routing.num_shards + 10

    cached_events = []
    cached = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=config,
        progress=cached_events.append,
    )
    assert cached.cache_hits == result.routing.num_shards
    assert cached.cache_misses == 0
    assert cached.shard_diagnostics == ()
    assert [event.phase for event in cached_events] == [
        "planning",
        "planning_cache_scan",
        "terminal",
    ]
    assert all(
        event.completed_shards == event.total_shards
        and event.remaining_shards == 0
        and event.active_workers == 0
        and event.buffered_shards == 0
        and event.queued_shards == 0
        for event in cached_events
    )


def test_parallel_profile_disabled_has_no_diagnostics(assignment_inputs, tmp_path):
    result = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=2,
            construction_workers=2,
            detailed_profiling=False,
            cache_directory=tmp_path / "cache",
            checkpoint_directory=tmp_path / "checkpoint",
        ),
    )

    assert result.status == "completed"
    assert result.shard_diagnostics == ()


def _load_all_sharded_arrays(result):
    shards = [
        load_fixed_routing_shard(routing=result.routing, descriptor=descriptor)
        for descriptor in result.plan.descriptors
    ]
    return (
        np.concatenate(
            [shard.effective_group_link_mask for shard in shards], axis=0
        ),
        np.concatenate(
            [shard.group_link_probability for shard in shards], axis=0
        ),
    )


def test_progress_intervals_do_not_change_routing_plan_or_arrays(
    assignment_inputs, tmp_path
):
    first = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=1,
            progress_interval_seconds=0.01,
            progress_interval_groups=1,
            cache_directory=tmp_path / "first-cache",
            checkpoint_directory=tmp_path / "first-checkpoint",
        ),
    )
    second = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=1,
            progress_interval_seconds=10.0,
            progress_interval_groups=100,
            cache_directory=tmp_path / "second-cache",
            checkpoint_directory=tmp_path / "second-checkpoint",
        ),
    )

    assert first.plan.plan_fingerprint == second.plan.plan_fingerprint
    assert (
        first.routing.provenance.preparation_fingerprint
        == second.routing.provenance.preparation_fingerprint
    )
    first_manifest = json.loads(
        (tmp_path / "first-checkpoint" / "manifest.json").read_text()
    )
    second_manifest = json.loads(
        (tmp_path / "second-checkpoint" / "manifest.json").read_text()
    )
    assert first_manifest["progress_interval_seconds"] == 0.01
    assert first_manifest["progress_interval_groups"] == 1
    assert second_manifest["progress_interval_seconds"] == 10.0
    assert second_manifest["progress_interval_groups"] == 100
    first_mask, first_probability = _load_all_sharded_arrays(first)
    second_mask, second_probability = _load_all_sharded_arrays(second)
    np.testing.assert_array_equal(first_mask, second_mask)
    np.testing.assert_allclose(first_probability, second_probability)


def test_legacy_progress_sink_failure_does_not_abort_sharded_preparation(
    assignment_inputs, tmp_path
):
    def broken_sink(_event):
        raise OSError("telemetry sink unavailable")

    result = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=2,
            cache_directory=tmp_path / "cache",
            checkpoint_directory=tmp_path / "checkpoint",
        ),
        progress=broken_sink,
    )

    assert result.status == "completed"


def test_reporting_enabled_preserves_routing_artifacts_and_fingerprints(
    assignment_inputs, tmp_path
):
    common = {
        "maximum_groups_per_shard": 2,
    }
    enabled_config = FixedRoutingPreparationConfig(
        **common,
        cache_directory=tmp_path / "enabled-cache",
        checkpoint_directory=tmp_path / "enabled-checkpoint",
    )
    disabled_config = FixedRoutingPreparationConfig(
        **common,
        cache_directory=tmp_path / "disabled-cache",
        checkpoint_directory=tmp_path / "disabled-checkpoint",
    )
    events = []
    enabled = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=enabled_config,
        progress=events.append,
    )
    disabled = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=disabled_config,
    )

    assert events
    assert enabled.plan.plan_fingerprint == disabled.plan.plan_fingerprint
    assert (
        enabled.routing.provenance.preparation_fingerprint
        == disabled.routing.provenance.preparation_fingerprint
    )
    for enabled_descriptor, disabled_descriptor in zip(
        enabled.plan.descriptors, disabled.plan.descriptors, strict=True
    ):
        enabled_path = fixed_routing_shard_path(enabled.routing, enabled_descriptor)
        disabled_path = fixed_routing_shard_path(disabled.routing, disabled_descriptor)
        assert enabled_path.read_bytes() == disabled_path.read_bytes()


def test_batched_shards_match_serial_with_partial_final_batch(
    assignment_inputs, tmp_path
):
    serial = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=1,
            cache_directory=tmp_path / "serial-cache",
            checkpoint_directory=tmp_path / "serial-checkpoint",
        ),
    )
    events = []
    batched = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=1,
            execution_strategy="batched",
            shards_per_execution_batch=4,
            detailed_profiling=True,
            cache_directory=tmp_path / "batched-cache",
            checkpoint_directory=tmp_path / "batched-checkpoint",
        ),
        progress=events.append,
    )

    serial_mask, serial_probability = _load_all_sharded_arrays(serial)
    batch_mask, batch_probability = _load_all_sharded_arrays(batched)
    np.testing.assert_array_equal(batch_mask, serial_mask)
    np.testing.assert_allclose(
        batch_probability, serial_probability, rtol=1.0e-6, atol=1.0e-7
    )
    assert batched.execution_strategy == "batched"
    assert batched.compilation_count == 1
    assert batched.batch_diagnostics
    assert batched.batch_diagnostics[-1].batch_size < 4
    assert batched.batch_diagnostics[-1].padded_batch_size == 4
    assert all(item.effective_cpu_cores >= 0.0 for item in batched.batch_diagnostics)
    assert any(event.phase == "batch_completed" for event in events)
    assert events[-1].phase == "terminal"


def test_thread_and_batched_strategies_reuse_each_others_shards(
    assignment_inputs, tmp_path
):
    common = {
        "maximum_groups_per_shard": 1,
        "cache_directory": tmp_path / "cache",
        "checkpoint_directory": tmp_path / "checkpoint",
    }
    threaded = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(construction_workers=2, **common),
    )
    batched_hit = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            execution_strategy="batched",
            shards_per_execution_batch=3,
            **common,
        ),
    )
    assert batched_hit.cache_hits == threaded.routing.num_shards
    assert (
        batched_hit.routing.provenance.preparation_fingerprint
        == threaded.routing.provenance.preparation_fingerprint
    )
    assert batched_hit.compiled_kernel_identity != threaded.compiled_kernel_identity

    reverse = dict(common)
    reverse["cache_directory"] = tmp_path / "reverse-cache"
    reverse["checkpoint_directory"] = tmp_path / "reverse-checkpoint"
    batched = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            execution_strategy="batched",
            shards_per_execution_batch=3,
            **reverse,
        ),
    )
    threaded_hit = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(construction_workers=2, **reverse),
    )
    assert threaded_hit.cache_hits == batched.routing.num_shards


def test_batched_deadline_and_temporary_memory_admission(
    assignment_inputs, tmp_path
):
    deadline = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=1,
            execution_strategy="batched",
            shards_per_execution_batch=3,
            initial_predicted_shard_seconds=10.0,
            dispatch_safety_margin_seconds=1.0,
            cache_directory=tmp_path / "deadline-cache",
            checkpoint_directory=tmp_path / "deadline-checkpoint",
        ),
        absolute_deadline=sharded_module.perf_counter() + 2.0,
    )
    assert deadline.status == "deadline_reached"
    assert deadline.completed_shards == 0
    assert deadline.dispatch_prevented_by_deadline
    assert deadline.deadline_phase == "predictive batch dispatch guard"

    one_group_temporary = (
        assignment_inputs.graph.num_links
        * (
            np.dtype(bool).itemsize
            + np.dtype(assignment_inputs.base_link_cost.dtype).itemsize
        )
        * 2
    )
    memory = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=1,
            maximum_temporary_bytes=one_group_temporary,
            execution_strategy="batched",
            shards_per_execution_batch=2,
            cache_directory=tmp_path / "memory-cache",
            checkpoint_directory=tmp_path / "memory-checkpoint",
        ),
    )
    assert memory.status == "memory_budget_reached"
    assert memory.completed_shards == 0


def test_batched_resume_after_partial_persistence(
    assignment_inputs, tmp_path, monkeypatch
):
    config = FixedRoutingPreparationConfig(
        maximum_groups_per_shard=1,
        execution_strategy="batched",
        shards_per_execution_batch=3,
        cache_directory=tmp_path / "cache",
        checkpoint_directory=tmp_path / "checkpoint",
    )
    original = sharded_module.save_fixed_routing_shard
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected persistence failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(sharded_module, "save_fixed_routing_shard", fail_second)
    with pytest.raises(OSError, match="injected persistence failure"):
        prepare_fixed_routing_sharded(
            inputs=assignment_inputs, theta=1.0, config=config
        )
    monkeypatch.setattr(sharded_module, "save_fixed_routing_shard", original)

    resumed = prepare_fixed_routing_sharded(
        inputs=assignment_inputs, theta=1.0, config=config
    )
    assert resumed.status == "completed"
    assert resumed.cache_hits == 1
    assert resumed.completed_shards == resumed.routing.num_shards


def test_worker_recommendation_requires_throughput_evidence(
    assignment_inputs, tmp_path
):
    plan = plan_fixed_routing_shards(
        inputs=assignment_inputs,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=1,
            cache_directory=tmp_path / "cache",
            checkpoint_directory=tmp_path / "checkpoint",
        ),
    )
    conservative = recommend_fixed_routing_workers(
        plan=plan,
        available_ram_bytes=8 * 1024**3,
        cpu_count=16,
        server=True,
    )
    assert conservative.workers == 1
    assert conservative.throughput_effective_workers is None
    assert conservative.memory_admissible_workers >= 1
    assert conservative.cpu_admissible_workers == 8

    calibrated = recommend_fixed_routing_workers(
        plan=plan,
        available_ram_bytes=8 * 1024**3,
        cpu_count=16,
        server=True,
        measured_throughput_by_workers={1: 1.0, 2: 1.8, 4: 1.7},
        measured_throughput_by_batch_size={1: 1.0, 2: 1.4, 4: 1.2},
    )
    assert calibrated.workers == 2
    assert calibrated.throughput_effective_workers == 2
    assert calibrated.recommended_batch_size == 2


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_detailed_profile_supports_requested_worker_counts(
    assignment_inputs, tmp_path, workers
):
    result = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=1,
            construction_workers=workers,
            detailed_profiling=True,
            cache_directory=tmp_path / f"cache-{workers}",
            checkpoint_directory=tmp_path / f"checkpoint-{workers}",
        ),
    )

    assert result.status == "completed"
    assert result.admitted_worker_count == workers
    assert len(result.shard_diagnostics) == result.routing.num_shards
    assert all(
        diagnostic.concurrent_jax_executions >= 1
        for diagnostic in result.shard_diagnostics
    )
    assert result.throughput_effective_worker_count is None


def test_parallel_predictive_deadline_guard_prevents_dispatch(
    assignment_inputs, tmp_path
):
    events = []
    result = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=2,
            construction_workers=2,
            initial_predicted_shard_seconds=60.0,
            dispatch_safety_margin_seconds=10.0,
            cache_directory=tmp_path / "cache",
            checkpoint_directory=tmp_path / "checkpoint",
        ),
        absolute_deadline=sharded_module.perf_counter() + 1.0,
        progress=events.append,
    )

    assert result.status == "deadline_reached"
    assert result.completed_shards == 0
    assert result.dispatch_prevented_by_deadline
    assert result.deadline_phase == "predictive dispatch guard"
    assert not list((tmp_path / "cache").glob("routing-shard-*.npz"))
    guard = [event for event in events if event.phase == "predictive_dispatch_guard"]
    assert len(guard) == 1
    assert guard[0].active_workers == 0
    assert guard[0].queued_shards == result.routing.num_shards
    assert events[-1].phase == "terminal"


def test_parallel_memory_admission_can_decline_before_compilation(
    assignment_inputs, tmp_path
):
    result = prepare_fixed_routing_sharded(
        inputs=assignment_inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            construction_workers=2,
            maximum_process_rss_bytes=1,
            cache_directory=tmp_path / "cache",
            checkpoint_directory=tmp_path / "checkpoint",
        ),
    )

    assert result.status == "memory_budget_reached"
    assert result.completed_shards == 0
    assert result.compilation_count == 0


def test_parallel_worker_failure_preserves_valid_artifacts_and_manifest(
    assignment_inputs, tmp_path, monkeypatch
):
    original = sharded_module.save_fixed_routing_shard

    def fail_second(*, routing, shard, durable=True):
        if shard.descriptor.shard_index == 2:
            raise OSError("simulated worker persistence failure")
        return original(routing=routing, shard=shard, durable=durable)

    monkeypatch.setattr(sharded_module, "save_fixed_routing_shard", fail_second)
    config = FixedRoutingPreparationConfig(
        maximum_groups_per_shard=2,
        construction_workers=2,
        cache_directory=tmp_path / "cache",
        checkpoint_directory=tmp_path / "checkpoint",
    )
    events = []

    with pytest.raises(RuntimeError, match="worker failed for shard 2"):
        prepare_fixed_routing_sharded(
            inputs=assignment_inputs,
            theta=1.0,
            config=config,
            progress=events.append,
        )

    failure = [event for event in events if event.phase == "worker_failed"]
    assert len(failure) == 1
    assert failure[0].failed_shards == 1
    assert (
        failure[0].completed_shards
        + failure[0].active_workers
        + failure[0].buffered_shards
        + failure[0].queued_shards
        + failure[0].failed_shards
        == failure[0].total_shards
    )
    assert events[-1].phase == "terminal"

    manifest = json.loads(
        (tmp_path / "checkpoint" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "interrupted"
    assert not list((tmp_path / "cache").glob("*.tmp"))
    monkeypatch.setattr(sharded_module, "save_fixed_routing_shard", original)
    resumed = prepare_fixed_routing_sharded(
        inputs=assignment_inputs, theta=1.0, config=config
    )
    assert resumed.status == "completed"
    assert resumed.cache_hits >= 2


def test_fixed_routing_depends_on_theta_and_is_deterministic(assignment_inputs):
    first = prepare_fixed_routing(inputs=assignment_inputs, theta=0.5)
    repeated = prepare_fixed_routing(inputs=assignment_inputs, theta=0.5)
    different = prepare_fixed_routing(inputs=assignment_inputs, theta=5.0)

    np.testing.assert_array_equal(
        first.group_link_probability,
        repeated.group_link_probability,
    )
    assert not np.allclose(
        np.asarray(first.group_link_probability),
        np.asarray(different.group_link_probability),
    )


def test_prepare_fixed_routing_supports_no_active_destination_groups(assignment_inputs):
    empty = replace(
        assignment_inputs,
        group_dest_node=jnp.empty((0,), dtype=jnp.int32),
        group_link_mask=jnp.empty(
            (0, assignment_inputs.graph.num_links),
            dtype=bool,
        ),
    )
    prepared = prepare_fixed_routing(inputs=empty, theta=1.0)

    assert prepared.group_dest_node.shape == (0,)
    assert prepared.effective_group_link_mask.shape == (
        0,
        assignment_inputs.graph.num_links,
    )
    assert prepared.group_link_probability.shape == (
        0,
        assignment_inputs.graph.num_links,
    )
def test_cached_probabilities_match_independent_destination_routing(assignment_inputs):
    prepared = prepare_fixed_routing(inputs=assignment_inputs, theta=1.0)
    group_index = 0
    destination = assignment_inputs.group_dest_node[group_index]
    enabled, cost = _routing_inputs_for_destination(
        graph=assignment_inputs.graph,
        base_link_cost=assignment_inputs.base_link_cost,
        group_link_mask=assignment_inputs.group_link_mask[group_index],
        dest_node=destination,
    )
    independent = prepare_destination_routing(
        graph=assignment_inputs.graph,
        link_cost=cost,
        enabled_link_mask=enabled,
        dest_node=destination,
        theta=1.0,
    )

    np.testing.assert_array_equal(
        prepared.effective_group_link_mask[group_index],
        independent.enabled_link_mask,
    )
    np.testing.assert_allclose(
        prepared.group_link_probability[group_index],
        independent.link_prob,
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_fixed_routing_rejects_incompatible_assignment_inputs(assignment_inputs):
    prepared = prepare_fixed_routing(inputs=assignment_inputs, theta=1.0)
    changed_cost = replace(
        assignment_inputs,
        base_link_cost=assignment_inputs.base_link_cost.at[0].add(1.0),
    )

    with pytest.raises(ValueError, match="base link costs"):
        validate_fixed_routing_compatibility(inputs=changed_cost, routing=prepared)


@pytest.mark.parametrize("scale", [0.0, 0.4, 1.0, 2.5])
def test_fixed_routing_loading_matches_dynamic_assignment(assignment_inputs, scale):
    theta = 1.0
    routing = prepare_fixed_routing(inputs=assignment_inputs, theta=theta)
    demand = jnp.linspace(
        0.0,
        20.0 * scale,
        assignment_inputs.od_origin_node.shape[0],
        dtype=jnp.float32,
    )

    dynamic = assign_link_flow(
        inputs=assignment_inputs,
        f=demand,
        theta=jnp.asarray(theta),
    )
    cached = assign_link_flow_fixed_routing(
        inputs=assignment_inputs,
        routing=routing,
        f=demand,
    )

    np.testing.assert_allclose(cached, dynamic, rtol=2.0e-6, atol=2.0e-6)


def test_fixed_routing_loading_gradient_matches_dynamic_assignment(assignment_inputs):
    theta = 1.0
    routing = prepare_fixed_routing(inputs=assignment_inputs, theta=theta)
    demand = jnp.linspace(
        0.1,
        12.0,
        assignment_inputs.od_origin_node.shape[0],
        dtype=jnp.float32,
    )

    def dynamic_objective(value):
        flow = assign_link_flow(
            inputs=assignment_inputs,
            f=value,
            theta=jnp.asarray(theta),
        )
        return jnp.square(flow).sum()

    def cached_objective(value):
        flow = assign_link_flow_fixed_routing(
            inputs=assignment_inputs,
            routing=routing,
            f=value,
        )
        return jnp.square(flow).sum()

    dynamic_gradient = jax.grad(dynamic_objective)(demand)
    cached_gradient = jax.grad(cached_objective)(demand)
    np.testing.assert_allclose(
        cached_gradient,
        dynamic_gradient,
        rtol=3.0e-5,
        atol=3.0e-5,
    )


def test_fixed_routing_custom_adjoint_matches_ordinary_autodiff(assignment_inputs):
    routing = prepare_fixed_routing(inputs=assignment_inputs, theta=1.0)
    demand = jnp.linspace(
        0.1,
        12.0,
        assignment_inputs.od_origin_node.shape[0],
        dtype=jnp.float32,
    )
    weight = jnp.sin(
        jnp.arange(assignment_inputs.graph.num_links, dtype=jnp.float32) * 0.13
    )

    def ordinary(value):
        return jnp.vdot(
            assign_link_flow_fixed_routing(
                inputs=assignment_inputs, routing=routing, f=value
            ),
            weight,
        )

    def explicit(value):
        return jnp.vdot(
            assign_link_flow_fixed_routing_custom_adjoint(
                inputs=assignment_inputs, routing=routing, f=value
            ),
            weight,
        )

    ordinary_value, ordinary_gradient = jax.value_and_grad(ordinary)(demand)
    explicit_value, explicit_gradient = jax.value_and_grad(explicit)(demand)
    np.testing.assert_allclose(explicit_value, ordinary_value, rtol=0, atol=0)
    np.testing.assert_allclose(
        explicit_gradient, ordinary_gradient, rtol=2e-5, atol=2e-5
    )


@pytest.mark.parametrize("theta", [0.0, -1.0, np.inf, np.nan])
def test_prepare_fixed_routing_rejects_invalid_theta(assignment_inputs, theta):
    with pytest.raises(ValueError, match="positive and finite"):
        prepare_fixed_routing(inputs=assignment_inputs, theta=theta)
