from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.domain import Scenario
from public_transportation.inference.assignment_adapter import (
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.block_coordinate.support_preflight import (
    SupportPreflightBudget,
    SupportPreflightConfig,
    SupportPreflightFingerprints,
    SupportPreflightMode,
    SupportPreflightStatus,
    SupportPreflightStopLocation,
    authorize_block_coordinate_pilot,
    load_support_preflight_checkpoint,
    run_support_preflight,
)
from public_transportation.inference.block_coordinate.config import BlockSizingConfig
from public_transportation.inference.block_coordinate.fixed_routing_selected_block_builder import (
    FixedRoutingSelectedBlockBuilder,
    SelectedBlockBuilderConfig,
    SelectedBlockBuilderProvenance,
    SelectedBlockConstructionDeadlineError,
    SelectedBlockDiagnosticStop,
    SelectedBlockJSONLProgressSink,
)
from public_transportation.inference.block_coordinate.operator import (
    SparseBlockLinearOperator,
)
from public_transportation.inference.block_coordinate.partition import (
    partition_assignment_od_blocks,
)
from public_transportation.inference.block_coordinate.selected_blocks import (
    BlockConstructionResourceError,
    construct_selected_block_operators,
    select_representative_block_ids,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.fixed_routing_origin_support import (
    OriginSupportConfig,
    analyze_fixed_routing_origin_support,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    prepare_fixed_routing_measurement_operator,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout
from public_transportation.measurement.mapping import AggregationSpec

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


class StepClock:
    def __init__(self, step: float = 0.01):
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


def deterministic_summary(summary):
    return (
        summary.group_id,
        summary.free_columns,
        summary.positive_fixed_columns,
        summary.measurement_support_rows,
        summary.exact_nonzeros,
        summary.unique_support_patterns,
        summary.estimated_temporary_bytes,
    )


def completed_preflight(setup):
    inputs, compact, spec, fingerprints, partition = setup
    result = run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=SupportPreflightConfig(
            mode=SupportPreflightMode.STREAMING_EXACT_SUPPORT
        ),
    )
    return inputs, compact, spec, fingerprints, partition, result


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    directory = tmp_path_factory.mktemp("support-preflight")
    for name in NETWORK_FILES:
        shutil.copy2(EXAMPLE / "data" / name, directory / name)
    shutil.copy2(
        EXAMPLE / "pre_processing/results/demand.csv", directory / "demand.csv"
    )
    scenario = Scenario.from_folder(directory, strict=True)
    artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
    full_inputs = build_assignment_inputs(artifacts=artifacts)
    count = int(full_inputs.od_origin_node.shape[0])
    layout = ODParameterLayout(
        num_od_total=count,
        od_keys=tuple((f"o{i}", "d", "t") for i in range(count)),
        free_od_indices=tuple(range(count)),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=tuple(1.0 for _ in range(count)),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    links = np.arange(min(12, int(inputs.graph.num_links)), dtype=np.int32)
    spec = AggregationSpec(
        num_measurements=4,
        measurement_index=np.arange(links.size, dtype=np.int32) % 4,
        link_index=links,
    )
    fingerprints = SupportPreflightFingerprints(
        scenario="scenario",
        assignment_inputs="assignment",
        od_layout=layout.fingerprint,
        fixed_demand="fixed",
        measurement_mapping="mapping",
        routing="theta-1",
        partition="all-free",
    )
    partition = partition_assignment_od_blocks(
        inputs=inputs,
        parameter_layout=layout,
        compact_layout=compact,
        sizing=BlockSizingConfig(mode="explicit", maximum_free_variables_per_block=2),
    )
    fingerprints = replace(fingerprints, partition=partition.fingerprint)
    return inputs, compact, spec, fingerprints, partition


def test_configuration_is_validated_and_fingerprinted():
    first = SupportPreflightConfig(sample_count=4, sampling_seed=3)
    second = SupportPreflightConfig(sample_count=4, sampling_seed=3)
    assert first.fingerprint == second.fingerprint
    changed_allowance = replace(
        first,
        budget=replace(first.budget, maximum_elapsed_seconds=7200.0),
    )
    assert changed_allowance.semantics_fingerprint == first.semantics_fingerprint
    assert changed_allowance.policy_fingerprint != first.policy_fingerprint
    assert changed_allowance.invocation_policy.budget.maximum_elapsed_seconds == 7200.0
    assert (
        replace(first, sampling_seed=4).semantics_fingerprint
        != first.semantics_fingerprint
    )
    with pytest.raises(ValueError, match="explicit authorization"):
        SupportPreflightConfig(mode=SupportPreflightMode.EXACT_MATERIALIZED_PLAN)
    with pytest.raises(ValueError, match="positive"):
        SupportPreflightBudget(maximum_temporary_bytes=0)


def test_streaming_exact_support_matches_materialized_small_example(setup):
    inputs, compact, spec, fingerprints, partition = setup
    result = run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=SupportPreflightConfig(
            mode=SupportPreflightMode.STREAMING_EXACT_SUPPORT
        ),
    )
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    materialized = analyze_fixed_routing_origin_support(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        config=OriginSupportConfig(origin_chunk_size=32, materialize=True),
    )
    assert result.status is SupportPreflightStatus.COMPLETED
    assert (
        sum(item.exact_nonzeros for item in result.destination_summaries)
        == materialized.metrics.origin_specific_entries
    )
    assert result.processed_free_columns == compact.num_free
    assert result.retained_state_bytes < 100_000


def test_timing_sink_failure_does_not_abort_support_analysis(setup):
    inputs, compact, spec, _fingerprints, _partition = setup
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)

    def broken_timing_sink(_timing):
        raise OSError("telemetry sink unavailable")

    result = analyze_fixed_routing_origin_support(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        config=OriginSupportConfig(origin_chunk_size=32, materialize=False),
        timing_callback=broken_timing_sink,
    )

    assert result.summaries


def test_interrupt_checkpoint_resume_is_exact(setup, tmp_path):
    inputs, compact, spec, fingerprints, partition = setup
    config = SupportPreflightConfig(
        mode=SupportPreflightMode.STREAMING_EXACT_SUPPORT,
        checkpoint_directory=tmp_path,
        checkpoint_interval_groups=1,
    )
    calls = 0

    def interrupt(_event):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt

    partial = run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=config,
        progress_callback=interrupt,
    )
    assert partial.status is SupportPreflightStatus.INTERRUPTED
    resumed = run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=config,
        resume=True,
    )
    uninterrupted = run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=replace(config, checkpoint_directory=None),
    )
    assert resumed.complete
    assert [deterministic_summary(item) for item in resumed.destination_summaries] == [
        deterministic_summary(item) for item in uninterrupted.destination_summaries
    ]
    assert len(resumed.completed_destination_groups) == len(
        set(resumed.completed_destination_groups)
    )
    incompatible = replace(fingerprints, routing="other-theta")
    with pytest.raises(ValueError, match="incompatible"):
        load_support_preflight_checkpoint(
            tmp_path, fingerprints=incompatible, config=config
        )


def test_rss_guard_returns_partial_result(setup):
    inputs, compact, spec, fingerprints, partition = setup
    result = run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=SupportPreflightConfig(
            mode=SupportPreflightMode.STREAMING_EXACT_SUPPORT,
            budget=SupportPreflightBudget(maximum_process_rss_bytes=100),
        ),
        resource_observer=lambda: 100,
    )
    assert result.status is SupportPreflightStatus.STOPPED_RSS
    assert not result.completed_destination_groups
    assert not authorize_block_coordinate_pilot(result).accepted


@pytest.mark.parametrize(
    ("budget", "status"),
    [
        (
            SupportPreflightBudget(maximum_elapsed_seconds=1.0e-12),
            SupportPreflightStatus.STOPPED_TIME,
        ),
        (
            SupportPreflightBudget(maximum_temporary_bytes=1),
            SupportPreflightStatus.RESOURCE_GUARD,
        ),
        (
            SupportPreflightBudget(maximum_retained_support_bytes=1),
            SupportPreflightStatus.STOPPED_RETAINED,
        ),
    ],
)
def test_resource_limits_return_valid_partial_results(setup, budget, status):
    inputs, compact, spec, fingerprints, partition = setup
    result = run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=SupportPreflightConfig(
            mode=SupportPreflightMode.STREAMING_EXACT_SUPPORT,
            budget=budget,
        ),
    )
    assert result.status is status
    assert result.reason


def test_sampling_is_deterministic_and_not_pilot_authorization(setup):
    inputs, compact, spec, fingerprints, partition = setup
    config = SupportPreflightConfig(
        mode=SupportPreflightMode.SAMPLED_EXACT_SUPPORT,
        sample_count=4,
        sampling_seed=27,
    )
    first = run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=config,
    )
    second = run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=config,
    )
    assert first.selected_destination_groups == second.selected_destination_groups
    assert first.extrapolation is not None
    assert second.extrapolation is not None
    assert (
        first.extrapolation.nonzero_count_range
        == second.extrapolation.nonzero_count_range
    )
    assert (
        first.extrapolation.storage_shards_range
        == second.extrapolation.storage_shards_range
    )
    assert not first.full_network_coverage
    assert not authorize_block_coordinate_pilot(first).accepted


def test_corrupt_checkpoint_is_rejected(setup, tmp_path):
    _inputs, _compact, _spec, fingerprints, _partition = setup
    (tmp_path / "support-preflight.json").write_text("{incomplete", encoding="utf-8")
    config = SupportPreflightConfig(checkpoint_directory=tmp_path)
    with pytest.raises(ValueError, match="corrupt"):
        load_support_preflight_checkpoint(
            tmp_path, fingerprints=fingerprints, config=config
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("scenario", "changed-scenario"),
        ("measurement_mapping", "changed-measurements"),
        ("routing", "changed-routing"),
        ("od_layout", "changed-layout"),
        ("partition", "changed-partition"),
    ],
)
def test_authoritative_fingerprint_changes_reject_resume(
    setup, tmp_path, field, replacement
):
    inputs, compact, spec, fingerprints, partition = setup
    config = SupportPreflightConfig(
        mode=SupportPreflightMode.STREAMING_EXACT_SUPPORT,
        destination_group_ids=(0,),
        checkpoint_directory=tmp_path,
    )
    run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=config,
    )
    with pytest.raises(ValueError, match="fingerprints are incompatible"):
        load_support_preflight_checkpoint(
            tmp_path,
            fingerprints=replace(fingerprints, **{field: replacement}),
            config=config,
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"sampling_seed": 91},
        {"destination_group_ids": (1,)},
        {"origin_chunk_size": 1},
        {"probability_tolerance": 1.0e-8},
    ],
)
def test_semantic_configuration_changes_reject_resume(setup, tmp_path, changed):
    inputs, compact, spec, fingerprints, partition = setup
    config = SupportPreflightConfig(
        mode=SupportPreflightMode.STREAMING_EXACT_SUPPORT,
        destination_group_ids=(0,),
        checkpoint_directory=tmp_path,
    )
    run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=config,
    )
    incompatible = replace(config, **changed)
    with pytest.raises(ValueError, match="semantic configuration"):
        load_support_preflight_checkpoint(
            tmp_path, fingerprints=fingerprints, config=incompatible
        )


def test_obsolete_checkpoint_schema_has_precise_error(setup, tmp_path):
    _inputs, _compact, _spec, fingerprints, _partition = setup
    (tmp_path / "support-preflight.json").write_text(
        '{"schema_version": 1}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema 1 cannot be resumed safely"):
        load_support_preflight_checkpoint(
            tmp_path,
            fingerprints=fingerprints,
            config=SupportPreflightConfig(checkpoint_directory=tmp_path),
        )


def test_short_time_budget_resumes_until_exact_completion(setup, tmp_path):
    inputs, compact, spec, fingerprints, partition = setup
    selected = (0, 1, 2)
    config = SupportPreflightConfig(
        mode=SupportPreflightMode.STREAMING_EXACT_SUPPORT,
        destination_group_ids=selected,
        checkpoint_directory=tmp_path,
        checkpoint_interval_groups=1,
        budget=SupportPreflightBudget(maximum_elapsed_seconds=0.06),
    )
    results = []
    resume = False
    for invocation in range(1, 8):
        policy = replace(
            config,
            budget=replace(
                config.budget,
                maximum_elapsed_seconds=0.06 + invocation * 0.001,
            ),
        )
        result = run_support_preflight(
            inputs=inputs,
            theta=1.0,
            spec=spec,
            compact_layout=compact,
            partition=partition,
            fingerprints=fingerprints,
            config=policy,
            resume=resume,
            clock=StepClock(),
        )
        results.append(result)
        assert result.invocation_count == invocation
        assert result.semantic_config_fingerprint == config.semantics_fingerprint
        assert result.policy_fingerprint == policy.policy_fingerprint
        assert result.invocation_policy == policy.invocation_policy
        assert (
            result.cumulative_elapsed_seconds
            >= result.previous_invocations_elapsed_seconds
        )
        assert result.cumulative_elapsed_seconds == pytest.approx(
            result.previous_invocations_elapsed_seconds
            + result.current_invocation_elapsed_seconds
        )
        assert result.invocation_allowance_overshoot_seconds >= 0.0
        if result.complete:
            break
        assert result.status is SupportPreflightStatus.STOPPED_TIME
        assert result.stop_location in {
            SupportPreflightStopLocation.BEFORE_GROUP,
            SupportPreflightStopLocation.INSIDE_CHUNK,
        }
        if len(results) > 1:
            assert result.previous_invocations_elapsed_seconds == pytest.approx(
                results[-2].cumulative_elapsed_seconds
            )
            assert len(result.completed_destination_groups) > len(
                results[-2].completed_destination_groups
            )
        resume = True
    assert results[-1].complete
    assert results[-1].completed_destination_groups == selected
    assert all(
        later.cumulative_elapsed_seconds >= earlier.cumulative_elapsed_seconds
        for earlier, later in zip(results, results[1:])
    )

    uninterrupted = run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=replace(
            config,
            checkpoint_directory=None,
            budget=replace(config.budget, maximum_elapsed_seconds=10.0),
        ),
        clock=StepClock(),
    )
    assert [
        deterministic_summary(item) for item in results[-1].destination_summaries
    ] == [deterministic_summary(item) for item in uninterrupted.destination_summaries]
    assert results[-1].block_summaries == uninterrupted.block_summaries


def test_inside_chunk_stop_discards_uncommitted_group(setup, tmp_path):
    inputs, compact, spec, fingerprints, partition = setup
    result = run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=SupportPreflightConfig(
            mode=SupportPreflightMode.STREAMING_EXACT_SUPPORT,
            destination_group_ids=(0,),
            checkpoint_directory=tmp_path,
            budget=SupportPreflightBudget(maximum_elapsed_seconds=0.02),
        ),
        clock=StepClock(),
    )
    assert result.status is SupportPreflightStatus.STOPPED_TIME
    assert result.stop_location is SupportPreflightStopLocation.INSIDE_CHUNK
    assert result.stop_group_id == 0
    assert not result.completed_destination_groups
    assert result.pending_destination_groups == (0,)
    assert not result.destination_summaries
    assert result.discarded_partial_group_seconds > 0.0


def test_resume_policy_can_expand_but_cannot_invalidate_persisted_state(
    setup, tmp_path
):
    inputs, compact, spec, fingerprints, partition = setup
    config = SupportPreflightConfig(
        mode=SupportPreflightMode.STREAMING_EXACT_SUPPORT,
        destination_group_ids=(0, 1),
        checkpoint_directory=tmp_path,
        checkpoint_interval_groups=1,
    )
    calls = 0

    def interrupt(_event):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt

    partial = run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=config,
        progress_callback=interrupt,
    )
    assert partial.completed_destination_groups == (0,)
    expanded = replace(
        config,
        construction_workers=2,
        budget=replace(
            config.budget,
            maximum_elapsed_seconds=7200.0,
            maximum_process_rss_bytes=32 * 1024**3,
            maximum_temporary_bytes=1024**3,
            maximum_retained_support_bytes=128 * 1024**2,
        ),
    )
    resumed = run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=expanded,
        resume=True,
    )
    assert resumed.complete
    assert resumed.invocation_count == 2
    assert resumed.completed_destination_groups == (0, 1)

    with pytest.raises(ValueError, match="retained state"):
        load_support_preflight_checkpoint(
            tmp_path,
            fingerprints=fingerprints,
            config=replace(
                config,
                budget=replace(
                    config.budget,
                    maximum_retained_support_bytes=1,
                ),
            ),
        )
    with pytest.raises(ValueError, match="operator-size limit"):
        load_support_preflight_checkpoint(
            tmp_path,
            fingerprints=fingerprints,
            config=replace(
                config,
                budget=replace(config.budget, maximum_block_operator_bytes=1),
            ),
        )


def test_selected_blocks_are_measured_and_rejected_before_allocation(setup):
    inputs, compact, spec, fingerprints, partition = setup
    result = run_support_preflight(
        inputs=inputs,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        fingerprints=fingerprints,
        config=SupportPreflightConfig(
            mode=SupportPreflightMode.STREAMING_EXACT_SUPPORT
        ),
    )
    selected = select_representative_block_ids(result)
    assert selected
    matrix = np.arange(spec.num_measurements * compact.num_free, dtype=float).reshape(
        spec.num_measurements, compact.num_free
    )

    def build(block):
        return SparseBlockLinearOperator(matrix[:, block.free_column_indices])

    measured = construct_selected_block_operators(
        result=result,
        partition=partition,
        builder=build,
        budget=SupportPreflightBudget(),
    )
    assert tuple(item.block_id for item in measured) == selected
    for item in measured:
        block = next(
            value for value in partition.blocks if value.block_id == item.block_id
        )
        vector = np.linspace(0.25, 1.25, block.num_free_variables)
        expected = matrix[:, block.free_column_indices] @ vector
        assert item.forward_checksum == pytest.approx(float(np.sum(expected)))
        cotangent = np.linspace(-0.5, 0.5, spec.num_measurements)
        expected_transpose = matrix[:, block.free_column_indices].T @ cotangent
        assert item.transpose_checksum == pytest.approx(
            float(np.sum(expected_transpose))
        )

    calls = 0

    def forbidden(_block):
        nonlocal calls
        calls += 1
        raise AssertionError("builder must not be called")

    with pytest.raises(BlockConstructionResourceError):
        construct_selected_block_operators(
            result=result,
            partition=partition,
            builder=forbidden,
            budget=SupportPreflightBudget(maximum_temporary_bytes=1),
        )
    assert calls == 0


def test_production_selected_block_matches_complete_operator_and_reuses_cache(
    setup, tmp_path, monkeypatch
):
    inputs, compact, spec, fingerprints, partition, preflight = completed_preflight(
        setup
    )
    summary = max(preflight.block_summaries, key=lambda item: item.exact_nonzeros)
    block = next(item for item in partition.blocks if item.block_id == summary.block_id)
    config = SelectedBlockBuilderConfig(
        cache_directory=tmp_path / "cache",
        support_directory=tmp_path / "support",
        od_chunk_size=2,
        measurement_chunk_size=2,
        mapped_edge_chunk_size=3,
    )
    provenance = SelectedBlockBuilderProvenance(
        fingerprints=fingerprints,
        semantic_preflight_fingerprint=SupportPreflightConfig().semantics_fingerprint,
        theta=1.0,
    )
    selected_progress = []
    builder = FixedRoutingSelectedBlockBuilder(
        inputs=inputs,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        provenance=provenance,
        config=config,
        progress=selected_progress.append,
    )
    import public_transportation.inference.block_coordinate.fixed_routing_selected_block_builder as builder_module

    original_zeros = builder_module.np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        if shape == (spec.num_measurements, block.num_free_variables):
            raise AssertionError("full-height dense block allocation")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(builder_module.np, "zeros", guarded_zeros)
    cold = builder.build_result(block)
    monkeypatch.undo()
    assert not cold.cache_hit
    assert cold.operator.shape == (spec.num_measurements, block.num_free_variables)
    assert selected_progress[-1].status == "completed"
    assert selected_progress[-1].predicted_remaining_seconds == 0.0
    first_completed = next(
        item for item in selected_progress if item.completed_od_chunks > 0
    )
    assert first_completed.predicted_remaining_seconds is not None
    assert set(cold.operator.measurement_support_indices).issubset(
        cold.support_artifact.support_rows
    )
    assert set(block.active_od_indices).isdisjoint(compact.fixed_compact_indices)

    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    complete = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint=fingerprints.assignment_inputs,
        compact_layout=compact,
        representation="dense",
        chunk_size=2,
    )
    reference = np.asarray(complete.matrix)[:, block.free_column_indices]
    vector = np.linspace(0.25, 1.25, block.num_free_variables)
    cotangent = np.linspace(-0.5, 0.5, spec.num_measurements)
    np.testing.assert_allclose(cold.operator.matvec(vector), reference @ vector)
    np.testing.assert_allclose(
        cold.operator.rmatvec(cotangent), reference.T @ cotangent
    )
    assert np.dot(cold.operator.matvec(vector), cotangent) == pytest.approx(
        np.dot(vector, cold.operator.rmatvec(cotangent))
    )
    assert builder.retained_bytes > 0
    assert cold.diagnostics.compiled_kernel_cache_misses == 2
    assert cold.diagnostics.compiled_kernel_cache_hits == 0
    assert cold.diagnostics.captured_constant_bytes == 0
    assert cold.diagnostics.jax_backend
    assert cold.diagnostics.jax_devices
    assert cold.diagnostics.reach_input_shapes
    assert all(
        value >= 0.0
        for value in (
            cold.diagnostics.jax_argument_transfer_seconds,
            cold.diagnostics.jax_tracing_seconds,
            cold.diagnostics.jax_lowering_seconds,
            cold.diagnostics.jax_compilation_seconds,
            cold.diagnostics.jax_execution_seconds,
            cold.diagnostics.jax_host_transfer_seconds,
        )
    )
    builder.release_all()
    assert builder.retained_bytes == 0

    # Removing only the completed numerical artifact forces reconstruction while
    # retaining the builder's compatible compiled executables.
    next((tmp_path / "cache").glob("block-*.npz")).unlink()
    reused_compilation = builder.build_result(
        block, absolute_deadline=builder.clock() + 60.0
    )
    assert reused_compilation.diagnostics.compiled_kernel_cache_hits == 2
    assert reused_compilation.diagnostics.compiled_kernel_cache_misses == 0
    assert (
        reused_compilation.diagnostics.compiled_kernel_identity
        == cold.diagnostics.compiled_kernel_identity
    )
    np.testing.assert_array_equal(
        reused_compilation.operator.compact_matrix.toarray(),
        cold.operator.compact_matrix.toarray(),
    )
    builder.release_all()

    fresh = FixedRoutingSelectedBlockBuilder(
        inputs=inputs,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        provenance=provenance,
        config=config,
    ).build_result(block)
    assert fresh.cache_hit
    assert fresh.construction_count == 0
    np.testing.assert_array_equal(
        fresh.operator.compact_matrix.toarray(),
        cold.operator.compact_matrix.toarray(),
    )

    cache_file = next((tmp_path / "cache").glob("block-*.npz"))
    support_file = next((tmp_path / "support").glob("support-*.npz"))
    integration_builder = FixedRoutingSelectedBlockBuilder(
        inputs=inputs,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        provenance=provenance,
        config=config,
    )
    measured = construct_selected_block_operators(
        result=preflight,
        partition=partition,
        builder=integration_builder,
        budget=SupportPreflightBudget(),
    )
    assert tuple(item.block_id for item in measured) == select_representative_block_ids(
        preflight
    )
    assert len(measured) == len(select_representative_block_ids(preflight))

    cache_file.write_bytes(b"corrupt")
    rebuilt = FixedRoutingSelectedBlockBuilder(
        inputs=inputs,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        provenance=provenance,
        config=config,
    ).build_result(block)
    assert not rebuilt.cache_hit
    assert rebuilt.construction_count == 1

    support_file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="invalid selected-block support artifact"):
        FixedRoutingSelectedBlockBuilder(
            inputs=inputs,
            spec=spec,
            compact_layout=compact,
            partition=partition,
            provenance=provenance,
            config=config,
        ).prepare_support(block)


def test_production_selected_block_rejects_before_support_discovery(setup, tmp_path):
    inputs, compact, spec, fingerprints, partition = setup
    block = partition.blocks[0]
    builder = FixedRoutingSelectedBlockBuilder(
        inputs=inputs,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        provenance=SelectedBlockBuilderProvenance(
            fingerprints=fingerprints,
            semantic_preflight_fingerprint=SupportPreflightConfig().semantics_fingerprint,
            theta=1.0,
        ),
        config=SelectedBlockBuilderConfig(
            cache_directory=tmp_path,
            maximum_temporary_bytes=1,
            per_worker_memory_ceiling_bytes=1,
        ),
    )
    with pytest.raises(BlockConstructionResourceError, match="support-discovery"):
        builder.build(block)
    assert not tuple(tmp_path.rglob("*.npz"))
    assert builder.last_result is None


def test_od_batching_reduces_routing_passes_and_shares_logical_cache(setup, tmp_path):
    inputs, compact, spec, fingerprints, partition, preflight = completed_preflight(
        setup
    )
    summary = max(
        (item for item in preflight.block_summaries if item.free_columns == 2),
        key=lambda item: item.exact_nonzeros,
    )
    block = next(item for item in partition.blocks if item.block_id == summary.block_id)
    provenance = SelectedBlockBuilderProvenance(
        fingerprints=fingerprints,
        semantic_preflight_fingerprint=SupportPreflightConfig().semantics_fingerprint,
        theta=1.0,
    )
    common = SelectedBlockBuilderConfig(
        cache_directory=tmp_path / "unused",
        support_directory=tmp_path / "support",
        od_chunk_size=1,
        measurement_chunk_size=2,
        mapped_edge_chunk_size=3,
    )

    single_builder = FixedRoutingSelectedBlockBuilder(
        inputs=inputs,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        provenance=provenance,
        config=replace(common, cache_directory=tmp_path / "single", od_batch_size=1),
    )
    single = single_builder.build_result(block)
    batched = FixedRoutingSelectedBlockBuilder(
        inputs=inputs,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        provenance=provenance,
        config=replace(common, cache_directory=tmp_path / "batched", od_batch_size=2),
    ).build_result(block)
    np.testing.assert_array_equal(
        single.operator.compact_matrix.toarray(),
        batched.operator.compact_matrix.toarray(),
    )
    assert single.diagnostics.routing_evaluations == 2
    assert batched.diagnostics.routing_evaluations == 1
    assert single.diagnostics.od_batches == 2
    assert batched.diagnostics.od_batches == 1
    assert (
        single.diagnostics.measurement_mapping_filtering_passes
        == batched.diagnostics.measurement_mapping_filtering_passes
    )
    assert (
        single.diagnostics.candidate_contributions_examined
        == batched.diagnostics.candidate_contributions_examined
    )
    assert single.diagnostics.accepted_nonzeros == batched.diagnostics.accepted_nonzeros
    assert batched.diagnostics.accepted_nonzeros == batched.operator.compact_matrix.nnz
    assert batched.diagnostics.routing_evaluations == batched.diagnostics.od_batches
    assert batched.diagnostics.candidate_contributions_examined >= (
        batched.diagnostics.accepted_nonzeros
    )
    assert all(
        value >= 0.0
        for value in (
            batched.diagnostics.measurement_index_preparation_seconds,
            batched.diagnostics.od_chunk_preparation_seconds,
            batched.diagnostics.routing_evaluation_seconds,
            batched.diagnostics.measurement_support_filtering_seconds,
            batched.diagnostics.sparse_triplet_generation_seconds,
            batched.diagnostics.duplicate_reduction_seconds,
            batched.diagnostics.csr_csc_assembly_seconds,
        )
    )

    shared = FixedRoutingSelectedBlockBuilder(
        inputs=inputs,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        provenance=provenance,
        config=replace(common, cache_directory=tmp_path / "single", od_batch_size=2),
    ).build_result(block)
    assert shared.cache_hit
    assert shared.diagnostics.routing_evaluations == 0

    artifact = single.support_artifact
    one = single.estimate
    two = batched.estimate
    ceiling = (one.peak_worker_bytes + two.peak_worker_bytes) // 2
    fallback = FixedRoutingSelectedBlockBuilder(
        inputs=inputs,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        provenance=provenance,
        config=replace(
            common,
            cache_directory=tmp_path / "fallback",
            od_batch_size=2,
            maximum_temporary_bytes=ceiling,
            per_worker_memory_ceiling_bytes=ceiling,
        ),
    ).estimate_resources(block, artifact)
    assert fallback.requested_od_batch_size == 2
    assert fallback.effective_od_batch_size == 1


def test_selected_block_deadline_discards_partial_numerical_cache(setup, tmp_path):
    inputs, compact, spec, fingerprints, partition, preflight = completed_preflight(
        setup
    )
    summary = max(preflight.block_summaries, key=lambda item: item.exact_nonzeros)
    block = next(item for item in partition.blocks if item.block_id == summary.block_id)

    class Clock:
        value = 0.0

        def __call__(self):
            return self.value

    clock = Clock()

    expired_builder = FixedRoutingSelectedBlockBuilder(
        inputs=inputs,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        provenance=SelectedBlockBuilderProvenance(
            fingerprints=fingerprints,
            semantic_preflight_fingerprint=SupportPreflightConfig().semantics_fingerprint,
            theta=1.0,
        ),
        config=SelectedBlockBuilderConfig(
            cache_directory=tmp_path / "expired-cache",
            support_directory=tmp_path / "expired-support",
        ),
        clock=clock,
    )
    clock.value = 11.0
    with pytest.raises(SelectedBlockConstructionDeadlineError) as expired:
        expired_builder.build_result(block, absolute_deadline=10.0)
    assert expired.value.diagnostics.phase == "builder_entry"
    assert not tuple((tmp_path / "expired-cache").glob("*.npz"))
    clock.value = 0.0

    def expire_after_first_chunk(_progress):
        clock.value = 11.0

    builder = FixedRoutingSelectedBlockBuilder(
        inputs=inputs,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        provenance=SelectedBlockBuilderProvenance(
            fingerprints=fingerprints,
            semantic_preflight_fingerprint=SupportPreflightConfig().semantics_fingerprint,
            theta=1.0,
        ),
        config=SelectedBlockBuilderConfig(
            cache_directory=tmp_path / "cache",
            support_directory=tmp_path / "support",
            od_chunk_size=1,
            od_batch_size=1,
            measurement_chunk_size=1,
            mapped_edge_chunk_size=2,
        ),
        progress=expire_after_first_chunk,
        clock=clock,
    )
    with pytest.raises(SelectedBlockConstructionDeadlineError) as error:
        builder.build_result(block, absolute_deadline=10.0)
    diagnostics = error.value.diagnostics
    assert diagnostics.block_id == block.block_id
    assert diagnostics.phase in {
        "measurement_support_filtering",
        "od_batch_preparation",
        "sparse_assembly",
    }
    assert diagnostics.completed_mapping_passes >= 1
    assert diagnostics.partial_work_discarded
    assert not diagnostics.numerical_cache_persistence_completed
    assert not diagnostics.valid_warm_cache_exists
    assert not tuple((tmp_path / "cache").glob("block-*.npz"))


def test_selected_block_durable_progress_survives_compilation_cancellation(
    setup, tmp_path
):
    inputs, compact, spec, fingerprints, partition, preflight = completed_preflight(
        setup
    )
    summary = max(preflight.block_summaries, key=lambda item: item.exact_nonzeros)
    block = next(item for item in partition.blocks if item.block_id == summary.block_id)
    progress_path = tmp_path / "progress.jsonl"
    sink = SelectedBlockJSONLProgressSink(progress_path, durable=True)

    def cancel_at_compilation(event):
        sink(event)
        if event.event == "jax_compilation_start":
            raise KeyboardInterrupt

    builder = FixedRoutingSelectedBlockBuilder(
        inputs=inputs,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        provenance=SelectedBlockBuilderProvenance(
            fingerprints=fingerprints,
            semantic_preflight_fingerprint=SupportPreflightConfig().semantics_fingerprint,
            theta=1.0,
        ),
        config=SelectedBlockBuilderConfig(
            cache_directory=tmp_path / "cache",
            support_directory=tmp_path / "support",
            od_chunk_size=1,
            od_batch_size=1,
            measurement_chunk_size=2,
            mapped_edge_chunk_size=3,
        ),
        phase_progress=cancel_at_compilation,
    )
    with pytest.raises(KeyboardInterrupt):
        builder.build_result(block)

    records = [json.loads(line) for line in progress_path.read_text().splitlines()]
    events = [record["event"] for record in records]
    assert events[-1] == "jax_compilation_start"
    assert "jax_tracing_start" in events
    assert "jax_tracing_complete" in events
    assert "jax_lowering_start" in events
    assert "jax_lowering_complete" in events
    assert "jax_compilation_complete" not in events
    assert not tuple((tmp_path / "cache").glob("block-*.npz"))
    assert all(record["schema_version"] == 1 for record in records)
    assert all("input_shapes" in record for record in records)
    assert progress_path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize(
    ("stop_after", "forbidden"),
    [
        ("tracing", ("jax_lowering_start", "jax_compilation_start")),
        ("lowering", ("jax_compilation_start", "jax_execution_start")),
        ("compilation", ("jax_execution_start", "host_transfer_start")),
    ],
)
def test_selected_block_probe_stops_at_requested_jax_boundary(
    setup, tmp_path, stop_after, forbidden
):
    inputs, compact, spec, fingerprints, partition, preflight = completed_preflight(
        setup
    )
    summary = max(preflight.block_summaries, key=lambda item: item.exact_nonzeros)
    block = next(item for item in partition.blocks if item.block_id == summary.block_id)
    events = []
    builder = FixedRoutingSelectedBlockBuilder(
        inputs=inputs,
        spec=spec,
        compact_layout=compact,
        partition=partition,
        provenance=SelectedBlockBuilderProvenance(
            fingerprints=fingerprints,
            semantic_preflight_fingerprint=SupportPreflightConfig().semantics_fingerprint,
            theta=1.0,
        ),
        config=SelectedBlockBuilderConfig(
            cache_directory=tmp_path / stop_after,
            support_directory=tmp_path / "support",
            od_chunk_size=1,
            od_batch_size=1,
            measurement_chunk_size=2,
            mapped_edge_chunk_size=3,
        ),
        phase_progress=lambda event: events.append(event.event),
        diagnostic_stop_after=stop_after,
    )
    with pytest.raises(SelectedBlockDiagnosticStop) as stopped:
        builder.build_result(block)
    assert stopped.value.event.event == f"jax_{stop_after}_complete"
    assert events[-1] == f"jax_{stop_after}_complete"
    assert not set(events).intersection(forbidden)
    assert not tuple((tmp_path / stop_after).glob("block-*.npz"))


def test_atomic_selected_block_write_does_not_publish_interrupted_file(
    tmp_path, monkeypatch
):
    import public_transportation.inference.block_coordinate.fixed_routing_selected_block_builder as builder_module

    target = tmp_path / "operator.npz"

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(builder_module.np, "savez", interrupt)
    with pytest.raises(KeyboardInterrupt):
        builder_module._atomic_npz(target, data=np.arange(3))
    assert not target.exists()
    assert not tuple(tmp_path.glob(".operator.npz.*.tmp"))
