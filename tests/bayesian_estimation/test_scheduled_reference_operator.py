from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.domain import Scenario
from public_transportation.inference.assignment_adapter import (
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.assignment_contract import (
    AssignmentCompatibilityError,
    AssignmentOperator,
    CanonicalMeasurement,
    CanonicalTimeInterval,
    build_canonical_assignment_index,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.direct_scheduled_temporal_builder import (
    DirectScheduledGravityOperator,
    activate_direct_scheduled_temporal_operator,
    prepare_direct_scheduled_temporal_operator,
)
from public_transportation.inference.construction_control import (
    ConstructionDeadline,
    ConstructionPhase,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    prepare_fixed_routing_measurement_operator,
)
from public_transportation.inference.fixed_routing_sharded_builder import (
    ShardedConstructionConfig,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout
from public_transportation.inference.measurement_support_preflight import (
    UnsupportedPositiveBoardingError,
)
from public_transportation.inference.scheduled_reference_operator import (
    ScheduledTimeExpandedReferenceOperator,
    build_scheduled_reference_artifact_identity,
)
from public_transportation.inference.temporal_assignment_blocks import (
    TemporalSupportProfileConfig,
    build_chunked_temporal_block_operator,
    build_exact_temporal_block_operator,
    profile_temporal_block_support,
)
from public_transportation.inference.temporal_assignment_sparse_backend import (
    CSRCSCTemporalAssignmentOperator,
)
from public_transportation.inference.sharded_sparse_operator import shard_path
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
)
from public_transportation.inference.temporal_assignment_persistence import (
    load_temporal_block_operator,
    reuse_or_build_temporal_block_operator,
    temporal_block_cache_path,
)
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


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("scheduled-reference")
    for name in NETWORK_FILES:
        shutil.copy2(EXAMPLE / "data" / name, directory / name)
    shutil.copy2(
        EXAMPLE / "pre_processing/results/demand.csv", directory / "demand.csv"
    )
    scenario = Scenario.from_folder(directory, strict=True)
    return prepare_assignment(scenario=scenario, config=AssignmentConfig())


def _spec(num_links: int) -> AggregationSpec:
    links = np.arange(min(num_links, 8), dtype=np.int32)
    return AggregationSpec(
        num_measurements=3,
        measurement_index=np.arange(links.size, dtype=np.int32) % 3,
        link_index=links,
    )


def _layout(num_od: int, *, fixed_positive: bool) -> ODParameterLayout:
    fixed = (0,) if fixed_positive else ()
    free = tuple(index for index in range(num_od) if index not in fixed)
    return ODParameterLayout(
        num_od_total=num_od,
        od_keys=tuple((f"origin-{index}", "destination", "period") for index in range(num_od)),
        free_od_indices=free,
        fixed_od_indices=fixed,
        fixed_od_values=(4.0,) if fixed_positive else (),
        free_baseline_values=tuple(1.0 for _ in free),
        fixed_zero_indices=(),
        fixed_positive_indices=fixed,
    )


def _canonical_index(layout: ODParameterLayout):
    return build_canonical_assignment_index(
        parameter_layout=layout,
        time_intervals=(CanonicalTimeInterval("period", 0, 3600),),
        measurements=tuple(
            CanonicalMeasurement(
                row_index=index,
                measurement_id=f"measurement-{index}",
                event="boarding" if index % 2 == 0 else "alighting",
                location_id=f"stop-{index}",
                interval_id="period",
            )
            for index in range(3)
        ),
    )


def _scheduled_operator(*, inputs, spec, index, theta=1.0):
    identity = build_scheduled_reference_artifact_identity(
        inputs=inputs,
        spec=spec,
        canonical_index=index,
        theta=theta,
        temporal_discretization_fingerprint="one-period",
        departure_choice_fingerprint="scenario-departure-bins",
        feasibility_fingerprint="assignment-config",
        coefficient_policy_fingerprint="exact-float32",
    )
    return ScheduledTimeExpandedReferenceOperator(
        inputs=inputs,
        spec=spec,
        canonical_index=index,
        theta=theta,
        identity=identity,
    )


@pytest.mark.parametrize("fixed_positive", [False, True])
def test_scheduled_reference_matches_fixed_routing_products_and_offset(
    artifacts, fixed_positive
):
    full_inputs = build_assignment_inputs(artifacts=artifacts)
    layout = _layout(
        int(full_inputs.od_origin_node.shape[0]), fixed_positive=fixed_positive
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    spec = _spec(inputs.graph.num_links)
    index = _canonical_index(layout)
    scheduled = _scheduled_operator(inputs=inputs, spec=spec, index=index)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    fixed = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="scheduled-reference-test",
        compact_layout=compact,
        od_layout_fingerprint=layout.fingerprint,
        representation="dense",
        chunk_size=8,
    )
    demand = jnp.linspace(0.25, 3.0, layout.num_free, dtype=jnp.float32)
    residual = jnp.asarray([0.5, -0.25, 1.5], dtype=jnp.float32)

    np.testing.assert_allclose(
        scheduled.matvec(demand), fixed.jax_matvec(demand), rtol=4e-5, atol=4e-5
    )
    np.testing.assert_allclose(
        scheduled.rmatvec(residual),
        fixed.jax_rmatvec(residual),
        rtol=4e-5,
        atol=4e-5,
    )
    np.testing.assert_allclose(
        scheduled.fixed_measurement_offset,
        fixed.fixed_measurement_offset,
        rtol=4e-5,
        atol=4e-5,
    )
    assert isinstance(scheduled, AssignmentOperator)
    assert scheduled.representation == "scheduled_time_expanded_reference"


def test_scheduled_reference_adjoint_identity(artifacts):
    inputs = build_assignment_inputs(artifacts=artifacts)
    layout = _layout(int(inputs.od_origin_node.shape[0]), fixed_positive=False)
    operator = _scheduled_operator(
        inputs=inputs,
        spec=_spec(inputs.graph.num_links),
        index=_canonical_index(layout),
    )
    demand = jnp.linspace(0.1, 2.0, layout.num_free, dtype=jnp.float32)
    residual = jnp.asarray([0.7, -0.4, 1.2], dtype=jnp.float32)

    lhs = jnp.vdot(operator.matvec(demand), residual)
    rhs = jnp.vdot(demand, operator.rmatvec(residual))
    np.testing.assert_allclose(lhs, rhs, rtol=4e-5, atol=4e-5)


def test_scheduled_reference_rejects_identity_mismatch(artifacts):
    inputs = build_assignment_inputs(artifacts=artifacts)
    layout = _layout(int(inputs.od_origin_node.shape[0]), fixed_positive=False)
    index = _canonical_index(layout)
    identity = build_scheduled_reference_artifact_identity(
        inputs=inputs,
        spec=_spec(inputs.graph.num_links),
        canonical_index=index,
        theta=2.0,
        temporal_discretization_fingerprint="one-period",
        departure_choice_fingerprint="scenario-departure-bins",
        feasibility_fingerprint="assignment-config",
        coefficient_policy_fingerprint="exact-float32",
    )

    with pytest.raises(AssignmentCompatibilityError, match="route-choice"):
        ScheduledTimeExpandedReferenceOperator(
            inputs=inputs,
            spec=_spec(inputs.graph.num_links),
            canonical_index=index,
            theta=1.0,
            identity=identity,
        )


def test_temporal_support_profile_is_conservative_and_temporally_bounded():
    layout = ODParameterLayout(
        num_od_total=2,
        od_keys=(("o", "d", "early"), ("o", "d", "late")),
        free_od_indices=(0, 1),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=(1.0, 1.0),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )
    index = build_canonical_assignment_index(
        parameter_layout=layout,
        time_intervals=(
            CanonicalTimeInterval("early", 0, 600),
            CanonicalTimeInterval("middle", 600, 1200),
            CanonicalTimeInterval("late", 3600, 4200),
        ),
        measurements=(
            CanonicalMeasurement(0, "early-board", "boarding", "a", "early"),
            CanonicalMeasurement(1, "middle-board", "boarding", "b", "middle"),
            CanonicalMeasurement(2, "late-alight", "alighting", "c", "late"),
        ),
    )
    profile = profile_temporal_block_support(
        canonical_index=index,
        config=TemporalSupportProfileConfig(maximum_journey_duration_seconds=900),
    )

    assert profile.dense_candidate_entries == 6
    assert profile.total_candidate_entries == 3
    assert profile.excluded_by_temporal_structure == 3
    assert {
        (item.key.measurement_interval_id, item.key.departure_interval_id)
        for item in profile.blocks
    } == {
        ("early", "early"),
        ("middle", "early"),
        ("late", "late"),
    }
    assert profile.projected_storage_bytes > 0


def test_exact_temporal_blocks_match_scheduled_reference_and_report_progress(
    artifacts,
):
    inputs = build_assignment_inputs(artifacts=artifacts)
    layout = _layout(int(inputs.od_origin_node.shape[0]), fixed_positive=False)
    reference = _scheduled_operator(
        inputs=inputs,
        spec=_spec(inputs.graph.num_links),
        index=_canonical_index(layout),
    )
    progress = []
    operator = build_exact_temporal_block_operator(
        reference=reference, progress=progress.append
    )
    demand = jnp.linspace(0.1, 2.0, layout.num_free, dtype=jnp.float32)
    residual = jnp.asarray([0.4, -0.7, 1.3], dtype=jnp.float32)

    np.testing.assert_allclose(
        operator.matvec(demand), reference.matvec(demand), rtol=4e-5, atol=4e-5
    )
    np.testing.assert_allclose(
        operator.rmatvec(residual), reference.rmatvec(residual), rtol=4e-5, atol=4e-5
    )
    assert progress[-1].completed_columns == layout.num_free
    assert progress[-1].predicted_remaining_seconds == 0.0
    assert operator.diagnostics.columns_processed == layout.num_free
    assert operator.diagnostics.removed_l1_mass == 0.0
    assert operator.diagnostics.nonzero_entries == sum(
        block.nonzero_entries for block in operator.blocks
    )


def test_temporal_block_construction_honors_deadline(artifacts):
    inputs = build_assignment_inputs(artifacts=artifacts)
    layout = _layout(int(inputs.od_origin_node.shape[0]), fixed_positive=False)
    reference = _scheduled_operator(
        inputs=inputs,
        spec=_spec(inputs.graph.num_links),
        index=_canonical_index(layout),
    )
    with pytest.raises(TimeoutError, match="0/"):
        build_exact_temporal_block_operator(
            reference=reference, absolute_deadline=0.0, clock=lambda: 1.0
        )


def test_temporal_block_artifact_reuses_and_rejects_incompatibility(
    artifacts, tmp_path
):
    inputs = build_assignment_inputs(artifacts=artifacts)
    layout = _layout(int(inputs.od_origin_node.shape[0]), fixed_positive=False)
    reference = _scheduled_operator(
        inputs=inputs,
        spec=_spec(inputs.graph.num_links),
        index=_canonical_index(layout),
    )
    first, first_reused = reuse_or_build_temporal_block_operator(
        cache_root=tmp_path, reference=reference
    )
    second, second_reused = reuse_or_build_temporal_block_operator(
        cache_root=tmp_path, reference=reference
    )
    demand = jnp.linspace(0.2, 1.7, layout.num_free, dtype=jnp.float32)

    assert not first_reused
    assert second_reused
    np.testing.assert_array_equal(second.matvec(demand), first.matvec(demand))
    incompatible = replace(reference.identity, timetable_fingerprint="changed")
    with pytest.raises(AssignmentCompatibilityError, match="timetable_fingerprint"):
        load_temporal_block_operator(
            temporal_block_cache_path(tmp_path, reference.identity),
            expected_identity=incompatible,
            expected_canonical_index=reference.canonical_index,
        )


def test_temporal_block_artifact_rejects_corrupt_payload(artifacts, tmp_path):
    inputs = build_assignment_inputs(artifacts=artifacts)
    layout = _layout(int(inputs.od_origin_node.shape[0]), fixed_positive=False)
    reference = _scheduled_operator(
        inputs=inputs,
        spec=_spec(inputs.graph.num_links),
        index=_canonical_index(layout),
    )
    operator, _ = reuse_or_build_temporal_block_operator(
        cache_root=tmp_path, reference=reference
    )
    assert operator.blocks
    directory = temporal_block_cache_path(tmp_path, reference.identity)
    block_path = sorted((directory / "blocks").glob("*.npz"))[0]
    with np.load(block_path, allow_pickle=False) as data:
        rows = data["row_indices"]
        columns = data["column_indices"]
        values = np.array(data["values"], copy=True)
    values[0] += 1.0
    np.savez(
        block_path, row_indices=rows, column_indices=columns, values=values
    )

    with pytest.raises(ValueError, match="missing or corrupt"):
        load_temporal_block_operator(
            directory,
            expected_identity=reference.identity,
            expected_canonical_index=reference.canonical_index,
        )


def test_chunked_constructor_compiles_once_and_pads_final_chunk(artifacts):
    inputs = build_assignment_inputs(artifacts=artifacts)
    layout = _layout(int(inputs.od_origin_node.shape[0]), fixed_positive=False)
    reference = _scheduled_operator(
        inputs=inputs,
        spec=_spec(inputs.graph.num_links),
        index=_canonical_index(layout),
    )
    exact = build_exact_temporal_block_operator(reference=reference)
    progress = []
    chunk_size = 5
    chunked = build_chunked_temporal_block_operator(
        reference=reference,
        chunk_size=chunk_size,
        progress=progress.append,
    )
    demand = jnp.linspace(0.1, 2.0, layout.num_free, dtype=jnp.float32)
    residual = jnp.asarray([0.3, -0.8, 1.1], dtype=jnp.float32)

    np.testing.assert_allclose(
        chunked.matvec(demand), exact.matvec(demand), rtol=4e-5, atol=4e-5
    )
    np.testing.assert_allclose(
        chunked.rmatvec(residual), exact.rmatvec(residual), rtol=4e-5, atol=4e-5
    )
    assert chunked.diagnostics.compilation_count == 1
    assert chunked.diagnostics.num_chunks == math.ceil(layout.num_free / chunk_size)
    assert chunked.diagnostics.chunk_shape == (chunk_size, 3)
    assert progress[-1].completed_columns == layout.num_free
    assert layout.num_free % chunk_size != 0


def test_csr_csc_backend_matches_blocks_and_supplies_jax_adjoint(artifacts):
    inputs = build_assignment_inputs(artifacts=artifacts)
    layout = _layout(int(inputs.od_origin_node.shape[0]), fixed_positive=False)
    reference = _scheduled_operator(
        inputs=inputs,
        spec=_spec(inputs.graph.num_links),
        index=_canonical_index(layout),
    )
    blocks = build_chunked_temporal_block_operator(
        reference=reference, chunk_size=3
    )
    sparse_backend = CSRCSCTemporalAssignmentOperator(blocks)
    demand = jnp.linspace(0.2, 1.8, layout.num_free, dtype=jnp.float32)
    residual = jnp.asarray([0.6, -0.2, 1.4], dtype=jnp.float32)

    np.testing.assert_allclose(
        sparse_backend.matvec(demand), blocks.matvec(demand), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        sparse_backend.rmatvec(residual),
        blocks.rmatvec(residual),
        rtol=1e-6,
        atol=1e-6,
    )
    gradient = jax.grad(
        lambda value: jnp.vdot(sparse_backend.matvec(value), residual)
    )(demand)
    np.testing.assert_allclose(
        gradient, sparse_backend.rmatvec(residual), rtol=1e-6, atol=1e-6
    )
    compiled = jax.jit(sparse_backend.matvec)(demand)
    np.testing.assert_allclose(compiled, blocks.matvec(demand), rtol=1e-6, atol=1e-6)
    assert sparse_backend.metrics.nonzero_entries == blocks.diagnostics.nonzero_entries
    assert sparse_backend.metrics.total_bytes > 0
    assert sparse_backend.representation == "temporal_csr_csc"


def _direct_config():
    return ShardedConstructionConfig(
        od_chunk_size=8,
        measurement_block_size=2,
        worker_memory_budget_bytes=100_000_000,
        target_nonzeros_per_storage_shard=2,
        maximum_nonzeros_per_storage_shard=16,
        manifest_checkpoint_shards=1,
    )


def test_direct_resumable_builder_matches_reference_and_reuses_artifact(
    artifacts, tmp_path
):
    full_inputs = build_assignment_inputs(artifacts=artifacts)
    layout = _layout(int(full_inputs.od_origin_node.shape[0]), fixed_positive=True)
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    index = _canonical_index(layout)
    reference = _scheduled_operator(inputs=inputs, spec=spec, index=index)
    first = prepare_direct_scheduled_temporal_operator(
        checkpoint_root=tmp_path / "checkpoints",
        artifact_root=tmp_path / "artifacts",
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        canonical_index=index,
        observations=np.zeros(index.number_of_measurements),
        identity=reference.identity,
        assignment_fingerprint="direct-scheduled-public-example",
        od_layout_fingerprint=layout.fingerprint,
        config=_direct_config(),
    )
    second = prepare_direct_scheduled_temporal_operator(
        checkpoint_root=tmp_path / "unused-checkpoints",
        artifact_root=tmp_path / "artifacts",
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        canonical_index=index,
        observations=np.zeros(index.number_of_measurements),
        identity=reference.identity,
        assignment_fingerprint="direct-scheduled-public-example",
        od_layout_fingerprint=layout.fingerprint,
        config=_direct_config(),
    )
    demand = jnp.linspace(0.1, 2.0, layout.num_free, dtype=jnp.float32)
    residual = jnp.asarray([0.25, -0.5, 1.3], dtype=jnp.float32)

    np.testing.assert_allclose(
        first.operator.matvec(demand), reference.matvec(demand), rtol=5e-5, atol=5e-5
    )
    np.testing.assert_allclose(
        first.operator.rmatvec(residual),
        reference.rmatvec(residual),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        first.operator.fixed_measurement_offset,
        reference.fixed_measurement_offset,
        rtol=5e-5,
        atol=5e-5,
    )
    assert first.source is not None
    assert first.source.manifest.complete
    assert not first.temporal_artifact_reused
    assert second.temporal_artifact_reused
    assert second.source is None
    np.testing.assert_array_equal(second.operator.matvec(demand), first.operator.matvec(demand))


def test_direct_builder_resumes_after_interruption(artifacts, tmp_path):
    inputs = build_assignment_inputs(artifacts=artifacts)
    layout = _layout(int(inputs.od_origin_node.shape[0]), fixed_positive=False)
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    index = _canonical_index(layout)
    reference = _scheduled_operator(inputs=inputs, spec=spec, index=index)

    def interrupt(event):
        if event["completed_shards"] == 1:
            raise RuntimeError("simulated direct-construction interruption")

    with pytest.raises(RuntimeError, match="simulated direct-construction"):
        prepare_direct_scheduled_temporal_operator(
            checkpoint_root=tmp_path,
            artifact_root=None,
            inputs=inputs,
            routing=routing,
            spec=spec,
            compact_layout=compact,
            canonical_index=index,
            observations=np.zeros(index.number_of_measurements),
            identity=reference.identity,
            assignment_fingerprint="direct-resume",
            od_layout_fingerprint=layout.fingerprint,
            config=_direct_config(),
            progress=interrupt,
        )
    resumed = prepare_direct_scheduled_temporal_operator(
        checkpoint_root=tmp_path,
        artifact_root=None,
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        canonical_index=index,
        observations=np.zeros(index.number_of_measurements),
        identity=reference.identity,
        assignment_fingerprint="direct-resume",
        od_layout_fingerprint=layout.fingerprint,
        config=_direct_config(),
    )

    assert resumed.source is not None
    assert resumed.source.reused_shards >= 1
    assert resumed.source.manifest.complete


def test_direct_builder_selectively_rebuilds_corrupt_shard(artifacts, tmp_path):
    inputs = build_assignment_inputs(artifacts=artifacts)
    layout = _layout(int(inputs.od_origin_node.shape[0]), fixed_positive=False)
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    index = _canonical_index(layout)
    reference = _scheduled_operator(inputs=inputs, spec=spec, index=index)
    first = prepare_direct_scheduled_temporal_operator(
        checkpoint_root=tmp_path,
        artifact_root=None,
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        canonical_index=index,
        observations=np.zeros(index.number_of_measurements),
        identity=reference.identity,
        assignment_fingerprint="direct-repair",
        od_layout_fingerprint=layout.fingerprint,
        config=_direct_config(),
    )
    assert first.source is not None
    damaged = shard_path(
        first.checkpoint_directory, first.source.manifest.expected_shards[0]
    )
    damaged.write_bytes(b"corrupt")
    repaired = prepare_direct_scheduled_temporal_operator(
        checkpoint_root=tmp_path,
        artifact_root=None,
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        canonical_index=index,
        observations=np.zeros(index.number_of_measurements),
        identity=reference.identity,
        assignment_fingerprint="direct-repair",
        od_layout_fingerprint=layout.fingerprint,
        config=_direct_config(),
    )

    assert repaired.source is not None
    assert repaired.source.rejected_shards == 1
    assert repaired.source.rebuilt_shards == 1


def _activation_arguments(artifacts, tmp_path):
    inputs = build_assignment_inputs(artifacts=artifacts)
    layout = _layout(int(inputs.od_origin_node.shape[0]), fixed_positive=False)
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    spec = _spec(inputs.graph.num_links)
    index = _canonical_index(layout)
    reference = _scheduled_operator(inputs=inputs, spec=spec, index=index)
    calls = []

    def routing_factory():
        calls.append(True)
        return prepare_fixed_routing(inputs=inputs, theta=1.0)

    return dict(
        checkpoint_root=tmp_path / "checkpoints",
        artifact_root=tmp_path / "artifacts",
        inputs=inputs,
        routing_factory=routing_factory,
        theta=1.0,
        spec=spec,
        compact_layout=compact,
        canonical_index=index,
        observations=np.zeros(index.number_of_measurements),
        identity=reference.identity,
        assignment_fingerprint="direct-activation",
        od_layout_fingerprint=layout.fingerprint,
        config=_direct_config(),
    ), calls


def test_direct_activation_declines_unjustified_build_without_routing(
    artifacts, tmp_path
):
    arguments, routing_calls = _activation_arguments(artifacts, tmp_path)
    result = activate_direct_scheduled_temporal_operator(
        mode="auto",
        expected_evaluations=5,
        construction_seconds=20.0,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        **arguments,
    )

    assert result.operator is None
    assert not result.decision.activated
    assert "does not exceed" in result.decision.reason
    assert result.decision.break_even_evaluations == pytest.approx(20.0 / 1.5)
    assert routing_calls == []


def test_direct_activation_rejects_unsupported_positive_boarding_before_routing(
    artifacts, tmp_path
):
    arguments, routing_calls = _activation_arguments(artifacts, tmp_path)
    inputs = arguments["inputs"]
    active_origins = np.asarray(inputs.od_origin_node)
    tails = np.asarray(inputs.graph.tail)
    ineligible = np.flatnonzero(~np.isin(tails, active_origins))
    assert ineligible.size
    spec = AggregationSpec(
        num_measurements=3,
        measurement_index=np.arange(3, dtype=np.int32),
        link_index=np.full(3, ineligible[0], dtype=np.int32),
    )
    arguments["spec"] = spec
    arguments["identity"] = _scheduled_operator(
        inputs=inputs,
        spec=spec,
        index=arguments["canonical_index"],
    ).identity
    arguments["observations"] = np.asarray([5.0, 0.0, 0.0])

    with pytest.raises(UnsupportedPositiveBoardingError) as caught:
        activate_direct_scheduled_temporal_operator(
            mode="direct",
            expected_evaluations=1,
            construction_seconds=None,
            reference_evaluation_seconds=2.0,
            operator_evaluation_seconds=0.5,
            **arguments,
        )

    assert routing_calls == []
    assert caught.value.report.stage == "canonical_origin_support"
    assert caught.value.report.issues[0].row_index == 0
    assert caught.value.report_path is not None
    assert caught.value.report_path.exists()


def test_direct_activation_builds_then_fresh_call_reuses_without_routing(
    artifacts, tmp_path
):
    arguments, routing_calls = _activation_arguments(artifacts, tmp_path)
    built = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=0,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        **arguments,
    )
    assert isinstance(built.operator, DirectScheduledGravityOperator)
    assert built.operator.representation == "direct_scheduled_temporal_blocks_csr_csc"
    assert len(routing_calls) == 1
    assert built.construction is not None

    reused = activate_direct_scheduled_temporal_operator(
        mode="auto",
        expected_evaluations=0,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        **arguments,
    )
    assert reused.operator is not None
    assert reused.decision.cache_reused
    assert reused.construction is None
    assert len(routing_calls) == 1
    vector = jnp.ones(reused.operator.num_free_od, dtype=jnp.float32)
    matrix = jnp.stack((vector, 2.0 * vector), axis=1)
    np.testing.assert_allclose(
        reused.operator.jax_matmat(matrix),
        jnp.stack(
            (
                reused.operator.jax_matvec(vector),
                reused.operator.jax_matvec(2.0 * vector),
            ),
            axis=1,
        ),
    )
    def expired_clock():
        return 50.0
    expired = ConstructionDeadline.from_absolute(49.0, clock=expired_clock)
    reused_after_deadline = activate_direct_scheduled_temporal_operator(
        mode="auto",
        expected_evaluations=0,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        deadline=expired,
        **arguments,
    )
    assert reused_after_deadline.operator is not None
    assert reused_after_deadline.decision.cache_reused
    assert reused_after_deadline.termination is None
    assert len(routing_calls) == 1


def test_direct_construction_quarantines_invalid_final_artifact(artifacts, tmp_path):
    arguments, routing_calls = _activation_arguments(artifacts, tmp_path)
    artifact = temporal_block_cache_path(
        arguments["artifact_root"], arguments["identity"]
    )
    artifact.mkdir(parents=True)
    (artifact / "manifest.json").write_text("old-schema", encoding="utf-8")

    result = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=1,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        **arguments,
    )

    assert result.operator is not None
    assert len(routing_calls) == 1
    assert artifact.exists()
    quarantined = tuple(artifact.parent.glob(f"{artifact.name}.invalid-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "manifest.json").read_text(encoding="utf-8") == "old-schema"


def test_activation_stops_before_routing_with_structured_status(artifacts, tmp_path):
    arguments, routing_calls = _activation_arguments(artifacts, tmp_path)
    now = [10.0]
    deadline = ConstructionDeadline.from_budget(
        2.0, safety_margin_seconds=0.5, clock=lambda: now[0]
    )
    events = []
    result = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=1,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        predicted_routing_seconds=1.51,
        deadline=deadline,
        progress=events.append,
        **arguments,
    )

    assert result.operator is None
    assert result.termination is not None
    assert result.termination.phase is ConstructionPhase.ROUTING_PREPARATION
    assert result.termination.checkpoint_reusable is False
    assert routing_calls == []
    assert events[-1]["status"] == "deadline_stopped"


def test_activation_stops_after_committed_shard_then_resumes(artifacts, tmp_path):
    arguments, routing_calls = _activation_arguments(artifacts, tmp_path)
    now = [10.0]
    deadline = ConstructionDeadline.from_budget(10.0, clock=lambda: now[0])
    events = []

    def stop_after_shard(event):
        events.append(event)
        if (
            event["phase"] == "shard_construction"
            and event["status"] == "running"
        ):
            now[0] = 21.0

    stopped = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=1,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        deadline=deadline,
        progress=stop_after_shard,
        **arguments,
    )
    assert stopped.termination is not None
    assert stopped.termination.phase is ConstructionPhase.SHARD_CONSTRUCTION
    assert stopped.termination.completed_units == 1
    assert stopped.termination.checkpoint_reusable

    resume_events = []
    resumed = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=1,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        progress=resume_events.append,
        **arguments,
    )
    assert resumed.operator is not None
    planning = [event for event in resume_events if event["phase"] == "planning"]
    assert planning and planning[-1]["cache_hits"] == 1
    assert resumed.construction is not None
    assert resumed.construction.source is not None
    assert resumed.construction.source.reused_shards >= 1
    assert len(routing_calls) == 1


def test_activation_reuses_persisted_support_after_planning_stop(
    artifacts, tmp_path, monkeypatch
):
    arguments, routing_calls = _activation_arguments(artifacts, tmp_path)
    now = [10.0]
    deadline = ConstructionDeadline.from_budget(10.0, clock=lambda: now[0])

    def stop_after_support(event):
        if event["phase"] == "support_discovery" and event["status"] == "completed":
            now[0] = 21.0

    stopped = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=1,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        deadline=deadline,
        progress=stop_after_support,
        **arguments,
    )
    assert stopped.termination is not None
    assert stopped.termination.phase is ConstructionPhase.PLANNING
    assert stopped.termination.checkpoint_reusable
    assert (Path(stopped.termination.checkpoint_location) / "support.npz").exists()
    assert len(routing_calls) == 1

    def forbidden_support_discovery(**_):
        raise AssertionError("compatible persisted support was recomputed")

    monkeypatch.setattr(
        "public_transportation.inference.fixed_routing_sharded_builder."
        "analyze_fixed_routing_origin_support",
        forbidden_support_discovery,
    )
    resumed = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=1,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        **arguments,
    )
    assert resumed.operator is not None
    assert len(routing_calls) == 1


def test_activation_resumes_origin_support_at_next_destination_group(
    artifacts, tmp_path
):
    arguments, routing_calls = _activation_arguments(artifacts, tmp_path)
    now = [10.0]
    deadline = ConstructionDeadline.from_budget(10.0, clock=lambda: now[0])

    def stop_after_first_group(event):
        if (
            event["phase"] == "support_discovery"
            and event["status"] == "running"
            and event["completed_units"] == 1
        ):
            now[0] = 21.0

    stopped = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=1,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        deadline=deadline,
        progress=stop_after_first_group,
        **arguments,
    )
    assert stopped.termination is not None
    assert stopped.termination.phase is ConstructionPhase.SUPPORT_DISCOVERY
    assert stopped.termination.completed_units == 1
    assert stopped.termination.checkpoint_reusable
    support_groups = Path(stopped.termination.checkpoint_location) / "support_groups"
    assert (support_groups / "group-000000.npz").exists()
    assert len(routing_calls) == 1

    resume_events = []
    resumed = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=1,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        progress=resume_events.append,
        **arguments,
    )
    assert resumed.operator is not None
    group_hits = [
        event
        for event in resume_events
        if event["phase"] == "support_discovery"
        and event["status"] == "running"
        and event["cache_hits"] == 1
    ]
    assert group_hits and group_hits[0]["completed_units"] == 1
    assert len(routing_calls) == 1


def test_activation_stops_and_resumes_persistent_sharded_routing(
    artifacts, tmp_path
):
    arguments, routing_calls = _activation_arguments(artifacts, tmp_path)
    routing_config = FixedRoutingPreparationConfig(
        maximum_groups_per_shard=1,
        maximum_retained_bytes_per_shard=8 * 1024 * 1024,
        maximum_temporary_bytes=64 * 1024 * 1024,
        checkpoint_directory=tmp_path / "ignored-checkpoint",
        cache_directory=tmp_path / "ignored-cache",
        dispatch_safety_margin_seconds=0.0,
    )
    now = [10.0]
    deadline = ConstructionDeadline.from_budget(10.0, clock=lambda: now[0])
    stopped_events: list[dict[str, object]] = []

    def stop_after_first_routing_shard(event):
        stopped_events.append(event)
        if (
            event["phase"] == "routing_preparation"
            and event.get("routing_phase") == "shard_persisted"
            and event["completed_units"] == 1
        ):
            now[0] = 21.0

    stopped = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=1,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        deadline=deadline,
        progress=stop_after_first_routing_shard,
        routing_preparation_config=routing_config,
        **arguments,
    )
    assert stopped.termination is not None
    assert stopped.termination.phase is ConstructionPhase.ROUTING_PREPARATION
    assert stopped.termination.completed_units == 1
    assert stopped.termination.checkpoint_reusable
    assert routing_calls == []
    json.dumps(stopped_events)
    stopped_routing_events = [
        event
        for event in stopped_events
        if event["phase"] == "routing_preparation"
    ]
    assert stopped_routing_events[0]["routing_phase"] == "planning"
    assert stopped_routing_events[0]["completed_units"] == 0
    assert stopped_routing_events[0]["total_units"] > 0
    assert stopped_routing_events[0]["total_destination_groups"] > 0
    assert stopped_routing_events[0]["eta_confidence"] == "unavailable"
    routing_subphases = {
        event.get("routing_phase")
        for event in stopped_events
        if event["phase"] == "routing_preparation"
    }
    assert {
        "trace",
        "lowering",
        "compilation",
        "batch_execution",
        "synchronization",
        "host_transfer",
        "shard_persisted",
    } <= routing_subphases
    assert max(
        int(event.get("resident_routing_batches", 0))
        for event in stopped_events
    ) <= 1

    resume_events: list[dict[str, object]] = []
    resumed = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=1,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        progress=resume_events.append,
        routing_preparation_config=routing_config,
        **arguments,
    )
    assert resumed.operator is not None
    routing_events = [
        event for event in resume_events if event["phase"] == "routing_preparation"
    ]
    assert routing_events
    assert any(int(event["cache_hits"] or 0) >= 1 for event in routing_events)
    assert routing_events[0]["routing_phase"] == "planning"
    assert routing_events[0]["completed_units"] >= 1
    assert routing_events[0]["total_units"] == stopped_routing_events[0]["total_units"]
    assert routing_calls == []

    dense = prepare_direct_scheduled_temporal_operator(
        checkpoint_root=tmp_path / "dense-checkpoints",
        artifact_root=None,
        inputs=arguments["inputs"],
        routing=prepare_fixed_routing(inputs=arguments["inputs"], theta=1.0),
        spec=arguments["spec"],
        compact_layout=arguments["compact_layout"],
        canonical_index=arguments["canonical_index"],
        observations=arguments["observations"],
        identity=arguments["identity"],
        assignment_fingerprint=arguments["assignment_fingerprint"],
        od_layout_fingerprint=arguments["od_layout_fingerprint"],
        config=arguments["config"],
    )
    demand = jnp.arange(1, resumed.operator.num_free_od + 1, dtype=jnp.float32)
    weights = jnp.arange(
        1, resumed.operator.num_measurements + 1, dtype=jnp.float32
    )
    np.testing.assert_allclose(
        resumed.operator.jax_matvec(demand),
        dense.operator.jax_matvec(demand),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        resumed.operator.jax_rmatvec(weights),
        dense.operator.jax_rmatvec(weights),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_activation_reuses_temporal_fragment_after_finalization_stop(
    artifacts, tmp_path
):
    arguments, _ = _activation_arguments(artifacts, tmp_path)
    now = [10.0]
    deadline = ConstructionDeadline.from_budget(10.0, clock=lambda: now[0])

    def stop_after_fragment(event):
        if (
            event["phase"] == "temporal_block_assembly"
            and event["status"] == "running"
        ):
            now[0] = 21.0

    stopped = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=1,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        deadline=deadline,
        progress=stop_after_fragment,
        **arguments,
    )
    assert stopped.termination is not None
    assert stopped.termination.phase is ConstructionPhase.TEMPORAL_BLOCK_ASSEMBLY
    checkpoint = Path(stopped.termination.checkpoint_location)
    assert tuple((checkpoint / "temporal_fragments").glob("*.npz"))

    events = []
    resumed = activate_direct_scheduled_temporal_operator(
        mode="direct",
        expected_evaluations=1,
        construction_seconds=None,
        reference_evaluation_seconds=2.0,
        operator_evaluation_seconds=0.5,
        progress=events.append,
        **arguments,
    )
    assert resumed.operator is not None
    assembly = [event for event in events if event["phase"] == "temporal_block_assembly"]
    assert any(event["cache_hits"] == 1 for event in assembly)
