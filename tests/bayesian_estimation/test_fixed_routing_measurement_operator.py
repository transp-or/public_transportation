from __future__ import annotations

import json
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
    assign_link_flow_fixed_routing,
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    choose_fixed_measurement_operator,
    fixed_routing_measurement_operator_cache_path,
    load_fixed_routing_measurement_operator,
    load_or_prepare_fixed_routing_measurement_operator,
    measurement_mapping_fingerprint,
    predict_measurements_fixed_operator,
    prepare_fixed_routing_measurement_operator,
    save_fixed_routing_measurement_operator,
    validate_fixed_routing_measurement_operator,
)
from public_transportation.inference.fixed_routing_linear_backend import (
    SparseOperatorSelectionConfig,
    prepare_fixed_routing_linear_measurement_backend,
)
from public_transportation.inference.fixed_routing_sharded_builder import (
    ShardedConstructionConfig,
    load_complete_sharded_fixed_routing_cache,
    plan_sharded_fixed_routing_operator,
    prepare_sharded_fixed_routing_measurement_operator,
)
from public_transportation.inference.fixed_routing_origin_support import (
    OriginSupportConfig,
    analyze_fixed_routing_origin_support,
    validate_origin_support_against_operator,
)
from public_transportation.inference.sharded_sparse_operator import (
    ShardedSparseLinearOperator,
    shard_path,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout
from public_transportation.inference.maximum_likelihood_pipeline import (
    build_od_theta_ml_problem,
)
from public_transportation.inference.pipeline import ODThetaEstimationRequest
from public_transportation.measurement.likelihood_jax import (
    predict_measurements_from_link_flow,
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
    directory = tmp_path_factory.mktemp("measurement-operator")
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


def _reference(inputs, routing, spec, demand):
    link_flow = assign_link_flow_fixed_routing(
        inputs=inputs,
        routing=routing,
        f=demand,
    )
    return predict_measurements_from_link_flow(
        link_flow,
        spec_num_measurements=spec.num_measurements,
        spec_measurement_index=jnp.asarray(spec.measurement_index),
        spec_link_index=jnp.asarray(spec.link_index),
    )


@pytest.fixture(scope="module")
def all_free_operator(artifacts):
    inputs = build_assignment_inputs(artifacts=artifacts)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    operator = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="simple-example-02",
        representation="dense",
        chunk_size=8,
    )
    return inputs, routing, spec, operator


@pytest.mark.parametrize("scale", [0.0, 0.25, 1.0, 2.0])
def test_dense_operator_matches_fixed_loader_and_scaling(all_free_operator, scale):
    inputs, routing, spec, operator = all_free_operator
    demand = jnp.linspace(0.0, 7.0 * scale, operator.num_free_od)

    direct = predict_measurements_fixed_operator(
        operator=operator,
        free_demand=demand,
        rho=jnp.asarray(1.0),
    )
    reference = _reference(inputs, routing, spec, demand)

    np.testing.assert_allclose(direct, reference, rtol=3e-5, atol=3e-5)


def test_operator_preserves_superposition(all_free_operator):
    _, _, _, operator = all_free_operator
    first = jnp.linspace(0.0, 2.0, operator.num_free_od)
    second = jnp.linspace(1.0, 0.0, operator.num_free_od)

    def predict(demand):
        return predict_measurements_fixed_operator(
            operator=operator, free_demand=demand, rho=jnp.asarray(1.0)
        )

    np.testing.assert_allclose(
        predict(first + second),
        predict(first) + predict(second),
        rtol=2e-6,
        atol=2e-6,
    )


def test_dense_operator_gradient_matches_reference(all_free_operator):
    inputs, routing, spec, operator = all_free_operator
    demand = jnp.linspace(0.1, 3.0, operator.num_free_od)

    direct_gradient = jax.grad(
        lambda value: jnp.square(
            predict_measurements_fixed_operator(
                operator=operator,
                free_demand=value,
                rho=jnp.asarray(0.7),
            )
        ).sum()
    )(demand)
    reference_gradient = jax.grad(
        lambda value: jnp.square(0.7 * _reference(inputs, routing, spec, value)).sum()
    )(demand)

    np.testing.assert_allclose(
        direct_gradient,
        reference_gradient,
        rtol=5e-5,
        atol=5e-5,
    )


def test_sparse_operator_matches_dense(all_free_operator):
    inputs, routing, spec, dense = all_free_operator
    sparse = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="simple-example-02",
        representation="bcoo",
        chunk_size=8,
    )
    demand = jnp.linspace(0.0, 4.0, dense.num_free_od)

    dense_value = predict_measurements_fixed_operator(
        operator=dense, free_demand=demand, rho=jnp.asarray(0.8)
    )
    sparse_value = predict_measurements_fixed_operator(
        operator=sparse, free_demand=demand, rho=jnp.asarray(0.8)
    )
    np.testing.assert_allclose(sparse_value, dense_value, rtol=2e-6, atol=2e-6)
    assert sparse.metrics.stored_bytes <= sparse.metrics.dense_bytes


def test_linear_sparse_backend_cache_hit_skips_routing_rebuild(artifacts, tmp_path):
    full_inputs = build_assignment_inputs(artifacts=artifacts)
    num_od = int(full_inputs.od_origin_node.shape[0])
    layout = ODParameterLayout(
        num_od_total=num_od,
        od_keys=tuple((f"o{i}", "d", "t") for i in range(num_od)),
        free_od_indices=tuple(range(num_od)),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=tuple(1.0 for _ in range(num_od)),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    config = SparseOperatorSelectionConfig(
        mode="sparse", memory_budget_bytes=100_000_000, chunk_size=8
    )
    built = prepare_fixed_routing_linear_measurement_backend(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        assignment_fingerprint="linear-backend-cache",
        od_layout_fingerprint=layout.fingerprint,
        cache_directory=tmp_path,
        config=config,
    )
    assert not built.metrics.cache_hit

    cached = prepare_fixed_routing_linear_measurement_backend(
        inputs=inputs,
        theta=1.0,
        routing_factory=lambda: pytest.fail("cache hit rebuilt fixed routing"),
        spec=spec,
        compact_layout=compact,
        assignment_fingerprint="linear-backend-cache",
        od_layout_fingerprint=layout.fingerprint,
        cache_directory=tmp_path,
        config=config,
    )
    assert cached.metrics.cache_hit
    demand = np.linspace(0.0, 2.0, num_od)
    np.testing.assert_allclose(cached.operator.matvec(demand), built.operator.matvec(demand))
    np.testing.assert_allclose(
        cached.fixed_measurement_offset, built.fixed_measurement_offset
    )


def test_sharded_builder_matches_monolithic_and_resumes(all_free_operator, tmp_path):
    inputs, routing, spec, monolithic = all_free_operator
    num_od = int(inputs.od_origin_node.shape[0])
    layout = ODParameterLayout(
        num_od_total=num_od,
        od_keys=tuple((f"o{i}", "d", "t") for i in range(num_od)),
        free_od_indices=tuple(range(num_od)),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=tuple(1.0 for _ in range(num_od)),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    config = ShardedConstructionConfig(
        od_chunk_size=8,
        measurement_block_size=2,
        worker_memory_budget_bytes=100_000_000,
    )
    built = prepare_sharded_fixed_routing_measurement_operator(
        directory=tmp_path,
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        assignment_fingerprint="sharded-public-example",
        od_layout_fingerprint=layout.fingerprint,
        config=config,
    )
    assert built.manifest.complete
    assert built.rebuilt_shards == built.plan.num_shards
    assert built.plan.construction_tasks >= built.plan.num_shards
    assert built.dispatch_count <= built.plan.construction_tasks
    assert built.synchronization_count == built.dispatch_count
    assert built.dispatch_count == len(built.origins_per_dispatch)
    assert built.plan.num_shards <= config.maximum_storage_shards
    assert built.plan.estimated_sparse_calls_per_product == built.plan.num_shards
    assert built.manifest_write_count <= (
        2 + built.plan.num_shards // config.manifest_checkpoint_shards
    )
    assert built.plan.maximum_shard_measurements <= 2
    assert built.plan.candidate_entries == built.manifest.aggregate_nonzeros
    assert built.plan.candidate_entries < built.plan.group_level_candidate_entries
    operator = ShardedSparseLinearOperator(tmp_path)
    demand = np.linspace(0.0, 2.0, num_od)
    cotangent = np.asarray([0.25, -0.5, 2.0])
    np.testing.assert_allclose(
        operator.matvec(demand),
        np.asarray(monolithic.matrix) @ demand,
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        operator.rmatvec(cotangent),
        np.asarray(monolithic.matrix).T @ cotangent,
        rtol=1e-4,
        atol=1e-4,
    )
    resumed = prepare_sharded_fixed_routing_measurement_operator(
        directory=tmp_path,
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        assignment_fingerprint="sharded-public-example",
        od_layout_fingerprint=layout.fingerprint,
        config=config,
    )
    assert resumed.reused_shards == built.plan.num_shards
    assert resumed.rebuilt_shards == 0
    assert resumed.compilation_seconds == 0.0

    fast_cached = load_complete_sharded_fixed_routing_cache(
        directory=tmp_path,
        inputs=inputs,
        spec=spec,
        compact_layout=compact,
        assignment_fingerprint="sharded-public-example",
        od_layout_fingerprint=layout.fingerprint,
        theta=1.0,
        config=config,
    )
    assert fast_cached is not None
    assert fast_cached.plan.num_shards == built.plan.num_shards
    assert fast_cached.support_discovery_seconds == 0.0
    assert fast_cached.compilation_seconds == 0.0

    damaged_path = shard_path(tmp_path, built.plan.expected_shards[0])
    with np.load(damaged_path, allow_pickle=False) as archive:
        damaged = {name: archive[name] for name in archive.files}
    damaged_metadata = json.loads(str(damaged["metadata"]))
    damaged_metadata["content_hash"] = "0" * 64
    damaged["metadata"] = np.asarray(json.dumps(damaged_metadata))
    np.savez(damaged_path, **damaged)
    repaired = prepare_sharded_fixed_routing_measurement_operator(
        directory=tmp_path,
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        assignment_fingerprint="sharded-public-example",
        od_layout_fingerprint=layout.fingerprint,
        config=config,
    )
    assert repaired.rejected_shards == 1
    assert repaired.rebuilt_shards == 1
    assert repaired.reused_shards == built.plan.num_shards - 1

    interrupted_directory = tmp_path / "interrupted"
    tiny_config = ShardedConstructionConfig(
        od_chunk_size=8,
        measurement_block_size=2,
        worker_memory_budget_bytes=100_000_000,
        target_nonzeros_per_storage_shard=1,
        maximum_nonzeros_per_storage_shard=1,
        manifest_checkpoint_shards=16,
    )

    def interrupt_after_first(event):
        if event["completed_shards"] == 1:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        prepare_sharded_fixed_routing_measurement_operator(
            directory=interrupted_directory,
            inputs=inputs,
            routing=routing,
            spec=spec,
            compact_layout=compact,
            assignment_fingerprint="sharded-interrupted-example",
            od_layout_fingerprint=layout.fingerprint,
            config=tiny_config,
            progress=interrupt_after_first,
        )
    resumed_after_interruption = prepare_sharded_fixed_routing_measurement_operator(
        directory=interrupted_directory,
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        assignment_fingerprint="sharded-interrupted-example",
        od_layout_fingerprint=layout.fingerprint,
        config=tiny_config,
    )
    assert resumed_after_interruption.reused_shards == 1
    interrupted_operator = ShardedSparseLinearOperator(interrupted_directory)
    np.testing.assert_allclose(
        interrupted_operator.matvec(demand),
        operator.matvec(demand),
        rtol=5e-5,
        atol=5e-5,
    )


def test_sharded_positive_fixed_flow_is_offset_only(artifacts, tmp_path):
    full_inputs = build_assignment_inputs(artifacts=artifacts)
    num_od = int(full_inputs.od_origin_node.shape[0])
    free_indices = tuple(range(1, num_od))
    layout = ODParameterLayout(
        num_od_total=num_od,
        od_keys=tuple((f"o{i}", "d", "t") for i in range(num_od)),
        free_od_indices=free_indices,
        fixed_od_indices=(0,),
        fixed_od_values=(4.0,),
        free_baseline_values=tuple(1.0 for _ in free_indices),
        fixed_zero_indices=(),
        fixed_positive_indices=(0,),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    reference = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="sharded-positive-fixed",
        compact_layout=compact,
        od_layout_fingerprint=layout.fingerprint,
        representation="dense",
        chunk_size=8,
    )
    origin_support = analyze_fixed_routing_origin_support(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
    )
    support_validation = validate_origin_support_against_operator(
        support=origin_support, operator=reference
    )
    assert support_validation.complete
    assert origin_support.positive_fixed_support.shape[1] == 1
    prepare_sharded_fixed_routing_measurement_operator(
        directory=tmp_path,
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        assignment_fingerprint="sharded-positive-fixed",
        od_layout_fingerprint=layout.fingerprint,
        config=ShardedConstructionConfig(
            od_chunk_size=8,
            measurement_block_size=2,
            worker_memory_budget_bytes=100_000_000,
        ),
    )
    sharded = ShardedSparseLinearOperator(tmp_path)
    demand = np.linspace(0.0, 2.0, layout.num_free)
    np.testing.assert_allclose(
        sharded.matvec(demand) + sharded.fixed_measurement_offset,
        np.asarray(reference.matrix) @ demand
        + np.asarray(reference.fixed_measurement_offset),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        sharded.fixed_measurement_offset,
        reference.fixed_measurement_offset,
        rtol=5e-5,
        atol=5e-5,
    )
    assert sharded.shape[1] == layout.num_free


def test_sharded_preflight_rejects_unsafe_kernel(all_free_operator):
    inputs, routing, spec, _ = all_free_operator
    num_od = int(inputs.od_origin_node.shape[0])
    layout = ODParameterLayout(
        num_od_total=num_od,
        od_keys=tuple((f"o{i}", "d", "t") for i in range(num_od)),
        free_od_indices=tuple(range(num_od)),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=tuple(1.0 for _ in range(num_od)),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    with pytest.raises(MemoryError, match="memory budget"):
        plan_sharded_fixed_routing_operator(
            inputs=inputs,
            routing=routing,
            spec=spec,
            compact_layout=compact,
            config=ShardedConstructionConfig(
                od_chunk_size=128,
                measurement_block_size=2048,
                worker_memory_budget_bytes=1,
            ),
        )


def test_origin_specific_support_contains_every_realized_entry(all_free_operator):
    inputs, routing, spec, operator = all_free_operator
    num_od = int(inputs.od_origin_node.shape[0])
    layout = ODParameterLayout(
        num_od_total=num_od,
        od_keys=tuple((f"o{i}", "d", "t") for i in range(num_od)),
        free_od_indices=tuple(range(num_od)),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=tuple(1.0 for _ in range(num_od)),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    support = analyze_fixed_routing_origin_support(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        config=OriginSupportConfig(origin_chunk_size=3),
    )
    validation = validate_origin_support_against_operator(
        support=support, operator=operator
    )
    assert validation.complete
    assert validation.missing_free_entries == 0
    assert support.metrics.origin_specific_entries <= (
        support.metrics.group_level_candidate_entries
    )
    assert support.metrics.free_support_entries >= operator.metrics.nonzero_entries


def test_origin_support_is_deterministic_across_chunk_sizes(all_free_operator):
    inputs, routing, spec, _ = all_free_operator
    num_od = int(inputs.od_origin_node.shape[0])
    layout = ODParameterLayout(
        num_od_total=num_od,
        od_keys=tuple((f"o{i}", "d", "t") for i in range(num_od)),
        free_od_indices=tuple(range(num_od)),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=tuple(1.0 for _ in range(num_od)),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    first = analyze_fixed_routing_origin_support(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        config=OriginSupportConfig(origin_chunk_size=1),
    )
    second = analyze_fixed_routing_origin_support(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        config=OriginSupportConfig(origin_chunk_size=8),
    )
    assert first.fingerprint == second.fingerprint
    np.testing.assert_array_equal(
        first.free_support.toarray(), second.free_support.toarray()
    )


def test_origin_support_summary_mode_and_memory_preflight(all_free_operator):
    inputs, routing, spec, _ = all_free_operator
    num_od = int(inputs.od_origin_node.shape[0])
    layout = ODParameterLayout(
        num_od_total=num_od,
        od_keys=tuple((f"o{i}", "d", "t") for i in range(num_od)),
        free_od_indices=tuple(range(num_od)),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=tuple(1.0 for _ in range(num_od)),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    summary = analyze_fixed_routing_origin_support(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        config=OriginSupportConfig(materialize=False),
    )
    assert not summary.materialized
    assert summary.metrics.origin_specific_entries > 0
    with pytest.raises(MemoryError, match="memory budget"):
        analyze_fixed_routing_origin_support(
            inputs=inputs,
            routing=routing,
            spec=spec,
            compact_layout=compact,
            config=OriginSupportConfig(worker_memory_budget_bytes=1),
        )


def test_compact_positive_frozen_flow_becomes_offset(artifacts):
    full_inputs = build_assignment_inputs(artifacts=artifacts)
    num_od = int(full_inputs.od_origin_node.shape[0])
    free_indices = tuple(range(1, num_od))
    layout = ODParameterLayout(
        num_od_total=num_od,
        od_keys=tuple((f"o{i}", "d", "t") for i in range(num_od)),
        free_od_indices=free_indices,
        fixed_od_indices=(0,),
        fixed_od_values=(4.0,),
        free_baseline_values=tuple(1.0 for _ in free_indices),
        fixed_zero_indices=(),
        fixed_positive_indices=(0,),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    operator = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="compact",
        compact_layout=compact,
        od_layout_fingerprint=layout.fingerprint,
        chunk_size=8,
    )
    free_demand = jnp.linspace(1.0, 3.0, layout.num_free)
    active_demand = jnp.zeros((compact.num_active,), dtype=free_demand.dtype)
    active_demand = active_demand.at[jnp.asarray(compact.free_compact_indices)].set(
        free_demand
    )
    active_demand = active_demand.at[jnp.asarray(compact.fixed_compact_indices)].set(
        jnp.asarray(compact.fixed_compact_values)
    )

    direct = predict_measurements_fixed_operator(
        operator=operator, free_demand=free_demand, rho=jnp.asarray(1.0)
    )
    reference = _reference(inputs, routing, spec, active_demand)
    np.testing.assert_allclose(direct, reference, rtol=3e-5, atol=3e-5)
    fixed_only = active_demand.at[jnp.asarray(compact.free_compact_indices)].set(0.0)
    np.testing.assert_allclose(
        operator.fixed_measurement_offset,
        _reference(inputs, routing, spec, fixed_only),
        rtol=3e-5,
        atol=3e-5,
    )


def test_provenance_mismatch_is_rejected(all_free_operator):
    inputs, routing, spec, operator = all_free_operator
    changed_spec = replace(
        spec,
        link_index=np.asarray([1], dtype=np.int32),
        measurement_index=np.asarray([0], dtype=np.int32),
    )
    assert measurement_mapping_fingerprint(changed_spec) != operator.mapping_fingerprint

    with pytest.raises(ValueError, match="mapping fingerprint mismatch"):
        validate_fixed_routing_measurement_operator(
            operator=operator,
            inputs=inputs,
            routing=routing,
            spec=changed_spec,
            assignment_fingerprint="simple-example-02",
            compact_layout=None,
            od_layout_fingerprint=None,
        )


def test_empty_compact_layout_produces_zero_measurements(artifacts):
    full_inputs = build_assignment_inputs(artifacts=artifacts)
    num_od = int(full_inputs.od_origin_node.shape[0])
    layout = ODParameterLayout(
        num_od_total=num_od,
        od_keys=tuple((f"o{i}", "d", "t") for i in range(num_od)),
        free_od_indices=(),
        fixed_od_indices=tuple(range(num_od)),
        fixed_od_values=tuple(0.0 for _ in range(num_od)),
        free_baseline_values=(),
        fixed_zero_indices=tuple(range(num_od)),
        fixed_positive_indices=(),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    operator = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="empty",
        compact_layout=compact,
        od_layout_fingerprint=layout.fingerprint,
    )

    prediction = predict_measurements_fixed_operator(
        operator=operator,
        free_demand=jnp.empty((0,), dtype=jnp.float32),
        rho=jnp.asarray(0.5),
    )
    assert operator.matrix.shape == (spec.num_measurements, 0)
    np.testing.assert_array_equal(prediction, np.zeros(spec.num_measurements))


def test_operator_records_memory_and_construction_metrics(all_free_operator):
    _, _, _, operator = all_free_operator
    metrics = operator.metrics
    assert metrics.construction_seconds >= 0.0
    assert metrics.dense_bytes == operator.num_measurements * operator.num_free_od * 4
    assert metrics.peak_construction_bytes >= 0
    assert 0 <= metrics.nonzero_entries <= metrics.total_entries
    assert metrics.density == pytest.approx(
        0.0
        if metrics.total_entries == 0
        else metrics.nonzero_entries / metrics.total_entries
    )
    assert metrics.compilation_count == 1
    assert metrics.chunk_shape == (8, operator.num_measurements)
    assert metrics.num_chunks > 0
    assert metrics.routing_loading_seconds >= 0.0
    assert metrics.device_synchronization_seconds >= 0.0
    assert metrics.numpy_transfer_seconds >= 0.0


def test_progress_is_reported_for_each_fixed_shape_chunk(artifacts):
    inputs = build_assignment_inputs(artifacts=artifacts)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    events = []
    operator = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=_spec(inputs.graph.num_links),
        assignment_fingerprint="progress",
        chunk_size=7,
        progress=events.append,
    )
    assert len(events) == operator.metrics.num_chunks
    assert all(
        tuple(event["shape"]) == (7, operator.num_measurements) for event in events
    )
    assert events[-1]["chunk"] == events[-1]["chunks"]


@pytest.mark.parametrize("representation", ["dense", "bcoo"])
def test_persistent_cache_reuse_and_invalid_file_rebuild(
    artifacts, tmp_path, representation
):
    inputs = build_assignment_inputs(artifacts=artifacts)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    kwargs = dict(
        cache_directory=tmp_path,
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="persistent",
        representation=representation,
        chunk_size=8,
    )
    built = load_or_prepare_fixed_routing_measurement_operator(**kwargs)
    loaded = load_or_prepare_fixed_routing_measurement_operator(**kwargs)
    assert not built.metrics.cache_hit
    assert loaded.metrics.cache_hit
    np.testing.assert_allclose(
        np.asarray(
            loaded.matrix.todense() if representation == "bcoo" else loaded.matrix
        ),
        np.asarray(
            built.matrix.todense() if representation == "bcoo" else built.matrix
        ),
    )
    path = fixed_routing_measurement_operator_cache_path(
        cache_directory=tmp_path,
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="persistent",
        representation=representation,
    )
    path.write_bytes(b"not a valid cache")
    rebuilt = load_or_prepare_fixed_routing_measurement_operator(**kwargs)
    assert not rebuilt.metrics.cache_hit


def test_sparse_cache_identity_includes_zero_tolerance(artifacts, tmp_path):
    inputs = build_assignment_inputs(artifacts=artifacts)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    common = dict(
        cache_directory=tmp_path,
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="zero-tolerance",
        representation="bcoo",
    )
    exact_path = fixed_routing_measurement_operator_cache_path(
        **common, zero_tolerance=0.0
    )
    truncated_path = fixed_routing_measurement_operator_cache_path(
        **common, zero_tolerance=0.1
    )
    assert exact_path != truncated_path

    exact = load_or_prepare_fixed_routing_measurement_operator(
        **common, zero_tolerance=0.0
    )
    truncated = load_or_prepare_fixed_routing_measurement_operator(
        **common, zero_tolerance=0.1
    )
    assert exact.zero_tolerance == 0.0
    assert truncated.zero_tolerance == 0.1
    assert exact_path.exists()
    assert truncated_path.exists()


def test_sparse_loader_rejects_out_of_bounds_indices(artifacts, tmp_path):
    inputs = build_assignment_inputs(artifacts=artifacts)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    operator = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="invalid-indices",
        representation="bcoo",
    )
    path = tmp_path / "operator.npz"
    save_fixed_routing_measurement_operator(operator, path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    indices = np.array(payload["indices"], copy=True)
    indices[0, 0] = operator.num_measurements
    payload["indices"] = indices
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="indices are out of bounds"):
        load_fixed_routing_measurement_operator(path)


def test_auto_activation_policy_respects_cache_and_break_even():
    assert (
        choose_fixed_measurement_operator(
            mode="off", cached=True, expected_evaluations=100, construction_seconds=1.0
        )
        is None
    )
    assert (
        choose_fixed_measurement_operator(
            mode="dense",
            cached=False,
            expected_evaluations=0,
            construction_seconds=None,
        )
        == "dense"
    )
    assert (
        choose_fixed_measurement_operator(
            mode="auto", cached=True, expected_evaluations=0, construction_seconds=None
        )
        == "bcoo"
    )
    assert (
        choose_fixed_measurement_operator(
            mode="auto",
            cached=False,
            expected_evaluations=5,
            construction_seconds=20.0,
            reference_evaluation_seconds=1.94,
        )
        is None
    )
    assert (
        choose_fixed_measurement_operator(
            mode="auto",
            cached=False,
            expected_evaluations=20,
            construction_seconds=20.0,
            reference_evaluation_seconds=1.94,
        )
        == "bcoo"
    )


def test_ml_problem_cache_hit_does_not_prepare_routing(
    artifacts, tmp_path, monkeypatch
):
    inputs = build_assignment_inputs(artifacts=artifacts)
    num_od = int(inputs.od_origin_node.shape[0])
    request = ODThetaEstimationRequest(
        fingerprint="routing-free-cache-hit",
        f0=jnp.ones((num_od,)),
        y_obs=jnp.asarray([2.0, 1.0, 3.0]),
        mapping_spec=_spec(inputs.graph.num_links),
        baseline_theta=1.0,
        estimate_theta=False,
        fixed_theta=1.0,
        assignment_artifacts=artifacts,
        fixed_measurement_operator="bcoo",
        fixed_measurement_operator_cache_directory=tmp_path,
        fixed_measurement_operator_chunk_size=8,
    )
    built = build_od_theta_ml_problem(request)
    assert built.fixed_measurement_operator is not None
    assert not built.fixed_measurement_operator.metrics.cache_hit

    import public_transportation.inference.maximum_likelihood_pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "prepare_fixed_routing",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("routing must not be prepared on a valid cache hit")
        ),
    )
    loaded = build_od_theta_ml_problem(request)
    assert loaded.fixed_measurement_operator is not None
    assert loaded.fixed_measurement_operator.metrics.cache_hit


def test_ml_likelihood_objective_gradient_and_solution_inputs_are_equivalent(artifacts):
    inputs = build_assignment_inputs(artifacts=artifacts)
    spec = _spec(inputs.graph.num_links)
    num_od = int(inputs.od_origin_node.shape[0])
    common = dict(
        fingerprint="simple-example-02",
        f0=jnp.linspace(1.0, 2.0, num_od),
        y_obs=jnp.asarray([2.0, 1.0, 3.0]),
        mapping_spec=spec,
        baseline_theta=1.0,
        estimate_theta=False,
        fixed_theta=1.0,
        rho=0.8,
        nb_dispersion=10.0,
        assignment_artifacts=artifacts,
    )
    reference = build_od_theta_ml_problem(
        ODThetaEstimationRequest(**common, fixed_measurement_operator="off")
    )
    optimized = build_od_theta_ml_problem(
        ODThetaEstimationRequest(**common, fixed_measurement_operator="dense")
    )

    def objective(problem, parameter):
        return -(problem.loglik(parameter, problem.data) + problem.logprior(parameter))

    for parameter in (
        jnp.zeros((num_od,)),
        jnp.linspace(-0.3, 0.4, num_od),
        jnp.linspace(0.2, -0.1, num_od),
    ):
        reference_value, reference_gradient = jax.value_and_grad(
            lambda value: objective(reference, value)
        )(parameter)
        optimized_value, optimized_gradient = jax.value_and_grad(
            lambda value: objective(optimized, value)
        )(parameter)
        np.testing.assert_allclose(
            optimized.loglik(parameter, optimized.data),
            reference.loglik(parameter, reference.data),
            rtol=5e-5,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            optimized_value,
            reference_value,
            rtol=5e-5,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            optimized_gradient,
            reference_gradient,
            rtol=8e-5,
            atol=8e-5,
        )
