from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import public_transportation.inference.fixed_routing_sharded_builder as sharded_builder

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
    ShardedConstructionPreflightError,
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
    load_sharded_operator_manifest,
    load_sparse_shard,
    shard_path,
)
from public_transportation.inference.construction_control import (
    ConstructionDeadline,
    ConstructionDeadlineStop,
)
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
    prepare_fixed_routing_sharded,
)
from public_transportation.inference.sharded_matrix_free_operator import (
    ShardedOperatorProductInterrupted,
    ShardedMatrixFreeFixedRoutingMeasurementOperator,
)
from public_transportation.inference.measurement_operator_protocol import (
    GravityMeasurementOperator,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout
from public_transportation.inference.maximum_likelihood_pipeline import (
    build_od_theta_ml_problem,
)
from public_transportation.inference.gravity import (
    GravityFeatures,
    GravityLikelihood,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    gravity_value_and_gradient_adjoint,
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


def test_sharded_matrix_free_products_match_complete_operator(
    all_free_operator, tmp_path
):
    inputs, _, spec, complete_operator = all_free_operator
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
    prepared = prepare_fixed_routing_sharded(
        inputs=inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=2,
            cache_directory=tmp_path,
            checkpoint_directory=tmp_path / "checkpoint",
            resident_shard_limit=1,
        ),
    )
    operator = ShardedMatrixFreeFixedRoutingMeasurementOperator(
        inputs=inputs,
        routing=prepared.routing,
        spec=spec,
        compact_layout=compact,
        resident_shard_limit=1,
        operator_shards_per_batch=3,
    )
    demand = jnp.linspace(0.1, 5.0, num_od, dtype=jnp.float32)
    weight = jnp.array([0.5, -0.2, 1.1], dtype=jnp.float32)

    expected = complete_operator.jax_matvec(demand)
    actual = operator.jax_matvec(demand)
    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(
        operator.jax_rmatvec(weight),
        complete_operator.jax_rmatvec(weight),
        rtol=4e-5,
        atol=4e-5,
    )
    compiled = jax.jit(operator.jax_matvec)(demand)
    np.testing.assert_allclose(compiled, expected, rtol=4e-5, atol=4e-5)
    gradient = jax.grad(lambda value: jnp.vdot(operator.jax_matvec(value), weight))(
        demand
    )
    np.testing.assert_allclose(
        gradient,
        complete_operator.jax_rmatvec(weight),
        rtol=4e-5,
        atol=4e-5,
    )
    assert operator.resident_shards <= 1
    assert operator.is_matrix_free
    assert operator.representation == "matrix_free_sharded"
    assert isinstance(operator, GravityMeasurementOperator)
    assert operator.product_capabilities.absolute_deadline
    assert operator.metrics.compilation_count == 2
    assert operator.metrics.product_count >= 2

    matrix = jnp.column_stack((demand, 2.0 * demand, jnp.zeros_like(demand)))
    np.testing.assert_allclose(
        operator.jax_matmat(matrix),
        jnp.column_stack(
            [complete_operator.jax_matvec(matrix[:, index]) for index in range(3)]
        ),
        rtol=4e-5,
        atol=4e-5,
    )

    progress = []
    operator.progress_callback = progress.append
    operator.matvec(np.asarray(demand))
    assert progress[0].phase == "product_started"
    assert progress[-1].phase == "product_completed"
    assert progress[-1].completed_shards == progress[-1].total_shards
    assert all(event.completed_shards <= event.total_shards for event in progress)
    assert operator.resident_shards <= operator.resident_shard_limit

    operator.absolute_deadline = 0.0
    with pytest.raises(ShardedOperatorProductInterrupted):
        operator.matvec(np.asarray(demand))
    operator.absolute_deadline = None

    predictive_progress = []
    operator.progress_callback = predictive_progress.append
    operator.initial_predicted_batch_seconds = 10.0
    operator._predicted_batch_seconds.clear()
    operator.absolute_deadline = perf_counter() + 1.0
    with pytest.raises(ShardedOperatorProductInterrupted) as interrupted:
        operator.matvec(np.asarray(demand))
    assert interrupted.value.completed_shards == 0
    assert predictive_progress[-1].phase == "product_deadline_infeasible"
    assert predictive_progress[-1].predicted_remaining_seconds is not None
    operator.absolute_deadline = None
    operator.initial_predicted_batch_seconds = None

    for strategy, options in (
        ("vectorized", {"operator_shards_per_batch": 3}),
        (
            "concurrent",
            {
                "shard_execution_strategy": "concurrent",
                "operator_concurrency": 2,
                "operator_shards_per_batch": 1,
            },
        ),
    ):
        candidate = ShardedMatrixFreeFixedRoutingMeasurementOperator(
            inputs=inputs,
            routing=prepared.routing,
            spec=spec,
            compact_layout=compact,
            resident_shard_limit=1,
            group_execution_strategy=(
                "vectorized" if strategy == "vectorized" else "scan"
            ),
            **options,
        )
        first_result = candidate.matvec(np.asarray(demand))
        second_result = candidate.matvec(np.asarray(demand))
        np.testing.assert_array_equal(first_result, second_result)
        np.testing.assert_allclose(first_result, expected, rtol=4e-5, atol=4e-5)
        transpose = candidate.rmatvec(np.asarray(weight))
        np.testing.assert_allclose(
            transpose,
            complete_operator.jax_rmatvec(weight),
            rtol=4e-5,
            atol=4e-5,
        )
        assert float(np.vdot(first_result, weight)) == pytest.approx(
            float(np.vdot(demand, transpose)), rel=4e-5, abs=4e-5
        )
        np.testing.assert_allclose(
            candidate._host_matmat(np.asarray(matrix)),
            np.column_stack(
                (first_result, 2.0 * first_result, np.zeros_like(first_result))
            ),
            rtol=4e-5,
            atol=4e-5,
        )
        assert candidate.resident_shards <= candidate.resident_shard_limit
        assert candidate.assignment_fingerprint == operator.assignment_fingerprint

    memory_limited = ShardedMatrixFreeFixedRoutingMeasurementOperator(
        inputs=inputs,
        routing=prepared.routing,
        spec=spec,
        compact_layout=compact,
        shard_execution_strategy="concurrent",
        operator_concurrency=100,
        maximum_concurrent_routing_bytes=1,
    )
    assert memory_limited.effective_operator_concurrency == 1

    deadline_events = []

    def make_remaining_product_infeasible(event):
        deadline_events.append(event)
        if event.phase == "batch_accumulated" and event.completed_shards == 2:
            assert event.recent_batch_seconds is not None
            concurrent_deadline.absolute_deadline = (
                perf_counter() + 0.25 * event.recent_batch_seconds
            )

    concurrent_deadline = ShardedMatrixFreeFixedRoutingMeasurementOperator(
        inputs=inputs,
        routing=prepared.routing,
        spec=spec,
        compact_layout=compact,
        resident_shard_limit=1,
        shard_execution_strategy="concurrent",
        operator_concurrency=2,
        progress_callback=make_remaining_product_infeasible,
    )
    with pytest.raises(ShardedOperatorProductInterrupted) as stopped:
        concurrent_deadline.matvec(np.asarray(demand))
    assert stopped.value.reason == "product_deadline_infeasible"
    assert stopped.value.completed_shards == 2
    assert stopped.value.predicted_remaining_seconds is not None
    assert stopped.value.discarded_partial_seconds > 0.0
    assert [event.phase for event in deadline_events].count("batch_loaded") == 1
    diagnostic = deadline_events[-1]
    assert diagnostic.phase == "product_deadline_infeasible"
    assert diagnostic.batch_size == 1
    assert diagnostic.concurrency == 2
    assert diagnostic.discarded_partial_seconds > 0.0

    failing = ShardedMatrixFreeFixedRoutingMeasurementOperator(
        inputs=inputs,
        routing=prepared.routing,
        spec=spec,
        compact_layout=compact,
        shard_execution_strategy="concurrent",
        operator_concurrency=2,
    )

    def fail_kernel(*_):
        raise RuntimeError("worker kernel failed")

    failing._compiled_forward = fail_kernel
    with pytest.raises(RuntimeError, match="worker kernel failed"):
        failing.matvec(np.asarray(demand))

    features = GravityFeatures(
        canonical_od_index=np.arange(num_od),
        origin_index=np.arange(num_od),
        destination_index=np.zeros(num_od, dtype=np.int32),
        departure_time_index=np.zeros(num_od, dtype=np.int32),
        origin_time_group_index=np.arange(num_od),
        journey_time=np.linspace(1.0, 2.0, num_od),
        transfer_count=np.zeros(num_od, dtype=np.int32),
        structural_feasible=np.ones(num_od, dtype=bool),
        origin_time_totals=np.linspace(1.0, 3.0, num_od),
        destination_attractiveness=np.ones(num_od),
        num_origins=num_od,
        num_destinations=1,
        num_departure_times=1,
        od_layout_fingerprint=compact.fingerprint,
        journey_time_scale=1.0,
    )
    parameter_layout = GravityParameterLayout(GravityModelSpecification())
    raw = parameter_layout.raw_from_physical((0.5, 1.0, 10.0))
    observations = np.asarray((2.0, 3.0, 1.0))
    for likelihood in (GravityLikelihood.POISSON, GravityLikelihood.NEGATIVE_BINOMIAL):
        complete_problem = GravityObjectiveProblem(
            features=features,
            parameter_layout=parameter_layout,
            operator=complete_operator,
            observations=observations,
            likelihood=likelihood,
        )
        sharded_problem = GravityObjectiveProblem(
            features=features,
            parameter_layout=parameter_layout,
            operator=operator,
            observations=observations,
            likelihood=likelihood,
        )
        expected_evaluation, expected_gradient = gravity_value_and_gradient_adjoint(
            raw, problem=complete_problem
        )
        actual_evaluation, actual_gradient = gravity_value_and_gradient_adjoint(
            raw, problem=sharded_problem
        )
        np.testing.assert_allclose(
            actual_evaluation.measurement_mean,
            expected_evaluation.measurement_mean,
            rtol=4e-5,
            atol=4e-5,
        )
        np.testing.assert_allclose(
            actual_gradient, expected_gradient, rtol=5e-5, atol=5e-5
        )


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
    np.testing.assert_allclose(
        cached.operator.matvec(demand), built.operator.matvec(demand)
    )
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
    assert (tmp_path / "manifest.json").stat().st_size <= (
        built.plan.estimated_manifest_bytes
    )
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


def _parallel_builder_case(
    all_free_operator,
    directory,
    *,
    workers,
    max_materialized_support_entries=125_000_000,
    maximum_storage_shards=256,
    **kwargs,
):
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
    config = ShardedConstructionConfig(
        od_chunk_size=8,
        measurement_block_size=1,
        worker_memory_budget_bytes=100_000_000,
        workers=workers,
        max_materialized_support_entries=max_materialized_support_entries,
        maximum_storage_shards=maximum_storage_shards,
        maximum_resident_shards=2,
        target_nonzeros_per_storage_shard=1,
        maximum_nonzeros_per_storage_shard=1,
        manifest_checkpoint_shards=1,
        progress_interval_seconds=0.0,
        deadline_safety_margin_seconds=0.0,
    )
    return prepare_sharded_fixed_routing_measurement_operator(
        directory=directory,
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        assignment_fingerprint="parallel-measurement-shards",
        od_layout_fingerprint=layout.fingerprint,
        config=config,
        **kwargs,
    )


@pytest.fixture(scope="module")
def sharded_preflight_case(all_free_operator):
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
    config = ShardedConstructionConfig(
        od_chunk_size=8,
        measurement_block_size=1,
        worker_memory_budget_bytes=1_000_000_000,
        target_nonzeros_per_storage_shard=1,
        maximum_nonzeros_per_storage_shard=1,
        maximum_patterns_per_storage_shard=256,
        maximum_storage_shards=1_000_000,
        maximum_manifest_bytes=1_000_000_000,
        maximum_filesystem_operations=1_000_000_000,
        maximum_sparse_calls_per_product=1_000_000,
        maximum_construction_dispatches=1_000_000_000,
    )
    plan, support = plan_sharded_fixed_routing_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        config=config,
    )
    return inputs, routing, spec, compact, config, plan, support


def test_sharded_preflight_safe_plan_is_complete_and_serializable(
    sharded_preflight_case,
):
    _, _, _, _, config, plan, _ = sharded_preflight_case
    assert plan.safe
    diagnostics = plan.preflight_diagnostics()
    assert all(not item["exceeded"] for item in diagnostics["limits"].values())
    assert plan.estimated_manifest_writes == (
        2 + math.ceil(plan.num_shards / config.manifest_checkpoint_shards)
    )
    assert plan.estimated_filesystem_operations == (
        plan.num_shards * 5 + plan.estimated_manifest_writes * 3
    )
    assert plan.estimated_sparse_calls_per_product == plan.num_shards
    assert plan.estimated_construction_dispatches >= plan.construction_tasks
    assert plan.estimated_worker_memory_bytes == (
        plan.estimated_kernel_bytes
        + plan.estimated_batch_temporary_bytes
        + plan.estimated_maximum_staged_shard_bytes
    )
    assert diagnostics["storage_shard_sizing"] == {
        "target_nonzeros": config.target_nonzeros_per_storage_shard,
        "maximum_nonzeros": config.maximum_nonzeros_per_storage_shard,
        "maximum_patterns": config.maximum_patterns_per_storage_shard,
    }
    json.dumps(diagnostics, sort_keys=True)
    json.dumps(asdict(plan), sort_keys=True)
    json.dumps(asdict(config), sort_keys=True)


@pytest.mark.parametrize(
    ("config_field", "diagnostic_name", "plan_field"),
    [
        ("maximum_storage_shards", "storage_shards", "num_shards"),
        ("maximum_manifest_bytes", "manifest_bytes", "estimated_manifest_bytes"),
        (
            "maximum_filesystem_operations",
            "filesystem_operations",
            "estimated_filesystem_operations",
        ),
        (
            "maximum_sparse_calls_per_product",
            "sparse_calls_per_product",
            "estimated_sparse_calls_per_product",
        ),
        (
            "maximum_construction_dispatches",
            "construction_dispatches",
            "estimated_construction_dispatches",
        ),
    ],
)
def test_each_sharded_operational_limit_reports_actual_and_permitted(
    sharded_preflight_case, config_field, diagnostic_name, plan_field
):
    inputs, routing, spec, compact, config, safe_plan, support = sharded_preflight_case
    actual = getattr(safe_plan, plan_field)
    assert actual > 1
    permitted = actual - 1
    rejected, _ = plan_sharded_fixed_routing_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        config=replace(config, **{config_field: permitted}),
        discovered_support=support,
    )
    assert not rejected.safe
    assert (
        f"{diagnostic_name}: actual={actual}, permitted={permitted}" in rejected.reason
    )
    detail = rejected.preflight_diagnostics()["limits"][diagnostic_name]
    assert detail == {
        "actual": actual,
        "permitted": permitted,
        "exceeded": True,
    }


def test_sharded_preflight_combines_failures_and_preserves_memoryerror(
    sharded_preflight_case,
):
    inputs, routing, spec, compact, config, _, support = sharded_preflight_case
    rejected, _ = plan_sharded_fixed_routing_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        config=replace(
            config,
            maximum_storage_shards=1,
            maximum_manifest_bytes=1,
            maximum_filesystem_operations=1,
            maximum_sparse_calls_per_product=1,
            maximum_construction_dispatches=1,
            worker_memory_budget_bytes=1,
        ),
        discovered_support=support,
    )
    diagnostics = rejected.preflight_diagnostics()
    assert not rejected.safe
    assert all(item["exceeded"] for item in diagnostics["limits"].values())
    error = ShardedConstructionPreflightError(rejected)
    assert isinstance(error, MemoryError)
    assert error.plan is rejected
    assert error.details == diagnostics
    assert str(error) == rejected.reason


def test_sharded_builder_raises_structured_preflight_memoryerror(
    all_free_operator, tmp_path
):
    with pytest.raises(ShardedConstructionPreflightError) as caught:
        _parallel_builder_case(
            all_free_operator,
            tmp_path,
            workers=1,
            maximum_storage_shards=1,
        )
    assert isinstance(caught.value, MemoryError)
    storage = caught.value.details["limits"]["storage_shards"]
    assert storage["actual"] > storage["permitted"]


def test_parallel_measurement_shards_match_serial_content(all_free_operator, tmp_path):
    serial = _parallel_builder_case(all_free_operator, tmp_path / "serial", workers=1)
    parallel = _parallel_builder_case(
        all_free_operator, tmp_path / "parallel", workers=2
    )
    assert serial.manifest.provenance == parallel.manifest.provenance
    assert serial.manifest.expected_shards == parallel.manifest.expected_shards
    assert serial.compilation_count == 2
    assert parallel.compilation_count == 2
    assert serial.reachability_evaluations == (
        serial.plan.estimated_reachability_evaluations
    )
    assert serial.edge_gather_evaluations == (
        serial.plan.estimated_edge_gather_evaluations
    )
    assert parallel.reachability_evaluations == serial.reachability_evaluations
    assert parallel.edge_gather_evaluations == serial.edge_gather_evaluations
    assert serial.edge_gather_evaluations >= serial.reachability_evaluations
    assert parallel.requested_workers == 2
    assert parallel.admitted_workers == 2
    for identity in serial.manifest.expected_shards:
        left = load_sparse_shard(shard_path(serial.directory, identity))
        right = load_sparse_shard(shard_path(parallel.directory, identity))
        assert left.metadata.content_hash == right.metadata.content_hash
        np.testing.assert_array_equal(left.row_indices, right.row_indices)
        np.testing.assert_array_equal(left.matrix.toarray(), right.matrix.toarray())
        np.testing.assert_array_equal(
            left.fixed_offset_values, right.fixed_offset_values
        )
    demand = np.linspace(0.0, 2.0, serial.plan.num_free_od)
    cotangent = np.linspace(-0.5, 1.0, serial.plan.num_measurements)
    serial_operator = ShardedSparseLinearOperator(serial.directory)
    parallel_operator = ShardedSparseLinearOperator(parallel.directory)
    np.testing.assert_allclose(
        serial_operator.matvec(demand), parallel_operator.matvec(demand)
    )
    np.testing.assert_allclose(
        serial_operator.rmatvec(cotangent),
        parallel_operator.rmatvec(cotangent),
    )


def test_kernel_version_rebuilds_shards_but_reuses_support_checkpoint(
    all_free_operator, tmp_path, monkeypatch
):
    built = _parallel_builder_case(all_free_operator, tmp_path, workers=1)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"].pop("measurement_kernel_algorithm", None)
    manifest["provenance_hash"] = hashlib.sha256(
        json.dumps(
            manifest["provenance"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    for identity in built.manifest.expected_shards:
        path = shard_path(tmp_path, identity)
        with np.load(path, allow_pickle=False) as archive:
            shard = {name: archive[name] for name in archive.files}
        metadata = json.loads(str(shard["metadata"]))
        metadata["provenance_hash"] = manifest["provenance_hash"]
        shard["metadata"] = np.asarray(json.dumps(metadata))
        np.savez(path, **shard)

    def support_must_be_reused(**_kwargs):
        raise AssertionError("support discovery should have been reused")

    monkeypatch.setattr(sharded_builder, "_discover_support", support_must_be_reused)
    rebuilt = _parallel_builder_case(all_free_operator, tmp_path, workers=1)
    assert rebuilt.rebuilt_shards == built.plan.num_shards
    assert rebuilt.rejected_shards == built.plan.num_shards
    assert (
        rebuilt.manifest.provenance["measurement_kernel_algorithm"]
        == sharded_builder.MEASUREMENT_KERNEL_ALGORITHM_VERSION
    )


def test_parallel_publication_is_canonical_when_workers_finish_out_of_order(
    all_free_operator, tmp_path, monkeypatch
):
    original = sharded_builder._construct_measurement_shard

    def delayed(**kwargs):
        position = kwargs["identity"].storage_shard
        if position == 0:
            time.sleep(0.05)
        return original(**kwargs)

    monkeypatch.setattr(sharded_builder, "_construct_measurement_shard", delayed)
    published = []
    result = _parallel_builder_case(
        all_free_operator,
        tmp_path,
        workers=2,
        progress=lambda event: (
            published.append(event.get("current_unit") or event["shard"])
            if (event.get("current_unit") or event.get("shard"))
            and event.get("nonzero_entries") is not None
            else None
        ),
    )
    assert result.maximum_buffered_shards >= 1
    assert published == [item.key for item in result.manifest.expected_shards]
    assert tuple(result.manifest.completed_shards) == tuple(
        item.key for item in result.manifest.expected_shards
    )


def test_parallel_deadline_waits_for_inflight_shards_and_resumes(
    all_free_operator, tmp_path, monkeypatch
):
    original = sharded_builder._construct_measurement_shard
    clock_state = {"now": 0.0}

    def expires_inflight(**kwargs):
        result = original(**kwargs)
        clock_state["now"] = 100.0
        return result

    monkeypatch.setattr(
        sharded_builder, "_construct_measurement_shard", expires_inflight
    )
    deadline = ConstructionDeadline(
        started_at=0.0,
        absolute_deadline=10.0,
        safety_margin_seconds=0.0,
        clock=lambda: clock_state["now"],
    )
    with pytest.raises(ConstructionDeadlineStop) as stopped:
        _parallel_builder_case(
            all_free_operator,
            tmp_path,
            workers=2,
            deadline=deadline,
        )
    assert stopped.value.termination.completed_units == 2
    partial = load_sharded_operator_manifest(tmp_path)
    assert len(partial.completed_shards) == 2
    monkeypatch.setattr(sharded_builder, "_construct_measurement_shard", original)
    resumed = _parallel_builder_case(all_free_operator, tmp_path, workers=2)
    assert resumed.manifest.complete
    assert resumed.reused_shards == 2


def test_parallel_failure_cleans_staging_and_checkpoint_remains_reusable(
    all_free_operator, tmp_path, monkeypatch
):
    original = sharded_builder._construct_measurement_shard

    def fails_second(**kwargs):
        if kwargs["identity"].storage_shard == 1:
            raise RuntimeError("injected worker failure")
        return original(**kwargs)

    monkeypatch.setattr(sharded_builder, "_construct_measurement_shard", fails_second)
    with pytest.raises(RuntimeError, match="measurement-shard worker failed"):
        _parallel_builder_case(all_free_operator, tmp_path, workers=2)
    assert not tuple(tmp_path.glob(".measurement-shards-*"))
    partial = load_sharded_operator_manifest(tmp_path)
    assert set(partial.completed_shards).issubset(
        {item.key for item in partial.expected_shards}
    )
    monkeypatch.setattr(sharded_builder, "_construct_measurement_shard", original)
    resumed = _parallel_builder_case(all_free_operator, tmp_path, workers=2)
    assert resumed.manifest.complete


def test_parallel_corrupted_shard_is_repaired(all_free_operator, tmp_path):
    built = _parallel_builder_case(all_free_operator, tmp_path, workers=2)
    damaged_path = shard_path(tmp_path, built.plan.expected_shards[0])
    with np.load(damaged_path, allow_pickle=False) as archive:
        damaged = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(damaged["metadata"]))
    metadata["content_hash"] = "0" * 64
    damaged["metadata"] = np.asarray(json.dumps(metadata))
    np.savez(damaged_path, **damaged)
    repaired = _parallel_builder_case(all_free_operator, tmp_path, workers=2)
    assert repaired.rejected_shards == 1
    assert repaired.rebuilt_shards == 1
    assert repaired.reused_shards == repaired.plan.num_shards - 1


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
    serial = prepare_sharded_fixed_routing_measurement_operator(
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
    parallel = prepare_sharded_fixed_routing_measurement_operator(
        directory=tmp_path / "parallel-fixed-offset",
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
            workers=2,
            maximum_resident_shards=2,
        ),
    )
    for identity in serial.manifest.expected_shards:
        serial_shard = load_sparse_shard(shard_path(serial.directory, identity))
        parallel_shard = load_sparse_shard(shard_path(parallel.directory, identity))
        assert serial_shard.metadata.content_hash == (
            parallel_shard.metadata.content_hash
        )
    sharded = ShardedSparseLinearOperator(tmp_path)
    parallel_sharded = ShardedSparseLinearOperator(parallel.directory)
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
    np.testing.assert_array_equal(
        sharded.fixed_measurement_offset,
        parallel_sharded.fixed_measurement_offset,
    )
    assert sharded.shape[1] == layout.num_free

    prepared_routing = prepare_fixed_routing_sharded(
        inputs=inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=2,
            cache_directory=tmp_path / "routing-probabilities",
            checkpoint_directory=tmp_path / "routing-checkpoints",
        ),
    )
    concurrent = ShardedMatrixFreeFixedRoutingMeasurementOperator(
        inputs=inputs,
        routing=prepared_routing.routing,
        spec=spec,
        compact_layout=compact,
        resident_shard_limit=1,
        shard_execution_strategy="concurrent",
        operator_concurrency=2,
    )
    np.testing.assert_allclose(
        concurrent.matvec(demand) + concurrent.fixed_measurement_offset,
        np.asarray(reference.matrix) @ demand
        + np.asarray(reference.fixed_measurement_offset),
        rtol=5e-5,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        concurrent.fixed_measurement_offset,
        reference.fixed_measurement_offset,
        rtol=5e-5,
        atol=5e-5,
    )


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
        config=OriginSupportConfig(materialize=False, max_materialized_entries=1),
    )
    assert not summary.materialized
    assert summary.metrics.origin_specific_entries > 0
    actual_entries = summary.metrics.origin_specific_entries
    assert actual_entries > 1
    with pytest.raises(
        MemoryError,
        match=(
            f"origin-specific support has {actual_entries} entries, exceeding "
            f"max_materialized_entries={actual_entries - 1}"
        ),
    ):
        analyze_fixed_routing_origin_support(
            inputs=inputs,
            routing=routing,
            spec=spec,
            compact_layout=compact,
            config=OriginSupportConfig(max_materialized_entries=actual_entries - 1),
        )
    sufficient = analyze_fixed_routing_origin_support(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        config=OriginSupportConfig(max_materialized_entries=actual_entries),
    )
    generous = analyze_fixed_routing_origin_support(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact,
        config=OriginSupportConfig(max_materialized_entries=actual_entries + 10_000),
    )
    assert sufficient.fingerprint == generous.fingerprint
    np.testing.assert_array_equal(
        sufficient.free_support.toarray(), generous.free_support.toarray()
    )
    with pytest.raises(MemoryError, match="memory budget"):
        analyze_fixed_routing_origin_support(
            inputs=inputs,
            routing=routing,
            spec=spec,
            compact_layout=compact,
            config=OriginSupportConfig(worker_memory_budget_bytes=1),
        )


def test_sharded_builder_support_cap_fails_then_reuses_group_checkpoints(
    all_free_operator, tmp_path
):
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
    actual_entries = summary.metrics.origin_specific_entries
    with pytest.raises(MemoryError, match=f"has {actual_entries} entries"):
        _parallel_builder_case(
            all_free_operator,
            tmp_path,
            workers=1,
            max_materialized_support_entries=actual_entries - 1,
        )
    checkpoints = tuple((tmp_path / "support_groups").glob("group-*.npz"))
    assert checkpoints
    completed = _parallel_builder_case(
        all_free_operator,
        tmp_path,
        workers=1,
        max_materialized_support_entries=actual_entries,
    )
    assert completed.manifest.complete
    assert tuple((tmp_path / "support_groups").glob("group-*.npz")) == checkpoints
    generous = _parallel_builder_case(
        all_free_operator,
        tmp_path / "generous",
        workers=1,
        max_materialized_support_entries=actual_entries + 10_000,
    )
    assert completed.manifest.provenance == generous.manifest.provenance
    for identity in completed.manifest.expected_shards:
        exact = load_sparse_shard(shard_path(completed.directory, identity))
        roomy = load_sparse_shard(shard_path(generous.directory, identity))
        assert exact.metadata.content_hash == roomy.metadata.content_hash
    demand = np.linspace(0.0, 2.0, completed.plan.num_free_od)
    np.testing.assert_allclose(
        ShardedSparseLinearOperator(completed.directory).matvec(demand),
        ShardedSparseLinearOperator(generous.directory).matvec(demand),
    )
    with pytest.raises(ValueError, match="max_materialized_support_entries"):
        ShardedConstructionConfig(max_materialized_support_entries=0)


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
