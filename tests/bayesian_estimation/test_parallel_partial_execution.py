from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from benchmarks.benchmark_sharded_gravity_operator import _inputs, _layout
from public_transportation.inference.parallel_partial_execution import (
    PartialExecutionBatch,
    PartialExecutionPlan,
    RoutingCostModel,
    plan_fixed_budget_routing_selection,
    RoutingWorkUnit,
    ShardedWorkInstrumentation,
    build_balanced_microshard_plan,
    routing_group_work_units,
)
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
    prepare_fixed_routing_sharded,
)
from public_transportation.inference.sharded_matrix_free_operator import (
    ShardedMatrixFreeFixedRoutingMeasurementOperator,
)
from public_transportation.measurement.mapping import AggregationSpec


def _work(index: int, cost: float, *, stratum: str = "default") -> RoutingWorkUnit:
    return RoutingWorkUnit(
        work_id=f"group-{index}",
        destination_group_indices=(index,),
        predicted_cost=cost,
        routing_bytes=100,
        active_od_cells=2,
        support_entries=3,
        measurement_support=1,
        stratum=stratum,
    )


def _operator(root: Path, *, progress_callback=None):
    inputs = _inputs(nodes=32, degree=2, groups=8, od_cells=32)
    prepared = prepare_fixed_routing_sharded(
        inputs=inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=2,
            cache_directory=root / "routing",
            checkpoint_directory=root / "checkpoints",
            resident_shard_limit=1,
        ),
    )
    measurements = 16
    spec = AggregationSpec(
        num_measurements=measurements,
        measurement_index=np.arange(measurements, dtype=np.int32),
        link_index=np.arange(measurements, dtype=np.int32) % inputs.graph.num_links,
    )
    return ShardedMatrixFreeFixedRoutingMeasurementOperator(
        inputs=inputs,
        routing=prepared.routing,
        spec=spec,
        compact_layout=_layout(32),
        resident_shard_limit=1,
        operator_shards_per_batch=2,
        progress_callback=progress_callback,
    )


def test_work_contract_validation_and_execution_plan_json_roundtrip():
    with pytest.raises(ValueError, match="positive and finite"):
        replace(_work(0, 1.0), predicted_cost=0.0)
    batch = PartialExecutionBatch("batch-0", ("a", "b"))
    plan = PartialExecutionPlan(
        problem_fingerprint="problem",
        microshard_plan_fingerprint="microshards",
        requested_effort_percent=25.0,
        realized_effort_percent=24.0,
        selected_work_ids=("a", "b"),
        batches=(batch,),
        selection_seed=4,
        execution_fingerprint="execution",
    )
    restored = PartialExecutionPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert restored == plan
    with pytest.raises(ValueError, match="partition selected work"):
        replace(plan, selected_work_ids=("a", "b", "c"))


def test_balanced_microshards_are_deterministic_complete_and_serializable():
    work = tuple(_work(index, cost) for index, cost in enumerate((9, 8, 7, 6, 5, 4)))
    first = build_balanced_microshard_plan(
        work, target_microshards=3, problem_fingerprint="problem"
    )
    second = build_balanced_microshard_plan(
        tuple(reversed(work)), target_microshards=3, problem_fingerprint="problem"
    )
    assert first.plan_fingerprint == second.plan_fingerprint
    assert first.microshards == second.microshards
    assert sorted(
        group for item in first.microshards for group in item.destination_group_indices
    ) == list(range(6))
    costs = [item.predicted_cost for item in first.microshards]
    assert max(costs) - min(costs) <= 1.0
    restored = type(first).from_dict(json.loads(json.dumps(first.to_dict())))
    assert restored == first
    changed = build_balanced_microshard_plan(
        work, target_microshards=2, problem_fingerprint="problem"
    )
    assert changed.plan_fingerprint != first.plan_fingerprint


def test_operator_metadata_builds_work_without_loading_routing_shards():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary))
        before = operator.metrics.cache_misses
        work = routing_group_work_units(
            operator,
            cost_model=RoutingCostModel(),
            stratum_by_group={0: "early", 1: "early"},
        )
        assert len(work) == operator.routing.num_destination_groups
        assert operator.metrics.cache_misses == before
        assert work[0].stratum == "early"
        assert all(item.predicted_cost > 0.0 for item in work)


def test_instrumentation_reports_batches_without_changing_exact_product():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        reference = _operator(root / "reference")
        profiler = ShardedWorkInstrumentation(reference.routing)
        observed = _operator(root / "observed", progress_callback=profiler)
        demand = np.linspace(0.1, 2.0, observed.num_free_od, dtype=np.float32)
        expected = reference.matvec(demand)
        actual = observed.matvec(demand)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
        cotangent = np.linspace(-1.0, 1.0, observed.num_measurements, dtype=np.float32)
        expected_reverse = reference.rmatvec(cotangent)
        actual_reverse = observed.rmatvec(cotangent)
        np.testing.assert_allclose(
            actual_reverse, expected_reverse, rtol=1.0e-6, atol=1.0e-6
        )
        report = profiler.report()
        assert report["totals"]["batches"] == 4
        assert report["totals"]["destination_groups"] == 16
        assert report["totals"]["predicted_routing_bytes"] > 0
        assert len(report["observations"]) == 4
        assert {item["operation"] for item in report["observations"]} == {
            "matvec",
            "rmatvec",
        }
        assert all(item["total_seconds"] >= 0.0 for item in report["observations"])


def test_fixed_budget_selection_is_nested_weighted_and_close_to_requested_cost():
    work = tuple(
        _work(index, 1.0, stratum="early" if index < 8 else "late")
        for index in range(16)
    )
    plan = build_balanced_microshard_plan(
        work, target_microshards=16, problem_fingerprint="problem"
    )
    low = plan_fixed_budget_routing_selection(plan, effort_percent=25, seed=7)
    medium = plan_fixed_budget_routing_selection(plan, effort_percent=50, seed=7)
    exact = plan_fixed_budget_routing_selection(plan, effort_percent=100, seed=7)
    assert set(low.selected_work_ids) < set(medium.selected_work_ids)
    assert set(medium.selected_work_ids) < set(exact.selected_work_ids)
    assert low.realized_effort_percent == 25.0
    assert medium.realized_effort_percent == 50.0
    assert set(low.expansion_weights) == {4.0}
    assert set(medium.expansion_weights) == {2.0}
    assert set(exact.expansion_weights) == {1.0}
    repeated = plan_fixed_budget_routing_selection(plan, effort_percent=25, seed=7)
    assert repeated == low
