from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from benchmarks.benchmark_sharded_gravity_operator import _problem
from public_transportation.inference.gravity import gravity_value_and_gradient_adjoint
from public_transportation.inference.parallel_exact_gate import (
    assess_parallel_exact_gate,
)
from public_transportation.inference.parallel_routing_executor import (
    ParallelApproximateRoutingOperator,
    ParallelExactRoutingOperator,
    ParallelRoutingExecutorConfig,
    PersistentParallelRoutingExecutor,
)
from public_transportation.inference.parallel_partial_execution import (
    plan_fixed_budget_routing_selection,
)
from tests.bayesian_estimation.test_parallel_routing_executor import _operator, _plan


def _gate(**overrides):
    values = {
        "reference_forward": np.asarray([1.0, 2.0]),
        "parallel_forward": np.asarray([1.0, 2.0]),
        "reference_reverse": np.asarray([3.0, 4.0]),
        "parallel_reverse": np.asarray([3.0, 4.0]),
        "reference_objective": 5.0,
        "parallel_objective": 5.0,
        "reference_gradient": np.asarray([2.0, -1.0]),
        "parallel_gradient": np.asarray([2.0, -1.0]),
        "existing_exact_seconds": 11.0,
        "parallel_exact_seconds": 10.0,
        "requested_workers": 8,
        "observed_worker_lanes": 8,
    }
    values.update(overrides)
    return assess_parallel_exact_gate(**values)


def test_gate_promotes_only_when_all_requirements_pass():
    report = _gate()
    assert report.promotion_passed
    assert report.recommendation == "promote_parallel"
    assert report.measured_speedup == 1.1
    assert report.to_dict()["schema_version"] == 1


def test_gate_rejects_numerical_performance_and_utilization_failures():
    numerical = _gate(parallel_gradient=np.asarray([3.0, -1.0]))
    assert not numerical.numerical_equivalence_passed
    assert numerical.recommendation == "retain_existing"
    performance = _gate(parallel_exact_seconds=10.5)
    assert not performance.performance_passed
    utilization = _gate(observed_worker_lanes=7)
    assert not utilization.worker_utilization_passed


def test_parallel_exact_adapter_matches_complete_gravity_objective_and_gradient():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary))
        plan = _plan(operator)
        problem, raw = _problem(operator)
        reference_evaluation, reference_gradient = gravity_value_and_gradient_adjoint(
            raw, problem=problem
        )
        with PersistentParallelRoutingExecutor(
            operator=operator,
            microshard_plan=plan,
            config=ParallelRoutingExecutorConfig(
                worker_count=4, supported_group_batch_sizes=(1, 2)
            ),
        ) as executor:
            parallel_operator = ParallelExactRoutingOperator(operator, executor)
            parallel_evaluation, parallel_gradient = gravity_value_and_gradient_adjoint(
                raw, problem=replace(problem, operator=parallel_operator)
            )
        np.testing.assert_allclose(
            parallel_evaluation.measurement_mean,
            reference_evaluation.measurement_mean,
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            parallel_evaluation.objective,
            reference_evaluation.objective,
            rtol=1.0e-5,
            atol=1.0e-3,
        )
        np.testing.assert_allclose(
            parallel_gradient, reference_gradient, rtol=1.0e-5, atol=1.0e-5
        )


def test_committed_public_gate_retains_existing_backend():
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/parallel_exact_gate_public.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    gate = report["promotion_gate"]
    assert gate["numerical_equivalence_passed"] is True
    assert gate["worker_utilization_passed"] is True
    assert gate["performance_passed"] is False
    assert gate["recommendation"] == "retain_existing"
    assert gate["measured_speedup"] < gate["config"]["minimum_speedup"]


def test_parallel_approximate_adapter_returns_consistent_finite_adjoint_result():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary))
        plan = _plan(operator)
        selection = plan_fixed_budget_routing_selection(
            plan, effort_percent=50, seed=11
        )
        problem, raw = _problem(operator)
        with PersistentParallelRoutingExecutor(
            operator=operator,
            microshard_plan=plan,
            config=ParallelRoutingExecutorConfig(
                worker_count=4,
                supported_group_batch_sizes=(1, 2),
                maximum_retained_batch_bytes=1024 * 1024,
            ),
        ) as executor:
            approximate = ParallelApproximateRoutingOperator(
                operator, executor, selection
            )
            evaluation, gradient = gravity_value_and_gradient_adjoint(
                raw, problem=replace(problem, operator=approximate)
            )
            assert approximate.release_pending() is False
        assert np.isfinite(float(evaluation.objective))
        assert np.all(np.isfinite(np.asarray(evaluation.measurement_mean)))
        assert np.all(np.isfinite(np.asarray(gradient)))
