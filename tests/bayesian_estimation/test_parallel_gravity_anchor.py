from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from benchmarks.benchmark_sharded_gravity_operator import _problem
from public_transportation.inference.gravity import gravity_value_and_gradient_adjoint
from public_transportation.inference.parallel_gravity_anchor import (
    ParallelGravityAnchor,
    create_parallel_gravity_anchor,
    parallel_anchored_value_and_gradient,
)
from public_transportation.inference.parallel_partial_execution import (
    plan_fixed_budget_routing_selection,
)
from public_transportation.inference.parallel_routing_executor import (
    ParallelApproximateRoutingOperator,
    ParallelRoutingExecutorConfig,
    PersistentParallelRoutingExecutor,
)
from tests.bayesian_estimation.test_parallel_routing_executor import _operator, _plan


def test_anchor_roundtrip_and_every_effort_is_exact_at_anchor():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary))
        plan = _plan(operator)
        problem, raw = _problem(operator)
        anchor = create_parallel_gravity_anchor(raw, problem=problem)
        restored = ParallelGravityAnchor.from_dict(
            json.loads(json.dumps(anchor.to_dict()))
        )
        selection = plan_fixed_budget_routing_selection(
            plan, effort_percent=25, seed=3
        )
        with PersistentParallelRoutingExecutor(
            operator=operator,
            microshard_plan=plan,
            config=ParallelRoutingExecutorConfig(
                worker_count=4, supported_group_batch_sizes=(1, 2)
            ),
        ) as executor:
            evaluation, gradient = parallel_anchored_value_and_gradient(
                raw,
                problem=problem,
                executor=executor,
                selection=selection,
                anchor=restored,
            )
        assert float(evaluation.objective) == pytest.approx(anchor.objective)
        np.testing.assert_allclose(evaluation.measurement_mean, anchor.measurement_mean)
        np.testing.assert_allclose(gradient, anchor.gradient)


def test_anchor_rejects_changed_problem_identity():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary))
        plan = _plan(operator)
        problem, raw = _problem(operator)
        anchor = create_parallel_gravity_anchor(raw, problem=problem)
        changed = replace(problem, observations=np.asarray(problem.observations) + 1.0)
        selection = plan_fixed_budget_routing_selection(
            plan, effort_percent=25, seed=3
        )
        with PersistentParallelRoutingExecutor(
            operator=operator, microshard_plan=plan
        ) as executor:
            with pytest.raises(ValueError, match="identity mismatch"):
                parallel_anchored_value_and_gradient(
                    raw,
                    problem=changed,
                    executor=executor,
                    selection=selection,
                    anchor=anchor,
                )


def test_anchor_reduces_mean_error_for_nearby_parameters_at_equal_effort():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary))
        plan = _plan(operator)
        problem, raw = _problem(operator)
        anchor = create_parallel_gravity_anchor(raw, problem=problem)
        target = np.asarray(raw) + np.asarray([0.03, -0.02, 0.01])
        exact, _ = gravity_value_and_gradient_adjoint(target, problem=problem)
        unanchored_errors = []
        anchored_errors = []
        with PersistentParallelRoutingExecutor(
            operator=operator,
            microshard_plan=plan,
            config=ParallelRoutingExecutorConfig(
                worker_count=4,
                supported_group_batch_sizes=(1, 2),
                maximum_retained_batch_bytes=1024 * 1024,
            ),
        ) as executor:
            for seed in range(5):
                selection = plan_fixed_budget_routing_selection(
                    plan, effort_percent=25, seed=seed
                )
                unanchored_operator = ParallelApproximateRoutingOperator(
                    operator, executor, selection
                )
                unanchored, _ = gravity_value_and_gradient_adjoint(
                    target, problem=replace(problem, operator=unanchored_operator)
                )
                anchored, _ = parallel_anchored_value_and_gradient(
                    target,
                    problem=problem,
                    executor=executor,
                    selection=selection,
                    anchor=anchor,
                )
                unanchored_errors.append(
                    np.linalg.norm(
                        np.asarray(unanchored.measurement_mean)
                        - np.asarray(exact.measurement_mean)
                    )
                )
                anchored_errors.append(
                    np.linalg.norm(
                        np.asarray(anchored.measurement_mean)
                        - np.asarray(exact.measurement_mean)
                    )
                )
        assert np.mean(anchored_errors) < np.mean(unanchored_errors)


def test_committed_anchor_benchmark_improves_gradient_and_count_errors():
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/parallel_gravity_anchor_public.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    for row in report["results"]:
        assert row["anchored_gradient_relative_error"] < row[
            "unanchored_gradient_relative_error"
        ]
        assert row["anchored_count_relative_error"] < row[
            "unanchored_count_relative_error"
        ]
        assert row["anchored_speedup"] > 1.0
