from __future__ import annotations

import numpy as np
import pytest

from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.fixed_routing_linear_solver import (
    FixedRoutingLinearSolverConfig,
    benchmark_fixed_routing_linear_solvers,
    solve_fixed_routing_linear,
)
from public_transportation.inference.fixed_routing_linear_trf_solver import (
    TRFLSMRConfig,
    TRFLSMRResult,
)


def _problem() -> FixedRoutingLinearProblem:
    return FixedRoutingLinearProblem(
        measurement_operator=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
        fixed_measurement_offset=np.zeros(4),
        observations=np.array([2.0, 3.0, 5.0, 7.0]),
        observation_weights=np.ones(4),
        prior_demand=np.array([1.0, 1.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        regularization_selection="none",
    )


@pytest.mark.parametrize("backend", ["dense_reference", "trf_lsmr"])
def test_registered_backends_expose_one_result_contract(backend):
    result = solve_fixed_routing_linear(
        _problem(), config=FixedRoutingLinearSolverConfig(backend=backend)
    )

    assert result.backend == backend
    assert result.success
    assert result.demand.shape == (2,)
    assert result.solver_variable.shape == (2,)
    assert result.evaluation.objective >= 0.0
    assert result.kkt.feasibility_inf_norm == 0.0
    assert result.elapsed_seconds >= 0.0
    with pytest.raises(ValueError, match="read-only"):
        result.demand[0] = 0.0
    if backend == "dense_reference":
        assert result.numerical_rank == 2
        assert result.matvec_count is None
        assert result.singular_values.size == 2
    else:
        assert result.numerical_rank is None
        assert result.matvec_count is not None
        assert isinstance(result.native_result, TRFLSMRResult)


def test_benchmark_compares_accuracy_and_work_without_false_dense_counts():
    records = benchmark_fixed_routing_linear_solvers(
        _problem(),
        configs=(
            FixedRoutingLinearSolverConfig(backend="dense_reference"),
            FixedRoutingLinearSolverConfig(
                backend="trf_lsmr",
                trf_lsmr=TRFLSMRConfig(tolerance=1.0e-10),
            ),
        ),
    )

    assert tuple(item.backend for item in records) == (
        "dense_reference",
        "trf_lsmr",
    )
    assert all(item.success for item in records)
    np.testing.assert_allclose(
        [item.objective for item in records], records[0].objective, rtol=1e-8, atol=1e-8
    )
    assert records[0].matvec_count is None
    assert records[1].matvec_count is not None
    assert min(item.objective_difference_from_best for item in records) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "unknown"},
        {"dense_tolerance": 0.0},
        {"dense_active_tolerance": -1.0},
        {"dense_max_iterations": 0},
        {"dense_max_materialized_entries": 0},
    ],
)
def test_solver_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        FixedRoutingLinearSolverConfig(**kwargs)


def test_benchmark_requires_unique_nonempty_backends():
    with pytest.raises(ValueError, match="at least one"):
        benchmark_fixed_routing_linear_solvers(_problem(), configs=())
    repeated = FixedRoutingLinearSolverConfig(backend="trf_lsmr")
    with pytest.raises(ValueError, match="unique"):
        benchmark_fixed_routing_linear_solvers(_problem(), configs=(repeated, repeated))
