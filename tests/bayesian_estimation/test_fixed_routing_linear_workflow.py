from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.fixed_routing_linear_results import (
    load_fixed_routing_linear_result,
)
from public_transportation.inference.fixed_routing_linear_workflow import (
    FixedRoutingLinearEstimationConfig,
    configure_fixed_routing_linear_regularization,
    run_fixed_routing_linear_estimation,
    run_fixed_routing_linear_estimation_scalable,
)
from public_transportation.inference.fixed_routing_linear_scalable_quality import (
    ScalableQualityConfig,
)


def _problem() -> FixedRoutingLinearProblem:
    return FixedRoutingLinearProblem(
        measurement_operator=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        fixed_measurement_offset=np.array([0.25, 0.5, 0.0]),
        observations=np.array([2.25, 3.5, 5.0]),
        observation_weights=np.array([1.0, 2.0, 0.5]),
        prior_demand=np.array([1.5, 2.5]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 2.0),
        variable_scales=np.array([1.5, 2.5]),
        free_od_indices=np.array([5, 2]),
    )


@pytest.mark.parametrize(
    ("choice", "strength", "selection", "block_name"),
    [
        ("none", None, "none", None),
        ("ridge_to_prior", 0.25, "configured", "ridge_to_prior"),
        (
            "scaled_ridge_to_prior",
            0.25,
            "configured",
            "scaled_ridge_to_prior",
        ),
    ],
)
def test_workflow_runs_each_explicit_regularization_choice(
    choice, strength, selection, block_name
):
    run = run_fixed_routing_linear_estimation(
        _problem(),
        config=FixedRoutingLinearEstimationConfig(
            regularization=choice,
            regularization_strength=strength,
        ),
    )

    assert run.problem.regularization_selection == selection
    assert tuple(block.name for block in run.problem.regularization_blocks) == (
        () if block_name is None else (block_name,)
    )
    assert run.iterative_result.success
    assert run.dense_reference.success
    assert run.output_path is None
    assert run.result.estimated_demand.shape == (2,)
    assert run.recommendation.automatic_selection_applied is False


def test_workflow_persists_shared_result_contract(tmp_path):
    output = tmp_path / "result.npz"
    run = run_fixed_routing_linear_estimation(
        _problem(),
        config=FixedRoutingLinearEstimationConfig(regularization="none"),
        output_path=output,
    )

    loaded = load_fixed_routing_linear_result(
        output, expected_od_layout_fingerprint="od"
    )
    assert run.output_path == output
    np.testing.assert_allclose(loaded.estimated_demand, run.iterative_result.demand)
    np.testing.assert_array_equal(loaded.free_od_indices, [5, 2])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"regularization": "none", "regularization_strength": 1.0},
        {"regularization": "ridge_to_prior"},
        {"regularization": "scaled_ridge_to_prior", "regularization_strength": -1.0},
        {"regularization": "unknown"},
    ],
)
def test_config_rejects_ambiguous_or_invalid_regularization(kwargs):
    with pytest.raises(ValueError):
        FixedRoutingLinearEstimationConfig(**kwargs)


def test_workflow_rejects_preconfigured_problem():
    configured = replace(_problem(), regularization_selection="none")
    config = FixedRoutingLinearEstimationConfig(regularization="none")
    with pytest.raises(ValueError, match="requires a base problem"):
        configure_fixed_routing_linear_regularization(configured, config)
    with pytest.raises(ValueError, match="requires a base problem"):
        run_fixed_routing_linear_estimation(configured, config=config)


def test_scalable_workflow_uses_products_without_reference_materialization(monkeypatch):
    def reject_materialization(*args, **kwargs):
        raise AssertionError("scalable workflow must not materialize the operator")

    monkeypatch.setattr(
        "public_transportation.inference.linear_operator.materialize_linear_operator",
        reject_materialization,
    )
    run = run_fixed_routing_linear_estimation_scalable(
        _problem(),
        config=FixedRoutingLinearEstimationConfig(
            regularization="ridge_to_prior",
            regularization_strength=0.5,
        ),
        quality_config=ScalableQualityConfig(
            resolution_samples=4,
            smallest_singular_values=1,
        ),
    )

    assert run.solver_result.success
    assert run.solver_result.backend == "trf_lsmr"
    assert run.quality.spectral_converged
    assert run.quality.resolution_converged_samples == 4
