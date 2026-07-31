from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from public_transportation.inference.fixed_routing_linear_results import (
    load_fixed_routing_linear_result,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "docs/source/examples/simple_example_01/estimation/run_linear_fixed_routing.py"
)


def _load_example_module():
    spec = importlib.util.spec_from_file_location("simple_example_01_linear", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_simple_example_01_linear_runs_end_to_end(tmp_path):
    module = _load_example_module()
    output = tmp_path / "linear_fixed_routing_results.npz"
    cache = tmp_path / "operator-cache"
    result = module.run_linear_fixed_routing(
        output_path=output,
        chunk_size=8,
        operator_cache_directory=cache,
    )

    assert output.exists()
    assert result.iterative_result.success
    assert result.operator_validation_max_abs_difference <= 4.0e-5
    assert result.iterative_result.kkt.feasibility_inf_norm <= 1.0e-8
    assert result.iterative_result.kkt.projected_gradient_inf_norm <= 5.0e-4
    assert result.iterative_result.evaluation.objective <= (
        result.dense_result.evaluation.objective * (1.0 + 2.0e-5) + 2.0e-5
    )
    np.testing.assert_allclose(
        result.iterative_result.demand,
        result.dense_result.demand,
        rtol=3.0e-4,
        atol=3.0e-4,
    )
    assert len(result.quality.classifications) == result.problem.num_free_od

    saved = load_fixed_routing_linear_result(
        output,
        expected_od_layout_fingerprint=result.problem.provenance.od_layout_fingerprint,
    )
    assert saved.schema_version == 1
    assert saved.mode == "fixed_routing_linear"
    assert saved.regularization_names == ()
    np.testing.assert_allclose(saved.estimated_demand, result.iterative_result.demand)
    np.testing.assert_allclose(
        saved.predicted_measurements,
        result.iterative_result.evaluation.data_fit.prediction,
    )
    np.testing.assert_array_equal(saved.free_od_indices, result.problem.free_od_indices)
    assert len(saved.classifications) == result.problem.num_free_od

    cached = module.run_linear_fixed_routing(
        output_path=output,
        chunk_size=8,
        operator_cache_directory=cache,
    )
    assert result.operator_cache_hit is False
    assert cached.operator_cache_hit is True
    np.testing.assert_allclose(
        cached.iterative_result.demand, result.iterative_result.demand
    )


def test_simple_example_01_linear_model_validation(tmp_path):
    module = _load_example_module()
    report_path = tmp_path / "linear_model_validation.json"
    validation = module.run_linear_model_validation(
        report_path=report_path,
        chunk_size=8,
        operator_cache_directory=tmp_path / "operator-cache",
    )

    assert validation.forward.passed
    assert [case.name for case in validation.forward.cases] == [
        "zero",
        "prior",
        "true",
        "seeded_random",
    ]
    assert validation.forward.worst_max_abs_difference <= 5.0e-5
    assert validation.recovery.result.success
    assert validation.recovery.measurement_residual_inf_norm <= 1.0e-8
    assert validation.recovery.identifiable_error_norm <= 1.0e-8
    if validation.recovery.measurement_nullity == 0:
        assert validation.recovery.estimation_error_norm <= 1.0e-8

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["forward_equivalence"]["passed"] is True
    assert report["noise_free_recovery"]["measurement_rank"] == (
        validation.recovery.measurement_rank
    )


def test_simple_example_01_actual_observation_accuracy_comparison(tmp_path):
    module = _load_example_module()
    comparison = module.run_actual_observation_accuracy_comparison(
        report_path=tmp_path / "linear_accuracy_comparison.json",
        nonlinear_results_path=None,
        chunk_size=8,
        operator_cache_directory=tmp_path / "operator-cache",
    )

    methods = comparison.report["methods"]
    unregularized = methods["linear_none"]
    ridge = methods["linear_ridge_to_prior"]
    scaled = methods["linear_scaled_ridge_to_prior"]

    assert comparison.report["problem"]["fixed_constraints_match_generating_truth"] is True
    assert unregularized["demand_rmse"] <= 1.0e-5
    assert unregularized["measurement_max_abs_error"] <= 1.0e-5
    assert unregularized["least_squares_data_objective"] < (
        ridge["least_squares_data_objective"]
    )
    assert ridge["distance_to_prior"] < unregularized["distance_to_prior"]
    assert scaled["distance_to_prior"] < unregularized["distance_to_prior"]

    saved = json.loads(comparison.report_path.read_text(encoding="utf-8"))
    assert saved["measurement_generation"]["added_observation_noise"] is False
    assert saved["methods"]["linear_none"]["solver_success"] is True
    assert saved["problem"]["fixed_od_constraints"] == [
        {
            "difference": 0.0,
            "fixed_value": 10.0,
            "generating_true_value": 10.0,
            "od_index": 0,
            "od_key": ["A", "B", "t0"],
        },
        {
            "difference": 0.0,
            "fixed_value": 50.0,
            "generating_true_value": 50.0,
            "od_index": 1,
            "od_key": ["A", "B", "t1"],
        },
    ]
