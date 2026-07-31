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
    / "docs/source/examples/simple_example_02/estimation/run_linear_fixed_routing.py"
)


def _load_example_module():
    spec = importlib.util.spec_from_file_location("simple_example_02_linear", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_simple_example_02_linear_runs_all_configurations(tmp_path):
    module = _load_example_module()
    output = tmp_path / "linear_fixed_routing"
    cache = tmp_path / "operator-cache"
    run = module.run_linear_fixed_routing(
        output_path=output,
        chunk_size=8,
        operator_cache_directory=cache,
    )

    assert output.exists()
    assert run.operator_validation_max_abs_difference <= 4.0e-5
    assert tuple(item.name for item in run.configurations) == (
        "none",
        "ridge_to_prior",
        "scaled_ridge_to_prior",
    )
    for configuration in run.configurations:
        iterative = configuration.iterative_result
        reference = configuration.dense_result
        assert iterative.success
        assert iterative.kkt.feasibility_inf_norm <= 1.0e-8
        assert iterative.kkt.projected_gradient_inf_norm <= 2.0e-3
        assert iterative.evaluation.objective <= (
            reference.evaluation.objective * (1.0 + 3.0e-5) + 3.0e-5
        )
        np.testing.assert_allclose(
            iterative.evaluation.data_fit.prediction,
            reference.evaluation.data_fit.prediction,
            rtol=3.0e-4,
            atol=3.0e-4,
        )
        assert (
            len(configuration.quality.classifications) == run.base_problem.num_free_od
        )

    ridge = run.configurations[1]
    scaled = run.configurations[2]
    np.testing.assert_allclose(
        ridge.iterative_result.demand,
        ridge.dense_result.demand,
        rtol=5.0e-4,
        atol=5.0e-4,
    )
    np.testing.assert_allclose(
        scaled.iterative_result.demand,
        scaled.dense_result.demand,
        rtol=5.0e-4,
        atol=5.0e-4,
    )

    for configuration in run.configurations:
        saved = load_fixed_routing_linear_result(
            output / f"{configuration.name}.npz",
            expected_mapping_fingerprint=(
                configuration.problem.provenance.mapping_fingerprint
            ),
        )
        np.testing.assert_allclose(
            saved.estimated_demand, configuration.iterative_result.demand
        )
        assert len(saved.classifications) == run.base_problem.num_free_od

    cached = module.run_linear_fixed_routing(
        output_path=output,
        chunk_size=8,
        operator_cache_directory=cache,
    )
    assert run.operator_cache_hit is False
    assert cached.operator_cache_hit is True
    for first, second in zip(run.configurations, cached.configurations, strict=True):
        np.testing.assert_allclose(
            first.iterative_result.demand, second.iterative_result.demand
        )


def test_simple_example_02_linear_model_validation(tmp_path):
    module = _load_example_module()
    validation = module.run_linear_model_validation(
        report_path=tmp_path / "linear_model_validation.json",
        chunk_size=8,
        operator_cache_directory=tmp_path / "operator-cache",
    )

    assert validation.report["fixed_constraints_match_generating_truth"] is True
    assert validation.forward.passed
    assert validation.forward.worst_max_abs_difference <= 5.0e-5
    assert validation.recovery.result.success
    assert validation.recovery.measurement_nullity == 0
    assert validation.recovery.measurement_residual_inf_norm <= 1.0e-7
    assert validation.recovery.estimation_error_norm <= 1.0e-5
    assert validation.report["actual_observations_at_true_demand"][
        "residual_inf_norm"
    ] <= 1.0e-4

    saved = json.loads(validation.report_path.read_text(encoding="utf-8"))
    assert saved["noise_free_recovery"]["measurement_rank"] == 70
    assert saved["fixed_od_constraints"] == [
        {
            "fixed_value": 18.0,
            "generating_true_value": 18.0,
            "od_index": 0,
            "od_key": ["A", "H", "t0"],
        },
        {
            "fixed_value": 42.0,
            "generating_true_value": 42.0,
            "od_index": 1,
            "od_key": ["A", "H", "t1"],
        },
    ]
