"""Regression checks for the fixed-routing optimization baseline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import numpy as np

from benchmarks.benchmark_fixed_routing_baseline import build_baseline_setup

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "benchmarks/fixed_routing_baseline.npz"


def test_simple_example_02_matches_fixed_routing_reference() -> None:
    with tempfile.TemporaryDirectory(prefix="pt-fixed-routing-test-") as temporary:
        setup = build_baseline_setup(Path(temporary))
        evaluate = jax.jit(
            lambda parameter: (
                setup.forward(parameter),
                setup.objective(parameter),
                jax.grad(setup.objective)(parameter),
            )
        )
        results = [evaluate(case.parameter) for case in setup.cases]
        evaluate_cached = jax.jit(
            lambda parameter: (
                setup.cached_forward(parameter),
                setup.cached_objective(parameter),
                jax.grad(setup.cached_objective)(parameter),
            )
        )
        cached_results = [evaluate_cached(case.parameter) for case in setup.cases]
        jax.block_until_ready(results)
        jax.block_until_ready(cached_results)

    assert setup.metadata["example"] == "simple_example_02"
    assert setup.metadata["implementation"] == "dynamic_routing_reference"
    assert setup.metadata["fixed_theta"] == 1.0
    assert setup.metadata["num_fixed_zero_od"] >= 1
    assert [case.name for case in setup.cases] == [
        "baseline",
        "perturbed",
        "alternating",
    ]

    objectives = []
    with np.load(REFERENCE) as reference:
        for case, dynamic, cached in zip(
            setup.cases, results, cached_results, strict=True
        ):
            (link_flow, prediction, loglik), objective, gradient = dynamic
            cached_outputs, cached_objective, cached_gradient = cached
            actual = {
                "parameter": np.asarray(case.parameter),
                "link_flow": np.asarray(link_flow),
                "prediction": np.asarray(prediction),
                "gradient": np.asarray(gradient),
            }
            for key, value in actual.items():
                np.testing.assert_allclose(
                    value,
                    reference[f"{case.name}_{key}"],
                    rtol=3.0e-5,
                    atol=3.0e-5,
                )
            np.testing.assert_allclose(cached_outputs[0], link_flow, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(cached_outputs[1], prediction, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(cached_outputs[2], loglik, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(cached_objective, objective, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(
                cached_gradient,
                gradient,
                rtol=3e-5,
                atol=3e-5,
            )
            assert np.all(np.isfinite(actual["link_flow"]))
            assert np.all(actual["link_flow"] >= 0.0)
            assert np.all(np.isfinite(actual["prediction"]))
            assert np.all(actual["prediction"] >= 0.0)
            assert np.isfinite(float(np.asarray(loglik)))
            assert np.isfinite(float(np.asarray(objective)))
            assert np.all(np.isfinite(actual["gradient"]))
            objectives.append(float(np.asarray(objective)))

    assert len(set(objectives)) == len(objectives)
