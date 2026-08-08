"""Compare departure quadrature strategies on deterministic public microcases."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import asdict

import numpy as np

from public_transportation.preprocessing.reduced_od import (
    DepartureTimeSamplingConfig,
    ResponseCellKey,
    SparseWeightedResponse,
    integrate_adaptive_departure_response,
)
from public_transportation.preprocessing.reduced_od.adaptive_departure_quadrature import (
    _integrate_adaptive_departure_response_depth_first,
)


def _response(case: str, seconds: float) -> SparseWeightedResponse | None:
    if case == "constant":
        return SparseWeightedResponse((0,), (1.0,))
    if case == "single_boundary":
        return None if seconds < 1800 else SparseWeightedResponse((0, 2), (1.0, 0.5))
    if seconds < 900:
        return SparseWeightedResponse((0,), (0.8,))
    if seconds < 2100:
        return SparseWeightedResponse((0, 1), (1.0, 0.4))
    if seconds < 3000:
        return SparseWeightedResponse((1, 2), (0.7, 1.0))
    return None


def _fixed(case: str, *, width: int, count: int | None = None) -> dict[str, object]:
    started = time.perf_counter()
    if count is None:
        edges = np.arange(0.0, 3600.0, width).tolist() + [3600.0]
    else:
        edges = np.linspace(0.0, 3600.0, count + 1).tolist()
    prediction: dict[int, float] = {}
    infeasible = 0.0
    for left, right in zip(edges, edges[1:]):
        weight = (right - left) / 3600.0
        response = _response(case, 0.5 * (left + right))
        if response is None:
            infeasible += weight
            continue
        for index, value in zip(response.measurement_indices, response.values, strict=True):
            prediction[index] = prediction.get(index, 0.0) + weight * value
    vector = np.asarray([prediction.get(index, 0.0) for index in range(3)])
    return {
        "routing_evaluations": len(edges) - 1,
        "preprocessing_seconds": time.perf_counter() - started,
        "prediction": vector.tolist(),
        "supported_measurement_rows": int(np.count_nonzero(vector)),
        "supported_positive_observed_mass": float(np.asarray([10, 20, 30])[vector != 0].sum()),
        "infeasible_time_fraction": infeasible,
        "unresolved_interval_weight": 0.0,
        "sample_cap_reached": False,
    }


def _adaptive(case: str) -> dict[str, object]:
    result = integrate_adaptive_departure_response(
        cell_key=ResponseCellKey("A", "B", "P"),
        start_seconds=0,
        end_seconds=3600,
        evaluator=lambda seconds: _response(case, seconds),
        config=DepartureTimeSamplingConfig(
            strategy="adaptive_service_aware",
            initial_interval_seconds=1800,
            minimum_interval_seconds=60,
            response_tolerance=1.0e-3,
            maximum_samples_per_cell=128,
            infeasible_policy="preserve_mass",
        ),
    )
    sparse = result.averaged_response
    prediction = np.zeros(3)
    prediction[list(sparse.measurement_indices)] = sparse.values
    return {
        "routing_evaluations": result.diagnostics.routing_evaluations,
        "preprocessing_seconds": result.diagnostics.elapsed_seconds,
        "prediction": prediction.tolist(),
        "supported_measurement_rows": int(np.count_nonzero(prediction)),
        "supported_positive_observed_mass": float(np.asarray([10, 20, 30])[prediction != 0].sum()),
        "infeasible_time_fraction": result.diagnostics.infeasible_time_fraction,
        "unresolved_interval_weight": result.diagnostics.unresolved_interval_weight,
        "sample_cap_reached": result.diagnostics.sample_cap_reached,
        "diagnostics": asdict(result.diagnostics),
    }


def _integral_adaptive(case: str) -> dict[str, object]:
    result = integrate_adaptive_departure_response(
        cell_key=ResponseCellKey("A", "B", "P"),
        start_seconds=0,
        end_seconds=3600,
        evaluator=lambda seconds: _response(case, seconds),
        config=DepartureTimeSamplingConfig(
            strategy="adaptive_service_aware",
            comparison_mode="integral_response",
            initial_interval_seconds=1800,
            minimum_interval_seconds=60,
            absolute_response_tolerance=1.0e-3,
            relative_response_tolerance=2.0e-2,
            maximum_samples_per_cell=128,
            infeasible_policy="preserve_mass",
        ),
        effective_comparison_mode="integral_response",
        service_boundary_seconds=(900, 1800, 2100, 3000),
    )
    sparse = result.averaged_response
    prediction = np.zeros(3)
    prediction[list(sparse.measurement_indices)] = sparse.values
    return {
        "routing_evaluations": result.diagnostics.routing_evaluations,
        "preprocessing_seconds": result.diagnostics.elapsed_seconds,
        "peak_rss_bytes": result.diagnostics.peak_rss_bytes,
        "prediction": prediction.tolist(),
        "supported_measurement_rows": int(np.count_nonzero(prediction)),
        "operator_nonzeros": int(np.count_nonzero(prediction)),
        "response_total": float(prediction.sum()),
        "supported_positive_observed_mass": float(
            np.asarray([10, 20, 30])[prediction != 0].sum()
        ),
        "infeasible_time_fraction": result.diagnostics.infeasible_time_fraction,
        "estimated_absolute_integration_error": (
            result.diagnostics.estimated_absolute_integration_error
        ),
        "estimated_relative_integration_error": (
            result.diagnostics.estimated_relative_response_error
        ),
        "unresolved_interval_weight": result.diagnostics.unresolved_interval_weight,
        "sample_cap_reached": result.diagnostics.sample_cap_reached,
        "diagnostics": asdict(result.diagnostics),
    }


def _discrete_response(seconds: float) -> SparseWeightedResponse:
    trip_row = min(11, int(seconds // 1800))
    return SparseWeightedResponse((trip_row,), (1.0,))


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _discrete_fixed(step: int) -> tuple[np.ndarray, int, float]:
    started = time.perf_counter()
    duration = 6 * 3600
    prediction = np.zeros(12)
    evaluations = 0
    for left in range(0, duration, step):
        right = min(duration, left + step)
        response = _discrete_response(0.5 * (left + right))
        prediction[response.measurement_indices[0]] += (right - left) / duration
        evaluations += 1
    return prediction, evaluations, time.perf_counter() - started


def _discrete_integral() -> tuple[np.ndarray, dict[str, object]]:
    result = integrate_adaptive_departure_response(
        cell_key=ResponseCellKey("A", "B", "DISCRETE"),
        start_seconds=0,
        end_seconds=6 * 3600,
        evaluator=_discrete_response,
        config=DepartureTimeSamplingConfig(
            strategy="adaptive_service_aware",
            comparison_mode="integral_response",
            initial_interval_seconds=3600,
            minimum_interval_seconds=60,
            absolute_response_tolerance=1.0e-4,
            relative_response_tolerance=1.0e-2,
            maximum_samples_per_cell=128,
            infeasible_policy="preserve_mass",
        ),
        effective_comparison_mode="integral_response",
        service_boundary_seconds=tuple(range(1800, 6 * 3600, 1800)),
    )
    prediction = np.zeros(12)
    sparse = result.averaged_response
    prediction[list(sparse.measurement_indices)] = sparse.values
    return prediction, asdict(result.diagnostics)


def _discrete_pointwise() -> tuple[np.ndarray, dict[str, object]]:
    result = integrate_adaptive_departure_response(
        cell_key=ResponseCellKey("A", "B", "DISCRETE"),
        start_seconds=0,
        end_seconds=6 * 3600,
        evaluator=_discrete_response,
        config=DepartureTimeSamplingConfig(
            strategy="adaptive_service_aware",
            comparison_mode="aggregate_response",
            initial_interval_seconds=3600,
            minimum_interval_seconds=60,
            response_tolerance=1.0e-3,
            maximum_samples_per_cell=128,
            infeasible_policy="preserve_mass",
        ),
        effective_comparison_mode="aggregate_response",
    )
    prediction = np.zeros(12)
    sparse = result.averaged_response
    prediction[list(sparse.measurement_indices)] = sparse.values
    return prediction, asdict(result.diagnostics)


def _long_response(seconds: float) -> SparseWeightedResponse:
    """Observation-visible response with many equivalent service departures."""
    if seconds < 13_500:
        return SparseWeightedResponse((0, 1), (1.0, 0.25))
    if seconds < 27_000:
        return SparseWeightedResponse((0, 1), (0.8, 0.4))
    return SparseWeightedResponse((0, 2), (0.6, 0.5))


def _long_fixed_reference() -> np.ndarray:
    values = np.zeros(3)
    for seconds in np.arange(150.0, 40_500.0, 300.0):
        response = _long_response(float(seconds))
        for index, value in zip(
            response.measurement_indices, response.values, strict=True
        ):
            values[index] += value / 135.0
    return values


def _long_adaptive_budget(budget: int) -> dict[str, object]:
    result = integrate_adaptive_departure_response(
        cell_key=ResponseCellKey("A", "B", "LONG"),
        start_seconds=0,
        end_seconds=40_500,
        evaluator=_long_response,
        config=DepartureTimeSamplingConfig(
            strategy="adaptive_service_aware",
            initial_interval_seconds=900,
            minimum_interval_seconds=60,
            response_tolerance=1.0e-3,
            maximum_samples_per_cell=budget,
            infeasible_policy="preserve_mass",
            comparison_mode="aggregate_response",
        ),
        effective_comparison_mode="aggregate_response",
    )
    prediction = np.zeros(3)
    prediction[list(result.averaged_response.measurement_indices)] = (
        result.averaged_response.values
    )
    reference = _long_fixed_reference()
    return {
        "evaluation_budget": budget,
        "routing_evaluations": result.diagnostics.routing_evaluations,
        "preprocessing_seconds": result.diagnostics.elapsed_seconds,
        "stable_interval_weight": result.diagnostics.stable_interval_weight,
        "unresolved_interval_weight": result.diagnostics.unresolved_interval_weight,
        "supported_measurement_rows": int(np.count_nonzero(prediction)),
        "supported_positive_observed_mass": float(
            np.asarray([10, 20, 30])[prediction != 0].sum()
        ),
        "relative_l1_from_five_minute": float(
            np.abs(prediction - reference).sum()
            / max(np.abs(reference).sum(), 1.0e-15)
        ),
        "sample_cap_reached": result.diagnostics.sample_cap_reached,
    }


def run() -> dict[str, object]:
    cases: dict[str, object] = {}
    for case in ("constant", "single_boundary", "multiple_changes"):
        methods = {
            "fixed_count_3": _fixed(case, width=0, count=3),
            "fixed_step_300": _fixed(case, width=300),
            "adaptive": _adaptive(case),
            "integral_adaptive": _integral_adaptive(case),
            "fixed_step_60": _fixed(case, width=60),
        }
        reference = np.asarray(methods["fixed_step_300"]["prediction"])
        fine = np.asarray(methods["fixed_step_60"]["prediction"])
        for values in methods.values():
            prediction = np.asarray(values["prediction"])
            values["relative_l1_from_five_minute"] = float(
                np.abs(prediction - reference).sum() / max(np.abs(reference).sum(), 1.0e-15)
            )
            values["relative_l1_from_one_minute"] = float(
                np.abs(prediction - fine).sum() / max(np.abs(fine).sum(), 1.0e-15)
            )
        cases[case] = methods
    def adversarial_evaluator(seconds: float) -> SparseWeightedResponse:
        return (
            SparseWeightedResponse((0,), (1.0,))
            if seconds >= 900 or int(seconds // 60) % 2 == 0
            else SparseWeightedResponse((0,), (2.0,))
        )
    adversarial_config = DepartureTimeSamplingConfig(
        strategy="adaptive_service_aware",
        initial_interval_seconds=900,
        minimum_interval_seconds=60,
        response_tolerance=1.0e-3,
        maximum_samples_per_cell=128,
        infeasible_policy="preserve_mass",
    )
    adversarial = integrate_adaptive_departure_response(
        cell_key=ResponseCellKey("A", "B", "LONG"),
        start_seconds=0,
        end_seconds=40500,
        evaluator=adversarial_evaluator,
        config=adversarial_config,
    )
    depth_first = _integrate_adaptive_departure_response_depth_first(
        cell_key=ResponseCellKey("A", "B", "LONG"),
        start_seconds=0,
        end_seconds=40500,
        evaluator=adversarial_evaluator,
        config=adversarial_config,
    )
    (
        discrete_reference,
        discrete_reference_evaluations,
        discrete_reference_seconds,
    ) = _discrete_fixed(300)
    discrete_fine, discrete_fine_evaluations, discrete_fine_seconds = _discrete_fixed(60)
    discrete_integral, discrete_diagnostics = _discrete_integral()
    discrete_pointwise, discrete_pointwise_diagnostics = _discrete_pointwise()
    discrete_three = np.zeros(12)
    for seconds in (3600.0, 10800.0, 18000.0):
        response = _discrete_response(seconds)
        discrete_three[response.measurement_indices[0]] += 1.0 / 3.0

    def discrete_metrics(
        prediction: np.ndarray, evaluations: int, seconds: float
    ) -> dict[str, object]:
        difference = prediction - discrete_reference
        fine_difference = prediction - discrete_fine
        return {
            "routing_evaluations": evaluations,
            "preprocessing_seconds": seconds,
            "peak_rss_bytes": _peak_rss_bytes(),
            "operator_dimensions": [12, 1],
            "operator_nonzeros": int(np.count_nonzero(prediction)),
            "response_total": float(prediction.sum()),
            "relative_l1_from_five_minute": float(
                np.abs(difference).sum()
                / max(np.abs(discrete_reference).sum(), 1.0e-15)
            ),
            "relative_l2_from_five_minute": float(
                np.linalg.norm(difference)
                / max(np.linalg.norm(discrete_reference), 1.0e-15)
            ),
            "relative_l1_from_one_minute": float(
                np.abs(fine_difference).sum()
                / max(np.abs(discrete_fine).sum(), 1.0e-15)
            ),
            "maximum_row_difference": float(np.max(np.abs(difference))),
            "supported_measurement_rows": int(np.count_nonzero(prediction)),
        }

    discrete_report = {
        "three_fixed_midpoints": discrete_metrics(discrete_three, 3, 0.0),
        "five_minute_fixed_step": discrete_metrics(
            discrete_reference,
            discrete_reference_evaluations,
            discrete_reference_seconds,
        ),
        "one_minute_fixed_step": discrete_metrics(
            discrete_fine, discrete_fine_evaluations, discrete_fine_seconds
        ),
        "integral_response": {
            **discrete_metrics(
                discrete_integral,
                int(discrete_diagnostics["routing_evaluations"]),
                float(discrete_diagnostics["elapsed_seconds"]),
            ),
            "estimated_absolute_integration_error": discrete_diagnostics[
                "estimated_absolute_integration_error"
            ],
            "estimated_relative_integration_error": discrete_diagnostics[
                "estimated_relative_response_error"
            ],
            "unresolved_interval_weight": discrete_diagnostics[
                "unresolved_interval_weight"
            ],
            "sample_cap_reached": discrete_diagnostics["sample_cap_reached"],
        },
        "old_pointwise_adaptive": {
            **discrete_metrics(
                discrete_pointwise,
                int(discrete_pointwise_diagnostics["routing_evaluations"]),
                float(discrete_pointwise_diagnostics["elapsed_seconds"]),
            ),
            "estimated_relative_response_error": discrete_pointwise_diagnostics[
                "estimated_relative_response_error"
            ],
            "unresolved_interval_weight": discrete_pointwise_diagnostics[
                "unresolved_interval_weight"
            ],
            "sample_cap_reached": discrete_pointwise_diagnostics[
                "sample_cap_reached"
            ],
        },
    }
    return {
        "schema_version": 4,
        "scope": "deterministic public quadrature microbenchmarks",
        "interval_seconds": 3600,
        "observed_counts": [10, 20, 30],
        "cases": cases,
        "adversarial_long_period": {
            "interval_seconds": 40500,
            "initial_subintervals": adversarial.diagnostics.initial_subintervals,
            "initial_subintervals_evaluated": adversarial.diagnostics.initial_subintervals_evaluated,
            "evaluation_budget": adversarial.diagnostics.evaluation_budget,
            "baseline_evaluations": adversarial.diagnostics.baseline_evaluations,
            "refinement_evaluations": adversarial.diagnostics.refinement_evaluations,
            "routing_evaluations": adversarial.diagnostics.routing_evaluations,
            "stable_interval_weight": adversarial.diagnostics.stable_interval_weight,
            "unresolved_interval_weight": adversarial.diagnostics.unresolved_interval_weight,
            "sample_cap_reached": adversarial.diagnostics.sample_cap_reached,
            "fingerprint": adversarial.diagnostics.fingerprint,
            "depth_first_reference": {
                "routing_evaluations": depth_first.diagnostics.routing_evaluations,
                "unresolved_interval_weight": depth_first.diagnostics.unresolved_interval_weight,
                "sample_cap_reached": depth_first.diagnostics.sample_cap_reached,
            },
        },
        "long_period_budget_comparison": {
            str(budget): _long_adaptive_budget(budget)
            for budget in (128, 256, 512)
        },
        "comparison_mode_evaluation": {
            "integral_response": (
                "embedded sparse interval comparison with global error control"
            ),
            "exact_service_identity": (
                "preserves exact trip-leg changes; diagnostic reference, often too strict"
            ),
            "measurement_support": (
                "preserves active measurement-row support but ignores share changes"
            ),
            "aggregate_response": (
                "preserves active sparse expected responses and is the integrated default"
            ),
            "two_stage": (
                "currently resolves to aggregate_response; optimization remains future work"
            ),
            "route_pattern_signature": (
                "stable fallback for studies without active measurements"
            ),
        },
        "timetable_discrete_measurement_rows": discrete_report,
        "recommendation": {
            "strategy": "adaptive_service_aware",
            "initial_interval_seconds": 900,
            "minimum_interval_seconds": 60,
            "response_tolerance": 1.0e-3,
            "comparison_mode": "integral_response",
            "absolute_response_tolerance": 1.0e-3,
            "relative_response_tolerance": 2.0e-2,
            "maximum_samples_per_cell": 128,
            "infeasible_policy": "preserve_mass",
            "caveat": "Validate on a 20--50 origin-period-group private pilot before a full rebuild.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    arguments = parser.parse_args()
    report = run()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        from pathlib import Path

        Path(arguments.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
