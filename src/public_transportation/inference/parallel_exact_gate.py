"""Numerical and performance promotion gate for the parallel exact backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


ParallelExactRecommendation = Literal["retain_existing", "promote_parallel"]


@dataclass(frozen=True, slots=True)
class ParallelExactGateConfig:
    forward_max_abs_tolerance: float = 5.0e-4
    reverse_max_abs_tolerance: float = 1.0e-5
    objective_abs_tolerance: float = 1.0e-3
    gradient_relative_tolerance: float = 1.0e-5
    minimum_speedup: float = 1.10
    require_all_worker_lanes: bool = True

    def __post_init__(self) -> None:
        for name in (
            "forward_max_abs_tolerance",
            "reverse_max_abs_tolerance",
            "objective_abs_tolerance",
            "gradient_relative_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not np.isfinite(self.minimum_speedup) or self.minimum_speedup <= 1.0:
            raise ValueError("minimum_speedup must be finite and greater than one.")


@dataclass(frozen=True, slots=True)
class ParallelExactGateReport:
    numerical_equivalence_passed: bool
    performance_passed: bool
    worker_utilization_passed: bool
    promotion_passed: bool
    recommendation: ParallelExactRecommendation
    forward_max_abs_error: float
    reverse_max_abs_error: float
    objective_abs_error: float
    gradient_relative_error: float
    existing_exact_seconds: float
    parallel_exact_seconds: float
    measured_speedup: float
    requested_workers: int
    observed_worker_lanes: int
    reasons: tuple[str, ...]
    config: ParallelExactGateConfig
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["config"] = asdict(self.config)
        return result


def assess_parallel_exact_gate(
    *,
    reference_forward: object,
    parallel_forward: object,
    reference_reverse: object,
    parallel_reverse: object,
    reference_objective: float,
    parallel_objective: float,
    reference_gradient: object,
    parallel_gradient: object,
    existing_exact_seconds: float,
    parallel_exact_seconds: float,
    requested_workers: int,
    observed_worker_lanes: int,
    config: ParallelExactGateConfig = ParallelExactGateConfig(),
) -> ParallelExactGateReport:
    """Require numerical equivalence, useful speedup, and worker utilization."""
    if existing_exact_seconds <= 0.0 or parallel_exact_seconds <= 0.0:
        raise ValueError("exact execution times must be positive.")
    if requested_workers <= 0 or observed_worker_lanes <= 0:
        raise ValueError("worker counts must be positive.")
    forward_error = float(
        np.max(np.abs(np.asarray(parallel_forward) - np.asarray(reference_forward)))
    )
    reverse_error = float(
        np.max(np.abs(np.asarray(parallel_reverse) - np.asarray(reference_reverse)))
    )
    objective_error = abs(float(parallel_objective) - float(reference_objective))
    reference_gradient_array = np.asarray(reference_gradient)
    gradient_error = float(
        np.linalg.norm(np.asarray(parallel_gradient) - reference_gradient_array)
        / max(np.linalg.norm(reference_gradient_array), np.finfo(float).eps)
    )
    numerical = (
        forward_error <= config.forward_max_abs_tolerance
        and reverse_error <= config.reverse_max_abs_tolerance
        and objective_error <= config.objective_abs_tolerance
        and gradient_error <= config.gradient_relative_tolerance
    )
    speedup = existing_exact_seconds / parallel_exact_seconds
    performance = speedup >= config.minimum_speedup
    utilization = (
        observed_worker_lanes >= requested_workers
        if config.require_all_worker_lanes
        else observed_worker_lanes > 0
    )
    passed = numerical and performance and utilization
    reasons = []
    if not numerical:
        reasons.append("numerical equivalence tolerance was not met")
    if not performance:
        reasons.append(
            f"measured speedup {speedup:.3f} is below {config.minimum_speedup:.3f}"
        )
    if not utilization:
        reasons.append("not all requested worker lanes performed routing work")
    if passed:
        reasons.append("all numerical, performance, and utilization gates passed")
    return ParallelExactGateReport(
        numerical_equivalence_passed=numerical,
        performance_passed=performance,
        worker_utilization_passed=utilization,
        promotion_passed=passed,
        recommendation="promote_parallel" if passed else "retain_existing",
        forward_max_abs_error=forward_error,
        reverse_max_abs_error=reverse_error,
        objective_abs_error=objective_error,
        gradient_relative_error=gradient_error,
        existing_exact_seconds=float(existing_exact_seconds),
        parallel_exact_seconds=float(parallel_exact_seconds),
        measured_speedup=float(speedup),
        requested_workers=requested_workers,
        observed_worker_lanes=observed_worker_lanes,
        reasons=tuple(reasons),
        config=config,
    )
