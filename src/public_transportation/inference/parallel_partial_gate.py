"""Accuracy and speed acceptance gate for public partial-routing benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class PartialEffortRequirement:
    effort_percent: float
    minimum_speedup: float
    maximum_gradient_relative_error: float
    maximum_count_relative_error: float


DEFAULT_PARTIAL_EFFORT_REQUIREMENTS = (
    PartialEffortRequirement(10.0, 3.0, 0.10, 0.20),
    PartialEffortRequirement(25.0, 2.5, 0.10, 0.10),
    PartialEffortRequirement(50.0, 1.5, 0.05, 0.06),
    PartialEffortRequirement(75.0, 1.1, 0.02, 0.03),
)


@dataclass(frozen=True, slots=True)
class PartialEffortGateResult:
    effort_percent: float
    passed: bool
    measured_speedup: float
    gradient_relative_error: float
    count_relative_error: float
    reasons: tuple[str, ...]
    requirement: PartialEffortRequirement


@dataclass(frozen=True, slots=True)
class ParallelPartialGateReport:
    passed: bool
    effort_results: tuple[PartialEffortGateResult, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "effort_results": [asdict(item) for item in self.effort_results],
        }


def assess_parallel_partial_gate(
    rows: Sequence[Mapping[str, object]],
    *,
    requirements: tuple[PartialEffortRequirement, ...] = (
        DEFAULT_PARTIAL_EFFORT_REQUIREMENTS
    ),
) -> ParallelPartialGateReport:
    """Require every declared effort level to meet speed and accuracy limits."""
    by_effort = {float(row["requested_effort_percent"]): row for row in rows}
    results = []
    for requirement in requirements:
        row = by_effort.get(requirement.effort_percent)
        if row is None:
            raise ValueError(
                f"missing partial benchmark effort {requirement.effort_percent}."
            )
        speedup = float(row["speedup_over_exact"])
        gradient_error = float(row["gradient_relative_norm_error"])
        count_error = float(row["predicted_count_relative_error"])
        if not all(np.isfinite(value) for value in (speedup, gradient_error, count_error)):
            raise ValueError("partial benchmark metrics must be finite.")
        reasons = []
        if speedup < requirement.minimum_speedup:
            reasons.append("speedup is below the required threshold")
        if gradient_error > requirement.maximum_gradient_relative_error:
            reasons.append("gradient error exceeds the required threshold")
        if count_error > requirement.maximum_count_relative_error:
            reasons.append("predicted-count error exceeds the required threshold")
        results.append(
            PartialEffortGateResult(
                effort_percent=requirement.effort_percent,
                passed=not reasons,
                measured_speedup=speedup,
                gradient_relative_error=gradient_error,
                count_relative_error=count_error,
                reasons=tuple(reasons),
                requirement=requirement,
            )
        )
    return ParallelPartialGateReport(
        passed=all(item.passed for item in results), effort_results=tuple(results)
    )
