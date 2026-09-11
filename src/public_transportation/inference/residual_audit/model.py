"""Public data contracts for residual auditing.

The residual-audit package deliberately operates on persisted tables.  It does
not know how a network, an assignment operator, or a private case adapter was
constructed, so an audit can be rerun without activating the scientific model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class ResidualAuditConfig:
    """Numerical thresholds used by a residual audit.

    ``positive_observation_threshold`` is also used to define the positive
    observation flag.  Relative residuals have a separate denominator
    threshold because very small positive counts can still be uninformative.
    """

    positive_observation_threshold: float = 0.5
    predicted_mean_floor: float = 1.0e-8
    pearson_threshold: float = 3.0
    variance_floor: float = 1.0e-8
    relative_observation_threshold: float = 1.0e-8

    def __post_init__(self) -> None:
        positive = (
            "positive_observation_threshold",
            "predicted_mean_floor",
            "pearson_threshold",
            "variance_floor",
            "relative_observation_threshold",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.predicted_mean_floor <= 0.0:
            raise ValueError("predicted_mean_floor must be strictly positive.")
        if self.variance_floor <= 0.0:
            raise ValueError("variance_floor must be strictly positive.")

    def to_dict(self) -> dict[str, float]:
        """Return deterministic JSON-compatible threshold metadata."""
        return {
            "positive_observation_threshold": float(self.positive_observation_threshold),
            "predicted_mean_floor": float(self.predicted_mean_floor),
            "pearson_threshold": float(self.pearson_threshold),
            "variance_floor": float(self.variance_floor),
            "relative_observation_threshold": float(self.relative_observation_threshold),
        }


@dataclass(frozen=True, slots=True)
class ResidualAuditRun:
    """Normalized persisted inputs for one model run."""

    observations: pd.DataFrame
    contributions: pd.DataFrame | None = None
    metadata: pd.DataFrame | None = None
    input_paths: dict[str, str] = field(default_factory=dict)
    input_fingerprints: dict[str, str] = field(default_factory=dict)
    model_fingerprints: tuple[str, ...] = ()
    specification_fingerprints: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResidualAuditResult:
    """In-memory result produced by :func:`audit_run`."""

    row_diagnostics: pd.DataFrame
    grouped_metrics: pd.DataFrame
    support_failures: pd.DataFrame
    contribution_summary: pd.DataFrame
    comparison: pd.DataFrame
    comparison_grouped_metrics: pd.DataFrame
    manifest: dict[str, Any]
    summary_markdown: str
    output_files: dict[str, Path] = field(default_factory=dict)
