"""Minimal reduced conditional-gravity model specification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


ReducedGravityLikelihood = Literal["poisson", "negative_binomial"]
ReducedGravityProductionMode = Literal["provided", "estimated_basis"]
ReducedGravityAttractivenessMode = Literal["provided", "estimated_basis"]


@dataclass(frozen=True, slots=True)
class MinimalGravitySpecification:
    likelihood: ReducedGravityLikelihood = "poisson"
    production_mode: ReducedGravityProductionMode = "provided"
    production_basis_columns: int = 0
    destination_attractiveness_mode: ReducedGravityAttractivenessMode = "provided"
    destination_attractiveness_basis_columns: int = 0
    journey_time_scale_seconds: float = 1800.0
    positivity_floor: float = 1.0e-6
    mean_floor: float = 1.0e-9

    def __post_init__(self) -> None:
        if self.likelihood not in {"poisson", "negative_binomial"}:
            raise ValueError("unsupported reduced gravity likelihood.")
        if self.production_mode not in {"provided", "estimated_basis"}:
            raise ValueError("unsupported reduced gravity production mode.")
        expected_columns = self.production_basis_columns
        if self.production_mode == "provided" and expected_columns != 0:
            raise ValueError("provided productions require zero basis columns.")
        if self.production_mode == "estimated_basis" and expected_columns <= 0:
            raise ValueError("estimated productions require positive basis columns.")
        if self.destination_attractiveness_mode not in {"provided", "estimated_basis"}:
            raise ValueError("unsupported destination-attractiveness mode.")
        destination_columns = self.destination_attractiveness_basis_columns
        if self.destination_attractiveness_mode == "provided" and destination_columns != 0:
            raise ValueError(
                "provided destination attractiveness requires zero basis columns."
            )
        if self.destination_attractiveness_mode == "estimated_basis" and destination_columns <= 0:
            raise ValueError(
                "estimated destination attractiveness requires positive basis columns."
            )
        for value, name in (
            (self.journey_time_scale_seconds, "journey_time_scale_seconds"),
            (self.positivity_floor, "positivity_floor"),
            (self.mean_floor, "mean_floor"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

    @property
    def parameter_count(self) -> int:
        return (
            2
            + (1 if self.likelihood == "negative_binomial" else 0)
            + self.production_basis_columns
            + self.destination_attractiveness_basis_columns
        )
