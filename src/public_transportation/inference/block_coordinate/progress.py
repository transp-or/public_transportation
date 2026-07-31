"""Structured progress contracts and precision semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ._canonical import canonical_json

DiagnosticKind = Literal["exact", "sampled", "stale", "deferred", "unavailable"]


@dataclass(frozen=True, slots=True)
class DiagnosticValue:
    value: float | None
    kind: DiagnosticKind
    computed_at_sweep: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"exact", "sampled", "stale", "deferred", "unavailable"}:
            raise ValueError("invalid diagnostic kind.")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("diagnostic value must be finite when available.")
        if self.kind in {"unavailable", "deferred"} and self.value is not None:
            raise ValueError("unavailable or deferred diagnostics cannot contain a value.")
        if self.kind not in {"unavailable", "deferred"} and self.value is None:
            raise ValueError("available diagnostics require a value.")
        if self.computed_at_sweep is not None and self.computed_at_sweep < 0:
            raise ValueError("computed_at_sweep must be non-negative.")


@dataclass(frozen=True, slots=True)
class BlockProgressEvent:
    sweep: int
    block_or_batch: str
    blocks_completed_in_sweep: int
    total_blocks: int
    variables_visited: int
    total_variables: int
    elapsed_seconds: float
    current_objective: float
    best_objective: float
    data_objective: float
    prior_objective: float
    latest_objective_improvement: float
    latest_block_flow_change: float
    latest_block_projected_gradient: float
    estimated_global_projected_gradient: DiagnosticValue
    exact_global_projected_gradient: DiagnosticValue
    last_exact_global_diagnostic_sweep: int | None
    checkpoint_committed: bool
    estimated_remaining_sweep_seconds: float | None
    initial_prediction_source: str = "compute"
    global_forward_count: int = 0
    global_forward_seconds: float = 0.0
    global_transpose_count: int = 0
    global_transpose_seconds: float = 0.0
    selected_block_construction_seconds: float = 0.0
    selected_block_cache_hits: int = 0
    selected_block_cache_misses: int = 0
    block_solve_seconds: float = 0.0
    checkpoint_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.sweep < 0:
            raise ValueError("sweep must be non-negative.")
        if not self.block_or_batch.strip():
            raise ValueError("block_or_batch must be nonempty.")
        if not 0 <= self.blocks_completed_in_sweep <= self.total_blocks:
            raise ValueError("completed block count is inconsistent.")
        if self.total_blocks <= 0:
            raise ValueError("total_blocks must be positive.")
        if not 0 <= self.variables_visited <= self.total_variables:
            raise ValueError("visited variable count is inconsistent.")
        if self.total_variables <= 0:
            raise ValueError("total_variables must be positive.")
        numeric = (
            self.elapsed_seconds,
            self.current_objective,
            self.best_objective,
            self.data_objective,
            self.prior_objective,
            self.latest_objective_improvement,
            self.latest_block_flow_change,
            self.latest_block_projected_gradient,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("progress numeric fields must be finite.")
        if self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be non-negative.")
        if self.best_objective > self.current_objective:
            raise ValueError("best_objective must not exceed current_objective.")
        if self.estimated_remaining_sweep_seconds is not None and (
            not math.isfinite(self.estimated_remaining_sweep_seconds)
            or self.estimated_remaining_sweep_seconds < 0.0
        ):
            raise ValueError("estimated remaining time must be finite and non-negative.")
        if not self.initial_prediction_source:
            raise ValueError("initial_prediction_source must be nonempty.")
        if any(
            value < 0
            for value in (
                self.global_forward_count,
                self.global_transpose_count,
                self.selected_block_cache_hits,
                self.selected_block_cache_misses,
            )
        ):
            raise ValueError("work counts must be non-negative.")

    def to_json_line(self) -> str:
        return canonical_json(self) + "\n"
