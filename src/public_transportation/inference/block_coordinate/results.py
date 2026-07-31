"""Immutable anytime state and result contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ._canonical import immutable_float_vector
from .checkpoint import BlockCoordinateFingerprints
from .progress import DiagnosticValue

BlockCoordinateStatus = Literal[
    "converged",
    "stopped_by_time_budget",
    "stopped_by_sweep_budget",
    "stopped_by_update_budget",
    "interrupted_with_approximate_solution",
    "resource_guard_triggered",
    "numerical_failure",
]

VALID_BLOCK_COORDINATE_STATUSES: tuple[BlockCoordinateStatus, ...] = (
    "converged",
    "stopped_by_time_budget",
    "stopped_by_sweep_budget",
    "stopped_by_update_budget",
    "interrupted_with_approximate_solution",
    "resource_guard_triggered",
    "numerical_failure",
)


@dataclass(frozen=True, slots=True)
class BlockObjectiveComponents:
    data: float
    prior: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.data) or not math.isfinite(self.prior):
            raise ValueError("objective components must be finite.")

    @property
    def total(self) -> float:
        return self.data + self.prior


@dataclass(frozen=True, slots=True)
class BlockConvergenceDiagnostics:
    latest_block_projected_gradient: DiagnosticValue
    estimated_global_projected_gradient: DiagnosticValue
    exact_global_projected_gradient: DiagnosticValue
    maximum_block_flow_change: float
    initialization_objective_improvement: float
    current_sweep_objective_improvement: float
    previous_sweep_objective_improvement: float | None

    def __post_init__(self) -> None:
        numeric = (
            self.maximum_block_flow_change,
            self.initialization_objective_improvement,
            self.current_sweep_objective_improvement,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("convergence diagnostics must be finite.")
        if self.previous_sweep_objective_improvement is not None and not math.isfinite(
            self.previous_sweep_objective_improvement
        ):
            raise ValueError("previous sweep improvement must be finite.")


@dataclass(frozen=True, slots=True)
class BlockCoordinateState:
    current_free_flow: np.ndarray
    best_free_flow: np.ndarray
    current_prediction: np.ndarray
    fixed_measurement_offset: np.ndarray
    current_objective: float
    best_objective: float
    current_components: BlockObjectiveComponents
    best_components: BlockObjectiveComponents
    sweep: int
    schedule_position: int
    accepted_updates: int
    rejected_updates: int
    elapsed_seconds: float
    block_schedule: tuple[str, ...]
    random_state_json: str
    diagnostics: BlockConvergenceDiagnostics
    fingerprints: BlockCoordinateFingerprints

    def __post_init__(self) -> None:
        current = immutable_float_vector(self.current_free_flow, name="current_free_flow")
        best = immutable_float_vector(self.best_free_flow, name="best_free_flow")
        prediction = immutable_float_vector(
            self.current_prediction, name="current_prediction"
        )
        offset = immutable_float_vector(
            self.fixed_measurement_offset, name="fixed_measurement_offset"
        )
        if current.shape != best.shape:
            raise ValueError("current and best free-flow vectors must have equal shape.")
        if prediction.shape != offset.shape:
            raise ValueError("prediction and fixed offset must have equal shape.")
        if np.any(current < 0.0) or np.any(best < 0.0):
            raise ValueError("free-flow vectors must be non-negative.")
        if not math.isfinite(self.current_objective) or not math.isfinite(
            self.best_objective
        ):
            raise ValueError("objectives must be finite.")
        if self.best_objective > self.current_objective:
            raise ValueError("best_objective must not exceed current_objective.")
        if not np.isclose(self.current_objective, self.current_components.total):
            raise ValueError("current objective components are inconsistent.")
        if not np.isclose(self.best_objective, self.best_components.total):
            raise ValueError("best objective components are inconsistent.")
        for name in (
            "sweep",
            "schedule_position",
            "accepted_updates",
            "rejected_updates",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        schedule = tuple(str(value).strip() for value in self.block_schedule)
        if not schedule or any(not value for value in schedule):
            raise ValueError("block_schedule must contain nonempty identifiers.")
        if self.schedule_position > len(schedule):
            raise ValueError("schedule_position exceeds the block schedule.")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative.")
        if not self.random_state_json.strip():
            raise ValueError("random_state_json must be nonempty.")
        object.__setattr__(self, "current_free_flow", current)
        object.__setattr__(self, "best_free_flow", best)
        object.__setattr__(self, "current_prediction", prediction)
        object.__setattr__(self, "fixed_measurement_offset", offset)
        object.__setattr__(self, "block_schedule", schedule)


@dataclass(frozen=True, slots=True)
class BlockCoordinateMAPResult:
    status: BlockCoordinateStatus
    message: str
    state: BlockCoordinateState
    checkpoint_directory: Path
    resume_configuration_fingerprint: str
    work: "BlockCoordinateWorkDiagnostics | None" = None

    def __post_init__(self) -> None:
        if self.status not in VALID_BLOCK_COORDINATE_STATUSES:
            raise ValueError("invalid block-coordinate result status.")
        if not self.message.strip():
            raise ValueError("result message must be nonempty.")
        if not self.resume_configuration_fingerprint.strip():
            raise ValueError("resume configuration fingerprint must be nonempty.")
        object.__setattr__(
            self, "checkpoint_directory", Path(self.checkpoint_directory).expanduser()
        )

    @property
    def latest_free_flow(self) -> np.ndarray:
        return self.state.current_free_flow

    @property
    def best_free_flow(self) -> np.ndarray:
        return self.state.best_free_flow


@dataclass(frozen=True, slots=True)
class BlockCoordinateWorkDiagnostics:
    """Observable costly work performed by one estimator invocation."""

    initial_prediction_source: str
    global_forward_count: int = 0
    global_forward_seconds: float = 0.0
    global_transpose_count: int = 0
    global_transpose_seconds: float = 0.0
    selected_block_construction_seconds: float = 0.0
    selected_block_cache_hits: int = 0
    selected_block_cache_misses: int = 0
    selected_block_construction_attempts: int = 0
    selected_block_constructions_completed: int = 0
    selected_block_constructions_deadline_stopped: int = 0
    block_solve_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    resume_prediction_validation: str = "not_applicable"
    final_prediction_validation: str = "deferred"
    deadline_exceeded_by_indivisible_operation: bool = False
    selected_block_deadline_phase: str | None = None
    selected_block_deadline_overshoot_seconds: float = 0.0
    solver_started: bool = False
    checkpoint_preserved: bool = False
    scheduled_block_not_attempted_by_solver: str | None = None

    def __post_init__(self) -> None:
        if not self.initial_prediction_source:
            raise ValueError("initial_prediction_source must be nonempty.")
        for name in (
            "global_forward_count",
            "global_transpose_count",
            "selected_block_cache_hits",
            "selected_block_cache_misses",
            "selected_block_construction_attempts",
            "selected_block_constructions_completed",
            "selected_block_constructions_deadline_stopped",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        for name in (
            "global_forward_seconds",
            "global_transpose_seconds",
            "selected_block_construction_seconds",
            "block_solve_seconds",
            "checkpoint_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.selected_block_deadline_overshoot_seconds < 0.0:
            raise ValueError("selected block deadline overshoot must be non-negative.")
