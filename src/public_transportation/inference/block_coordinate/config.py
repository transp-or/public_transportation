"""Validated configuration contracts for block-coordinate MAP estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ._canonical import fingerprint

BlockOrder = Literal["cyclic", "shuffled", "interleaved"]
BlockSizingMode = Literal["explicit", "auto"]
InitialPredictionMode = Literal["compute", "provided"]
PredictionValidationMode = Literal["exact", "sampled", "deferred"]


@dataclass(frozen=True, slots=True)
class GlobalProductPolicy:
    """Control costly complete-operator products independently of block updates."""

    initial_prediction_mode: InitialPredictionMode = "compute"
    initial_prediction_validation: PredictionValidationMode = "deferred"
    initial_exact_gradient: bool = True
    resume_prediction_validation: PredictionValidationMode = "exact"
    final_prediction_validation: PredictionValidationMode = "deferred"
    final_exact_gradient: bool = False

    def __post_init__(self) -> None:
        if self.initial_prediction_mode not in {"compute", "provided"}:
            raise ValueError("initial_prediction_mode must be 'compute' or 'provided'.")
        for name in (
            "initial_prediction_validation",
            "resume_prediction_validation",
            "final_prediction_validation",
        ):
            if getattr(self, name) not in {"exact", "sampled", "deferred"}:
                raise ValueError(f"{name} must be 'exact', 'sampled', or 'deferred'.")


@dataclass(frozen=True, slots=True)
class BlockSizingConfig:
    mode: BlockSizingMode
    maximum_free_variables_per_block: int | None = None
    maximum_operator_nonzeros_per_block: int | None = None
    maximum_worker_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"explicit", "auto"}:
            raise ValueError("mode must be 'explicit' or 'auto'.")
        for name in (
            "maximum_free_variables_per_block",
            "maximum_operator_nonzeros_per_block",
            "maximum_worker_memory_bytes",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided.")
        if self.mode == "explicit" and all(
            value is None
            for value in (
                self.maximum_free_variables_per_block,
                self.maximum_operator_nonzeros_per_block,
                self.maximum_worker_memory_bytes,
            )
        ):
            raise ValueError("explicit block sizing requires at least one hard ceiling.")


@dataclass(frozen=True, slots=True)
class BlockCoordinateMAPConfig:
    maximum_sweeps: int | None = None
    maximum_block_updates: int | None = None
    maximum_elapsed_seconds: float | None = None
    block_solver_max_iterations: int = 50
    block_solver_tolerance: float = 1.0e-8
    global_projected_gradient_tolerance: float | None = 1.0e-6
    relative_sweep_objective_tolerance: float | None = 1.0e-8
    maximum_flow_change_tolerance: float | None = None
    block_order: BlockOrder = "cyclic"
    random_seed: int = 0
    update_damping: float = 1.0
    checkpoint_directory: Path = Path("block-coordinate-checkpoint")
    save_after_every_block: bool = True
    compact_checkpoint_every_blocks: int = 100
    compact_checkpoint_every_seconds: float = 300.0
    exact_global_diagnostic_every_sweeps: int | None = 1
    global_product_policy: GlobalProductPolicy = GlobalProductPolicy()
    sampled_gradient_blocks: int = 1
    construction_workers: int = 1
    solver_workers: int = 1
    threads_per_worker: int = 1
    pilot_block_schedule: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("maximum_sweeps", "maximum_block_updates"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided.")
        if self.maximum_elapsed_seconds is not None and (
            not math.isfinite(self.maximum_elapsed_seconds)
            or self.maximum_elapsed_seconds <= 0.0
        ):
            raise ValueError("maximum_elapsed_seconds must be finite and positive.")
        if self.block_solver_max_iterations <= 0:
            raise ValueError("block_solver_max_iterations must be positive.")
        for name in (
            "block_solver_tolerance",
            "global_projected_gradient_tolerance",
            "relative_sweep_objective_tolerance",
            "maximum_flow_change_tolerance",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.block_solver_tolerance == 0.0:
            raise ValueError("block_solver_tolerance must be strictly positive.")
        if self.block_order not in {"cyclic", "shuffled", "interleaved"}:
            raise ValueError("block_order is invalid.")
        if not 0.0 < self.update_damping <= 1.0:
            raise ValueError("update_damping must be in (0, 1].")
        checkpoint = Path(self.checkpoint_directory).expanduser()
        if not str(checkpoint):
            raise ValueError("checkpoint_directory must be nonempty.")
        object.__setattr__(self, "checkpoint_directory", checkpoint)
        if self.compact_checkpoint_every_blocks <= 0:
            raise ValueError("compact_checkpoint_every_blocks must be positive.")
        if (
            not math.isfinite(self.compact_checkpoint_every_seconds)
            or self.compact_checkpoint_every_seconds <= 0.0
        ):
            raise ValueError("compact_checkpoint_every_seconds must be finite and positive.")
        if (
            self.exact_global_diagnostic_every_sweeps is not None
            and self.exact_global_diagnostic_every_sweeps <= 0
        ):
            raise ValueError("exact_global_diagnostic_every_sweeps must be positive.")
        if self.sampled_gradient_blocks < 0:
            raise ValueError("sampled_gradient_blocks must be non-negative.")
        for name in ("construction_workers", "solver_workers", "threads_per_worker"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.pilot_block_schedule is not None:
            schedule = tuple(str(value).strip() for value in self.pilot_block_schedule)
            if not schedule or any(not value for value in schedule):
                raise ValueError("pilot_block_schedule must contain nonempty block IDs.")
            if len(schedule) != len(set(schedule)):
                raise ValueError("pilot_block_schedule must not repeat block IDs.")
            object.__setattr__(self, "pilot_block_schedule", schedule)

    @property
    def fingerprint(self) -> str:
        # Invocation limits may be extended on resume; numerical semantics may not.
        return fingerprint(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name
                not in {
                    "maximum_sweeps",
                    "maximum_block_updates",
                    "maximum_elapsed_seconds",
                    "checkpoint_directory",
                }
            }
        )
