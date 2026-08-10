"""Structured policy for matrix-free versus resumable sharded operators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .fixed_routing_sharded_builder import (
    ShardedConstructionPlan,
    ShardedConstructionPreflightError,
)

ShardedCacheStatus = Literal["none", "partial", "complete"]


@dataclass(frozen=True, slots=True)
class ShardedSelectionConfig:
    mode: Literal["matrix_free", "sharded", "auto"] = "auto"
    expected_products: int = 40
    estimated_construction_seconds: float | None = None
    matrix_free_product_seconds: float | None = None
    sharded_product_seconds: float | None = None
    expected_cache_reuses: int = 1
    disk_budget_bytes: int | None = None
    estimated_bytes_per_nonzero: float = 16.0

    def __post_init__(self) -> None:
        if self.mode not in {"matrix_free", "sharded", "auto"}:
            raise ValueError("invalid sharded selection mode.")
        if self.expected_products < 0 or self.expected_cache_reuses <= 0:
            raise ValueError("product and cache reuse counts are invalid.")
        for name in (
            "estimated_construction_seconds",
            "matrix_free_product_seconds",
            "sharded_product_seconds",
            "estimated_bytes_per_nonzero",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.disk_budget_bytes is not None and self.disk_budget_bytes <= 0:
            raise ValueError("disk_budget_bytes must be positive when provided.")


@dataclass(frozen=True, slots=True)
class ShardedSelectionDecision:
    selected_mode: Literal["matrix_free", "sharded"]
    reason: str
    cache_status: ShardedCacheStatus
    candidate_density_upper_bound: float
    estimated_disk_bytes_upper_bound: int
    disk_safe: bool
    kernel_memory_safe: bool
    operational_plan_safe: bool
    expected_products_across_reuses: int
    estimated_break_even_products: float | None


def select_sharded_fixed_routing_backend(
    *,
    plan: ShardedConstructionPlan,
    cache_status: ShardedCacheStatus,
    config: ShardedSelectionConfig | None = None,
) -> ShardedSelectionDecision:
    """Choose without constructing a shard or allocating global sparse storage."""
    config = ShardedSelectionConfig() if config is None else config
    if cache_status not in {"none", "partial", "complete"}:
        raise ValueError("invalid cache status.")
    logical = plan.num_measurements * plan.num_free_od
    density = 0.0 if logical == 0 else min(1.0, plan.candidate_entries / logical)
    disk = int(math.ceil(plan.candidate_entries * config.estimated_bytes_per_nonzero))
    disk_safe = config.disk_budget_bytes is None or disk <= config.disk_budget_bytes
    expected = config.expected_products * config.expected_cache_reuses
    saving = (
        None
        if config.matrix_free_product_seconds is None
        or config.sharded_product_seconds is None
        else config.matrix_free_product_seconds - config.sharded_product_seconds
    )
    break_even = (
        None
        if saving is None
        or saving <= 0
        or config.estimated_construction_seconds is None
        else config.estimated_construction_seconds / saving
    )
    kernel_memory_safe = (
        (plan.estimated_worker_memory_bytes or plan.estimated_kernel_bytes)
        <= plan.worker_memory_budget_bytes
    )
    operational_plan_safe = plan.safe and kernel_memory_safe
    safe = operational_plan_safe and disk_safe
    selected: Literal["matrix_free", "sharded"]
    if config.mode == "matrix_free":
        selected, reason = "matrix_free", "explicit matrix-free override"
    elif config.mode == "sharded":
        if not operational_plan_safe:
            raise ShardedConstructionPreflightError(plan)
        if not disk_safe:
            raise MemoryError("sharded construction plan exceeds memory or disk budget.")
        selected, reason = "sharded", "explicit bounded sharded override"
    elif cache_status == "complete":
        selected, reason = "sharded", "completed compatible cache is a sunk cost"
    elif not safe:
        selected, reason = "matrix_free", "sharded memory or disk preflight is unsafe"
    elif cache_status == "partial":
        selected, reason = "sharded", "compatible partial cache can be resumed"
    elif break_even is not None and expected <= break_even:
        selected, reason = "matrix_free", "cold construction does not amortize"
    elif break_even is None and expected < 4:
        selected, reason = "matrix_free", "one-use product count is too small"
    else:
        selected, reason = "sharded", "bounded construction amortizes across expected reuse"
    return ShardedSelectionDecision(
        selected_mode=selected,
        reason=reason,
        cache_status=cache_status,
        candidate_density_upper_bound=density,
        estimated_disk_bytes_upper_bound=disk,
        disk_safe=disk_safe,
        kernel_memory_safe=kernel_memory_safe,
        operational_plan_safe=operational_plan_safe,
        expected_products_across_reuses=expected,
        estimated_break_even_products=break_even,
    )
