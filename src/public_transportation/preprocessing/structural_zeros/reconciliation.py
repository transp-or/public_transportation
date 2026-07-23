"""Strict reconciliation of detected structural zeros with fixed demand."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from public_transportation.domain.fixed_demand import (
    FixedODDemand,
    FixedODKey,
    FixedODRecord,
    read_fixed_demand_csv,
)

from .config import StructuralZeroConfig
from .errors import StructuralZeroConflictError
from .types import StructuralZeroAnalysisResult

if TYPE_CHECKING:
    from public_transportation.domain.scenario import Scenario


@dataclass(frozen=True, slots=True)
class FixedDemandReconciliationResult:
    """Merged fixed demand and deterministic reconciliation counts."""

    fixed_demand: FixedODDemand
    num_existing: int
    num_structural_zero: int
    num_existing_structural_zero: int
    num_added_structural_zero: int

    def __post_init__(self) -> None:
        if not isinstance(self.fixed_demand, FixedODDemand):
            raise TypeError("fixed_demand must be a FixedODDemand.")
        values = (
            self.num_existing,
            self.num_structural_zero,
            self.num_existing_structural_zero,
            self.num_added_structural_zero,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("Reconciliation counts must be non-negative integers.")
        if (
            self.num_existing_structural_zero + self.num_added_structural_zero
            != self.num_structural_zero
        ):
            raise ValueError("Structural-zero reconciliation counts are inconsistent.")
        if self.num_existing_structural_zero > self.num_existing:
            raise ValueError(
                "Existing structural-zero count exceeds existing demand count."
            )
        if len(self.fixed_demand) != self.num_existing + self.num_added_structural_zero:
            raise ValueError("Merged fixed-demand count is inconsistent.")

    @property
    def num_merged(self) -> int:
        return len(self.fixed_demand)


def reconcile_fixed_demand(
    analysis: StructuralZeroAnalysisResult,
    existing: FixedODDemand | None = None,
) -> FixedDemandReconciliationResult:
    """Merge detected zero cells with existing fixed values.

    A nonzero existing value on a detected structural-zero key is always an
    error. Compatible existing values retain their exact values. Every newly
    detected key is added with value zero.
    """
    analysis_keys = {record.key.tuple for record in analysis.records}
    structural_zero_keys = {
        record.key.tuple for record in analysis.records if record.is_structural_zero
    }
    existing_records = () if existing is None else existing.records

    existing_by_key: dict[FixedODKey, float] = {}
    for record in existing_records:
        if not isinstance(record, FixedODRecord):
            raise TypeError("Existing fixed demand must contain FixedODRecord values.")
        key = record.key
        if key in existing_by_key:
            raise ValueError(f"Existing fixed demand contains duplicate key {key!r}.")
        if key not in analysis_keys:
            raise ValueError(
                f"Existing fixed-demand key {key!r} is outside the analyzed OD/time universe."
            )
        value = float(record.fixed_flow)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Existing fixed-demand value for {key!r} must be finite and non-negative."
            )
        existing_by_key[key] = value

    conflicts = tuple(
        (*key, existing_by_key[key])
        for key in sorted(structural_zero_keys & set(existing_by_key))
        if existing_by_key[key] != 0.0
    )
    if conflicts:
        raise StructuralZeroConflictError(conflicts)

    merged = dict(existing_by_key)
    for key in structural_zero_keys:
        merged.setdefault(key, 0.0)
    fixed_demand = FixedODDemand(
        records=tuple(
            FixedODRecord(
                origin_stop_id=key[0],
                dest_stop_id=key[1],
                time_bin_id=key[2],
                fixed_flow=value,
            )
            for key, value in sorted(merged.items())
        )
    )
    overlap = len(structural_zero_keys & set(existing_by_key))
    return FixedDemandReconciliationResult(
        fixed_demand=fixed_demand,
        num_existing=len(existing_by_key),
        num_structural_zero=len(structural_zero_keys),
        num_existing_structural_zero=overlap,
        num_added_structural_zero=len(structural_zero_keys) - overlap,
    )


def load_and_reconcile_fixed_demand(
    analysis: StructuralZeroAnalysisResult,
    config: StructuralZeroConfig,
    *,
    scenario: Scenario,
) -> FixedDemandReconciliationResult:
    """Load the configured existing file, when present, and reconcile it."""
    existing = None
    if config.existing_fixed_demand is not None:
        existing = read_fixed_demand_csv(
            config.existing_fixed_demand.file,
            scenario=scenario,
        )
    return reconcile_fixed_demand(analysis, existing)
