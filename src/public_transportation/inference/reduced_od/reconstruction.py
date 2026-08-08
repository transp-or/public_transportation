"""Explicit post-fit OD reconstruction and one-shot detailed validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from public_transportation.preprocessing.reduced_od.response_atoms import (
    ResponseCellKey,
)

from .contracts import JourneyODTimeKey, ReducedODProblemContract


@dataclass(frozen=True, slots=True)
class ReconstructedODRow:
    key: JourneyODTimeKey
    demand: float
    estimated: bool


@dataclass(frozen=True, slots=True)
class ReconstructedOD:
    """Canonical full OD vector, created only for output or validation."""

    keys: tuple[JourneyODTimeKey, ...]
    demand: np.ndarray
    estimated: np.ndarray

    def __post_init__(self) -> None:
        demand = np.array(self.demand, dtype=np.float64, copy=True)
        estimated = np.array(self.estimated, dtype=np.bool_, copy=True)
        if demand.shape != (len(self.keys),) or estimated.shape != demand.shape:
            raise ValueError("reconstructed arrays must align with keys.")
        if not np.all(np.isfinite(demand)) or np.any(demand < 0.0):
            raise ValueError("reconstructed demand must be finite and non-negative.")
        demand.setflags(write=False)
        estimated.setflags(write=False)
        object.__setattr__(self, "demand", demand)
        object.__setattr__(self, "estimated", estimated)

    @property
    def rows(self) -> tuple[ReconstructedODRow, ...]:
        return tuple(
            ReconstructedODRow(key, float(value), bool(estimated))
            for key, value, estimated in zip(
                self.keys, self.demand, self.estimated, strict=True
            )
        )


def reconstruct_full_od(
    *,
    contract: ReducedODProblemContract,
    free_cell_keys: tuple[ResponseCellKey, ...],
    free_demand: object,
) -> ReconstructedOD:
    """Place compact fitted demand and fixed constants in canonical OD order."""
    values = np.asarray(free_demand, dtype=np.float64)
    if values.shape != (contract.num_free_od,):
        raise ValueError("free_demand does not match the contract free dimension.")
    expected = tuple(
        JourneyODTimeKey(
            key.origin_physical_stop_id,
            key.destination_physical_stop_id,
            key.origin_time_period_id,
        )
        for key in free_cell_keys
    )
    contract_free_keys = tuple(
        contract.od_keys[index] for index in contract.free_od_indices
    )
    if expected != contract_free_keys:
        raise ValueError("free_cell_keys do not match the contract free OD order.")
    full = np.empty(contract.num_od, dtype=np.float64)
    full[contract.free_od_indices] = values
    full[contract.fixed_od_indices] = contract.fixed_od_values
    estimated = np.zeros(contract.num_od, dtype=np.bool_)
    estimated[contract.free_od_indices] = True
    return ReconstructedOD(keys=contract.od_keys, demand=full, estimated=estimated)


@dataclass(frozen=True, slots=True)
class DetailedAssignmentOutput:
    measurement_prediction: np.ndarray
    transfer_audit: Mapping[str, float]

    def __post_init__(self) -> None:
        prediction = np.array(self.measurement_prediction, dtype=np.float64, copy=True)
        if prediction.ndim != 1 or not np.all(np.isfinite(prediction)):
            raise ValueError("detailed measurement prediction must be a finite vector.")
        prediction.setflags(write=False)
        object.__setattr__(self, "measurement_prediction", prediction)
        audit = {str(key): float(value) for key, value in self.transfer_audit.items()}
        if any(not np.isfinite(value) for value in audit.values()):
            raise ValueError("transfer audit values must be finite.")
        object.__setattr__(self, "transfer_audit", audit)


@dataclass(frozen=True, slots=True)
class DetailedValidationResult:
    compact_prediction: np.ndarray
    detailed_prediction: np.ndarray
    difference: np.ndarray
    maximum_absolute_error: float
    root_mean_square_error: float
    relative_l2_error: float
    transfer_audit: Mapping[str, float]


DetailedAssignmentRunner = Callable[[ReconstructedOD], DetailedAssignmentOutput]


def validate_with_detailed_assignment(
    *,
    reconstructed_od: ReconstructedOD,
    compact_prediction: object,
    run_detailed_assignment: DetailedAssignmentRunner,
) -> DetailedValidationResult:
    """Run detailed assignment exactly once after fitting and compare counts."""
    compact = np.asarray(compact_prediction, dtype=np.float64)
    if compact.ndim != 1 or not np.all(np.isfinite(compact)):
        raise ValueError("compact_prediction must be a finite vector.")
    detailed = run_detailed_assignment(reconstructed_od)
    if detailed.measurement_prediction.shape != compact.shape:
        raise ValueError("compact and detailed predictions must have equal shape.")
    difference = detailed.measurement_prediction - compact
    denominator = float(np.linalg.norm(compact))
    relative = (
        float(np.linalg.norm(difference) / denominator)
        if denominator
        else (0.0 if not np.any(difference) else float("inf"))
    )
    return DetailedValidationResult(
        compact_prediction=np.array(compact, copy=True),
        detailed_prediction=np.array(detailed.measurement_prediction, copy=True),
        difference=np.array(difference, copy=True),
        maximum_absolute_error=float(np.max(np.abs(difference), initial=0.0)),
        root_mean_square_error=(
            float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0
        ),
        relative_l2_error=relative,
        transfer_audit=detailed.transfer_audit,
    )
