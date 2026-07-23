"""Immutable outputs shared by structural-zero analysis and persistence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True, order=True)
class ODTimeKey:
    origin_stop_id: str
    dest_stop_id: str
    time_bin_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("origin_stop_id", self.origin_stop_id),
            ("dest_stop_id", self.dest_stop_id),
            ("time_bin_id", self.time_bin_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")

    @property
    def tuple(self) -> tuple[str, str, str]:
        return (self.origin_stop_id, self.dest_stop_id, self.time_bin_id)


class StructuralZeroReason(str, Enum):
    SAME_STOP = "same_stop"
    NO_FEASIBLE_PATH = "no_feasible_path"
    MAXIMUM_TRANSFERS_EXCEEDED = "maximum_transfers_exceeded"
    MAXIMUM_INITIAL_WAIT_EXCEEDED = "maximum_initial_wait_exceeded"
    MAXIMUM_JOURNEY_TIME_EXCEEDED = "maximum_journey_time_exceeded"
    INSUFFICIENT_FEASIBLE_DEPARTURES = "insufficient_feasible_departures"


@dataclass(frozen=True, slots=True)
class ODPathMetrics:
    feasible: bool
    minimum_transfers: int | None
    minimum_initial_wait_minutes: float | None
    minimum_journey_time_minutes: float | None
    feasible_departure_count: int
    earliest_arrival_seconds: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be a Boolean.")
        if isinstance(self.feasible_departure_count, bool) or not isinstance(
            self.feasible_departure_count, int
        ):
            raise TypeError("feasible_departure_count must be an integer.")
        metrics = (
            self.minimum_transfers,
            self.minimum_initial_wait_minutes,
            self.minimum_journey_time_minutes,
            self.earliest_arrival_seconds,
        )
        if not self.feasible:
            if (
                any(value is not None for value in metrics)
                or self.feasible_departure_count != 0
            ):
                raise ValueError(
                    "Infeasible path metrics must contain no path values or departures."
                )
            return
        if (
            self.minimum_transfers is None
            or isinstance(self.minimum_transfers, bool)
            or not isinstance(self.minimum_transfers, int)
            or self.minimum_transfers < 0
        ):
            raise ValueError(
                "A feasible path must have a non-negative minimum_transfers value."
            )
        if (
            self.minimum_initial_wait_minutes is None
            or not math.isfinite(self.minimum_initial_wait_minutes)
            or self.minimum_initial_wait_minutes < 0.0
        ):
            raise ValueError("A feasible path must have a non-negative initial wait.")
        if (
            self.minimum_journey_time_minutes is None
            or not math.isfinite(self.minimum_journey_time_minutes)
            or self.minimum_journey_time_minutes < 0.0
        ):
            raise ValueError("A feasible path must have a non-negative journey time.")
        if self.feasible_departure_count < 1:
            raise ValueError(
                "A feasible path must have at least one feasible departure."
            )
        if (
            self.earliest_arrival_seconds is None
            or isinstance(self.earliest_arrival_seconds, bool)
            or not isinstance(self.earliest_arrival_seconds, int)
            or self.earliest_arrival_seconds < 0
        ):
            raise ValueError(
                "A feasible path must have a non-negative earliest arrival time."
            )

    @classmethod
    def unreachable(cls) -> ODPathMetrics:
        return cls(
            feasible=False,
            minimum_transfers=None,
            minimum_initial_wait_minutes=None,
            minimum_journey_time_minutes=None,
            feasible_departure_count=0,
            earliest_arrival_seconds=None,
        )


@dataclass(frozen=True, slots=True)
class ODPathMetricRecord:
    """Path metrics for one OD/time cell before rule classification."""

    key: ODTimeKey
    metrics: ODPathMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.key, ODTimeKey):
            raise TypeError("key must be an ODTimeKey.")
        if not isinstance(self.metrics, ODPathMetrics):
            raise TypeError("metrics must be ODPathMetrics.")


@dataclass(frozen=True, slots=True)
class StructuralZeroRecord:
    key: ODTimeKey
    is_structural_zero: bool
    primary_reason: StructuralZeroReason | None
    triggered_rules: tuple[StructuralZeroReason, ...]
    metrics: ODPathMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.key, ODTimeKey):
            raise TypeError("key must be an ODTimeKey.")
        if not isinstance(self.is_structural_zero, bool):
            raise TypeError("is_structural_zero must be a Boolean.")
        if not isinstance(self.triggered_rules, tuple):
            raise TypeError("triggered_rules must be a tuple.")
        if any(
            not isinstance(reason, StructuralZeroReason)
            for reason in self.triggered_rules
        ):
            raise TypeError("triggered_rules must contain StructuralZeroReason values.")
        if self.primary_reason is not None and not isinstance(
            self.primary_reason, StructuralZeroReason
        ):
            raise TypeError("primary_reason must be a StructuralZeroReason or None.")
        if not isinstance(self.metrics, ODPathMetrics):
            raise TypeError("metrics must be ODPathMetrics.")
        if len(set(self.triggered_rules)) != len(self.triggered_rules):
            raise ValueError("triggered_rules must not contain duplicates.")
        if self.is_structural_zero:
            if not self.triggered_rules or self.primary_reason is None:
                raise ValueError(
                    "A structural zero must have triggered rules and a primary reason."
                )
            if self.primary_reason not in self.triggered_rules:
                raise ValueError("primary_reason must be one of triggered_rules.")
        elif self.primary_reason is not None or self.triggered_rules:
            raise ValueError("A retained cell cannot have structural-zero reasons.")


@dataclass(frozen=True, slots=True)
class StructuralZeroAnalysisResult:
    records: tuple[StructuralZeroRecord, ...]
    scenario_fingerprint: str
    graph_fingerprint: str
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple):
            raise TypeError("records must be a tuple.")
        if any(not isinstance(record, StructuralZeroRecord) for record in self.records):
            raise TypeError("records must contain StructuralZeroRecord values.")
        keys = tuple(record.key for record in self.records)
        if keys != tuple(sorted(keys)):
            raise ValueError("records must be sorted by OD/time key.")
        if len(keys) != len(set(keys)):
            raise ValueError("records must contain unique OD/time keys.")
        for name, value in (
            ("scenario_fingerprint", self.scenario_fingerprint),
            ("graph_fingerprint", self.graph_fingerprint),
            ("configuration_fingerprint", self.configuration_fingerprint),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")

    @property
    def num_cells(self) -> int:
        return len(self.records)

    @property
    def num_structural_zero(self) -> int:
        return sum(record.is_structural_zero for record in self.records)

    @property
    def num_retained(self) -> int:
        return self.num_cells - self.num_structural_zero

    @property
    def reason_counts(self) -> dict[str, int]:
        counts = {reason.value: 0 for reason in StructuralZeroReason}
        for record in self.records:
            if record.primary_reason is not None:
                counts[record.primary_reason.value] += 1
        return counts
