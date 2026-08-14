"""Fail-fast observed-measurement support checks for expensive construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Literal, Mapping

import numpy as np

from public_transportation.measurement.mapping import MappingInfo

from .assignment_contract import CanonicalAssignmentIndex

POSITIVE_BOARDING_PREFLIGHT_SCHEMA_VERSION = 1

PositiveBoardingPreflightStage = Literal[
    "canonical_origin_support",
    "routing_support",
    "realized_operator_support",
]
PositiveBoardingFailureCause = Literal[
    "origin_interval_absent_from_demand",
    "origin_interval_all_fixed_zero",
    "mapped_boarding_access_link_has_no_active_origin",
    "no_retained_route_to_boarding_event",
]


@dataclass(frozen=True, slots=True)
class PositiveBoardingSupportIssue:
    """One observed boarding row that the current assignment cannot predict."""

    row_index: int
    measurement_id: str
    observed_value: float
    location_id: str
    interval_id: str
    cause: PositiveBoardingFailureCause
    explanation: str
    remediation: str
    physical_origin_cells: int
    free_origin_cells: int
    fixed_positive_origin_cells: int
    fixed_zero_origin_cells: int
    fixed_zero_reason_counts: tuple[tuple[str, int], ...] = ()
    method_id: str | None = None
    time_hms: str | None = None
    trip_id: str | None = None
    line_id: str | None = None


@dataclass(frozen=True, slots=True)
class PositiveBoardingSupportReport:
    """Machine-readable result produced before expensive numerical mapping."""

    stage: PositiveBoardingPreflightStage
    number_of_measurements: int
    positive_boarding_rows: int
    supported_positive_boarding_rows: int
    unsupported_positive_boarding_rows: int
    unsupported_positive_boarding_mass: float
    issues: tuple[PositiveBoardingSupportIssue, ...]
    schema_version: int = POSITIVE_BOARDING_PREFLIGHT_SCHEMA_VERSION

    @property
    def safe(self) -> bool:
        return self.unsupported_positive_boarding_rows == 0

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-ready diagnostic payload."""
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "safe": self.safe,
            "number_of_measurements": self.number_of_measurements,
            "positive_boarding_rows": self.positive_boarding_rows,
            "supported_positive_boarding_rows": (
                self.supported_positive_boarding_rows
            ),
            "unsupported_positive_boarding_rows": (
                self.unsupported_positive_boarding_rows
            ),
            "unsupported_positive_boarding_mass": (
                self.unsupported_positive_boarding_mass
            ),
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class PositiveBoardingPreflightContext:
    """Observed data and optional provenance used by sharded construction."""

    canonical_index: CanonicalAssignmentIndex
    observations: object
    report_path: Path | None = None
    mapping_info: MappingInfo | None = None
    fixed_zero_reasons_by_full_index: Mapping[int, str] | None = None
    canonical_supported_measurement_rows: object | None = None


class UnsupportedPositiveBoardingError(ValueError):
    """Raised before construction when positive boarding rows have no support."""

    def __init__(
        self,
        report: PositiveBoardingSupportReport,
        *,
        report_path: Path | None = None,
    ) -> None:
        preview = "; ".join(
            f"row {issue.row_index} ({issue.measurement_id!r}, "
            f"value={issue.observed_value:g}): {issue.cause}"
            for issue in report.issues[:5]
        )
        suffix = "" if len(report.issues) <= 5 else "; ..."
        location = (
            ""
            if report_path is None
            else f" Full machine-readable report: {report_path}."
        )
        super().__init__(
            f"{report.unsupported_positive_boarding_rows} positive boarding "
            "measurement row(s), with observed mass "
            f"{report.unsupported_positive_boarding_mass:g}, have no model support "
            f"at {report.stage}. Expensive linear-map construction was stopped. "
            f"{preview}{suffix}.{location}"
        )
        self.report = report
        self.report_path = report_path
        self.details = report.to_payload()


@dataclass(frozen=True, slots=True)
class _OriginCounts:
    physical: int
    free: int
    fixed_positive: int
    fixed_zero: int
    fixed_zero_indices: tuple[int, ...]

    @property
    def active(self) -> int:
        return self.free + self.fixed_positive


def _validated_observations(
    canonical_index: CanonicalAssignmentIndex, observations: object
) -> np.ndarray:
    try:
        values = np.asarray(observations, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("observations must contain real numeric values.") from error
    expected = (canonical_index.number_of_measurements,)
    if values.shape != expected:
        raise ValueError(f"observations must have shape {expected}, got {values.shape}.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("observations must be finite and nonnegative.")
    return values


def _validated_mapping_entries(
    *,
    canonical_index: CanonicalAssignmentIndex,
    observations: np.ndarray,
    mapping_info: MappingInfo | None,
) -> dict[int, object]:
    if mapping_info is None:
        return {}
    if len(mapping_info.entries) != canonical_index.number_of_measurements:
        raise ValueError("mapping-info rows must match the canonical measurements.")
    entries: dict[int, object] = {}
    for position, (entry, measurement) in enumerate(
        zip(mapping_info.entries, canonical_index.measurements, strict=True)
    ):
        if entry.row_index != position:
            raise ValueError("mapping-info rows must use contiguous canonical indices.")
        if entry.measurement_type != measurement.event:
            raise ValueError("mapping-info event types differ from the canonical index.")
        if entry.stop_id != measurement.location_id:
            raise ValueError("mapping-info stop identifiers differ from the canonical index.")
        if not math.isclose(
            float(entry.observed_value),
            float(observations[position]),
            rel_tol=1.0e-6,
            abs_tol=1.0e-8,
        ):
            raise ValueError("mapping-info observed values differ from observations.")
        entries[position] = entry
    return entries


def _origin_counts(
    canonical_index: CanonicalAssignmentIndex,
) -> dict[tuple[str, str], _OriginCounts]:
    mutable: dict[tuple[str, str], dict[str, object]] = {}
    for cell in canonical_index.demand_cells:
        key = (cell.origin_id, cell.departure_interval_id)
        counts = mutable.setdefault(
            key,
            {
                "physical": 0,
                "free": 0,
                "fixed_positive": 0,
                "fixed_zero": 0,
                "fixed_zero_indices": [],
            },
        )
        counts["physical"] = int(counts["physical"]) + 1
        counts[cell.role] = int(counts[cell.role]) + 1
        if cell.role == "fixed_zero":
            indices = counts["fixed_zero_indices"]
            assert isinstance(indices, list)
            indices.append(cell.full_index)
    return {
        key: _OriginCounts(
            physical=int(value["physical"]),
            free=int(value["free"]),
            fixed_positive=int(value["fixed_positive"]),
            fixed_zero=int(value["fixed_zero"]),
            fixed_zero_indices=tuple(value["fixed_zero_indices"]),
        )
        for key, value in mutable.items()
    }


def _reason_counts(
    counts: _OriginCounts,
    reasons: Mapping[int, str] | None,
) -> tuple[tuple[str, int], ...]:
    if reasons is None:
        return ()
    labels = []
    for index in counts.fixed_zero_indices:
        label = str(reasons.get(index, "reason_not_provided")).strip()
        labels.append(label or "reason_not_provided")
    return tuple(sorted(Counter(labels).items()))


def _issue(
    *,
    row_index: int,
    canonical_index: CanonicalAssignmentIndex,
    observations: np.ndarray,
    counts: _OriginCounts,
    cause: PositiveBoardingFailureCause,
    mapping_entries: dict[int, object],
    reasons: Mapping[int, str] | None,
) -> PositiveBoardingSupportIssue:
    measurement = canonical_index.measurements[row_index]
    entry = mapping_entries.get(row_index)
    reason_counts = _reason_counts(counts, reasons)
    if cause == "origin_interval_absent_from_demand":
        explanation = (
            "No canonical OD cell starts at this boarding location in this "
            "departure interval. The strict boarding mapper counts access links "
            "into the scheduled departure event, so no modeled demand can enter "
            "that link. Transfer boardings are not represented by this access-link "
            "measurement contract."
        )
        remediation = (
            "Correct the OD location/time domain or use a scientifically validated "
            "measurement mapping that represents the intended boarding event."
        )
    elif cause == "origin_interval_all_fixed_zero":
        explanation = (
            f"All {counts.fixed_zero} canonical OD cell(s) starting at this "
            "boarding location in this departure interval are fixed at zero and "
            "removed from compact assignment. There is neither free demand nor a "
            "positive fixed offset that can enter the mapped access link. Transfer "
            "boardings are not represented by this access-link measurement contract."
        )
        if reason_counts:
            explanation += " Fixed-zero causes: " + ", ".join(
                f"{name}={count}" for name, count in reason_counts
            ) + "."
        else:
            explanation += (
                " The canonical assignment index does not retain the upstream "
                "structural-zero reason; supply fixed_zero_reasons_by_full_index "
                "to include those preprocessing causes in this report."
            )
        remediation = (
            "Review the fixed-demand and structural-zero policy for this origin "
            "and interval (especially timing/initial-wait rules), or exclude the "
            "row under an explicit support-aware observation policy."
        )
    elif cause == "mapped_boarding_access_link_has_no_active_origin":
        explanation = (
            f"This stop/interval has {counts.active} active origin OD cell(s), "
            "but none of the mapped boarding access links starts at an active "
            "assignment origin node. The row cannot receive first-boarding flow. "
            "This commonly indicates a departure-time/initial-wait exclusion; a "
            "transfer boarding is also outside the strict access-link contract."
        )
        remediation = (
            "Inspect the mapped event's access-link tails and the structural-zero "
            "reason for the corresponding OD origins before rebuilding."
        )
    else:
        explanation = (
            f"This stop/interval has {counts.active} active origin OD cell(s), but "
            "exact fixed-routing support contains no contribution to the mapped "
            "boarding event. The event is therefore unreachable under the retained "
            "route/timing constraints, or it is a transfer boarding that the strict "
            "access-link mapper does not represent."
        )
        remediation = (
            "Inspect route/timing feasibility and first-boarding versus transfer "
            "semantics before changing the observation set or rebuilding."
        )
    return PositiveBoardingSupportIssue(
        row_index=row_index,
        measurement_id=measurement.measurement_id,
        observed_value=float(observations[row_index]),
        location_id=measurement.location_id,
        interval_id=measurement.interval_id,
        cause=cause,
        explanation=explanation,
        remediation=remediation,
        physical_origin_cells=counts.physical,
        free_origin_cells=counts.free,
        fixed_positive_origin_cells=counts.fixed_positive,
        fixed_zero_origin_cells=counts.fixed_zero,
        fixed_zero_reason_counts=reason_counts,
        method_id=None if entry is None else str(getattr(entry, "method_id")),
        time_hms=None if entry is None else str(getattr(entry, "time_hms")),
        trip_id=None if entry is None else getattr(entry, "trip_id"),
        line_id=None if entry is None else getattr(entry, "line_id"),
    )


def audit_positive_boarding_support(
    *,
    canonical_index: CanonicalAssignmentIndex,
    observations: object,
    supported_measurement_rows: object | None = None,
    stage: PositiveBoardingPreflightStage = "canonical_origin_support",
    mapping_info: MappingInfo | None = None,
    fixed_zero_reasons_by_full_index: Mapping[int, str] | None = None,
) -> PositiveBoardingSupportReport:
    """Audit positive boarding support without constructing numerical columns.

    The canonical stage is an intentionally cheap necessary-condition check.
    Routing and realized-operator stages receive exact supported row indices and
    therefore detect every unsupported positive boarding row.
    """
    values = _validated_observations(canonical_index, observations)
    entries = _validated_mapping_entries(
        canonical_index=canonical_index,
        observations=values,
        mapping_info=mapping_info,
    )
    if supported_measurement_rows is None:
        if stage != "canonical_origin_support":
            raise ValueError("exact support audit requires supported_measurement_rows.")
        supported_rows: set[int] | None = None
    else:
        array = np.asarray(supported_measurement_rows, dtype=np.int64)
        if array.ndim != 1 or (
            array.size
            and (
                np.any(array < 0)
                or np.any(array >= canonical_index.number_of_measurements)
            )
        ):
            raise ValueError("supported measurement rows must be one-dimensional and in range.")
        supported_rows = set(int(value) for value in array)

    counts_by_key = _origin_counts(canonical_index)
    empty = _OriginCounts(0, 0, 0, 0, ())
    issues = []
    positive_rows = 0
    for measurement in canonical_index.measurements:
        row = measurement.row_index
        if measurement.event != "boarding" or values[row] <= 0.0:
            continue
        positive_rows += 1
        counts = counts_by_key.get(
            (measurement.location_id, measurement.interval_id), empty
        )
        cause: PositiveBoardingFailureCause | None = None
        if supported_rows is not None and row in supported_rows:
            cause = None
        elif counts.physical == 0:
            cause = "origin_interval_absent_from_demand"
        elif counts.active == 0:
            cause = "origin_interval_all_fixed_zero"
        elif stage == "canonical_origin_support" and (
            supported_rows is not None and row not in supported_rows
        ):
            cause = "mapped_boarding_access_link_has_no_active_origin"
        elif supported_rows is not None and row not in supported_rows:
            cause = "no_retained_route_to_boarding_event"
        if cause is not None:
            issues.append(
                _issue(
                    row_index=row,
                    canonical_index=canonical_index,
                    observations=values,
                    counts=counts,
                    cause=cause,
                    mapping_entries=entries,
                    reasons=fixed_zero_reasons_by_full_index,
                )
            )
    issue_tuple = tuple(issues)
    return PositiveBoardingSupportReport(
        stage=stage,
        number_of_measurements=canonical_index.number_of_measurements,
        positive_boarding_rows=positive_rows,
        supported_positive_boarding_rows=positive_rows - len(issue_tuple),
        unsupported_positive_boarding_rows=len(issue_tuple),
        unsupported_positive_boarding_mass=float(
            sum(issue.observed_value for issue in issue_tuple)
        ),
        issues=issue_tuple,
    )


def boarding_access_supported_measurement_rows(
    *,
    active_origin_nodes: object,
    graph_link_tails: object,
    measurement_index: object,
    link_index: object,
) -> np.ndarray:
    """Return rows whose mapped link starts at an active assignment origin.

    This graph-level necessary condition is cheap and more precise than matching
    stop/interval labels: distinct departure intervals may use distinct centroid
    nodes even when they share the same physical stop identifier.
    """
    origins = np.asarray(active_origin_nodes, dtype=np.int64)
    tails = np.asarray(graph_link_tails, dtype=np.int64)
    measurements = np.asarray(measurement_index, dtype=np.int64)
    links = np.asarray(link_index, dtype=np.int64)
    if origins.ndim != 1 or tails.ndim != 1:
        raise ValueError("origin nodes and graph link tails must be one-dimensional.")
    if measurements.ndim != 1 or links.shape != measurements.shape:
        raise ValueError("measurement and link mapping arrays must be aligned.")
    if links.size and (np.any(links < 0) or np.any(links >= tails.size)):
        raise ValueError("measurement mapping contains an out-of-range link.")
    if np.any(origins < 0) or np.any(tails < 0):
        raise ValueError("assignment node indices must be nonnegative.")
    maximum_node = max(
        int(origins.max()) if origins.size else -1,
        int(tails.max()) if tails.size else -1,
    )
    active = np.zeros(maximum_node + 1, dtype=bool)
    active[origins] = True
    eligible = active[tails[links]]
    return np.unique(measurements[eligible]).astype(np.int64, copy=False)


def write_positive_boarding_support_report(
    report: PositiveBoardingSupportReport, path: str | Path
) -> Path:
    """Atomically persist the complete deterministic preflight report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report.to_payload(), stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def enforce_positive_boarding_support(
    report: PositiveBoardingSupportReport,
    *,
    report_path: str | Path | None = None,
) -> Path | None:
    """Persist a report and raise before expensive work when it is unsafe."""
    destination = (
        None
        if report_path is None
        else write_positive_boarding_support_report(report, report_path)
    )
    if not report.safe:
        raise UnsupportedPositiveBoardingError(report, report_path=destination)
    return destination
