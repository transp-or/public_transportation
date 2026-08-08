"""Canonical indexing and backend-neutral assignment-operator contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import jax
import numpy as np

from .compact_od_assignment_layout import build_compact_od_assignment_layout
from .od_parameter_layout import ODParameterLayout

DemandCellRole = Literal["free", "fixed_positive", "fixed_zero"]
MeasurementEvent = Literal["boarding", "alighting"]
ASSIGNMENT_CONTRACT_SCHEMA_VERSION = 1


def _nonempty(value: object, *, name: str) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be nonempty.")
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def fixed_routing_route_choice_fingerprint(theta: float) -> str:
    """Identify the fixed route-choice dispersion used by assignment."""
    value = float(theta)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("fixed-routing theta must be positive and finite.")
    return _fingerprint({"fixed_routing_theta": value})


@dataclass(frozen=True, slots=True)
class CanonicalTimeInterval:
    """Stable half-open temporal interval used by demand and measurements."""

    interval_id: str
    start_seconds: int
    end_seconds: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "interval_id", _nonempty(self.interval_id, name="interval_id")
        )
        if isinstance(self.start_seconds, bool) or isinstance(self.end_seconds, bool):
            raise TypeError("interval boundaries must be integer seconds.")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("time intervals must be nonempty and start at or after zero.")


@dataclass(frozen=True, slots=True)
class CanonicalODTimeCell:
    """One physical OD--departure-interval cell in authoritative order."""

    full_index: int
    origin_id: str
    destination_id: str
    departure_interval_id: str
    role: DemandCellRole
    operator_column: int | None
    fixed_value: float | None = None

    def __post_init__(self) -> None:
        if self.full_index < 0:
            raise ValueError("full_index must be nonnegative.")
        for name in ("origin_id", "destination_id", "departure_interval_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        if self.role not in {"free", "fixed_positive", "fixed_zero"}:
            raise ValueError("unsupported demand-cell role.")
        if self.role == "free":
            if self.operator_column is None or self.operator_column < 0:
                raise ValueError("free demand cells require an operator column.")
            if self.fixed_value is not None:
                raise ValueError("free demand cells cannot have a fixed value.")
            return
        if self.operator_column is not None:
            raise ValueError("fixed demand cells cannot have an operator column.")
        if self.fixed_value is None or not np.isfinite(self.fixed_value):
            raise ValueError("fixed demand cells require a finite fixed value.")
        if self.role == "fixed_zero" and self.fixed_value != 0.0:
            raise ValueError("fixed-zero cells must have value zero.")
        if self.role == "fixed_positive" and self.fixed_value <= 0.0:
            raise ValueError("fixed-positive cells must have a positive value.")

    @property
    def physical_key(self) -> tuple[str, str, str]:
        return self.origin_id, self.destination_id, self.departure_interval_id


@dataclass(frozen=True, slots=True)
class CanonicalMeasurement:
    """One boarding or alighting row in authoritative operator order."""

    row_index: int
    measurement_id: str
    event: MeasurementEvent
    location_id: str
    interval_id: str

    def __post_init__(self) -> None:
        if self.row_index < 0:
            raise ValueError("row_index must be nonnegative.")
        for name in ("measurement_id", "location_id", "interval_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        if self.event not in {"boarding", "alighting"}:
            raise ValueError("measurement event must be boarding or alighting.")


@dataclass(frozen=True, slots=True)
class CanonicalAssignmentIndex:
    """Canonical physical tables and reduced operator-coordinate binding.

    ``artifact_fingerprint`` deliberately excludes free/fixed roles and fixed
    values.  Those fields describe the current estimation binding, whereas the
    physical assignment map is identified only by OD-time keys, measurements,
    and temporal intervals.
    """

    time_intervals: tuple[CanonicalTimeInterval, ...]
    demand_cells: tuple[CanonicalODTimeCell, ...]
    measurements: tuple[CanonicalMeasurement, ...]
    source_od_layout_fingerprint: str | None = None
    source_compact_layout_fingerprint: str | None = None
    schema_version: int = ASSIGNMENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ASSIGNMENT_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported canonical assignment-index schema.")
        interval_ids = tuple(item.interval_id for item in self.time_intervals)
        if len(set(interval_ids)) != len(interval_ids):
            raise ValueError("time-interval identifiers must be unique.")
        previous_end = None
        for item in self.time_intervals:
            if previous_end is not None and item.start_seconds < previous_end:
                raise ValueError("canonical time intervals must not overlap.")
            previous_end = item.end_seconds
        interval_id_set = set(interval_ids)

        full_indices = tuple(item.full_index for item in self.demand_cells)
        if full_indices != tuple(range(len(self.demand_cells))):
            raise ValueError("demand cells must use contiguous canonical full indices.")
        physical_keys = tuple(item.physical_key for item in self.demand_cells)
        if len(set(physical_keys)) != len(physical_keys):
            raise ValueError("physical OD-time keys must be unique.")
        if any(
            item.departure_interval_id not in interval_id_set
            for item in self.demand_cells
        ):
            raise ValueError("every demand cell must reference a canonical interval.")
        columns = tuple(
            item.operator_column
            for item in self.demand_cells
            if item.operator_column is not None
        )
        if columns != tuple(range(len(columns))):
            raise ValueError("free demand cells must use contiguous operator columns.")

        rows = tuple(item.row_index for item in self.measurements)
        if rows != tuple(range(len(self.measurements))):
            raise ValueError("measurements must use contiguous canonical row indices.")
        measurement_ids = tuple(item.measurement_id for item in self.measurements)
        if len(set(measurement_ids)) != len(measurement_ids):
            raise ValueError("measurement identifiers must be unique.")
        if any(item.interval_id not in interval_id_set for item in self.measurements):
            raise ValueError("every measurement must reference a canonical interval.")

    @property
    def number_of_physical_demand_cells(self) -> int:
        return len(self.demand_cells)

    @property
    def number_of_demand_cells(self) -> int:
        return sum(item.operator_column is not None for item in self.demand_cells)

    @property
    def number_of_measurements(self) -> int:
        return len(self.measurements)

    @property
    def artifact_fingerprint(self) -> str:
        """Fingerprint only physical coordinates reusable across demand models."""
        return _fingerprint(
            {
                "schema_version": self.schema_version,
                "time_intervals": [
                    [item.interval_id, item.start_seconds, item.end_seconds]
                    for item in self.time_intervals
                ],
                "demand_cells": [list(item.physical_key) for item in self.demand_cells],
                "measurements": [
                    [
                        item.measurement_id,
                        item.event,
                        item.location_id,
                        item.interval_id,
                    ]
                    for item in self.measurements
                ],
            }
        )

    @property
    def binding_fingerprint(self) -> str:
        """Fingerprint the current free/fixed reduction separately."""
        return _fingerprint(
            {
                "artifact_fingerprint": self.artifact_fingerprint,
                "source_od_layout_fingerprint": self.source_od_layout_fingerprint,
                "source_compact_layout_fingerprint": (
                    self.source_compact_layout_fingerprint
                ),
                "binding": [
                    [item.role, item.operator_column, item.fixed_value]
                    for item in self.demand_cells
                ],
            }
        )


def build_canonical_assignment_index(
    *,
    parameter_layout: ODParameterLayout,
    time_intervals: tuple[CanonicalTimeInterval, ...],
    measurements: tuple[CanonicalMeasurement, ...],
) -> CanonicalAssignmentIndex:
    """Build canonical tables from the existing authoritative OD layout."""
    fixed_values = dict(
        zip(
            parameter_layout.fixed_od_indices,
            parameter_layout.fixed_od_values,
            strict=True,
        )
    )
    free_columns = {
        full_index: column
        for column, full_index in enumerate(parameter_layout.free_od_indices)
    }
    fixed_zero = set(parameter_layout.fixed_zero_indices)
    cells = []
    for full_index, key in enumerate(parameter_layout.od_keys):
        if full_index in free_columns:
            role: DemandCellRole = "free"
            column = free_columns[full_index]
            fixed_value = None
        elif full_index in fixed_zero:
            role = "fixed_zero"
            column = None
            fixed_value = 0.0
        else:
            role = "fixed_positive"
            column = None
            fixed_value = fixed_values[full_index]
        origin_id, destination_id, interval_id = key
        cells.append(
            CanonicalODTimeCell(
                full_index=full_index,
                origin_id=origin_id,
                destination_id=destination_id,
                departure_interval_id=interval_id,
                role=role,
                operator_column=column,
                fixed_value=fixed_value,
            )
        )
    return CanonicalAssignmentIndex(
        time_intervals=tuple(time_intervals),
        demand_cells=tuple(cells),
        measurements=tuple(measurements),
        source_od_layout_fingerprint=parameter_layout.fingerprint,
        source_compact_layout_fingerprint=build_compact_od_assignment_layout(
            parameter_layout=parameter_layout
        ).fingerprint,
    )


@dataclass(frozen=True, slots=True)
class AssignmentArtifactIdentity:
    """Supply-dependent identity of one reusable assignment artifact."""

    canonical_index_fingerprint: str
    network_fingerprint: str
    timetable_fingerprint: str
    temporal_discretization_fingerprint: str
    route_choice_fingerprint: str
    departure_choice_fingerprint: str
    feasibility_fingerprint: str
    measurement_mapping_fingerprint: str
    coefficient_policy_fingerprint: str
    numeric_dtype: str
    schema_version: int = ASSIGNMENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ASSIGNMENT_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported assignment-artifact identity schema.")
        for name in (
            "canonical_index_fingerprint",
            "network_fingerprint",
            "timetable_fingerprint",
            "temporal_discretization_fingerprint",
            "route_choice_fingerprint",
            "departure_choice_fingerprint",
            "feasibility_fingerprint",
            "measurement_mapping_fingerprint",
            "coefficient_policy_fingerprint",
            "numeric_dtype",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            }
        )


class AssignmentCompatibilityError(ValueError):
    """Raised when canonical coordinates or artifact dependencies differ."""


def assert_assignment_artifact_compatible(
    *, expected: AssignmentArtifactIdentity, actual: AssignmentArtifactIdentity
) -> None:
    """Fail explicitly and diagnostically on any artifact-identity mismatch."""
    mismatches = [
        name
        for name in expected.__dataclass_fields__
        if getattr(expected, name) != getattr(actual, name)
    ]
    if mismatches:
        details = ", ".join(
            f"{name}: expected={getattr(expected, name)!r}, "
            f"got={getattr(actual, name)!r}"
            for name in mismatches
        )
        raise AssignmentCompatibilityError(
            f"assignment artifact is incompatible ({details})."
        )


def build_fixed_routing_artifact_identity(
    *,
    operator: object,
    canonical_index: CanonicalAssignmentIndex,
    temporal_discretization_fingerprint: str,
    departure_choice_fingerprint: str,
    feasibility_fingerprint: str,
    coefficient_policy_fingerprint: str,
) -> AssignmentArtifactIdentity:
    """Derive a canonical artifact identity from an existing fixed-routing operator."""
    return AssignmentArtifactIdentity(
        canonical_index_fingerprint=canonical_index.artifact_fingerprint,
        network_fingerprint=str(getattr(operator, "graph_fingerprint")),
        timetable_fingerprint=str(getattr(operator, "assignment_fingerprint")),
        temporal_discretization_fingerprint=temporal_discretization_fingerprint,
        route_choice_fingerprint=fixed_routing_route_choice_fingerprint(
            float(getattr(operator, "theta"))
        ),
        departure_choice_fingerprint=departure_choice_fingerprint,
        feasibility_fingerprint=feasibility_fingerprint,
        measurement_mapping_fingerprint=str(
            getattr(operator, "mapping_fingerprint")
        ),
        coefficient_policy_fingerprint=coefficient_policy_fingerprint,
        numeric_dtype=np.dtype(getattr(operator, "dtype")).name,
    )


@runtime_checkable
class AssignmentOperator(Protocol):
    """Backend-neutral canonical forward and adjoint assignment contract."""

    @property
    def number_of_demand_cells(self) -> int: ...

    @property
    def number_of_measurements(self) -> int: ...

    @property
    def canonical_index_fingerprint(self) -> str: ...

    @property
    def artifact_fingerprint(self) -> str: ...

    def matvec(self, demand: object) -> jax.Array: ...

    def rmatvec(self, residual: object) -> jax.Array: ...


@dataclass(frozen=True, slots=True)
class FixedRoutingAssignmentOperatorAdapter:
    """Expose an existing fixed-routing operator through the canonical contract."""

    operator: object
    canonical_index: CanonicalAssignmentIndex
    identity: AssignmentArtifactIdentity

    def __post_init__(self) -> None:
        if (
            self.identity.canonical_index_fingerprint
            != self.canonical_index.artifact_fingerprint
        ):
            raise AssignmentCompatibilityError(
                "artifact identity and canonical index have different fingerprints."
            )
        demand_cells = int(getattr(self.operator, "num_free_od"))
        measurements = int(getattr(self.operator, "num_measurements"))
        if demand_cells != self.canonical_index.number_of_demand_cells:
            raise AssignmentCompatibilityError(
                "fixed-routing demand dimension does not match canonical columns: "
                f"operator={demand_cells}, canonical={self.canonical_index.number_of_demand_cells}."
            )
        if measurements != self.canonical_index.number_of_measurements:
            raise AssignmentCompatibilityError(
                "fixed-routing measurement dimension does not match canonical rows: "
                f"operator={measurements}, canonical={self.canonical_index.number_of_measurements}."
            )
        expected_od_layout = self.canonical_index.source_od_layout_fingerprint
        operator_od_layout = getattr(self.operator, "od_layout_fingerprint", None)
        if (
            expected_od_layout is not None
            and operator_od_layout is not None
            and str(operator_od_layout) != expected_od_layout
        ):
            raise AssignmentCompatibilityError(
                "fixed-routing OD-layout fingerprint is incompatible."
            )
        expected_compact_layout = (
            self.canonical_index.source_compact_layout_fingerprint
        )
        operator_compact_layout = getattr(
            self.operator, "compact_layout_fingerprint", None
        )
        if (
            expected_compact_layout is not None
            and operator_compact_layout is not None
            and str(operator_compact_layout) != expected_compact_layout
        ):
            raise AssignmentCompatibilityError(
                "fixed-routing compact-layout fingerprint is incompatible."
            )
        mapping = str(getattr(self.operator, "mapping_fingerprint"))
        if mapping != self.identity.measurement_mapping_fingerprint:
            raise AssignmentCompatibilityError(
                "fixed-routing measurement mapping fingerprint is incompatible."
            )
        graph = str(getattr(self.operator, "graph_fingerprint"))
        if graph != self.identity.network_fingerprint:
            raise AssignmentCompatibilityError(
                "fixed-routing graph fingerprint is incompatible."
            )
        assignment = str(getattr(self.operator, "assignment_fingerprint"))
        if assignment != self.identity.timetable_fingerprint:
            raise AssignmentCompatibilityError(
                "fixed-routing assignment fingerprint is incompatible."
            )
        route_choice = fixed_routing_route_choice_fingerprint(
            float(getattr(self.operator, "theta"))
        )
        if route_choice != self.identity.route_choice_fingerprint:
            raise AssignmentCompatibilityError(
                "fixed-routing route-choice fingerprint is incompatible."
            )
        dtype = np.dtype(getattr(self.operator, "dtype")).name
        if dtype != np.dtype(self.identity.numeric_dtype).name:
            raise AssignmentCompatibilityError(
                f"fixed-routing dtype is incompatible: operator={dtype}, "
                f"identity={self.identity.numeric_dtype}."
            )

    @property
    def number_of_demand_cells(self) -> int:
        return self.canonical_index.number_of_demand_cells

    @property
    def number_of_measurements(self) -> int:
        return self.canonical_index.number_of_measurements

    @property
    def canonical_index_fingerprint(self) -> str:
        return self.canonical_index.artifact_fingerprint

    @property
    def artifact_fingerprint(self) -> str:
        return self.identity.fingerprint

    @property
    def fixed_measurement_offset(self) -> object:
        return getattr(self.operator, "fixed_measurement_offset")

    def matvec(self, demand: object) -> jax.Array:
        """Delegate the unchanged fixed-routing forward product."""
        return getattr(self.operator, "jax_matvec")(demand)

    def rmatvec(self, residual: object) -> jax.Array:
        """Delegate the unchanged fixed-routing adjoint product."""
        return getattr(self.operator, "jax_rmatvec")(residual)
