"""Direct sparse journey-event responses for strict APC measurements."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from public_transportation.measurement.schema import (
    MeasurementTable,
    MeasurementType,
)

from .artifacts import canonical_json
from .equivalence import ResponseEquivalence, build_response_equivalence
from .journey_choices import (
    JourneyChoiceResult,
    JourneyChoiceSet,
    JourneyEventKind,
)
from .progress import ReducedODProgress, ReducedODProgressEmitter
from .timetable_index import TimetableIndex


@dataclass(frozen=True, slots=True, order=True)
class ResponseCellKey:
    """Physical-stop journey cell represented by one response column."""

    origin_physical_stop_id: str
    destination_physical_stop_id: str
    origin_time_period_id: str

    def __post_init__(self) -> None:
        if not all(
            value
            for value in (
                self.origin_physical_stop_id,
                self.destination_physical_stop_id,
                self.origin_time_period_id,
            )
        ):
            raise ValueError("response cell identifiers must be non-empty.")

    @property
    def tuple(self) -> tuple[str, str, str]:
        return (
            self.origin_physical_stop_id,
            self.destination_physical_stop_id,
            self.origin_time_period_id,
        )


@dataclass(frozen=True, slots=True)
class ResolvedMeasurement:
    """One atomic observation resolved to exactly one scheduled trip event."""

    row_index: int
    method_id: str
    measurement_type: MeasurementType
    scenario_stop_id: str
    physical_stop_id: str
    seconds: int
    trip_id: str
    line_id: str
    observed_value: float

    @property
    def event_key(self) -> tuple[str, str, str, int]:
        return (
            self.trip_id,
            self.measurement_type.value,
            self.physical_stop_id,
            self.seconds,
        )


def _immutable_array(value: object, dtype: np.dtype[Any], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class MeasurementResponseArtifact:
    """Sparse free-cell response, fixed offset, observations, and provenance."""

    configuration_fingerprint: str
    timetable_fingerprint: str
    journey_choice_fingerprint: str
    measurement_fingerprint: str
    free_cell_keys: tuple[ResponseCellKey, ...]
    fixed_cell_keys: tuple[ResponseCellKey, ...]
    resolved_measurements: tuple[ResolvedMeasurement, ...]
    observed_values: np.ndarray
    measurement_index: np.ndarray
    free_cell_index: np.ndarray
    response_values: np.ndarray
    fixed_offset: np.ndarray
    equivalence: ResponseEquivalence

    def __post_init__(self) -> None:
        for value, name in (
            (self.configuration_fingerprint, "configuration_fingerprint"),
            (self.timetable_fingerprint, "timetable_fingerprint"),
            (self.journey_choice_fingerprint, "journey_choice_fingerprint"),
            (self.measurement_fingerprint, "measurement_fingerprint"),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty.")
        if self.free_cell_keys != tuple(sorted(self.free_cell_keys)):
            raise ValueError("free_cell_keys must be sorted.")
        if self.fixed_cell_keys != tuple(sorted(self.fixed_cell_keys)):
            raise ValueError("fixed_cell_keys must be sorted.")
        if set(self.free_cell_keys) & set(self.fixed_cell_keys):
            raise ValueError("free and fixed cell keys must be disjoint.")
        if tuple(item.row_index for item in self.resolved_measurements) != tuple(
            range(len(self.resolved_measurements))
        ):
            raise ValueError("resolved measurements must retain contiguous row order.")
        observed = _immutable_array(self.observed_values, np.dtype(np.float64), "observed_values")
        rows = _immutable_array(self.measurement_index, np.dtype(np.int64), "measurement_index")
        columns = _immutable_array(self.free_cell_index, np.dtype(np.int64), "free_cell_index")
        values = _immutable_array(self.response_values, np.dtype(np.float64), "response_values")
        offset = _immutable_array(self.fixed_offset, np.dtype(np.float64), "fixed_offset")
        measurements = len(self.resolved_measurements)
        if observed.size != measurements or offset.size != measurements:
            raise ValueError("observation and offset arrays must match measurement rows.")
        if not (rows.size == columns.size == values.size):
            raise ValueError("sparse response arrays must have equal length.")
        if rows.size and (np.any(rows < 0) or np.any(rows >= measurements)):
            raise ValueError("measurement_index is outside measurement rows.")
        if columns.size and (
            np.any(columns < 0) or np.any(columns >= len(self.free_cell_keys))
        ):
            raise ValueError("free_cell_index is outside free cells.")
        if not np.all(np.isfinite(observed)) or np.any(observed < 0.0):
            raise ValueError("observed values must be finite and non-negative.")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("response values must be finite and positive.")
        if not np.all(np.isfinite(offset)) or np.any(offset < 0.0):
            raise ValueError("fixed offset must be finite and non-negative.")
        if rows.size and np.any(
            np.lexsort((columns, rows)) != np.arange(rows.size)
        ):
            raise ValueError("sparse response entries must be sorted by row and cell.")
        if self.equivalence.number_of_cells != len(self.free_cell_keys):
            raise ValueError("equivalence must partition every free response cell.")
        object.__setattr__(self, "observed_values", observed)
        object.__setattr__(self, "measurement_index", rows)
        object.__setattr__(self, "free_cell_index", columns)
        object.__setattr__(self, "response_values", values)
        object.__setattr__(self, "fixed_offset", offset)

    @property
    def number_of_measurements(self) -> int:
        return len(self.resolved_measurements)

    @property
    def number_of_free_cells(self) -> int:
        return len(self.free_cell_keys)

    @property
    def nnz(self) -> int:
        return int(self.response_values.size)

    def predict(self, free_demand: np.ndarray) -> np.ndarray:
        demand = np.asarray(free_demand, dtype=np.float64)
        if demand.shape != (self.number_of_free_cells,):
            raise ValueError("free_demand has an invalid shape.")
        if not np.all(np.isfinite(demand)) or np.any(demand < 0.0):
            raise ValueError("free_demand must be finite and non-negative.")
        prediction = np.array(self.fixed_offset, copy=True)
        np.add.at(
            prediction,
            self.measurement_index,
            self.response_values * demand[self.free_cell_index],
        )
        return prediction

    @property
    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "arrays": {
                name: _array_descriptor(value)
                for name, value in (
                    ("fixed_offset", self.fixed_offset),
                    ("free_cell_index", self.free_cell_index),
                    ("measurement_index", self.measurement_index),
                    ("observed_values", self.observed_values),
                    ("response_values", self.response_values),
                    ("class_by_cell", self.equivalence.class_by_cell),
                    (
                        "representative_cell_indices",
                        self.equivalence.representative_cell_indices,
                    ),
                    ("member_indptr", self.equivalence.member_indptr),
                    ("member_cell_indices", self.equivalence.member_cell_indices),
                )
            },
            "configuration_fingerprint": self.configuration_fingerprint,
            "fixed_cell_keys": [list(key.tuple) for key in self.fixed_cell_keys],
            "free_cell_keys": [list(key.tuple) for key in self.free_cell_keys],
            "journey_choice_fingerprint": self.journey_choice_fingerprint,
            "measurement_fingerprint": self.measurement_fingerprint,
            "resolved_measurements": [
                [
                    item.row_index,
                    item.method_id,
                    item.measurement_type.value,
                    item.scenario_stop_id,
                    item.physical_stop_id,
                    item.seconds,
                    item.trip_id,
                    item.line_id,
                    item.observed_value,
                ]
                for item in self.resolved_measurements
            ],
            "timetable_fingerprint": self.timetable_fingerprint,
        }

    @property
    def fingerprint_payload_json(self) -> str:
        return canonical_json(self.fingerprint_payload)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.fingerprint_payload_json.encode("utf-8")).hexdigest()

    @property
    def retained_bytes(self) -> int:
        arrays = (
            self.observed_values,
            self.measurement_index,
            self.free_cell_index,
            self.response_values,
            self.fixed_offset,
            self.equivalence.class_by_cell,
            self.equivalence.representative_cell_indices,
            self.equivalence.member_indptr,
            self.equivalence.member_cell_indices,
        )
        return sum(int(array.nbytes) for array in arrays)


def _array_descriptor(array: np.ndarray) -> dict[str, object]:
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {
        "digest": digest.hexdigest(),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def measurement_table_fingerprint(table: MeasurementTable) -> str:
    payload = [
        [
            record.method_id,
            record.measurement_type.value,
            str(record.stop_id),
            int(record.time.seconds_from_midnight),
            None if record.trip_id is None else str(record.trip_id),
            None if record.line_id is None else str(record.line_id),
            float(record.value),
        ]
        for record in table.records
    ]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _resolve_measurements(
    timetable: TimetableIndex, table: MeasurementTable
) -> tuple[ResolvedMeasurement, ...]:
    stop_index = {stop_id: index for index, stop_id in enumerate(timetable.stop_ids)}
    trip_index = {trip_id: index for index, trip_id in enumerate(timetable.trip_ids)}
    line_index = {line_id: index for index, line_id in enumerate(timetable.line_ids)}
    trip_lines = timetable.array("trip_line_index")
    event_stops = timetable.array("stop_time_stop_index")
    event_physical = timetable.array("stop_time_physical_stop_index")
    event_trips = timetable.array("stop_time_trip_index")
    arrivals = timetable.array("arrival_seconds")
    departures = timetable.array("departure_seconds")
    resolved: list[ResolvedMeasurement] = []
    for row_index, record in enumerate(table.records):
        if record.measurement_type not in {
            MeasurementType.BOARDING,
            MeasurementType.ALIGHTING,
        }:
            raise ValueError(
                f"measurement row {row_index} uses unsupported type "
                f"{record.measurement_type.value!r}."
            )
        value = float(record.value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"measurement row {row_index} value must be finite and non-negative."
            )
        stop_id = str(record.stop_id)
        if stop_id not in stop_index:
            raise ValueError(f"measurement row {row_index} has unknown stop {stop_id!r}.")
        if record.trip_id is not None:
            requested_trip = str(record.trip_id)
            if requested_trip not in trip_index:
                raise ValueError(
                    f"measurement row {row_index} has unknown trip {requested_trip!r}."
                )
            candidate_trips = [trip_index[requested_trip]]
        else:
            assert record.line_id is not None
            requested_line = str(record.line_id)
            if requested_line not in line_index:
                raise ValueError(
                    f"measurement row {row_index} has unknown line {requested_line!r}."
                )
            candidate_trips = np.flatnonzero(
                trip_lines == line_index[requested_line]
            ).tolist()
        if record.line_id is not None:
            requested_line = str(record.line_id)
            if requested_line not in line_index or any(
                int(trip_lines[index]) != line_index[requested_line]
                for index in candidate_trips
            ):
                raise ValueError(
                    f"measurement row {row_index} has inconsistent trip/line identity."
                )
        seconds = int(record.time.seconds_from_midnight)
        event_times = (
            departures
            if record.measurement_type is MeasurementType.BOARDING
            else arrivals
        )
        matches = np.flatnonzero(
            (event_stops == stop_index[stop_id])
            & np.isin(event_trips, np.asarray(candidate_trips, dtype=np.int64))
            & (event_times == seconds)
        )
        if matches.size == 0:
            raise ValueError(
                f"measurement row {row_index} does not match a timetable event."
            )
        if matches.size != 1:
            raise ValueError(
                f"measurement row {row_index} is ambiguous across {matches.size} events."
            )
        event = int(matches[0])
        selected_trip = int(event_trips[event])
        selected_line = int(trip_lines[selected_trip])
        resolved.append(
            ResolvedMeasurement(
                row_index=row_index,
                method_id=str(record.method_id),
                measurement_type=record.measurement_type,
                scenario_stop_id=stop_id,
                physical_stop_id=timetable.physical_stop_ids[int(event_physical[event])],
                seconds=seconds,
                trip_id=timetable.trip_ids[selected_trip],
                line_id=timetable.line_ids[selected_line],
                observed_value=value,
            )
        )
    return tuple(resolved)


def resolve_measurements(
    timetable: TimetableIndex, table: MeasurementTable
) -> tuple[ResolvedMeasurement, ...]:
    """Resolve canonical measurements to exactly one timetable event each.

    This public wrapper keeps case-study adapters from depending on the
    response-builder's private implementation while preserving its strict
    zero/one-match contract.
    """
    return _resolve_measurements(timetable, table)


def _cell_key(choice_set: JourneyChoiceSet) -> ResponseCellKey:
    return ResponseCellKey(
        choice_set.origin_physical_stop_id,
        choice_set.destination_physical_stop_id,
        choice_set.origin_time_period_id,
    )


def build_measurement_response(
    *,
    timetable: TimetableIndex,
    journey_choices: JourneyChoiceResult,
    measurements: MeasurementTable,
    configuration_fingerprint: str,
    fixed_demand: Mapping[ResponseCellKey, float] | None = None,
    progress: ReducedODProgress | None = None,
) -> MeasurementResponseArtifact:
    """Build ``B`` and a fixed offset directly from journey event atoms."""
    if not configuration_fingerprint:
        raise ValueError("configuration_fingerprint must be non-empty.")
    resolved = _resolve_measurements(timetable, measurements)
    measurement_lookup: dict[tuple[str, str, str, int], list[int]] = {}
    for item in resolved:
        measurement_lookup.setdefault(item.event_key, []).append(item.row_index)
    choice_by_key: dict[ResponseCellKey, JourneyChoiceSet] = {}
    index_progress = ReducedODProgressEmitter(
        progress,
        phase="measurement_response_index",
        total=len(journey_choices.choice_sets),
    )
    index_progress.start()
    for choice_position, choice_set in enumerate(
        journey_choices.choice_sets, start=1
    ):
        key = _cell_key(choice_set)
        if key in choice_by_key:
            raise ValueError(f"duplicate journey choice cell {key.tuple!r}.")
        choice_by_key[key] = choice_set
        index_progress.update(choice_position, current_unit="|".join(key.tuple))
    supplied_fixed = dict(fixed_demand or {})
    for key, value in supplied_fixed.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"fixed demand for {key.tuple!r} must be non-negative.")
    unknown_positive = sorted(
        key
        for key, value in supplied_fixed.items()
        if key not in choice_by_key and value > 0.0
    )
    if unknown_positive:
        raise ValueError(
            "positive fixed demand contains cells without journey choices: "
            f"{[item.tuple for item in unknown_positive]}."
        )
    fixed_keys = tuple(sorted(supplied_fixed))
    free_keys = tuple(sorted(set(choice_by_key) - set(supplied_fixed)))
    free_index = {key: index for index, key in enumerate(free_keys)}
    coefficient_by_cell: dict[ResponseCellKey, dict[int, float]] = {}
    ordered_choice_keys = sorted(choice_by_key)
    coefficient_progress = ReducedODProgressEmitter(
        progress,
        phase="measurement_response_coefficients",
        total=len(ordered_choice_keys),
    )
    coefficient_progress.start()
    for key_position, key in enumerate(ordered_choice_keys, start=1):
        coefficients: dict[int, float] = {}
        choice_set = choice_by_key[key]
        for alternative, share in zip(
            choice_set.alternatives, choice_set.initial_shares, strict=True
        ):
            share *= choice_set.served_time_fraction
            for event in alternative.events:
                leg = alternative.transit_legs[event.leg_index]
                measurement_type = (
                    MeasurementType.BOARDING
                    if event.event_kind
                    in {
                        JourneyEventKind.FIRST_BOARDING,
                        JourneyEventKind.TRANSFER_BOARDING,
                    }
                    else MeasurementType.ALIGHTING
                )
                event_key = (
                    leg.trip_id,
                    measurement_type.value,
                    event.physical_stop_id,
                    event.seconds,
                )
                for row_index in measurement_lookup.get(event_key, ()):
                    coefficients[row_index] = (
                        coefficients.get(row_index, 0.0) + share
                    )
        coefficient_by_cell[key] = coefficients
        coefficient_progress.update(
            key_position, current_unit="|".join(key.tuple)
        )
    sparse: list[tuple[int, int, float]] = []
    sparse_progress = ReducedODProgressEmitter(
        progress,
        phase="measurement_response_sparse_rows",
        total=len(free_keys),
    )
    sparse_progress.start()
    for free_position, key in enumerate(free_keys, start=1):
        for row, value in coefficient_by_cell[key].items():
            if value > 0.0:
                sparse.append((row, free_index[key], value))
        sparse_progress.update(free_position, current_unit="|".join(key.tuple))
    sparse.sort()
    measurement_index = np.asarray([item[0] for item in sparse], dtype=np.int64)
    free_cell_index = np.asarray([item[1] for item in sparse], dtype=np.int64)
    response_values = np.asarray([item[2] for item in sparse], dtype=np.float64)
    fixed_offset = np.zeros(len(resolved), dtype=np.float64)
    fixed_progress = ReducedODProgressEmitter(
        progress,
        phase="measurement_response_fixed_offset",
        total=len(fixed_keys),
    )
    fixed_progress.start()
    for fixed_position, key in enumerate(fixed_keys, start=1):
        demand = float(supplied_fixed[key])
        if demand == 0.0:
            fixed_progress.update(
                fixed_position, current_unit="|".join(key.tuple)
            )
            continue
        for row, coefficient in coefficient_by_cell.get(key, {}).items():
            fixed_offset[row] += demand * coefficient
        fixed_progress.update(fixed_position, current_unit="|".join(key.tuple))
    equivalence = build_response_equivalence(
        number_of_cells=len(free_keys),
        measurement_index=measurement_index,
        cell_index=free_cell_index,
        values=response_values,
    )
    return MeasurementResponseArtifact(
        configuration_fingerprint=configuration_fingerprint,
        timetable_fingerprint=timetable.fingerprint,
        journey_choice_fingerprint=journey_choices.fingerprint,
        measurement_fingerprint=measurement_table_fingerprint(measurements),
        free_cell_keys=free_keys,
        fixed_cell_keys=fixed_keys,
        resolved_measurements=resolved,
        observed_values=np.asarray(
            [item.observed_value for item in resolved], dtype=np.float64
        ),
        measurement_index=measurement_index,
        free_cell_index=free_cell_index,
        response_values=response_values,
        fixed_offset=fixed_offset,
        equivalence=equivalence,
    )
