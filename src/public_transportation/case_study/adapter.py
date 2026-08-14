"""Canonical-file adapter used by the generic case-study runner."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

from public_transportation.domain import Scenario, TimeOfDay, read_fixed_demand_csv
from public_transportation.measurement import (
    MeasurementRecord,
    MeasurementTable,
    MeasurementType,
)
from public_transportation.preprocessing.reduced_od import (
    DepartureTimeSamplingConfig,
    Footpath,
    JourneyTimePeriod,
    ResponseCellKey,
    preflight_reduced_od_time_periods,
    prepare_reduced_od_timetable,
    resolve_measurements,
)

from .config import CaseStudyConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_time(value: Any, *, location: str) -> TimeOfDay:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{location} is missing; timestamp semantics are event_time.")
    if isinstance(value, bool):
        raise ValueError(f"{location} must be HH:MM[:SS] or seconds.")
    if isinstance(value, (int, float)):
        if not float(value).is_integer() or value < 0:
            raise ValueError(f"{location} must be a non-negative integer number of seconds.")
        return TimeOfDay(seconds_from_midnight=int(value))
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"{location} must be HH:MM[:SS].")
    try:
        hour, minute = int(parts[0]), int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as error:
        raise ValueError(f"{location} must be HH:MM[:SS].") from error
    if hour < 0 or minute not in range(60) or second not in range(60):
        raise ValueError(f"{location} contains an invalid timestamp.")
    return TimeOfDay(seconds_from_midnight=hour * 3600 + minute * 60 + second)


def _required_text(value: Any, *, location: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{location} must be a non-empty value.")
    return str(value).strip()


def load_canonical_measurements(
    path: str | Path,
    *,
    config: CaseStudyConfig,
) -> MeasurementTable:
    """Load a mapped measurement table without cleaning or dropping rows."""
    source = Path(path).resolve()
    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"measurement file has no header: {source}")
        headers = [str(item).strip() for item in reader.fieldnames]
        if len(headers) != len(set(headers)):
            raise ValueError("measurement file contains duplicate column names.")
        mapping = config.observations
        required = {
            mapping.measurement_type_column,
            mapping.stop_id_column,
            mapping.timestamp_column,
            mapping.value_column,
        }
        if mapping.method_id_column is not None:
            required.add(mapping.method_id_column)
        for optional in (mapping.trip_id_column, mapping.line_id_column):
            if optional is not None:
                required.add(optional)
        missing = sorted(required - set(headers))
        if missing:
            raise ValueError(f"measurement file is missing configured columns: {missing}")
        unknown = sorted(set(headers) - required - {"method_id"})
        if unknown:
            raise ValueError(f"measurement file contains unknown columns: {unknown}")
        records: list[MeasurementRecord] = []
        method_column = mapping.method_id_column
        if method_column is not None and method_column not in headers:
            raise ValueError(f"measurement file is missing configured method-id column: {method_column!r}")
        if method_column is None and "method_id" in headers:
            method_column = "method_id"
        for row_number, raw in enumerate(reader, start=2):
            row = {str(key).strip(): value for key, value in raw.items() if key is not None}
            if raw.get(None) and any(str(item).strip() for item in raw[None]):
                raise ValueError(f"measurement row {row_number} contains extra fields.")
            type_text = _required_text(row.get(mapping.measurement_type_column), location=f"measurement row {row_number} measurement type")
            try:
                measurement_type = MeasurementType(type_text)
            except ValueError as error:
                raise ValueError(f"measurement row {row_number} has unknown measurement type {type_text!r}.") from error
            if measurement_type is MeasurementType.LOAD:
                raise ValueError("load measurements are not supported by reduced-OD boarding/alighting response.")
            stop_id = _required_text(row.get(mapping.stop_id_column), location=f"measurement row {row_number} stop_id")
            time = _parse_time(row.get(mapping.timestamp_column), location=f"measurement row {row_number} timestamp")
            try:
                value = float(row.get(mapping.value_column, ""))
            except (TypeError, ValueError) as error:
                raise ValueError(f"measurement row {row_number} value must be numeric.") from error
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"measurement row {row_number} value must be finite and non-negative.")
            trip_id = (
                None
                if mapping.trip_id_column is None
                else (str(row.get(mapping.trip_id_column, "")).strip() or None)
            )
            line_id = (
                None
                if mapping.line_id_column is None
                else (str(row.get(mapping.line_id_column, "")).strip() or None)
            )
            if trip_id is None and line_id is None:
                raise ValueError(f"measurement row {row_number} must identify a trip or line.")
            method_id = str(row.get(method_column, "case-study")).strip() if method_column else "case-study"
            if not method_id:
                raise ValueError(f"measurement row {row_number} method_id must be non-empty.")
            records.append(MeasurementRecord(method_id, measurement_type, stop_id, time, value, trip_id, line_id))
    return MeasurementTable.from_records(records)


@dataclass(frozen=True, slots=True)
class GenericCaseData:
    scenario: Scenario
    measurements: MeasurementTable
    fixed_demand: dict[ResponseCellKey, float]
    time_periods: tuple[JourneyTimePeriod, ...]
    production_inputs: dict[tuple[str, str], float]
    destination_attractiveness: dict[tuple[str, str], float]
    departure_seconds_by_origin: dict[str, tuple[int, ...]]
    departure_sampling: DepartureTimeSamplingConfig
    physical_stop_mapping: dict[str, str] | None
    footpaths: tuple[Footpath, ...]


@dataclass(frozen=True, slots=True)
class GenericCaseAudit:
    case_name: str
    configuration_fingerprint: str
    source_checksums: dict[str, str]
    scenario_stop_count: int
    demand_cell_count: int
    candidate_od_pair_count: int
    measurement_count: int
    measurement_type_counts: dict[str, int]
    resolved_measurement_count: int
    time_period_count: int
    period_preflight: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


GenericCaseHook = Callable[[CaseStudyConfig, GenericCaseData], GenericCaseData]


class GenericCaseAdapter:
    """Load one canonical case and construct explicit reduced-OD inputs."""

    def __init__(self, config: CaseStudyConfig, *, custom_hook: GenericCaseHook | None = None):
        self.config = config
        self.custom_hook = custom_hook
        self._data: GenericCaseData | None = None

    @property
    def data(self) -> GenericCaseData:
        if self._data is None:
            self._data = self._load()
        return self._data

    def _load(self) -> GenericCaseData:
        paths = self.config.paths
        for name, path in (
            ("scenario_directory", paths.scenario_directory),
            ("measurements", paths.measurements),
            ("candidate_demand", paths.candidate_demand),
        ):
            if not path.exists():
                raise FileNotFoundError(f"configured {name} does not exist: {path}")
        if paths.fixed_demand is not None and not paths.fixed_demand.exists():
            raise FileNotFoundError(f"configured fixed_demand does not exist: {paths.fixed_demand}")
        scenario = Scenario.from_folder(paths.scenario_directory, strict=True, demand_file=paths.candidate_demand)
        measurements = load_canonical_measurements(paths.measurements, config=self.config)
        fixed: dict[ResponseCellKey, float] = {}
        if paths.fixed_demand is not None:
            fixed_domain = read_fixed_demand_csv(paths.fixed_demand, scenario=scenario)
            fixed = {
                ResponseCellKey(origin, destination, period): float(value)
                for (origin, destination, period), value in fixed_domain.as_dict().items()
            }
        periods = tuple(
            JourneyTimePeriod(str(item.bin_id), int(item.start.seconds_from_midnight), int(item.end.seconds_from_midnight))
            for item in scenario.time_bins
        )
        demand_rows = tuple(scenario.demand.records)
        origins = sorted({str(item.origin_stop_id) for item in demand_rows})
        production: dict[tuple[str, str], float] = {}
        attraction: dict[tuple[str, str], float] = {}
        for record in demand_rows:
            origin_key = (str(record.origin_stop_id), str(record.time_bin_id))
            destination_key = (str(record.dest_stop_id), str(record.time_bin_id))
            production[origin_key] = production.get(origin_key, 0.0) + float(record.flow)
            attraction[destination_key] = attraction.get(destination_key, 0.0) + float(record.flow)
        if any(value <= 0.0 for value in attraction.values()):
            raise ValueError("candidate demand gives a non-positive destination attractiveness.")
        sampling = DepartureTimeSamplingConfig(
            strategy=self.config.sampling.strategy,
            samples_per_period=self.config.sampling.samples_per_period,
            time_step_seconds=self.config.sampling.time_step_seconds,
        )
        departures = {
            origin: tuple(period.start_seconds + (period.end_seconds - period.start_seconds) // 2 for period in periods)
            for origin in origins
        }
        data = GenericCaseData(
            scenario=scenario,
            measurements=measurements,
            fixed_demand=fixed,
            time_periods=periods,
            production_inputs=production,
            destination_attractiveness=attraction,
            departure_seconds_by_origin=departures,
            departure_sampling=sampling,
            physical_stop_mapping=None,
            footpaths=(),
        )
        if self.custom_hook is not None:
            transformed = self.custom_hook(self.config, data)
            if not isinstance(transformed, GenericCaseData):
                raise TypeError("custom case-study hook must return GenericCaseData.")
            data = transformed
        return data

    def _resolved_timetable(self):
        timetable = prepare_reduced_od_timetable(
            self.data.scenario,
            configuration_fingerprint=self.config.reduced_od_config.fingerprint,
            physical_stop_mapping=self.data.physical_stop_mapping,
            mapping_policy=self.config.reduced_od_config.stops.mapping_policy,
        )
        return timetable

    def audit(self) -> GenericCaseAudit:
        data = self.data
        timetable = self._resolved_timetable()
        resolved = resolve_measurements(timetable, data.measurements)
        event_seconds = tuple(
            int(value)
            for stop_time in data.scenario.timetable.stop_times  # type: ignore[union-attr]
            for value in (stop_time.arrival.seconds_from_midnight, stop_time.departure.seconds_from_midnight)
            if self.config.time_discretization.horizon_start_s <= int(value) < self.config.time_discretization.horizon_end_s
        )
        period_report = preflight_reduced_od_time_periods(
            data.time_periods,
            relevant_event_seconds=event_seconds,
            sampling_config=data.departure_sampling,
            require_contiguous=True,
        )
        if not period_report.valid:
            raise ValueError(f"time-period preflight failed: {period_report.to_dict()}")
        pair_count = len({(str(item.origin_stop_id), str(item.dest_stop_id)) for item in data.scenario.demand.records})
        configured_pairs = self.config.time_discretization.num_od_pairs
        if configured_pairs is not None and configured_pairs != pair_count:
            raise ValueError(f"configured num_od_pairs={configured_pairs} does not match candidate demand count {pair_count}.")
        counts: dict[str, int] = {}
        for item in data.measurements.records:
            counts[item.measurement_type.value] = counts.get(item.measurement_type.value, 0) + 1
        checksums = {
            "measurements": _sha256(self.config.paths.measurements),
            "candidate_demand": _sha256(self.config.paths.candidate_demand),
            **({"fixed_demand": _sha256(self.config.paths.fixed_demand)} if self.config.paths.fixed_demand else {}),
        }
        for source in sorted(path for path in self.config.paths.scenario_directory.rglob("*") if path.is_file()):
            checksums[f"scenario/{source.relative_to(self.config.paths.scenario_directory)}"] = _sha256(source)
        for configuration_path in self.config.package_config_paths:
            if configuration_path.is_file():
                checksums[f"configuration/{configuration_path.name}"] = _sha256(configuration_path)
        return GenericCaseAudit(
            self.config.case_name,
            self.config.fingerprint,
            checksums,
            len(data.scenario.stops),
            len(data.scenario.demand.records),
            pair_count,
            len(data.measurements.records),
            counts,
            len(resolved),
            len(data.time_periods),
            period_report.to_dict(),
        )

    def preparation_inputs(self):
        from public_transportation.inference.reduced_od import ReducedODPreparationInputs

        data = self.data
        return ReducedODPreparationInputs(
            departure_seconds_by_origin=data.departure_seconds_by_origin,
            production_inputs=data.production_inputs,
            destination_attractiveness=data.destination_attractiveness,
            physical_stop_mapping=data.physical_stop_mapping,
            footpaths=data.footpaths,
            time_periods=data.time_periods,
            fixed_demand=data.fixed_demand,
            departure_time_sampling=data.departure_sampling,
        )
