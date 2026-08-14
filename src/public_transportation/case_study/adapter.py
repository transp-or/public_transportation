"""Canonical-file adapter used by the generic case-study runner."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable, Mapping
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
from public_transportation.preprocessing.od_universe import (
    CandidateODTimeCell,
    CandidateODUniverse,
    ODTimeExpansion,
    PriorGenerationResult,
    expand_candidate_od_time_cells,
    generate_candidate_od_pairs,
    generate_prior_demand,
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
    if text.isdigit():
        return TimeOfDay(seconds_from_midnight=int(text))
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


def _load_physical_stop_mapping(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    mapping: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"stop_id", "physical_stop_id"}.issubset(reader.fieldnames):
            raise ValueError(
                "physical-stop mapping must contain stop_id,physical_stop_id columns."
            )
        for row_number, row in enumerate(reader, start=2):
            stop_id = _required_text(row.get("stop_id"), location=f"physical mapping row {row_number} stop_id")
            physical_id = _required_text(
                row.get("physical_stop_id"),
                location=f"physical mapping row {row_number} physical_stop_id",
            )
            if stop_id in mapping:
                raise ValueError(f"physical-stop mapping contains duplicate stop_id {stop_id!r}.")
            mapping[stop_id] = physical_id
    return mapping


def _load_time_periods(path: Path) -> tuple[JourneyTimePeriod, ...]:
    periods: list[JourneyTimePeriod] = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"bin_id", "start_s", "end_s"}.issubset(reader.fieldnames):
            raise ValueError("time-bin file must contain bin_id,start_s,end_s columns.")
        for row_number, row in enumerate(reader, start=2):
            period_id = _required_text(row.get("bin_id"), location=f"time-bin row {row_number} bin_id")
            start = _parse_time(row.get("start_s"), location=f"time-bin row {row_number} start_s")
            end = _parse_time(row.get("end_s"), location=f"time-bin row {row_number} end_s")
            if end.seconds_from_midnight <= start.seconds_from_midnight:
                raise ValueError(f"time-bin row {row_number} must have end_s > start_s.")
            periods.append(JourneyTimePeriod(period_id, start.seconds_from_midnight, end.seconds_from_midnight))
    ordered = tuple(periods)
    if ordered != tuple(
        sorted(ordered, key=lambda item: (item.start_seconds, item.end_seconds, item.period_id))
    ):
        raise ValueError("time-bin file must list periods in sorted start/end/id order.")
    if any(
        left.end_seconds > right.start_seconds
        for left, right in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError("time-bin file contains overlapping periods.")
    return ordered


def _load_component_values(path: Path, *, component: str) -> dict[tuple[str, str], float]:
    """Load explicit origin/destination-period component values."""
    values: dict[tuple[str, str], float] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{component} input has no header: {path}")
        first = "origin_stop_id" if component == "production" else "destination_stop_id"
        value_column = "value" if "value" in reader.fieldnames else f"{component}_value"
        required = {first, "time_bin_id", value_column}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{component} input is missing required columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            key = (
                _required_text(row.get(first), location=f"{component} row {row_number} identifier"),
                _required_text(row.get("time_bin_id"), location=f"{component} row {row_number} time_bin_id"),
            )
            if key in values:
                raise ValueError(f"{component} input contains duplicate key {key!r}.")
            try:
                value = float(row.get(value_column, ""))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{component} row {row_number} value must be numeric.") from error
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{component} row {row_number} value must be finite and positive.")
            values[key] = value
    return values


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
    candidate_od_universe: CandidateODUniverse | None = None
    od_time_expansion: ODTimeExpansion | None = None
    prior_demand: Mapping[CandidateODTimeCell, float] | None = None
    prior_generation: PriorGenerationResult | None = None
    input_semantics: str = "legacy_time_dependent_demand"


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
    od_universe: dict[str, object] | None = None
    od_time_expansion: dict[str, object] | None = None
    prior_generation: dict[str, object] | None = None
    input_semantics: str = "legacy_time_dependent_demand"

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
        for name, path in (("scenario_directory", paths.scenario_directory), ("measurements", paths.measurements)):
            if not path.exists():
                raise FileNotFoundError(f"configured {name} does not exist: {path}")
        if self.config.od_universe.source == "legacy_time_dependent_demand":
            if paths.candidate_demand is None or not paths.candidate_demand.exists():
                raise FileNotFoundError(f"configured candidate_demand does not exist: {paths.candidate_demand}")
        elif self.config.od_universe.source == "file":
            pair_file = self.config.od_universe.pair_file or paths.od_pairs
            if pair_file is None or not pair_file.exists():
                raise FileNotFoundError(f"configured OD-pair file does not exist: {pair_file}")
        if paths.fixed_demand is not None and not paths.fixed_demand.exists():
            raise FileNotFoundError(f"configured fixed_demand does not exist: {paths.fixed_demand}")
        legacy = self.config.od_universe.source == "legacy_time_dependent_demand"
        scenario = Scenario.from_folder(
            paths.scenario_directory,
            strict=True,
            demand_file=paths.candidate_demand if legacy else None,
            allow_missing_demand=not legacy,
        )
        measurements = load_canonical_measurements(paths.measurements, config=self.config)
        fixed: dict[ResponseCellKey, float] = {}
        if paths.fixed_demand is not None:
            fixed_domain = read_fixed_demand_csv(paths.fixed_demand, scenario=scenario)
            fixed = {
                ResponseCellKey(origin, destination, period): float(value)
                for (origin, destination, period), value in fixed_domain.as_dict().items()
            }
        generated_bins = self.config.paths.results_directory / "generated_inputs/time_bins.csv"
        periods = (
            _load_time_periods(generated_bins)
            if generated_bins.is_file()
            else tuple(
                JourneyTimePeriod(str(item.bin_id), int(item.start.seconds_from_midnight), int(item.end.seconds_from_midnight))
                for item in scenario.time_bins
            )
        )
        physical_mapping = _load_physical_stop_mapping(
            self.config.reduced_od_config.stops.physical_stop_mapping_path
        )
        candidate_universe: CandidateODUniverse | None = None
        expansion: ODTimeExpansion | None = None
        prior_values: Mapping[CandidateODTimeCell, float] | None = None
        prior_generation: PriorGenerationResult | None = None
        if legacy:
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
            input_semantics = "legacy_time_dependent_demand"
        else:
            candidate_universe = generate_candidate_od_pairs(
                scenario,
                source=self.config.od_universe.source,  # type: ignore[arg-type]
                level=self.config.od_universe.level,  # type: ignore[arg-type]
                include_same_stop=self.config.od_universe.include_same_stop,
                active_service_only=self.config.od_universe.active_service_only,
                connectivity_policy=self.config.od_universe.connectivity_policy,  # type: ignore[arg-type]
                od_pairs_path=self.config.od_universe.pair_file or paths.od_pairs,
                physical_stop_mapping=physical_mapping,
            )
            reduced = self.config.reduced_od_config
            expansion = expand_candidate_od_time_cells(
                candidate_universe,
                periods,
                scenario=scenario,
                maximum_transfers=reduced.journeys.maximum_transfers,
                maximum_initial_wait_seconds=(
                    reduced.journeys.maximum_waiting_seconds
                ),
                maximum_journey_seconds=reduced.journeys.maximum_journey_seconds,
                maximum_waiting_seconds=reduced.journeys.maximum_waiting_seconds,
                timetable_policy="required",
            )
            if expansion.cell_count == 0:
                raise ValueError(
                    "OD-time expansion retained no cells; review pair-level and "
                    "timetable-feasibility rules before estimation."
                )
            generated_fixed = {
                ResponseCellKey(
                    item.origin_stop_id,
                    item.destination_stop_id,
                    item.time_bin_id,
                ): 0.0
                for item in expansion.exclusions
            }
            conflicts = sorted(
                key.tuple
                for key, value in generated_fixed.items()
                if key in fixed and fixed[key] > 0.0
            )
            if conflicts:
                raise ValueError(
                    "positive fixed demand conflicts with an OD-time structural zero: "
                    f"{conflicts}"
                )
            fixed.update(generated_fixed)
            prior_values_result = generate_prior_demand(
                expansion,
                source=self.config.prior_demand.source,
                value=float(self.config.prior_demand.value or 1.0),
                semantics=self.config.prior_demand.semantics,
                prior_file=self.config.prior_demand.input_file,
            )
            prior_generation = prior_values_result
            prior_values = prior_values_result.values
            origins = sorted({cell.origin_stop_id for cell in expansion.cells})
            production, attraction = self._explicit_components(expansion)
            input_semantics = "independent_od_universe"
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
            physical_stop_mapping=physical_mapping,
            footpaths=(),
            candidate_od_universe=candidate_universe,
            od_time_expansion=expansion,
            prior_demand=prior_values,
            prior_generation=prior_generation,
            input_semantics=input_semantics,
        )
        if self.custom_hook is not None:
            transformed = self.custom_hook(self.config, data)
            if not isinstance(transformed, GenericCaseData):
                raise TypeError("custom case-study hook must return GenericCaseData.")
            data = transformed
        return data

    def _explicit_components(
        self, expansion: ODTimeExpansion
    ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
        """Build declared production/attractiveness baselines without demand sums."""
        productions_spec = self.config.model.get("production")
        attraction_spec = self.config.model.get("destination_attractiveness")
        if not isinstance(productions_spec, Mapping) or not isinstance(attraction_spec, Mapping):
            raise ValueError(
                "independent OD workflow requires explicit [production] and "
                "[destination_attractiveness] model specifications."
            )

        origin_groups = sorted({(cell.origin_stop_id, cell.time_bin_id) for cell in expansion.cells})
        destination_groups = sorted({(cell.destination_stop_id, cell.time_bin_id) for cell in expansion.cells})

        def values_for(spec: Mapping[str, object], groups: list[tuple[str, str]], component: str) -> dict[tuple[str, str], float]:
            mode = str(spec["mode"])
            if mode == "provided":
                file_value = spec.get("input_file")
                if not file_value:
                    raise ValueError(f"{component} provided mode requires input_file.")
                values = _load_component_values(Path(str(file_value)), component=component)
                missing = sorted(set(groups) - set(values))
                extra = sorted(set(values) - set(groups))
                if missing or extra:
                    raise ValueError(f"{component} values do not match retained groups; missing={missing}, extra={extra}.")
                return {key: values[key] for key in groups}
            baseline = spec.get("baseline")
            if isinstance(baseline, (int, float)) and not isinstance(baseline, bool):
                value = float(baseline)
            elif str(baseline) == "ones":
                value = 1.0
            else:
                raise ValueError(f"{component} baseline must be numeric or 'ones'.")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{component} baseline must be finite and positive.")
            return {key: value for key in groups}

        return (
            values_for(productions_spec, origin_groups, "production"),
            values_for(attraction_spec, destination_groups, "destination_attractiveness"),
        )

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
        pair_count = (
            data.candidate_od_universe.pair_count
            if data.candidate_od_universe is not None
            else len({(str(item.origin_stop_id), str(item.dest_stop_id)) for item in data.scenario.demand.records})
        )
        configured_pairs = self.config.time_discretization.num_od_pairs
        if configured_pairs is not None and configured_pairs != pair_count and data.input_semantics == "legacy_time_dependent_demand":
            raise ValueError(f"configured num_od_pairs={configured_pairs} does not match candidate demand count {pair_count}.")
        counts: dict[str, int] = {}
        for item in data.measurements.records:
            counts[item.measurement_type.value] = counts.get(item.measurement_type.value, 0) + 1
        checksums = {
            "measurements": _sha256(self.config.paths.measurements),
            **({"candidate_demand": _sha256(self.config.paths.candidate_demand)} if self.config.paths.candidate_demand else {}),
            **({"fixed_demand": _sha256(self.config.paths.fixed_demand)} if self.config.paths.fixed_demand else {}),
        }
        pair_file = self.config.od_universe.pair_file
        if pair_file is not None:
            checksums["od_pairs"] = _sha256(pair_file)
        if self.config.prior_demand.input_file is not None:
            checksums["prior_demand"] = _sha256(self.config.prior_demand.input_file)
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
            None if data.candidate_od_universe is None else data.candidate_od_universe.audit,
            None if data.od_time_expansion is None else data.od_time_expansion.audit,
            None if data.prior_generation is None else data.prior_generation.audit,
            data.input_semantics,
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
            candidate_od_pairs=(
                None
                if data.candidate_od_universe is None
                else tuple(pair.tuple for pair in data.candidate_od_universe.pairs)
            ),
            prior_demand=data.prior_demand,
        )
