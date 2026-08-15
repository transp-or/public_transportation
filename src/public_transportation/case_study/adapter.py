"""Canonical-file adapter used by the generic case-study runner."""

from __future__ import annotations

import csv
import hashlib
import math
import json
from dataclasses import asdict, dataclass, field
from importlib.metadata import distribution
from pathlib import Path
import re
from collections.abc import Callable, Mapping
from types import MappingProxyType
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
    CandidateODPair,
    CandidateODUniverse,
    ODTimeExclusion,
    ODTimeExpansion,
    ODUniverseExclusion,
    PriorGenerationResult,
    TimetableFeasibilityIndex,
    expand_candidate_od_time_cells,
    generate_candidate_od_pairs,
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
class GenericCaseBaseData:
    """Data loaded by every stage without OD--time feasibility work."""

    scenario: Scenario
    measurements: MeasurementTable
    fixed_demand: dict[ResponseCellKey, float]
    time_periods: tuple[JourneyTimePeriod, ...]
    departure_sampling: DepartureTimeSamplingConfig
    physical_stop_mapping: dict[str, str] | None
    footpaths: tuple[Footpath, ...]
    production_inputs: dict[tuple[str, str], float] = field(default_factory=dict)
    destination_attractiveness: dict[tuple[str, str], float] = field(default_factory=dict)
    departure_seconds_by_origin: dict[str, tuple[int, ...]] = field(default_factory=dict)
    input_semantics: str = "legacy_time_dependent_demand"


@dataclass(frozen=True, slots=True)
class GenericCaseData(GenericCaseBaseData):
    """Fully materialized case data used by reduced-OD preparation."""

    candidate_od_universe: CandidateODUniverse | None = None
    od_time_expansion: ODTimeExpansion | None = None
    prior_demand: Mapping[CandidateODTimeCell, float] | None = None
    prior_generation: PriorGenerationResult | None = None


@dataclass(frozen=True, slots=True)
class GenericCaseAudit:
    case_name: str
    configuration_fingerprint: str
    source_checksums: dict[str, str]
    scenario_stop_count: int
    demand_cell_count: int
    candidate_od_pair_count: int | None
    measurement_count: int
    measurement_type_counts: dict[str, int]
    resolved_measurement_count: int
    time_period_count: int
    period_preflight: dict[str, object]
    od_universe_status: str = "not_run"
    od_time_expansion_status: str = "not_run"
    timetable_feasibility_status: str = "not_run"
    complexity: dict[str, object] = field(default_factory=dict)
    od_universe: dict[str, object] | None = None
    od_time_expansion: dict[str, object] | None = None
    prior_generation: dict[str, object] | None = None
    input_semantics: str = "legacy_time_dependent_demand"
    source_time_bins: dict[str, object] = field(default_factory=dict)
    approved_time_bins: dict[str, object] = field(default_factory=dict)
    stale_artifacts: list[dict[str, str]] = field(default_factory=list)
    active_candidate_od_time_count: int | None = None
    legacy_demand_file_present: bool = False
    legacy_demand_row_count: int = 0
    legacy_demand_used: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


GenericCaseHook = Callable[[CaseStudyConfig, GenericCaseData], GenericCaseData]


class GenericCaseAdapter:
    """Load one canonical case and construct explicit reduced-OD inputs."""

    def __init__(self, config: CaseStudyConfig, *, custom_hook: GenericCaseHook | None = None):
        self.config = config
        self.custom_hook = custom_hook
        self._base_data: GenericCaseBaseData | None = None
        self._universe: CandidateODUniverse | None = None
        self._expansion: ODTimeExpansion | None = None
        self._data: GenericCaseData | None = None

    def _current_package_revision(self) -> str:
        """Return the package identity used by generated-artifact contracts."""
        lockfile = self.config.source_file.parent.parent / "uv.lock"
        lock_text = lockfile.read_text(encoding="utf-8") if lockfile.is_file() else ""
        marker = 'name = "public-transportation"'
        if marker in lock_text:
            block = lock_text[lock_text.index(marker) : lock_text.index(marker) + 4096]
            match = re.search(r'\brev = "([0-9a-fA-F]+)"', block)
            if match:
                return match.group(1)
        try:
            return str(distribution("public_transportation").version)
        except Exception:  # pragma: no cover - only malformed environments
            return "unknown"

    def _current_source_checksums(self) -> dict[str, str]:
        """Checksum all configured source and contract files.

        This mirrors the runner's provenance payload so that a persisted
        artifact can be checked without running any expensive preparation.
        """
        paths = self.config.paths
        candidates = [
            ("measurements", paths.measurements),
            ("candidate_demand", paths.candidate_demand),
            ("od_pairs", paths.od_pairs),
            ("prior_demand", paths.prior_demand),
            ("fixed_demand", paths.fixed_demand),
            ("production_inputs", paths.production_inputs),
            ("destination_attractiveness", paths.destination_attractiveness),
            ("od_universe_pair_file", self.config.od_universe.pair_file),
            ("prior_input_file", self.config.prior_demand.input_file),
        ]
        checksums = {
            name: _sha256(path)
            for name, path in candidates
            if path is not None and path.is_file()
        }
        if paths.scenario_directory.is_dir():
            for path in sorted(item for item in paths.scenario_directory.rglob("*") if item.is_file()):
                checksums[f"scenario/{path.relative_to(paths.scenario_directory)}"] = _sha256(path)
        for path in self.config.package_config_paths:
            if path.is_file():
                checksums[f"configuration/{path.name}"] = _sha256(path)
        return checksums

    @property
    def data(self) -> GenericCaseBaseData:
        """Return base data only; this property never runs timetable search."""
        return self.load_base_data()

    def load_base_data(self) -> GenericCaseBaseData:
        """Load scenario, measurements, periods, mappings, and fixed inputs."""
        if self._base_data is not None:
            return self._base_data
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
        # Base loading always uses the scenario's source periods.  A generated
        # time_bins.csv is a reviewed downstream artifact and is admitted only
        # through _approved_time_periods(), after its manifest is validated.
        periods = tuple(
            JourneyTimePeriod(str(item.bin_id), int(item.start.seconds_from_midnight), int(item.end.seconds_from_midnight))
            for item in scenario.time_bins
        )
        physical_mapping = _load_physical_stop_mapping(
            self.config.reduced_od_config.stops.physical_stop_mapping_path
        )
        input_semantics = "legacy_time_dependent_demand" if legacy else "independent_od_universe"
        production: dict[tuple[str, str], float] = {}
        attraction: dict[tuple[str, str], float] = {}
        origins: list[str] = []
        if legacy:
            demand_rows = tuple(scenario.demand.records)
            origins = sorted({str(item.origin_stop_id) for item in demand_rows})
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
            origin: tuple(
                period.start_seconds + (period.end_seconds - period.start_seconds) // 2
                for period in periods
            )
            for origin in origins
        }
        self._base_data = GenericCaseBaseData(
            scenario=scenario,
            measurements=measurements,
            fixed_demand=fixed,
            time_periods=periods,
            departure_sampling=sampling,
            physical_stop_mapping=physical_mapping,
            footpaths=(),
            production_inputs=production,
            destination_attractiveness=attraction,
            departure_seconds_by_origin=departures,
            input_semantics=input_semantics,
        )
        return self._base_data

    def build_od_universe(self) -> CandidateODUniverse | None:
        """Generate/validate the pair universe without using time bins."""
        if self._universe is not None:
            return self._universe
        base = self.load_base_data()
        if self.config.od_universe.source == "legacy_time_dependent_demand":
            return None
        self._universe = generate_candidate_od_pairs(
            base.scenario,
            source=self.config.od_universe.source,  # type: ignore[arg-type]
            level=self.config.od_universe.level,  # type: ignore[arg-type]
            include_same_stop=self.config.od_universe.include_same_stop,
            active_service_only=self.config.od_universe.active_service_only,
            connectivity_policy=self.config.od_universe.connectivity_policy,  # type: ignore[arg-type]
            od_pairs_path=self.config.od_universe.pair_file or self.config.paths.od_pairs,
            physical_stop_mapping=base.physical_stop_mapping,
        )
        return self._universe

    def _od_universe_identity(self) -> tuple[str, int, dict[str, object] | None]:
        """Return the current pair-universe identity without using stale audits."""
        if self.config.od_universe.source == "legacy_time_dependent_demand":
            base = self.load_base_data()
            pairs = sorted({(str(item.origin_stop_id), str(item.dest_stop_id)) for item in base.scenario.demand.records})
            encoded = json.dumps(pairs, separators=(",", ":")).encode("utf-8")
            return f"legacy-demand:{hashlib.sha256(encoded).hexdigest()}", len(pairs), None
        payload = self.load_current_od_universe_audit(required=True)
        return str(payload["fingerprint"]), int(payload["retained_pair_count"]), payload

    def load_current_od_universe_audit(self, *, required: bool = False) -> dict[str, object] | None:
        """Load the current OD audit and reject stale pair-universe artifacts."""
        if self.config.od_universe.source == "legacy_time_dependent_demand":
            return None
        path = self.config.paths.results_directory / "audit/od_universe.json"
        if not path.is_file():
            if required:
                raise FileNotFoundError(
                    "run od-universe before time-discretization for an independent OD case"
                )
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"OD-universe audit is not valid JSON: {path}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"OD-universe audit must contain an object: {path}")
        reasons: list[str] = []
        if payload.get("configuration_fingerprint") != self.config.fingerprint:
            reasons.append("configuration_fingerprint_mismatch")
        package = payload.get("package")
        package_revision = payload.get("package_revision")
        if package_revision is None and isinstance(package, Mapping):
            package_revision = package.get("locked_revision") or package.get("distribution_version")
        if package_revision != self._current_package_revision():
            reasons.append("package_revision_mismatch")
        source_checksums = payload.get("source_checksums")
        if source_checksums != self._current_source_checksums():
            reasons.append("source_checksum_mismatch")
        try:
            universe = self.load_persisted_universe()
        except (FileNotFoundError, ValueError) as error:
            reasons.append(f"pair_artifact_invalid:{error}")
            universe = None
        if universe is not None:
            if payload.get("fingerprint") != universe.fingerprint:
                reasons.append("od_universe_fingerprint_mismatch")
            try:
                retained_pair_count = int(payload.get("retained_pair_count", -1))
            except (TypeError, ValueError):
                retained_pair_count = -1
            if retained_pair_count != universe.pair_count:
                reasons.append("retained_pair_count_mismatch")
        if reasons:
            raise ValueError(
                "STALE ARTIFACT: " + str(path) + "\nReason: " + ", ".join(reasons)
                + "\nAction: rerun od-universe before continuing."
            )
        return payload

    def _time_bin_artifact_paths(self) -> tuple[Path, Path]:
        root = self.config.paths.results_directory / "generated_inputs"
        return root / "time_bins.csv", root / "time_bins_manifest.json"

    def _time_bin_staleness(self) -> list[dict[str, str]]:
        """Return explicit reasons why a generated-bin artifact is unusable."""
        generated_bins, manifest_path = self._time_bin_artifact_paths()
        if not generated_bins.is_file():
            if manifest_path.is_file():
                return [{"path": str(manifest_path), "reason": "missing_time_bins"}]
            return []
        reasons: list[str] = []
        if not manifest_path.is_file():
            return [{"path": str(generated_bins), "reason": "missing_manifest"}]
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return [{"path": str(generated_bins), "reason": "invalid_manifest"}]
        if not isinstance(manifest, Mapping):
            return [{"path": str(generated_bins), "reason": "invalid_manifest"}]
        expected_config = self.config.fingerprint
        if manifest.get("configuration_fingerprint") != expected_config:
            reasons.append("configuration_fingerprint_mismatch")
        if manifest.get("package_revision") != self._current_package_revision():
            reasons.append("package_revision_mismatch")
        if manifest.get("time_discretization_fingerprint") != self.config.time_discretization_fingerprint:
            reasons.append("time_discretization_fingerprint_mismatch")
        try:
            od_fingerprint, pair_count, _ = self._od_universe_identity()
        except (FileNotFoundError, ValueError):
            od_fingerprint, pair_count = "unavailable", -1
            reasons.append("od_universe_not_current")
        if manifest.get("od_universe_fingerprint") != od_fingerprint:
            reasons.append("od_universe_fingerprint_mismatch")
        if manifest.get("retained_pair_count") != pair_count:
            reasons.append("retained_pair_count_mismatch")
        if manifest.get("source_checksums") != self._current_source_checksums():
            reasons.append("source_checksum_mismatch")
        recommendation = manifest.get("recommendation")
        recommendation_path = Path(str(recommendation)) if recommendation else self.config.paths.results_directory / "audit/time_discretization_recommendation.json"
        if not recommendation_path.is_file():
            reasons.append("recommendation_missing")
        elif manifest.get("recommendation_fingerprint") != _sha256(recommendation_path):
            reasons.append("recommendation_fingerprint_mismatch")
        try:
            periods = _load_time_periods(generated_bins)
            actual_fingerprint = self._time_bins_fingerprint(periods)
            if manifest.get("time_bins_fingerprint") != actual_fingerprint:
                reasons.append("time_bins_fingerprint_mismatch")
        except ValueError:
            reasons.append("invalid_time_bins")
        return [{"path": str(generated_bins), "reason": reason} for reason in sorted(set(reasons))]

    def stale_artifacts(self) -> list[dict[str, str]]:
        """Expose generated artifacts that are present but not current."""
        return self._time_bin_staleness()

    def _approved_time_periods(self, *, require_materialized: bool) -> tuple[JourneyTimePeriod, ...]:
        base = self.load_base_data()
        generated_bins, _ = self._time_bin_artifact_paths()
        if generated_bins.is_file():
            stale = self._time_bin_staleness()
            if stale:
                reasons = ", ".join(item["reason"] for item in stale)
                raise ValueError(
                    f"STALE ARTIFACT: {generated_bins}\nReason: {reasons}\n"
                    "Action: rerun time-discretization and materialize-bins "
                    "(time-bin fingerprint is not current)."
                )
            return _load_time_periods(generated_bins)
        if require_materialized and self.config.od_universe.source != "legacy_time_dependent_demand":
            raise FileNotFoundError("run materialize-bins before expand-od")
        return base.time_periods

    def complexity_preflight(
        self,
        *,
        pair_count: int | None = None,
        time_bin_count: int | None = None,
        pair_level_exclusions: int | None = None,
        raise_on_exceed: bool = True,
    ) -> dict[str, object]:
        base = self.load_base_data()
        if pair_count is None:
            pair_count = self._persisted_pair_count()
            if pair_count is None and self.config.od_universe.source == "legacy_time_dependent_demand":
                pair_count = self._pair_file_count()
        if time_bin_count is None:
            # A missing value means that no approved bins are admitted.  In
            # particular, never infer it from an unvalidated generated file.
            time_bin_count = 0
        estimated = None if pair_count is None else int(pair_count) * int(time_bin_count)
        maximum = self.config.time_discretization.max_od_cells
        report = {
            "raw_network_nodes": len(base.scenario.stops),
            "candidate_pair_count": pair_count,
            "pair_level_exclusions": pair_level_exclusions,
            "retained_pair_count": pair_count,
            "approved_time_bin_count": time_bin_count,
            "estimated_od_time_cells": estimated,
            "maximum_configured_od_cells": maximum,
            "estimated_timetable_feasibility_calls": estimated,
        }
        if raise_on_exceed and estimated is not None and maximum is not None and estimated > maximum:
            raise ValueError(
                "estimated OD-time cells exceed max_od_cells before timetable expansion: "
                f"{estimated} > {maximum} (pairs={pair_count}, bins={time_bin_count})."
            )
        return report

    def build_od_time_expansion(
        self,
        universe: CandidateODUniverse | None = None,
    ) -> ODTimeExpansion:
        """Run timetable feasibility for the approved bins exactly once."""
        base = self.load_base_data()
        universe = universe or self.build_od_universe()
        if universe is None:
            raise ValueError("independent OD-time expansion is unavailable for legacy demand.csv.")
        periods = self._approved_time_periods(require_materialized=True)
        self.complexity_preflight(
            pair_count=universe.pair_count,
            pair_level_exclusions=len(universe.exclusions),
            time_bin_count=len(periods),
        )
        reduced = self.config.reduced_od_config
        index = TimetableFeasibilityIndex.from_scenario(
            base.scenario,
            physical_stop_mapping=universe.physical_stop_mapping,
        )
        expansion = expand_candidate_od_time_cells(
            universe,
            periods,
            scenario=base.scenario,
            maximum_transfers=reduced.journeys.maximum_transfers,
            maximum_initial_wait_seconds=reduced.journeys.maximum_waiting_seconds,
            maximum_journey_seconds=reduced.journeys.maximum_journey_seconds,
            maximum_waiting_seconds=reduced.journeys.maximum_waiting_seconds,
            timetable_policy="required",
            feasibility_index=index,
        )
        if expansion.cell_count == 0:
            raise ValueError(
                "OD-time expansion retained no cells; review pair-level and "
                "timetable-feasibility rules before estimation."
            )
        self._expansion = expansion
        return expansion

    def _pair_file_count(self) -> int | None:
        path = self.config.od_universe.pair_file or self.config.paths.od_pairs
        if path is None or not path.is_file():
            return None
        with path.open("r", encoding="utf-8-sig") as stream:
            return max(sum(1 for _ in stream) - 1, 0)

    def _persisted_pair_count(self) -> int | None:
        if self.config.od_universe.source == "legacy_time_dependent_demand":
            base = self.load_base_data()
            return len({(str(item.origin_stop_id), str(item.dest_stop_id)) for item in base.scenario.demand.records})
        try:
            payload = self.load_current_od_universe_audit(required=False)
        except ValueError:
            return None
        if payload is None:
            return None
        value = payload.get("retained_pair_count")
        return int(value) if value is not None else None

    def _pair_mapping(self, base: GenericCaseBaseData) -> dict[str, str]:
        if self.config.od_universe.level != "physical_stop":
            return {str(stop.stop_id): str(stop.stop_id) for stop in base.scenario.stops}
        mapping = dict(base.physical_stop_mapping or {})
        return {**mapping, **{value: value for value in mapping.values()}}

    def load_persisted_universe(self) -> CandidateODUniverse:
        """Read and validate the pair artifact produced by ``od-universe``."""
        audit_path = self.config.paths.results_directory / "audit/od_universe.json"
        pair_path = self.config.paths.results_directory / "audit/od_pairs.csv"
        exclusion_path = self.config.paths.results_directory / "audit/od_universe_exclusions.csv"
        if not audit_path.is_file() or not pair_path.is_file() or not exclusion_path.is_file():
            raise FileNotFoundError("run od-universe before expand-od")
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        if payload.get("configuration_fingerprint") != self.config.fingerprint:
            raise ValueError("OD-universe artifact configuration fingerprint does not match current configuration.")
        base = self.load_base_data()
        pairs: list[CandidateODPair] = []
        with pair_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["origin_stop_id", "destination_stop_id"]:
                raise ValueError("persisted OD-pair artifact has an invalid header.")
            for row in reader:
                pairs.append(CandidateODPair(row["origin_stop_id"], row["destination_stop_id"]))
        exclusions: list[ODUniverseExclusion] = []
        with exclusion_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"origin_stop_id", "destination_stop_id", "reason", "detail"}
            if reader.fieldnames is None or set(reader.fieldnames) != required:
                raise ValueError("persisted OD-universe exclusion artifact has an invalid header.")
            for row in reader:
                exclusions.append(
                    ODUniverseExclusion(
                        row["origin_stop_id"], row["destination_stop_id"], row["reason"], row["detail"]
                    )
                )
        universe = CandidateODUniverse(
            pairs=tuple(sorted(pairs)),
            exclusions=tuple(sorted(exclusions, key=lambda item: item.tuple)),
            source=self.config.od_universe.source,  # type: ignore[arg-type]
            level=self.config.od_universe.level,  # type: ignore[arg-type]
            include_same_stop=self.config.od_universe.include_same_stop,
            active_service_only=self.config.od_universe.active_service_only,
            connectivity_policy=self.config.od_universe.connectivity_policy,  # type: ignore[arg-type]
            physical_stop_mapping=MappingProxyType(self._pair_mapping(base)),
            generator_fingerprint=str(payload.get("generator_fingerprint", "")),
        )
        expected = payload.get("fingerprint")
        if not expected or universe.fingerprint != expected:
            raise ValueError("OD-universe artifact fingerprint does not match its persisted contents.")
        package = payload.get("package")
        package_revision = payload.get("package_revision")
        if package_revision is None and isinstance(package, Mapping):
            package_revision = package.get("locked_revision") or package.get("distribution_version")
        if package_revision != self._current_package_revision():
            raise ValueError("OD-universe artifact package revision does not match the current package.")
        if payload.get("source_checksums") != self._current_source_checksums():
            raise ValueError("OD-universe artifact source checksums do not match current inputs.")
        try:
            retained_pair_count = int(payload.get("retained_pair_count", -1))
        except (TypeError, ValueError):
            retained_pair_count = -1
        if retained_pair_count != universe.pair_count:
            raise ValueError("OD-universe artifact retained pair count does not match its pair file.")
        self._universe = universe
        return universe

    @staticmethod
    def _time_bins_fingerprint(periods: tuple[JourneyTimePeriod, ...]) -> str:
        payload = json.dumps(
            [[item.period_id, item.start_seconds, item.end_seconds] for item in periods],
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def load_persisted_expansion(self) -> ODTimeExpansion:
        """Read and validate the expansion artifact; never rebuild it."""
        audit_path = self.config.paths.results_directory / "audit/od_time_expansion.json"
        cell_path = self.config.paths.results_directory / "generated_inputs/candidate_od_time.csv"
        exclusion_path = self.config.paths.results_directory / "audit/od_time_exclusions.csv"
        if not audit_path.is_file() or not cell_path.is_file() or not exclusion_path.is_file():
            raise FileNotFoundError("run expand-od before structural-zeros")
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        if payload.get("configuration_fingerprint") != self.config.fingerprint:
            raise ValueError("OD-time expansion artifact configuration fingerprint does not match current configuration.")
        universe = self.load_persisted_universe()
        if payload.get("universe_fingerprint") != universe.fingerprint:
            raise ValueError("OD-time expansion artifact universe fingerprint does not match current pair artifact.")
        periods = self._approved_time_periods(require_materialized=True)
        bins = tuple((item.period_id, item.start_seconds, item.end_seconds) for item in periods)
        bins_fingerprint = self._time_bins_fingerprint(periods)
        if payload.get("approved_time_bins_fingerprint") != bins_fingerprint:
            raise ValueError("OD-time expansion artifact time-bin fingerprint does not match approved bins.")
        cells: list[CandidateODTimeCell] = []
        with cell_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["origin_stop_id", "destination_stop_id", "time_bin_id"]:
                raise ValueError("persisted candidate OD-time artifact has an invalid header.")
            for row in reader:
                cells.append(CandidateODTimeCell(row["origin_stop_id"], row["destination_stop_id"], row["time_bin_id"]))
        exclusions: list[ODTimeExclusion] = []
        with exclusion_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"origin_stop_id", "destination_stop_id", "time_bin_id", "reason", "detail"}
            if reader.fieldnames is None or set(reader.fieldnames) != required:
                raise ValueError("persisted OD-time exclusion artifact has an invalid header.")
            for row in reader:
                exclusions.append(
                    ODTimeExclusion(
                        row["origin_stop_id"], row["destination_stop_id"], row["time_bin_id"], row["reason"], row["detail"]
                    )
                )
        policies = payload.get("feasibility_settings", payload.get("policies", {}))
        expansion = ODTimeExpansion(
            universe_fingerprint=universe.fingerprint,
            cells=tuple(sorted(cells)),
            exclusions=tuple(sorted(exclusions, key=lambda item: (item.origin_stop_id, item.destination_stop_id, item.time_bin_id, item.reason))),
            time_bins=bins,
            policies=MappingProxyType(dict(policies)),
            fingerprint=str(payload.get("expansion_fingerprint", payload.get("fingerprint", ""))),
        )
        if expansion.fingerprint != payload.get("expansion_fingerprint", payload.get("fingerprint")):
            raise ValueError("OD-time expansion artifact fingerprint does not match its persisted contents.")
        self._universe = universe
        self._expansion = expansion
        return expansion

    def _load_persisted_prior(
        self, expansion: ODTimeExpansion
    ) -> PriorGenerationResult:
        path = self.config.paths.results_directory / "generated_inputs/prior_demand.csv"
        audit_path = self.config.paths.results_directory / "audit/prior_generation.json"
        if not path.is_file() or not audit_path.is_file():
            raise FileNotFoundError("run expand-od before prepare")
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        if payload.get("od_time_expansion_fingerprint") != expansion.fingerprint:
            raise ValueError("prior artifact expansion fingerprint does not match current expansion.")
        values: dict[CandidateODTimeCell, float] = {}
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"origin_stop_id", "destination_stop_id", "time_bin_id", "prior_value"}
            if reader.fieldnames is None or set(reader.fieldnames) != required:
                raise ValueError("persisted prior-demand artifact has an invalid header.")
            for row in reader:
                cell = CandidateODTimeCell(row["origin_stop_id"], row["destination_stop_id"], row["time_bin_id"])
                value = float(row["prior_value"])
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError("persisted prior-demand values must be finite and positive.")
                values[cell] = value
        if set(values) != set(expansion.cells):
            raise ValueError("persisted prior-demand cells do not match retained expansion cells.")
        return PriorGenerationResult(
            values=MappingProxyType(dict(sorted(values.items()))),
            source=str(payload.get("source", self.config.prior_demand.source)),
            semantics=str(payload.get("semantics", self.config.prior_demand.semantics)),
            parameters=dict(payload.get("parameters", {})),
            generator_fingerprint=str(payload.get("generator_fingerprint", "")),
            fingerprint=str(payload.get("fingerprint", "")),
        )

    def _compose_data(
        self,
        base: GenericCaseBaseData,
        *,
        universe: CandidateODUniverse | None = None,
        expansion: ODTimeExpansion | None = None,
        prior_generation: PriorGenerationResult | None = None,
    ) -> GenericCaseData:
        if base.input_semantics == "legacy_time_dependent_demand":
            data = GenericCaseData(
                scenario=base.scenario,
                measurements=base.measurements,
                fixed_demand=dict(base.fixed_demand),
                time_periods=base.time_periods,
                production_inputs=dict(base.production_inputs),
                destination_attractiveness=dict(base.destination_attractiveness),
                departure_seconds_by_origin=dict(base.departure_seconds_by_origin),
                departure_sampling=base.departure_sampling,
                physical_stop_mapping=base.physical_stop_mapping,
                footpaths=base.footpaths,
                input_semantics=base.input_semantics,
            )
        else:
            if universe is None or expansion is None or prior_generation is None:
                raise ValueError("independent preparation requires persisted OD-universe, expansion, and prior artifacts.")
            fixed = dict(base.fixed_demand)
            generated_fixed = {
                ResponseCellKey(item.origin_stop_id, item.destination_stop_id, item.time_bin_id): 0.0
                for item in expansion.exclusions
            }
            conflicts = sorted(
                key.tuple for key, value in generated_fixed.items() if key in fixed and value > 0.0
            )
            if conflicts:
                raise ValueError(f"positive fixed demand conflicts with an OD-time structural zero: {conflicts}")
            fixed.update(generated_fixed)
            production, attraction = self._explicit_components(expansion)
            periods = tuple(
                JourneyTimePeriod(item[0], item[1], item[2]) for item in expansion.time_bins
            )
            origins = sorted({cell.origin_stop_id for cell in expansion.cells})
            departures = {
                origin: tuple(
                    period.start_seconds + (period.end_seconds - period.start_seconds) // 2
                    for period in periods
                )
                for origin in origins
            }
            data = GenericCaseData(
                scenario=base.scenario,
                measurements=base.measurements,
                fixed_demand=fixed,
                time_periods=periods,
                production_inputs=production,
                destination_attractiveness=attraction,
                departure_seconds_by_origin=departures,
                departure_sampling=base.departure_sampling,
                physical_stop_mapping=base.physical_stop_mapping,
                footpaths=base.footpaths,
                candidate_od_universe=universe,
                od_time_expansion=expansion,
                prior_demand=prior_generation.values,
                prior_generation=prior_generation,
                input_semantics=base.input_semantics,
            )
        if self.custom_hook is not None:
            transformed = self.custom_hook(self.config, data)
            if not isinstance(transformed, GenericCaseData):
                raise TypeError("custom case-study hook must return GenericCaseData.")
            data = transformed
        self._data = data
        return data

    def load_persisted_data(self) -> GenericCaseData:
        """Build preparation data from persisted artifacts only."""
        base = self.load_base_data()
        if base.input_semantics == "legacy_time_dependent_demand":
            return self._compose_data(base)
        expansion = self.load_persisted_expansion()
        summary_path = self.config.paths.results_directory / "audit/structural_zero_summary.json"
        fixed_path = self.config.paths.results_directory / "generated_inputs/fixed_demand.csv"
        if not summary_path.is_file() or not fixed_path.is_file():
            raise FileNotFoundError("run expand-od before structural-zeros, then run structural-zeros before prepare")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("configuration_fingerprint") != self.config.fingerprint:
            raise ValueError("structural-zero artifact configuration fingerprint does not match current configuration.")
        if summary.get("expansion_fingerprint") != expansion.fingerprint:
            raise ValueError("structural-zero artifact expansion fingerprint does not match current expansion.")
        persisted_fixed: dict[ResponseCellKey, float] = {}
        with fixed_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"origin_stop_id", "dest_stop_id", "time_bin_id", "fixed_flow"}
            if reader.fieldnames is None or set(reader.fieldnames) != required:
                raise ValueError("persisted structural-zero artifact has an invalid header.")
            for row in reader:
                key = ResponseCellKey(
                    _required_text(row.get("origin_stop_id"), location="fixed-demand origin_stop_id"),
                    _required_text(row.get("dest_stop_id"), location="fixed-demand dest_stop_id"),
                    _required_text(row.get("time_bin_id"), location="fixed-demand time_bin_id"),
                )
                value = float(row.get("fixed_flow", ""))
                if not math.isfinite(value) or value != 0.0:
                    raise ValueError("persisted structural-zero values must be exactly zero.")
                if key in persisted_fixed:
                    raise ValueError(f"persisted structural-zero artifact contains duplicate key {key.tuple!r}.")
                persisted_fixed[key] = value
        expected_fixed = {
            ResponseCellKey(item.origin_stop_id, item.destination_stop_id, item.time_bin_id): 0.0
            for item in expansion.exclusions
        }
        if persisted_fixed != expected_fixed:
            raise ValueError("structural-zero artifact does not match persisted OD-time exclusions.")
        prior = self._load_persisted_prior(expansion)
        return self._compose_data(base, universe=self._universe, expansion=expansion, prior_generation=prior)

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
        base = self.load_base_data()
        timetable = prepare_reduced_od_timetable(
            base.scenario,
            configuration_fingerprint=self.config.reduced_od_config.fingerprint,
            physical_stop_mapping=base.physical_stop_mapping,
            mapping_policy=self.config.reduced_od_config.stops.mapping_policy,
        )
        return timetable

    def audit(self) -> GenericCaseAudit:
        data = self.load_base_data()
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
        legacy_demand_file = (
            (self.config.paths.candidate_demand is not None and self.config.paths.candidate_demand.is_file())
            or (self.config.paths.scenario_directory / "demand.csv").is_file()
        )
        legacy_demand_row_count = len(data.scenario.demand.records) if legacy_demand_file else 0
        pair_count = self._persisted_pair_count()
        configured_pairs = self.config.time_discretization.num_od_pairs
        if configured_pairs is not None and pair_count is not None and configured_pairs != pair_count and data.input_semantics == "legacy_time_dependent_demand":
            raise ValueError(f"configured num_od_pairs={configured_pairs} does not match candidate demand count {pair_count}.")
        counts: dict[str, int] = {}
        for item in data.measurements.records:
            counts[item.measurement_type.value] = counts.get(item.measurement_type.value, 0) + 1
        checksums = self._current_source_checksums()
        stale = self.stale_artifacts()
        generated_bins, _ = self._time_bin_artifact_paths()
        approved_bin_count = 0
        approved_status = "not_available"
        if generated_bins.is_file() and not stale:
            approved_bin_count = len(_load_time_periods(generated_bins))
            approved_status = "available"
        complexity = self.complexity_preflight(
            pair_count=pair_count,
            time_bin_count=approved_bin_count,
            raise_on_exceed=False,
        )
        return GenericCaseAudit(
            case_name=self.config.case_name,
            configuration_fingerprint=self.config.fingerprint,
            source_checksums=checksums,
            scenario_stop_count=len(data.scenario.stops),
            demand_cell_count=(len(data.scenario.demand.records) if data.input_semantics == "legacy_time_dependent_demand" else 0),
            candidate_od_pair_count=pair_count,
            measurement_count=len(data.measurements.records),
            measurement_type_counts=counts,
            resolved_measurement_count=len(resolved),
            time_period_count=len(data.time_periods),
            period_preflight=period_report.to_dict(),
            od_universe_status=("current" if data.input_semantics != "legacy_time_dependent_demand" and pair_count is not None else ("not_run" if data.input_semantics != "legacy_time_dependent_demand" else "legacy_compatibility")),
            od_time_expansion_status="not_run",
            timetable_feasibility_status="not_run",
            complexity=complexity,
            od_universe=None,
            od_time_expansion=None,
            prior_generation=None,
            input_semantics=data.input_semantics,
            source_time_bins={"count": len(data.time_periods), "semantics": "legacy_source_bins"},
            approved_time_bins={"status": approved_status, "count": approved_bin_count if approved_status == "available" else None},
            stale_artifacts=stale,
            active_candidate_od_time_count=None,
            legacy_demand_file_present=legacy_demand_file,
            legacy_demand_row_count=legacy_demand_row_count,
            legacy_demand_used=data.input_semantics == "legacy_time_dependent_demand",
        )

    def build_preparation_inputs(self):
        from public_transportation.inference.reduced_od import ReducedODPreparationInputs

        data = self.load_persisted_data()
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

    def preparation_inputs(self):
        """Backward-compatible name for the persisted-artifact preparation path."""
        return self.build_preparation_inputs()
