"""Strict, deterministic configuration for a generic case-study workflow."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from public_transportation.preprocessing.reduced_od.config import (
    ReducedODConfig,
    load_reduced_od_config,
)
from public_transportation.preprocessing.structural_zeros.config import (
    StructuralZeroConfig,
    load_structural_zero_config,
)


CASE_CONFIG_SCHEMA_VERSION = 1


class CaseStudyConfigError(ValueError):
    """Raised when a generic case-study configuration is incomplete or invalid."""


def _canonical(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _table(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = root.get(name)
    if not isinstance(value, Mapping):
        raise CaseStudyConfigError(f"{name} must be a TOML table.")
    return value


def _keys(table: Mapping[str, Any], *, name: str, required: set[str], optional: set[str] = set()) -> None:
    unknown = sorted(set(table) - required - optional)
    missing = sorted(required - set(table))
    if unknown:
        raise CaseStudyConfigError(f"{name} contains unknown fields: {unknown}.")
    if missing:
        raise CaseStudyConfigError(f"{name} is missing required fields: {missing}.")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaseStudyConfigError(f"{name} must be a non-empty string.")
    return value.strip()


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise CaseStudyConfigError(f"{name} must be true or false.")
    return value


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaseStudyConfigError(f"{name} must be an integer.")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise CaseStudyConfigError(f"{name} must be {qualifier}.")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaseStudyConfigError(f"{name} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise CaseStudyConfigError(f"{name} must be finite.")
    return parsed


def _seconds(value: Any, name: str, *, required: bool = True) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise CaseStudyConfigError(f"{name} must be supplied explicitly.")
        return None
    if isinstance(value, bool):
        raise CaseStudyConfigError(f"{name} must be HH:MM[:SS] or seconds.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        parts = value.strip().split(":")
        if len(parts) not in {2, 3}:
            raise CaseStudyConfigError(f"{name} must be HH:MM[:SS] or seconds.")
        try:
            hour, minute = int(parts[0]), int(parts[1])
            second = int(parts[2]) if len(parts) == 3 else 0
        except ValueError as error:
            raise CaseStudyConfigError(f"{name} must be HH:MM[:SS] or seconds.") from error
        if hour < 0 or minute not in range(60) or second not in range(60):
            raise CaseStudyConfigError(f"{name} contains an invalid time.")
        parsed = hour * 3600 + minute * 60 + second
    else:
        raise CaseStudyConfigError(f"{name} must be HH:MM[:SS] or seconds.")
    if parsed < 0:
        raise CaseStudyConfigError(f"{name} must be non-negative.")
    return parsed


@dataclass(frozen=True, slots=True)
class CasePaths:
    scenario_directory: Path
    measurements: Path
    candidate_demand: Path
    fixed_demand: Path | None
    results_directory: Path


@dataclass(frozen=True, slots=True)
class ObservationMapping:
    method_id_column: str | None
    measurement_type_column: str
    stop_id_column: str
    timestamp_column: str
    value_column: str
    trip_id_column: str | None
    line_id_column: str | None
    timestamp_semantics: str
    missing_value_policy: str
    ambiguous_event_policy: str


@dataclass(frozen=True, slots=True)
class TimeDiscretizationSettings:
    configuration_file: Path | None
    required_when_timestamps_exist: bool
    base_resolution_minutes: int
    min_bin_minutes: int
    max_bin_minutes: int
    max_bins: int
    num_od_pairs: int | None
    max_od_cells: int | None
    horizon_start_s: int | None
    horizon_end_s: int | None
    candidate: str | None


@dataclass(frozen=True, slots=True)
class SamplingSettings:
    strategy: str
    samples_per_period: int | dict[str, int]
    time_step_seconds: int | dict[str, int]


@dataclass(frozen=True, slots=True)
class CaseStudyConfig:
    """Validated generic case configuration with resolved paths."""

    schema_version: int
    source_file: Path
    case_name: str
    service_day: str
    timezone: str
    after_midnight_convention: str
    paths: CasePaths
    observations: ObservationMapping
    time_discretization: TimeDiscretizationSettings
    structural_zero_config_file: Path
    reduced_od_config_file: Path
    model_config_file: Path
    sampling: SamplingSettings
    reduced_od_config: ReducedODConfig
    structural_zero_config: StructuralZeroConfig
    model: dict[str, Any]

    @property
    def fingerprint_payload_json(self) -> str:
        payload = _canonical(asdict(self))
        payload.pop("source_file", None)
        payload.pop("reduced_od_config", None)
        payload.pop("structural_zero_config", None)
        payload["reduced_od_fingerprint"] = self.reduced_od_config.fingerprint
        payload["structural_zero_fingerprint"] = self.structural_zero_config.fingerprint
        payload["model"] = _canonical(self.model)
        return _json(payload)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.fingerprint_payload_json.encode("utf-8")).hexdigest()

    @property
    def package_config_paths(self) -> tuple[Path, ...]:
        return (
            self.source_file,
            *(() if self.time_discretization.configuration_file is None else (self.time_discretization.configuration_file,)),
            self.structural_zero_config_file,
            self.reduced_od_config_file,
            self.model_config_file,
        )


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CaseStudyConfigError(f"Configuration file does not exist: {path}")
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise CaseStudyConfigError(f"Invalid TOML in {path}: {error}") from error
    return value


def _resolved_path(value: Any, name: str, case_root: Path, *, optional: bool = False) -> Path | None:
    if value is None and optional:
        return None
    path = Path(_text(value, name)).expanduser()
    return (case_root / path).resolve() if not path.is_absolute() else path.resolve()


def _mapping(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    return _text(value, name)


def _load_model(path: Path) -> dict[str, Any]:
    root = _load_toml(path)
    version = _positive_int(root.get("schema_version"), "model.schema_version")
    if version != 1:
        raise CaseStudyConfigError("model.schema_version must be 1.")
    _keys(root, name="model", required={"schema_version", "likelihood", "production_mode"}, optional={"maximum_iterations", "gradient_tolerance", "function_tolerance", "dispersion", "prior_scale"})
    likelihood = _text(root["likelihood"], "model.likelihood")
    if likelihood not in {"poisson", "negative_binomial"}:
        raise CaseStudyConfigError("model.likelihood must be poisson or negative_binomial.")
    production_mode = _text(root["production_mode"], "model.production_mode")
    if production_mode not in {"provided", "estimated_basis"}:
        raise CaseStudyConfigError("model.production_mode must be provided or estimated_basis.")
    for name in ("maximum_iterations",):
        if name in root:
            _positive_int(root[name], f"model.{name}")
    for name in ("gradient_tolerance", "function_tolerance"):
        if name in root and _finite(root[name], f"model.{name}") <= 0.0:
            raise CaseStudyConfigError(f"model.{name} must be positive.")
    for name in ("dispersion", "prior_scale"):
        if name in root and _finite(root[name], f"model.{name}") <= 0.0:
            raise CaseStudyConfigError(f"model.{name} must be positive.")
    return {str(key): value for key, value in root.items()}


def _validate_time_discretization_file(path: Path, values: Mapping[str, Any]) -> None:
    """Validate the optional standalone time-discretization contract."""
    root = _load_toml(path)
    _keys(
        root,
        name="time_discretization_file",
        required={
            "schema_version",
            "required_when_timestamps_exist",
            "base_resolution_minutes",
            "min_bin_minutes",
            "max_bin_minutes",
            "max_bins",
            "num_od_pairs",
            "max_od_cells",
            "horizon_start",
            "horizon_end",
            "candidate",
        },
    )
    if _positive_int(root["schema_version"], "time_discretization_file.schema_version") != 1:
        raise CaseStudyConfigError("time_discretization_file.schema_version must be 1.")
    for name in (
        "required_when_timestamps_exist",
        "base_resolution_minutes",
        "min_bin_minutes",
        "max_bin_minutes",
        "max_bins",
        "num_od_pairs",
        "max_od_cells",
        "horizon_start",
        "horizon_end",
        "candidate",
    ):
        if root[name] != values[name]:
            raise CaseStudyConfigError(
                f"time-discretization value {name!r} disagrees between case.toml and {path}."
            )


def load_case_study_config(path: str | Path, *, case_root: str | Path | None = None) -> CaseStudyConfig:
    """Load and validate ``config/case.toml`` and all referenced contracts."""
    source = Path(path).expanduser().resolve()
    root_dir = Path(case_root).expanduser().resolve() if case_root is not None else source.parent.parent
    root = _load_toml(source)
    _keys(root, name="configuration", required={"schema_version", "case", "paths", "observations", "time_discretization", "structural_zeros", "model"}, optional={"sampling"})
    version = _positive_int(root["schema_version"], "schema_version")
    if version != CASE_CONFIG_SCHEMA_VERSION:
        raise CaseStudyConfigError(f"Unsupported schema_version {version}; expected {CASE_CONFIG_SCHEMA_VERSION}.")
    case = _table(root, "case")
    _keys(case, name="case", required={"name", "service_day", "timezone", "after_midnight_convention"})
    case_name = _text(case["name"], "case.name")
    service_day = _text(case["service_day"], "case.service_day")
    timezone = _text(case["timezone"], "case.timezone")
    convention = _text(case["after_midnight_convention"], "case.after_midnight_convention")
    try:
        date.fromisoformat(service_day)
    except ValueError as error:
        raise CaseStudyConfigError("case.service_day must be an ISO date (YYYY-MM-DD).") from error
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as error:
        raise CaseStudyConfigError(f"case.timezone is not a known IANA timezone: {timezone!r}.") from error
    if convention != "seconds_from_service_day":
        raise CaseStudyConfigError("case.after_midnight_convention must be seconds_from_service_day.")

    paths = _table(root, "paths")
    _keys(paths, name="paths", required={"scenario_directory", "measurements", "candidate_demand", "results_directory"}, optional={"fixed_demand", "production_inputs", "destination_attractiveness"})
    resolved_paths = CasePaths(
        scenario_directory=_resolved_path(paths["scenario_directory"], "paths.scenario_directory", root_dir),
        measurements=_resolved_path(paths["measurements"], "paths.measurements", root_dir),
        candidate_demand=_resolved_path(paths["candidate_demand"], "paths.candidate_demand", root_dir),
        fixed_demand=_resolved_path(paths.get("fixed_demand"), "paths.fixed_demand", root_dir, optional=True),
        results_directory=_resolved_path(paths["results_directory"], "paths.results_directory", root_dir),
    )

    observations = _table(root, "observations")
    _keys(observations, name="observations", required={"measurement_type_column", "stop_id_column", "timestamp_column", "value_column", "timestamp_semantics", "missing_value_policy", "ambiguous_event_policy"}, optional={"method_id_column", "trip_id_column", "line_id_column"})
    mapping = ObservationMapping(
        method_id_column=_mapping(observations.get("method_id_column"), "observations.method_id_column", optional=True),
        measurement_type_column=_text(observations["measurement_type_column"], "observations.measurement_type_column"),
        stop_id_column=_text(observations["stop_id_column"], "observations.stop_id_column"),
        timestamp_column=_text(observations["timestamp_column"], "observations.timestamp_column"),
        value_column=_text(observations["value_column"], "observations.value_column"),
        trip_id_column=_mapping(observations.get("trip_id_column"), "observations.trip_id_column", optional=True),
        line_id_column=_mapping(observations.get("line_id_column"), "observations.line_id_column", optional=True),
        timestamp_semantics=_text(observations["timestamp_semantics"], "observations.timestamp_semantics"),
        missing_value_policy=_text(observations["missing_value_policy"], "observations.missing_value_policy"),
        ambiguous_event_policy=_text(observations["ambiguous_event_policy"], "observations.ambiguous_event_policy"),
    )
    if mapping.timestamp_semantics != "event_time":
        raise CaseStudyConfigError("observations.timestamp_semantics must be event_time.")
    if mapping.missing_value_policy != "error":
        raise CaseStudyConfigError("observations.missing_value_policy must be error.")
    if mapping.ambiguous_event_policy != "error":
        raise CaseStudyConfigError("observations.ambiguous_event_policy must be error.")
    if mapping.trip_id_column is None and mapping.line_id_column is None:
        raise CaseStudyConfigError("At least one of trip_id_column or line_id_column is required.")

    td = _table(root, "time_discretization")
    _keys(td, name="time_discretization", required={"required_when_timestamps_exist", "base_resolution_minutes", "min_bin_minutes", "max_bin_minutes", "max_bins", "num_od_pairs", "max_od_cells", "horizon_start", "horizon_end", "candidate"}, optional={"configuration_file"})
    num_pairs = _positive_int(td["num_od_pairs"], "time_discretization.num_od_pairs", allow_zero=True) or None
    max_cells = _positive_int(td["max_od_cells"], "time_discretization.max_od_cells", allow_zero=True) or None
    if max_cells is None:
        raise CaseStudyConfigError("time_discretization.max_od_cells must be explicitly approved and positive.")
    time_file = _resolved_path(td.get("configuration_file"), "time_discretization.configuration_file", root_dir, optional=True)
    if time_file is not None:
        _validate_time_discretization_file(time_file, td)
    settings = TimeDiscretizationSettings(
        configuration_file=time_file,
        required_when_timestamps_exist=_bool(td["required_when_timestamps_exist"], "time_discretization.required_when_timestamps_exist"),
        base_resolution_minutes=_positive_int(td["base_resolution_minutes"], "time_discretization.base_resolution_minutes"),
        min_bin_minutes=_positive_int(td["min_bin_minutes"], "time_discretization.min_bin_minutes"),
        max_bin_minutes=_positive_int(td["max_bin_minutes"], "time_discretization.max_bin_minutes"),
        max_bins=_positive_int(td["max_bins"], "time_discretization.max_bins"),
        num_od_pairs=num_pairs,
        max_od_cells=max_cells,
        horizon_start_s=_seconds(td["horizon_start"], "time_discretization.horizon_start"),
        horizon_end_s=_seconds(td["horizon_end"], "time_discretization.horizon_end"),
        candidate=_text(td["candidate"], "time_discretization.candidate"),
    )
    if settings.max_bin_minutes < settings.min_bin_minutes:
        raise CaseStudyConfigError("time_discretization.max_bin_minutes must be >= min_bin_minutes.")
    if settings.num_od_pairs is not None and settings.max_od_cells < settings.num_od_pairs:
        raise CaseStudyConfigError("max_od_cells must allow at least one time bin.")
    assert settings.horizon_start_s is not None and settings.horizon_end_s is not None
    if settings.horizon_end_s <= settings.horizon_start_s:
        raise CaseStudyConfigError("time-discretization horizon_end must exceed horizon_start.")

    structural = _table(root, "structural_zeros")
    _keys(structural, name="structural_zeros", required={"configuration_file"})
    reduced = _table(root, "model")
    _keys(reduced, name="model", required={"configuration_file"}, optional={"settings_file"})
    structural_file = _resolved_path(structural["configuration_file"], "structural_zeros.configuration_file", root_dir)
    reduced_file = _resolved_path(reduced["configuration_file"], "model.configuration_file", root_dir)
    model_file = _resolved_path(root["model"].get("settings_file", "config/model.toml"), "model.settings_file", root_dir)
    # The top-level model table intentionally accepts only the reference to the
    # reduced-OD contract and optional model-settings file.
    reduced_config = load_reduced_od_config(reduced_file)
    structural_config = load_structural_zero_config(structural_file)
    model = _load_model(model_file)
    if reduced_config.observations.service_day != service_day:
        raise CaseStudyConfigError(
            "case.service_day must agree with reduced-OD observations.service_day."
        )
    if (
        reduced_config.observations.analysis_start_seconds != settings.horizon_start_s
        or reduced_config.observations.analysis_end_seconds != settings.horizon_end_s
    ):
        raise CaseStudyConfigError(
            "time-discretization horizon must agree with the reduced-OD analysis interval."
        )
    if reduced_config.observations.after_midnight_convention != "service_day_extended":
        raise CaseStudyConfigError(
            "reduced-OD observations.after_midnight_convention must be service_day_extended."
        )
    if model["likelihood"] != reduced_config.model.likelihood:
        raise CaseStudyConfigError(
            "model.likelihood must agree with the reduced-OD model likelihood."
        )
    if model["production_mode"] != reduced_config.productions.mode:
        raise CaseStudyConfigError(
            "model.production_mode must agree with the reduced-OD production mode."
        )

    sampling_root = root.get("sampling", {"strategy": "fixed_count", "samples_per_period": 1, "time_step_seconds": 300})
    if not isinstance(sampling_root, Mapping):
        raise CaseStudyConfigError("sampling must be a TOML table.")
    _keys(sampling_root, name="sampling", required={"strategy", "samples_per_period", "time_step_seconds"})
    strategy = _text(sampling_root["strategy"], "sampling.strategy")
    if strategy not in {"uniform_midpoint", "fixed_count", "fixed_time_step", "adaptive_service_aware"}:
        raise CaseStudyConfigError("sampling.strategy is unsupported.")
    def _sample_values(value: Any, name: str) -> int | dict[str, int]:
        if isinstance(value, int) and not isinstance(value, bool):
            return _positive_int(value, name)
        if isinstance(value, Mapping):
            result = {str(key): _positive_int(item, f"{name}.{key}") for key, item in value.items()}
            if not result:
                raise CaseStudyConfigError(f"{name} must not be empty.")
            return result
        raise CaseStudyConfigError(f"{name} must be an integer or table of integers.")
    sampling = SamplingSettings(strategy, _sample_values(sampling_root["samples_per_period"], "sampling.samples_per_period"), _sample_values(sampling_root["time_step_seconds"], "sampling.time_step_seconds"))
    return CaseStudyConfig(
        version,
        source,
        case_name,
        service_day,
        timezone,
        convention,
        resolved_paths,
        mapping,
        settings,
        structural_file,
        reduced_file,
        model_file,
        sampling,
        reduced_config,
        structural_config,
        model,
    )
