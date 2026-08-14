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
    candidate_demand: Path | None
    od_pairs: Path | None
    prior_demand: Path | None
    fixed_demand: Path | None
    production_inputs: Path | None
    destination_attractiveness: Path | None
    results_directory: Path


@dataclass(frozen=True, slots=True)
class ODUniverseSettings:
    source: str
    level: str
    include_same_stop: bool
    active_service_only: bool
    connectivity_policy: str
    pair_file: Path | None


@dataclass(frozen=True, slots=True)
class PriorDemandSettings:
    source: str
    value: float | None
    semantics: str
    expansion: str
    input_file: Path | None


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
    od_universe: ODUniverseSettings
    prior_demand: PriorDemandSettings
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
    path = (
        value.expanduser()
        if isinstance(value, Path)
        else Path(_text(value, name)).expanduser()
    )
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
    _keys(
        root,
        name="model",
        required={"schema_version", "likelihood"},
        optional={
            "production_mode",
            "maximum_iterations",
            "gradient_tolerance",
            "function_tolerance",
            "dispersion",
            "prior_scale",
            "production",
            "destination_attractiveness",
        },
    )
    likelihood = _text(root["likelihood"], "model.likelihood")
    if likelihood not in {"poisson", "negative_binomial"}:
        raise CaseStudyConfigError("model.likelihood must be poisson or negative_binomial.")
    production_mode = root.get("production_mode")
    if production_mode is not None:
        production_mode = _text(production_mode, "model.production_mode")
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
    parsed = {str(key): value for key, value in root.items()}
    for component_name in ("production", "destination_attractiveness"):
        if component_name not in root:
            continue
        component = _table(root, component_name)
        _keys(
            component,
            name=f"model.{component_name}",
            required={"mode", "baseline", "correction_scope", "transformation", "constraint", "regularization", "prior_scale"},
            optional={"input_file"},
        )
        mode = _text(component["mode"], f"model.{component_name}.mode")
        if mode not in {"provided", "fixed", "estimated"}:
            raise CaseStudyConfigError(
                f"model.{component_name}.mode must be provided, fixed, or estimated."
            )
        prior_scale = _finite(component["prior_scale"], f"model.{component_name}.prior_scale")
        if prior_scale <= 0.0:
            raise CaseStudyConfigError(f"model.{component_name}.prior_scale must be positive.")
        correction_scope = _text(
            component["correction_scope"],
            f"model.{component_name}.correction_scope",
        )
        allowed_scopes = {
            "global",
            "origin",
            "destination",
            "origin_time",
            "destination_time",
        }
        if correction_scope not in allowed_scopes:
            raise CaseStudyConfigError(
                f"model.{component_name}.correction_scope must be one of "
                f"{sorted(allowed_scopes)}."
            )
        transformation = _text(
            component["transformation"],
            f"model.{component_name}.transformation",
        )
        if transformation not in {"log", "log_multiplier", "additive"}:
            raise CaseStudyConfigError(
                f"model.{component_name}.transformation must be log, "
                "log_multiplier, or additive."
            )
        constraint = _text(
            component["constraint"], f"model.{component_name}.constraint"
        )
        if constraint not in {"sum_to_zero", "reference", "none"}:
            raise CaseStudyConfigError(
                f"model.{component_name}.constraint must be sum_to_zero, "
                "reference, or none."
            )
        regularization = _text(
            component["regularization"],
            f"model.{component_name}.regularization",
        )
        if regularization not in {"gaussian", "ridge", "none"}:
            raise CaseStudyConfigError(
                f"model.{component_name}.regularization must be gaussian, ridge, or none."
            )
        if mode == "estimated" and constraint == "none":
            raise CaseStudyConfigError(
                f"model.{component_name} estimated corrections require an "
                "identifiability constraint."
            )
        if mode == "estimated" and regularization == "none":
            raise CaseStudyConfigError(
                f"model.{component_name} estimated corrections require explicit regularization."
            )
        input_file = component.get("input_file")
        if mode != "provided" and input_file is not None:
            raise CaseStudyConfigError(
                f"model.{component_name}.input_file is only allowed when mode='provided'."
            )
        parsed[component_name] = {
            **{str(key): value for key, value in component.items()},
            "input_file": None if input_file is None else str((path.parent / _text(input_file, f"model.{component_name}.input_file")).resolve()),
        }
    if production_mode is None and "production" not in parsed:
        raise CaseStudyConfigError(
            "model must declare production_mode or a [production] component specification."
        )
    production_spec = parsed.get("production")
    attraction_spec = parsed.get("destination_attractiveness")
    if (
        isinstance(production_spec, Mapping)
        and isinstance(attraction_spec, Mapping)
        and production_spec.get("mode") == "estimated"
        and attraction_spec.get("mode") == "estimated"
        and production_spec.get("correction_scope") == "global"
        and attraction_spec.get("correction_scope") == "global"
    ):
        raise CaseStudyConfigError(
            "global production and global destination-attractiveness corrections "
            "are confounded; constrain one component or use grouped corrections."
        )
    return parsed


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
            "max_od_cells",
            "horizon_start",
            "horizon_end",
            "candidate",
        },
        optional={"num_od_pairs"},
    )
    if _positive_int(root["schema_version"], "time_discretization_file.schema_version") != 1:
        raise CaseStudyConfigError("time_discretization_file.schema_version must be 1.")
    for name in (
        "required_when_timestamps_exist",
        "base_resolution_minutes",
        "min_bin_minutes",
        "max_bin_minutes",
        "max_bins",
        "max_od_cells",
        "horizon_start",
        "horizon_end",
        "candidate",
    ):
        if name not in root:
            continue
        if root[name] != values.get(name, 0):
            raise CaseStudyConfigError(
                f"time-discretization value {name!r} disagrees between case.toml and {path}."
            )


def load_case_study_config(path: str | Path, *, case_root: str | Path | None = None) -> CaseStudyConfig:
    """Load and validate ``config/case.toml`` and all referenced contracts."""
    source = Path(path).expanduser().resolve()
    root_dir = Path(case_root).expanduser().resolve() if case_root is not None else source.parent.parent
    root = _load_toml(source)
    _keys(
        root,
        name="configuration",
        required={"schema_version", "case", "paths", "observations", "time_discretization", "structural_zeros", "model"},
        optional={"sampling", "od_universe", "prior_demand"},
    )
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
    _keys(
        paths,
        name="paths",
        required={"scenario_directory", "measurements", "results_directory"},
        optional={
            "candidate_demand",
            "od_pairs",
            "prior_demand",
            "fixed_demand",
            "production_inputs",
            "destination_attractiveness",
        },
    )
    resolved_paths = CasePaths(
        scenario_directory=_resolved_path(paths["scenario_directory"], "paths.scenario_directory", root_dir),
        measurements=_resolved_path(paths["measurements"], "paths.measurements", root_dir),
        candidate_demand=_resolved_path(paths.get("candidate_demand"), "paths.candidate_demand", root_dir, optional=True),
        od_pairs=_resolved_path(paths.get("od_pairs"), "paths.od_pairs", root_dir, optional=True),
        prior_demand=_resolved_path(paths.get("prior_demand"), "paths.prior_demand", root_dir, optional=True),
        fixed_demand=_resolved_path(paths.get("fixed_demand"), "paths.fixed_demand", root_dir, optional=True),
        production_inputs=_resolved_path(paths.get("production_inputs"), "paths.production_inputs", root_dir, optional=True),
        destination_attractiveness=_resolved_path(paths.get("destination_attractiveness"), "paths.destination_attractiveness", root_dir, optional=True),
        results_directory=_resolved_path(paths["results_directory"], "paths.results_directory", root_dir),
    )

    od_root = root.get(
        "od_universe",
        {
            "source": "legacy_time_dependent_demand",
            "level": "stop",
            "include_same_stop": False,
            "active_service_only": False,
            "connectivity_policy": "none",
        },
    )
    if not isinstance(od_root, Mapping):
        raise CaseStudyConfigError("od_universe must be a TOML table.")
    _keys(
        od_root,
        name="od_universe",
        required={"source", "level", "include_same_stop", "active_service_only", "connectivity_policy"},
        optional={"pair_file"},
    )
    od_source = _text(od_root["source"], "od_universe.source")
    if od_source not in {"file", "network_ordered_pairs", "legacy_time_dependent_demand"}:
        raise CaseStudyConfigError(
            "od_universe.source must be file, network_ordered_pairs, or legacy_time_dependent_demand."
        )
    od_level = _text(od_root["level"], "od_universe.level")
    if od_level not in {"stop", "physical_stop"}:
        raise CaseStudyConfigError("od_universe.level must be stop or physical_stop.")
    od_connectivity = _text(od_root["connectivity_policy"], "od_universe.connectivity_policy")
    if od_connectivity not in {"none", "directed_reachable"}:
        raise CaseStudyConfigError("od_universe.connectivity_policy must be none or directed_reachable.")
    od_pair_file = _resolved_path(
        od_root.get("pair_file"),
        "od_universe.pair_file",
        root_dir,
        optional=True,
    )
    if od_source == "file" and od_pair_file is None and resolved_paths.od_pairs is None:
        raise CaseStudyConfigError("od_universe.source='file' requires od_universe.pair_file or paths.od_pairs.")
    if od_source == "file" and od_pair_file is None:
        od_pair_file = resolved_paths.od_pairs
    if od_source != "file" and od_pair_file is not None:
        raise CaseStudyConfigError("od_universe.pair_file is only allowed for source='file'.")
    if od_source == "legacy_time_dependent_demand" and resolved_paths.candidate_demand is None:
        raise CaseStudyConfigError(
            "legacy_time_dependent_demand requires paths.candidate_demand."
        )
    od_universe = ODUniverseSettings(
        source=od_source,
        level=od_level,
        include_same_stop=_bool(od_root["include_same_stop"], "od_universe.include_same_stop"),
        active_service_only=_bool(od_root["active_service_only"], "od_universe.active_service_only"),
        connectivity_policy=od_connectivity,
        pair_file=od_pair_file,
    )

    prior_root = root.get(
        "prior_demand",
        {
            "source": "legacy_time_dependent_demand",
            "semantics": "legacy_time_dependent_demand",
            "expansion": "legacy",
        },
    )
    if not isinstance(prior_root, Mapping):
        raise CaseStudyConfigError("prior_demand must be a TOML table.")
    _keys(
        prior_root,
        name="prior_demand",
        required={"source", "semantics", "expansion"},
        optional={"value", "input_file"},
    )
    prior_source = _text(prior_root["source"], "prior_demand.source")
    if prior_source not in {"all_ones", "external_file", "legacy_time_dependent_demand"}:
        raise CaseStudyConfigError(
            "prior_demand.source must be all_ones, external_file, or legacy_time_dependent_demand."
        )
    prior_value = None if "value" not in prior_root else _finite(prior_root["value"], "prior_demand.value")
    if prior_source == "all_ones" and (prior_value is None or prior_value <= 0.0):
        raise CaseStudyConfigError("prior_demand.value must be positive for all_ones.")
    prior_file = _resolved_path(
        prior_root.get("input_file", resolved_paths.prior_demand),
        "prior_demand.input_file",
        root_dir,
        optional=True,
    )
    if prior_source == "external_file" and prior_file is None:
        raise CaseStudyConfigError("prior_demand.source='external_file' requires input_file or paths.prior_demand.")
    if prior_source != "external_file" and prior_file is not None:
        raise CaseStudyConfigError("prior_demand.input_file is only allowed for source='external_file'.")
    prior_demand = PriorDemandSettings(
        source=prior_source,
        value=prior_value,
        semantics=_text(prior_root["semantics"], "prior_demand.semantics"),
        expansion=_text(prior_root["expansion"], "prior_demand.expansion"),
        input_file=prior_file,
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
    _keys(td, name="time_discretization", required={"required_when_timestamps_exist", "base_resolution_minutes", "min_bin_minutes", "max_bin_minutes", "max_bins", "max_od_cells", "horizon_start", "horizon_end", "candidate"}, optional={"configuration_file", "num_od_pairs"})
    num_pairs = _positive_int(td.get("num_od_pairs", 0), "time_discretization.num_od_pairs", allow_zero=True) or None
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
    for component_name, fallback_path in (
        ("production", resolved_paths.production_inputs),
        ("destination_attractiveness", resolved_paths.destination_attractiveness),
    ):
        component = model.get(component_name)
        if not isinstance(component, dict):
            continue
        if component.get("mode") == "provided" and component.get("input_file") is None:
            if fallback_path is None:
                raise CaseStudyConfigError(
                    f"model.{component_name} provided mode requires input_file "
                    f"or paths.{component_name if component_name == 'production' else 'destination_attractiveness'}."
                )
            component["input_file"] = str(fallback_path)
    if od_source != "legacy_time_dependent_demand":
        missing_components = [
            name
            for name in ("production", "destination_attractiveness")
            if name not in model
        ]
        if missing_components:
            raise CaseStudyConfigError(
                "independent OD workflow requires explicit model component tables; "
                f"missing {missing_components}."
            )
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
    legacy_production_mode = model.get("production_mode")
    if legacy_production_mode is not None and legacy_production_mode != reduced_config.productions.mode:
        raise CaseStudyConfigError(
            "model.production_mode must agree with the reduced-OD production mode."
        )
    component_production = model.get("production")
    if isinstance(component_production, Mapping):
        effective_mode = (
            "estimated_basis"
            if component_production.get("mode") == "estimated"
            else "provided"
        )
        if effective_mode != reduced_config.productions.mode:
            raise CaseStudyConfigError(
                "model.production component mode must agree with "
                "reduced-OD productions.mode."
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
        od_universe,
        prior_demand,
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
