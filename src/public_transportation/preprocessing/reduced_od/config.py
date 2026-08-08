"""Strict, versioned configuration for reduced-dimensional OD preprocessing."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


REDUCED_OD_CONFIG_SCHEMA_VERSION = 2

MeasurementKind = Literal["boarding", "alighting"]
ProductionMode = Literal["provided", "estimated_basis"]
ProductionBasis = Literal["origin_period"]
ProductionSemantics = Literal[
    "external_journey_productions",
    "transfer_adjusted_journey_productions",
    "estimated_production_basis",
    "route_leg_baseline",
]
StopMappingPolicy = Literal["identity", "authoritative", "reviewed_generated"]
OutputSpatialLevel = Literal["scenario_stop", "physical_stop"]


class ReducedODConfigError(ValueError):
    """The reduced-OD TOML configuration is invalid or unsupported."""


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    service_day: str
    analysis_start_seconds: int
    analysis_end_seconds: int
    after_midnight_convention: Literal["service_day_extended"]
    apc_policy_identifier: str
    sensor_coverage_policy: str
    sensor_outage_policy: str
    unit: Literal["timetable_event"]
    accepted_types: tuple[MeasurementKind, ...]
    missing_policy: Literal["exclude"]
    duplicate_policy: Literal["error"]
    ambiguous_event_policy: Literal["error"]
    cleaning_stage: Literal["external"]


@dataclass(frozen=True, slots=True)
class JourneyConfig:
    origin_semantics: Literal["first_boarding"]
    destination_semantics: Literal["final_alighting"]
    time_bin_membership: Literal["half_open"]
    maximum_transfers: int
    maximum_waiting_seconds: int
    maximum_journey_seconds: int
    maximum_alternatives_per_cell: int
    transfer_footpath_policy: str
    route_shares: Literal["fixed_within_fit"]


@dataclass(frozen=True, slots=True)
class ProductionConfig:
    mode: ProductionMode
    semantics: ProductionSemantics
    input_path: Path | None
    basis: ProductionBasis | None


@dataclass(frozen=True, slots=True)
class StopConfig:
    mapping_policy: StopMappingPolicy
    physical_stop_mapping_path: Path | None
    footpaths_path: Path | None


@dataclass(frozen=True, slots=True)
class ReducedODOutputConfig:
    spatial_level: OutputSpatialLevel
    reconstruct_full_od: bool


@dataclass(frozen=True, slots=True)
class ReducedODValidationConfig:
    detailed_assignment: Literal["explicit_only"]


@dataclass(frozen=True, slots=True)
class ReducedODModelConfig:
    likelihood: Literal["poisson", "negative_binomial"]


@dataclass(frozen=True, slots=True)
class ReducedODConfig:
    """Fully validated semantic configuration with resolved optional paths."""

    schema_version: int
    source_file: Path
    observations: ObservationConfig
    journeys: JourneyConfig
    productions: ProductionConfig
    stops: StopConfig
    outputs: ReducedODOutputConfig
    model: ReducedODModelConfig
    validation: ReducedODValidationConfig

    @property
    def fingerprint_payload_json(self) -> str:
        payload = _jsonable(asdict(self))
        payload.pop("source_file", None)
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.fingerprint_payload_json.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _table(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ReducedODConfigError(f"{location} must be a TOML table.")
    return value


def _check_keys(
    table: Mapping[str, Any],
    *,
    location: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = set() if optional is None else optional
    unknown = sorted(set(table) - required - optional)
    missing = sorted(required - set(table))
    if unknown:
        raise ReducedODConfigError(
            f"{location} contains unknown parameters: {unknown}."
        )
    if missing:
        raise ReducedODConfigError(
            f"{location} is missing required parameters: {missing}."
        )


def _literal(value: Any, location: str, allowed: set[str]) -> Any:
    """Validate a string literal while preserving the caller's narrow type."""
    if not isinstance(value, str) or value not in allowed:
        raise ReducedODConfigError(
            f"{location} must be one of {sorted(allowed)}, got {value!r}."
        )
    return value


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ReducedODConfigError(f"{location} must be true or false.")
    return value


def _nonnegative_integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReducedODConfigError(f"{location} must be an integer.")
    if value < 0:
        raise ReducedODConfigError(f"{location} must be at least 0, got {value}.")
    return value


def _positive_integer(value: Any, location: str) -> int:
    parsed = _nonnegative_integer(value, location)
    if parsed == 0:
        raise ReducedODConfigError(f"{location} must be positive.")
    return parsed


def _identifier(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReducedODConfigError(f"{location} must be a non-empty string.")
    return value.strip()


def _optional_path(
    table: Mapping[str, Any],
    name: str,
    *,
    location: str,
    base: Path,
) -> Path | None:
    if name not in table:
        return None
    value = table[name]
    if not isinstance(value, str) or not value.strip():
        raise ReducedODConfigError(
            f"{location}.{name} must be a non-empty path string."
        )
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _accepted_types(value: Any) -> tuple[MeasurementKind, ...]:
    if not isinstance(value, list) or not value:
        raise ReducedODConfigError(
            "observations.accepted_types must be a non-empty array."
        )
    if any(not isinstance(item, str) for item in value):
        raise ReducedODConfigError(
            "observations.accepted_types must contain strings."
        )
    unknown = sorted(set(value) - {"boarding", "alighting"})
    if unknown:
        raise ReducedODConfigError(
            "observations.accepted_types contains unsupported values: "
            f"{unknown}."
        )
    if len(value) != len(set(value)):
        raise ReducedODConfigError(
            "observations.accepted_types must not contain duplicates."
        )
    return tuple(sorted(value))


def load_reduced_od_config(path: str | Path) -> ReducedODConfig:
    """Load the strict Phase-1 reduced-OD TOML configuration."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReducedODConfigError(f"Configuration file does not exist: {source}")
    try:
        with source.open("rb") as stream:
            root = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ReducedODConfigError(f"Invalid TOML in {source}: {error}") from error

    _check_keys(
        root,
        location="configuration",
        required={
            "schema_version",
            "observations",
            "journeys",
            "productions",
            "stops",
            "outputs",
            "model",
            "validation",
        },
    )
    version = _nonnegative_integer(root["schema_version"], "schema_version")
    if version != REDUCED_OD_CONFIG_SCHEMA_VERSION:
        raise ReducedODConfigError(
            f"Unsupported schema_version {version}; supported version is "
            f"{REDUCED_OD_CONFIG_SCHEMA_VERSION}."
        )
    base = source.parent

    observation_table = _table(root["observations"], "observations")
    _check_keys(
        observation_table,
        location="observations",
        required={
            "service_day",
            "analysis_start_seconds",
            "analysis_end_seconds",
            "after_midnight_convention",
            "apc_policy_identifier",
            "sensor_coverage_policy",
            "sensor_outage_policy",
            "unit",
            "accepted_types",
            "missing_policy",
            "duplicate_policy",
            "ambiguous_event_policy",
            "cleaning_stage",
        },
    )
    observations = ObservationConfig(
        service_day=_identifier(
            observation_table["service_day"], "observations.service_day"
        ),
        analysis_start_seconds=_nonnegative_integer(
            observation_table["analysis_start_seconds"],
            "observations.analysis_start_seconds",
        ),
        analysis_end_seconds=_positive_integer(
            observation_table["analysis_end_seconds"],
            "observations.analysis_end_seconds",
        ),
        after_midnight_convention=_literal(
            observation_table["after_midnight_convention"],
            "observations.after_midnight_convention",
            {"service_day_extended"},
        ),
        apc_policy_identifier=_identifier(
            observation_table["apc_policy_identifier"],
            "observations.apc_policy_identifier",
        ),
        sensor_coverage_policy=_identifier(
            observation_table["sensor_coverage_policy"],
            "observations.sensor_coverage_policy",
        ),
        sensor_outage_policy=_identifier(
            observation_table["sensor_outage_policy"],
            "observations.sensor_outage_policy",
        ),
        unit=_literal(
            observation_table["unit"], "observations.unit", {"timetable_event"}
        ),
        accepted_types=_accepted_types(observation_table["accepted_types"]),
        missing_policy=_literal(
            observation_table["missing_policy"],
            "observations.missing_policy",
            {"exclude"},
        ),
        duplicate_policy=_literal(
            observation_table["duplicate_policy"],
            "observations.duplicate_policy",
            {"error"},
        ),
        ambiguous_event_policy=_literal(
            observation_table["ambiguous_event_policy"],
            "observations.ambiguous_event_policy",
            {"error"},
        ),
        cleaning_stage=_literal(
            observation_table["cleaning_stage"],
            "observations.cleaning_stage",
            {"external"},
        ),
    )
    if observations.analysis_end_seconds <= observations.analysis_start_seconds:
        raise ReducedODConfigError(
            "observations.analysis_end_seconds must exceed analysis_start_seconds."
        )

    journey_table = _table(root["journeys"], "journeys")
    _check_keys(
        journey_table,
        location="journeys",
        required={
            "origin_semantics",
            "destination_semantics",
            "time_bin_membership",
            "maximum_transfers",
            "maximum_waiting_seconds",
            "maximum_journey_seconds",
            "maximum_alternatives_per_cell",
            "transfer_footpath_policy",
            "route_shares",
        },
    )
    journeys = JourneyConfig(
        origin_semantics=_literal(
            journey_table["origin_semantics"],
            "journeys.origin_semantics",
            {"first_boarding"},
        ),
        destination_semantics=_literal(
            journey_table["destination_semantics"],
            "journeys.destination_semantics",
            {"final_alighting"},
        ),
        time_bin_membership=_literal(
            journey_table["time_bin_membership"],
            "journeys.time_bin_membership",
            {"half_open"},
        ),
        maximum_transfers=_nonnegative_integer(
            journey_table["maximum_transfers"], "journeys.maximum_transfers"
        ),
        maximum_waiting_seconds=_positive_integer(
            journey_table["maximum_waiting_seconds"],
            "journeys.maximum_waiting_seconds",
        ),
        maximum_journey_seconds=_positive_integer(
            journey_table["maximum_journey_seconds"],
            "journeys.maximum_journey_seconds",
        ),
        maximum_alternatives_per_cell=_positive_integer(
            journey_table["maximum_alternatives_per_cell"],
            "journeys.maximum_alternatives_per_cell",
        ),
        transfer_footpath_policy=_identifier(
            journey_table["transfer_footpath_policy"],
            "journeys.transfer_footpath_policy",
        ),
        route_shares=_literal(
            journey_table["route_shares"],
            "journeys.route_shares",
            {"fixed_within_fit"},
        ),
    )

    production_table = _table(root["productions"], "productions")
    _check_keys(
        production_table,
        location="productions",
        required={"mode", "semantics"},
        optional={"input_path", "basis"},
    )
    production_mode = _literal(
        production_table["mode"],
        "productions.mode",
        {"provided", "estimated_basis"},
    )
    production_semantics = _literal(
        production_table["semantics"],
        "productions.semantics",
        {
            "external_journey_productions",
            "transfer_adjusted_journey_productions",
            "estimated_production_basis",
            "route_leg_baseline",
        },
    )
    input_path = _optional_path(
        production_table, "input_path", location="productions", base=base
    )
    basis_raw = production_table.get("basis")
    basis = (
        None
        if basis_raw is None
        else _literal(basis_raw, "productions.basis", {"origin_period"})
    )
    if production_mode == "provided":
        if production_semantics == "estimated_production_basis":
            raise ReducedODConfigError(
                "provided productions cannot use estimated_production_basis semantics."
            )
        if input_path is None:
            raise ReducedODConfigError(
                "productions.input_path is required when mode is 'provided'."
            )
        if basis is not None:
            raise ReducedODConfigError(
                "productions.basis is not allowed when mode is 'provided'."
            )
    else:
        if production_semantics != "estimated_production_basis":
            raise ReducedODConfigError(
                "estimated_basis mode requires estimated_production_basis semantics."
            )
        if input_path is not None:
            raise ReducedODConfigError(
                "productions.input_path is not allowed when mode is "
                "'estimated_basis'."
            )
        if basis is None:
            raise ReducedODConfigError(
                "productions.basis is required when mode is 'estimated_basis'."
            )
    productions = ProductionConfig(
        mode=production_mode,
        semantics=production_semantics,
        input_path=input_path,
        basis=basis,
    )

    stop_table = _table(root["stops"], "stops")
    _check_keys(
        stop_table,
        location="stops",
        required={"mapping_policy"},
        optional={"physical_stop_mapping_path", "footpaths_path"},
    )
    stops = StopConfig(
        mapping_policy=_literal(
            stop_table["mapping_policy"],
            "stops.mapping_policy",
            {"identity", "authoritative", "reviewed_generated"},
        ),
        physical_stop_mapping_path=_optional_path(
            stop_table,
            "physical_stop_mapping_path",
            location="stops",
            base=base,
        ),
        footpaths_path=_optional_path(
            stop_table, "footpaths_path", location="stops", base=base
        ),
    )

    output_table = _table(root["outputs"], "outputs")
    _check_keys(
        output_table,
        location="outputs",
        required={"spatial_level", "reconstruct_full_od"},
    )
    outputs = ReducedODOutputConfig(
        spatial_level=_literal(
            output_table["spatial_level"],
            "outputs.spatial_level",
            {"scenario_stop", "physical_stop"},
        ),
        reconstruct_full_od=_boolean(
            output_table["reconstruct_full_od"], "outputs.reconstruct_full_od"
        ),
    )

    model_table = _table(root["model"], "model")
    _check_keys(model_table, location="model", required={"likelihood"})
    model = ReducedODModelConfig(
        likelihood=_literal(
            model_table["likelihood"],
            "model.likelihood",
            {"poisson", "negative_binomial"},
        )
    )

    validation_table = _table(root["validation"], "validation")
    _check_keys(
        validation_table,
        location="validation",
        required={"detailed_assignment"},
    )
    validation = ReducedODValidationConfig(
        detailed_assignment=_literal(
            validation_table["detailed_assignment"],
            "validation.detailed_assignment",
            {"explicit_only"},
        )
    )

    return ReducedODConfig(
        schema_version=version,
        source_file=source,
        observations=observations,
        journeys=journeys,
        productions=productions,
        stops=stops,
        outputs=outputs,
        model=model,
        validation=validation,
    )
