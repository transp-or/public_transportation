"""Strict, versioned TOML configuration for structural-zero preprocessing."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import StructuralZeroConfigError


SUPPORTED_VERSION = 1
RULE_NAMES = (
    "same_stop",
    "no_feasible_path",
    "maximum_transfers",
    "maximum_initial_wait",
    "maximum_journey_time",
    "minimum_feasible_departures",
)


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    folder: Path
    demand_file: Path | None = None


@dataclass(frozen=True, slots=True)
class OutputConfig:
    folder: Path
    include_retained_cells_in_report: bool = True


@dataclass(frozen=True, slots=True)
class EnabledRulesConfig:
    same_stop: bool
    no_feasible_path: bool
    maximum_transfers: bool
    maximum_initial_wait: bool
    maximum_journey_time: bool
    minimum_feasible_departures: bool

    def is_enabled(self, name: str) -> bool:
        if name not in RULE_NAMES:
            raise KeyError(name)
        return bool(getattr(self, name))


@dataclass(frozen=True, slots=True)
class SameStopRuleConfig:
    pass


@dataclass(frozen=True, slots=True)
class NoFeasiblePathRuleConfig:
    pass


@dataclass(frozen=True, slots=True)
class MaximumTransfersRuleConfig:
    max_transfers: int


@dataclass(frozen=True, slots=True)
class MaximumInitialWaitRuleConfig:
    max_initial_wait_minutes: float


@dataclass(frozen=True, slots=True)
class MaximumJourneyTimeRuleConfig:
    max_journey_time_minutes: float


@dataclass(frozen=True, slots=True)
class MinimumFeasibleDeparturesRuleConfig:
    min_feasible_departures: int


@dataclass(frozen=True, slots=True)
class RulesConfig:
    enabled: EnabledRulesConfig
    same_stop: SameStopRuleConfig | None
    no_feasible_path: NoFeasiblePathRuleConfig | None
    maximum_transfers: MaximumTransfersRuleConfig | None
    maximum_initial_wait: MaximumInitialWaitRuleConfig | None
    maximum_journey_time: MaximumJourneyTimeRuleConfig | None
    minimum_feasible_departures: MinimumFeasibleDeparturesRuleConfig | None


@dataclass(frozen=True, slots=True)
class StructuralZeroAssignmentConfig:
    max_access_deviation_minutes: float = 15.0
    max_transfer_wait_minutes: float = 30.0
    minimum_dwell_seconds: int = 1


@dataclass(frozen=True, slots=True)
class ExistingFixedDemandConfig:
    file: Path


@dataclass(frozen=True, slots=True)
class StructuralZeroConfig:
    """Fully validated configuration with absolute resolved paths."""

    version: int
    source_file: Path
    scenario: ScenarioConfig
    output: OutputConfig
    rules: RulesConfig
    assignment: StructuralZeroAssignmentConfig
    existing_fixed_demand: ExistingFixedDemandConfig | None

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

    def to_resolved_toml(self) -> str:
        """Serialize all effective values to deterministic TOML."""
        lines = [f"version = {self.version}", ""]
        lines.extend(
            ("[scenario]", f"folder = {_toml_string(str(self.scenario.folder))}")
        )
        if self.scenario.demand_file is not None:
            lines.append(
                f"demand_file = {_toml_string(str(self.scenario.demand_file))}"
            )
        lines.append("")
        lines.extend(
            (
                "[output]",
                f"folder = {_toml_string(str(self.output.folder))}",
                "include_retained_cells_in_report = "
                f"{_toml_bool(self.output.include_retained_cells_in_report)}",
                "",
                "[rules.enabled]",
            )
        )
        for name in RULE_NAMES:
            lines.append(f"{name} = {_toml_bool(self.rules.enabled.is_enabled(name))}")
        lines.append("")
        _append_rule_sections(lines, self.rules)
        lines.extend(
            (
                "[assignment]",
                f"max_access_deviation_minutes = {self.assignment.max_access_deviation_minutes}",
                f"max_transfer_wait_minutes = {self.assignment.max_transfer_wait_minutes}",
                f"minimum_dwell_seconds = {self.assignment.minimum_dwell_seconds}",
                "",
            )
        )
        if self.existing_fixed_demand is not None:
            lines.extend(
                (
                    "[existing_fixed_demand]",
                    f"file = {_toml_string(str(self.existing_fixed_demand.file))}",
                    "",
                )
            )
        return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _append_rule_sections(lines: list[str], rules: RulesConfig) -> None:
    for name in RULE_NAMES:
        section = getattr(rules, name)
        if section is None:
            continue
        lines.append(f"[rules.{name}]")
        if isinstance(section, MaximumTransfersRuleConfig):
            lines.append(f"max_transfers = {section.max_transfers}")
        elif isinstance(section, MaximumInitialWaitRuleConfig):
            lines.append(
                f"max_initial_wait_minutes = {section.max_initial_wait_minutes}"
            )
        elif isinstance(section, MaximumJourneyTimeRuleConfig):
            lines.append(
                f"max_journey_time_minutes = {section.max_journey_time_minutes}"
            )
        elif isinstance(section, MinimumFeasibleDeparturesRuleConfig):
            lines.append(f"min_feasible_departures = {section.min_feasible_departures}")
        lines.append("")


def _table(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise StructuralZeroConfigError(f"{location} must be a TOML table.")
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
        raise StructuralZeroConfigError(
            f"{location} contains unknown parameters: {unknown}."
        )
    if missing:
        raise StructuralZeroConfigError(
            f"{location} is missing required parameters: {missing}."
        )


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise StructuralZeroConfigError(f"{location} must be true or false.")
    return value


def _integer(value: Any, location: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StructuralZeroConfigError(f"{location} must be an integer.")
    if value < minimum:
        raise StructuralZeroConfigError(
            f"{location} must be at least {minimum}, got {value}."
        )
    return value


def _number(
    value: Any,
    location: str,
    *,
    minimum: float,
    strict: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuralZeroConfigError(f"{location} must be a number.")
    parsed = float(value)
    valid = parsed > minimum if strict else parsed >= minimum
    if not math.isfinite(parsed) or not valid:
        operator = "greater than" if strict else "at least"
        raise StructuralZeroConfigError(
            f"{location} must be finite and {operator} {minimum}, got {value!r}."
        )
    return parsed


def _path(value: Any, location: str, *, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise StructuralZeroConfigError(f"{location} must be a non-empty path string.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _rule_table(
    rules_table: Mapping[str, Any],
    *,
    name: str,
    enabled: bool,
) -> Mapping[str, Any] | None:
    value = rules_table.get(name)
    if value is None:
        if enabled:
            raise StructuralZeroConfigError(
                f"rules.{name} must be present when rules.enabled.{name} is true."
            )
        return None
    return _table(value, f"rules.{name}")


def load_structural_zero_config(path: str | Path) -> StructuralZeroConfig:
    """Load a strict configuration and resolve paths relative to its TOML file."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise StructuralZeroConfigError(f"Configuration file does not exist: {source}")
    try:
        with source.open("rb") as stream:
            root = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise StructuralZeroConfigError(f"Invalid TOML in {source}: {error}") from error

    _check_keys(
        root,
        location="configuration",
        required={"version", "scenario", "output", "rules", "assignment"},
        optional={"existing_fixed_demand"},
    )
    version = _integer(root["version"], "version", minimum=1)
    if version != SUPPORTED_VERSION:
        raise StructuralZeroConfigError(
            f"Unsupported configuration version {version}; supported version is {SUPPORTED_VERSION}."
        )
    base = source.parent

    scenario_table = _table(root["scenario"], "scenario")
    _check_keys(
        scenario_table,
        location="scenario",
        required={"folder"},
        optional={"demand_file"},
    )
    scenario_folder = _path(scenario_table["folder"], "scenario.folder", base=base)
    if not scenario_folder.is_dir():
        raise StructuralZeroConfigError(
            f"scenario.folder must be an existing directory: {scenario_folder}"
        )
    demand_file = None
    if "demand_file" in scenario_table:
        demand_file = _path(
            scenario_table["demand_file"], "scenario.demand_file", base=base
        )
        if not demand_file.is_file():
            raise StructuralZeroConfigError(
                f"scenario.demand_file must be an existing file: {demand_file}"
            )

    output_table = _table(root["output"], "output")
    _check_keys(
        output_table,
        location="output",
        required={"folder"},
        optional={"include_retained_cells_in_report"},
    )
    output_folder = _path(output_table["folder"], "output.folder", base=base)
    if output_folder == scenario_folder:
        raise StructuralZeroConfigError(
            "output.folder must differ from scenario.folder."
        )
    include_retained = _boolean(
        output_table.get("include_retained_cells_in_report", True),
        "output.include_retained_cells_in_report",
    )

    rules_table = _table(root["rules"], "rules")
    _check_keys(
        rules_table,
        location="rules",
        required={"enabled"},
        optional=set(RULE_NAMES),
    )
    enabled_table = _table(rules_table["enabled"], "rules.enabled")
    _check_keys(enabled_table, location="rules.enabled", required=set(RULE_NAMES))
    enabled = EnabledRulesConfig(
        **{
            name: _boolean(enabled_table[name], f"rules.enabled.{name}")
            for name in RULE_NAMES
        }
    )
    parsed_rules = _parse_rules(rules_table, enabled)

    assignment_table = _table(root["assignment"], "assignment")
    _check_keys(
        assignment_table,
        location="assignment",
        required=set(),
        optional={
            "max_access_deviation_minutes",
            "max_transfer_wait_minutes",
            "minimum_dwell_seconds",
        },
    )
    assignment = StructuralZeroAssignmentConfig(
        max_access_deviation_minutes=_number(
            assignment_table.get("max_access_deviation_minutes", 15.0),
            "assignment.max_access_deviation_minutes",
            minimum=0.0,
        ),
        max_transfer_wait_minutes=_number(
            assignment_table.get("max_transfer_wait_minutes", 30.0),
            "assignment.max_transfer_wait_minutes",
            minimum=0.0,
        ),
        minimum_dwell_seconds=_integer(
            assignment_table.get("minimum_dwell_seconds", 1),
            "assignment.minimum_dwell_seconds",
            minimum=1,
        ),
    )

    existing = None
    if "existing_fixed_demand" in root:
        existing_table = _table(root["existing_fixed_demand"], "existing_fixed_demand")
        _check_keys(existing_table, location="existing_fixed_demand", required={"file"})
        fixed_file = _path(
            existing_table["file"], "existing_fixed_demand.file", base=base
        )
        if not fixed_file.is_file():
            raise StructuralZeroConfigError(
                f"existing_fixed_demand.file must exist: {fixed_file}"
            )
        existing = ExistingFixedDemandConfig(file=fixed_file)

    return StructuralZeroConfig(
        version=version,
        source_file=source,
        scenario=ScenarioConfig(folder=scenario_folder, demand_file=demand_file),
        output=OutputConfig(
            folder=output_folder,
            include_retained_cells_in_report=include_retained,
        ),
        rules=parsed_rules,
        assignment=assignment,
        existing_fixed_demand=existing,
    )


def _parse_rules(
    rules_table: Mapping[str, Any], enabled: EnabledRulesConfig
) -> RulesConfig:
    same = _rule_table(rules_table, name="same_stop", enabled=enabled.same_stop)
    if same is not None:
        _check_keys(same, location="rules.same_stop", required=set())
    no_path = _rule_table(
        rules_table, name="no_feasible_path", enabled=enabled.no_feasible_path
    )
    if no_path is not None:
        _check_keys(no_path, location="rules.no_feasible_path", required=set())

    transfers = _rule_table(
        rules_table, name="maximum_transfers", enabled=enabled.maximum_transfers
    )
    if transfers is not None:
        _check_keys(
            transfers,
            location="rules.maximum_transfers",
            required={"max_transfers"},
        )
    initial_wait = _rule_table(
        rules_table,
        name="maximum_initial_wait",
        enabled=enabled.maximum_initial_wait,
    )
    if initial_wait is not None:
        _check_keys(
            initial_wait,
            location="rules.maximum_initial_wait",
            required={"max_initial_wait_minutes"},
        )
    journey = _rule_table(
        rules_table,
        name="maximum_journey_time",
        enabled=enabled.maximum_journey_time,
    )
    if journey is not None:
        _check_keys(
            journey,
            location="rules.maximum_journey_time",
            required={"max_journey_time_minutes"},
        )
    departures = _rule_table(
        rules_table,
        name="minimum_feasible_departures",
        enabled=enabled.minimum_feasible_departures,
    )
    if departures is not None:
        _check_keys(
            departures,
            location="rules.minimum_feasible_departures",
            required={"min_feasible_departures"},
        )

    return RulesConfig(
        enabled=enabled,
        same_stop=(None if same is None else SameStopRuleConfig()),
        no_feasible_path=(None if no_path is None else NoFeasiblePathRuleConfig()),
        maximum_transfers=(
            None
            if transfers is None
            else MaximumTransfersRuleConfig(
                max_transfers=_integer(
                    transfers["max_transfers"],
                    "rules.maximum_transfers.max_transfers",
                    minimum=0,
                )
            )
        ),
        maximum_initial_wait=(
            None
            if initial_wait is None
            else MaximumInitialWaitRuleConfig(
                max_initial_wait_minutes=_number(
                    initial_wait["max_initial_wait_minutes"],
                    "rules.maximum_initial_wait.max_initial_wait_minutes",
                    minimum=0.0,
                )
            )
        ),
        maximum_journey_time=(
            None
            if journey is None
            else MaximumJourneyTimeRuleConfig(
                max_journey_time_minutes=_number(
                    journey["max_journey_time_minutes"],
                    "rules.maximum_journey_time.max_journey_time_minutes",
                    minimum=0.0,
                    strict=True,
                )
            )
        ),
        minimum_feasible_departures=(
            None
            if departures is None
            else MinimumFeasibleDeparturesRuleConfig(
                min_feasible_departures=_integer(
                    departures["min_feasible_departures"],
                    "rules.minimum_feasible_departures.min_feasible_departures",
                    minimum=1,
                )
            )
        ),
    )
