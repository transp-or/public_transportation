"""Declarative Phase-1 gravity-model complexity contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from public_transportation.inference.block_coordinate._canonical import (
    canonical_json,
    fingerprint,
)


class GravityEffectScope(str, Enum):
    NONE = "none"
    GLOBAL = "global"
    ORIGIN_ZONE = "origin_zone"
    DESTINATION_ZONE = "destination_zone"
    ZONE_PAIR = "zone_pair"
    TIME_PERIOD = "time_period"


@dataclass(frozen=True, slots=True)
class GravityModelSpecification:
    """Minimal model now, with explicit stable slots for later relaxations."""

    origin_total_correction_scope: GravityEffectScope = GravityEffectScope.NONE
    destination_attractiveness_scope: GravityEffectScope = GravityEffectScope.NONE
    journey_time_scope: GravityEffectScope = GravityEffectScope.GLOBAL
    transfer_scope: GravityEffectScope = GravityEffectScope.GLOBAL
    waiting_time_scope: GravityEffectScope = GravityEffectScope.NONE
    temporal_basis_scope: GravityEffectScope = GravityEffectScope.NONE
    dispersion_scope: GravityEffectScope = GravityEffectScope.GLOBAL
    residual_demand_scope: GravityEffectScope = GravityEffectScope.NONE
    estimate_global_production_correction: bool = False
    destination_zone_count: int = 0
    time_period_count: int = 0
    origin_zone_count: int = 0
    destination_zone_ridge: float = 1.0
    time_period_ridge: float = 1.0
    origin_zone_ridge: float = 1.0
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version not in (1, 2):
            raise ValueError("unsupported gravity specification schema version.")
        expected = {
            "journey_time_scope": GravityEffectScope.GLOBAL,
            "transfer_scope": GravityEffectScope.GLOBAL,
            "waiting_time_scope": GravityEffectScope.NONE,
            "dispersion_scope": GravityEffectScope.GLOBAL,
            "residual_demand_scope": GravityEffectScope.NONE,
        }
        for name, required in expected.items():
            if getattr(self, name) is not required:
                raise NotImplementedError(
                    f"unsupported gravity scope {name}={getattr(self, name).value!r}."
                )
        allowed = {
            "origin_total_correction_scope": (GravityEffectScope.NONE, GravityEffectScope.ORIGIN_ZONE),
            "destination_attractiveness_scope": (GravityEffectScope.NONE, GravityEffectScope.DESTINATION_ZONE),
            "temporal_basis_scope": (GravityEffectScope.NONE, GravityEffectScope.TIME_PERIOD),
        }
        for name, choices in allowed.items():
            if getattr(self, name) not in choices:
                raise NotImplementedError(f"unsupported gravity scope {name}={getattr(self, name).value!r}.")
        blocks = (
            ("destination_attractiveness_scope", "destination_zone_count", "destination_zone_ridge"),
            ("temporal_basis_scope", "time_period_count", "time_period_ridge"),
            ("origin_total_correction_scope", "origin_zone_count", "origin_zone_ridge"),
        )
        for scope_name, count_name, ridge_name in blocks:
            active = getattr(self, scope_name) is not GravityEffectScope.NONE
            count = getattr(self, count_name)
            ridge = getattr(self, ridge_name)
            if count < 0 or (active and count < 2) or (not active and count != 0):
                raise ValueError(f"{count_name} must be at least two exactly when {scope_name} is active.")
            if not isinstance(ridge, (int, float)) or not float("-inf") < ridge < float("inf") or ridge < 0:
                raise ValueError(f"{ridge_name} must be finite and non-negative.")
        if self.estimate_global_production_correction:
            raise NotImplementedError(
                "global production correction is reserved for the estimator phase."
            )

    @property
    def parameter_count(self) -> int:
        return 3 + sum(max(value - 1, 0) for value in (
            self.destination_zone_count, self.time_period_count, self.origin_zone_count
        ))

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names = ["beta_time", "beta_transfer", "dispersion"]
        for prefix, count in (
            ("destination_zone_deviation", self.destination_zone_count),
            ("time_period_deviation", self.time_period_count),
            ("origin_zone_deviation", self.origin_zone_count),
        ):
            names.extend(f"{prefix}[{index}]" for index in range(max(count - 1, 0)))
        return tuple(names)

    @property
    def canonical_json(self) -> str:
        return canonical_json(asdict(self))

    @property
    def fingerprint(self) -> str:
        return fingerprint(asdict(self))

    def to_dict(self) -> dict[str, object]:
        return {
            name: value.value if isinstance(value, GravityEffectScope) else value
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GravityModelSpecification:
        scope_names = {
            "origin_total_correction_scope",
            "destination_attractiveness_scope",
            "journey_time_scope",
            "transfer_scope",
            "waiting_time_scope",
            "temporal_basis_scope",
            "dispersion_scope",
            "residual_demand_scope",
        }
        values = dict(payload)
        if values.get("schema_version") == 1:
            values["schema_version"] = 2
        for name in scope_names:
            values[name] = GravityEffectScope(str(values[name]))
        return cls(**values)  # type: ignore[arg-type]
