"""Declarative, fingerprinted gravity-model specification contracts."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from enum import Enum
from typing import ClassVar

from public_transportation.inference.block_coordinate._canonical import (
    canonical_json,
    fingerprint,
)


class GravityEffectScope(str, Enum):
    """Spatial or temporal scope of one gravity-model component."""

    NONE = "none"
    FIXED = "fixed"
    GLOBAL = "global"
    ORIGIN = "origin"
    DESTINATION = "destination"
    TIME_PERIOD = "time_period"
    ORIGIN_TIME = "origin_time"
    DESTINATION_TIME = "destination_time"
    ORIGIN_ZONE = "origin_zone"
    DESTINATION_ZONE = "destination_zone"
    ZONE_PAIR = "zone_pair"
    CUSTOM_GROUP = "custom_group"
    SMOOTH_BASIS = "smooth_basis"


class GravityConstraint(str, Enum):
    """Identifying constraint applied to a categorical parameter block."""

    NONE = "none"
    SUM_ZERO = "sum_zero"
    REFERENCE = "reference"


class GravityParameterization(str, Enum):
    """Raw-to-physical transformation for a parameter block."""

    FIXED = "fixed"
    POSITIVE = "positive"
    LOG_MULTIPLIER = "log_multiplier"
    ADDITIVE = "additive"


class GravityRegularizationType(str, Enum):
    NONE = "none"
    RIDGE = "ridge"


_GROUP_SCOPES = frozenset(
    {
        GravityEffectScope.ORIGIN,
        GravityEffectScope.DESTINATION,
        GravityEffectScope.TIME_PERIOD,
        GravityEffectScope.ORIGIN_TIME,
        GravityEffectScope.DESTINATION_TIME,
        GravityEffectScope.ORIGIN_ZONE,
        GravityEffectScope.DESTINATION_ZONE,
        GravityEffectScope.ZONE_PAIR,
        GravityEffectScope.CUSTOM_GROUP,
    }
)
_BASIS_SCOPES = frozenset({GravityEffectScope.SMOOTH_BASIS})


@dataclass(frozen=True, slots=True)
class GravityRegularization:
    """Optional regularization attached to one explicit parameter block."""

    kind: GravityRegularizationType = GravityRegularizationType.NONE
    strength: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, GravityRegularizationType):
            object.__setattr__(self, "kind", GravityRegularizationType(self.kind))
        if not isinstance(self.strength, (int, float)) or not (
            float("-inf") < float(self.strength) < float("inf")
        ):
            raise ValueError("regularization strength must be finite.")
        if self.strength < 0:
            raise ValueError("regularization strength must be non-negative.")
        if self.kind is GravityRegularizationType.NONE and self.strength != 0:
            raise ValueError("regularization kind 'none' requires strength zero.")
        if self.kind is GravityRegularizationType.RIDGE and self.strength <= 0:
            raise ValueError("ridge regularization requires positive strength.")

    def to_dict(self) -> dict[str, object]:
        return {"type": self.kind.value, "strength": float(self.strength)}

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> GravityRegularization:
        if payload is None:
            return cls()
        return cls(
            kind=GravityRegularizationType(str(payload.get("type", "none"))),
            strength=float(payload.get("strength", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class GravityComponentSpecification:
    """Complete declarative contract for one model component."""

    name: str
    scope: GravityEffectScope
    parameterization: GravityParameterization
    grouping: str | None = None
    group_count: int = 0
    constraint: GravityConstraint = GravityConstraint.NONE
    reference_category: int | None = None
    regularization: GravityRegularization = GravityRegularization()
    fixed_value: float | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("gravity component name must be nonempty.")
        for field_name, enum_type in (
            ("scope", GravityEffectScope),
            ("parameterization", GravityParameterization),
            ("constraint", GravityConstraint),
        ):
            value = getattr(self, field_name)
            if not isinstance(value, enum_type):
                object.__setattr__(self, field_name, enum_type(value))
        if not isinstance(self.regularization, GravityRegularization):
            raise TypeError("regularization must be a GravityRegularization.")
        grouped = self.scope in _GROUP_SCOPES
        basis = self.scope in _BASIS_SCOPES
        if grouped:
            if self.group_count < 2:
                raise ValueError(
                    f"component {self.name!r} with scope {self.scope.value!r} "
                    "requires group_count >= 2."
                )
            if self.constraint is GravityConstraint.NONE:
                raise ValueError(
                    f"component {self.name!r} requires sum_zero or reference "
                    "constraint for grouped scope."
                )
            if self.scope is GravityEffectScope.CUSTOM_GROUP and not self.grouping:
                raise ValueError("custom_group scope requires an explicit grouping name.")
        elif basis:
            if self.group_count < 1 or not self.grouping:
                raise ValueError(
                    "smooth_basis scope requires a grouping name and at least one "
                    "basis column."
                )
            if self.constraint is not GravityConstraint.NONE:
                raise ValueError("smooth-basis coefficients cannot be centered.")
        elif self.group_count != 0:
            raise ValueError("group_count is valid only for grouped scopes.")
        if self.constraint is GravityConstraint.REFERENCE:
            if not grouped or self.reference_category is None:
                raise ValueError("reference constraint requires reference_category.")
            if not 0 <= self.reference_category < self.group_count:
                raise ValueError("reference_category is outside the declared groups.")
        elif self.reference_category is not None:
            raise ValueError("reference_category is valid only with reference constraint.")
        fixed = self.scope in (GravityEffectScope.NONE, GravityEffectScope.FIXED)
        if fixed and self.parameterization is not GravityParameterization.FIXED:
            raise ValueError("none/fixed scopes require fixed parameterization.")
        if not fixed and self.parameterization is GravityParameterization.FIXED:
            raise ValueError("estimated scopes cannot use fixed parameterization.")
        if self.fixed_value is not None and not fixed:
            raise ValueError("fixed_value is valid only for none/fixed scopes.")
        if self.fixed_value is not None and not (
            float("-inf") < float(self.fixed_value) < float("inf")
        ):
            raise ValueError("fixed_value must be finite.")
        if self.scope is GravityEffectScope.NONE and self.fixed_value not in (None, 0):
            raise ValueError("scope 'none' cannot carry a nonzero fixed value.")

    @property
    def grouped(self) -> bool:
        return self.scope in _GROUP_SCOPES

    @property
    def deviation_count(self) -> int:
        if not self.grouped:
            return 0
        return self.group_count - 1

    @property
    def parameter_count(self) -> int:
        if self.scope in (GravityEffectScope.NONE, GravityEffectScope.FIXED):
            return 0
        if self.scope is GravityEffectScope.GLOBAL:
            return 1
        if self.scope is GravityEffectScope.SMOOTH_BASIS:
            return self.group_count
        if self.parameterization is GravityParameterization.POSITIVE:
            return 1 + self.deviation_count
        return self.deviation_count

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scope": self.scope.value,
            "parameterization": self.parameterization.value,
            "grouping": self.grouping,
            "group_count": self.group_count,
            "constraint": self.constraint.value,
            "reference_category": self.reference_category,
            "regularization": self.regularization.to_dict(),
            "fixed_value": self.fixed_value,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GravityComponentSpecification:
        return cls(
            name=str(payload["name"]),
            scope=GravityEffectScope(str(payload["scope"])),
            parameterization=GravityParameterization(
                str(payload["parameterization"])
            ),
            grouping=(
                None if payload.get("grouping") is None else str(payload["grouping"])
            ),
            group_count=int(payload.get("group_count", 0)),
            constraint=GravityConstraint(str(payload.get("constraint", "none"))),
            reference_category=(
                None
                if payload.get("reference_category") is None
                else int(payload["reference_category"])
            ),
            regularization=GravityRegularization.from_dict(
                payload.get("regularization")  # type: ignore[arg-type]
            ),
            fixed_value=(
                None
                if payload.get("fixed_value") is None
                else float(payload["fixed_value"])
            ),
            source=None if payload.get("source") is None else str(payload["source"]),
        )


@dataclass(frozen=True, slots=True)
class GravityLikelihoodSpecification:
    family: str = "negative_binomial"
    calibration_mask: str = "supported_measurements"
    detection_rate_estimated: bool = False

    def __post_init__(self) -> None:
        if self.family not in ("negative_binomial", "poisson"):
            raise ValueError(f"unsupported gravity likelihood family {self.family!r}.")
        if self.calibration_mask not in (
            "supported_measurements",
            "all_measurements",
            "explicit",
        ):
            raise ValueError("unsupported gravity calibration-mask policy.")


@dataclass(frozen=True, slots=True)
class GravityTimeSpecification:
    units: str = "index"
    interpretation: str = "categorical departure-time bins"
    bin_labels: tuple[str, ...] = ()
    smooth_basis_name: str | None = None

    def __post_init__(self) -> None:
        if not self.units or not self.interpretation:
            raise ValueError("time units and interpretation must be nonempty.")
        if len(set(self.bin_labels)) != len(self.bin_labels):
            raise ValueError("time-bin labels must be unique.")


_COMPONENT_ORDER = (
    "journey_time",
    "transfer",
    "dispersion",
    "waiting_time",
    "production",
    "destination_attractiveness",
    "temporal",
    "residual_demand",
)


@dataclass(frozen=True, slots=True)
class GravityModelSpecification:
    """Backward-compatible gravity model plus explicit component overrides."""

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
    model_name: str = "minimal_three_parameter"
    components: tuple[GravityComponentSpecification, ...] = ()
    likelihood: GravityLikelihoodSpecification = GravityLikelihoodSpecification()
    time: GravityTimeSpecification = GravityTimeSpecification()
    schema_version: int = 3

    SUPPORTED_SCHEMA_VERSIONS: ClassVar[tuple[int, ...]] = (1, 2, 3)

    def __post_init__(self) -> None:
        if self.schema_version not in self.SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("unsupported gravity specification schema version.")
        if not self.model_name:
            raise ValueError("gravity model_name must be nonempty.")
        scope_fields = (
            "origin_total_correction_scope",
            "destination_attractiveness_scope",
            "journey_time_scope",
            "transfer_scope",
            "waiting_time_scope",
            "temporal_basis_scope",
            "dispersion_scope",
            "residual_demand_scope",
        )
        for name in scope_fields:
            value = getattr(self, name)
            if not isinstance(value, GravityEffectScope):
                object.__setattr__(self, name, GravityEffectScope(value))
        names = [item.name for item in self.components]
        if len(set(names)) != len(names):
            raise ValueError("gravity component overrides must have unique names.")
        unknown = set(names) - set(_COMPONENT_ORDER)
        if unknown:
            raise ValueError(f"unknown gravity components: {sorted(unknown)}.")
        if self.estimate_global_production_correction and (
            self.origin_total_correction_scope is not GravityEffectScope.NONE
        ):
            raise ValueError(
                "legacy global production correction cannot be combined with "
                "origin_total_correction_scope."
            )
        self._validate_legacy_blocks()
        self._validate_components()

    def _validate_legacy_blocks(self) -> None:
        allowed = {
            "origin_total_correction_scope": (
                GravityEffectScope.NONE,
                GravityEffectScope.ORIGIN_ZONE,
            ),
            "destination_attractiveness_scope": (
                GravityEffectScope.NONE,
                GravityEffectScope.DESTINATION_ZONE,
            ),
            "temporal_basis_scope": (
                GravityEffectScope.NONE,
                GravityEffectScope.TIME_PERIOD,
            ),
            "journey_time_scope": (GravityEffectScope.GLOBAL,),
            "transfer_scope": (GravityEffectScope.GLOBAL,),
            "waiting_time_scope": (GravityEffectScope.NONE,),
            "dispersion_scope": (GravityEffectScope.GLOBAL,),
            "residual_demand_scope": (GravityEffectScope.NONE,),
        }
        for name, choices in allowed.items():
            if getattr(self, name) not in choices and not self.components:
                raise ValueError(
                    f"legacy field {name} does not support scope "
                    f"{getattr(self, name).value!r}; use an explicit component."
                )
        blocks = (
            (
                "destination_attractiveness_scope",
                "destination_zone_count",
                "destination_zone_ridge",
            ),
            ("temporal_basis_scope", "time_period_count", "time_period_ridge"),
            (
                "origin_total_correction_scope",
                "origin_zone_count",
                "origin_zone_ridge",
            ),
        )
        for scope_name, count_name, ridge_name in blocks:
            active = getattr(self, scope_name) is not GravityEffectScope.NONE
            count = getattr(self, count_name)
            ridge = getattr(self, ridge_name)
            if count < 0 or (active and count < 2) or (not active and count != 0):
                raise ValueError(
                    f"{count_name} must be at least two exactly when "
                    f"{scope_name} is active."
                )
            if not isinstance(ridge, (int, float)) or not (
                float("-inf") < float(ridge) < float("inf")
            ) or ridge < 0:
                raise ValueError(f"{ridge_name} must be finite and non-negative.")

    def _validate_components(self) -> None:
        allowed = {
            "production": {
                GravityEffectScope.NONE,
                GravityEffectScope.FIXED,
                GravityEffectScope.GLOBAL,
                GravityEffectScope.ORIGIN,
                GravityEffectScope.TIME_PERIOD,
                GravityEffectScope.ORIGIN_TIME,
                GravityEffectScope.ORIGIN_ZONE,
                GravityEffectScope.CUSTOM_GROUP,
            },
            "destination_attractiveness": {
                GravityEffectScope.NONE,
                GravityEffectScope.FIXED,
                GravityEffectScope.GLOBAL,
                GravityEffectScope.DESTINATION,
                GravityEffectScope.TIME_PERIOD,
                GravityEffectScope.DESTINATION_TIME,
                GravityEffectScope.DESTINATION_ZONE,
                GravityEffectScope.CUSTOM_GROUP,
            },
            "journey_time": set(GravityEffectScope)
            - {GravityEffectScope.SMOOTH_BASIS},
            "transfer": set(GravityEffectScope)
            - {GravityEffectScope.SMOOTH_BASIS},
            "waiting_time": set(GravityEffectScope),
            "temporal": {
                GravityEffectScope.NONE,
                GravityEffectScope.FIXED,
                GravityEffectScope.GLOBAL,
                GravityEffectScope.TIME_PERIOD,
                GravityEffectScope.ORIGIN_TIME,
                GravityEffectScope.DESTINATION_TIME,
                GravityEffectScope.CUSTOM_GROUP,
                GravityEffectScope.SMOOTH_BASIS,
            },
            "dispersion": {
                GravityEffectScope.NONE,
                GravityEffectScope.FIXED,
                GravityEffectScope.GLOBAL,
            },
            "residual_demand": {
                GravityEffectScope.NONE,
            },
        }
        for component in self.components:
            if component.scope not in allowed[component.name]:
                raise ValueError(
                    f"unsupported gravity scope {component.name}="
                    f"{component.scope.value!r}."
                )
            expected_parameterization = {
                "production": GravityParameterization.LOG_MULTIPLIER,
                "destination_attractiveness": GravityParameterization.ADDITIVE,
                "journey_time": GravityParameterization.POSITIVE,
                "transfer": GravityParameterization.POSITIVE,
                "waiting_time": GravityParameterization.POSITIVE,
                "temporal": GravityParameterization.ADDITIVE,
                "dispersion": GravityParameterization.POSITIVE,
                "residual_demand": GravityParameterization.FIXED,
            }[component.name]
            if component.scope not in (
                GravityEffectScope.NONE,
                GravityEffectScope.FIXED,
            ) and component.parameterization is not expected_parameterization:
                raise ValueError(
                    f"component {component.name!r} requires parameterization "
                    f"{expected_parameterization.value!r}."
                )
            if component.name in ("production", "destination_attractiveness", "temporal"):
                if component.grouped and component.constraint is GravityConstraint.NONE:
                    raise ValueError(
                        f"component {component.name!r} must be normalized."
                    )
            if component.name == "production" and component.scope in (
                GravityEffectScope.ORIGIN_TIME,
                GravityEffectScope.CUSTOM_GROUP,
            ) and component.regularization.kind is not GravityRegularizationType.RIDGE:
                raise ValueError(
                    f"high-dimensional production scope {component.scope.value!r} "
                    "requires ridge regularization."
                )
            if component.name in (
                "journey_time",
                "transfer",
                "waiting_time",
                "dispersion",
            ) and component.scope is GravityEffectScope.FIXED:
                if component.fixed_value is None or component.fixed_value <= 0:
                    raise ValueError(
                        f"fixed {component.name!r} requires a strictly positive "
                        "fixed_value."
                    )
            if component.name == "destination_attractiveness" and component.source not in (
                "feature_cache",
                "external",
                "baseline_derived",
            ):
                raise ValueError(
                    "destination attractiveness requires source feature_cache, "
                    "external, or baseline_derived."
                )
            if component.name == "production" and component.source != "origin_time_totals":
                raise ValueError(
                    "production requires source 'origin_time_totals'."
                )
        temporal = self.component("temporal")
        if temporal.scope is GravityEffectScope.SMOOTH_BASIS:
            if self.time.smooth_basis_name is None:
                raise ValueError(
                    "smooth temporal effects require time.smooth_basis_name."
                )
            if temporal.grouping != self.time.smooth_basis_name:
                raise ValueError(
                    "temporal smooth-basis grouping and time.smooth_basis_name differ."
                )
        production = self.component("production")
        if (
            production.scope is GravityEffectScope.GLOBAL
            and self.likelihood.detection_rate_estimated
        ):
            raise ValueError(
                "global production scale is confounded with an estimated "
                "detection-rate scale."
            )
        if self.likelihood.detection_rate_estimated:
            raise ValueError(
                "an estimated detection-rate scale is not implemented by the "
                "gravity observation model."
            )
        dispersion = self.component("dispersion")
        if self.likelihood.family == "negative_binomial" and (
            dispersion.scope is GravityEffectScope.NONE
        ):
            raise ValueError(
                "negative-binomial likelihood requires global or fixed dispersion."
            )
        if self.likelihood.family == "poisson" and dispersion.parameter_count:
            raise ValueError(
                "Poisson likelihood cannot estimate an unused dispersion parameter."
            )

    def _legacy_component(self, name: str) -> GravityComponentSpecification:
        if name == "journey_time":
            return GravityComponentSpecification(
                name, self.journey_time_scope, GravityParameterization.POSITIVE
                if self.journey_time_scope not in (GravityEffectScope.NONE, GravityEffectScope.FIXED)
                else GravityParameterization.FIXED
            )
        if name == "transfer":
            return GravityComponentSpecification(
                name, self.transfer_scope, GravityParameterization.POSITIVE
                if self.transfer_scope not in (GravityEffectScope.NONE, GravityEffectScope.FIXED)
                else GravityParameterization.FIXED
            )
        if name == "dispersion":
            return GravityComponentSpecification(
                name,
                self.dispersion_scope,
                GravityParameterization.POSITIVE
                if self.dispersion_scope is GravityEffectScope.GLOBAL
                else GravityParameterization.FIXED,
            )
        if name == "waiting_time":
            return GravityComponentSpecification(
                name,
                self.waiting_time_scope,
                GravityParameterization.POSITIVE
                if self.waiting_time_scope not in (GravityEffectScope.NONE, GravityEffectScope.FIXED)
                else GravityParameterization.FIXED,
                fixed_value=0.0 if self.waiting_time_scope is GravityEffectScope.NONE else None,
            )
        if name == "production":
            if self.estimate_global_production_correction:
                return GravityComponentSpecification(
                    name,
                    GravityEffectScope.GLOBAL,
                    GravityParameterization.LOG_MULTIPLIER,
                    source="origin_time_totals",
                )
            if self.origin_total_correction_scope is GravityEffectScope.ORIGIN_ZONE:
                return GravityComponentSpecification(
                    name,
                    GravityEffectScope.ORIGIN_ZONE,
                    GravityParameterization.LOG_MULTIPLIER,
                    grouping="origin_zone_index",
                    group_count=self.origin_zone_count,
                    constraint=GravityConstraint.SUM_ZERO,
                    regularization=(
                        GravityRegularization(
                            GravityRegularizationType.RIDGE,
                            self.origin_zone_ridge,
                        )
                        if self.origin_zone_ridge > 0
                        else GravityRegularization()
                    ),
                    source="origin_time_totals",
                )
            return GravityComponentSpecification(
                name,
                GravityEffectScope.NONE,
                GravityParameterization.FIXED,
                source="origin_time_totals",
            )
        if name == "destination_attractiveness":
            if self.destination_attractiveness_scope is GravityEffectScope.DESTINATION_ZONE:
                return GravityComponentSpecification(
                    name,
                    GravityEffectScope.DESTINATION_ZONE,
                    GravityParameterization.ADDITIVE,
                    grouping="destination_zone_index",
                    group_count=self.destination_zone_count,
                    constraint=GravityConstraint.SUM_ZERO,
                    regularization=(
                        GravityRegularization(
                            GravityRegularizationType.RIDGE,
                            self.destination_zone_ridge,
                        )
                        if self.destination_zone_ridge > 0
                        else GravityRegularization()
                    ),
                    source="feature_cache",
                )
            return GravityComponentSpecification(
                name,
                GravityEffectScope.FIXED,
                GravityParameterization.FIXED,
                source="feature_cache",
            )
        if name == "temporal":
            if self.temporal_basis_scope is GravityEffectScope.TIME_PERIOD:
                return GravityComponentSpecification(
                    name,
                    GravityEffectScope.TIME_PERIOD,
                    GravityParameterization.ADDITIVE,
                    grouping="time_period_index",
                    group_count=self.time_period_count,
                    constraint=GravityConstraint.SUM_ZERO,
                    regularization=(
                        GravityRegularization(
                            GravityRegularizationType.RIDGE,
                            self.time_period_ridge,
                        )
                        if self.time_period_ridge > 0
                        else GravityRegularization()
                    ),
                )
            return GravityComponentSpecification(
                name, GravityEffectScope.NONE, GravityParameterization.FIXED
            )
        return GravityComponentSpecification(
            name, GravityEffectScope.NONE, GravityParameterization.FIXED
        )

    def component(self, name: str) -> GravityComponentSpecification:
        if name not in _COMPONENT_ORDER:
            raise KeyError(f"unknown gravity component {name!r}.")
        return next(
            (item for item in self.components if item.name == name),
            self._legacy_component(name),
        )

    @property
    def active_components(self) -> tuple[GravityComponentSpecification, ...]:
        return tuple(self.component(name) for name in _COMPONENT_ORDER)

    @property
    def parameter_count(self) -> int:
        return sum(item.parameter_count for item in self.active_components)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for component in self.active_components:
            count = component.parameter_count
            if not count:
                continue
            if component.name == "journey_time" and component.scope is GravityEffectScope.GLOBAL:
                names.append("beta_time")
            elif component.name == "transfer" and component.scope is GravityEffectScope.GLOBAL:
                names.append("beta_transfer")
            elif component.name == "dispersion" and component.scope is GravityEffectScope.GLOBAL:
                names.append("dispersion")
            elif component.name == "production" and component.scope is GravityEffectScope.GLOBAL:
                names.append("production_scale")
            elif component.parameterization is GravityParameterization.POSITIVE:
                names.append(
                    {
                        "journey_time": "beta_time",
                        "transfer": "beta_transfer",
                        "waiting_time": "beta_waiting",
                        "dispersion": "dispersion",
                    }.get(component.name, f"{component.name}.base")
                )
                names.extend(
                    f"{component.name}.deviation[{index}]"
                    for index in range(count - 1)
                )
            else:
                legacy_prefix = {
                    ("destination_attractiveness", GravityEffectScope.DESTINATION_ZONE): "destination_zone_deviation",
                    ("temporal", GravityEffectScope.TIME_PERIOD): "time_period_deviation",
                    ("production", GravityEffectScope.ORIGIN_ZONE): "origin_zone_deviation",
                }.get((component.name, component.scope))
                prefix = legacy_prefix or f"{component.name}.deviation"
                names.extend(f"{prefix}[{index}]" for index in range(count))
        return tuple(names)

    @property
    def required_feature_mappings(self) -> tuple[str, ...]:
        required: list[str] = []
        default = {
            GravityEffectScope.ORIGIN: "origin_index",
            GravityEffectScope.DESTINATION: "destination_index",
            GravityEffectScope.TIME_PERIOD: "time_period_index",
            GravityEffectScope.ORIGIN_TIME: "origin_time_group_index",
            GravityEffectScope.DESTINATION_TIME: "destination_time_group_index",
            GravityEffectScope.ORIGIN_ZONE: "origin_zone_index",
            GravityEffectScope.DESTINATION_ZONE: "destination_zone_index",
            GravityEffectScope.ZONE_PAIR: "zone_pair_index",
        }
        for component in self.active_components:
            if not component.grouped and component.scope is not GravityEffectScope.SMOOTH_BASIS:
                continue
            mapping = component.grouping or default.get(component.scope)
            if mapping is None:
                raise ValueError(
                    f"component {component.name!r} has no feature grouping."
                )
            required.append(mapping)
        return tuple(dict.fromkeys(required))

    def identifiability_warnings(self) -> tuple[str, ...]:
        result: list[str] = []
        production = self.component("production")
        attractiveness = self.component("destination_attractiveness")
        if attractiveness.scope is GravityEffectScope.GLOBAL:
            result.append(
                "A global additive attractiveness correction cancels within each "
                "origin-time softmax and is weakly identified."
            )
        if attractiveness.scope is GravityEffectScope.TIME_PERIOD:
            result.append(
                "A time-period-only attractiveness correction is constant within "
                "each origin-time softmax and is not separately identified."
            )
        temporal = self.component("temporal")
        if temporal.scope in (
            GravityEffectScope.GLOBAL,
            GravityEffectScope.TIME_PERIOD,
            GravityEffectScope.ORIGIN,
            GravityEffectScope.ORIGIN_TIME,
            GravityEffectScope.ORIGIN_ZONE,
        ):
            result.append(
                f"Temporal utility scope {temporal.scope.value!r} is constant "
                "within an origin-time softmax; prefer a time-specific production "
                "correction or a journey-time coefficient interaction."
            )
        if production.scope is GravityEffectScope.GLOBAL and (
            attractiveness.scope is GravityEffectScope.GLOBAL
        ):
            result.append(
                "Global production and global attractiveness corrections should "
                "not be interpreted separately."
            )
        if production.grouped and attractiveness.grouped:
            result.append(
                "Grouped production and destination-attractiveness corrections "
                "are normalized but may remain strongly correlated; inspect "
                "held-out performance and Hessian diagnostics."
            )
        for component in self.active_components:
            if component.grouped and component.regularization.kind is GravityRegularizationType.NONE:
                result.append(
                    f"Component {component.name!r} has no ridge regularization; "
                    "check group support and curvature diagnostics."
                )
        return tuple(result)

    def emit_identifiability_warnings(self) -> tuple[str, ...]:
        messages = self.identifiability_warnings()
        for message in messages:
            warnings.warn(message, UserWarning, stacklevel=2)
        return messages

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        if (
            not self.components
            and self.model_name == "minimal_three_parameter"
            and self.likelihood == GravityLikelihoodSpecification()
            and self.time == GravityTimeSpecification()
        ):
            return {
                name: value.value if isinstance(value, GravityEffectScope) else value
                for name, value in asdict(self).items()
                if name
                not in {
                    "components",
                    "likelihood",
                    "time",
                    "model_name",
                    "schema_version",
                }
            } | {"schema_version": 2}
        return {
            "schema_version": 3,
            "model_name": self.model_name,
            "legacy": {
                name: value.value if isinstance(value, GravityEffectScope) else value
                for name, value in asdict(self).items()
                if name
                not in {"components", "likelihood", "time", "model_name", "schema_version"}
            },
            "components": [item.to_dict() for item in self.components],
            "likelihood": asdict(self.likelihood),
            "time": asdict(self.time),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GravityModelSpecification:
        schema = int(payload.get("schema_version", 1))
        if schema in (1, 2):
            values = dict(payload)
            values["schema_version"] = 3
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
            for name in scope_names:
                values[name] = GravityEffectScope(str(values[name]))
            return cls(**values)  # type: ignore[arg-type]
        if schema != 3:
            raise ValueError("unsupported gravity specification schema version.")
        legacy = dict(payload.get("legacy", {}))  # type: ignore[arg-type]
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
        for name in scope_names & legacy.keys():
            legacy[name] = GravityEffectScope(str(legacy[name]))
        likelihood_payload = dict(payload.get("likelihood", {}))  # type: ignore[arg-type]
        time_payload = dict(payload.get("time", {}))  # type: ignore[arg-type]
        return cls(
            **legacy,  # type: ignore[arg-type]
            model_name=str(payload.get("model_name", "gravity_model")),
            components=tuple(
                GravityComponentSpecification.from_dict(item)
                for item in payload.get("components", [])  # type: ignore[union-attr]
            ),
            likelihood=GravityLikelihoodSpecification(**likelihood_payload),
            time=GravityTimeSpecification(
                units=str(time_payload.get("units", "index")),
                interpretation=str(
                    time_payload.get(
                        "interpretation", "categorical departure-time bins"
                    )
                ),
                bin_labels=tuple(time_payload.get("bin_labels", ())),
                smooth_basis_name=(
                    None
                    if time_payload.get("smooth_basis_name") is None
                    else str(time_payload["smooth_basis_name"])
                ),
            ),
            schema_version=3,
        )
