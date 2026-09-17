"""Declarative, fingerprinted gravity-model specification contracts."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import ClassVar

import numpy as np

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
    ORIGIN_ZONE_TIME = "origin_zone_time"
    DESTINATION_ZONE = "destination_zone"
    DESTINATION_ZONE_TIME = "destination_zone_time"
    ZONE_PAIR = "zone_pair"
    CUSTOM_GROUP = "custom_group"
    SMOOTH_BASIS = "smooth_basis"


class GravityConstraint(str, Enum):
    """Identifying constraint applied to a categorical parameter block."""

    NONE = "none"
    SUM_ZERO = "sum_zero"
    REFERENCE = "reference"
    TWO_WAY_CENTERED = "two_way_centered"


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
        GravityEffectScope.ORIGIN_ZONE_TIME,
        GravityEffectScope.DESTINATION_ZONE,
        GravityEffectScope.DESTINATION_ZONE_TIME,
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
class GravityDeviationSpecification:
    """Centered categorical deviations attached to a base component.

    A deviation block is deliberately separate from its base component.  This
    lets the production scale remain unpenalized while the time-regime
    deviations carry their own centering and regularization contract.
    """

    scope: GravityEffectScope
    grouping: str | None = None
    group_count: int = 0
    constraint: GravityConstraint = GravityConstraint.SUM_ZERO
    reference_category: int | None = None
    regularization: GravityRegularization = GravityRegularization()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, GravityEffectScope):
            object.__setattr__(self, "scope", GravityEffectScope(self.scope))
        if not isinstance(self.constraint, GravityConstraint):
            object.__setattr__(self, "constraint", GravityConstraint(self.constraint))
        if self.scope not in _GROUP_SCOPES:
            raise ValueError("gravity deviations require a grouped scope.")
        if self.group_count < 2:
            raise ValueError("gravity deviations require group_count >= 2.")
        if self.constraint is GravityConstraint.NONE:
            raise ValueError(
                "gravity deviations require sum_zero or reference constraint."
            )
        if self.scope is GravityEffectScope.CUSTOM_GROUP and not self.grouping:
            raise ValueError(
                "custom-group deviations require an explicit grouping name."
            )
        if self.constraint is GravityConstraint.REFERENCE:
            if self.reference_category is None:
                raise ValueError("reference deviations require reference_category.")
            if not 0 <= self.reference_category < self.group_count:
                raise ValueError(
                    "deviation reference_category is outside the declared groups."
                )
        elif self.reference_category is not None:
            raise ValueError(
                "reference_category is valid only with reference constraint."
            )
        if not isinstance(self.regularization, GravityRegularization):
            raise TypeError("deviation regularization must be a GravityRegularization.")

    @property
    def parameter_count(self) -> int:
        return self.group_count - 1

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope.value,
            "grouping": self.grouping,
            "group_count": self.group_count,
            "constraint": self.constraint.value,
            "reference_category": self.reference_category,
            "regularization": self.regularization.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, object] | None
    ) -> GravityDeviationSpecification | None:
        if payload is None:
            return None
        return cls(
            scope=GravityEffectScope(str(payload["scope"])),
            grouping=None
            if payload.get("grouping") is None
            else str(payload["grouping"]),
            group_count=int(payload.get("group_count", 0)),
            constraint=GravityConstraint(str(payload.get("constraint", "sum_zero"))),
            reference_category=(
                None
                if payload.get("reference_category") is None
                else int(payload["reference_category"])
            ),
            regularization=GravityRegularization.from_dict(
                payload.get("regularization")  # type: ignore[arg-type]
            ),
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
    deviation: GravityDeviationSpecification | None = None

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
        if self.deviation is not None and not isinstance(
            self.deviation, GravityDeviationSpecification
        ):
            raise TypeError("deviation must be a GravityDeviationSpecification.")
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
                raise ValueError(
                    "custom_group scope requires an explicit grouping name."
                )
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
            raise ValueError(
                "reference_category is valid only with reference constraint."
            )
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
        if self.deviation is not None and self.scope is not GravityEffectScope.GLOBAL:
            raise ValueError("component deviations require a global base component.")

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
            return 1 + (0 if self.deviation is None else self.deviation.parameter_count)
        if self.scope is GravityEffectScope.SMOOTH_BASIS:
            return self.group_count
        if self.parameterization is GravityParameterization.POSITIVE:
            return 1 + self.deviation_count
        return self.deviation_count

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
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
        if self.deviation is not None:
            payload["deviation"] = self.deviation.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GravityComponentSpecification:
        return cls(
            name=str(payload["name"]),
            scope=GravityEffectScope(str(payload["scope"])),
            parameterization=GravityParameterization(str(payload["parameterization"])),
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
            deviation=GravityDeviationSpecification.from_dict(
                payload.get("deviation")  # type: ignore[arg-type]
            ),
        )


@dataclass(frozen=True, slots=True)
class GravityTermSpecification:
    """One additive term in a production or destination predictor.

    Terms are deliberately separate from the historical component contract.
    This makes it possible to compose, for example, an origin-zone main effect
    and an origin-zone-by-time interaction while retaining the old component
    API unchanged.
    """

    name: str
    target: str
    scope: GravityEffectScope
    grouping: str | None = None
    group_count: int = 0
    constraint: GravityConstraint = GravityConstraint.SUM_ZERO
    reference_category: int | None = None
    regularization: GravityRegularization = GravityRegularization()
    parameterization: GravityParameterization = GravityParameterization.ADDITIVE
    fixed_value: float | None = None
    row_grouping: str | None = None
    column_grouping: str | None = None
    row_group_count: int = 0
    column_group_count: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("gravity term name must be nonempty.")
        target = self.target.lower().strip()
        aliases = {
            "production_log_total": "production",
            "destination_utility": "destination_attractiveness",
            "destination": "destination_attractiveness",
        }
        target = aliases.get(target, target)
        if target not in {"production", "destination_attractiveness"}:
            raise ValueError(
                "gravity term target must be 'production' or "
                "'destination_attractiveness'."
            )
        object.__setattr__(self, "target", target)
        if not isinstance(self.scope, GravityEffectScope):
            object.__setattr__(self, "scope", GravityEffectScope(self.scope))
        if not isinstance(self.constraint, GravityConstraint):
            object.__setattr__(self, "constraint", GravityConstraint(self.constraint))
        if not isinstance(self.parameterization, GravityParameterization):
            object.__setattr__(
                self,
                "parameterization",
                GravityParameterization(self.parameterization),
            )
        if not isinstance(self.regularization, GravityRegularization):
            raise TypeError("term regularization must be a GravityRegularization.")
        if self.target == "production":
            allowed = {
                GravityEffectScope.NONE,
                GravityEffectScope.FIXED,
                GravityEffectScope.GLOBAL,
                GravityEffectScope.ORIGIN,
                GravityEffectScope.TIME_PERIOD,
                GravityEffectScope.ORIGIN_TIME,
                GravityEffectScope.ORIGIN_ZONE,
                GravityEffectScope.ORIGIN_ZONE_TIME,
                GravityEffectScope.CUSTOM_GROUP,
            }
        else:
            allowed = {
                GravityEffectScope.NONE,
                GravityEffectScope.FIXED,
                GravityEffectScope.DESTINATION,
                GravityEffectScope.DESTINATION_TIME,
                GravityEffectScope.DESTINATION_ZONE,
                GravityEffectScope.DESTINATION_ZONE_TIME,
                GravityEffectScope.CUSTOM_GROUP,
            }
        if self.scope not in allowed:
            raise ValueError(
                f"scope {self.scope.value!r} is not valid for {self.target} terms."
            )
        grouped = self.scope in _GROUP_SCOPES
        centered = self.constraint is GravityConstraint.TWO_WAY_CENTERED
        if grouped:
            if self.group_count < 2:
                raise ValueError("grouped gravity terms require group_count >= 2.")
            if (
                self.scope
                in (
                    GravityEffectScope.ORIGIN_ZONE_TIME,
                    GravityEffectScope.DESTINATION_ZONE_TIME,
                )
                and centered
            ):
                if self.row_group_count < 2 or self.column_group_count < 2:
                    raise ValueError(
                        "two-way centered terms require row and column group counts >= 2."
                    )
                if self.group_count != self.row_group_count * self.column_group_count:
                    raise ValueError(
                        "two-way centered term group_count must equal row_count*column_count."
                    )
                if not self.row_grouping or not self.column_grouping:
                    raise ValueError(
                        "two-way centered terms require row_grouping and column_grouping."
                    )
            elif self.constraint is GravityConstraint.NONE:
                raise ValueError("grouped gravity terms require a constraint.")
            if self.scope is GravityEffectScope.CUSTOM_GROUP and not self.grouping:
                raise ValueError(
                    "custom-group terms require an explicit grouping name."
                )
        elif self.group_count != 0:
            raise ValueError("group_count is valid only for grouped gravity terms.")
        if centered and self.scope not in (
            GravityEffectScope.ORIGIN_ZONE_TIME,
            GravityEffectScope.DESTINATION_ZONE_TIME,
        ):
            raise ValueError("two-way centering is valid only for zone-time terms.")
        if self.constraint is GravityConstraint.REFERENCE:
            if not grouped or self.reference_category is None:
                raise ValueError("reference terms require reference_category.")
            if not 0 <= self.reference_category < self.group_count:
                raise ValueError(
                    "term reference_category is outside the declared groups."
                )
        elif self.reference_category is not None:
            raise ValueError(
                "reference_category is valid only with reference constraint."
            )
        fixed = self.scope in (GravityEffectScope.NONE, GravityEffectScope.FIXED)
        if fixed and self.parameterization is not GravityParameterization.FIXED:
            raise ValueError("none/fixed terms require fixed parameterization.")
        if not fixed and self.parameterization is GravityParameterization.FIXED:
            raise ValueError("estimated terms cannot use fixed parameterization.")
        if self.fixed_value is not None and not fixed:
            raise ValueError("fixed_value is valid only for none/fixed terms.")
        if self.fixed_value is not None and not (
            float("-inf") < float(self.fixed_value) < float("inf")
        ):
            raise ValueError("fixed_value must be finite.")

    @property
    def grouped(self) -> bool:
        return self.scope in _GROUP_SCOPES

    @property
    def parameter_count(self) -> int:
        if self.scope in (GravityEffectScope.NONE, GravityEffectScope.FIXED):
            return 0
        if self.scope is GravityEffectScope.GLOBAL:
            return 1
        if self.constraint is GravityConstraint.TWO_WAY_CENTERED:
            return (self.row_group_count - 1) * (self.column_group_count - 1)
        return self.group_count - 1

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "target": self.target,
            "scope": self.scope.value,
            "grouping": self.grouping,
            "group_count": self.group_count,
            "constraint": self.constraint.value,
            "reference_category": self.reference_category,
            "regularization": self.regularization.to_dict(),
            "parameterization": self.parameterization.value,
            "fixed_value": self.fixed_value,
            "row_grouping": self.row_grouping,
            "column_grouping": self.column_grouping,
            "row_group_count": self.row_group_count,
            "column_group_count": self.column_group_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> GravityTermSpecification:
        return cls(
            name=str(payload["name"]),
            target=str(payload["target"]),
            scope=GravityEffectScope(str(payload["scope"])),
            grouping=None
            if payload.get("grouping") is None
            else str(payload["grouping"]),
            group_count=int(payload.get("group_count", 0)),
            constraint=GravityConstraint(str(payload.get("constraint", "sum_zero"))),
            reference_category=(
                None
                if payload.get("reference_category") is None
                else int(payload["reference_category"])
            ),
            regularization=GravityRegularization.from_dict(
                payload.get("regularization")  # type: ignore[arg-type]
            ),
            parameterization=GravityParameterization(
                str(payload.get("parameterization", "additive"))
            ),
            fixed_value=(
                None
                if payload.get("fixed_value") is None
                else float(payload["fixed_value"])
            ),
            row_grouping=None
            if payload.get("row_grouping") is None
            else str(payload["row_grouping"]),
            column_grouping=None
            if payload.get("column_grouping") is None
            else str(payload["column_grouping"]),
            row_group_count=int(payload.get("row_group_count", 0)),
            column_group_count=int(payload.get("column_group_count", 0)),
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


_OBSOLETE_OD_MATRIX_SOURCES = frozenset(
    {
        "od_matrix",
        "a_priori_od_matrix",
        "a-priori_od_matrix",
        "a-priori-od-matrix",
        "a_priori_od",
        "a-priori-od",
        "a_priori_matrix",
        "a-priori-matrix",
        "apriori_od_matrix",
        "apriori-od-matrix",
        "apriori_od",
        "apriori_matrix",
        "prior_od_matrix",
        "prior_od",
        "prior_matrix",
        "baseline_od_matrix",
        "baseline-od-matrix",
        "external_od_matrix",
        "external-od-matrix",
        "demand_matrix",
        "prior_demand",
    }
)
_OBSOLETE_OD_MATRIX_KEYS = frozenset(
    {
        "od_matrix",
        "a_priori_od_matrix",
        "a-priori_od_matrix",
        "a-priori-od-matrix",
        "a_priori_od",
        "a-priori-od",
        "a_priori_matrix",
        "a-priori-matrix",
        "apriori_od_matrix",
        "apriori-od-matrix",
        "apriori_od",
        "apriori_matrix",
        "prior_od_matrix",
        "prior_od",
        "prior_matrix",
        "baseline_od_matrix",
        "baseline-od-matrix",
        "external_od_matrix",
        "external-od-matrix",
        "demand_matrix",
        "prior_demand",
    }
)


def _reject_obsolete_od_matrix_source(source: object, *, context: str) -> None:
    """Reject removed matrix-valued production sources explicitly."""

    if isinstance(source, (Mapping, list, tuple, np.ndarray)):
        raise ValueError(
            "a priori OD matrix is obsolete and unsupported; "
            "matrix-valued a priori OD matrices are not accepted; "
            f"{context} must be a source name, not a matrix-valued object. Use "
            "production_source='unit_exposure' or the one-dimensional "
            "'origin_time_totals' source."
        )
    normalized = str(source).strip().lower()
    if normalized in _OBSOLETE_OD_MATRIX_SOURCES:
        raise ValueError(
            "a priori OD matrix is obsolete and unsupported; "
            "matrix-valued a priori OD matrices are not accepted; "
            f"{context}={source!r} is no longer accepted. Use "
            "production_source='unit_exposure' or the one-dimensional "
            "'origin_time_totals' source."
        )


def _reject_obsolete_od_matrix_fields(
    payload: Mapping[str, object], *, context: str
) -> None:
    """Reject removed matrix-valued fields before generic schema validation."""

    present = sorted(
        str(key)
        for key in payload
        if str(key).strip().lower() in _OBSOLETE_OD_MATRIX_KEYS
    )
    if present:
        raise ValueError(
            "a priori OD matrix is obsolete and unsupported; "
            "matrix-valued a priori OD matrices are not accepted; "
            f"remove {context} field(s) {present} and use "
            "production_source='unit_exposure' or a one-dimensional "
            "'origin_time_totals' vector."
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
    terms: tuple[GravityTermSpecification, ...] = ()
    preset: str | None = None
    production_source: str = "origin_time_totals"
    destination_attractiveness_source: str = "feature_cache"
    likelihood: GravityLikelihoodSpecification = GravityLikelihoodSpecification()
    time: GravityTimeSpecification = GravityTimeSpecification()
    additive_flow_blocks: tuple[dict[str, object], ...] = ()
    schema_version: int = 3

    SUPPORTED_SCHEMA_VERSIONS: ClassVar[tuple[int, ...]] = (1, 2, 3, 4)

    def __post_init__(self) -> None:
        if self.schema_version not in self.SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("unsupported gravity specification schema version.")
        if not self.model_name:
            raise ValueError("gravity model_name must be nonempty.")
        additive_blocks: list[dict[str, object]] = []
        additive_names: set[str] = set()
        for item in self.additive_flow_blocks:
            if not isinstance(item, Mapping):
                raise TypeError("additive_flow_blocks must contain mappings.")
            block = dict(item)
            name = str(block.get("name", ""))
            if not name:
                raise ValueError("additive flow block names must be non-empty.")
            if name in additive_names:
                raise ValueError(f"duplicate additive flow block {name!r}.")
            additive_names.add(name)
            additive_blocks.append(block)
        object.__setattr__(self, "additive_flow_blocks", tuple(additive_blocks))
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
        if self.preset is not None and not self.preset:
            raise ValueError("gravity preset must be nonempty when supplied.")
        _reject_obsolete_od_matrix_source(
            self.production_source, context="production_source"
        )
        if self.production_source not in {"origin_time_totals", "unit_exposure"}:
            raise ValueError(
                "production_source must be 'origin_time_totals' or 'unit_exposure'."
            )
        if self.destination_attractiveness_source not in {
            "feature_cache",
            "external",
            "baseline_derived",
        }:
            raise ValueError("unsupported destination_attractiveness_source.")
        term_names = [item.name for item in self.terms]
        if len(set(term_names)) != len(term_names):
            raise ValueError("gravity term names must be unique.")
        if any(not isinstance(item, GravityTermSpecification) for item in self.terms):
            raise TypeError("terms must contain GravityTermSpecification instances.")
        if self.terms and any(
            item.name in {"production", "destination_attractiveness"}
            for item in self.components
        ):
            raise ValueError(
                "composable terms cannot be combined with explicit production or "
                "destination-attractiveness components."
            )
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
            if (
                not isinstance(ridge, (int, float))
                or not (float("-inf") < float(ridge) < float("inf"))
                or ridge < 0
            ):
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
                GravityEffectScope.ORIGIN_ZONE_TIME,
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
                GravityEffectScope.DESTINATION_ZONE_TIME,
                GravityEffectScope.CUSTOM_GROUP,
            },
            "journey_time": set(GravityEffectScope) - {GravityEffectScope.SMOOTH_BASIS},
            "transfer": set(GravityEffectScope) - {GravityEffectScope.SMOOTH_BASIS},
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
            if (
                component.scope
                not in (
                    GravityEffectScope.NONE,
                    GravityEffectScope.FIXED,
                )
                and component.parameterization is not expected_parameterization
            ):
                raise ValueError(
                    f"component {component.name!r} requires parameterization "
                    f"{expected_parameterization.value!r}."
                )
            if component.name in (
                "production",
                "destination_attractiveness",
                "temporal",
            ):
                if component.grouped and component.constraint is GravityConstraint.NONE:
                    raise ValueError(
                        f"component {component.name!r} must be normalized."
                    )
            if (
                component.name == "production"
                and component.scope
                in (
                    GravityEffectScope.ORIGIN_TIME,
                    GravityEffectScope.CUSTOM_GROUP,
                )
                and component.regularization.kind is not GravityRegularizationType.RIDGE
            ):
                raise ValueError(
                    f"high-dimensional production scope {component.scope.value!r} "
                    "requires ridge regularization."
                )
            if (
                component.name
                in (
                    "journey_time",
                    "transfer",
                    "waiting_time",
                    "dispersion",
                )
                and component.scope is GravityEffectScope.FIXED
            ):
                if component.fixed_value is None or component.fixed_value <= 0:
                    raise ValueError(
                        f"fixed {component.name!r} requires a strictly positive "
                        "fixed_value."
                    )
            if (
                component.name == "destination_attractiveness"
                and component.source
                not in (
                    "feature_cache",
                    "external",
                    "baseline_derived",
                )
            ):
                raise ValueError(
                    "destination attractiveness requires source feature_cache, "
                    "external, or baseline_derived."
                )
            if (
                component.name == "production"
                and component.source != "origin_time_totals"
            ):
                _reject_obsolete_od_matrix_source(
                    component.source, context="production component source"
                )
                raise ValueError("production requires source 'origin_time_totals'.")
            if component.deviation is not None:
                if component.name != "production":
                    raise ValueError(
                        "deviation blocks are currently supported only for production."
                    )
                if component.scope is not GravityEffectScope.GLOBAL:
                    raise ValueError(
                        "production deviations require a global production scale."
                    )
                if (
                    component.parameterization
                    is not GravityParameterization.LOG_MULTIPLIER
                ):
                    raise ValueError(
                        "production deviations require log_multiplier production."
                    )
                if (
                    component.deviation.regularization.kind
                    is not GravityRegularizationType.RIDGE
                ):
                    raise ValueError(
                        "production deviations require ridge regularization."
                    )
                if component.regularization.kind is not GravityRegularizationType.NONE:
                    raise ValueError(
                        "the global production scale is unregularized; attach ridge "
                        "regularization to the deviation block."
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
        self._validate_terms()

    def _validate_terms(self) -> None:
        production_terms = [item for item in self.terms if item.target == "production"]
        destination_terms = [
            item for item in self.terms if item.target == "destination_attractiveness"
        ]
        if any(
            item.scope in (GravityEffectScope.GLOBAL, GravityEffectScope.TIME_PERIOD)
            for item in destination_terms
        ):
            raise ValueError(
                "destination global/time-period terms cancel in the within-origin-time softmax."
            )
        if production_terms and self.production_source not in {
            "origin_time_totals",
            "unit_exposure",
        }:
            raise ValueError("unsupported production source for composable terms.")
        for target, items in (
            ("production", production_terms),
            ("destination_attractiveness", destination_terms),
        ):
            seen_scopes: set[tuple[GravityEffectScope, str | None]] = set()
            for item in items:
                key = (item.scope, item.grouping)
                if key in seen_scopes:
                    raise ValueError(f"duplicate {target} term scope/grouping {key!r}.")
                seen_scopes.add(key)
                if (
                    item.scope
                    in (
                        GravityEffectScope.ORIGIN_ZONE_TIME,
                        GravityEffectScope.DESTINATION_ZONE_TIME,
                    )
                    and item.constraint is GravityConstraint.NONE
                ):
                    raise ValueError("zone-time interactions must be constrained.")
                if (
                    target == "production"
                    and item.scope
                    in (
                        GravityEffectScope.ORIGIN_TIME,
                        GravityEffectScope.ORIGIN_ZONE_TIME,
                    )
                    and item.regularization.kind is not GravityRegularizationType.RIDGE
                ):
                    raise ValueError(
                        f"high-dimensional production term {item.name!r} requires ridge regularization."
                    )
            unrestricted = [
                item for item in items if item.constraint is GravityConstraint.NONE
            ]
            if unrestricted and len(items) > 1:
                raise ValueError(
                    f"unrestricted {target} terms cannot be combined with other terms."
                )

    def _legacy_component(self, name: str) -> GravityComponentSpecification:
        if name == "journey_time":
            return GravityComponentSpecification(
                name,
                self.journey_time_scope,
                GravityParameterization.POSITIVE
                if self.journey_time_scope
                not in (GravityEffectScope.NONE, GravityEffectScope.FIXED)
                else GravityParameterization.FIXED,
            )
        if name == "transfer":
            return GravityComponentSpecification(
                name,
                self.transfer_scope,
                GravityParameterization.POSITIVE
                if self.transfer_scope
                not in (GravityEffectScope.NONE, GravityEffectScope.FIXED)
                else GravityParameterization.FIXED,
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
                if self.waiting_time_scope
                not in (GravityEffectScope.NONE, GravityEffectScope.FIXED)
                else GravityParameterization.FIXED,
                fixed_value=0.0
                if self.waiting_time_scope is GravityEffectScope.NONE
                else None,
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
            if (
                self.destination_attractiveness_scope
                is GravityEffectScope.DESTINATION_ZONE
            ):
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
    def production_terms(self) -> tuple[GravityTermSpecification, ...]:
        return tuple(item for item in self.terms if item.target == "production")

    @property
    def destination_attractiveness_terms(self) -> tuple[GravityTermSpecification, ...]:
        return tuple(
            item for item in self.terms if item.target == "destination_attractiveness"
        )

    @property
    def uses_external_od_matrix(self) -> bool:
        """Whether an externally supplied OD matrix is consumed.

        Neither supported production source is an OD matrix: ``origin_time_totals``
        are one-dimensional totals, while ``unit_exposure`` is neutral.
        """
        return False

    @property
    def uses_external_production_totals(self) -> bool:
        return self.production_source == "origin_time_totals"

    @property
    def expanded_specification(self) -> dict[str, object]:
        """Canonical fully expanded payload suitable for a run manifest."""
        return self.to_dict()

    @property
    def parameter_count(self) -> int:
        return sum(item.parameter_count for item in self.active_components) + sum(
            item.parameter_count for item in self.terms
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for component in self.active_components:
            count = component.parameter_count
            if not count:
                continue
            if (
                component.name == "journey_time"
                and component.scope is GravityEffectScope.GLOBAL
            ):
                names.append("beta_time")
            elif (
                component.name == "transfer"
                and component.scope is GravityEffectScope.GLOBAL
            ):
                names.append("beta_transfer")
            elif (
                component.name == "dispersion"
                and component.scope is GravityEffectScope.GLOBAL
            ):
                names.append("dispersion")
            elif (
                component.name == "production"
                and component.scope is GravityEffectScope.GLOBAL
            ):
                names.append("production_scale")
                if component.deviation is not None:
                    names.extend(
                        f"production_time_deviation[{index}]"
                        for index in range(component.deviation.parameter_count)
                    )
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
                    f"{component.name}.deviation[{index}]" for index in range(count - 1)
                )
            else:
                legacy_prefix = {
                    (
                        "destination_attractiveness",
                        GravityEffectScope.DESTINATION_ZONE,
                    ): "destination_zone_deviation",
                    (
                        "temporal",
                        GravityEffectScope.TIME_PERIOD,
                    ): "time_period_deviation",
                    (
                        "production",
                        GravityEffectScope.ORIGIN_ZONE,
                    ): "origin_zone_deviation",
                }.get((component.name, component.scope))
                prefix = legacy_prefix or f"{component.name}.deviation"
                names.extend(f"{prefix}[{index}]" for index in range(count))
        for term in self.terms:
            if term.parameter_count == 0:
                continue
            if term.scope is GravityEffectScope.GLOBAL:
                names.append(term.name)
            elif term.constraint is GravityConstraint.TWO_WAY_CENTERED:
                names.extend(
                    f"{term.name}[{index}]" for index in range(term.parameter_count)
                )
            else:
                names.extend(
                    f"{term.name}[{index}]" for index in range(term.parameter_count)
                )
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
            GravityEffectScope.ORIGIN_ZONE_TIME: "origin_zone_time_index",
            GravityEffectScope.DESTINATION_ZONE: "destination_zone_index",
            GravityEffectScope.DESTINATION_ZONE_TIME: "destination_zone_time_index",
            GravityEffectScope.ZONE_PAIR: "zone_pair_index",
        }
        for component in self.active_components:
            if component.grouped or component.scope is GravityEffectScope.SMOOTH_BASIS:
                mapping = component.grouping or default.get(component.scope)
                if mapping is None:
                    raise ValueError(
                        f"component {component.name!r} has no feature grouping."
                    )
                required.append(mapping)
            if component.deviation is not None:
                mapping = component.deviation.grouping or default.get(
                    component.deviation.scope
                )
                if mapping is None:
                    raise ValueError(
                        f"component {component.name!r} deviation has no feature grouping."
                    )
                required.append(mapping)
        for term in self.terms:
            if term.grouped:
                mapping = term.grouping or default.get(term.scope)
                if mapping is None:
                    raise ValueError(f"term {term.name!r} has no feature grouping.")
                required.append(mapping)
                if term.constraint is GravityConstraint.TWO_WAY_CENTERED:
                    if term.row_grouping:
                        required.append(term.row_grouping)
                    if term.column_grouping:
                        required.append(term.column_grouping)
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
            if (
                component.grouped
                and component.regularization.kind is GravityRegularizationType.NONE
            ):
                result.append(
                    f"Component {component.name!r} has no ridge regularization; "
                    "check group support and curvature diagnostics."
                )
        for term in self.terms:
            if (
                term.grouped
                and term.regularization.kind is GravityRegularizationType.NONE
            ):
                result.append(
                    f"Term {term.name!r} has no ridge regularization; check group support."
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
            and not self.terms
            and self.preset is None
            and self.production_source == "origin_time_totals"
            and self.destination_attractiveness_source == "feature_cache"
            and self.model_name == "minimal_three_parameter"
            and self.likelihood == GravityLikelihoodSpecification()
            and self.time == GravityTimeSpecification()
            and not self.additive_flow_blocks
        ):
            return {
                name: value.value if isinstance(value, GravityEffectScope) else value
                for name, value in asdict(self).items()
                if name
                not in {
                    "components",
                    "terms",
                    "preset",
                    "production_source",
                    "destination_attractiveness_source",
                    "likelihood",
                    "time",
                    "model_name",
                    "additive_flow_blocks",
                    "schema_version",
                }
            } | {"schema_version": 2}
        schema_version = (
            4
            if self.terms
            or self.preset is not None
            or self.production_source != "origin_time_totals"
            or self.destination_attractiveness_source != "feature_cache"
            else 3
        )
        return {
            "schema_version": schema_version,
            "model_name": self.model_name,
            "preset": self.preset,
            "production_source": self.production_source,
            "destination_attractiveness_source": self.destination_attractiveness_source,
            "legacy": {
                name: value.value if isinstance(value, GravityEffectScope) else value
                for name, value in asdict(self).items()
                if name
                not in {
                    "components",
                    "terms",
                    "preset",
                    "production_source",
                    "destination_attractiveness_source",
                    "likelihood",
                    "time",
                    "additive_flow_blocks",
                    "model_name",
                    "schema_version",
                }
            },
            "components": [item.to_dict() for item in self.components],
            "terms": [item.to_dict() for item in self.terms],
            "likelihood": asdict(self.likelihood),
            "time": asdict(self.time),
            "additive_flow_blocks": [dict(item) for item in self.additive_flow_blocks],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GravityModelSpecification:
        _reject_obsolete_od_matrix_fields(payload, context="gravity specification")
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
        if schema not in (3, 4):
            raise ValueError("unsupported gravity specification schema version.")
        legacy = dict(payload.get("legacy", {}))  # type: ignore[arg-type]
        _reject_obsolete_od_matrix_fields(
            legacy, context="gravity legacy specification"
        )
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
            terms=tuple(
                GravityTermSpecification.from_dict(item)
                for item in payload.get("terms", [])  # type: ignore[union-attr]
            ),
            preset=(None if payload.get("preset") is None else str(payload["preset"])),
            production_source=str(
                payload.get("production_source", "origin_time_totals")
            ),
            destination_attractiveness_source=str(
                payload.get("destination_attractiveness_source", "feature_cache")
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
            additive_flow_blocks=tuple(
                dict(item)
                for item in payload.get("additive_flow_blocks", [])  # type: ignore[union-attr]
            ),
            schema_version=schema,
        )

    @classmethod
    def from_preset(cls, name: str, **kwargs: object) -> GravityModelSpecification:
        """Expand a named production/destination preset."""
        return gravity_model_specification_from_preset(name, **kwargs)


_GRAVITY_PRESETS = (
    "minimal_fixed",
    "time_regime",
    "zone",
    "zone_time",
    "origin_time_full",
)


def _preset_count(
    features: object | None,
    attribute: str,
    explicit: int | None,
) -> int:
    if explicit is not None:
        return int(explicit)
    if features is None:
        raise ValueError(
            f"features or an explicit {attribute} is required to expand a gravity preset."
        )
    return int(getattr(features, attribute))


def _preset_mapping_count(features: object | None, mapping: str, fallback: int) -> int:
    if features is None:
        return fallback
    values = getattr(features, mapping, None)
    if values is None:
        raise ValueError(f"gravity preset requires feature mapping {mapping!r}.")
    return int(np.unique(np.asarray(values)).size)


def _preset_ridge(strength: float) -> GravityRegularization:
    return (
        GravityRegularization(GravityRegularizationType.RIDGE, float(strength))
        if strength > 0
        else GravityRegularization()
    )


def gravity_model_specification_from_preset(
    name: str,
    *,
    features: object | None = None,
    production_source: str = "unit_exposure",
    destination_attractiveness_source: str = "feature_cache",
    origin_zone_count: int | None = None,
    destination_zone_count: int | None = None,
    time_period_count: int | None = None,
    regularization_strength: float = 1.0,
    model_name: str | None = None,
    likelihood: GravityLikelihoodSpecification | None = None,
    time: GravityTimeSpecification | None = None,
) -> GravityModelSpecification:
    """Expand a named production/destination specification deterministically.

    ``minimal_fixed`` and ``time_regime`` deliberately use the legacy
    components so existing estimates retain their exact numerical contract.
    The zone-oriented presets use neutral unit exposure and explicit additive
    terms; no observed OD matrix is required.
    """

    preset = str(name).strip().lower()
    if preset not in _GRAVITY_PRESETS:
        raise ValueError(
            f"unknown gravity preset {name!r}; choose from {_GRAVITY_PRESETS}."
        )
    if regularization_strength < 0 or not float("-inf") < float(
        regularization_strength
    ) < float("inf"):
        raise ValueError("regularization_strength must be finite and non-negative.")
    if preset in {"minimal_fixed", "time_regime"}:
        if preset == "minimal_fixed":
            return GravityModelSpecification(
                model_name=model_name or "minimal_three_parameter",
                preset=preset,
                production_source="origin_time_totals",
                destination_attractiveness_source=destination_attractiveness_source,
                likelihood=likelihood or GravityLikelihoodSpecification(),
                time=time or GravityTimeSpecification(),
                schema_version=4,
            )
        return GravityModelSpecification(
            model_name=model_name or "time_regime",
            preset=preset,
            estimate_global_production_correction=True,
            production_source="origin_time_totals",
            destination_attractiveness_source=destination_attractiveness_source,
            likelihood=likelihood or GravityLikelihoodSpecification(),
            time=time or GravityTimeSpecification(),
            schema_version=4,
        )
    oz = _preset_count(features, "num_origins", origin_zone_count)
    dz = _preset_count(features, "num_destinations", destination_zone_count)
    tp = _preset_count(features, "num_departure_times", time_period_count)
    if preset in {"zone", "zone_time", "origin_time_full"}:
        oz = _preset_mapping_count(features, "origin_zone_index", oz)
        dz = _preset_mapping_count(features, "destination_zone_index", dz)
    if preset == "zone_time":
        tp = _preset_mapping_count(features, "time_period_index", tp)
    reg = _preset_ridge(regularization_strength)
    terms: list[GravityTermSpecification] = []
    terms.append(
        GravityTermSpecification(
            "production.global",
            "production",
            GravityEffectScope.GLOBAL,
        )
    )
    if preset in {"zone", "zone_time"}:
        terms.extend(
            (
                GravityTermSpecification(
                    "production.origin_zone",
                    "production",
                    GravityEffectScope.ORIGIN_ZONE,
                    grouping="origin_zone_index",
                    group_count=oz,
                    regularization=reg,
                ),
                GravityTermSpecification(
                    "destination.zone",
                    "destination_attractiveness",
                    GravityEffectScope.DESTINATION_ZONE,
                    grouping="destination_zone_index",
                    group_count=dz,
                    regularization=reg,
                ),
            )
        )
    if preset == "zone_time":
        terms.extend(
            (
                GravityTermSpecification(
                    "production.time_period",
                    "production",
                    GravityEffectScope.TIME_PERIOD,
                    grouping="time_period_index",
                    group_count=tp,
                    regularization=reg,
                ),
                GravityTermSpecification(
                    "production.origin_zone_time",
                    "production",
                    GravityEffectScope.ORIGIN_ZONE_TIME,
                    grouping="origin_zone_time_index",
                    group_count=oz * tp,
                    constraint=GravityConstraint.TWO_WAY_CENTERED,
                    regularization=reg,
                    row_grouping="origin_zone_index",
                    column_grouping="time_period_index",
                    row_group_count=oz,
                    column_group_count=tp,
                ),
                GravityTermSpecification(
                    "destination.zone_time",
                    "destination_attractiveness",
                    GravityEffectScope.DESTINATION_ZONE_TIME,
                    grouping="destination_zone_time_index",
                    group_count=dz * tp,
                    constraint=GravityConstraint.TWO_WAY_CENTERED,
                    regularization=reg,
                    row_grouping="destination_zone_index",
                    column_grouping="time_period_index",
                    row_group_count=dz,
                    column_group_count=tp,
                ),
            )
        )
    if preset == "origin_time_full":
        terms.extend(
            (
                GravityTermSpecification(
                    "production.origin_time",
                    "production",
                    GravityEffectScope.ORIGIN_TIME,
                    grouping="origin_time_group_index",
                    group_count=(
                        int(getattr(features, "num_origin_time_groups"))
                        if features is not None
                        else oz * tp
                    ),
                    regularization=reg,
                ),
                GravityTermSpecification(
                    "destination.zone",
                    "destination_attractiveness",
                    GravityEffectScope.DESTINATION_ZONE,
                    grouping="destination_zone_index",
                    group_count=dz,
                    regularization=reg,
                ),
            )
        )
    return GravityModelSpecification(
        model_name=model_name or preset,
        preset=preset,
        production_source=production_source,
        destination_attractiveness_source=destination_attractiveness_source,
        terms=tuple(terms),
        likelihood=likelihood or GravityLikelihoodSpecification(),
        time=time or GravityTimeSpecification(),
        schema_version=4,
    )
