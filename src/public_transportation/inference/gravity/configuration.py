"""Strict YAML loading and user-facing validation for gravity specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

import numpy as np
import yaml

from .features import GravityFeatures
from .parameters import (
    GravityParameterLayout,
    validate_gravity_relaxation_features,
)
from .specification import (
    GravityComponentSpecification,
    GravityConstraint,
    GravityEffectScope,
    GravityLikelihoodSpecification,
    GravityModelSpecification,
    GravityParameterization,
    GravityRegularization,
    GravityTimeSpecification,
)


_DEFAULT_GROUPINGS = {
    GravityEffectScope.ORIGIN: "origin_index",
    GravityEffectScope.DESTINATION: "destination_index",
    GravityEffectScope.TIME_PERIOD: "time_period_index",
    GravityEffectScope.ORIGIN_TIME: "origin_time_group_index",
    GravityEffectScope.DESTINATION_TIME: "destination_time_group_index",
    GravityEffectScope.ORIGIN_ZONE: "origin_zone_index",
    GravityEffectScope.DESTINATION_ZONE: "destination_zone_index",
    GravityEffectScope.ZONE_PAIR: "zone_pair_index",
    GravityEffectScope.SMOOTH_BASIS: "smooth_time_basis",
}


@dataclass(frozen=True, slots=True)
class GravitySpecificationValidation:
    """Resolved specification, layout, diagnostics, and printable summary."""

    specification: GravityModelSpecification
    parameter_layout: GravityParameterLayout
    warnings: tuple[str, ...]
    calibration_rows: int | None
    excluded_unsupported_rows: int | None
    holdout_rows: int | None
    feature_cache_fingerprint: str | None
    structural_zero_fingerprint: str | None

    @property
    def summary(self) -> str:
        lines = [
            f"Gravity model: {self.specification.model_name}",
            f"Specification fingerprint: {self.specification.fingerprint}",
            f"Parameters: {self.parameter_layout.size}",
            "Active components:",
        ]
        for component in self.specification.active_components:
            lines.append(
                f"  - {component.name}: {component.scope.value} "
                f"({component.parameter_count} parameters)"
            )
        lines.append("Parameter blocks:")
        for block in self.parameter_layout.blocks:
            regularization = (
                "none"
                if block.regularization_strength == 0
                else f"ridge={block.regularization_strength:g}"
            )
            lines.append(
                f"  - {block.component}[{block.parameter_slice.start}:"
                f"{block.parameter_slice.stop}]: {', '.join(block.names)}; "
                f"{regularization}"
            )
        required = self.specification.required_feature_mappings
        lines.append(
            "Required feature mappings: " + (", ".join(required) if required else "none")
        )
        lines.extend(
            (
                f"Calibration rows: {self.calibration_rows if self.calibration_rows is not None else 'not supplied'}",
                "Excluded unsupported rows: "
                f"{self.excluded_unsupported_rows if self.excluded_unsupported_rows is not None else 'not supplied'}",
                f"Holdout rows: {self.holdout_rows if self.holdout_rows is not None else 'not supplied'}",
                f"Feature-cache fingerprint: {self.feature_cache_fingerprint or 'not supplied'}",
                "Structural-zero fingerprint: "
                f"{self.structural_zero_fingerprint or 'not supplied'}",
            )
        )
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {message}" for message in self.warnings)
        else:
            lines.append("Warnings: none")
        return "\n".join(lines)


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a YAML mapping.")
    return cast(Mapping[str, object], value)


def _reject_unknown(
    payload: Mapping[str, object], allowed: set[str], *, context: str
) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unsupported {context} options: {sorted(unknown)}.")


def _component(
    *,
    name: str,
    payload: Mapping[str, object],
    parameterization: GravityParameterization,
    features: GravityFeatures | None,
    default_scope: GravityEffectScope,
    source: str | None = None,
) -> GravityComponentSpecification:
    _reject_unknown(
        payload,
        {
            "scope",
            "grouping",
            "group_count",
            "constraint",
            "reference_category",
            "regularization",
            "fixed_value",
            "source",
            "parameterization",
        },
        context=f"{name} component",
    )
    raw_scope = str(payload.get("scope", default_scope.value))
    if name == "temporal" and raw_scope == "categorical":
        raw_scope = GravityEffectScope.TIME_PERIOD.value
    scope = GravityEffectScope(raw_scope)
    fixed = scope in (GravityEffectScope.NONE, GravityEffectScope.FIXED)
    declared_parameterization = GravityParameterization(
        str(
            payload.get(
                "parameterization",
                GravityParameterization.FIXED.value if fixed else parameterization.value,
            )
        )
    )
    grouping = (
        None
        if payload.get("grouping") is None
        else str(payload.get("grouping"))
    ) or _DEFAULT_GROUPINGS.get(scope)
    group_count = int(payload.get("group_count", 0))
    if scope not in (
        GravityEffectScope.NONE,
        GravityEffectScope.FIXED,
        GravityEffectScope.GLOBAL,
    ):
        if grouping is None:
            raise ValueError(f"component {name!r} requires a grouping mapping.")
        if features is not None:
            values = features.mapping(grouping)
            if values is None:
                raise ValueError(
                    f"feature mapping {grouping!r} is required by component {name!r}."
                )
            array = np.asarray(values)
            inferred = array.shape[1] if scope is GravityEffectScope.SMOOTH_BASIS else np.unique(array).size
            if group_count not in (0, inferred):
                raise ValueError(
                    f"component {name!r} declares group_count={group_count}, but "
                    f"mapping {grouping!r} contains {inferred} groups."
                )
            group_count = int(inferred)
        elif group_count == 0:
            raise ValueError(
                f"component {name!r} requires group_count when features are not supplied."
            )
    constraint_default = (
        GravityConstraint.NONE
        if scope in (
            GravityEffectScope.NONE,
            GravityEffectScope.FIXED,
            GravityEffectScope.GLOBAL,
            GravityEffectScope.SMOOTH_BASIS,
        )
        else GravityConstraint.SUM_ZERO
    )
    regularization = GravityRegularization.from_dict(
        dict(_mapping(payload.get("regularization"), context=f"{name}.regularization"))
    )
    return GravityComponentSpecification(
        name=name,
        scope=scope,
        parameterization=declared_parameterization,
        grouping=grouping,
        group_count=group_count,
        constraint=GravityConstraint(
            str(payload.get("constraint", constraint_default.value))
        ),
        reference_category=(
            None
            if payload.get("reference_category") is None
            else int(cast(int, payload["reference_category"]))
        ),
        regularization=regularization,
        fixed_value=(
            None
            if payload.get("fixed_value") is None
            else float(cast(float, payload["fixed_value"]))
        ),
        source=str(payload.get("source", source)) if payload.get("source", source) is not None else None,
    )


def gravity_model_specification_from_mapping(
    payload: Mapping[str, object], *, features: GravityFeatures | None = None
) -> GravityModelSpecification:
    """Construct a strict declarative model from parsed YAML data."""
    _reject_unknown(
        payload,
        {
            "schema_version",
            "model_name",
            "likelihood",
            "time",
            "production",
            "destination_attractiveness",
            "utility",
            "dispersion",
            "residual_demand",
        },
        context="gravity specification",
    )
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("unsupported gravity YAML schema_version; expected 1.")
    likelihood_payload = _mapping(payload.get("likelihood"), context="likelihood")
    _reject_unknown(
        likelihood_payload,
        {"family", "calibration_mask", "detection_rate_estimated"},
        context="likelihood",
    )
    time_payload = _mapping(payload.get("time"), context="time")
    _reject_unknown(
        time_payload,
        {"units", "interpretation", "bin_labels", "smooth_basis_name"},
        context="time",
    )
    production_payload = _mapping(payload.get("production"), context="production")
    _reject_unknown(production_payload, {"baseline", "correction"}, context="production")
    if production_payload.get("baseline", "origin_time_totals") != "origin_time_totals":
        raise ValueError("only production baseline 'origin_time_totals' is implemented.")
    destination_payload = _mapping(
        payload.get("destination_attractiveness"),
        context="destination_attractiveness",
    )
    _reject_unknown(
        destination_payload,
        {"source", "correction"},
        context="destination_attractiveness",
    )
    utility = _mapping(payload.get("utility"), context="utility")
    _reject_unknown(
        utility,
        {"journey_time", "transfers", "transfer", "waiting_time", "departure_time"},
        context="utility",
    )
    if "transfers" in utility and "transfer" in utility:
        raise ValueError("utility may define transfers or transfer, not both.")
    likelihood_family = str(
        likelihood_payload.get("family", "negative_binomial")
    )
    components = (
        _component(
            name="journey_time",
            payload=_mapping(utility.get("journey_time"), context="utility.journey_time"),
            parameterization=GravityParameterization.POSITIVE,
            features=features,
            default_scope=(
                GravityEffectScope.NONE
                if likelihood_family == "poisson"
                else GravityEffectScope.GLOBAL
            ),
        ),
        _component(
            name="transfer",
            payload=_mapping(utility.get("transfers", utility.get("transfer")), context="utility.transfers"),
            parameterization=GravityParameterization.POSITIVE,
            features=features,
            default_scope=GravityEffectScope.GLOBAL,
        ),
        _component(
            name="dispersion",
            payload=_mapping(payload.get("dispersion"), context="dispersion"),
            parameterization=GravityParameterization.POSITIVE,
            features=features,
            default_scope=GravityEffectScope.GLOBAL,
        ),
        _component(
            name="waiting_time",
            payload=_mapping(utility.get("waiting_time"), context="utility.waiting_time"),
            parameterization=GravityParameterization.POSITIVE,
            features=features,
            default_scope=GravityEffectScope.NONE,
        ),
        _component(
            name="production",
            payload=_mapping(production_payload.get("correction"), context="production.correction"),
            parameterization=GravityParameterization.LOG_MULTIPLIER,
            features=features,
            default_scope=GravityEffectScope.NONE,
            source="origin_time_totals",
        ),
        _component(
            name="destination_attractiveness",
            payload=_mapping(destination_payload.get("correction"), context="destination_attractiveness.correction"),
            parameterization=GravityParameterization.ADDITIVE,
            features=features,
            default_scope=GravityEffectScope.FIXED,
            source=str(destination_payload.get("source", "feature_cache")),
        ),
        _component(
            name="temporal",
            payload=_mapping(utility.get("departure_time"), context="utility.departure_time"),
            parameterization=GravityParameterization.ADDITIVE,
            features=features,
            default_scope=GravityEffectScope.NONE,
        ),
        _component(
            name="residual_demand",
            payload=_mapping(payload.get("residual_demand"), context="residual_demand"),
            parameterization=GravityParameterization.FIXED,
            features=features,
            default_scope=GravityEffectScope.NONE,
        ),
    )
    return GravityModelSpecification(
        model_name=str(payload.get("model_name", "gravity_model")),
        components=components,
        likelihood=GravityLikelihoodSpecification(
            family=likelihood_family,
            calibration_mask=str(
                likelihood_payload.get("calibration_mask", "supported_measurements")
            ),
            detection_rate_estimated=bool(
                likelihood_payload.get("detection_rate_estimated", False)
            ),
        ),
        time=GravityTimeSpecification(
            units=str(time_payload.get("units", "index")),
            interpretation=str(
                time_payload.get("interpretation", "categorical departure-time bins")
            ),
            bin_labels=tuple(str(item) for item in time_payload.get("bin_labels", ())),  # type: ignore[union-attr]
            smooth_basis_name=(
                None
                if time_payload.get("smooth_basis_name") is None
                else str(time_payload["smooth_basis_name"])
            ),
        ),
    )


def validate_gravity_model_specification(
    specification: GravityModelSpecification,
    *,
    features: GravityFeatures | None = None,
    calibration_mask: object | None = None,
    unsupported_measurement_mask: object | None = None,
    holdout_mask: object | None = None,
    structural_zero_fingerprint: str | None = None,
) -> GravitySpecificationValidation:
    """Validate mappings and observation masks and construct the final layout."""
    if features is not None:
        validate_gravity_relaxation_features(features, specification)
        if specification.time.bin_labels and (
            len(specification.time.bin_labels) != features.num_departure_times
        ):
            raise ValueError(
                "time.bin_labels must match the number of departure-time bins."
            )
        waiting = specification.component("waiting_time")
        if waiting.scope not in (GravityEffectScope.NONE, GravityEffectScope.FIXED) and features.initial_waiting_time is None:
            raise ValueError("initial_waiting_time is required by the waiting-time model.")
    masks: dict[str, np.ndarray | None] = {}
    for name, value in (
        ("calibration", calibration_mask),
        ("unsupported", unsupported_measurement_mask),
        ("holdout", holdout_mask),
    ):
        masks[name] = None if value is None else np.asarray(value, dtype=bool)
    supplied_lengths = {item.size for item in masks.values() if item is not None}
    if len(supplied_lengths) > 1:
        raise ValueError("calibration, unsupported, and holdout masks must have equal length.")
    calibration = masks["calibration"]
    unsupported = masks["unsupported"]
    if calibration is not None and unsupported is not None and np.any(calibration & unsupported):
        raise ValueError("unsupported measurement rows must be excluded from calibration.")
    layout = GravityParameterLayout(specification)
    return GravitySpecificationValidation(
        specification=specification,
        parameter_layout=layout,
        warnings=specification.identifiability_warnings(),
        calibration_rows=None if calibration is None else int(np.count_nonzero(calibration)),
        excluded_unsupported_rows=None if unsupported is None else int(np.count_nonzero(unsupported)),
        holdout_rows=None if masks["holdout"] is None else int(np.count_nonzero(masks["holdout"])),
        feature_cache_fingerprint=None if features is None else features.fingerprint,
        structural_zero_fingerprint=structural_zero_fingerprint,
    )


def load_gravity_model_specification(
    path: str | Path,
    *,
    features: GravityFeatures | None = None,
    calibration_mask: object | None = None,
    unsupported_measurement_mask: object | None = None,
    holdout_mask: object | None = None,
    structural_zero_fingerprint: str | None = None,
) -> GravitySpecificationValidation:
    """Read strict YAML, resolve feature-dependent groups, and return diagnostics."""
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read gravity specification {source}.") from error
    if not isinstance(payload, Mapping):
        raise ValueError("gravity specification YAML must contain one mapping.")
    specification = gravity_model_specification_from_mapping(payload, features=features)
    return validate_gravity_model_specification(
        specification,
        features=features,
        calibration_mask=calibration_mask,
        unsupported_measurement_mask=unsupported_measurement_mask,
        holdout_mask=holdout_mask,
        structural_zero_fingerprint=structural_zero_fingerprint,
    )
