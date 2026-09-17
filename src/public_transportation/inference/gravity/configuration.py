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
    GravityDeviationSpecification,
    GravityEffectScope,
    GravityLikelihoodSpecification,
    GravityModelSpecification,
    GravityParameterization,
    GravityRegularization,
    GravityTimeSpecification,
    GravityTermSpecification,
    gravity_model_specification_from_preset,
)


_DEFAULT_GROUPINGS = {
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
    GravityEffectScope.SMOOTH_BASIS: "smooth_time_basis",
}

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
        if self.specification.terms:
            lines.append("Expanded predictor terms:")
            for term in self.specification.terms:
                lines.append(
                    f"  - {term.name}: target={term.target}, scope={term.scope.value} "
                    f"({term.parameter_count} parameters)"
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
            "Required feature mappings: "
            + (", ".join(required) if required else "none")
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


def _reject_obsolete_od_matrix_fields(
    payload: Mapping[str, object], *, context: str
) -> None:
    """Reject removed matrix-valued gravity inputs before schema parsing."""

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
            "deviation",
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
                GravityParameterization.FIXED.value
                if fixed
                else parameterization.value,
            )
        )
    )
    grouping = (
        None if payload.get("grouping") is None else str(payload.get("grouping"))
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
            inferred = (
                array.shape[1]
                if scope is GravityEffectScope.SMOOTH_BASIS
                else np.unique(array).size
            )
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
        if scope
        in (
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
    deviation_payload = _mapping(payload.get("deviation"), context=f"{name}.deviation")
    deviation: GravityDeviationSpecification | None = None
    if deviation_payload:
        _reject_unknown(
            deviation_payload,
            {
                "scope",
                "grouping",
                "group_count",
                "constraint",
                "reference_category",
                "regularization",
            },
            context=f"{name}.deviation",
        )
        deviation_scope = GravityEffectScope(
            str(deviation_payload.get("scope", GravityEffectScope.TIME_PERIOD.value))
        )
        deviation_grouping = (
            None
            if deviation_payload.get("grouping") is None
            else str(deviation_payload.get("grouping"))
        ) or _DEFAULT_GROUPINGS.get(deviation_scope)
        deviation_count = int(deviation_payload.get("group_count", 0))
        if deviation_grouping is None:
            raise ValueError(
                f"component {name!r} deviation requires a grouping mapping."
            )
        if features is not None:
            values = features.mapping(deviation_grouping)
            if values is None:
                raise ValueError(
                    f"feature mapping {deviation_grouping!r} is required by component "
                    f"{name!r} deviation."
                )
            inferred = np.unique(np.asarray(values)).size
            if deviation_count not in (0, inferred):
                raise ValueError(
                    f"component {name!r} deviation declares group_count={deviation_count}, "
                    f"but mapping {deviation_grouping!r} contains {inferred} groups."
                )
            deviation_count = int(inferred)
        if deviation_count == 0:
            raise ValueError(
                f"component {name!r} deviation requires group_count when features "
                "are not supplied."
            )
        deviation_constraint = GravityConstraint(
            str(deviation_payload.get("constraint", GravityConstraint.SUM_ZERO.value))
        )
        deviation = GravityDeviationSpecification(
            scope=deviation_scope,
            grouping=deviation_grouping,
            group_count=deviation_count,
            constraint=deviation_constraint,
            reference_category=(
                None
                if deviation_payload.get("reference_category") is None
                else int(deviation_payload["reference_category"])
            ),
            regularization=GravityRegularization.from_dict(
                dict(
                    _mapping(
                        deviation_payload.get("regularization"),
                        context=f"{name}.deviation.regularization",
                    )
                )
            ),
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
        source=str(payload.get("source", source))
        if payload.get("source", source) is not None
        else None,
        deviation=deviation,
    )


def _term(
    payload: Mapping[str, object],
    *,
    target: str,
    features: GravityFeatures | None,
) -> GravityTermSpecification:
    """Parse one composable production/destination term."""
    _reject_unknown(
        payload,
        {
            "name",
            "target",
            "scope",
            "grouping",
            "group_count",
            "constraint",
            "reference_category",
            "regularization",
            "parameterization",
            "fixed_value",
            "row_grouping",
            "column_grouping",
            "row_group_count",
            "column_group_count",
        },
        context="gravity term",
    )
    scope = GravityEffectScope(str(payload.get("scope", "none")))
    grouping = None if payload.get("grouping") is None else str(payload["grouping"])
    grouping = grouping or _DEFAULT_GROUPINGS.get(scope)
    group_count = int(payload.get("group_count", 0))
    if scope not in (
        GravityEffectScope.NONE,
        GravityEffectScope.FIXED,
        GravityEffectScope.GLOBAL,
    ):
        if grouping is None:
            raise ValueError("grouped gravity terms require a grouping mapping.")
        if features is not None:
            values = features.mapping(grouping)
            if values is None:
                raise ValueError(
                    f"feature mapping {grouping!r} is required by the term."
                )
            inferred = int(np.unique(np.asarray(values)).size)
            if group_count not in (0, inferred):
                raise ValueError(
                    f"term {payload.get('name', '<unnamed>')!r} declares group_count={group_count}, "
                    f"but mapping {grouping!r} contains {inferred} groups."
                )
            group_count = inferred
        elif group_count == 0:
            raise ValueError("group_count is required when features are not supplied.")
    constraint_default = (
        GravityConstraint.NONE
        if scope
        in (
            GravityEffectScope.NONE,
            GravityEffectScope.FIXED,
            GravityEffectScope.GLOBAL,
        )
        else GravityConstraint.SUM_ZERO
    )
    parameterization_default = (
        GravityParameterization.FIXED
        if scope in (GravityEffectScope.NONE, GravityEffectScope.FIXED)
        else GravityParameterization.ADDITIVE
    )
    row_grouping = (
        None if payload.get("row_grouping") is None else str(payload["row_grouping"])
    )
    column_grouping = (
        None
        if payload.get("column_grouping") is None
        else str(payload["column_grouping"])
    )
    if scope is GravityEffectScope.ORIGIN_ZONE_TIME:
        row_grouping = row_grouping or "origin_zone_index"
        column_grouping = column_grouping or "time_period_index"
    elif scope is GravityEffectScope.DESTINATION_ZONE_TIME:
        row_grouping = row_grouping or "destination_zone_index"
        column_grouping = column_grouping or "time_period_index"
    row_group_count = int(payload.get("row_group_count", 0))
    column_group_count = int(payload.get("column_group_count", 0))
    if features is not None:
        for grouping_name, count in (
            (row_grouping, row_group_count),
            (column_grouping, column_group_count),
        ):
            if grouping_name is None or count:
                continue
            values = features.mapping(grouping_name)
            if values is None:
                raise ValueError(
                    f"feature mapping {grouping_name!r} is required by the term."
                )
            inferred = int(np.unique(np.asarray(values)).size)
            if grouping_name == row_grouping:
                row_group_count = inferred
            else:
                column_group_count = inferred
    return GravityTermSpecification(
        name=str(payload.get("name", "")),
        target=target,
        scope=scope,
        grouping=grouping,
        group_count=group_count,
        constraint=GravityConstraint(
            str(payload.get("constraint", constraint_default.value))
        ),
        reference_category=(
            None
            if payload.get("reference_category") is None
            else int(payload["reference_category"])
        ),
        regularization=GravityRegularization.from_dict(
            dict(_mapping(payload.get("regularization"), context="term.regularization"))
        ),
        parameterization=GravityParameterization(
            str(payload.get("parameterization", parameterization_default.value))
        ),
        fixed_value=(
            None
            if payload.get("fixed_value") is None
            else float(payload["fixed_value"])
        ),
        row_grouping=row_grouping,
        column_grouping=column_grouping,
        row_group_count=row_group_count,
        column_group_count=column_group_count,
    )


def gravity_model_specification_from_mapping(
    payload: Mapping[str, object], *, features: GravityFeatures | None = None
) -> GravityModelSpecification:
    """Construct a strict declarative model from parsed YAML data."""
    _reject_obsolete_od_matrix_fields(payload, context="gravity specification")
    _reject_unknown(
        payload,
        {
            "schema_version",
            "model_name",
            "preset",
            "production_source",
            "likelihood",
            "time",
            "production",
            "destination_attractiveness",
            "utility",
            "dispersion",
            "residual_demand",
            "terms",
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
    _reject_obsolete_od_matrix_fields(production_payload, context="production")
    if "baseline" in production_payload:
        baseline = production_payload["baseline"]
        if isinstance(baseline, (Mapping, list, tuple, np.ndarray)):
            raise ValueError(
                "a priori OD matrix is obsolete and unsupported; "
                "matrix-valued a priori OD matrices are not accepted; "
                "production.baseline must be a source name, not a matrix-valued "
                "object. Use production_source='unit_exposure' or the one-dimensional "
                "'origin_time_totals' source."
            )
    _reject_unknown(
        production_payload,
        {"baseline", "source", "correction", "terms"},
        context="production",
    )
    preset_name = (
        None if payload.get("preset") is None else str(payload["preset"]).lower()
    )
    default_production_source = (
        "unit_exposure"
        if preset_name in {"zone", "zone_time", "origin_time_full"}
        else "origin_time_totals"
    )
    production_source_value = payload.get(
        "production_source",
        production_payload.get(
            "source", production_payload.get("baseline", default_production_source)
        ),
    )
    if isinstance(production_source_value, (Mapping, list, tuple, np.ndarray)):
        raise ValueError(
            "a priori OD matrix is obsolete and unsupported; "
            "matrix-valued a priori OD matrices are not accepted; "
            "production source must be a name, not a matrix-valued object. Use "
            "production_source='unit_exposure' or the one-dimensional "
            "'origin_time_totals' source."
        )
    production_source = str(production_source_value)
    destination_payload = _mapping(
        payload.get("destination_attractiveness"),
        context="destination_attractiveness",
    )
    destination_source = str(destination_payload.get("source", "feature_cache"))
    if destination_source not in {"feature_cache", "external", "baseline_derived"}:
        raise ValueError("unsupported destination attractiveness source.")
    _reject_unknown(
        destination_payload,
        {"source", "correction", "terms"},
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
    likelihood_family = str(likelihood_payload.get("family", "negative_binomial"))
    components = (
        _component(
            name="journey_time",
            payload=_mapping(
                utility.get("journey_time"), context="utility.journey_time"
            ),
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
            payload=_mapping(
                utility.get("transfers", utility.get("transfer")),
                context="utility.transfers",
            ),
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
            payload=_mapping(
                utility.get("waiting_time"), context="utility.waiting_time"
            ),
            parameterization=GravityParameterization.POSITIVE,
            features=features,
            default_scope=GravityEffectScope.NONE,
        ),
        _component(
            name="production",
            payload=_mapping(
                production_payload.get("correction"), context="production.correction"
            ),
            parameterization=GravityParameterization.LOG_MULTIPLIER,
            features=features,
            default_scope=GravityEffectScope.NONE,
            source="origin_time_totals",
        ),
        _component(
            name="destination_attractiveness",
            payload=_mapping(
                destination_payload.get("correction"),
                context="destination_attractiveness.correction",
            ),
            parameterization=GravityParameterization.ADDITIVE,
            features=features,
            default_scope=GravityEffectScope.FIXED,
            source=str(destination_payload.get("source", "feature_cache")),
        ),
        _component(
            name="temporal",
            payload=_mapping(
                utility.get("departure_time"), context="utility.departure_time"
            ),
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
    raw_terms = payload.get("terms", ())
    if raw_terms is None:
        raw_terms = ()
    if not isinstance(raw_terms, (list, tuple)):
        raise TypeError("gravity terms must be a sequence.")
    production_terms_payload = production_payload.get("terms", ())
    destination_terms_payload = destination_payload.get("terms", ())
    if not isinstance(production_terms_payload, (list, tuple)) or not isinstance(
        destination_terms_payload, (list, tuple)
    ):
        raise TypeError("production/destination terms must be sequences.")
    terms = (
        [
            _term(
                cast(Mapping[str, object], item), target="production", features=features
            )
            for item in production_terms_payload
        ]
        + [
            _term(
                cast(Mapping[str, object], item),
                target="destination_attractiveness",
                features=features,
            )
            for item in destination_terms_payload
        ]
        + [
            _term(
                cast(Mapping[str, object], item),
                target=str(
                    cast(Mapping[str, object], item).get("target", "production")
                ),
                features=features,
            )
            for item in raw_terms
        ]
    )
    if payload.get("preset") is not None:
        if terms:
            raise ValueError(
                "a named gravity preset cannot be combined with explicit terms."
            )
        return gravity_model_specification_from_preset(
            str(payload["preset"]),
            features=features,
            production_source=production_source,
            destination_attractiveness_source=destination_source,
            model_name=(
                None
                if payload.get("model_name") is None
                else str(payload["model_name"])
            ),
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
                    time_payload.get(
                        "interpretation", "categorical departure-time bins"
                    )
                ),
                bin_labels=tuple(
                    str(item) for item in time_payload.get("bin_labels", ())
                ),
                smooth_basis_name=(
                    None
                    if time_payload.get("smooth_basis_name") is None
                    else str(time_payload["smooth_basis_name"])
                ),
            ),
        )
    if terms:
        components = tuple(
            item
            for item in components
            if item.name not in {"production", "destination_attractiveness"}
        )
    return GravityModelSpecification(
        model_name=str(payload.get("model_name", "gravity_model")),
        components=components,
        terms=tuple(terms),
        production_source=production_source,
        destination_attractiveness_source=destination_source,
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
        schema_version=(
            4
            if terms
            or production_source != "origin_time_totals"
            or destination_source != "feature_cache"
            else 3
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
        if (
            waiting.scope not in (GravityEffectScope.NONE, GravityEffectScope.FIXED)
            and features.initial_waiting_time is None
        ):
            raise ValueError(
                "initial_waiting_time is required by the waiting-time model."
            )
    masks: dict[str, np.ndarray | None] = {}
    for name, value in (
        ("calibration", calibration_mask),
        ("unsupported", unsupported_measurement_mask),
        ("holdout", holdout_mask),
    ):
        masks[name] = None if value is None else np.asarray(value, dtype=bool)
    supplied_lengths = {item.size for item in masks.values() if item is not None}
    if len(supplied_lengths) > 1:
        raise ValueError(
            "calibration, unsupported, and holdout masks must have equal length."
        )
    calibration = masks["calibration"]
    unsupported = masks["unsupported"]
    if (
        calibration is not None
        and unsupported is not None
        and np.any(calibration & unsupported)
    ):
        raise ValueError(
            "unsupported measurement rows must be excluded from calibration."
        )
    layout = GravityParameterLayout(specification)
    return GravitySpecificationValidation(
        specification=specification,
        parameter_layout=layout,
        warnings=specification.identifiability_warnings(),
        calibration_rows=None
        if calibration is None
        else int(np.count_nonzero(calibration)),
        excluded_unsupported_rows=None
        if unsupported is None
        else int(np.count_nonzero(unsupported)),
        holdout_rows=None
        if masks["holdout"] is None
        else int(np.count_nonzero(masks["holdout"])),
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
