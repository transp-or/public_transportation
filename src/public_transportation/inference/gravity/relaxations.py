"""Atomic, auditable Phase-5 gravity-model relaxations."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .features import GravityFeatures
from .specification import GravityEffectScope, GravityModelSpecification


@dataclass(frozen=True, slots=True)
class GravityRelaxationInfo:
    scope: GravityEffectScope
    added_parameter_count: int
    description: str
    execution_impact: str


def add_gravity_relaxation(
    parent: GravityModelSpecification,
    *,
    features: GravityFeatures,
    scope: GravityEffectScope,
    ridge: float = 1.0,
) -> tuple[GravityModelSpecification, GravityRelaxationInfo]:
    """Create one nested child specification from a user-supplied mapping."""
    definitions = {
        GravityEffectScope.DESTINATION_ZONE: (
            "destination_zone_index",
            "destination_attractiveness_scope",
            "destination_zone_count",
            "destination_zone_ridge",
            "Adds centered destination-zone log-attractiveness deviations.",
            "Adds K-1 parameters and one indexed cell-vector addition.",
        ),
        GravityEffectScope.TIME_PERIOD: (
            "time_period_index",
            "temporal_basis_scope",
            "time_period_count",
            "time_period_ridge",
            "Adds centered broad-period deviations to the journey-time coefficient.",
            "Adds K-1 parameters and one indexed coefficient multiplication.",
        ),
        GravityEffectScope.ORIGIN_ZONE: (
            "origin_zone_index",
            "origin_total_correction_scope",
            "origin_zone_count",
            "origin_zone_ridge",
            "Adds centered multiplicative origin-zone production corrections.",
            "Adds K-1 parameters and one indexed group-total multiplication.",
        ),
    }
    if scope not in definitions:
        raise ValueError(f"scope {scope.value!r} is not an atomic Phase-5 relaxation.")
    feature_name, scope_name, _, _, description, impact = definitions[scope]
    if getattr(parent, scope_name) is not GravityEffectScope.NONE:
        raise ValueError(f"{scope_name} is already active.")
    mapping = getattr(features, feature_name)
    if mapping is None:
        raise ValueError(f"{feature_name} is required for {scope.value}.")
    unique = np.unique(mapping)
    if not np.array_equal(unique, np.arange(unique.size)) or unique.size < 2:
        raise ValueError(f"{feature_name} must contain at least two contiguous indices from zero.")
    count = int(unique.size)
    if scope is GravityEffectScope.DESTINATION_ZONE:
        child = replace(
            parent,
            destination_attractiveness_scope=scope,
            destination_zone_count=count,
            destination_zone_ridge=ridge,
        )
    elif scope is GravityEffectScope.TIME_PERIOD:
        child = replace(
            parent,
            temporal_basis_scope=scope,
            time_period_count=count,
            time_period_ridge=ridge,
        )
    else:
        child = replace(
            parent,
            origin_total_correction_scope=scope,
            origin_zone_count=count,
            origin_zone_ridge=ridge,
        )
    return child, GravityRelaxationInfo(
        scope=scope,
        added_parameter_count=int(unique.size - 1),
        description=description,
        execution_impact=impact,
    )


def remove_gravity_relaxation(
    child: GravityModelSpecification, scope: GravityEffectScope
) -> GravityModelSpecification:
    """Return the exact parent contract for one atomic relaxation."""
    fields = {
        GravityEffectScope.DESTINATION_ZONE: "destination_attractiveness_scope",
        GravityEffectScope.TIME_PERIOD: "temporal_basis_scope",
        GravityEffectScope.ORIGIN_ZONE: "origin_total_correction_scope",
    }
    if scope not in fields:
        raise ValueError(f"scope {scope.value!r} is not an atomic Phase-5 relaxation.")
    scope_name = fields[scope]
    if getattr(child, scope_name) is not scope:
        raise ValueError(f"{scope_name} is not active.")
    if scope is GravityEffectScope.DESTINATION_ZONE:
        return replace(
            child,
            destination_attractiveness_scope=GravityEffectScope.NONE,
            destination_zone_count=0,
            destination_zone_ridge=1.0,
        )
    if scope is GravityEffectScope.TIME_PERIOD:
        return replace(
            child,
            temporal_basis_scope=GravityEffectScope.NONE,
            time_period_count=0,
            time_period_ridge=1.0,
        )
    return replace(
        child,
        origin_total_correction_scope=GravityEffectScope.NONE,
        origin_zone_count=0,
        origin_zone_ridge=1.0,
    )
