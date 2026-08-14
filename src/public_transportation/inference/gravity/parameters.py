"""Explicit, serializable parameter blocks for declarative gravity models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.inference.block_coordinate._canonical import fingerprint

from .features import GravityFeatures
from .specification import (
    GravityComponentSpecification,
    GravityConstraint,
    GravityEffectScope,
    GravityModelSpecification,
    GravityParameterization,
    GravityRegularizationType,
)


class MinimalGravityParameters(NamedTuple):
    """Legacy physical view retained for likelihood compatibility."""

    beta_time: jax.Array
    beta_transfer: jax.Array
    dispersion: jax.Array


_DEFAULT_MAPPINGS = {
    GravityEffectScope.ORIGIN: "origin_index",
    GravityEffectScope.DESTINATION: "destination_index",
    GravityEffectScope.TIME_PERIOD: "time_period_index",
    GravityEffectScope.ORIGIN_TIME: "origin_time_group_index",
    GravityEffectScope.DESTINATION_TIME: "destination_time_group_index",
    GravityEffectScope.ORIGIN_ZONE: "origin_zone_index",
    GravityEffectScope.DESTINATION_ZONE: "destination_zone_index",
    GravityEffectScope.ZONE_PAIR: "zone_pair_index",
}


@dataclass(frozen=True, slots=True)
class GravityParameterBlock:
    """One deterministic slice of the flat optimizer vector."""

    component: str
    scope: GravityEffectScope
    parameterization: GravityParameterization
    constraint: GravityConstraint
    mapping: str | None
    group_count: int
    reference_category: int | None
    parameter_slice: slice
    names: tuple[str, ...]
    regularization_type: GravityRegularizationType
    regularization_strength: float

    @property
    def size(self) -> int:
        return self.parameter_slice.stop - self.parameter_slice.start

    def to_dict(self) -> dict[str, object]:
        return {
            "component": self.component,
            "scope": self.scope.value,
            "parameterization": self.parameterization.value,
            "constraint": self.constraint.value,
            "mapping": self.mapping,
            "group_count": self.group_count,
            "reference_category": self.reference_category,
            "start": self.parameter_slice.start,
            "stop": self.parameter_slice.stop,
            "names": list(self.names),
            "regularization": {
                "type": self.regularization_type.value,
                "strength": self.regularization_strength,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> GravityParameterBlock:
        regularization = cast(
            Mapping[str, object], payload.get("regularization", {})
        )
        return cls(
            component=str(payload["component"]),
            scope=GravityEffectScope(str(payload["scope"])),
            parameterization=GravityParameterization(
                str(payload["parameterization"])
            ),
            constraint=GravityConstraint(str(payload["constraint"])),
            mapping=(
                None if payload.get("mapping") is None else str(payload["mapping"])
            ),
            group_count=int(payload.get("group_count", 0)),
            reference_category=(
                None
                if payload.get("reference_category") is None
                else int(payload["reference_category"])
            ),
            parameter_slice=slice(int(payload["start"]), int(payload["stop"])),
            names=tuple(str(item) for item in cast(list[object], payload["names"])),
            regularization_type=GravityRegularizationType(
                str(regularization.get("type", "none"))
            ),
            regularization_strength=float(regularization.get("strength", 0.0)),
        )


def _component_names(component: GravityComponentSpecification) -> tuple[str, ...]:
    count = component.parameter_count
    if count == 0:
        return ()
    legacy = {
        ("journey_time", GravityEffectScope.GLOBAL): "beta_time",
        ("transfer", GravityEffectScope.GLOBAL): "beta_transfer",
        ("dispersion", GravityEffectScope.GLOBAL): "dispersion",
        ("production", GravityEffectScope.GLOBAL): "production_scale",
    }.get((component.name, component.scope))
    if legacy is not None:
        return (legacy,)
    result: list[str] = []
    if component.parameterization is GravityParameterization.POSITIVE:
        result.append(
            {
                "journey_time": "beta_time",
                "transfer": "beta_transfer",
                "waiting_time": "beta_waiting",
                "dispersion": "dispersion",
            }.get(component.name, f"{component.name}.base")
        )
    legacy_prefix = {
        (
            "destination_attractiveness",
            GravityEffectScope.DESTINATION_ZONE,
        ): "destination_zone_deviation",
        ("temporal", GravityEffectScope.TIME_PERIOD): "time_period_deviation",
        ("production", GravityEffectScope.ORIGIN_ZONE): "origin_zone_deviation",
    }.get((component.name, component.scope))
    prefix = legacy_prefix or f"{component.name}.deviation"
    result.extend(f"{prefix}[{index}]" for index in range(count - len(result)))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class GravityParameterLayout:
    """Complete flat-vector contract derived from a model specification."""

    specification: GravityModelSpecification
    positivity_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        if not np.isfinite(self.positivity_floor) or self.positivity_floor <= 0:
            raise ValueError("positivity_floor must be finite and positive.")

    @property
    def blocks(self) -> tuple[GravityParameterBlock, ...]:
        result: list[GravityParameterBlock] = []
        start = 0
        for component in self.specification.active_components:
            names = _component_names(component)
            if not names:
                continue
            stop = start + len(names)
            result.append(
                GravityParameterBlock(
                    component=component.name,
                    scope=component.scope,
                    parameterization=component.parameterization,
                    constraint=component.constraint,
                    mapping=component.grouping
                    or _DEFAULT_MAPPINGS.get(component.scope),
                    group_count=component.group_count,
                    reference_category=component.reference_category,
                    parameter_slice=slice(start, stop),
                    names=names,
                    regularization_type=component.regularization.kind,
                    regularization_strength=component.regularization.strength,
                )
            )
            start = stop
        return tuple(result)

    @property
    def size(self) -> int:
        return sum(block.size for block in self.blocks)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for block in self.blocks for name in block.names)

    @property
    def slices(self) -> dict[str, slice]:
        result: dict[str, slice] = {}
        aliases = {
            "destination_attractiveness": "destination_zone",
            "temporal": "time_period",
            "production": "origin_zone",
        }
        for block in self.blocks:
            result[block.component] = block.parameter_slice
            for name, index in zip(block.names, range(block.parameter_slice.start, block.parameter_slice.stop), strict=True):
                result[name] = slice(index, index + 1)
            legacy = aliases.get(block.component)
            if legacy is not None and block.scope in (
                GravityEffectScope.DESTINATION_ZONE,
                GravityEffectScope.TIME_PERIOD,
                GravityEffectScope.ORIGIN_ZONE,
            ):
                result[legacy] = block.parameter_slice
        return result

    def block(self, component: str) -> GravityParameterBlock | None:
        return next((item for item in self.blocks if item.component == component), None)

    def _raw(self, raw_parameters: object) -> jax.Array:
        raw = jnp.asarray(raw_parameters)
        if raw.ndim != 1 or raw.shape[0] != self.size:
            raise ValueError(f"raw_parameters must have shape ({self.size},).")
        return raw

    def _positive(self, value: jax.Array) -> jax.Array:
        return jax.nn.softplus(value) + jnp.asarray(
            self.positivity_floor, dtype=value.dtype
        )

    def constrained_deviations(
        self, raw_parameters: object, component: str
    ) -> jax.Array:
        """Expand free categorical deviations under the declared constraint."""
        raw = self._raw(raw_parameters)
        block = self.block(component)
        if block is None or block.group_count == 0:
            return jnp.empty((0,), dtype=raw.dtype)
        values = raw[block.parameter_slice]
        if block.parameterization is GravityParameterization.POSITIVE:
            values = values[1:]
        if block.scope is GravityEffectScope.SMOOTH_BASIS:
            return values
        if block.constraint is GravityConstraint.SUM_ZERO:
            return jnp.concatenate((values, -jnp.sum(values, keepdims=True)))
        if block.constraint is GravityConstraint.REFERENCE:
            reference = block.reference_category
            assert reference is not None
            return jnp.concatenate(
                (values[:reference], jnp.zeros((1,), dtype=raw.dtype), values[reference:])
            )
        raise ValueError(f"grouped component {component!r} lacks a constraint.")

    def scalar_or_base(self, raw_parameters: object, component: str) -> jax.Array:
        """Return a fixed/global value or the positive base of a grouped block."""
        raw = self._raw(raw_parameters)
        specification = self.specification.component(component)
        block = self.block(component)
        if block is None:
            value = 0.0 if specification.fixed_value is None else specification.fixed_value
            return jnp.asarray(value, dtype=raw.dtype)
        value = raw[block.parameter_slice.start]
        if block.parameterization is GravityParameterization.POSITIVE:
            return self._positive(value)
        if block.parameterization is GravityParameterization.LOG_MULTIPLIER:
            return value
        return value

    def cell_effect(
        self,
        raw_parameters: object,
        component: str,
        features: GravityFeatures,
    ) -> jax.Array:
        """Return a scalar or per-cell physical effect for one component."""
        raw = self._raw(raw_parameters)
        specification = self.specification.component(component)
        block = self.block(component)
        if block is None or specification.scope in (
            GravityEffectScope.NONE,
            GravityEffectScope.FIXED,
        ):
            return self.scalar_or_base(raw, component)
        if specification.scope is GravityEffectScope.GLOBAL:
            return self.scalar_or_base(raw, component)
        if specification.scope is GravityEffectScope.SMOOTH_BASIS:
            assert block.mapping is not None
            basis = jnp.asarray(features.mapping(block.mapping), dtype=raw.dtype)
            return basis @ raw[block.parameter_slice]
        assert block.mapping is not None
        indices = jnp.asarray(features.mapping(block.mapping), dtype=jnp.int32)
        deviations = self.constrained_deviations(raw, component)[indices]
        if specification.parameterization is GravityParameterization.POSITIVE:
            return self.scalar_or_base(raw, component) * jnp.exp(deviations)
        return deviations

    def production_group_log_multiplier(
        self, raw_parameters: object, features: GravityFeatures
    ) -> jax.Array:
        """Return one production log multiplier per origin-time total."""
        raw = self._raw(raw_parameters)
        component = self.specification.component("production")
        if component.scope in (GravityEffectScope.NONE, GravityEffectScope.FIXED):
            fixed = 0.0 if component.fixed_value is None else component.fixed_value
            return jnp.full(
                (features.num_origin_time_groups,), fixed, dtype=raw.dtype
            )
        if component.scope is GravityEffectScope.GLOBAL:
            return jnp.full(
                (features.num_origin_time_groups,),
                self.scalar_or_base(raw, "production"),
                dtype=raw.dtype,
            )
        cell_effect = self.cell_effect(raw, "production", features)
        groups = jnp.asarray(features.origin_time_group_index, dtype=jnp.int32)
        totals = jax.ops.segment_sum(
            cell_effect, groups, num_segments=features.num_origin_time_groups
        )
        counts = jax.ops.segment_sum(
            jnp.ones(features.num_cells, dtype=raw.dtype),
            groups,
            num_segments=features.num_origin_time_groups,
        )
        return totals / counts

    def production_log_scale(self, raw_parameters: object) -> jax.Array:
        """Legacy accessor for the global production log scale."""
        component = self.specification.component("production")
        if component.scope is GravityEffectScope.GLOBAL:
            return self.scalar_or_base(raw_parameters, "production")
        raw = self._raw(raw_parameters)
        return jnp.asarray(0.0, dtype=raw.dtype)

    @property
    def fingerprint(self) -> str:
        if not self.specification.components:
            return fingerprint(
                {
                    "schema_version": 1,
                    "specification": self.specification.to_dict(),
                    "names": self.names,
                    "positivity_floor": self.positivity_floor,
                }
            )
        return fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "specification": self.specification.to_dict(),
            "blocks": [block.to_dict() for block in self.blocks],
            "names": list(self.names),
            "positivity_floor": self.positivity_floor,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GravityParameterLayout:
        if payload.get("schema_version") not in (1, 2):
            raise ValueError("unsupported gravity parameter-layout schema version.")
        layout = cls(
            GravityModelSpecification.from_dict(
                payload["specification"]  # type: ignore[arg-type]
            ),
            positivity_floor=float(payload.get("positivity_floor", 1.0e-6)),
        )
        if payload.get("names") is not None and tuple(payload["names"]) != layout.names:  # type: ignore[arg-type]
            raise ValueError("serialized gravity parameter names do not match the specification.")
        if payload.get("blocks") is not None:
            restored_blocks = tuple(
                GravityParameterBlock.from_dict(cast(Mapping[str, object], item))
                for item in cast(list[object], payload["blocks"])
            )
            if tuple(block.to_dict() for block in restored_blocks) != tuple(
                block.to_dict() for block in layout.blocks
            ):
                raise ValueError(
                    "serialized gravity parameter blocks do not match the specification."
                )
        return layout

    def transform(self, raw_parameters: object) -> MinimalGravityParameters:
        raw = self._raw(raw_parameters)
        return MinimalGravityParameters(
            self.scalar_or_base(raw, "journey_time"),
            self.scalar_or_base(raw, "transfer"),
            self.scalar_or_base(raw, "dispersion"),
        )

    def centered_effect(self, raw_parameters: object, block: str) -> jax.Array:
        """Legacy centered-effect accessor for the original three relaxations."""
        component = {
            "destination_zone": "destination_attractiveness",
            "time_period": "temporal",
            "origin_zone": "production",
        }.get(block)
        if component is None:
            raise ValueError(f"unknown gravity relaxation block {block!r}.")
        return self.constrained_deviations(raw_parameters, component)

    def regularization(self, raw_parameters: object) -> jax.Array:
        raw = self._raw(raw_parameters)
        result = jnp.asarray(0.0, dtype=raw.dtype)
        for block in self.blocks:
            if block.regularization_type is not GravityRegularizationType.RIDGE:
                continue
            if block.group_count:
                effect = self.constrained_deviations(raw, block.component)
            else:
                effect = raw[block.parameter_slice]
            result = result + 0.5 * jnp.asarray(
                block.regularization_strength, dtype=raw.dtype
            ) * jnp.sum(effect * effect)
        return result

    def physical_vector(self, raw_parameters: object) -> jax.Array:
        raw = self._raw(raw_parameters)
        result = raw
        for block in self.blocks:
            position = block.parameter_slice.start
            if block.parameterization is GravityParameterization.POSITIVE:
                result = result.at[position].set(self._positive(raw[position]))
            elif (
                block.parameterization is GravityParameterization.LOG_MULTIPLIER
                and block.scope is GravityEffectScope.GLOBAL
            ):
                result = result.at[position].set(jnp.exp(raw[position]))
        return result

    def raw_from_physical(self, physical_parameters: object) -> np.ndarray:
        values = np.asarray(physical_parameters)
        if values.shape != (self.size,) or values.dtype.kind not in "iuf":
            raise ValueError(f"physical_parameters must have shape ({self.size},).")
        if not np.all(np.isfinite(values)):
            raise ValueError("physical parameters must be finite.")
        result = values.astype(np.float64, copy=True)
        for block in self.blocks:
            position = block.parameter_slice.start
            if block.parameterization is GravityParameterization.POSITIVE:
                shifted = float(values[position]) - self.positivity_floor
                if shifted <= 0:
                    raise ValueError(
                        f"physical parameter {block.names[0]!r} must exceed positivity_floor."
                    )
                result[position] = shifted + np.log(-np.expm1(-shifted))
            elif (
                block.parameterization is GravityParameterization.LOG_MULTIPLIER
                and block.scope is GravityEffectScope.GLOBAL
            ):
                scale = float(values[position])
                if scale <= 0:
                    raise ValueError("global production scale must be strictly positive.")
                result[position] = np.log(scale)
        return result


def warm_start_gravity_parameters(
    parent_layout: GravityParameterLayout,
    child_layout: GravityParameterLayout,
    parent_raw_parameters: object,
) -> np.ndarray:
    """Embed a parent iterate in a nested child; new deviations start at zero."""
    parent = np.asarray(parent_raw_parameters)
    if parent.shape != (parent_layout.size,):
        raise ValueError(
            f"parent_raw_parameters must have shape ({parent_layout.size},)."
        )
    child = np.zeros(child_layout.size, dtype=parent.dtype)
    parent_positions = {name: index for index, name in enumerate(parent_layout.names)}
    for index, name in enumerate(child_layout.names):
        if name in parent_positions:
            child[index] = parent[parent_positions[name]]
    missing = set(parent_layout.names) - set(child_layout.names)
    if missing:
        raise ValueError("child layout does not contain every parent parameter.")
    return child


def validate_gravity_relaxation_features(
    features: GravityFeatures, specification: GravityModelSpecification
) -> None:
    """Validate every feature mapping required by the declarative specification."""
    for component in specification.active_components:
        if component.scope not in _DEFAULT_MAPPINGS and component.scope not in (
            GravityEffectScope.CUSTOM_GROUP,
            GravityEffectScope.SMOOTH_BASIS,
        ):
            continue
        mapping = component.grouping or _DEFAULT_MAPPINGS.get(component.scope)
        if mapping is None:
            raise ValueError(
                f"component {component.name!r} requires an explicit feature mapping."
            )
        constant = component.name == "production" or mapping in {
            "origin_index",
            "time_period_index",
            "origin_time_group_index",
            "origin_zone_index",
        }
        features.validate_mapping(
            mapping,
            group_count=component.group_count,
            constant_within_origin_time=constant,
            smooth_basis=component.scope is GravityEffectScope.SMOOTH_BASIS,
        )
