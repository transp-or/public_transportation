"""Explicit, serializable parameter blocks for declarative gravity models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.inference.block_coordinate._canonical import (
    canonical_json,
    fingerprint,
)

from .features import GravityFeatures
from .additive import GravityAdditiveFlowBlock
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
    parent_component: str | None = None

    @property
    def size(self) -> int:
        return self.parameter_slice.stop - self.parameter_slice.start

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
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
        if self.parent_component is not None:
            payload["parent_component"] = self.parent_component
        return payload

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
            parent_component=(
                None
                if payload.get("parent_component") is None
                else str(payload["parent_component"])
            ),
        )


def _component_names(component: GravityComponentSpecification) -> tuple[str, ...]:
    if component.scope in (GravityEffectScope.NONE, GravityEffectScope.FIXED):
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
    result.extend(
        f"{prefix}[{index}]"
        for index in range(component.parameter_count - len(result))
    )
    return tuple(result)


def _deviation_names(component: GravityComponentSpecification) -> tuple[str, ...]:
    deviation = component.deviation
    if deviation is None:
        return ()
    prefix = (
        f"{component.name}_time_deviation"
        if component.name == "production"
        and deviation.scope is GravityEffectScope.TIME_PERIOD
        else f"{component.name}_deviation"
    )
    return tuple(
        f"{prefix}[{index}]" for index in range(deviation.parameter_count)
    )


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
                if component.deviation is None:
                    continue
            stop = start + len(names)
            if names:
                result.append(
                    GravityParameterBlock(
                        component=component.name,
                        scope=component.scope,
                        parameterization=component.parameterization,
                        constraint=component.constraint,
                        mapping=component.grouping
                        or _DEFAULT_MAPPINGS.get(component.scope),
                        group_count=0 if component.deviation is not None else component.group_count,
                        reference_category=(
                            None if component.deviation is not None else component.reference_category
                        ),
                        parameter_slice=slice(start, stop),
                        names=names,
                        regularization_type=(
                            GravityRegularizationType.NONE
                            if component.deviation is not None
                            else component.regularization.kind
                        ),
                        regularization_strength=(
                            0.0 if component.deviation is not None else component.regularization.strength
                        ),
                    )
                )
                start = stop
            deviation_names = _deviation_names(component)
            if deviation_names:
                deviation = component.deviation
                assert deviation is not None
                stop = start + len(deviation_names)
                result.append(
                    GravityParameterBlock(
                        component=(
                            "production_time_deviation"
                            if component.name == "production"
                            and deviation.scope is GravityEffectScope.TIME_PERIOD
                            else f"{component.name}_deviation"
                        ),
                        scope=deviation.scope,
                        parameterization=GravityParameterization.ADDITIVE,
                        constraint=deviation.constraint,
                        mapping=deviation.grouping
                        or _DEFAULT_MAPPINGS.get(deviation.scope),
                        group_count=deviation.group_count,
                        reference_category=deviation.reference_category,
                        parameter_slice=slice(start, stop),
                        names=deviation_names,
                        regularization_type=deviation.regularization.kind,
                        regularization_strength=deviation.regularization.strength,
                        parent_component=component.name,
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

    def deviation_block(self, component: str) -> GravityParameterBlock | None:
        """Return the separate deviation block attached to ``component``."""
        return next(
            (item for item in self.blocks if item.parent_component == component), None
        )

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
            attached = self.deviation_block(component)
            if attached is not None:
                block = attached
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
            base = self.scalar_or_base(raw, "production")
            if component.deviation is None:
                return jnp.full(
                    (features.num_origin_time_groups,),
                    base,
                    dtype=raw.dtype,
                )
            deviation_block = self.deviation_block("production")
            assert deviation_block is not None and deviation_block.mapping is not None
            deviations = self.constrained_deviations(
                raw, deviation_block.component
            )
            indices = jnp.asarray(
                features.mapping(deviation_block.mapping), dtype=jnp.int32
            )
            cell_deviation = deviations[indices]
            groups = jnp.asarray(features.origin_time_group_index, dtype=jnp.int32)
            totals = jax.ops.segment_sum(
                cell_deviation,
                groups,
                num_segments=features.num_origin_time_groups,
            )
            counts = jax.ops.segment_sum(
                jnp.ones(features.num_cells, dtype=raw.dtype),
                groups,
                num_segments=features.num_origin_time_groups,
            )
            return base + totals / counts
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


@dataclass(frozen=True, slots=True)
class GravityJointParameterLayout:
    """Flat layout combining gravity and additive-flow parameters.

    The legacy :class:`GravityParameterLayout` remains the exact layout when
    no additive blocks are supplied.  This wrapper delegates every gravity
    transformation to that layout and appends deterministic slices for each
    named flow block.
    """

    gravity_layout: GravityParameterLayout
    additive_flow_blocks: tuple[GravityAdditiveFlowBlock, ...] = ()

    def __post_init__(self) -> None:
        names: list[str] = list(self.gravity_layout.names)
        block_names: set[str] = set()
        for block in self.additive_flow_blocks:
            if block.name in block_names:
                raise ValueError(f"duplicate additive flow block {block.name!r}.")
            block_names.add(block.name)
            overlap = set(block.parameter_names) & set(names)
            if overlap:
                raise ValueError(
                    "additive-flow parameter names overlap gravity parameters: "
                    + ", ".join(sorted(overlap))
                )
            names.extend(block.parameter_names)
        desired = tuple(block.to_dict() for block in self.additive_flow_blocks)
        existing = tuple(self.gravity_layout.specification.additive_flow_blocks)
        if existing and not desired:
            raise ValueError(
                "gravity specification declares additive flow blocks, but none were configured."
            )
        if existing and canonical_json(existing) != canonical_json(desired):
            raise ValueError(
                "gravity specification additive_flow_blocks do not match the configured blocks."
            )
        if desired and not existing:
            specification = replace(
                self.gravity_layout.specification,
                additive_flow_blocks=desired,
            )
            object.__setattr__(
                self,
                "gravity_layout",
                GravityParameterLayout(
                    specification,
                    positivity_floor=self.gravity_layout.positivity_floor,
                ),
            )

    @property
    def specification(self) -> GravityModelSpecification:
        return self.gravity_layout.specification

    @property
    def gravity_parameter_slice(self) -> slice:
        return slice(0, self.gravity_layout.size)

    @property
    def block_slices(self) -> dict[str, slice]:
        start = self.gravity_layout.size
        result: dict[str, slice] = {}
        for block in self.additive_flow_blocks:
            result[block.name] = slice(start, start + block.num_parameters)
            start += block.num_parameters
        return result

    @property
    def size(self) -> int:
        return self.gravity_layout.size + sum(
            block.num_parameters for block in self.additive_flow_blocks
        )

    @property
    def names(self) -> tuple[str, ...]:
        return self.gravity_layout.names + tuple(
            name for block in self.additive_flow_blocks for name in block.parameter_names
        )

    @property
    def blocks(self) -> tuple[GravityParameterBlock, ...]:
        result = list(self.gravity_layout.blocks)
        for block in self.additive_flow_blocks:
            parameter_slice = self.block_slices[block.name]
            result.append(
                GravityParameterBlock(
                    component=block.name,
                    scope=GravityEffectScope.GLOBAL,
                    parameterization=GravityParameterization.ADDITIVE,
                    constraint=GravityConstraint.NONE,
                    mapping=None,
                    group_count=0,
                    reference_category=None,
                    parameter_slice=parameter_slice,
                    names=block.parameter_names,
                    regularization_type=(
                        GravityRegularizationType.RIDGE
                        if block.regularization_strength > 0
                        else GravityRegularizationType.NONE
                    ),
                    regularization_strength=block.regularization_strength,
                )
            )
        return tuple(result)

    @property
    def slices(self) -> dict[str, slice]:
        result = dict(self.gravity_layout.slices)
        for block in self.additive_flow_blocks:
            result[block.name] = self.block_slices[block.name]
            for index, name in enumerate(block.parameter_names):
                position = self.block_slices[block.name].start + index
                result[name] = slice(position, position + 1)
        return result

    def block(self, component: str) -> GravityParameterBlock | None:
        return next((item for item in self.blocks if item.component == component), None)

    def deviation_block(self, component: str) -> GravityParameterBlock | None:
        return self.gravity_layout.deviation_block(component)

    def _raw(self, raw_parameters: object) -> jax.Array:
        raw = jnp.asarray(raw_parameters)
        if raw.ndim != 1 or raw.shape[0] != self.size:
            raise ValueError(f"raw_parameters must have shape ({self.size},).")
        return raw

    def gravity_raw(self, raw_parameters: object) -> jax.Array:
        return self._raw(raw_parameters)[self.gravity_parameter_slice]

    def block_raw(self, raw_parameters: object, name: str) -> jax.Array:
        try:
            parameter_slice = self.block_slices[name]
        except KeyError as error:
            raise ValueError(f"unknown additive flow block {name!r}.") from error
        return self._raw(raw_parameters)[parameter_slice]

    def additive_flow(self, raw_parameters: object, name: str) -> jax.Array:
        block = next(
            (item for item in self.additive_flow_blocks if item.name == name), None
        )
        if block is None:
            raise ValueError(f"unknown additive flow block {name!r}.")
        return block.flow_from_raw(self.block_raw(raw_parameters, name))

    def transform(self, raw_parameters: object) -> MinimalGravityParameters:
        return self.gravity_layout.transform(self.gravity_raw(raw_parameters))

    def constrained_deviations(self, raw_parameters: object, component: str) -> jax.Array:
        return self.gravity_layout.constrained_deviations(self.gravity_raw(raw_parameters), component)

    def scalar_or_base(self, raw_parameters: object, component: str) -> jax.Array:
        return self.gravity_layout.scalar_or_base(self.gravity_raw(raw_parameters), component)

    def cell_effect(
        self, raw_parameters: object, component: str, features: GravityFeatures
    ) -> jax.Array:
        return self.gravity_layout.cell_effect(
            self.gravity_raw(raw_parameters), component, features
        )

    def production_group_log_multiplier(
        self, raw_parameters: object, features: GravityFeatures
    ) -> jax.Array:
        return self.gravity_layout.production_group_log_multiplier(
            self.gravity_raw(raw_parameters), features
        )

    def production_log_scale(self, raw_parameters: object) -> jax.Array:
        return self.gravity_layout.production_log_scale(self.gravity_raw(raw_parameters))

    def centered_effect(self, raw_parameters: object, block: str) -> jax.Array:
        return self.gravity_layout.centered_effect(self.gravity_raw(raw_parameters), block)

    def regularization(self, raw_parameters: object) -> jax.Array:
        raw = self._raw(raw_parameters)
        result = self.gravity_layout.regularization(raw[self.gravity_parameter_slice])
        for block in self.additive_flow_blocks:
            result = result + block.regularization(self.block_raw(raw, block.name))
        return result

    def physical_vector(self, raw_parameters: object) -> jax.Array:
        raw = self._raw(raw_parameters)
        values = [self.gravity_layout.physical_vector(raw[self.gravity_parameter_slice])]
        values.extend(
            jnp.asarray(block.latent_flow_model.physical_parameters(self.block_raw(raw, block.name)), dtype=raw.dtype)
            for block in self.additive_flow_blocks
        )
        return jnp.concatenate(values)

    def raw_from_physical(self, physical_parameters: object) -> np.ndarray:
        values = np.asarray(physical_parameters, dtype=np.float64)
        if values.shape != (self.size,) or not np.all(np.isfinite(values)):
            raise ValueError(f"physical_parameters must have shape ({self.size},).")
        result = [
            self.gravity_layout.raw_from_physical(
                values[self.gravity_parameter_slice]
            )
        ]
        for block in self.additive_flow_blocks:
            result.append(
                block.latent_flow_model.raw_from_physical(
                    values[self.block_slices[block.name]]
                )
            )
        return np.concatenate(result)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": 1,
            "gravity_layout": self.gravity_layout.to_dict(),
            "additive_flow_blocks": [block.to_dict() for block in self.additive_flow_blocks],
            "names": list(self.names),
        }
        payload["fingerprint"] = fingerprint(payload)
        return payload

    @property
    def fingerprint(self) -> str:
        payload = self.to_dict()
        return str(payload["fingerprint"])

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        operators: Mapping[str, object],
    ) -> "GravityJointParameterLayout":
        """Restore a joint layout using caller-supplied additive operators."""
        gravity_payload = payload.get("gravity_layout")
        block_payloads = payload.get("additive_flow_blocks", ())
        if not isinstance(gravity_payload, Mapping):
            raise ValueError("gravity_layout is missing or invalid.")
        if not isinstance(block_payloads, (list, tuple)):
            raise ValueError("additive_flow_blocks must be a sequence.")
        gravity_layout = GravityParameterLayout.from_dict(gravity_payload)
        blocks: list[GravityAdditiveFlowBlock] = []
        for raw_block in block_payloads:
            if not isinstance(raw_block, Mapping):
                raise ValueError("additive flow block payload must be a mapping.")
            name = str(raw_block.get("name", ""))
            if name not in operators:
                raise ValueError(f"no operator supplied for additive flow block {name!r}.")
            blocks.append(
                GravityAdditiveFlowBlock.from_dict(
                    raw_block,
                    operator=cast(object, operators[name]),
                )
            )
        result = cls(gravity_layout, tuple(blocks))
        names = payload.get("names")
        if names is not None and tuple(str(item) for item in cast(list[object], names)) != result.names:
            raise ValueError("joint parameter names do not match the restored layout.")
        stored = payload.get("fingerprint")
        if stored is not None and str(stored) != result.fingerprint:
            raise ValueError("joint parameter-layout fingerprint mismatch.")
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
        if component.scope in _DEFAULT_MAPPINGS or component.scope in (
            GravityEffectScope.CUSTOM_GROUP,
            GravityEffectScope.SMOOTH_BASIS,
        ):
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
        if component.deviation is not None:
            deviation = component.deviation
            deviation_mapping = deviation.grouping or _DEFAULT_MAPPINGS.get(
                deviation.scope
            )
            if deviation_mapping is None:
                raise ValueError(
                    f"component {component.name!r} deviation requires an explicit "
                    "feature mapping."
                )
            features.validate_mapping(
                deviation_mapping,
                group_count=deviation.group_count,
                constant_within_origin_time=(
                    component.name == "production"
                    or deviation_mapping
                    in {
                        "origin_index",
                        "time_period_index",
                        "origin_time_group_index",
                        "origin_zone_index",
                    }
                ),
            )
