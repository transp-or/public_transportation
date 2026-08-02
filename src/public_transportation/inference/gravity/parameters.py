"""Explicit flat-parameter layout for the minimal gravity model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.inference.block_coordinate._canonical import fingerprint

from .specification import GravityEffectScope, GravityModelSpecification


class MinimalGravityParameters(NamedTuple):
    beta_time: jax.Array
    beta_transfer: jax.Array
    dispersion: jax.Array


@dataclass(frozen=True, slots=True)
class GravityParameterLayout:
    specification: GravityModelSpecification
    positivity_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        if not np.isfinite(self.positivity_floor) or self.positivity_floor <= 0:
            raise ValueError("positivity_floor must be finite and positive.")

    @property
    def size(self) -> int:
        return self.specification.parameter_count

    @property
    def names(self) -> tuple[str, ...]:
        return self.specification.parameter_names

    @property
    def slices(self) -> dict[str, slice]:
        start = 3
        result = {name: slice(index, index + 1) for index, name in enumerate(self.names[:3])}
        for name, count in (
            ("destination_zone", self.specification.destination_zone_count),
            ("time_period", self.specification.time_period_count),
            ("origin_zone", self.specification.origin_zone_count),
        ):
            width = max(count - 1, 0)
            result[name] = slice(start, start + width)
            start += width
        return result

    @property
    def fingerprint(self) -> str:
        return fingerprint(
            {
                "schema_version": 1,
                "specification": self.specification.to_dict(),
                "names": self.names,
                "positivity_floor": self.positivity_floor,
            }
        )

    def transform(self, raw_parameters: object) -> MinimalGravityParameters:
        raw = jnp.asarray(raw_parameters)
        if raw.ndim != 1 or raw.shape[0] != self.size:
            raise ValueError(f"raw_parameters must have shape ({self.size},).")
        positive = jax.nn.softplus(raw[:3]) + jnp.asarray(
            self.positivity_floor, dtype=raw.dtype
        )
        return MinimalGravityParameters(*positive)

    def centered_effect(self, raw_parameters: object, block: str) -> jax.Array:
        """Return a sum-zero block, deriving its final component exactly."""
        raw = jnp.asarray(raw_parameters)
        if raw.ndim != 1 or raw.shape[0] != self.size:
            raise ValueError(f"raw_parameters must have shape ({self.size},).")
        if block not in ("destination_zone", "time_period", "origin_zone"):
            raise ValueError(f"unknown gravity relaxation block {block!r}.")
        free = raw[self.slices[block]]
        if free.size == 0:
            return free
        return jnp.concatenate((free, -jnp.sum(free, keepdims=True)))

    def regularization(self, raw_parameters: object) -> jax.Array:
        raw = jnp.asarray(raw_parameters)
        if raw.ndim != 1 or raw.shape[0] != self.size:
            raise ValueError(f"raw_parameters must have shape ({self.size},).")
        result = jnp.asarray(0.0, dtype=raw.dtype)
        for block, ridge in (
            ("destination_zone", self.specification.destination_zone_ridge),
            ("time_period", self.specification.time_period_ridge),
            ("origin_zone", self.specification.origin_zone_ridge),
        ):
            effect = self.centered_effect(raw, block)
            result = result + 0.5 * jnp.asarray(ridge, dtype=raw.dtype) * jnp.sum(effect * effect)
        return result

    def physical_vector(self, raw_parameters: object) -> jax.Array:
        base = jnp.asarray(self.transform(raw_parameters))
        return jnp.concatenate((base, *(self.centered_effect(raw_parameters, block)[:-1] for block in ("destination_zone", "time_period", "origin_zone"))))

    def raw_from_physical(self, physical_parameters: object) -> np.ndarray:
        values = np.asarray(physical_parameters)
        if values.shape != (self.size,) or values.dtype.kind not in "iuf":
            raise ValueError(f"physical_parameters must have shape ({self.size},).")
        shifted = values[:3].astype(np.float64) - self.positivity_floor
        if not np.all(np.isfinite(values)) or np.any(shifted <= 0):
            raise ValueError("physical parameters must exceed positivity_floor.")
        result = values.astype(np.float64, copy=True)
        result[:3] = shifted + np.log(-np.expm1(-shifted))
        return result


def warm_start_gravity_parameters(
    parent_layout: GravityParameterLayout,
    child_layout: GravityParameterLayout,
    parent_raw_parameters: object,
) -> np.ndarray:
    """Embed a parent iterate in a nested child; new deviations start at zero."""
    parent = np.asarray(parent_raw_parameters)
    if parent.shape != (parent_layout.size,):
        raise ValueError(f"parent_raw_parameters must have shape ({parent_layout.size},).")
    child = np.zeros(child_layout.size, dtype=parent.dtype)
    parent_positions = {name: index for index, name in enumerate(parent_layout.names)}
    for index, name in enumerate(child_layout.names):
        if name in parent_positions:
            child[index] = parent[parent_positions[name]]
    missing = set(parent_layout.names) - set(child_layout.names)
    if missing:
        raise ValueError("child layout does not contain every parent parameter.")
    return child


def validate_gravity_relaxation_features(features: object, specification: GravityModelSpecification) -> None:
    """Validate user-supplied compact mappings required by active relaxations."""
    requirements = (
        (GravityEffectScope.DESTINATION_ZONE, specification.destination_attractiveness_scope, "destination_zone_index", specification.destination_zone_count, False),
        (GravityEffectScope.TIME_PERIOD, specification.temporal_basis_scope, "time_period_index", specification.time_period_count, True),
        (GravityEffectScope.ORIGIN_ZONE, specification.origin_total_correction_scope, "origin_zone_index", specification.origin_zone_count, True),
    )
    groups = np.asarray(getattr(features, "origin_time_group_index"))
    for active_scope, scope, name, count, group_constant in requirements:
        if scope is not active_scope:
            continue
        values = getattr(features, name)
        if values is None:
            raise ValueError(f"{name} is required by the active gravity relaxation.")
        array = np.asarray(values)
        if not np.array_equal(np.unique(array), np.arange(count)):
            raise ValueError(f"{name} must use every contiguous index from zero to {count - 1}.")
        if group_constant:
            for group in range(int(groups.max()) + 1):
                if np.unique(array[groups == group]).size != 1:
                    raise ValueError(f"{name} must be constant within each origin-time group.")
