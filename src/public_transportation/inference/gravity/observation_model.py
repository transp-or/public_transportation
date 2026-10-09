"""Observation-process models for row-aligned gravity measurements.

The journey latent-class model in this module uses the *aggregate independent
journey* interpretation.  Each underlying journey is assigned a class
independently, so a row containing an aggregate count has expected mean equal
to the assigned flow multiplied by the probability-weighted class effect.  The
ordinary Poisson or negative-binomial count likelihood is then evaluated on
that aggregate mean.  A shared-class/joint-journey likelihood is deliberately
not inferred from row metadata: callers must use a future joint model when
that is the intended statistical unit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.inference.block_coordinate._canonical import fingerprint


@dataclass(frozen=True, slots=True)
class GravityObservationModel:
    """Fixed boarding/alighting scales aligned with measurement rows.

    The existing scalar ``rho`` remains an outer global scale in
    :class:`GravityObjectiveProblem`.  This object supplies optional row-type
    multipliers; estimated scales and dispersion blocks are intentionally left
    for a later phase.
    """

    measurement_types: tuple[str, ...] | None = None
    boarding_scale: float = 1.0
    alighting_scale: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("boarding_scale", self.boarding_scale),
            ("alighting_scale", self.alighting_scale),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.measurement_types is not None:
            types = tuple(str(value) for value in self.measurement_types)
            invalid = set(types) - {"boarding", "alighting"}
            if invalid:
                raise ValueError(
                    "measurement_types may contain only 'boarding' and 'alighting'."
                )
            object.__setattr__(self, "measurement_types", types)
        elif self.boarding_scale != 1.0 or self.alighting_scale != 1.0:
            raise ValueError(
                "measurement_types are required when boarding or alighting scales differ from one."
            )

    def scale_vector(self, num_rows: int, *, dtype: object = np.float32) -> jax.Array:
        if self.measurement_types is None:
            return jnp.ones((num_rows,), dtype=dtype)
        if len(self.measurement_types) != num_rows:
            raise ValueError("measurement_types must match the measurement dimension.")
        return jnp.asarray(
            [
                self.boarding_scale if value == "boarding" else self.alighting_scale
                for value in self.measurement_types
            ],
            dtype=dtype,
        )

    def supported_rows(self, num_rows: int) -> tuple[np.ndarray, np.ndarray]:
        if self.measurement_types is None:
            return np.zeros(num_rows, dtype=bool), np.zeros(num_rows, dtype=bool)
        if len(self.measurement_types) != num_rows:
            raise ValueError("measurement_types must match the measurement dimension.")
        values = np.asarray(self.measurement_types, dtype=object)
        return values == "boarding", values == "alighting"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "measurement_types": (
                None if self.measurement_types is None else list(self.measurement_types)
            ),
            "boarding_scale": float(self.boarding_scale),
            "alighting_scale": float(self.alighting_scale),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GravityObservationModel":
        if not isinstance(payload, Mapping):
            raise TypeError("observation-model payload must be a mapping.")
        raw_types = payload.get("measurement_types")
        measurement_types = (
            None if raw_types is None else tuple(str(value) for value in raw_types)  # type: ignore[union-attr]
        )
        return cls(
            measurement_types=measurement_types,
            boarding_scale=float(payload.get("boarding_scale", 1.0)),
            alighting_scale=float(payload.get("alighting_scale", 1.0)),
        )

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class GravityJourneyLatentClassModel:
    """Bounded journey-level latent bias in the measurement equation.

    ``class_probabilities`` and ``class_effects`` may be supplied as fixed
    values.  If either is omitted, the corresponding ``num_classes - 1``
    free parameters are appended by :class:`GravityMeasurementParameterLayout`.
    Class ``reference_class`` has a fixed logit of zero and a fixed effect of
    one, which removes both label and global-scale redundancy.  Free effects
    are transformed as ``exp(max_log_effect * tanh(raw))`` and are therefore
    finite and strictly positive for every optimizer iterate.
    """

    num_classes: int = 2
    class_probabilities: tuple[float, ...] | None = None
    class_effects: tuple[float, ...] | None = None
    reference_class: int = 0
    max_logit: float = 30.0
    max_log_effect: float = 2.0
    regularization_strength: float = 0.0
    aggregation_unit: str = "independent_journey_aggregate"

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two.")
        if not 0 <= self.reference_class < self.num_classes:
            raise ValueError("reference_class is outside the declared classes.")
        if self.aggregation_unit != "independent_journey_aggregate":
            raise ValueError(
                "only 'independent_journey_aggregate' is currently supported; "
                "a shared-class joint likelihood must be implemented explicitly."
            )
        if not np.isfinite(self.max_logit) or not 0 < self.max_logit <= 30.0:
            raise ValueError("max_logit must be finite and lie in (0, 30].")
        if not np.isfinite(self.max_log_effect) or not 0 < self.max_log_effect <= 30.0:
            raise ValueError("max_log_effect must be finite and lie in (0, 30].")
        if (
            not np.isfinite(self.regularization_strength)
            or self.regularization_strength < 0
        ):
            raise ValueError("regularization_strength must be finite and non-negative.")
        for name, values in (
            ("class_probabilities", self.class_probabilities),
            ("class_effects", self.class_effects),
        ):
            if values is None:
                continue
            normalized = tuple(float(value) for value in values)
            if len(normalized) != self.num_classes:
                raise ValueError(f"{name} must contain num_classes values.")
            if not np.all(np.isfinite(normalized)):
                raise ValueError(f"{name} must contain finite values.")
            if name == "class_probabilities":
                if any(value <= 0 for value in normalized):
                    raise ValueError("class_probabilities must be strictly positive.")
                if not np.isclose(sum(normalized), 1.0, rtol=1e-10, atol=1e-12):
                    raise ValueError("class_probabilities must sum to one.")
            else:
                if any(value <= 0 for value in normalized):
                    raise ValueError("class_effects must be strictly positive.")
                if not np.isclose(normalized[self.reference_class], 1.0, atol=1e-10):
                    raise ValueError(
                        "the reference class effect must equal one to identify scale."
                    )
            object.__setattr__(self, name, normalized)

    @property
    def free_classes(self) -> tuple[int, ...]:
        return tuple(
            index for index in range(self.num_classes) if index != self.reference_class
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.class_probabilities is None:
            names.extend(f"journey_bias.logit[{index}]" for index in self.free_classes)
        if self.class_effects is None:
            names.extend(
                f"journey_bias.log_effect[{index}]" for index in self.free_classes
            )
        return tuple(names)

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_names)

    @property
    def initial_raw_parameters(self) -> np.ndarray:
        return np.zeros(self.parameter_count, dtype=np.float64)

    def _split_raw(self, raw_parameters: object) -> tuple[jax.Array, jax.Array]:
        raw = jnp.asarray(raw_parameters)
        if raw.ndim != 1 or raw.shape[0] != self.parameter_count:
            raise ValueError(
                f"measurement-model raw parameters must have shape ({self.parameter_count},)."
            )
        position = 0
        if self.class_probabilities is None:
            logits_free = raw[position : position + len(self.free_classes)]
            position += len(self.free_classes)
        else:
            logits_free = jnp.zeros((0,), dtype=raw.dtype)
        if self.class_effects is None:
            effects_free = raw[position : position + len(self.free_classes)]
        else:
            effects_free = jnp.zeros((0,), dtype=raw.dtype)
        return logits_free, effects_free

    def class_parameters(
        self, raw_parameters: object, *, dtype: object | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Return stable class probabilities and positive class effects."""

        raw = jnp.asarray(raw_parameters)
        if dtype is not None:
            raw = raw.astype(dtype)
        logits_free, effects_free = self._split_raw(raw)
        if self.class_probabilities is None:
            logits = jnp.zeros((self.num_classes,), dtype=raw.dtype)
            logits = logits.at[jnp.asarray(self.free_classes, dtype=jnp.int32)].set(
                jnp.asarray(self.max_logit, dtype=raw.dtype) * jnp.tanh(logits_free)
            )
            log_probabilities = jax.nn.log_softmax(logits)
            probabilities = jnp.exp(log_probabilities)
        else:
            probabilities = jnp.asarray(self.class_probabilities, dtype=raw.dtype)
        if self.class_effects is None:
            log_effects = jnp.zeros((self.num_classes,), dtype=raw.dtype)
            log_effects = log_effects.at[
                jnp.asarray(self.free_classes, dtype=jnp.int32)
            ].set(
                jnp.asarray(self.max_log_effect, dtype=raw.dtype)
                * jnp.tanh(effects_free)
            )
            effects = jnp.exp(log_effects)
        else:
            effects = jnp.asarray(self.class_effects, dtype=raw.dtype)
        return probabilities, effects

    def aggregate_scale(
        self, raw_parameters: object, *, dtype: object | None = None
    ) -> jax.Array:
        probabilities, effects = self.class_parameters(raw_parameters, dtype=dtype)
        scale = jnp.sum(probabilities * effects)
        return jnp.where(jnp.isfinite(scale) & (scale > 0), scale, jnp.nan)

    def regularization(
        self, raw_parameters: object, *, dtype: object | None = None
    ) -> jax.Array:
        raw = jnp.asarray(raw_parameters)
        if dtype is not None:
            raw = raw.astype(dtype)
        return (
            0.5
            * jnp.asarray(self.regularization_strength, dtype=raw.dtype)
            * jnp.sum(raw * raw)
        )

    def physical_free_parameters(
        self, raw_parameters: object, *, dtype: object | None = None
    ) -> jax.Array:
        probabilities, effects = self.class_parameters(raw_parameters, dtype=dtype)
        free = jnp.asarray(self.free_classes, dtype=jnp.int32)
        return jnp.concatenate((probabilities[free], effects[free]))

    def raw_from_physical(self, physical_parameters: object) -> np.ndarray:
        values = np.asarray(physical_parameters, dtype=np.float64)
        if values.shape != (self.parameter_count,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"measurement-model physical parameters must have shape ({self.parameter_count},)."
            )
        position = 0
        result: list[float] = []
        if self.class_probabilities is None:
            probabilities = np.zeros(self.num_classes, dtype=np.float64)
            probabilities[self.reference_class] = 1.0 - np.sum(
                values[: len(self.free_classes)]
            )
            probabilities[list(self.free_classes)] = values[: len(self.free_classes)]
            if np.any(probabilities <= 0):
                raise ValueError(
                    "physical class probabilities must be positive and sum to one."
                )
            logits = np.asarray(
                [
                    np.log(probabilities[index] / probabilities[self.reference_class])
                    / self.max_logit
                    for index in self.free_classes
                ],
                dtype=np.float64,
            )
            if np.any(np.abs(logits) >= 1.0):
                raise ValueError("physical class probabilities exceed max_logit.")
            result.extend(float(np.arctanh(value)) for value in logits)
            position += len(self.free_classes)
        if self.class_effects is None:
            effects = values[position:]
            if np.any(effects <= 0):
                raise ValueError("physical free class effects must be positive.")
            bounded = np.log(effects) / self.max_log_effect
            if np.any(np.abs(bounded) >= 1.0):
                raise ValueError("physical class effects exceed max_log_effect.")
            result.extend(float(np.arctanh(value)) for value in bounded)
        return np.asarray(result, dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "type": "journey_latent_class",
            "aggregation_unit": self.aggregation_unit,
            "num_classes": self.num_classes,
            "reference_class": self.reference_class,
            "max_logit": float(self.max_logit),
            "class_probabilities": (
                None
                if self.class_probabilities is None
                else list(self.class_probabilities)
            ),
            "class_effects": (
                None if self.class_effects is None else list(self.class_effects)
            ),
            "max_log_effect": float(self.max_log_effect),
            "regularization_strength": float(self.regularization_strength),
            "parameter_names": list(self.parameter_names),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "GravityJourneyLatentClassModel":
        if payload.get("type", "journey_latent_class") != "journey_latent_class":
            raise ValueError("unsupported gravity measurement-model type.")
        return cls(
            num_classes=int(payload["num_classes"]),
            reference_class=int(payload.get("reference_class", 0)),
            max_logit=float(payload.get("max_logit", 30.0)),
            class_probabilities=(
                None
                if payload.get("class_probabilities") is None
                else tuple(float(value) for value in payload["class_probabilities"])  # type: ignore[union-attr]
            ),
            class_effects=(
                None
                if payload.get("class_effects") is None
                else tuple(float(value) for value in payload["class_effects"])  # type: ignore[union-attr]
            ),
            max_log_effect=float(payload.get("max_log_effect", 2.0)),
            regularization_strength=float(payload.get("regularization_strength", 0.0)),
            aggregation_unit=str(
                payload.get("aggregation_unit", "independent_journey_aggregate")
            ),
        )

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class GravityMeasurementParameterLayout:
    """Append latent measurement parameters to an existing gravity layout.

    Prepared assignment artifacts are owned by ``base_layout`` and are not
    modified.  Only the fit-time optimizer vector and its model fingerprint
    change.
    """

    base_layout: object
    measurement_model: GravityJourneyLatentClassModel

    def __post_init__(self) -> None:
        if not isinstance(self.measurement_model, GravityJourneyLatentClassModel):
            raise TypeError(
                "measurement_model must be a GravityJourneyLatentClassModel."
            )
        names = set(getattr(self.base_layout, "names"))
        overlap = names.intersection(self.measurement_model.parameter_names)
        if overlap:
            raise ValueError(
                f"measurement parameter names overlap gravity names: {sorted(overlap)}"
            )

    @property
    def specification(self):
        return self.base_layout.specification

    @property
    def positivity_floor(self):
        return self.base_layout.positivity_floor

    @property
    def gravity_parameter_slice(self) -> slice:
        return slice(0, self.base_layout.size)

    @property
    def measurement_parameter_slice(self) -> slice:
        return slice(self.base_layout.size, self.size)

    @property
    def size(self) -> int:
        return int(self.base_layout.size) + self.measurement_model.parameter_count

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.base_layout.names) + self.measurement_model.parameter_names

    @property
    def blocks(self):
        from .parameters import GravityParameterBlock
        from .specification import (
            GravityConstraint,
            GravityEffectScope,
            GravityParameterization,
            GravityRegularizationType,
        )

        result = list(self.base_layout.blocks)
        if self.measurement_model.parameter_count:
            result.append(
                GravityParameterBlock(
                    component="journey_latent_bias",
                    scope=GravityEffectScope.GLOBAL,
                    parameterization=GravityParameterization.ADDITIVE,
                    constraint=GravityConstraint.NONE,
                    mapping=None,
                    group_count=0,
                    reference_category=None,
                    parameter_slice=self.measurement_parameter_slice,
                    names=self.measurement_model.parameter_names,
                    regularization_type=(
                        GravityRegularizationType.RIDGE
                        if self.measurement_model.regularization_strength > 0
                        else GravityRegularizationType.NONE
                    ),
                    regularization_strength=self.measurement_model.regularization_strength,
                )
            )
        return tuple(result)

    @property
    def slices(self) -> dict[str, slice]:
        result = dict(getattr(self.base_layout, "slices", {}))
        for index, name in enumerate(self.measurement_model.parameter_names):
            position = self.measurement_parameter_slice.start + index
            result[name] = slice(position, position + 1)
        result["journey_latent_bias"] = self.measurement_parameter_slice
        return result

    @property
    def additive_flow_blocks(self):
        return getattr(self.base_layout, "additive_flow_blocks", ())

    @property
    def fingerprint(self) -> str:
        return str(self.to_dict()["fingerprint"])

    def _raw(self, raw_parameters: object) -> jax.Array:
        raw = jnp.asarray(raw_parameters)
        if raw.ndim != 1 or raw.shape[0] != self.size:
            raise ValueError(f"raw_parameters must have shape ({self.size},).")
        return raw

    def gravity_raw(self, raw_parameters: object) -> jax.Array:
        return self._raw(raw_parameters)[self.gravity_parameter_slice]

    def measurement_raw(self, raw_parameters: object) -> jax.Array:
        return self._raw(raw_parameters)[self.measurement_parameter_slice]

    def block(self, component: str):
        if component == "journey_latent_bias":
            return next(
                (block for block in self.blocks if block.component == component), None
            )
        return self.base_layout.block(component)

    def block_raw(self, raw_parameters: object, name: str) -> jax.Array:
        if name == "journey_latent_bias":
            return self.measurement_raw(raw_parameters)
        return self.base_layout.block_raw(self.gravity_raw(raw_parameters), name)

    def deviation_block(self, component: str):
        return self.base_layout.deviation_block(component)

    def transform(self, raw_parameters: object):
        return self.base_layout.transform(self.gravity_raw(raw_parameters))

    def regularization(self, raw_parameters: object) -> jax.Array:
        raw = self._raw(raw_parameters)
        return self.base_layout.regularization(
            self.gravity_raw(raw)
        ) + self.measurement_model.regularization(
            self.measurement_raw(raw), dtype=raw.dtype
        )

    def physical_vector(self, raw_parameters: object) -> jax.Array:
        raw = self._raw(raw_parameters)
        base = self.base_layout.physical_vector(self.gravity_raw(raw))
        model = self.measurement_model.physical_free_parameters(
            self.measurement_raw(raw), dtype=raw.dtype
        )
        return jnp.concatenate((base, model))

    def raw_from_physical(self, physical_parameters: object) -> np.ndarray:
        values = np.asarray(physical_parameters, dtype=np.float64)
        if values.shape != (self.size,):
            raise ValueError(f"physical_parameters must have shape ({self.size},).")
        base = self.base_layout.raw_from_physical(values[: self.base_layout.size])
        model = self.measurement_model.raw_from_physical(
            values[self.base_layout.size :]
        )
        return np.concatenate((base, model))

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 1,
            "type": "measurement_parameter_layout",
            "base_layout": self.base_layout.to_dict(),
            "measurement_model": self.measurement_model.to_dict(),
            "names": list(self.names),
        }
        payload["fingerprint"] = fingerprint(payload)
        return payload

    # Demand evaluation delegates its gravity portion to the unchanged base
    # layout, keeping prepared-assignment semantics entirely separate.
    def cell_effect(self, raw_parameters: object, component: str, features: object):
        return self.base_layout.cell_effect(
            self.gravity_raw(raw_parameters), component, features
        )

    def term_effect(self, raw_parameters: object, term_name: str, features: object):
        return self.base_layout.term_effect(
            self.gravity_raw(raw_parameters), term_name, features
        )

    def destination_utility_effect(self, raw_parameters: object, features: object):
        return self.base_layout.destination_utility_effect(
            self.gravity_raw(raw_parameters), features
        )

    def production_group_log_multiplier(self, raw_parameters: object, features: object):
        return self.base_layout.production_group_log_multiplier(
            self.gravity_raw(raw_parameters), features
        )

    def production_log_scale(self, raw_parameters: object):
        return self.base_layout.production_log_scale(self.gravity_raw(raw_parameters))

    def scalar_or_base(self, raw_parameters: object, component: str):
        return self.base_layout.scalar_or_base(
            self.gravity_raw(raw_parameters), component
        )

    def centered_effect(self, raw_parameters: object, block: str):
        return self.base_layout.centered_effect(self.gravity_raw(raw_parameters), block)
