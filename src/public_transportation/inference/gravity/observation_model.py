"""Observation-process scales for row-aligned gravity measurements."""

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
                self.boarding_scale
                if value == "boarding"
                else self.alighting_scale
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
                None
                if self.measurement_types is None
                else list(self.measurement_types)
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
            None
            if raw_types is None
            else tuple(str(value) for value in raw_types)  # type: ignore[union-attr]
        )
        return cls(
            measurement_types=measurement_types,
            boarding_scale=float(payload.get("boarding_scale", 1.0)),
            alighting_scale=float(payload.get("alighting_scale", 1.0)),
        )

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())
