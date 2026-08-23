"""Concrete optional-observation channels for aggregate route responses.

Phase 3 provides one attribute-response operator per histogram.  This module
connects those operators to the Phase-1 observation protocol without changing
the count-only path.  A channel includes the fixed-demand offset in its
prediction, while the likelihood remains conditional on the predicted mass
within each stratum.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.inference.block_coordinate._canonical import fingerprint

from .aggregate import (
    GravityAggregateHistogram,
    GravityAggregateObservation,
    GravityAggregateUncertainty,
)
from .attribute_operator import (
    GravityAttributeResponseOperator,
    GravityAttributeSupportError,
    validate_aggregate_support,
)
from .likelihoods import aggregate_histogram_log_likelihood
from .observations import GravityObservationBundle


@dataclass(frozen=True, slots=True)
class GravityAggregateObservationChannel:
    """One aggregate histogram connected to one route-response operator."""

    histogram: GravityAggregateHistogram
    uncertainty: GravityAggregateUncertainty
    operator: GravityAttributeResponseOperator
    aggregate_fingerprint: str
    channel_name: str | None = None

    def __post_init__(self) -> None:
        name = self.channel_name
        if name is None:
            name = f"gps_trip_attributes:{self.histogram.attribute}"
        if not isinstance(name, str) or not name.strip():
            raise ValueError("channel_name must be a non-empty string.")
        object.__setattr__(self, "channel_name", name.strip())
        if (
            not isinstance(self.aggregate_fingerprint, str)
            or not self.aggregate_fingerprint.strip()
        ):
            raise ValueError("aggregate_fingerprint must be a non-empty string.")
        if not isinstance(self.operator, GravityAttributeResponseOperator):
            raise TypeError("operator must be GravityAttributeResponseOperator.")
        if not isinstance(self.uncertainty, GravityAggregateUncertainty):
            raise TypeError("uncertainty must be GravityAggregateUncertainty.")

    @property
    def name(self) -> str:
        assert self.channel_name is not None
        return self.channel_name

    @property
    def kind(self) -> str:
        return "trip_attribute_distribution"

    @property
    def observed(self) -> GravityAggregateHistogram:
        return self.histogram

    @property
    def fingerprint(self) -> str:
        return fingerprint(
            {
                "schema_version": 1,
                "name": self.name,
                "kind": self.kind,
                "aggregate_fingerprint": self.aggregate_fingerprint,
                "histogram": self.histogram.to_dict(),
                "uncertainty": self.uncertainty.to_dict(),
                "operator_fingerprint": self.operator.fingerprint,
            }
        )

    def _support_report(self) -> dict[str, object]:
        try:
            return validate_aggregate_support(self.operator, self.histogram)
        except GravityAttributeSupportError as error:
            return dict(error.report)

    def validate(self, *, num_free_od: int, dtype: np.dtype) -> None:
        del dtype
        if self.operator.num_free_od != num_free_od:
            raise ValueError(
                f"aggregate channel {self.name!r} expects {self.operator.num_free_od} "
                f"free OD cells, got {num_free_od}."
            )
        # This also checks attribute/unit vocabulary and raises on every
        # positive observed category without model support.
        validate_aggregate_support(self.operator, self.histogram)

    def predict(self, demand: jax.Array) -> jax.Array:
        value = jnp.asarray(demand)
        routed = self.operator.jax_matvec(value)
        offset = jnp.asarray(self.operator.fixed_attribute_offset, dtype=value.dtype)
        return routed + offset

    def log_likelihood(
        self, *, prediction: jax.Array, raw_parameters: jax.Array
    ) -> jax.Array:
        del raw_parameters
        return aggregate_histogram_log_likelihood(
            prediction,
            histogram=self.histogram,
            uncertainty=self.uncertainty,
        )

    def report(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "attribute": self.histogram.attribute,
            "unit": self.histogram.unit,
            "aggregate_fingerprint": self.aggregate_fingerprint,
            "channel_fingerprint": self.fingerprint,
            "operator_fingerprint": self.operator.fingerprint,
            "uncertainty": self.uncertainty.to_dict(),
            "histogram": self.histogram.to_dict(),
            "support_audit": self._support_report(),
        }


def build_gravity_aggregate_observation_bundle(
    observation: GravityAggregateObservation,
    operators: Mapping[str, GravityAttributeResponseOperator],
    *,
    num_free_od: int | None = None,
    dtype: np.dtype = np.dtype("float64"),
) -> GravityObservationBundle:
    """Build and validate one channel for every aggregate histogram."""

    if not isinstance(observation, GravityAggregateObservation):
        raise TypeError("observation must be GravityAggregateObservation.")
    channels: list[GravityAggregateObservationChannel] = []
    for histogram in observation.histograms:
        try:
            operator = operators[histogram.attribute]
        except KeyError as error:
            raise KeyError(
                f"no attribute-response operator was supplied for {histogram.attribute!r}."
            ) from error
        channel = GravityAggregateObservationChannel(
            histogram=histogram,
            uncertainty=observation.uncertainty,
            operator=operator,
            aggregate_fingerprint=observation.fingerprint,
        )
        channels.append(channel)
    bundle = GravityObservationBundle(tuple(channels))
    if num_free_od is None:
        num_free_od = channels[0].operator.num_free_od
    bundle.validate(num_free_od=num_free_od, dtype=dtype)
    return bundle


__all__ = [
    "GravityAggregateObservationChannel",
    "build_gravity_aggregate_observation_bundle",
]
