"""Resolved count-observation families for the generic demand estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from public_transportation.measurement.likelihood_jax import (
    negbinom_logpmf_mu_r,
    poisson_logpmf,
)


class ObservationModel(Protocol):
    def log_likelihood(
        self,
        observations: object,
        mean: object,
        *,
        dispersion: object | None = None,
        inflation_logits: object | None = None,
    ) -> jax.Array: ...
    def negative_log_likelihood(
        self,
        observations: object,
        mean: object,
        *,
        dispersion: object | None = None,
        inflation_logits: object | None = None,
    ) -> jax.Array: ...
    def variance(
        self,
        mean: object,
        *,
        dispersion: object | None = None,
        inflation_logits: object | None = None,
    ) -> jax.Array: ...
    def zero_probability(
        self,
        mean: object,
        *,
        dispersion: object | None = None,
        inflation_logits: object | None = None,
    ) -> jax.Array: ...


@dataclass(frozen=True, slots=True)
class CountObservationModel:
    family: str

    def __post_init__(self) -> None:
        if self.family not in {"poisson", "negative_binomial", "zip", "zinb"}:
            raise ValueError("unsupported count-observation family.")

    def _count_logpmf(
        self, observations: object, mean: object, dispersion: object | None
    ) -> jax.Array:
        y, mu = jnp.asarray(observations), jnp.asarray(mean)
        if self.family in {"poisson", "zip"}:
            return poisson_logpmf(y, mu)
        if dispersion is None:
            raise ValueError("negative-binomial families require dispersion.")
        return negbinom_logpmf_mu_r(y, mu, jnp.asarray(dispersion))

    def log_likelihood(
        self,
        observations: object,
        mean: object,
        *,
        dispersion: object | None = None,
        inflation_logits: object | None = None,
    ) -> jax.Array:
        y = jnp.asarray(observations)
        count = self._count_logpmf(y, mean, dispersion)
        if self.family not in {"zip", "zinb"}:
            if inflation_logits is not None:
                raise ValueError(
                    "non-inflated families do not accept inflation logits."
                )
            return count
        if inflation_logits is None:
            raise ValueError("zero-inflated families require inflation logits.")
        logits = jnp.asarray(inflation_logits)
        log_pi = jax.nn.log_sigmoid(logits)
        log_one_minus_pi = jax.nn.log_sigmoid(-logits)
        zero = jnp.logaddexp(log_pi, log_one_minus_pi + count)
        return jnp.where(y == 0, zero, log_one_minus_pi + count)

    def negative_log_likelihood(
        self,
        observations: object,
        mean: object,
        *,
        dispersion: object | None = None,
        inflation_logits: object | None = None,
    ) -> jax.Array:
        return -self.log_likelihood(
            observations, mean, dispersion=dispersion, inflation_logits=inflation_logits
        )

    def mean(
        self, count_mean: object, *, inflation_logits: object | None = None
    ) -> jax.Array:
        mu = jnp.asarray(count_mean)
        return (
            mu
            if inflation_logits is None
            else jax.nn.sigmoid(-jnp.asarray(inflation_logits)) * mu
        )

    def variance(
        self,
        mean: object,
        *,
        dispersion: object | None = None,
        inflation_logits: object | None = None,
    ) -> jax.Array:
        mu = jnp.asarray(mean)
        count_variance = mu
        if self.family in {"negative_binomial", "zinb"}:
            if dispersion is None:
                raise ValueError("negative-binomial families require dispersion.")
            count_variance = mu + mu * mu / jnp.asarray(dispersion)
        if inflation_logits is None:
            return count_variance
        pi = jax.nn.sigmoid(jnp.asarray(inflation_logits))
        return (1.0 - pi) * count_variance + pi * (1.0 - pi) * mu * mu

    def zero_probability(
        self,
        mean: object,
        *,
        dispersion: object | None = None,
        inflation_logits: object | None = None,
    ) -> jax.Array:
        mu = jnp.asarray(mean)
        if self.family in {"poisson", "zip"}:
            count_zero = jnp.exp(-mu)
        else:
            if dispersion is None:
                raise ValueError("negative-binomial families require dispersion.")
            r = jnp.asarray(dispersion)
            count_zero = jnp.exp(r * (jnp.log(r) - jnp.log(r + mu)))
        if inflation_logits is None:
            return count_zero
        pi = jax.nn.sigmoid(jnp.asarray(inflation_logits))
        return pi + (1.0 - pi) * count_zero


def resolve_observation_model(family: str) -> CountObservationModel:
    return CountObservationModel(family)
