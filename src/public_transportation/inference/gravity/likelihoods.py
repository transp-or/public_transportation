"""JAX-compatible likelihoods for aggregate gravity observations.

This module is intentionally independent of the gravity objective.  It maps
predicted attribute-bin masses to conditional distributions within each
departure stratum and evaluates one of the uncertainty models declared by the
Phase-2 aggregate contract:

* multinomial;
* Dirichlet--multinomial; or
* a tempered multinomial composite likelihood.

The GPS sample contributes relative frequencies within a stratum.  Its total
row count is therefore used as the multinomial size, but never as an absolute
passenger-volume exposure.  Objective integration is deferred to Phase 5.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

from .aggregate import (
    GravityAggregateHistogram,
    GravityAggregateObservation,
    GravityAggregateUncertainty,
)

_PROBABILITY_FLOOR = 1.0e-30
_STRATUM_MASS_FLOOR = 1.0e-30


def _is_tracer(value: object) -> bool:
    return isinstance(value, jax.core.Tracer)


def _validate_concrete_masses(value: object) -> None:
    if _is_tracer(value):
        return
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError("predicted aggregate masses must be finite.")
    if np.any(array < 0.0):
        raise ValueError("predicted aggregate masses must be non-negative.")


def _prepare_masses(
    predicted_masses: object, *, histogram: GravityAggregateHistogram
) -> jax.Array:
    _validate_concrete_masses(predicted_masses)
    value = jnp.asarray(predicted_masses)
    expected = (len(histogram.strata), len(histogram.bins))
    if value.ndim == 1:
        if value.shape != (expected[0] * expected[1],):
            raise ValueError(
                "predicted aggregate masses must have shape "
                f"{expected} or ({expected[0] * expected[1]},), got {value.shape}."
            )
        value = value.reshape(expected)
    elif value.shape != expected:
        raise ValueError(
            f"predicted aggregate masses must have shape {expected}, got {value.shape}."
        )
    return value


def normalize_aggregate_masses(
    predicted_masses: object,
    *,
    histogram: GravityAggregateHistogram,
) -> jax.Array:
    """Normalize predicted masses independently within every stratum.

    A zero-mass stratum has no observed information when its observed total is
    zero.  It is represented by an all-zero probability row, so its likelihood
    contribution is exactly zero.  A positive observed count in such a stratum
    is driven to a very small probability and should already have been stopped
    by the Phase-3 support audit.
    """

    masses = _prepare_masses(predicted_masses, histogram=histogram)
    totals = jnp.sum(masses, axis=1, keepdims=True)
    safe_totals = jnp.maximum(
        totals, jnp.asarray(_STRATUM_MASS_FLOOR, dtype=masses.dtype)
    )
    return masses / safe_totals


def _observed_counts(histogram: GravityAggregateHistogram, *, dtype: Any) -> jax.Array:
    return jnp.asarray(
        [list(stratum.counts) for stratum in histogram.strata],
        dtype=dtype,
    )


def _multinomial_log_likelihood(
    probabilities: jax.Array,
    counts: jax.Array,
) -> jax.Array:
    totals = jnp.sum(counts, axis=1)
    log_probability = jnp.log(jnp.maximum(probabilities, _PROBABILITY_FLOOR))
    combinatorial = jsp.gammaln(totals + 1.0) - jnp.sum(
        jsp.gammaln(counts + 1.0), axis=1
    )
    weighted_log_probability = jnp.sum(
        jnp.where(counts > 0.0, counts * log_probability, 0.0), axis=1
    )
    return jnp.sum(combinatorial + weighted_log_probability)


def _log_rising_factorial(
    base: jax.Array, counts: jax.Array, *, maximum: int
) -> jax.Array:
    """Evaluate ``log((base)_n)`` without large-gamma cancellation."""

    if maximum == 0:
        return jnp.zeros_like(counts, dtype=base.dtype)
    indices = jnp.arange(maximum, dtype=base.dtype)
    terms = jnp.log(jnp.maximum(base[..., None] + indices, _PROBABILITY_FLOOR))
    return jnp.sum(
        jnp.where(indices < counts[..., None], terms, 0.0),
        axis=-1,
    )


def _dirichlet_multinomial_log_likelihood(
    probabilities: jax.Array,
    counts: jax.Array,
    concentration: float,
    *,
    maximum_count: int,
    maximum_total: int,
) -> jax.Array:
    totals = jnp.sum(counts, axis=1)
    concentration_value = jnp.asarray(concentration, dtype=probabilities.dtype)
    alpha = concentration_value * probabilities
    safe_alpha = jnp.maximum(alpha, _PROBABILITY_FLOOR)
    combinatorial = jsp.gammaln(totals + 1.0) - jnp.sum(
        jsp.gammaln(counts + 1.0), axis=1
    )
    normalization = -_log_rising_factorial(
        concentration_value,
        totals,
        maximum=maximum_total,
    )
    category_terms = _log_rising_factorial(
        safe_alpha,
        counts,
        maximum=maximum_count,
    )
    return jnp.sum(combinatorial + normalization + jnp.sum(category_terms, axis=1))


def _validate_uncertainty(uncertainty: GravityAggregateUncertainty) -> None:
    for name, value in (
        ("concentration", uncertainty.concentration),
        ("effective_sample_size", uncertainty.effective_sample_size),
    ):
        if value is not None and (not np.isfinite(float(value)) or float(value) <= 0.0):
            raise ValueError(f"uncertainty.{name} must be finite and positive.")
    if uncertainty.tempering is not None and (
        not np.isfinite(float(uncertainty.tempering))
        or not 0.0 <= float(uncertainty.tempering) <= 1.0
    ):
        raise ValueError("uncertainty.tempering must lie in [0, 1].")
    if uncertainty.likelihood == "multinomial":
        if any(
            value is not None
            for value in (
                uncertainty.concentration,
                uncertainty.effective_sample_size,
                uncertainty.tempering,
            )
        ):
            raise ValueError(
                "multinomial uncertainty must not include a scaling parameter."
            )
        return
    if uncertainty.likelihood == "dirichlet_multinomial":
        values = (
            uncertainty.concentration,
            uncertainty.effective_sample_size,
        )
        if sum(value is not None for value in values) != 1:
            raise ValueError(
                "dirichlet_multinomial requires exactly one concentration or effective sample size."
            )
        return
    if uncertainty.likelihood == "tempered_multinomial":
        if uncertainty.tempering is None:
            raise ValueError("tempered_multinomial requires a tempering factor.")
        if (
            uncertainty.concentration is not None
            or uncertainty.effective_sample_size is not None
        ):
            raise ValueError(
                "tempered_multinomial must not also include concentration or effective sample size."
            )
        return
    raise ValueError(f"unsupported aggregate likelihood {uncertainty.likelihood!r}.")


def aggregate_histogram_log_likelihood(
    predicted_masses: object,
    *,
    histogram: GravityAggregateHistogram,
    uncertainty: GravityAggregateUncertainty,
) -> jax.Array:
    """Evaluate one histogram's aggregate log likelihood.

    ``predicted_masses`` may be flattened in stratum-major order or have shape
    ``(num_strata, num_bins)``.  The uncertainty declaration is static metadata
    and is not included in the differentiable argument.
    """

    _validate_uncertainty(uncertainty)
    masses = _prepare_masses(predicted_masses, histogram=histogram)
    probabilities = normalize_aggregate_masses(masses, histogram=histogram)
    counts = _observed_counts(histogram, dtype=probabilities.dtype)
    if uncertainty.likelihood == "multinomial":
        return _multinomial_log_likelihood(probabilities, counts)
    if uncertainty.likelihood == "dirichlet_multinomial":
        concentration = uncertainty.concentration
        if concentration is None:
            concentration = uncertainty.effective_sample_size
        assert concentration is not None
        return _dirichlet_multinomial_log_likelihood(
            probabilities,
            counts,
            float(concentration),
            maximum_count=max(max(stratum.counts) for stratum in histogram.strata),
            maximum_total=max(stratum.total for stratum in histogram.strata),
        )
    assert uncertainty.tempering is not None
    multinomial = _multinomial_log_likelihood(probabilities, counts)
    return jnp.asarray(uncertainty.tempering, dtype=multinomial.dtype) * multinomial


@dataclass(frozen=True, slots=True)
class GravityAggregateLikelihoodEvaluation:
    """Breakdown of independent histogram likelihood contributions."""

    log_likelihood: jax.Array
    per_histogram_log_likelihood: tuple[jax.Array, ...]
    probabilities: tuple[jax.Array, ...]


def evaluate_gravity_aggregate_likelihood(
    predicted_masses: Mapping[str, object],
    *,
    observation: GravityAggregateObservation,
) -> GravityAggregateLikelihoodEvaluation:
    """Evaluate all histograms in a validated aggregate observation.

    ``predicted_masses`` is keyed by the histogram ``attribute`` name.  Each
    value is the output of the corresponding Phase-3 attribute response
    operator before normalization and fixed-offset addition.
    """

    if not isinstance(predicted_masses, Mapping):
        raise TypeError("predicted_masses must be a mapping keyed by attribute name.")
    values: list[jax.Array] = []
    probabilities: list[jax.Array] = []
    for histogram in observation.histograms:
        if histogram.attribute not in predicted_masses:
            raise KeyError(
                f"predicted aggregate masses are missing {histogram.attribute!r}."
            )
        value = predicted_masses[histogram.attribute]
        probabilities.append(normalize_aggregate_masses(value, histogram=histogram))
        values.append(
            aggregate_histogram_log_likelihood(
                value,
                histogram=histogram,
                uncertainty=observation.uncertainty,
            )
        )
    return GravityAggregateLikelihoodEvaluation(
        log_likelihood=jnp.sum(jnp.stack(values)),
        per_histogram_log_likelihood=tuple(values),
        probabilities=tuple(probabilities),
    )


# Descriptive aliases for callers that prefer a gravity-prefixed function.
gravity_aggregate_log_likelihood = aggregate_histogram_log_likelihood


__all__ = [
    "GravityAggregateLikelihoodEvaluation",
    "aggregate_histogram_log_likelihood",
    "evaluate_gravity_aggregate_likelihood",
    "gravity_aggregate_log_likelihood",
    "normalize_aggregate_masses",
]
