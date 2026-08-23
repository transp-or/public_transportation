from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gammaln

from public_transportation.inference.gravity import (
    GravityAggregateBin,
    GravityAggregateHistogram,
    GravityAggregateLikelihoodEvaluation,
    GravityAggregateObservation,
    GravityAggregateStratum,
    GravityAggregateUncertainty,
    aggregate_histogram_log_likelihood,
    evaluate_gravity_aggregate_likelihood,
    normalize_aggregate_masses,
)


def histogram() -> GravityAggregateHistogram:
    return GravityAggregateHistogram(
        attribute="travel_time",
        unit="seconds",
        support=(0.0, 7200.0),
        bins=(
            GravityAggregateBin("short", 0.0, 600.0),
            GravityAggregateBin("long", 600.0, 7200.0),
        ),
        strata=(
            GravityAggregateStratum("morning", (3, 1), 4),
            GravityAggregateStratum("evening", (1, 3), 4),
        ),
    )


def second_histogram() -> GravityAggregateHistogram:
    return GravityAggregateHistogram(
        attribute="transfers",
        unit="count",
        support=(0.0, 3.0),
        bins=(
            GravityAggregateBin("none", 0.0, 1.0),
            GravityAggregateBin("one_or_more", 1.0, 3.0),
        ),
        strata=(
            GravityAggregateStratum("morning", (2, 2), 4),
            GravityAggregateStratum("evening", (3, 1), 4),
        ),
    )


def observation(
    uncertainty: GravityAggregateUncertainty,
) -> GravityAggregateObservation:
    return GravityAggregateObservation(
        schema_version=1,
        channel_name="gps_trip_attributes",
        kind="trip_attribute_distribution",
        histograms=(histogram(), second_histogram()),
        metadata={
            "collection_period": "fixture",
            "valid_journeys": 8,
            "excluded_journeys": 0,
            "cleaning_reasons": {},
            "apc_overlap_policy": "recorded",
        },
        uncertainty=uncertainty,
        source_path="fixture.json",
        file_sha256="file",
        content_sha256="content",
    )


def test_normalization_is_independent_within_each_stratum():
    probabilities = normalize_aggregate_masses(
        np.asarray((3.0, 1.0, 2.0, 6.0)), histogram=histogram()
    )
    np.testing.assert_allclose(
        np.asarray(probabilities), np.asarray(((0.75, 0.25), (1 / 4, 3 / 4)))
    )
    np.testing.assert_allclose(np.asarray(probabilities).sum(axis=1), (1.0, 1.0))


def test_multinomial_matches_closed_form():
    with jax.enable_x64():
        item = histogram()
        masses = np.asarray((3.0, 1.0, 1.0, 3.0))
        value = aggregate_histogram_log_likelihood(
            masses,
            histogram=item,
            uncertainty=GravityAggregateUncertainty("multinomial"),
        )
        expected = 2 * (
            gammaln(5) - gammaln(4) - gammaln(2) + 3 * np.log(0.75) + np.log(0.25)
        )
        np.testing.assert_allclose(float(value), expected)


def test_dirichlet_multinomial_large_concentration_approaches_multinomial():
    with jax.enable_x64():
        item = histogram()
        masses = np.asarray((3.0, 1.0, 1.0, 3.0))
        multinomial = aggregate_histogram_log_likelihood(
            masses,
            histogram=item,
            uncertainty=GravityAggregateUncertainty("multinomial"),
        )
        dirichlet = aggregate_histogram_log_likelihood(
            masses,
            histogram=item,
            uncertainty=GravityAggregateUncertainty(
                "dirichlet_multinomial", concentration=1.0e8
            ),
        )
        np.testing.assert_allclose(float(dirichlet), float(multinomial), rtol=2e-6)


def test_small_dirichlet_concentration_allows_extra_dispersion_for_extreme_counts():
    item = replace(
        histogram(),
        strata=(
            GravityAggregateStratum("morning", (4, 0), 4),
            GravityAggregateStratum("evening", (0, 4), 4),
        ),
    )
    masses = np.asarray((1.0, 1.0, 1.0, 1.0))
    multinomial = aggregate_histogram_log_likelihood(
        masses,
        histogram=item,
        uncertainty=GravityAggregateUncertainty("multinomial"),
    )
    dispersed = aggregate_histogram_log_likelihood(
        masses,
        histogram=item,
        uncertainty=GravityAggregateUncertainty(
            "dirichlet_multinomial", concentration=2.0
        ),
    )
    assert float(dispersed) > float(multinomial)


def test_effective_sample_size_is_used_as_dirichlet_concentration():
    item = histogram()
    masses = np.asarray((3.0, 1.0, 1.0, 3.0))
    concentration = aggregate_histogram_log_likelihood(
        masses,
        histogram=item,
        uncertainty=GravityAggregateUncertainty(
            "dirichlet_multinomial", concentration=12.0
        ),
    )
    effective = aggregate_histogram_log_likelihood(
        masses,
        histogram=item,
        uncertainty=GravityAggregateUncertainty(
            "dirichlet_multinomial", effective_sample_size=12.0
        ),
    )
    np.testing.assert_allclose(concentration, effective)


def test_tempered_multinomial_zero_weight_removes_auxiliary_evidence():
    value = aggregate_histogram_log_likelihood(
        np.asarray((3.0, 1.0, 1.0, 3.0)),
        histogram=histogram(),
        uncertainty=GravityAggregateUncertainty("tempered_multinomial", tempering=0.0),
    )
    assert float(value) == pytest.approx(0.0)


def test_tempering_scales_multinomial_and_is_not_combined_with_dirichlet():
    item = histogram()
    masses = np.asarray((3.0, 1.0, 1.0, 3.0))
    base = aggregate_histogram_log_likelihood(
        masses,
        histogram=item,
        uncertainty=GravityAggregateUncertainty("multinomial"),
    )
    tempered = aggregate_histogram_log_likelihood(
        masses,
        histogram=item,
        uncertainty=GravityAggregateUncertainty("tempered_multinomial", tempering=0.25),
    )
    np.testing.assert_allclose(tempered, 0.25 * base)
    with pytest.raises(ValueError, match="exactly one"):
        aggregate_histogram_log_likelihood(
            masses,
            histogram=item,
            uncertainty=GravityAggregateUncertainty(
                "dirichlet_multinomial", concentration=2.0, effective_sample_size=2.0
            ),
        )


def test_likelihood_is_differentiable_and_matches_finite_difference():
    with jax.enable_x64():
        item = histogram()
        uncertainty = GravityAggregateUncertainty(
            "dirichlet_multinomial", concentration=8.0
        )
        raw = jnp.asarray((3.0, 1.0, 1.5, 2.5), dtype=jnp.float64)
        gradient = jax.grad(
            lambda value: aggregate_histogram_log_likelihood(
                value, histogram=item, uncertainty=uncertainty
            )
        )(raw)
        numerical = []
        step = 1.0e-5
        for index in range(raw.size):
            delta = np.zeros(4, dtype=np.float64)
            delta[index] = step
            plus = aggregate_histogram_log_likelihood(
                np.asarray(raw) + delta,
                histogram=item,
                uncertainty=uncertainty,
            )
            minus = aggregate_histogram_log_likelihood(
                np.asarray(raw) - delta,
                histogram=item,
                uncertainty=uncertainty,
            )
            numerical.append((float(plus) - float(minus)) / (2 * step))
        np.testing.assert_allclose(
            np.asarray(gradient), numerical, rtol=2e-6, atol=2e-6
        )


def test_multiple_histograms_are_summed_and_reported_separately():
    item = observation(GravityAggregateUncertainty("multinomial"))
    result = evaluate_gravity_aggregate_likelihood(
        {
            "travel_time": np.asarray((3.0, 1.0, 1.0, 3.0)),
            "transfers": np.asarray((2.0, 2.0, 3.0, 1.0)),
        },
        observation=item,
    )
    assert isinstance(result, GravityAggregateLikelihoodEvaluation)
    assert len(result.per_histogram_log_likelihood) == 2
    np.testing.assert_allclose(
        result.log_likelihood,
        sum(result.per_histogram_log_likelihood),
    )
    for probability in result.probabilities:
        np.testing.assert_allclose(np.asarray(probability).sum(axis=1), (1.0, 1.0))


@pytest.mark.parametrize(
    ("uncertainty", "message"),
    (
        (GravityAggregateUncertainty("dirichlet_multinomial"), "exactly one"),
        (
            GravityAggregateUncertainty("tempered_multinomial", tempering=1.5),
            "must lie in",
        ),
        (
            GravityAggregateUncertainty("dirichlet_multinomial", concentration=-1.0),
            "positive",
        ),
    ),
)
def test_invalid_uncertainty_is_rejected(uncertainty, message):
    with pytest.raises(ValueError, match=message):
        aggregate_histogram_log_likelihood(
            np.asarray((3.0, 1.0, 1.0, 3.0)),
            histogram=histogram(),
            uncertainty=uncertainty,
        )


def test_shape_and_negative_mass_errors_are_clear():
    with pytest.raises(ValueError, match="shape"):
        normalize_aggregate_masses(np.ones(3), histogram=histogram())
    with pytest.raises(ValueError, match="non-negative"):
        normalize_aggregate_masses(
            np.asarray((3.0, -1.0, 1.0, 3.0)), histogram=histogram()
        )
