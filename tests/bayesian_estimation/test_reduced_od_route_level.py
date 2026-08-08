from __future__ import annotations

import numpy as np
import pytest

from public_transportation.inference.reduced_od import (
    RouteLevelCounts,
    RouteLevelIPFConfig,
    RouteLevelInfeasibleError,
    estimate_route_level_ipf,
)


def _counts(
    boarding=(10.0, 5.0, 5.0, 0.0),
    alighting=(0.0, 4.0, 9.0, 7.0),
    *,
    boarding_observed=None,
    alighting_observed=None,
) -> RouteLevelCounts:
    size = len(boarding)
    return RouteLevelCounts(
        route_pattern_id="route_pattern_000001",
        service_period_id="weekday",
        stop_ids=tuple(f"S{index}" for index in range(size)),
        boarding_counts=np.asarray(boarding),
        alighting_counts=np.asarray(alighting),
        boarding_observed=np.ones(size, dtype=bool)
        if boarding_observed is None
        else np.asarray(boarding_observed),
        alighting_observed=np.ones(size, dtype=bool)
        if alighting_observed is None
        else np.asarray(alighting_observed),
    )


def test_exact_marginals_and_triangular_support_are_recovered() -> None:
    result = estimate_route_level_ipf(_counts())

    np.testing.assert_allclose(
        result.fitted_boarding_counts, [10.0, 5.0, 5.0, 0.0], atol=1e-8
    )
    np.testing.assert_allclose(
        result.fitted_alighting_counts, [0.0, 4.0, 9.0, 7.0], atol=1e-8
    )
    assert np.all(result.leg_od_matrix[np.tril_indices(4)] == 0.0)
    np.testing.assert_allclose(
        result.alighting_probabilities[:3].sum(axis=1), 1.0
    )
    assert result.diagnostics.converged
    assert result.diagnostics.maximum_relative_residual <= 1e-10
    assert result.diagnostics.structural_support_size == 6


def test_explicit_noisy_total_reconciliation_is_audited() -> None:
    result = estimate_route_level_ipf(
        _counts(alighting=(0.0, 4.0, 9.0, 8.0)),
        config=RouteLevelIPFConfig(
            total_reconciliation="average_observed_totals"
        ),
    )

    assert result.reconciled_boarding_targets.sum() == pytest.approx(20.5)
    assert result.reconciled_alighting_targets.sum() == pytest.approx(20.5)
    quality = result.diagnostics.data_quality
    assert quality.original_total_imbalance == pytest.approx(-1.0)
    assert quality.maximum_marginal_adjustment > 0.0
    assert "explicitly rescaled" in quality.warnings[0]


def test_inconsistent_totals_fail_with_diagnostic_report() -> None:
    with pytest.raises(
        RouteLevelInfeasibleError, match="totals differ"
    ) as caught:
        estimate_route_level_ipf(
            _counts(alighting=(0.0, 4.0, 9.0, 8.0))
        )

    assert caught.value.data_quality.original_total_imbalance == -1.0
    assert caught.value.data_quality.total_reconciliation == "error"


@pytest.mark.parametrize(
    ("boarding", "alighting", "message"),
    [
        ((10.0, 0.0, 1.0), (0.0, 5.0, 6.0), "final stop"),
        ((0.0, 10.0, 0.0), (0.0, 5.0, 5.0), "cumulative alightings"),
        ((10.0, 0.0, 0.0), (1.0, 0.0, 9.0), "first stop"),
    ],
)
def test_impossible_triangular_marginals_fail_diagnostically(
    boarding, alighting, message
) -> None:
    with pytest.raises(RouteLevelInfeasibleError, match=message):
        estimate_route_level_ipf(_counts(boarding, alighting))


def test_zero_rows_and_columns_remain_exactly_zero() -> None:
    result = estimate_route_level_ipf(
        _counts(
            boarding=(10.0, 0.0, 0.0),
            alighting=(0.0, 0.0, 10.0),
        )
    )

    np.testing.assert_array_equal(
        result.leg_od_matrix,
        np.asarray([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    assert np.all(result.alighting_probabilities[1:] == 0.0)


def test_missing_marginal_is_unconstrained_and_reported() -> None:
    result = estimate_route_level_ipf(
        _counts(
            boarding=(10.0, 5.0, 0.0),
            alighting=(0.0, 4.0, 11.0),
            alighting_observed=(False, True, True),
        )
    )

    np.testing.assert_allclose(result.fitted_boarding_counts, [10.0, 5.0, 0.0])
    np.testing.assert_allclose(result.fitted_alighting_counts[1:], [4.0, 11.0])
    quality = result.diagnostics.data_quality
    assert quality.underdetermined
    assert "seed-dependent" in quality.warnings[0]


def test_transfer_semantics_are_explicitly_leg_level() -> None:
    result = estimate_route_level_ipf(_counts())

    assert result.level == "leg_level"
    assert result.boarding_semantics == "leg_boarding_unclassified"
    assert result.journey_od_compatible is False


def test_arrays_are_owned_and_immutable() -> None:
    boarding = np.asarray([10.0, 0.0, 0.0])
    counts = _counts(boarding=boarding, alighting=(0.0, 0.0, 10.0))
    boarding[0] = 99.0
    result = estimate_route_level_ipf(counts)

    assert counts.boarding_counts[0] == 10.0
    assert not counts.boarding_counts.flags.writeable
    assert not result.leg_od_matrix.flags.writeable
    assert not result.alighting_probabilities.flags.writeable


def test_invalid_masks_seed_and_nonconvergence_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero placeholders"):
        _counts(
            boarding=(10.0, 5.0, 0.0),
            alighting=(0.0, 4.0, 11.0),
            boarding_observed=(True, False, True),
        )

    seed = np.triu(np.ones((4, 4)), k=1)
    seed[0, 2] = 0.0
    with pytest.raises(ValueError, match="positive on every"):
        estimate_route_level_ipf(_counts(), seed_matrix=seed)

    with pytest.raises(RouteLevelInfeasibleError, match="did not converge"):
        estimate_route_level_ipf(
            _counts(),
            config=RouteLevelIPFConfig(
                tolerance=1e-15,
                max_iterations=1,
            ),
        )
