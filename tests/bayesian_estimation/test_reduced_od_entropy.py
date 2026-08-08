from __future__ import annotations

import numpy as np
import pytest

from public_transportation.inference.reduced_od import (
    EntropyConfig,
    EntropyInfeasibleError,
    EntropySupport,
    JourneyMarginals,
    estimate_entropy_transport,
)


def _support() -> EntropySupport:
    return EntropySupport(
        origin_index=np.asarray([0, 0, 1, 1]),
        destination_index=np.asarray([0, 1, 0, 1]),
        generalized_cost=np.asarray([0.0, 1.0, 2.0, 0.5]),
        number_of_origins=2,
        number_of_destinations=2,
    )


def test_balanced_entropy_recovers_both_journey_marginals() -> None:
    marginals = JourneyMarginals(
        origin=np.asarray([3.0, 2.0]), destination=np.asarray([1.0, 4.0])
    )
    result = estimate_entropy_transport(
        _support(),
        marginals,
        config=EntropyConfig(epsilon=0.7, tolerance=1e-11),
    )
    np.testing.assert_allclose(result.fitted_origin, marginals.origin, atol=1e-9)
    np.testing.assert_allclose(
        result.fitted_destination, marginals.destination, atol=1e-9
    )
    assert result.diagnostics.converged
    assert result.diagnostics.mode == "balanced"
    assert not result.cell_flow.flags.writeable


def test_unbalanced_entropy_handles_inconsistent_totals_and_penalty_sensitivity() -> None:
    marginals = JourneyMarginals(
        origin=np.asarray([5.0, 3.0]), destination=np.asarray([2.0, 2.0])
    )
    low = estimate_entropy_transport(
        _support(),
        marginals,
        config=EntropyConfig(
            mode="unbalanced",
            epsilon=1.0,
            origin_penalty=1.0,
            destination_penalty=1.0,
            tolerance=1e-10,
        ),
    )
    high = estimate_entropy_transport(
        _support(),
        marginals,
        config=EntropyConfig(
            mode="unbalanced",
            epsilon=1.0,
            origin_penalty=100.0,
            destination_penalty=100.0,
            tolerance=1e-10,
        ),
    )
    assert low.diagnostics.fitted_total != pytest.approx(8.0)
    high_total_error = abs(high.cell_flow.sum() - 6.0)
    low_total_error = abs(low.cell_flow.sum() - 6.0)
    assert high_total_error < low_total_error


def test_balanced_disconnected_support_and_unequal_totals_fail_diagnostically() -> None:
    disconnected = EntropySupport(
        origin_index=np.asarray([0]),
        destination_index=np.asarray([0]),
        generalized_cost=np.asarray([1.0]),
        number_of_origins=2,
        number_of_destinations=2,
    )
    with pytest.raises(EntropyInfeasibleError, match="unsupported"):
        estimate_entropy_transport(
            disconnected,
            JourneyMarginals(
                origin=np.asarray([1.0, 1.0]),
                destination=np.asarray([1.0, 1.0]),
            ),
        )
    with pytest.raises(EntropyInfeasibleError, match="totals"):
        estimate_entropy_transport(
            _support(),
            JourneyMarginals(
                origin=np.asarray([1.0, 1.0]),
                destination=np.asarray([1.0, 2.0]),
            ),
        )


def test_apc_semantics_are_explicitly_rejected() -> None:
    with pytest.raises(ValueError, match="raw APC"):
        JourneyMarginals(
            origin=np.asarray([1.0]),
            destination=np.asarray([1.0]),
            semantics="apc_boarding_alighting",  # type: ignore[arg-type]
        )


def test_log_domain_stability_and_warm_scalings() -> None:
    support = EntropySupport(
        origin_index=np.asarray([0, 0]),
        destination_index=np.asarray([0, 1]),
        generalized_cost=np.asarray([1.0e6, 1.0e6 + 1.0]),
        number_of_origins=1,
        number_of_destinations=2,
    )
    marginals = JourneyMarginals(
        origin=np.asarray([2.0]), destination=np.asarray([1.0, 1.0])
    )
    first = estimate_entropy_transport(
        support,
        marginals,
        config=EntropyConfig(epsilon=0.1, tolerance=1e-10),
    )
    warm = estimate_entropy_transport(
        support,
        marginals,
        config=EntropyConfig(epsilon=0.1, tolerance=1e-10),
        initial_log_origin_scaling=first.log_origin_scaling,
        initial_log_destination_scaling=first.log_destination_scaling,
    )
    assert np.all(np.isfinite(warm.cell_flow))
    np.testing.assert_allclose(warm.cell_flow, [1.0, 1.0], atol=1e-8)
    assert warm.diagnostics.iterations <= first.diagnostics.iterations
