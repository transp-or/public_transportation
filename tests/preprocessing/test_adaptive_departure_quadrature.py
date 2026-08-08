from __future__ import annotations

import json

import pytest

from public_transportation.preprocessing.reduced_od import (
    DepartureTimeSamplingConfig,
    JourneyTimePeriod,
    ResponseCellKey,
    SparseIntervalContribution,
    SparseWeightedResponse,
    add_sparse_interval_contributions,
    generate_fixed_time_step_samples,
    integrate_adaptive_departure_response,
    integrate_adaptive_departure_responses,
)


CELL = ResponseCellKey("A", "B", "P")
ONE = SparseWeightedResponse((1,), (1.0,))
TWO = SparseWeightedResponse((1,), (2.0,))
OTHER = SparseWeightedResponse((2,), (1.0,))


def _adaptive(**changes: object) -> DepartureTimeSamplingConfig:
    values: dict[str, object] = {
        "strategy": "adaptive_service_aware",
        "infeasible_policy": "preserve_mass",
        "initial_interval_seconds": 100,
        "minimum_interval_seconds": 1,
        "response_tolerance": 1.0e-6,
        "maximum_samples_per_cell": 128,
    }
    values.update(changes)
    return DepartureTimeSamplingConfig(**values)  # type: ignore[arg-type]


def _integral(**changes: object) -> DepartureTimeSamplingConfig:
    values: dict[str, object] = {
        "comparison_mode": "integral_response",
        "absolute_response_tolerance": 0.0,
        "relative_response_tolerance": 1.0e-3,
    }
    values.update(changes)
    return _adaptive(**values)


def test_integral_constant_response_needs_only_embedded_baseline() -> None:
    result = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=lambda _: ONE,
        config=_integral(),
    )
    assert result.diagnostics.routing_evaluations == 3
    assert result.diagnostics.refined_subintervals == 0
    assert result.diagnostics.global_target_achieved
    assert result.diagnostics.estimated_absolute_integration_error == 0.0
    assert result.diagnostics.quadrature_rule == "embedded_midpoint_integral"
    assert result.averaged_response == ONE


def test_integral_quadratic_error_decreases_with_refinement() -> None:
    def response(seconds: float) -> SparseWeightedResponse:
        return SparseWeightedResponse((1,), ((seconds / 100.0) ** 2,))

    loose = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=response,
        config=_integral(relative_response_tolerance=0.2),
    )
    strict = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=response,
        config=_integral(relative_response_tolerance=0.01),
    )
    assert strict.diagnostics.routing_evaluations > loose.diagnostics.routing_evaluations
    assert (
        strict.diagnostics.estimated_absolute_integration_error
        < loose.diagnostics.estimated_absolute_integration_error
    )


def test_integral_support_change_is_error_not_mandatory_instability() -> None:
    result = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=lambda seconds: ONE if seconds < 50 else OTHER,
        config=_integral(relative_response_tolerance=0.02),
    )
    assert result.diagnostics.support_additions + result.diagnostics.support_removals > 0
    assert result.diagnostics.global_target_achieved
    assert result.diagnostics.unresolved_interval_weight == 0.0
    assert result.averaged_response.values == pytest.approx((0.5, 0.5), abs=0.02)


def test_integral_infeasible_parent_and_feasible_child_preserves_mass() -> None:
    result = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=lambda seconds: ONE if seconds < 40 else None,
        config=_integral(relative_response_tolerance=0.02),
    )
    assert result.diagnostics.refined_subintervals > 0
    assert result.diagnostics.feasible_time_fraction > 0.0
    assert result.diagnostics.infeasible_time_fraction > 0.0
    assert result.diagnostics.total_quadrature_weight == pytest.approx(1.0)


def test_integral_narrow_feasible_window_is_seeded_by_service_boundaries() -> None:
    result = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=lambda seconds: ONE if 45 <= seconds < 55 else None,
        config=_integral(relative_response_tolerance=0.02),
        service_boundary_seconds=(45, 55),
    )
    assert result.diagnostics.feasible_time_fraction > 0.0
    assert result.averaged_response.values[0] > 0.0
    assert result.diagnostics.service_boundary_safeguard.startswith(
        "bounded_timetable_edges"
    )


def test_integral_budget_exhaustion_separates_error_and_probability_mass() -> None:
    result = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=lambda seconds: SparseWeightedResponse((1,), (seconds**2,)),
        config=_integral(maximum_samples_per_cell=7, relative_response_tolerance=0.0),
    )
    assert result.diagnostics.sample_cap_reached
    assert result.diagnostics.unresolved_estimated_error > 0.0
    assert result.diagnostics.unresolved_interval_weight > 0.0
    assert result.diagnostics.infeasible_time_fraction == 0.0


def test_integral_mode_has_distinct_schema_identity_and_deterministic_cache() -> None:
    evaluated: list[float] = []

    def response(seconds: float) -> SparseWeightedResponse:
        evaluated.append(seconds)
        return SparseWeightedResponse((int(seconds >= 50),), (1.0,))

    integral = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=response,
        config=_integral(relative_response_tolerance=0.05),
        service_boundary_seconds=(50,),
    )
    pointwise = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=lambda seconds: SparseWeightedResponse(
            (int(seconds >= 50),), (1.0,)
        ),
        config=_adaptive(comparison_mode="aggregate_response"),
    )
    assert len(evaluated) == len(set(evaluated))
    assert integral.diagnostics.fingerprint != pointwise.diagnostics.fingerprint
    assert integral.diagnostics.requested_comparison_mode == "integral_response"
    assert integral.diagnostics.effective_comparison_mode == "integral_response"


def test_sparse_interval_contributions_add_without_dense_materialization() -> None:
    left = SparseIntervalContribution(
        0,
        50,
        "test",
        (25,),
        SparseWeightedResponse((1, 1000), (0.2, 0.3)),
        0.5,
        0.0,
        0.5,
    )
    right = SparseIntervalContribution(
        50,
        100,
        "test",
        (75,),
        SparseWeightedResponse((2, 1000), (0.1, 0.4)),
        0.25,
        0.25,
        0.5,
    )
    total = add_sparse_interval_contributions(left, right)
    assert total.weighted_response.measurement_indices == (1, 2, 1000)
    assert total.weighted_response.values == pytest.approx((0.2, 0.1, 0.7))
    assert total.feasible_time_weight == pytest.approx(0.75)
    assert total.infeasible_time_weight == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("minutes", "expected_weights"),
    [
        (3, (1 / 3, 1 / 3, 1 / 3)),
        (5, (0.2,) * 5),
        (7, (1 / 7,) * 7),
        (13, (1 / 13,) * 13),
        (60, (1 / 60,) * 60),
    ],
)
def test_fixed_one_minute_bins_have_exact_partial_weights(
    minutes: int, expected_weights: tuple[float, ...]
) -> None:
    samples = generate_fixed_time_step_samples(
        origin_period_groups=(("A", "P"),),
        time_periods=(JourneyTimePeriod("P", 0, minutes * 60),),
        config=DepartureTimeSamplingConfig(
            strategy="fixed_time_step", time_step_seconds=60
        ),
    )
    assert tuple(item.original_weight for item in samples) == pytest.approx(
        expected_weights
    )
    assert sum(item.original_weight for item in samples) == pytest.approx(1.0)


def test_thirteen_minute_five_minute_reference_uses_5_5_3_bins() -> None:
    samples = generate_fixed_time_step_samples(
        origin_period_groups=(("A", "P"),),
        time_periods=(JourneyTimePeriod("P", 0, 13 * 60),),
        config=DepartureTimeSamplingConfig(
            strategy="fixed_time_step", time_step_seconds=300
        ),
    )
    assert [item.desired_departure_seconds for item in samples] == [150, 450, 690]
    assert [item.original_weight for item in samples] == pytest.approx(
        [5 / 13, 5 / 13, 3 / 13]
    )


def test_constant_response_is_accepted_and_merged_without_over_refinement() -> None:
    result = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=lambda _: ONE,
        config=_adaptive(),
    )
    assert result.diagnostics.routing_evaluations == 3
    assert result.diagnostics.refined_subintervals == 0
    assert result.diagnostics.accepted_subintervals == 1
    assert result.diagnostics.quadrature_converged
    assert result.responses[0].weight == pytest.approx(1.0)
    assert result.averaged_response == ONE


def test_service_boundary_refines_locally_and_preserves_infeasible_mass() -> None:
    evaluated: list[float] = []

    def response(seconds: float) -> SparseWeightedResponse | None:
        evaluated.append(seconds)
        return None if seconds < 50 else ONE

    result = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=response,
        config=_adaptive(),
    )
    assert result.diagnostics.refined_subintervals >= 1
    assert result.diagnostics.infeasible_time_fraction == pytest.approx(0.5)
    assert result.diagnostics.feasible_time_fraction == pytest.approx(0.5)
    assert result.averaged_response.values == pytest.approx((0.5,))
    assert len(evaluated) == len(set(evaluated))
    assert len(evaluated) < 100


def test_multiple_support_and_probability_changes_are_detected() -> None:
    def response(seconds: float) -> SparseWeightedResponse:
        if seconds < 30:
            return ONE
        if seconds < 70:
            return TWO
        return OTHER

    result = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=response,
        config=_adaptive(minimum_interval_seconds=2),
    )
    assert result.diagnostics.refined_subintervals > 1
    assert result.diagnostics.response_support_changes > 0
    assert result.diagnostics.unique_responses == 3


def test_completely_infeasible_interval_is_valid_zero_response() -> None:
    result = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=lambda _: None,
        config=_adaptive(),
    )
    assert result.diagnostics.infeasible_time_fraction == 1.0
    assert result.diagnostics.total_quadrature_weight == pytest.approx(1.0)
    assert result.averaged_response == SparseWeightedResponse((), ())


def test_sample_cap_and_minimum_resolution_are_explicit_quality_warnings() -> None:
    def continuous(seconds: float) -> SparseWeightedResponse:
        return SparseWeightedResponse((1,), (seconds + 1.0,))
    events: list[dict[str, object]] = []
    capped = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=continuous,
        config=_adaptive(maximum_samples_per_cell=3),
        progress=lambda event: events.append(dict(event)),
    )
    assert capped.diagnostics.sample_cap_reached
    assert capped.diagnostics.unresolved_interval_weight > 0
    assert not capped.diagnostics.quadrature_converged
    assert events[-1]["sample_cap_reached"] is True
    assert events[-1]["estimated_remaining_routing_evaluations"] == 0
    assert events[-1]["estimated_remaining_seconds"] == pytest.approx(0.0)
    minimum = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=continuous,
        config=_adaptive(minimum_interval_seconds=25),
    )
    assert minimum.diagnostics.minimum_resolution_reached_with_instability


def test_determinism_fingerprint_invalidation_and_json_diagnostics() -> None:
    first = integrate_adaptive_departure_response(
        cell_key=CELL, start_seconds=0, end_seconds=100,
        evaluator=lambda _: ONE, config=_adaptive()
    )
    second = integrate_adaptive_departure_response(
        cell_key=CELL, start_seconds=0, end_seconds=100,
        evaluator=lambda _: ONE, config=_adaptive()
    )
    changed = integrate_adaptive_departure_response(
        cell_key=CELL, start_seconds=0, end_seconds=100,
        evaluator=lambda _: ONE, config=_adaptive(response_tolerance=0.1)
    )
    assert first.diagnostics.fingerprint == second.diagnostics.fingerprint
    assert first.diagnostics.fingerprint != changed.diagnostics.fingerprint
    json.dumps(first.diagnostics.__dict__ if hasattr(first.diagnostics, "__dict__") else {
        name: getattr(first.diagnostics, name)
        for name in first.diagnostics.__dataclass_fields__
    })


def test_batch_progress_is_monotonic_and_contains_eta_fields() -> None:
    events: list[dict[str, object]] = []
    cells = (
        (ResponseCellKey("A", "B", "P"), 0.0, 100.0),
        (ResponseCellKey("A", "C", "P"), 0.0, 100.0),
    )
    integrate_adaptive_departure_responses(
        cells=cells,
        evaluator=lambda _cell, _seconds: ONE,
        config=_adaptive(),
        progress=lambda event: events.append(dict(event)),
    )
    assert [event["completed_cells"] for event in events] == [1, 2]
    required = {
        "routing_evaluations",
        "mean_evaluations_per_cell",
        "accepted_subintervals",
        "refined_subintervals",
        "cache_hits",
        "elapsed_seconds",
        "throughput_cells_per_second",
        "estimated_remaining_seconds",
        "current_infeasible_fraction",
        "sample_cap_count",
        "unresolved_count",
    }
    assert required <= events[-1].keys()


def test_long_interval_baseline_covers_every_coarse_interval_before_refinement() -> None:
    evaluated: list[float] = []

    def adversarial(seconds: float) -> SparseWeightedResponse:
        evaluated.append(seconds)
        if seconds < 900:
            return ONE if int(seconds // 100) % 2 else TWO
        return OTHER

    result = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=4500,
        evaluator=adversarial,
        config=_adaptive(
            initial_interval_seconds=900,
            minimum_interval_seconds=30,
            maximum_samples_per_cell=13,
        ),
    )
    assert result.diagnostics.initial_subintervals == 5
    assert result.diagnostics.reserved_baseline_evaluations == 11
    assert result.diagnostics.initial_subintervals_evaluated == 5
    assert result.diagnostics.baseline_evaluations == 11
    assert result.diagnostics.refinement_evaluations == 2
    assert max(evaluated[:11]) == 4500
    assert result.diagnostics.unresolved_interval_weight <= 0.2 + 1.0e-12


def test_budget_smaller_than_whole_period_baseline_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 11"):
        integrate_adaptive_departure_response(
            cell_key=CELL,
            start_seconds=0,
            end_seconds=4500,
            evaluator=lambda _: ONE,
            config=_adaptive(
                initial_interval_seconds=900,
                maximum_samples_per_cell=10,
            ),
        )


def test_requested_and_effective_comparison_modes_are_explicit() -> None:
    events: list[dict[str, object]] = []
    result = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=lambda _: ONE,
        config=_adaptive(comparison_mode="assignment_response"),
        effective_comparison_mode="service_signature",
        progress=lambda event: events.append(dict(event)),
    )
    assert result.diagnostics.requested_comparison_mode == "assignment_response"
    assert result.diagnostics.effective_comparison_mode == "service_signature"
    assert events[-1]["requested_comparison_mode"] == "assignment_response"
    assert events[-1]["effective_comparison_mode"] == "service_signature"


def test_response_changes_below_tolerance_do_not_refine() -> None:
    result = integrate_adaptive_departure_response(
        cell_key=CELL,
        start_seconds=0,
        end_seconds=100,
        evaluator=lambda seconds: SparseWeightedResponse(
            (1,), (1.0 + seconds * 1.0e-8,)
        ),
        config=_adaptive(response_tolerance=1.0e-5),
    )
    assert result.diagnostics.refined_subintervals == 0
    assert result.diagnostics.quadrature_converged


@pytest.mark.parametrize(
    "changes",
    [
        {"time_step_seconds": 0},
        {"maximum_samples_per_cell": 0},
        {"response_tolerance": -1},
        {"minimum_interval_seconds": 901},
        {"comparison_mode": "unknown"},
    ],
)
def test_new_configuration_validation(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _adaptive(**changes)
