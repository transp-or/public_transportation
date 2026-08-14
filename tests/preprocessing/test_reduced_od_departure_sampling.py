from __future__ import annotations

import json

import numpy as np
import pytest

from public_transportation.preprocessing.reduced_od import (
    DepartureSampleInfeasibilityReason,
    DepartureSamplingDiagnosticsReport,
    DepartureSamplingEvaluation,
    DepartureTimeSamplingConfig,
    DesiredDepartureRoutingResult,
    DesiredDepartureSample,
    JourneyAlternative,
    JourneyChoiceDiagnostics,
    JourneyChoiceResult,
    JourneyChoiceSet,
    JourneyEvent,
    JourneyEventKind,
    JourneyTimePeriod,
    RaptorTransitLeg,
    ResponseCellKey,
    average_desired_departure_response,
    build_departure_sampling_diagnostics,
    compare_departure_sampling_levels,
    generate_uniform_midpoint_samples,
    merge_sampled_journey_choices,
    preflight_departure_sampling,
    preflight_reduced_od_time_periods,
    recommend_departure_sampling_actions,
    canonicalize_probability_mass,
)


def _alternative(
    identifier: str,
    *,
    trip: str = "T",
    boarding_period: str = "P",
    desired_period: str | None = None,
) -> JourneyAlternative:
    leg = RaptorTransitLeg(trip, "A", "B", 100, 200)
    return JourneyAlternative(
        alternative_id=identifier * 64,
        origin_physical_stop_id="A",
        destination_physical_stop_id="B",
        origin_time_period_id=boarding_period,
        query_departure_seconds=90,
        arrival_seconds=200,
        travel_seconds=110,
        wait_seconds=10,
        walk_seconds=0,
        in_vehicle_seconds=100,
        transfers=0,
        transit_legs=(leg,),
        events=(
            JourneyEvent(
                100, JourneyEventKind.FIRST_BOARDING, "A", boarding_period, 0
            ),
            JourneyEvent(200, JourneyEventKind.FINAL_ALIGHTING, "B", boarding_period, 0),
        ),
        route_pattern_ids=(f"route-{trip}",),
        desired_departure_time_period_id=desired_period,
    )


def test_merge_accepts_joint_cross_boarding_period_choice_set() -> None:
    sample = DesiredDepartureSample("s", "A", "P", 90.0, 1.0)
    early = _alternative("e", trip="EARLY", desired_period="P")
    late = _alternative(
        "l", trip="LATE", boarding_period="NEXT", desired_period="P"
    )
    result = JourneyChoiceResult(
        (JourneyChoiceSet("A", "B", "P", (early, late), (0.4, 0.6)),),
        JourneyChoiceDiagnostics(1, 2, 2, 0, 1, 2, 0, 0),
    )
    merged = merge_sampled_journey_choices(
        samples=(sample,),
        sample_choices=(result,),
        candidate_cells=(ResponseCellKey("A", "B", "P"),),
    )
    choice = merged.journey_choices.choice_sets[0]
    assert {item.first_boarding_time_period_id for item in choice.alternatives} == {
        "P",
        "NEXT",
    }
    assert choice.initial_shares == pytest.approx((0.4, 0.6))


def test_merge_rejects_genuine_duplicate_canonical_demand_cell() -> None:
    sample = DesiredDepartureSample("s", "A", "P", 90.0, 1.0)
    alternative = _alternative("d", desired_period="P")
    duplicate = JourneyChoiceSet("A", "B", "P", (alternative,), (1.0,))
    result = JourneyChoiceResult(
        (duplicate, duplicate),
        JourneyChoiceDiagnostics(1, 2, 2, 0, 2, 1, 0, 0),
    )
    with pytest.raises(ValueError, match="duplicate canonical"):
        merge_sampled_journey_choices(
            samples=(sample,),
            sample_choices=(result,),
            candidate_cells=(ResponseCellKey("A", "B", "P"),),
        )


def test_uniform_midpoints_are_exact_and_supply_independent() -> None:
    periods = (JourneyTimePeriod("P", 100, 111),)
    config = DepartureTimeSamplingConfig(samples_per_period=3)
    first = generate_uniform_midpoint_samples(
        origin_physical_stop_ids=("A",), time_periods=periods, config=config
    )
    second = generate_uniform_midpoint_samples(
        origin_physical_stop_ids=("A",), time_periods=periods, config=config
    )

    assert [item.desired_departure_seconds for item in first] == pytest.approx(
        [100 + 11 / 6, 100 + 11 / 2, 100 + 55 / 6]
    )
    assert [item.original_weight for item in first] == pytest.approx([1 / 3] * 3)
    assert first == second
    assert config.count_for_period("P") == 3
    preflight = preflight_departure_sampling(
        origin_period_groups=1001,
        samples_per_group=12,
        single_sample_support_nonzeros=100,
        observed_query_seconds=0.1,
        temporary_bytes_per_query=1024,
    )
    assert preflight.total_journey_queries == 12012
    assert preflight.estimated_wall_seconds == pytest.approx(1201.2)
    assert preflight.estimated_temporary_bytes == 1024


def test_time_period_preflight_reports_unequal_durations_and_coverage() -> None:
    periods = (
        JourneyTimePeriod("short", 0, 300),
        JourneyTimePeriod("long", 300, 1200),
    )
    report = preflight_reduced_od_time_periods(
        periods,
        relevant_event_seconds=(150, 900),
        sampling_config=DepartureTimeSamplingConfig(
            strategy="fixed_count", samples_per_period={"short": 3, "long": 9}
        ),
    )
    assert report.valid
    assert dict(report.durations_seconds) == {"short": 300, "long": 900}
    assert report.covered_event_count == 2
    assert report.sampling_resolution_seconds == {"short": 100.0, "long": 100.0}


def test_time_period_preflight_blocks_overlap_gap_event_and_missing_resolution() -> None:
    periods = (
        JourneyTimePeriod("first", 0, 600),
        JourneyTimePeriod("second", 500, 1200),
        JourneyTimePeriod("third", 1500, 1800),
    )
    report = preflight_reduced_od_time_periods(
        periods,
        relevant_event_seconds=(1400,),
        sampling_config=DepartureTimeSamplingConfig(
            strategy="fixed_count", samples_per_period={"first": 2, "second": 2}
        ),
    )
    assert not report.valid
    assert {issue.code for issue in report.issues} >= {
        "overlapping_periods",
        "period_gap",
        "event_outside_periods",
        "sampling_configuration_missing",
    }
def test_sparse_origin_period_support_avoids_cartesian_queries() -> None:
    periods = (
        JourneyTimePeriod("AM", 0, 10),
        JourneyTimePeriod("PM", 10, 20),
    )
    samples = generate_uniform_midpoint_samples(
        origin_period_groups=(("A", "AM"), ("B", "PM")),
        time_periods=periods,
        config=DepartureTimeSamplingConfig(samples_per_period=3),
    )
    assert {
        (item.origin_physical_stop_id, item.time_period_id) for item in samples
    } == {("A", "AM"), ("B", "PM")}
    assert len(samples) == 6
    preflight = preflight_departure_sampling(
        origin_period_groups=2,
        samples_per_group=3,
        distinct_origins=2,
        number_of_periods=2,
        observed_query_seconds=0.5,
    )
    assert preflight.cartesian_group_count == 4
    assert preflight.avoided_cartesian_groups == 2
    assert preflight.hypothetical_cartesian_queries == 12
    assert preflight.avoided_queries == 6
    assert preflight.expected_wall_seconds_saved == pytest.approx(3.0)


def test_canonical_fixed_status_is_not_changed_by_timetable_feasibility() -> None:
    samples = (DesiredDepartureSample("s", "A", "P", 100.0, 1.0),)
    result = JourneyChoiceResult(
        choice_sets=(JourneyChoiceSet("A", "B", "P", (_alternative("z"),), (1.0,)),),
        diagnostics=JourneyChoiceDiagnostics(1, 1, 1, 0, 1, 1, 0, 0),
    )
    key = ResponseCellKey("A", "B", "P")
    merged = merge_sampled_journey_choices(
        samples=samples,
        sample_choices=(result,),
        candidate_cells=(key,),
        cell_status={key: "fixed_zero"},
    )
    assert merged.journey_choices.choice_sets == ()
    assert merged.cells[0].cell_status == "fixed_zero"
    assert merged.cells[0].timetable_feasible_fixed_zero

    positive = merge_sampled_journey_choices(
        samples=samples,
        sample_choices=(result,),
        candidate_cells=(key,),
        cell_status={key: "fixed_positive"},
    )
    assert len(positive.journey_choices.choice_sets) == 1

    empty = JourneyChoiceResult(
        choice_sets=(),
        diagnostics=JourneyChoiceDiagnostics(0, 0, 0, 0, 0, 0, 0, 0),
    )
    with pytest.raises(ValueError, match="fixed-positive cell cannot be assigned"):
        merge_sampled_journey_choices(
            samples=samples,
            sample_choices=(empty,),
            candidate_cells=(key,),
            cell_status={key: "fixed_positive"},
        )


@pytest.mark.parametrize(
    ("fraction", "classification"),
    [
        (0.0, "frozen_no_feasible_sample"),
        (0.25, "excluded_low_feasibility"),
        (0.5, "warning"),
        (0.75, "warning"),
        (0.9, "normal"),
        (1.0, "normal"),
    ],
)
def test_feasibility_thresholds_and_conditioning(
    fraction: float, classification: str
) -> None:
    total = 20
    feasible_count = round(fraction * total)
    alternative = _alternative("a")
    results = []
    for index in range(total):
        sample = DesiredDepartureSample(f"s{index}", "A", "P", 100 + index, 1.0 / total)
        feasible = index < feasible_count
        results.append(
            DesiredDepartureRoutingResult(
                sample=sample,
                feasible=feasible,
                infeasibility_reason=(
                    None
                    if feasible
                    else DepartureSampleInfeasibilityReason.NO_SERVICE_AFTER_DESIRED_TIME
                ),
                alternatives=(alternative,) if feasible else (),
                conditional_route_shares=(1.0,) if feasible else (),
            )
        )
    averaged = average_desired_departure_response(
        cell_key=ResponseCellKey("A", "B", "P"),
        routing_results=results,
        alternative_responses={alternative.alternative_id: {0: 1.0, 2: 2.0}},
    )
    assert averaged.original_feasible_weight == pytest.approx(fraction)
    assert averaged.original_infeasible_weight == pytest.approx(1.0 - fraction)
    assert averaged.classification == classification
    if feasible_count:
        assert sum(averaged.conditional_sample_weights) == pytest.approx(1.0)
        assert averaged.averaged_response.values == pytest.approx((1.0, 2.0))
    else:
        assert averaged.averaged_response.values == ()


def test_identical_responses_accumulate_original_sample_weights() -> None:
    alternative = _alternative("b")
    samples = (
        DesiredDepartureSample("s0", "A", "P", 100.0, 0.25),
        DesiredDepartureSample("s1", "A", "P", 200.0, 0.75),
    )
    routing = tuple(
        DesiredDepartureRoutingResult(sample, True, None, (alternative,), (1.0,))
        for sample in samples
    )
    averaged = average_desired_departure_response(
        cell_key=ResponseCellKey("A", "B", "P"),
        routing_results=routing,
        alternative_responses={alternative.alternative_id: {4: 3.0}},
    )
    assert averaged.conditional_sample_weights == pytest.approx((0.25, 0.75))
    assert averaged.averaged_response.measurement_indices == (4,)
    assert averaged.averaged_response.values == pytest.approx((3.0,))


def test_preserve_mass_keeps_conditional_shares_but_records_served_fraction() -> None:
    alternative = _alternative("p")
    samples = (
        DesiredDepartureSample("s0", "A", "P", 100.0, 0.25),
        DesiredDepartureSample("s1", "A", "P", 200.0, 0.75),
    )
    feasible = JourneyChoiceResult(
        (JourneyChoiceSet("A", "B", "P", (alternative,), (1.0,)),),
        JourneyChoiceDiagnostics(1, 1, 1, 0, 1, 1, 0, 0),
    )
    infeasible = JourneyChoiceResult(
        (), JourneyChoiceDiagnostics(0, 0, 0, 0, 0, 0, 0, 0)
    )
    merged = merge_sampled_journey_choices(
        samples=samples,
        sample_choices=(feasible, infeasible),
        config=DepartureTimeSamplingConfig(
            samples_per_period=2,
            infeasible_policy="preserve_mass",
            minimum_feasible_fraction=0.0,
            warning_feasible_fraction=0.0,
        ),
        candidate_cells=(ResponseCellKey("A", "B", "P"),),
    )
    choice = merged.journey_choices.choice_sets[0]
    assert choice.initial_shares == (1.0,)
    assert choice.served_time_fraction == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("raw", "expected", "applied"),
    [
        (0.0, 0.0, False),
        (1.0, 1.0, False),
        (1.0 - 5.0e-13, 1.0 - 5.0e-13, False),
        (1.0 + 5.0e-13, 1.0, True),
        (-5.0e-13, 0.0, True),
    ],
)
def test_probability_mass_canonicalization_is_tolerance_bounded(
    raw: float, expected: float, applied: bool
) -> None:
    result = canonicalize_probability_mass(raw, tolerance=1.0e-12)
    assert result.canonical_value == expected
    assert result.applied is applied
    assert result.delta == pytest.approx(expected - raw)


@pytest.mark.parametrize("raw", [1.0 + 2.0e-12, -2.0e-12, np.inf, np.nan])
def test_probability_mass_canonicalization_rejects_material_errors(raw: float) -> None:
    with pytest.raises(ValueError):
        canonicalize_probability_mass(raw, tolerance=1.0e-12)


def test_roundoff_above_one_is_canonicalized_before_choice_construction() -> None:
    alternative = _alternative("r")
    samples = (
        DesiredDepartureSample("s0", "A", "P", 100.0, 0.5),
        DesiredDepartureSample("s1", "A", "P", 200.0, 0.5 + 5.0e-13),
    )
    choice_result = JourneyChoiceResult(
        (JourneyChoiceSet("A", "B", "P", (alternative,), (1.0,)),),
        JourneyChoiceDiagnostics(1, 1, 1, 0, 1, 1, 0, 0),
    )
    merged = merge_sampled_journey_choices(
        samples=samples,
        sample_choices=(choice_result, choice_result),
        config=DepartureTimeSamplingConfig(
            samples_per_period=2,
            infeasible_policy="preserve_mass",
            minimum_feasible_fraction=0.0,
            warning_feasible_fraction=0.0,
        ),
        candidate_cells=(ResponseCellKey("A", "B", "P"),),
    )
    cell = merged.cells[0]
    assert cell.raw_feasible_time_fraction == pytest.approx(1.0 + 5.0e-13)
    assert cell.canonical_feasible_time_fraction == 1.0
    assert cell.mass_canonicalization_applied
    assert cell.original_feasible_weight + cell.original_infeasible_weight == 1.0
    assert merged.journey_choices.choice_sets[0].served_time_fraction == 1.0
    diagnostics = build_departure_sampling_diagnostics(
        sampled=merged,
        config=DepartureTimeSamplingConfig(samples_per_period=2),
    )
    assert diagnostics.network_totals["mass_canonicalization_count"] == 1
    assert diagnostics.network_totals[
        "maximum_absolute_mass_canonicalization_delta"
    ] == pytest.approx(5.0e-13)
    json.dumps(
        {
            "raw": cell.raw_feasible_time_fraction,
            "canonical": cell.canonical_feasible_time_fraction,
            "applied": cell.mass_canonicalization_applied,
            "delta": cell.mass_canonicalization_delta,
        }
    )


def test_sampled_journey_merging_preserves_joint_weights_and_desired_period() -> None:
    first = _alternative("c", trip="T1")
    second = _alternative("d", trip="T2")
    samples = (
        DesiredDepartureSample("s0", "A", "P", 100.0, 0.25),
        DesiredDepartureSample("s1", "A", "P", 200.0, 0.75),
    )

    def choices(alternative: JourneyAlternative) -> JourneyChoiceResult:
        return JourneyChoiceResult(
            choice_sets=(JourneyChoiceSet("A", "B", "P", (alternative,), (1.0,)),),
            diagnostics=JourneyChoiceDiagnostics(1, 1, 1, 0, 1, 1, 0, 0),
        )

    merged = merge_sampled_journey_choices(
        samples=samples,
        sample_choices=(choices(first), choices(second)),
        candidate_cells=(
            ResponseCellKey("A", "B", "P"),
            ResponseCellKey("A", "C", "P"),
        ),
    )
    cell = merged.journey_choices.choice_sets[0]
    assert sorted(cell.initial_shares) == pytest.approx((0.25, 0.75))
    assert all(item.demand_time_period_id == "P" for item in cell.alternatives)
    frozen = next(
        item
        for item in merged.cells
        if item.cell_key.destination_physical_stop_id == "C"
    )
    assert frozen.classification == "frozen_no_feasible_sample"

    diagnostics = build_departure_sampling_diagnostics(
        sampled=merged,
        config=DepartureTimeSamplingConfig(samples_per_period=2),
        observations=np.asarray([0.0, 2.0]),
        predicted_counts=np.asarray([1.0, 1.5]),
        measurement_types=("boarding", "alighting"),
        measurement_period_ids=("P", "P"),
        vehicle_journey_ids=("T1", "T2"),
    )
    assert diagnostics.accounting_valid
    assert diagnostics.network_totals["od_period_cells"] == 2
    assert diagnostics.observation_alignment is not None
    assert diagnostics.observation_alignment[
        "fraction_prediction_on_zero_rows"
    ] == pytest.approx(0.4)


def test_sampling_progress_and_convergence_are_serializable() -> None:
    events: list[dict[str, object]] = []
    generate_uniform_midpoint_samples(
        origin_physical_stop_ids=("A", "B"),
        time_periods=(JourneyTimePeriod("P", 0, 10),),
        config=DepartureTimeSamplingConfig(
            samples_per_period=3, progress_interval_groups=1
        ),
        progress=lambda event: events.append(dict(event)),
    )
    report = compare_departure_sampling_levels(
        evaluator=lambda level: DepartureSamplingEvaluation(
            level=level,
            predicted_counts=np.asarray([1.0 + 1.0 / level, 2.0]),
            journey_search_queries=level,
            preprocessing_seconds=0.01,
            classifications={"A-B-P": "normal"},
            support_ids=frozenset({f"T{index}" for index in range(level)}),
        ),
        levels=(3, 6, 12),
        observations=np.asarray([0.0, 2.0]),
        relative_change_tolerance=0.2,
        progress=lambda event: events.append(dict(event)),
    )
    assert len(report.changes) == 2
    assert report.changes[0].support_added == 3
    completed = [
        int(event.get("completed_origin_period_groups", 0))
        for event in events
        if event.get("phase") == "departure_sampling"
    ]
    assert completed == sorted(completed)
    for event in events:
        json.dumps(event)


def test_recommendations_are_advisory_and_evidence_based() -> None:
    convergence = compare_departure_sampling_levels(
        evaluator=lambda level: DepartureSamplingEvaluation(
            level=level,
            predicted_counts=np.asarray([float(level), 1.0]),
            journey_search_queries=level,
            preprocessing_seconds=0.0,
        ),
        levels=(3, 6),
        relative_change_tolerance=0.01,
    )
    report = DepartureSamplingDiagnosticsReport(
        configuration={},
        network_totals={},
        per_period=(),
        per_origin=(),
        per_destination=(),
        cells=(
            {
                "original_feasible_fraction": 0.0,
                "conditional_weight_concentration": 1.0,
                "outside_horizon_weight": 0.1,
            },
        ),
        observation_alignment={"fraction_prediction_on_zero_rows": 0.2},
        convergence=convergence,
        accounting_valid=False,
    )
    recommendations = recommend_departure_sampling_actions(report)
    codes = {item.code for item in recommendations}
    assert "stop_before_estimation" in codes
    assert "increase_sampling_resolution" in codes
    assert "exclude_or_freeze_low_feasibility" in codes
    assert "extend_timetable_horizon" in codes
    assert "aggregate_observations" in codes
    assert "investigate_response_concentration" in codes
