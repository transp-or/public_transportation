"""Hand-calculable desired-departure sampling and convergence example."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from public_transportation.preprocessing.reduced_od import (
    DepartureSampleInfeasibilityReason,
    DepartureSamplingEvaluation,
    DepartureTimeSamplingConfig,
    DesiredDepartureRoutingResult,
    JourneyAlternative,
    JourneyEvent,
    JourneyEventKind,
    JourneyTimePeriod,
    RaptorTransitLeg,
    ResponseCellKey,
    average_desired_departure_response,
    compare_departure_sampling_levels,
    generate_uniform_midpoint_samples,
)


def alternative(identifier: str, trip: str, board: int) -> JourneyAlternative:
    leg = RaptorTransitLeg(trip, "A", "B", board, board + 600)
    return JourneyAlternative(
        alternative_id=identifier * 64,
        origin_physical_stop_id="A",
        destination_physical_stop_id="B",
        origin_time_period_id="morning",
        query_departure_seconds=board - 60,
        arrival_seconds=board + 600,
        travel_seconds=660,
        wait_seconds=60,
        walk_seconds=0,
        in_vehicle_seconds=600,
        transfers=0,
        transit_legs=(leg,),
        events=(
            JourneyEvent(board, JourneyEventKind.FIRST_BOARDING, "A", "morning", 0),
            JourneyEvent(
                board + 600,
                JourneyEventKind.FINAL_ALIGHTING,
                "B",
                "morning",
                0,
            ),
        ),
        route_pattern_ids=("A-B",),
    )


def main() -> None:
    config = DepartureTimeSamplingConfig(samples_per_period=3)
    samples = generate_uniform_midpoint_samples(
        origin_physical_stop_ids=("A",),
        time_periods=(JourneyTimePeriod("morning", 8 * 3600, 9 * 3600),),
        config=config,
    )
    early = alternative("a", "T1", 8 * 3600 + 10 * 60)
    late = alternative("b", "T2", 8 * 3600 + 50 * 60)
    routing = (
        DesiredDepartureRoutingResult(samples[0], True, None, (early,), (1.0,)),
        DesiredDepartureRoutingResult(samples[1], True, None, (late,), (1.0,)),
        DesiredDepartureRoutingResult(
            samples[2],
            False,
            DepartureSampleInfeasibilityReason.NO_SERVICE_AFTER_DESIRED_TIME,
            (),
            (),
        ),
    )
    averaged = average_desired_departure_response(
        cell_key=ResponseCellKey("A", "B", "morning"),
        routing_results=routing,
        alternative_responses={
            early.alternative_id: {0: 1.0},
            late.alternative_id: {1: 1.0},
        },
        config=config,
    )
    convergence = compare_departure_sampling_levels(
        evaluator=lambda level: DepartureSamplingEvaluation(
            level=level,
            predicted_counts=np.asarray([10.0 + 1.0 / level, 10.0 - 1.0 / level]),
            journey_search_queries=level,
            preprocessing_seconds=0.0,
            response_nonzeros=2,
            operator_fingerprint=f"public-level-{level}",
        ),
        levels=(3, 6, 12),
    )
    print(
        json.dumps(
            {
                "sample_times": [item.desired_departure_seconds for item in samples],
                "original_feasible_weight": averaged.original_feasible_weight,
                "conditional_weights": averaged.conditional_sample_weights,
                "classification": averaged.classification,
                "averaged_response": asdict(averaged.averaged_response),
                "convergence": [asdict(item) for item in convergence.changes],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
