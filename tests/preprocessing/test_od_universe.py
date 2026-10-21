from __future__ import annotations

import csv
from pathlib import Path

import pytest

from public_transportation.domain import (
    Metadata,
    ODDemand,
    Scenario,
    Stop,
    StopTime,
    TimeBin,
    TimeOfDay,
    Timetable,
    Trip,
    TransferGeometry,
)
from public_transportation.domain.line import Line
from public_transportation.preprocessing.od_universe import (
    CandidateODPair,
    CandidateODUniverse,
    ScheduledFeasibilityContract,
    TimetableFeasibilityIndex,
    feasibility_station_mapping_fingerprint,
    _timetable_feasible,
    expand_candidate_od_time_cells,
    generate_candidate_od_pairs,
    generate_prior_demand,
)
from public_transportation.preprocessing import fingerprint_scenario


def _scenario() -> Scenario:
    stops = [
        Stop("A", "A", 0.0, 0.0),
        Stop("B", "B", 0.0, 0.1),
        Stop("C", "C", 0.0, 0.2),
        Stop("X", "inactive", 0.0, 0.3),
    ]
    timetable = Timetable(
        trips=[Trip("T1", "L1")],
        stop_times=[
            StopTime("T1", "A", 1, TimeOfDay(8 * 3600), TimeOfDay(8 * 3600)),
            StopTime("T1", "B", 2, TimeOfDay(8 * 3600 + 600), TimeOfDay(8 * 3600 + 600)),
            StopTime("T1", "C", 3, TimeOfDay(8 * 3600 + 1200), TimeOfDay(8 * 3600 + 1200)),
        ],
    )
    return Scenario(
        metadata=Metadata("od-universe-test"),
        stops=stops,
        lines=[Line("L1")],
        time_bins=[TimeBin("old", TimeOfDay(8 * 3600), TimeOfDay(9 * 3600))],
        demand=ODDemand([]),
        timetable=timetable,
    )


def _hub_route_scenario(
    *,
    second_platform_station: str = "H",
    second_departure_s: int = 180,
    distance_m: float = 1_000.0,
) -> Scenario:
    stops = [Stop(value, value, 0.0, 0.0) for value in ("A", "P1", "P2", "D")]
    timetable = Timetable(
        trips=[Trip("T1", "L1"), Trip("T2", "L2")],
        stop_times=[
            StopTime("T1", "A", 1, TimeOfDay(50), TimeOfDay(60)),
            StopTime("T1", "P1", 2, TimeOfDay(120), TimeOfDay(121)),
            StopTime(
                "T2",
                "P2",
                1,
                TimeOfDay(second_departure_s - 1),
                TimeOfDay(second_departure_s),
            ),
            StopTime("T2", "D", 2, TimeOfDay(240), TimeOfDay(241)),
        ],
    )
    return Scenario(
        metadata=Metadata("station-transfer"),
        stops=stops,
        lines=[Line("L1"), Line("L2")],
        time_bins=[TimeBin("t0", TimeOfDay(0), TimeOfDay(1000))],
        demand=ODDemand([]),
        timetable=timetable,
        transfer_geometry=TransferGeometry(
            {
                "A": "A",
                "P1": "H",
                "P2": second_platform_station,
                "D": "D",
            },
            {
                ("H", "P1"): distance_m,
                (second_platform_station, "P2"): distance_m,
            },
        ),
    )


def _three_change_scenario() -> Scenario:
    stops = [Stop(value, value, 0.0, 0.0) for value in ("A", "P1", "P2", "Y", "Z", "D")]
    trips = [Trip("T1", "L1"), Trip("T2", "L2"), Trip("T3", "L3"), Trip("T4", "L4")]
    events = [
        ("T1", "A", 1, 50, 60),
        ("T1", "P1", 2, 120, 121),
        ("T2", "P2", 1, 179, 180),
        ("T2", "Y", 2, 240, 241),
        ("T3", "Y", 1, 299, 300),
        ("T3", "Z", 2, 360, 361),
        ("T4", "Z", 1, 419, 420),
        ("T4", "D", 2, 480, 481),
    ]
    timetable = Timetable(
        trips=trips,
        stop_times=[
            StopTime(trip, stop, sequence, TimeOfDay(arrival), TimeOfDay(departure))
            for trip, stop, sequence, arrival, departure in events
        ],
    )
    return Scenario(
        metadata=Metadata("three-changes"),
        stops=stops,
        lines=[Line(f"L{index}") for index in range(1, 5)],
        time_bins=[TimeBin("t0", TimeOfDay(0), TimeOfDay(1000))],
        demand=ODDemand([]),
        timetable=timetable,
        transfer_geometry=TransferGeometry(
            {"A": "A", "P1": "H", "P2": "H", "Y": "Y", "Z": "Z", "D": "D"},
            {("H", "P1"): 0.0, ("H", "P2"): 0.0},
        ),
    )


def test_pair_file_is_time_independent_and_duplicates_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "od_pairs.csv"
    path.write_text(
        "origin_stop_id,destination_stop_id\nA,C\n",
        encoding="utf-8",
    )
    universe = generate_candidate_od_pairs(
        _scenario(),
        source="file",
        od_pairs_path=path,
        active_service_only=False,
        connectivity_policy="none",
    )
    assert universe.pairs[0].tuple == ("A", "C")
    assert "time_bin" not in path.read_text(encoding="utf-8")

    path.write_text(
        "origin_stop_id,destination_stop_id,time_bin_id\nA,C,morning\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pair-only"):
        generate_candidate_od_pairs(
            _scenario(),
            source="file",
            od_pairs_path=path,
            active_service_only=False,
            connectivity_policy="none",
        )
    path.write_text(
        "origin_stop_id,destination_stop_id\nA,C\nA,C\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        generate_candidate_od_pairs(
            _scenario(),
            source="file",
            od_pairs_path=path,
            active_service_only=False,
            connectivity_policy="none",
        )


def test_station_transfer_feasibility_counts_line_changes_and_ignores_walk_time():
    scenario = _hub_route_scenario(second_departure_s=150, distance_m=10_000.0)
    index = TimetableFeasibilityIndex.from_scenario(scenario)
    pair = CandidateODPair("A", "D")
    contract = ScheduledFeasibilityContract(
        maximum_transfers=1,
        maximum_initial_wait_seconds=300,
        maximum_journey_seconds=600,
        maximum_waiting_seconds=60,
        station_mapping_fingerprint=feasibility_station_mapping_fingerprint(scenario),
    )
    assert contract.is_feasible(index, pair, ("p", 0, 500))
    assert _timetable_feasible(
        scenario,
        pair,
        ("p", 0, 500),
        mapping={"A": "A", "P1": "P1", "P2": "P2", "D": "D"},
        maximum_transfers=1,
        maximum_initial_wait_seconds=300,
        maximum_journey_seconds=600,
        maximum_waiting_seconds=60,
    )
    assert contract.fingerprint


def test_maximum_transfers_applies_to_line_changes_not_trip_ids():
    scenario = _three_change_scenario()
    index = TimetableFeasibilityIndex.from_scenario(scenario)
    pair = CandidateODPair("A", "D")
    period = ("p", 0, 1000)
    for limit in (0, 1, 2):
        assert not index.is_feasible(
            pair,
            period,
            maximum_transfers=limit,
            maximum_initial_wait_seconds=100,
            maximum_waiting_seconds=60,
            maximum_journey_seconds=600,
        )
    assert index.is_feasible(
        pair,
        period,
        maximum_transfers=3,
        maximum_initial_wait_seconds=100,
        maximum_waiting_seconds=60,
        maximum_journey_seconds=600,
    )

    same_line = Scenario(
        metadata=Metadata("same-line-reboarding"),
        stops=[Stop(value, value, 0.0, 0.0) for value in ("A", "B", "C")],
        lines=[Line("L1")],
        time_bins=[TimeBin("t0", TimeOfDay(0), TimeOfDay(1000))],
        demand=ODDemand([]),
        timetable=Timetable(
            trips=[Trip("T1", "L1"), Trip("T2", "L1")],
            stop_times=[
                StopTime("T1", "A", 1, TimeOfDay(50), TimeOfDay(60)),
                StopTime("T1", "B", 2, TimeOfDay(120), TimeOfDay(121)),
                StopTime("T2", "B", 1, TimeOfDay(179), TimeOfDay(180)),
                StopTime("T2", "C", 2, TimeOfDay(240), TimeOfDay(241)),
            ],
        ),
    )
    same_line_index = TimetableFeasibilityIndex.from_scenario(same_line)
    assert not same_line_index.is_feasible(
        CandidateODPair("A", "C"),
        period,
        maximum_transfers=2,
        maximum_initial_wait_seconds=100,
        maximum_waiting_seconds=60,
        maximum_journey_seconds=600,
    )


def test_feasibility_mapping_identity_changes_only_affected_expansion(tmp_path: Path):
    original = _hub_route_scenario(second_departure_s=150, distance_m=10.0)
    changed_distance = _hub_route_scenario(second_departure_s=150, distance_m=500.0)
    changed_station = _hub_route_scenario(
        second_platform_station="OTHER", second_departure_s=150, distance_m=10.0
    )
    assert feasibility_station_mapping_fingerprint(original) == (
        feasibility_station_mapping_fingerprint(changed_distance)
    )
    assert feasibility_station_mapping_fingerprint(original) != (
        feasibility_station_mapping_fingerprint(changed_station)
    )
    # The base scenario-data fingerprint intentionally excludes assignment-only
    # transfer geometry, preserving unaffected L0/L1 artifacts.
    assert fingerprint_scenario(original) == fingerprint_scenario(changed_station)

    universe = CandidateODUniverse(
        pairs=(CandidateODPair("A", "D"),),
        exclusions=(),
        source="file",
        level="stop",
        include_same_stop=False,
        active_service_only=False,
        connectivity_policy="none",
        physical_stop_mapping={
            stop.stop_id: stop.stop_id for stop in original.stops
        },
        generator_fingerprint="fixture",
    )
    bins = [("p", 0, 500)]
    original_expansion = expand_candidate_od_time_cells(
        universe, bins, scenario=original
    )
    distance_expansion = expand_candidate_od_time_cells(
        universe, bins, scenario=changed_distance
    )
    station_expansion = expand_candidate_od_time_cells(
        universe, bins, scenario=changed_station
    )
    assert original_expansion.fingerprint == distance_expansion.fingerprint
    assert original_expansion.fingerprint != station_expansion.fingerprint


def test_feasibility_contract_rejects_a_different_station_mapping():
    scenario = _hub_route_scenario()
    index = TimetableFeasibilityIndex.from_scenario(scenario)
    contract = ScheduledFeasibilityContract(
        station_mapping_fingerprint="different-map"
    )
    with pytest.raises(ValueError, match="different platform-to-station mappings"):
        contract.path_metrics(index, origin="A", period=("p", 0, 500))


def test_network_generation_is_deterministic_and_filters_activity_and_connectivity() -> None:
    first = generate_candidate_od_pairs(
        _scenario(),
        source="network_ordered_pairs",
        active_service_only=True,
        connectivity_policy="directed_reachable",
    )
    second = generate_candidate_od_pairs(
        _scenario(),
        source="network_ordered_pairs",
        active_service_only=True,
        connectivity_policy="directed_reachable",
    )
    assert first.pairs == second.pairs
    assert first.fingerprint == second.fingerprint
    assert first.pairs == tuple(
        pair for pair in first.pairs if pair.tuple in {("A", "B"), ("A", "C"), ("B", "C")}
    )
    assert first.audit["exclusion_counts"].get("inactive_origin", 0) > 0
    assert first.audit["exclusion_counts"].get("static_unreachable", 0) > 0


def test_same_node_filtering_and_physical_level_are_explicit() -> None:
    universe = generate_candidate_od_pairs(
        _scenario(),
        source="network_ordered_pairs",
        level="physical_stop",
        include_same_stop=False,
        active_service_only=False,
        connectivity_policy="none",
        physical_stop_mapping={"A": "P", "B": "P", "C": "Q", "X": "X"},
    )
    assert all(pair.origin_stop_id != pair.destination_stop_id for pair in universe.pairs)
    assert universe.level == "physical_stop"
    assert universe.audit["exclusion_counts"]["same_node"] == 3


def test_physical_level_pair_file_uses_physical_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "physical_od_pairs.csv"
    path.write_text(
        "origin_stop_id,destination_stop_id\nP,Q\n",
        encoding="utf-8",
    )
    universe = generate_candidate_od_pairs(
        _scenario(),
        source="file",
        level="physical_stop",
        od_pairs_path=path,
        physical_stop_mapping={"A": "P", "B": "P", "C": "Q", "X": "X"},
        active_service_only=False,
        connectivity_policy="none",
    )
    assert tuple(pair.tuple for pair in universe.pairs) == (("P", "Q"),)


def test_time_expansion_depends_on_bins_but_pair_fingerprint_does_not() -> None:
    scenario = _scenario()
    universe = generate_candidate_od_pairs(
        scenario,
        source="network_ordered_pairs",
        active_service_only=True,
        connectivity_policy="directed_reachable",
    )
    morning = expand_candidate_od_time_cells(
        universe, [("morning", 8 * 3600, 9 * 3600)], scenario=scenario
    )
    later = expand_candidate_od_time_cells(
        universe, [("later", 9 * 3600, 10 * 3600)], scenario=scenario
    )
    assert universe.fingerprint == generate_candidate_od_pairs(
        scenario,
        source="network_ordered_pairs",
        active_service_only=True,
        connectivity_policy="directed_reachable",
    ).fingerprint
    assert morning.fingerprint != later.fingerprint
    assert morning.cell_count > later.cell_count
    assert any(item.reason == "timetable_infeasible" for item in later.exclusions)
    assert morning.audit["input_pair_count"] == len(universe.pairs)
    assert (
        sum(morning.audit["exclusion_counts"].values()) + morning.audit["retained_cell_count"]
        == morning.audit["expanded_od_time_count"]
    )


def test_origin_period_feasibility_cache_matches_pairwise_search() -> None:
    scenario = _scenario()
    universe = generate_candidate_od_pairs(
        scenario,
        source="network_ordered_pairs",
        active_service_only=True,
        connectivity_policy="directed_reachable",
    )
    bins = [("morning", 8 * 3600, 9 * 3600), ("late", 9 * 3600, 10 * 3600)]
    indexed = expand_candidate_od_time_cells(
        universe,
        bins,
        scenario=scenario,
        feasibility_index=TimetableFeasibilityIndex.from_scenario(
            scenario, physical_stop_mapping=universe.physical_stop_mapping
        ),
    )

    def pairwise(pair, period):
        return _timetable_feasible(
            scenario,
            pair,
            period,
            mapping=universe.physical_stop_mapping,
            maximum_transfers=2,
            maximum_initial_wait_seconds=3600,
            maximum_journey_seconds=7200,
            maximum_waiting_seconds=3600,
        )

    reference = expand_candidate_od_time_cells(
        universe,
        bins,
        scenario=scenario,
        timetable_feasibility=pairwise,
    )
    assert indexed.cells == reference.cells
    assert indexed.exclusions == reference.exclusions
    assert indexed.fingerprint == reference.fingerprint


def test_in_memory_time_expansion_reports_bounded_eta_progress() -> None:
    scenario = _scenario()
    universe = generate_candidate_od_pairs(
        scenario,
        source="network_ordered_pairs",
        active_service_only=True,
        connectivity_policy="directed_reachable",
    )
    events: list[dict[str, object]] = []
    expansion = expand_candidate_od_time_cells(
        universe,
        [("morning", 8 * 3600, 9 * 3600), ("late", 9 * 3600, 10 * 3600)],
        scenario=scenario,
        progress=events.append,
    )

    cell_events = [
        event for event in events if event.get("phase") == "expand_od_time_cells"
    ]
    assert cell_events
    assert cell_events[0]["completed_units"] == 0
    assert cell_events[-1]["completed_units"] == cell_events[-1]["total_units"]
    assert cell_events[-1]["predicted_remaining_seconds"] == pytest.approx(0.0)
    assert any(
        event["completed_units"] > 0
        and event["predicted_remaining_seconds"] is not None
        for event in cell_events
    )
    assert expansion.cell_count + len(expansion.exclusions) == cell_events[-1]["total_units"]


def test_all_ones_prior_is_neutral_and_external_pair_prior_repeats_per_bin(tmp_path: Path) -> None:
    universe = generate_candidate_od_pairs(
        _scenario(),
        source="network_ordered_pairs",
        active_service_only=True,
        connectivity_policy="directed_reachable",
    )
    expansion = expand_candidate_od_time_cells(
        universe,
        [("morning", 8 * 3600, 9 * 3600), ("late", 9 * 3600, 10 * 3600)],
        scenario=_scenario(),
        timetable_policy="defer",
    )
    neutral = generate_prior_demand(expansion, source="all_ones", semantics="neutral_seed")
    assert set(neutral.values.values()) == {1.0}
    assert neutral.audit["semantics"] == "neutral_seed"

    path = tmp_path / "prior_demand.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("origin_stop_id", "destination_stop_id", "prior_value"))
        for pair in universe.pairs:
            writer.writerow((*pair.tuple, 3.0))
    external = generate_prior_demand(
        expansion,
        source="external_file",
        semantics="external_prior",
        prior_file=path,
    )
    assert set(external.values.values()) == {3.0}
