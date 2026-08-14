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
)
from public_transportation.domain.line import Line
from public_transportation.preprocessing.od_universe import (
    expand_candidate_od_time_cells,
    generate_candidate_od_pairs,
    generate_prior_demand,
)


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
