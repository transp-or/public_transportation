from __future__ import annotations

from public_transportation.domain.demand import ODDemand, ODRecord
from public_transportation.domain.issues import Severity
from public_transportation.domain.metadata import Metadata
from public_transportation.domain.scenario import Scenario
from public_transportation.domain.stop import Stop
from public_transportation.domain.time_bin import TimeBin
from public_transportation.domain.time_of_day import TimeOfDay


def _codes(rep):
    return {iss.code for iss in rep.issues}


def _find(rep, code: str):
    return [iss for iss in rep.issues if iss.code == code]


def _mk_stop(stop_id: str, *, name: str = "X", lat: float = 46.5, lon: float = 6.6) -> Stop:
    return Stop(stop_id=stop_id, name=name, lat=lat, lon=lon)


def _mk_bin(bin_id: str, start_s: int, end_s: int) -> TimeBin:
    return TimeBin(
        bin_id=bin_id,
        start=TimeOfDay(seconds_from_midnight=start_s),
        end=TimeOfDay(seconds_from_midnight=end_s),
    )


def _mk_scenario(
    *,
    stops: list[Stop] | None = None,
    time_bins: list[TimeBin] | None = None,
    demand: ODDemand | None = None,
) -> Scenario:
    return Scenario(
        metadata=Metadata(title="Test scenario"),
        stops=stops if stops is not None else [],
        time_bins=time_bins if time_bins is not None else [],
        demand=demand if demand is not None else ODDemand(records=[]),
        timetable=None,
    )


def test_empty_scenario_is_valid():
    sc = _mk_scenario()
    rep = sc.validate()
    # No local validation errors, no duplicates, no missing references because demand is empty.
    assert rep.issues == []


def test_valid_scenario_has_no_issues():
    sc = _mk_scenario(
        stops=[_mk_stop("A"), _mk_stop("B")],
        time_bins=[_mk_bin("T1", 8 * 3600, 8 * 3600 + 900)],
        demand=ODDemand(records=[
            ODRecord(origin_stop_id="A", dest_stop_id="B", time_bin_id="T1", flow=10.0),
        ]),
    )

    rep = sc.validate()
    assert rep.issues == []


def test_duplicate_stop_ids_reported_as_error():
    sc = _mk_scenario(
        stops=[_mk_stop("A"), _mk_stop("A")],
        time_bins=[_mk_bin("T1", 0, 60)],
        demand=ODDemand(records=[]),
    )

    rep = sc.validate()
    issues = _find(rep, "DUPLICATE_ID")
    assert len(issues) >= 1
    # Ensure at least one duplicate refers to stops
    assert any(iss.severity == Severity.ERROR and iss.location == "stops" for iss in issues)
    assert any(iss.context is not None and iss.context.get("id") == "A" for iss in issues)


def test_duplicate_time_bin_ids_reported_as_error():
    sc = _mk_scenario(
        stops=[_mk_stop("A"), _mk_stop("B")],
        time_bins=[_mk_bin("T1", 0, 60), _mk_bin("T1", 60, 120)],
        demand=ODDemand(records=[]),
    )

    rep = sc.validate()
    issues = _find(rep, "DUPLICATE_ID")
    assert len(issues) >= 1
    assert any(iss.severity == Severity.ERROR and iss.location == "time_bins" for iss in issues)
    assert any(iss.context is not None and iss.context.get("id") == "T1" for iss in issues)


def test_demand_unknown_origin_is_error():
    sc = _mk_scenario(
        stops=[_mk_stop("A"), _mk_stop("B")],
        time_bins=[_mk_bin("T1", 0, 60)],
        demand=ODDemand(records=[
            ODRecord(origin_stop_id="X", dest_stop_id="B", time_bin_id="T1", flow=1.0),
        ]),
    )

    rep = sc.validate()
    issues = _find(rep, "DEMAND_ORIGIN_UNKNOWN")
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR
    assert issues[0].location == "demand.records[0]"


def test_demand_unknown_dest_is_error():
    sc = _mk_scenario(
        stops=[_mk_stop("A"), _mk_stop("B")],
        time_bins=[_mk_bin("T1", 0, 60)],
        demand=ODDemand(records=[
            ODRecord(origin_stop_id="A", dest_stop_id="Y", time_bin_id="T1", flow=1.0),
        ]),
    )

    rep = sc.validate()
    issues = _find(rep, "DEMAND_DEST_UNKNOWN")
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR
    assert issues[0].location == "demand.records[0]"


def test_demand_unknown_time_bin_is_error():
    sc = _mk_scenario(
        stops=[_mk_stop("A"), _mk_stop("B")],
        time_bins=[_mk_bin("T1", 0, 60)],
        demand=ODDemand(records=[
            ODRecord(origin_stop_id="A", dest_stop_id="B", time_bin_id="T_X", flow=1.0),
        ]),
    )

    rep = sc.validate()
    issues = _find(rep, "DEMAND_TIMEBIN_UNKNOWN")
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR
    assert issues[0].location == "demand.records[0]"


def test_multiple_reference_errors_all_reported():
    sc = _mk_scenario(
        stops=[_mk_stop("A")],
        time_bins=[_mk_bin("T1", 0, 60)],
        demand=ODDemand(records=[
            ODRecord(origin_stop_id="X", dest_stop_id="Y", time_bin_id="T_X", flow=1.0),
        ]),
    )

    rep = sc.validate()
    codes = _codes(rep)

    assert "DEMAND_ORIGIN_UNKNOWN" in codes
    assert "DEMAND_DEST_UNKNOWN" in codes
    assert "DEMAND_TIMEBIN_UNKNOWN" in codes
    # Three distinct issues expected from _validate_references()
    assert len(_find(rep, "DEMAND_ORIGIN_UNKNOWN")) == 1
    assert len(_find(rep, "DEMAND_DEST_UNKNOWN")) == 1
    assert len(_find(rep, "DEMAND_TIMEBIN_UNKNOWN")) == 1