# tests/domain/test_demand.py
from __future__ import annotations

from public_transportation.domain.demand import ODRecord, ODDemand
from public_transportation.domain.issues import Severity


def _codes(report):
    return {iss.code for iss in report.issues}


def _find_issues(report, code: str):
    return [iss for iss in report.issues if iss.code == code]


# ---------------------------------------------------------
# Valid demand
# ---------------------------------------------------------


def test_valid_demand_has_no_issues():
    d = ODDemand(records=[
        ODRecord(origin_stop_id="A", dest_stop_id="B", time_bin_id="T1", flow=0.0),
        ODRecord(origin_stop_id="A", dest_stop_id="C", time_bin_id="T1", flow=12.5),
        ODRecord(origin_stop_id="B", dest_stop_id="C", time_bin_id="T2", flow=3.0),
    ])

    rep = d.validate()

    assert rep.issues == []


def test_empty_records_is_valid():
    d = ODDemand(records=[])

    rep = d.validate()

    assert rep.issues == []


# ---------------------------------------------------------
# OD ids
# ---------------------------------------------------------


def test_empty_origin_or_destination_is_error():
    d = ODDemand(records=[
        ODRecord(origin_stop_id="", dest_stop_id="B", time_bin_id="T1", flow=1.0),
        ODRecord(origin_stop_id="A", dest_stop_id="", time_bin_id="T1", flow=1.0),
        ODRecord(origin_stop_id="", dest_stop_id="", time_bin_id="T1", flow=1.0),
    ])

    rep = d.validate()
    issues = _find_issues(rep, "DEMAND_OD_EMPTY")

    assert len(issues) == 3
    assert all(iss.severity == Severity.ERROR for iss in issues)
    assert all(iss.location.startswith("demand.records[") for iss in issues)


# ---------------------------------------------------------
# time_bin_id
# ---------------------------------------------------------


def test_empty_time_bin_id_is_error():
    d = ODDemand(records=[
        ODRecord(origin_stop_id="A", dest_stop_id="B", time_bin_id="", flow=1.0),
        ODRecord(origin_stop_id="A", dest_stop_id="C", time_bin_id="", flow=2.0),
    ])

    rep = d.validate()
    issues = _find_issues(rep, "DEMAND_TIMEBIN_EMPTY")

    assert len(issues) == 2
    assert all(iss.severity == Severity.ERROR for iss in issues)


# ---------------------------------------------------------
# flow
# ---------------------------------------------------------


def test_negative_flow_is_error_and_context_contains_flow():
    d = ODDemand(records=[
        ODRecord(origin_stop_id="A", dest_stop_id="B", time_bin_id="T1", flow=-0.1),
    ])

    rep = d.validate()
    issues = _find_issues(rep, "DEMAND_FLOW_NEGATIVE")

    assert len(issues) == 1
    iss = issues[0]
    assert iss.severity == Severity.ERROR
    assert iss.location.endswith(".flow")
    assert iss.context is not None
    assert iss.context["flow"] == -0.1


# ---------------------------------------------------------
# Multiple issues aggregation
# ---------------------------------------------------------


def test_multiple_issues_in_single_record_are_all_reported():
    d = ODDemand(records=[
        ODRecord(origin_stop_id="", dest_stop_id="", time_bin_id="", flow=-3.0),
    ])

    rep = d.validate()
    codes = _codes(rep)

    assert "DEMAND_OD_EMPTY" in codes
    assert "DEMAND_TIMEBIN_EMPTY" in codes
    assert "DEMAND_FLOW_NEGATIVE" in codes
    assert len(rep.issues) == 3


def test_multiple_records_accumulate_issues_with_correct_locations():
    d = ODDemand(records=[
        ODRecord(origin_stop_id="", dest_stop_id="B", time_bin_id="T1", flow=1.0),     # OD empty
        ODRecord(origin_stop_id="A", dest_stop_id="C", time_bin_id="", flow=1.0),     # timebin empty
        ODRecord(origin_stop_id="A", dest_stop_id="D", time_bin_id="T2", flow=-1.0),  # flow negative
    ])

    rep = d.validate()

    # Exactly one of each.
    assert len(_find_issues(rep, "DEMAND_OD_EMPTY")) == 1
    assert len(_find_issues(rep, "DEMAND_TIMEBIN_EMPTY")) == 1
    assert len(_find_issues(rep, "DEMAND_FLOW_NEGATIVE")) == 1

    # Locations reflect record index.
    od_iss = _find_issues(rep, "DEMAND_OD_EMPTY")[0]
    tb_iss = _find_issues(rep, "DEMAND_TIMEBIN_EMPTY")[0]
    fl_iss = _find_issues(rep, "DEMAND_FLOW_NEGATIVE")[0]

    assert od_iss.location == "demand.records[0]"
    assert tb_iss.location == "demand.records[1]"
    assert fl_iss.location == "demand.records[2].flow"