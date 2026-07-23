from __future__ import annotations

from types import SimpleNamespace

import pytest

from public_transportation.domain.fixed_demand import (
    FixedODDemand,
    FixedODRecord,
    read_fixed_demand_csv,
)


def _scenario():
    return SimpleNamespace(
        stops=[
            SimpleNamespace(stop_id="A"),
            SimpleNamespace(stop_id="B"),
            SimpleNamespace(stop_id="C"),
        ],
        time_bins=[
            SimpleNamespace(bin_id="t0"),
            SimpleNamespace(bin_id="t1"),
        ],
        demand=SimpleNamespace(
            records=[
                SimpleNamespace(origin_stop_id="A", dest_stop_id="B", time_bin_id="t0"),
                SimpleNamespace(origin_stop_id="A", dest_stop_id="B", time_bin_id="t1"),
                SimpleNamespace(origin_stop_id="A", dest_stop_id="C", time_bin_id="t0"),
                SimpleNamespace(origin_stop_id="B", dest_stop_id="C", time_bin_id="t0"),
            ]
        ),
    )


def _write(tmp_path, text: str):
    path = tmp_path / "fixed_demand.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_zero_and_positive_fixed_values(tmp_path):
    path = _write(
        tmp_path,
        "origin_stop_id,dest_stop_id,time_bin_id,fixed_flow\n"
        "A,B,t0,0\n"
        "A,C,t0,12.5\n",
    )

    result = read_fixed_demand_csv(path, scenario=_scenario())

    assert isinstance(result, FixedODDemand)
    assert result.records == (
        FixedODRecord("A", "B", "t0", 0.0),
        FixedODRecord("A", "C", "t0", 12.5),
    )
    assert result.as_dict() == {
        ("A", "B", "t0"): 0.0,
        ("A", "C", "t0"): 12.5,
    }


@pytest.mark.parametrize(
    "text",
    [
        "origin_stop_id,dest_stop_id,time_bin_id,fixed_flow\nA,B,t0,\n",
        "origin_stop_id,dest_stop_id,time_bin_id\nA,B,t0\n",
    ],
)
def test_missing_or_blank_fixed_flow_defaults_to_zero(tmp_path, text):
    result = read_fixed_demand_csv(_write(tmp_path, text), scenario=_scenario())
    assert result.records == (FixedODRecord("A", "B", "t0", 0.0),)


def test_header_only_file_returns_empty_collection(tmp_path):
    path = _write(tmp_path, "origin_stop_id,dest_stop_id,time_bin_id,fixed_flow\n")
    result = read_fixed_demand_csv(path, scenario=_scenario())
    assert result.records == ()
    assert len(result) == 0
    assert result.as_dict() == {}


def test_output_order_is_canonical_and_independent_of_csv_order(tmp_path):
    path = _write(
        tmp_path,
        "origin_stop_id,dest_stop_id,time_bin_id,fixed_flow\n"
        "B,C,t0,3\n"
        "A,C,t0,2\n"
        "A,B,t1,1\n"
        "A,B,t0,0\n",
    )
    result = read_fixed_demand_csv(path, scenario=_scenario())
    assert [record.key for record in result.records] == [
        ("A", "B", "t0"),
        ("A", "B", "t1"),
        ("A", "C", "t0"),
        ("B", "C", "t0"),
    ]


def test_same_od_in_different_time_bins_are_distinct_keys(tmp_path):
    path = _write(
        tmp_path,
        "origin_stop_id,dest_stop_id,time_bin_id,fixed_flow\n"
        "A,B,t0,1\n"
        "A,B,t1,2\n",
    )
    result = read_fixed_demand_csv(path, scenario=_scenario())
    assert result.as_dict() == {("A", "B", "t0"): 1.0, ("A", "B", "t1"): 2.0}


def test_identifiers_and_headers_are_stripped(tmp_path):
    path = _write(
        tmp_path,
        " origin_stop_id , dest_stop_id , time_bin_id , fixed_flow \n"
        " A , B , t0 , 4 \n",
    )
    result = read_fixed_demand_csv(path, scenario=_scenario())
    assert result.records == (FixedODRecord("A", "B", "t0", 4.0),)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "must contain a header"),
        (
            "origin_stop_id,dest_stop_id,fixed_flow\nA,B,0\n",
            "missing required columns.*time_bin_id",
        ),
        (
            "origin_stop_id,dest_stop_id,time_bin_id,flow\nA,B,t0,0\n",
            "unexpected columns.*flow",
        ),
        (
            "origin_stop_id,dest_stop_id,time_bin_id,time_bin_id\nA,B,t0,t0\n",
            "duplicate column names",
        ),
        (
            "origin_stop_id,dest_stop_id,time_bin_id\n,B,t0\n",
            "origin_stop_id must be a non-empty identifier",
        ),
        (
            "origin_stop_id,dest_stop_id,time_bin_id\nA,,t0\n",
            "dest_stop_id must be a non-empty identifier",
        ),
        (
            "origin_stop_id,dest_stop_id,time_bin_id\nA,B,\n",
            "time_bin_id must be a non-empty identifier",
        ),
        (
            "origin_stop_id,dest_stop_id,time_bin_id,fixed_flow\nA,B,t0,nope\n",
            "fixed_flow must be numeric",
        ),
        (
            "origin_stop_id,dest_stop_id,time_bin_id,fixed_flow\nA,B,t0,-1\n",
            "fixed_flow must be non-negative",
        ),
        (
            "origin_stop_id,dest_stop_id,time_bin_id,fixed_flow\nA,B,t0,nan\n",
            "fixed_flow must be finite",
        ),
        (
            "origin_stop_id,dest_stop_id,time_bin_id,fixed_flow\nA,B,t0,inf\n",
            "fixed_flow must be finite",
        ),
        (
            "origin_stop_id,dest_stop_id,time_bin_id\nA,B,t0,extra\n",
            "more values than the header",
        ),
    ],
)
def test_rejects_malformed_files(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        read_fixed_demand_csv(_write(tmp_path, text), scenario=_scenario())


def test_rejects_duplicate_logical_key_and_reports_first_row(tmp_path):
    path = _write(
        tmp_path,
        "origin_stop_id,dest_stop_id,time_bin_id,fixed_flow\n"
        "A,B,t0,1\n"
        " A , B , t0 ,2\n",
    )
    with pytest.raises(ValueError, match="duplicate OD/time-bin key.*first defined on row 2"):
        read_fixed_demand_csv(path, scenario=_scenario())


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("X,B,t0,0", "unknown origin_stop_id 'X'"),
        ("A,X,t0,0", "unknown dest_stop_id 'X'"),
        ("A,B,tx,0", "unknown time_bin_id 'tx'"),
        (
            "C,A,t0,0",
            "OD/time-bin key.*is not present in scenario.demand.records",
        ),
    ],
)
def test_rejects_keys_not_defined_by_scenario(tmp_path, row, message):
    path = _write(
        tmp_path,
        "origin_stop_id,dest_stop_id,time_bin_id,fixed_flow\n" + row + "\n",
    )
    with pytest.raises(ValueError, match=message):
        read_fixed_demand_csv(path, scenario=_scenario())


def test_accepts_utf8_bom(tmp_path):
    path = _write(
        tmp_path,
        "\ufefforigin_stop_id,dest_stop_id,time_bin_id,fixed_flow\nA,B,t0,1\n",
    )
    result = read_fixed_demand_csv(path, scenario=_scenario())
    assert result.records == (FixedODRecord("A", "B", "t0", 1.0),)
