from __future__ import annotations

import json

import pytest

from public_transportation.inference.gravity.section_usage import (
    SECTION_USAGE_COLUMNS,
    SectionUsageStorageBudgetExceeded,
    aggregate_line_usage_rows,
    aggregate_section_usage_rows,
    iter_section_usage_rows,
    read_section_usage_manifest,
    write_line_usage_dataset,
    write_section_usage_dataset,
)


def _rows():
    return (
        {"departure_time_bin": "am", "section_id": "S1", "origin_stop_id": "O",
         "destination_stop_id": "D", "line_id": "L", "trip_id": "T1",
         "assigned_contribution": 1.0},
        {"departure_time_bin": "am", "section_id": "S1", "origin_stop_id": "O",
         "destination_stop_id": "D", "line_id": "L", "trip_id": "T1",
         "assigned_contribution": 2.0},
        {"departure_time_bin": "am", "section_id": "S1", "origin_stop_id": "O",
         "destination_stop_id": "D", "line_id": "L", "trip_id": "T2",
         "assigned_contribution": 4.0},
    )


def test_exact_section_and_line_aggregation():
    compact = aggregate_section_usage_rows(_rows())
    assert len(compact) == 2
    assert sum(row["assigned_contribution"] for row in compact) == pytest.approx(7.0)
    line = aggregate_line_usage_rows(compact)
    assert len(line) == 1
    assert line[0]["trip_id"] == ""
    assert line[0]["assigned_contribution"] == pytest.approx(7.0)


def test_partition_manifest_and_bounded_reader(tmp_path):
    dataset = tmp_path / "dataset"
    manifest = write_section_usage_dataset(_rows(), dataset, format="csv", shard_rows=1)
    assert tuple(manifest["columns"]) == SECTION_USAGE_COLUMNS
    assert manifest["rows_written"] == 2
    assert read_section_usage_manifest(dataset)["status"] == "completed"
    batches = list(iter_section_usage_rows(dataset, section_id="S1", batch_size=1))
    assert [len(batch) for batch in batches] == [1, 1]
    assert sum(row["assigned_contribution"] for batch in batches for row in batch) == pytest.approx(7.0)


def test_line_dataset_is_an_exact_separate_aggregate(tmp_path):
    dataset = tmp_path / "line-dataset"
    manifest = write_line_usage_dataset(_rows(), dataset, format="csv")
    assert manifest["grain"] == "section_usage_line"
    rows = list(iter_section_usage_rows(dataset, section_id="S1"))[0]
    assert rows[0]["trip_id"] == ""
    assert rows[0]["assigned_contribution"] == pytest.approx(7.0)


def test_storage_budget_stops_at_durable_shard_boundary(tmp_path):
    with pytest.raises(SectionUsageStorageBudgetExceeded):
        write_section_usage_dataset(_rows(), tmp_path / "budget", format="csv", storage_budget_bytes=1)
    partial = json.loads((tmp_path / "budget" / "manifest.json").read_text(encoding="utf-8"))
    assert partial["status"] == "storage_budget_exceeded"
