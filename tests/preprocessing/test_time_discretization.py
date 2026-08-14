from __future__ import annotations

import json

import pytest

from public_transportation.domain import TimeOfDay
from public_transportation.measurement.io import write_measurements_csv
from public_transportation.measurement.schema import (
    MeasurementRecord,
    MeasurementTable,
    MeasurementType,
)
from public_transportation.preprocessing.time_discretization import (
    TimeDiscretizationConfig,
    recommend_time_discretization,
    recommend_time_discretization_from_csv,
)
from public_transportation.preprocessing.materialize_time_bins import materialize_time_bins


def _profile_records() -> tuple[MeasurementRecord, ...]:
    records = []
    for minute in range(6 * 60, 9 * 60, 5):
        value = 12.0 if 7 * 60 + 30 <= minute < 8 * 60 + 15 else 1.0
        records.append(
            MeasurementRecord(
                method_id="synthetic",
                measurement_type=MeasurementType.BOARDING,
                stop_id="S1",
                time=TimeOfDay.from_hms(minute // 60, minute % 60),
                value=value,
                trip_id=f"T{minute}",
            )
        )
    return tuple(records)


def test_recommendation_detects_peak_and_reports_complexity() -> None:
    report = recommend_time_discretization(
        _profile_records(),
        TimeDiscretizationConfig(
            base_resolution_minutes=5,
            min_bin_minutes=10,
            max_bin_minutes=60,
            max_bins=24,
            num_od_pairs=100,
            max_od_cells=1_000,
        ),
    )

    assert report["schema_version"] == 1
    assert report["peak_intervals"]
    recommendation = report["recommendation"]
    assert recommendation["valid"] is True
    assert recommendation["estimated_od_cells"] == 100 * recommendation["num_bins"]
    assert recommendation["edges"][0]["time"] == "06:00:00"
    assert recommendation["edges"][-1]["time"] == "09:00:00"
    assert recommendation["time_bins"][0]["bin_id"] == "t0"
    assert recommendation["time_bins"][-1]["end"] == "09:00:00"


def test_recommendation_is_deterministic_and_respects_event_floor() -> None:
    config = TimeDiscretizationConfig(
        base_resolution_minutes=5,
        min_bin_minutes=10,
        max_bin_minutes=60,
        max_bins=24,
        min_events_per_bin=0.0,
    )
    first = recommend_time_discretization(_profile_records(), config)
    second = recommend_time_discretization(tuple(reversed(_profile_records())), config)
    assert first == second
    assert all(
        value >= config.min_events_per_bin
        for value in first["recommendation"]["events_per_bin"]
    )


def test_csv_entry_point_writes_json_compatible_report(tmp_path) -> None:
    path = tmp_path / "measurements.csv"
    write_measurements_csv(MeasurementTable.from_records(_profile_records()), path)
    report = recommend_time_discretization_from_csv(path)
    encoded = json.dumps(report)
    assert json.loads(encoded)["recommendation"]["num_bins"] > 0


def test_invalid_counts_and_configuration_are_rejected() -> None:
    invalid = MeasurementRecord(
        method_id="synthetic",
        measurement_type=MeasurementType.BOARDING,
        stop_id="S1",
        time=TimeOfDay.from_hms(7, 0),
        value=-1.0,
        trip_id="T1",
    )
    with pytest.raises(ValueError, match="non-negative"):
        recommend_time_discretization((invalid,))
    with pytest.raises(ValueError, match="max_bin_minutes"):
        TimeDiscretizationConfig(min_bin_minutes=30, max_bin_minutes=10)
    with pytest.raises(ValueError, match="max_od_cells"):
        TimeDiscretizationConfig(num_od_pairs=100, max_od_cells=50)


def test_materialize_reviewed_recommendation_as_canonical_csv(tmp_path) -> None:
    report = recommend_time_discretization(_profile_records())
    report_path = tmp_path / "recommendation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output_path = tmp_path / "time_bins.csv"

    bins = materialize_time_bins(report_path, output_path)

    assert bins == [
        {key: item[key] for key in ("bin_id", "start_s", "end_s")}
        for item in report["recommendation"]["time_bins"]
    ]
    assert output_path.read_text(encoding="utf-8").splitlines()[0] == "bin_id,start_s,end_s"
    assert output_path.read_text(encoding="utf-8").splitlines()[1].startswith("t0,")
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_time_bins(report_path, output_path)


def test_materialize_rejects_invalid_or_noncontiguous_candidate(tmp_path) -> None:
    report_path = tmp_path / "recommendation.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recommendation": {
                    "valid": False,
                    "invalid_reasons": ["exceeds_max_od_cells"],
                    "time_bins": [],
                },
                "candidates": [
                    {
                        "name": "broken",
                        "valid": True,
                        "time_bins": [
                            {"bin_id": "t0", "start_s": 0, "end_s": 60},
                            {"bin_id": "t1", "start_s": 120, "end_s": 180},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid"):
        materialize_time_bins(report_path, tmp_path / "time_bins.csv")
    with pytest.raises(ValueError, match="contiguous"):
        materialize_time_bins(
            report_path,
            tmp_path / "time_bins.csv",
            candidate_name="broken",
        )
