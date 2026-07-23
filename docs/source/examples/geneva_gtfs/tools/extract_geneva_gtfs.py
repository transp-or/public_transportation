"""Build the committed Geneva example from a Swiss national GTFS archive.

The script is intentionally standard-library only.  It streams the large GTFS
tables from the ZIP archive, verifies the source archive, selects TPG tram lines
12, 14 and 18 on 2026-06-02, and writes the compact scenario used by the example.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


ARCHIVE_NAME = "mdb-2898-202605290027.zip"
ARCHIVE_SHA256 = "c6f06bdad9f20349ed08b45daf2ff6114f116a3c231afdd48abe80608382c5dd"
MOBILITY_DATABASE_URL = (
    "https://files.mobilitydatabase.org/mdb-2898/"
    "mdb-2898-202605290027/mdb-2898-202605290027.zip"
)
OFFICIAL_DATASET_URL = (
    "https://data.opentransportdata.swiss/dataset/timetable-2026-gtfs2020"
)
SERVICE_DATE = date(2026, 6, 2)
WINDOW_START_S = 7 * 3600
WINDOW_END_S = 9 * 3600
TPG_AGENCY_ID = "881"
SELECTED_LINES = ("12", "14", "18")
SELECTED_ROUTE_IDS = {
    "91-12-B-j26-1": "12",
    "91-14-D-j26-1": "14",
    "91-18-j26-1": "18",
}
TIME_BINS = (
    ("t0700", "07:00:00", "07:30:00"),
    ("t0730", "07:30:00", "08:00:00"),
    ("t0800", "08:00:00", "08:30:00"),
    ("t0830", "08:30:00", "09:00:00"),
)


def _rows(archive: zipfile.ZipFile, name: str):
    stream = io.TextIOWrapper(archive.open(name), encoding="utf-8-sig", newline="")
    return csv.DictReader(stream)


def _seconds(value: str) -> int:
    hour, minute, second = (int(part) for part in value.split(":"))
    return hour * 3600 + minute * 60 + second


def _hms(value: int) -> str:
    hour, remainder = divmod(int(value), 3600)
    minute, second = divmod(remainder, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_").upper()


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _active_services(archive: zipfile.ZipFile) -> set[str]:
    date_text = SERVICE_DATE.strftime("%Y%m%d")
    weekday = SERVICE_DATE.strftime("%A").lower()
    active = {
        row["service_id"]
        for row in _rows(archive, "calendar.txt")
        if row["start_date"] <= date_text <= row["end_date"] and row[weekday] == "1"
    }
    for row in _rows(archive, "calendar_dates.txt"):
        if row["date"] != date_text:
            continue
        if row["exception_type"] == "1":
            active.add(row["service_id"])
        elif row["exception_type"] == "2":
            active.discard(row["service_id"])
    return active


def _choose_free_od_pairs(patterns: dict[str, Counter[tuple[str, ...]]]) -> list[tuple[str, str]]:
    """Select reproducible, timetable-feasible OD pairs over each line."""
    pairs: set[tuple[str, str]] = set()
    for line in SELECTED_LINES:
        # The two most common full patterns normally represent opposite directions.
        representatives = [pattern for pattern, _ in patterns[line].most_common(2)]
        for pattern in representatives:
            n = len(pattern)
            index_pairs = (
                (0, n - 1),
                (0, (2 * n) // 3),
                (n // 3, n - 1),
                (n // 4, (3 * n) // 4),
            )
            for origin_index, destination_index in index_pairs:
                origin = pattern[min(origin_index, n - 2)]
                destination = pattern[max(destination_index, origin_index + 1)]
                if origin != destination:
                    pairs.add((origin, destination))
    return sorted(pairs)


def build(archive_path: Path, output_dir: Path, *, verify_checksum: bool = True) -> dict[str, object]:
    if verify_checksum:
        actual = _sha256(archive_path)
        if actual != ARCHIVE_SHA256:
            raise ValueError(f"GTFS checksum mismatch: expected {ARCHIVE_SHA256}, got {actual}")

    with zipfile.ZipFile(archive_path) as archive:
        agencies = {row["agency_id"]: row["agency_name"] for row in _rows(archive, "agency.txt")}
        if agencies.get(TPG_AGENCY_ID) != "Transports Publics Genevois":
            raise ValueError("The expected TPG agency record is absent from the source archive.")

        active = _active_services(archive)
        trip_line: dict[str, str] = {}
        trip_headsign: dict[str, str] = {}
        for row in _rows(archive, "trips.txt"):
            line = SELECTED_ROUTE_IDS.get(row["route_id"])
            if line is not None and row["service_id"] in active:
                trip_line[row["trip_id"]] = line
                trip_headsign[row["trip_id"]] = row.get("trip_headsign", "")

        raw_events: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in _rows(archive, "stop_times.txt"):
            trip_id = row["trip_id"]
            if trip_id not in trip_line:
                continue
            raw_events[trip_id].append(
                {
                    "raw_stop_id": row["stop_id"],
                    "sequence": int(row["stop_sequence"]),
                    "arrival_s": _seconds(row["arrival_time"]),
                    "departure_s": _seconds(row["departure_time"]),
                }
            )

        selected_trip_ids = {
            trip_id
            for trip_id, events in raw_events.items()
            if any(WINDOW_START_S <= int(event["departure_s"]) < WINDOW_END_S for event in events)
        }
        raw_stop_ids = {
            str(event["raw_stop_id"])
            for trip_id in selected_trip_ids
            for event in raw_events[trip_id]
        }

        raw_stops: dict[str, dict[str, str]] = {}
        for row in _rows(archive, "stops.txt"):
            if row["stop_id"] in raw_stop_ids:
                raw_stops[row["stop_id"]] = row

    # The Swiss feed exposes platforms as distinct IDs.  Exact stop-name
    # consolidation produces physical boarding/alighting locations.
    stops_by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_stops.values():
        stops_by_name[row["stop_name"].strip()].append(row)

    name_to_id: dict[str, str] = {}
    used_ids: set[str] = set()
    for name in sorted(stops_by_name):
        base = _slug(name)
        stop_id = base
        suffix = 2
        while stop_id in used_ids:
            stop_id = f"{base}_{suffix}"
            suffix += 1
        used_ids.add(stop_id)
        name_to_id[name] = stop_id
    raw_to_physical = {
        raw_id: name_to_id[row["stop_name"].strip()] for raw_id, row in raw_stops.items()
    }

    stop_rows: list[dict[str, object]] = []
    for name in sorted(stops_by_name):
        members = stops_by_name[name]
        stop_rows.append(
            {
                "stop_id": name_to_id[name],
                "name": name,
                "lat": f"{sum(float(row['stop_lat']) for row in members) / len(members):.7f}",
                "lon": f"{sum(float(row['stop_lon']) for row in members) / len(members):.7f}",
            }
        )

    trip_rows: list[dict[str, object]] = []
    stop_time_rows: list[dict[str, object]] = []
    patterns: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    scheduled_paths: list[tuple[tuple[str, int], ...]] = []
    trip_number_by_line: Counter[str] = Counter()
    for source_trip_id in sorted(selected_trip_ids, key=lambda value: (trip_line[value], value)):
        line = trip_line[source_trip_id]
        trip_number_by_line[line] += 1
        trip_id = f"L{line}_T{trip_number_by_line[line]:03d}"
        events = sorted(raw_events[source_trip_id], key=lambda event: int(event["sequence"]))
        consolidated: list[dict[str, object]] = []
        for event in events:
            stop_id = raw_to_physical[str(event["raw_stop_id"])]
            if consolidated and consolidated[-1]["stop_id"] == stop_id:
                consolidated[-1]["departure_s"] = event["departure_s"]
                continue
            consolidated.append({**event, "stop_id": stop_id})
        if len(consolidated) < 2:
            continue
        patterns[line][tuple(str(event["stop_id"]) for event in consolidated)] += 1
        scheduled_paths.append(
            tuple((str(event["stop_id"]), int(event["departure_s"])) for event in consolidated)
        )
        trip_rows.append(
            {
                "trip_id": trip_id,
                "line_id": f"L{line}",
                "capacity": 180,
                "source_gtfs_trip_id": source_trip_id,
                "headsign": trip_headsign[source_trip_id],
            }
        )
        previous_departure_s = -1
        for sequence, event in enumerate(consolidated, start=1):
            arrival_s = max(int(event["arrival_s"]), previous_departure_s + 1)
            # The Swiss GTFS commonly uses zero scheduled dwell.  The domain
            # model requires a strictly positive dwell, matching the assignment
            # builder's documented one-second regularization policy.
            departure_s = max(int(event["departure_s"]), arrival_s + 1)
            previous_departure_s = departure_s
            stop_time_rows.append(
                {
                    "trip_id": trip_id,
                    "stop_id": event["stop_id"],
                    "sequence": sequence,
                    "arrival_s": _hms(arrival_s),
                    "departure_s": _hms(departure_s),
                }
            )

    candidate_pairs = _choose_free_od_pairs(patterns)
    bin_bounds = [(_seconds(start), _seconds(end)) for _, start, end in TIME_BINS]

    def has_service(origin: str, destination: str, start_s: int, end_s: int) -> bool:
        for path in scheduled_paths:
            positions = {stop_id: index for index, (stop_id, _) in enumerate(path)}
            if origin not in positions or destination not in positions:
                continue
            origin_index = positions[origin]
            if origin_index >= positions[destination]:
                continue
            departure_s = path[origin_index][1]
            if start_s - 15 * 60 <= departure_s <= end_s + 15 * 60:
                return True
        return False

    free_pairs = [
        pair
        for pair in candidate_pairs
        if all(has_service(*pair, start_s, end_s) for start_s, end_s in bin_bounds)
    ]
    bin_ids = [item[0] for item in TIME_BINS]
    true_profiles = (
        (18.0, 32.0, 54.0, 27.0),
        (42.0, 25.0, 16.0, 38.0),
        (14.0, 46.0, 31.0, 20.0),
        (35.0, 18.0, 44.0, 24.0),
    )
    prior_profiles = (
        (55.0, 11.0, 18.0, 48.0),
        (12.0, 50.0, 41.0, 13.0),
        (47.0, 15.0, 12.0, 44.0),
        (16.0, 43.0, 20.0, 52.0),
    )
    free_keys = {(origin, destination, bin_id) for origin, destination in free_pairs for bin_id in bin_ids}
    true_rows: list[dict[str, object]] = []
    prior_rows: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []
    stop_ids = sorted(name_to_id.values())
    pair_index = {pair: index for index, pair in enumerate(free_pairs)}
    for origin in stop_ids:
        for destination in stop_ids:
            if origin == destination:
                continue
            for bin_index, bin_id in enumerate(bin_ids):
                key = (origin, destination, bin_id)
                if key in free_keys:
                    index = pair_index[(origin, destination)]
                    scale = 0.75 + 0.1 * (index % 6)
                    true_flow = round(true_profiles[index % len(true_profiles)][bin_index] * scale, 3)
                    prior_flow = round(prior_profiles[index % len(prior_profiles)][bin_index] / scale, 3)
                else:
                    true_flow = 0.0
                    prior_flow = 0.0
                    fixed_rows.append(
                        {
                            "origin_stop_id": origin,
                            "dest_stop_id": destination,
                            "time_bin_id": bin_id,
                            "fixed_flow": 0,
                        }
                    )
                common = {
                    "origin_stop_id": origin,
                    "dest_stop_id": destination,
                    "time_bin_id": bin_id,
                }
                true_rows.append({**common, "flow": true_flow})
                prior_rows.append({**common, "flow": prior_flow})

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "stops.csv", ("stop_id", "name", "lat", "lon"), stop_rows)
    _write_csv(
        output_dir / "lines.csv",
        ("line_id", "short_name"),
        [{"line_id": f"L{line}", "short_name": f"TPG tram {line}"} for line in SELECTED_LINES],
    )
    _write_csv(
        output_dir / "trips.csv",
        ("trip_id", "line_id", "capacity", "source_gtfs_trip_id", "headsign"),
        trip_rows,
    )
    _write_csv(
        output_dir / "stop_times.csv",
        ("trip_id", "stop_id", "sequence", "arrival_s", "departure_s"),
        stop_time_rows,
    )
    _write_csv(
        output_dir / "time_bins.csv",
        ("bin_id", "start_s", "end_s"),
        [{"bin_id": item[0], "start_s": item[1], "end_s": item[2]} for item in TIME_BINS],
    )
    demand_fields = ("origin_stop_id", "dest_stop_id", "time_bin_id", "flow")
    _write_csv(output_dir / "true_demand.csv", demand_fields, true_rows)
    _write_csv(output_dir / "prior_demand.csv", demand_fields, prior_rows)
    _write_csv(
        output_dir / "fixed_demand.csv",
        ("origin_stop_id", "dest_stop_id", "time_bin_id", "fixed_flow"),
        fixed_rows,
    )

    summary: dict[str, object] = {
        "archive_name": ARCHIVE_NAME,
        "archive_sha256": ARCHIVE_SHA256,
        "official_dataset_url": OFFICIAL_DATASET_URL,
        "archived_download_url": MOBILITY_DATABASE_URL,
        "service_date": SERVICE_DATE.isoformat(),
        "agency_id": TPG_AGENCY_ID,
        "agency_name": "Transports Publics Genevois",
        "selected_lines": list(SELECTED_LINES),
        "selection_window": "07:00:00-09:00:00",
        "trip_selection": "complete trips having at least one departure in the half-open selection window",
        "stop_consolidation": "platform stop_ids consolidated by exact stripped stop_name",
        "num_physical_stops": len(stop_rows),
        "num_trips": len(trip_rows),
        "num_stop_times": len(stop_time_rows),
        "num_time_bins": len(TIME_BINS),
        "num_dense_od_cells": len(true_rows),
        "num_free_od_cells": len(free_keys),
        "num_frozen_zero_od_cells": len(fixed_rows),
        "true_total_demand": sum(float(row["flow"]) for row in true_rows),
        "prior_total_demand": sum(float(row["flow"]) for row in prior_rows),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "title": "Geneva TPG tram OD estimation example",
                "description": "Synthetic time-dependent OD estimation on real TPG lines 12, 14 and 18 derived from the 2026 Swiss GTFS timetable.",
                "timezone": "Europe/Zurich",
                "cost_unit": "minutes",
                "extra": {
                    "service_date": SERVICE_DATE.isoformat(),
                    "source": OFFICIAL_DATASET_URL,
                    "source_archive_sha256": ARCHIVE_SHA256,
                    "selected_lines": list(SELECTED_LINES),
                    "selection_window": "07:00-09:00",
                    "demand_unit": "synthetic passenger trips per 30-minute bin",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to the immutable Swiss GTFS ZIP archive")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Scenario data directory",
    )
    parser.add_argument("--skip-checksum", action="store_true", help="Allow an explicitly different archive")
    args = parser.parse_args()
    summary = build(args.archive, args.output, verify_checksum=not args.skip_checksum)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
