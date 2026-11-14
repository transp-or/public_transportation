"""Compact, checksummed measurement-row lookup for gravity viewer maps."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np


_SCHEMA_VERSION = 1
_PARTITION_ROWS = 500
_INSERT_BATCH = 10_000
MEASUREMENT_LOCATION_COLUMNS = (
    "measurement_index",
    "measurement_type",
    "line",
    "direction",
    "observation_time",
    "time_period",
    "stop",
    "observed",
    "modeled",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def read_measurement_location_manifest(root: str | Path, payload: Mapping[str, object]) -> Mapping[str, object]:
    """Read the small lookup index after checking its bundle-bound checksum."""
    relative = Path(str(payload.get("manifest_path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or not relative.as_posix().startswith("partitions/measurement_locations/"):
        raise ValueError("measurement location lookup manifest path is unsafe.")
    path = Path(root) / relative
    if not path.is_file() or _sha256(path) != payload.get("manifest_sha256"):
        raise ValueError("measurement location lookup manifest checksum mismatch.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("measurement location lookup manifest is corrupt.") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("measurement location lookup manifest is invalid.")
    return manifest


def write_measurement_location_lookup(
    *,
    destination: str | Path,
    predicted_measurements_path: str | Path,
    network_stops_path: str | Path,
    source_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Index prediction rows by exact stop ID without retaining the full table."""
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    partitions_root = target / "stops"
    partitions_root.mkdir()

    with Path(network_stops_path).open(newline="", encoding="utf-8") as stream:
        stop_ids = {row["stop_id"] for row in csv.DictReader(stream)}
    predicted_path = Path(predicted_measurements_path)
    total_measurements = 0
    missing_stop_rows = 0
    indexed_rows = 0
    stops: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix=".measurement-locations-", dir=target.parent) as working:
        connection = sqlite3.connect(Path(working) / "locations.sqlite")
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE rows ("
                "stop TEXT NOT NULL, measurement_index INTEGER PRIMARY KEY, "
                "measurement_type TEXT NOT NULL, line TEXT NOT NULL, "
                "direction TEXT NOT NULL, observation_time TEXT NOT NULL, "
                "time_period TEXT NOT NULL, observed REAL NOT NULL, modeled REAL NOT NULL)"
            )
            connection.execute("CREATE TABLE seen_measurements (measurement_index INTEGER PRIMARY KEY)")
            batch: list[tuple[object, ...]] = []
            seen_batch: list[tuple[int]] = []
            with predicted_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                required = {"row_index", "stop", "observed", "modeled"}
                if not required.issubset(reader.fieldnames or ()):
                    raise ValueError("predicted_measurements.csv lacks required measurement-location columns.")
                for row in reader:
                    total_measurements += 1
                    try:
                        measurement_index = int(row["row_index"])
                        observed = float(row["observed"])
                        modeled = float(row["modeled"])
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError("predicted_measurements.csv has an invalid measurement row.") from error
                    if measurement_index < 0 or not np.isfinite(observed) or not np.isfinite(modeled):
                        raise ValueError("predicted_measurements.csv has an invalid measurement row.")
                    seen_batch.append((measurement_index,))
                    stop = str(row.get("stop") or "")
                    if stop == "":
                        missing_stop_rows += 1
                        if len(seen_batch) >= _INSERT_BATCH:
                            connection.executemany("INSERT INTO seen_measurements VALUES (?)", seen_batch)
                            seen_batch.clear()
                        continue
                    batch.append(
                        (
                            stop,
                            measurement_index,
                            str(row.get("measurement_type") or ""),
                            str(row.get("line") or ""),
                            str(row.get("direction") or ""),
                            str(row.get("observation_time") or ""),
                            str(row.get("time_period") or ""),
                            observed,
                            modeled,
                        )
                    )
                    if len(batch) >= _INSERT_BATCH:
                        connection.executemany(
                            "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            batch,
                        )
                        batch.clear()
                    if len(seen_batch) >= _INSERT_BATCH:
                        connection.executemany("INSERT INTO seen_measurements VALUES (?)", seen_batch)
                        seen_batch.clear()
            if batch:
                connection.executemany("INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
            if seen_batch:
                connection.executemany("INSERT INTO seen_measurements VALUES (?)", seen_batch)
            if connection.execute("SELECT COUNT(*) FROM seen_measurements").fetchone()[0] != total_measurements:
                raise ValueError("predicted_measurements.csv contains duplicate row_index values.")
            if total_measurements:
                minimum, maximum = connection.execute("SELECT MIN(measurement_index), MAX(measurement_index) FROM seen_measurements").fetchone()
                if minimum != 0 or maximum != total_measurements - 1:
                    raise ValueError("predicted_measurements.csv row_index values must cover the canonical range.")
            connection.execute("CREATE INDEX rows_by_stop ON rows(stop, measurement_index)")
            connection.commit()

            for (stop,) in connection.execute("SELECT DISTINCT stop FROM rows ORDER BY stop"):
                cursor = connection.execute(
                    "SELECT measurement_index, measurement_type, line, direction, "
                    "observation_time, time_period, stop, observed, modeled "
                    "FROM rows WHERE stop=? ORDER BY measurement_index",
                    (stop,),
                )
                descriptors: list[dict[str, object]] = []
                chunk_index = 0
                while batch := cursor.fetchmany(_PARTITION_ROWS):
                    output = partitions_root / f"{_slug(stop)}-{chunk_index:05d}.csv"
                    with output.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(MEASUREMENT_LOCATION_COLUMNS)
                        writer.writerows(batch)
                    row_count = len(batch)
                    indexed_rows += row_count
                    descriptors.append(
                        {
                            "key": stop,
                            "chunk_index": chunk_index,
                            "path": output.relative_to(target.parent.parent).as_posix(),
                            "row_count": row_count,
                            "size_bytes": output.stat().st_size,
                            "sha256": _sha256(output),
                        }
                    )
                    chunk_index += 1
                count = sum(int(item["row_count"]) for item in descriptors)
                stops[str(stop)] = {"row_count": count, "partitions": descriptors}
        except sqlite3.IntegrityError as error:
            raise ValueError("predicted_measurements.csv contains duplicate row_index values.") from error
        finally:
            connection.close()

    unmatched_stop_ids = sorted(set(stops) - stop_ids)
    manifest: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "dataset_type": "gravity_measurement_location_lookup",
        "available": True,
        "number_of_measurements": total_measurements,
        "indexed_measurements": indexed_rows,
        "missing_stop_rows": missing_stop_rows,
        "number_of_stops": len(stops),
        "unmatched_stop_ids": unmatched_stop_ids,
        "partition_rows_limit": _PARTITION_ROWS,
        "source_identity": dict(source_identity or {}),
        "columns": list(MEASUREMENT_LOCATION_COLUMNS),
        "stops": stops,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "available": True,
        "schema_version": _SCHEMA_VERSION,
        "manifest_path": manifest_path.relative_to(target.parent.parent).as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "number_of_measurements": total_measurements,
        "indexed_measurements": indexed_rows,
        "missing_stop_rows": missing_stop_rows,
        "number_of_stops": len(stops),
        "unmatched_stop_count": len(unmatched_stop_ids),
        "source_identity": dict(source_identity or {}),
    }


def validate_measurement_location_lookup(root: str | Path, payload: Mapping[str, object]) -> Mapping[str, object]:
    """Validate the compact stop index and checksums of all bounded partitions."""
    if payload.get("available") is not True or payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("measurement location lookup metadata is invalid.")
    manifest = read_measurement_location_manifest(root, payload)
    if manifest.get("schema_version") != _SCHEMA_VERSION or manifest.get("dataset_type") != "gravity_measurement_location_lookup":
        raise ValueError("measurement location lookup schema is incompatible.")
    if tuple(manifest.get("columns", ())) != MEASUREMENT_LOCATION_COLUMNS or manifest.get("partition_rows_limit") != _PARTITION_ROWS:
        raise ValueError("measurement location lookup columns or partition bound are invalid.")
    if (
        manifest.get("number_of_measurements") != payload.get("number_of_measurements")
        or manifest.get("indexed_measurements") != payload.get("indexed_measurements")
        or manifest.get("missing_stop_rows") != payload.get("missing_stop_rows")
        or manifest.get("number_of_stops") != payload.get("number_of_stops")
        or manifest.get("source_identity", {}) != payload.get("source_identity", {})
    ):
        raise ValueError("measurement location lookup counts differ from bundle metadata.")
    stops = manifest.get("stops")
    if not isinstance(stops, Mapping):
        raise ValueError("measurement location lookup stops index is invalid.")
    expected_columns = tuple(MEASUREMENT_LOCATION_COLUMNS)
    total_rows = 0
    for stop, entry in stops.items():
        if not isinstance(stop, str) or stop == "" or not isinstance(entry, Mapping):
            raise ValueError("measurement location lookup stop entry is invalid.")
        descriptors = entry.get("partitions")
        if not isinstance(descriptors, list):
            raise ValueError("measurement location lookup partitions are invalid.")
        stop_count = 0
        previous_index = -1
        for chunk_index, descriptor in enumerate(descriptors):
            if not isinstance(descriptor, Mapping):
                raise ValueError("measurement location lookup partition is invalid.")
            relative = Path(str(descriptor.get("path", "")))
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.as_posix().startswith("partitions/measurement_locations/")
                or descriptor.get("key") != stop
                or descriptor.get("chunk_index") != chunk_index
            ):
                raise ValueError("measurement location lookup partition path is unsafe.")
            path = Path(root) / relative
            if not path.is_file() or path.stat().st_size != descriptor.get("size_bytes") or _sha256(path) != descriptor.get("sha256"):
                raise ValueError("measurement location lookup partition checksum mismatch.")
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != expected_columns:
                    raise ValueError("measurement location lookup columns are invalid.")
                count = 0
                for row in reader:
                    try:
                        index = int(row["measurement_index"])
                        numbers = (float(row["observed"]), float(row["modeled"]))
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError("measurement location lookup row is invalid.") from error
                    if (
                        row["stop"] != stop
                        or index <= previous_index
                        or index < 0
                        or index >= int(manifest.get("number_of_measurements", -1))
                        or not np.all(np.isfinite(numbers))
                    ):
                        raise ValueError("measurement location lookup row is inconsistent.")
                    previous_index = index
                    count += 1
            if count != descriptor.get("row_count") or count <= 0 or count > _PARTITION_ROWS:
                raise ValueError("measurement location lookup row count is invalid.")
            stop_count += count
        if stop_count != entry.get("row_count"):
            raise ValueError("measurement location lookup stop row count mismatch.")
        total_rows += stop_count
    unmatched = manifest.get("unmatched_stop_ids")
    if not isinstance(unmatched, list) or any(not isinstance(value, str) for value in unmatched) or unmatched != sorted(set(unmatched)):
        raise ValueError("measurement location lookup unmatched stop IDs are invalid.")
    if (
        total_rows != manifest.get("indexed_measurements")
        or total_rows + int(manifest.get("missing_stop_rows", -1)) != manifest.get("number_of_measurements")
        or len(stops) != manifest.get("number_of_stops")
        or len(unmatched) != payload.get("unmatched_stop_count")
    ):
        raise ValueError("measurement location lookup aggregate counts are invalid.")
    if (Path(root) / "network" / "stops.csv").is_file():
        with (Path(root) / "network" / "stops.csv").open(newline="", encoding="utf-8") as stream:
            network_ids = {row["stop_id"] for row in csv.DictReader(stream)}
        if sorted(set(stops) - network_ids) != manifest.get("unmatched_stop_ids"):
            raise ValueError("measurement location lookup unmatched stop IDs are invalid.")
    return manifest


__all__ = [
    "MEASUREMENT_LOCATION_COLUMNS",
    "read_measurement_location_manifest",
    "validate_measurement_location_lookup",
    "write_measurement_location_lookup",
]
