"""Portable, validated gravity-result bundles for independent viewers.

The bundle API deliberately stops at data access.  It does not import or
activate an assignment operator and it never recomputes the fitted model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from public_transportation.inference.block_coordinate._canonical import canonical_json
from public_transportation.inference.od_parameter_layout import ODParameterLayout

from .estimator import GravityEstimationResult
from .identifiability import (
    GravityODIdentifiability,
    read_gravity_od_identifiability,
    write_gravity_od_identifiability,
)
from .objective import GravityLikelihood
from .reporting import (
    GRAVITY_REPORT_PROVENANCE_FIELDS,
    GravityDetailedReport,
)
from .specification import GravityModelSpecification
from .validation import GravityAdequacyReport, GravityValidationMetadata


GRAVITY_VIEWER_BUNDLE_SCHEMA_VERSION = 1
GRAVITY_VIEWER_BUNDLE_TYPE = "gravity_viewer_bundle"
GRAVITY_VIEWER_CACHE_FORMAT_VERSION = 2
GRAVITY_VIEWER_ROUTE_SECTION_SCHEMA_VERSION = 1
_MODEL_SPECIFICATION_SCHEMA_VERSION = 1
_MODEL_SPECIFICATION_FILE = "model_specification.json"
_VIEWER_SUMMARY_FILE = "viewer_summary.json"
_PARTITION_ROOT = "partitions"
_OD_PARTITION_ROOT = "partitions/od"
_ROUTE_FLOW_PARTITION_ROOT = "partitions/route_flows"
_ROUTE_SECTION_PARTITION_ROOT = "partitions/route_sections"
_SECTION_LOAD_PARTITION_ROOT = "partitions/section_loads"
_IDENTIFIABILITY_MESSAGE = (
    "OD identifiability diagnostics were not available; inference_score remains "
    "only a structural fixed/free indicator."
)
_TABLE_FILES = (
    "full_od.csv",
    "predicted_measurements.csv",
    "residuals.csv",
    "grouped_residuals.csv",
    "parameters.csv",
)
_OPTIONAL_DERIVED_FILES = ("od_map.csv", "stop_summary.csv")
_REPORT_FILES = ("report.json", "executive_summary.md")
_OPTIONAL_REPORT_FILES = ("report.md",)
_FULL_OD_COLUMNS = (
    "od_index",
    "origin_stop_id",
    "destination_stop_id",
    "departure_time_bin",
    "estimated_demand",
    "cell_status",
    "inference_basis",
    "inference_score",
    "fixed_value",
    "prior_demand",
    "estimated_to_prior_ratio",
    "count_information_share",
    "regularization_information_share",
    "auxiliary_information_share",
    "local_information_variance",
    "identifiability_class",
    "identifiability_reason",
)
_PREDICTED_COLUMNS = (
    "row_index",
    "measurement_type",
    "line",
    "direction",
    "stop",
    "observation_time",
    "time_period",
    "observed",
    "modeled",
)
_RESIDUAL_COLUMNS = (*_PREDICTED_COLUMNS, "residual", "variance", "standardized_residual")
_PARAMETER_COLUMNS = ("parameter_index", "parameter_name", "raw_value", "physical_value")
_OD_MAP_COLUMNS = (
    "origin_stop_id",
    "destination_stop_id",
    "departure_time_bin",
    "observed_demand",
    "modeled_demand",
    "residual",
    "fixed_free_status",
    "identifiability_class",
)
_STOP_SUMMARY_COLUMNS = (
    "stop_id",
    "observed_total",
    "modeled_total",
    "residual",
    "incoming_total",
    "outgoing_total",
)
_NETWORK_FILES = ("stops.csv", "lines.csv", "trips.csv", "stop_times.csv", "shapes.csv")
_CANONICAL_PROVENANCE = tuple(GRAVITY_REPORT_PROVENANCE_FIELDS)
_ROUTE_FLOW_COLUMNS = (
    "origin_stop_id",
    "destination_stop_id",
    "departure_time_bin",
    "line_id",
    "trip_id",
    "boarding_stop_id",
    "alighting_stop_id",
    "route_leg_index",
    "route_leg_count",
    "route_legs",
    "assigned_demand",
    "path_share",
    "transfer_count",
    "flow_semantics",
    "route_path_id",
    "pattern_id",
    "direction",
    "assignment_fingerprint",
)
_ROUTE_FLOW_REQUIRED_COLUMNS = (
    "origin_stop_id",
    "destination_stop_id",
    "departure_time_bin",
    "assigned_demand",
    "flow_semantics",
)
_SECTION_CONTRIBUTION_COLUMNS = (
    "section_id",
    "origin_stop_id",
    "destination_stop_id",
    "departure_time_bin",
    "route_path_id",
    "route_leg_index",
    "route_leg_count",
    "line_id",
    "pattern_id",
    "direction",
    "trip_id",
    "boarding_stop_id",
    "alighting_stop_id",
    "assigned_demand",
    "assigned_contribution",
    "path_share",
    "transfer_count",
    "flow_semantics",
    "assignment_fingerprint",
)
_SECTION_LOAD_COLUMNS = (
    "section_id",
    "line_id",
    "pattern_id",
    "direction",
    "from_stop_id",
    "to_stop_id",
    "assigned_flow",
    "distinct_od_count",
)


def _json_safe(value: object) -> object:
    """Convert public result/config values to strict JSON-compatible values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("bundle metadata must not contain non-finite numbers.")
        return value
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path} has no header.")
        return list(reader.fieldnames), list(reader)


def _iter_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    """Yield CSV rows without retaining the complete table in memory."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path} has no header.")
        yield from reader


def _csv_fieldnames(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            return list(next(reader))
        except StopIteration as error:
            raise ValueError(f"CSV file {path} has no header.") from error


def _csv_row_count(path: Path) -> int:
    """Count data rows while streaming through a CSV file."""
    with path.open(newline="", encoding="utf-8") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def _partition_slug(value: object) -> str:
    text = str(value or "unassigned").strip() or "unassigned"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "unassigned"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:64]}-{digest}"


def _time_bin_sort_key(value: object) -> tuple[int, int, str]:
    text = str(value)
    match = re.search(r"(\d+)", text)
    return (0 if match else 1, int(match.group(1)) if match else 0, text)


def _finite_optional(value: object) -> str:
    if value is None or str(value).strip() == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"route-flow value {value!r} is not numeric.") from error
    if not np.isfinite(number):
        raise ValueError("route-flow numeric values must be finite.")
    return format(number, ".17g")


def _route_flow_row(raw: Mapping[str, object]) -> dict[str, object]:
    """Normalize one sparse route-flow record without changing its semantics."""
    if "minimum_transfers" in raw:
        raise ValueError(
            "minimum_transfers is a demand feature and cannot be used as route-flow transfer_count."
        )
    aliases = {
        "origin": "origin_stop_id",
        "destination": "destination_stop_id",
        "line": "line_id",
        "trip": "trip_id",
        "boarding_stop": "boarding_stop_id",
        "alighting_stop": "alighting_stop_id",
        "legs": "route_legs",
        "demand": "assigned_demand",
        "flow": "assigned_demand",
        "share": "path_share",
        "transfers": "transfer_count",
        "path_id": "route_path_id",
        "route_id": "route_path_id",
        "pattern": "pattern_id",
        "direction_id": "direction",
    }
    values: dict[str, object] = {}
    for column in _ROUTE_FLOW_COLUMNS:
        value = raw.get(column)
        if value is None:
            for alias, target in aliases.items():
                if target == column and alias in raw:
                    value = raw[alias]
                    break
        values[column] = value
    missing = [
        column
        for column in _ROUTE_FLOW_REQUIRED_COLUMNS
        if values[column] is None or str(values[column]).strip() == ""
    ]
    if missing:
        raise ValueError("route-flow row is missing required fields: " + ", ".join(missing))
    semantics = str(values["flow_semantics"]).strip()
    semantics = {
        "link_marginal": "service_marginal",
        "trip_marginal": "service_marginal",
        "expected_service_usage": "service_marginal",
    }.get(semantics, semantics)
    allowed = {"path_probability", "service_marginal", "passenger_leg_usage"}
    if semantics not in allowed:
        raise ValueError(
            "flow_semantics must be path_probability, service_marginal, "
            "or passenger_leg_usage."
        )
    assigned = float(values["assigned_demand"])
    if not np.isfinite(assigned) or assigned < 0:
        raise ValueError("assigned_demand must be a finite non-negative number.")
    path_share = values["path_share"]
    if path_share is not None and str(path_share).strip() != "":
        path_share_value = float(path_share)
        if not np.isfinite(path_share_value) or not 0 <= path_share_value <= 1:
            raise ValueError("path_share must be finite and between zero and one.")
        path_share = format(path_share_value, ".17g")
    else:
        path_share = ""
    transfer_count = values["transfer_count"]
    if transfer_count is not None and str(transfer_count).strip() != "":
        transfer_value = float(transfer_count)
        if not np.isfinite(transfer_value) or transfer_value < 0 or transfer_value != np.floor(transfer_value):
            raise ValueError("transfer_count must be a finite non-negative integer.")
        transfer_count = str(int(transfer_value))
    else:
        transfer_count = ""
    return {
        "origin_stop_id": str(values["origin_stop_id"]),
        "destination_stop_id": str(values["destination_stop_id"]),
        "departure_time_bin": str(values["departure_time_bin"]),
        "line_id": str(values["line_id"] or ""),
        "trip_id": str(values["trip_id"] or ""),
        "boarding_stop_id": str(values["boarding_stop_id"] or ""),
        "alighting_stop_id": str(values["alighting_stop_id"] or ""),
        "route_leg_index": str(values["route_leg_index"] or ""),
        "route_leg_count": str(values["route_leg_count"] or ""),
        "route_legs": (
            json.dumps(values["route_legs"], separators=(",", ":"))
            if isinstance(values["route_legs"], (list, tuple))
            else str(values["route_legs"] or "")
        ),
        "assigned_demand": format(assigned, ".17g"),
        "path_share": path_share,
        "transfer_count": transfer_count,
        "flow_semantics": semantics,
        "route_path_id": str(values.get("route_path_id") or ""),
        "pattern_id": str(values.get("pattern_id") or ""),
        "direction": str(values.get("direction") or ""),
        "assignment_fingerprint": str(values.get("assignment_fingerprint") or ""),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    """Write a small deterministic CSV atomically."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_partitioned_table(
    *,
    source: Path,
    destination: Path,
    partition_field: str,
    fieldnames: Sequence[str],
    row_transform: Any | None = None,
) -> tuple[list[dict[str, object]], dict[str, set[str]]]:
    """Partition a CSV by one field using bounded, streaming writes."""
    destination.mkdir(parents=True, exist_ok=True)
    descriptors: list[dict[str, object]] = []
    values: dict[str, set[str]] = {
        "origins": set(),
        "destinations": set(),
        "lines": set(),
        "trips": set(),
        "semantics": set(),
        "transfers": set(),
        "patterns": set(),
        "directions": set(),
    }
    writers: dict[str, csv.DictWriter] = {}
    streams: dict[str, Any] = {}
    counts: dict[str, int] = {}
    paths: dict[str, Path] = {}
    try:
        for source_row in _iter_csv_rows(source):
            row = row_transform(source_row) if row_transform is not None else dict(source_row)
            key = str(row.get(partition_field, "") or "unassigned")
            slug = _partition_slug(key)
            if slug not in writers:
                path = destination / f"{slug}.csv"
                stream = path.open("w", newline="", encoding="utf-8")
                writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="raise")
                writer.writeheader()
                writers[slug] = writer
                streams[slug] = stream
                counts[slug] = 0
                paths[slug] = path
            writers[slug].writerow({name: row.get(name, "") for name in fieldnames})
            counts[slug] += 1
            if "origin_stop_id" in row:
                values["origins"].add(str(row["origin_stop_id"]))
            if "destination_stop_id" in row:
                values["destinations"].add(str(row["destination_stop_id"]))
            if "line_id" in row and str(row["line_id"]):
                values["lines"].add(str(row["line_id"]))
            if "trip_id" in row and str(row["trip_id"]):
                values["trips"].add(str(row["trip_id"]))
            if "flow_semantics" in row and str(row["flow_semantics"]):
                values["semantics"].add(str(row["flow_semantics"]))
            if "transfer_count" in row and str(row["transfer_count"]):
                values["transfers"].add(str(row["transfer_count"]))
            if "pattern_id" in row and str(row["pattern_id"]):
                values["patterns"].add(str(row["pattern_id"]))
            if "direction" in row and str(row["direction"]):
                values["directions"].add(str(row["direction"]))
    finally:
        for stream in streams.values():
            stream.close()
    for slug, path in sorted(paths.items()):
        descriptors.append(
            {
                "key": next(
                    (
                        str(row.get(partition_field, "") or "unassigned")
                        for row in _iter_csv_rows(path)
                    ),
                    "unassigned",
                ),
                "path": path.relative_to(destination.parent.parent).as_posix(),
                "file": path.relative_to(destination.parent.parent).as_posix(),
                "row_count": counts[slug],
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return descriptors, values


def _network_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str | None:
    return _first_column(fieldnames, candidates)


def _read_network_patterns(root: Path) -> tuple[
    dict[str, dict[str, object]], dict[str, tuple[str, ...]], dict[tuple[str, str], str]
]:
    """Return validated trip patterns, ordered stops, and trip-to-pattern IDs."""
    network = root / "network"
    trips_path = network / "trips.csv"
    stop_times_path = network / "stop_times.csv"
    if not trips_path.is_file() or not stop_times_path.is_file():
        return {}, {}, {}
    trip_fields = _csv_fieldnames(trips_path)
    trip_id_field = _network_column(trip_fields, ("trip_id", "trip", "id"))
    line_field = _network_column(trip_fields, ("line_id", "route_id", "line"))
    direction_field = _network_column(trip_fields, ("direction_id", "direction", "dir"))
    if trip_id_field is None:
        return {}, {}, {}
    trip_meta: dict[str, tuple[str, str, str]] = {}
    for row in _iter_csv_rows(trips_path):
        trip = str(row.get(trip_id_field, "")).strip()
        if not trip:
            continue
        trip_meta[trip] = (
            str(row.get(line_field, "") if line_field else "").strip(),
            str(row.get(direction_field, "") if direction_field else "").strip(),
            str(row.get("shape_id", "")).strip(),
        )
    st_fields = _csv_fieldnames(stop_times_path)
    st_trip = _network_column(st_fields, ("trip_id", "trip"))
    st_stop = _network_column(st_fields, ("stop_id", "stop"))
    st_sequence = _network_column(st_fields, ("stop_sequence", "sequence", "seq"))
    if st_trip is None or st_stop is None:
        return {}, {}, {}
    grouped: dict[str, list[tuple[float, int, str]]] = {}
    ordinal = 0
    for row in _iter_csv_rows(stop_times_path):
        trip = str(row.get(st_trip, "")).strip()
        stop = str(row.get(st_stop, "")).strip()
        if not trip or not stop:
            continue
        raw_sequence = row.get(st_sequence, "") if st_sequence else ""
        try:
            sequence = float(raw_sequence)
        except (TypeError, ValueError):
            sequence = float(ordinal)
        grouped.setdefault(trip, []).append((sequence, ordinal, stop))
        ordinal += 1
    patterns: dict[str, dict[str, object]] = {}
    trip_patterns: dict[tuple[str, str], str] = {}
    trip_sequences: dict[str, tuple[str, ...]] = {}
    for trip, items in grouped.items():
        sequence = tuple(item[2] for item in sorted(items, key=lambda item: (item[0], item[1])))
        if len(sequence) < 2:
            continue
        line, direction, shape_id = trip_meta.get(trip, ("", "", ""))
        key = canonical_json({"line_id": line, "direction": direction, "shape_id": shape_id, "stops": sequence})
        pattern_id = "p-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        patterns.setdefault(
            pattern_id,
            {"line_id": line, "direction": direction, "shape_id": shape_id,
             "stops": sequence, "trips": []},
        )["trips"].append(trip)
        trip_patterns[(trip, line)] = pattern_id
        trip_sequences[trip] = sequence
    return patterns, trip_sequences, trip_patterns


def _write_line_geometry(staging: Path) -> dict[str, object]:
    """Write a compact line/section catalog and GeoJSON layer from the snapshot."""
    patterns, _trip_sequences, _trip_patterns = _read_network_patterns(staging)
    network = staging / "network"
    stops_path = network / "stops.csv"
    if not patterns or not stops_path.is_file():
        return {
            "available": False,
            "message": "validated trips/stop_times and stop coordinates are required",
            "schema_version": GRAVITY_VIEWER_ROUTE_SECTION_SCHEMA_VERSION,
        }
    stop_fields = _csv_fieldnames(stops_path)
    stop_id = _network_column(stop_fields, ("stop_id", "stop", "id"))
    lat_field = _network_column(stop_fields, ("lat", "latitude", "y"))
    lon_field = _network_column(stop_fields, ("lon", "longitude", "lng", "x"))
    if stop_id is None or lat_field is None or lon_field is None:
        return {
            "available": False,
            "message": "stop coordinates are unavailable for line geometry",
            "schema_version": GRAVITY_VIEWER_ROUTE_SECTION_SCHEMA_VERSION,
        }
    coordinates: dict[str, tuple[float, float]] = {}
    for row in _iter_csv_rows(stops_path):
        identifier = str(row.get(stop_id, "")).strip()
        try:
            lat = float(row.get(lat_field, ""))
            lon = float(row.get(lon_field, ""))
        except (TypeError, ValueError):
            continue
        if identifier and np.isfinite(lat) and np.isfinite(lon):
            coordinates[identifier] = (lon, lat)
    shape_geometries: dict[str, list[list[float]]] = {}
    shapes_path = network / "shapes.csv"
    if shapes_path.is_file():
        shape_fields = _csv_fieldnames(shapes_path)
        shape_id_field = _network_column(shape_fields, ("shape_id", "shape"))
        shape_lat_field = _network_column(shape_fields, ("shape_pt_lat", "lat", "latitude"))
        shape_lon_field = _network_column(shape_fields, ("shape_pt_lon", "lon", "longitude"))
        shape_seq_field = _network_column(shape_fields, ("shape_pt_sequence", "sequence", "seq"))
        if shape_id_field and shape_lat_field and shape_lon_field:
            grouped_shapes: dict[str, list[tuple[float, int, list[float]]]] = {}
            for ordinal, row in enumerate(_iter_csv_rows(shapes_path)):
                try:
                    point = [float(row[shape_lon_field]), float(row[shape_lat_field])]
                    sequence = float(row.get(shape_seq_field, ordinal)) if shape_seq_field else float(ordinal)
                except (TypeError, ValueError):
                    continue
                grouped_shapes.setdefault(str(row[shape_id_field]), []).append((sequence, ordinal, point))
            shape_geometries = {
                key: [point for _sequence, _ordinal, point in sorted(points, key=lambda item: (item[0], item[1]))]
                for key, points in grouped_shapes.items() if len(points) >= 2
            }
    catalog_rows: list[dict[str, object]] = []
    features: list[dict[str, object]] = []
    for pattern_id, pattern in sorted(patterns.items()):
        stops = tuple(pattern["stops"])
        line = str(pattern.get("line_id", ""))
        direction = str(pattern.get("direction", ""))
        for index, (from_stop, to_stop) in enumerate(zip(stops, stops[1:], strict=False)):
            section_key = canonical_json(
                {"line_id": line, "pattern_id": pattern_id, "direction": direction,
                 "sequence": index, "from_stop_id": from_stop, "to_stop_id": to_stop}
            )
            section_id = "s-" + hashlib.sha256(section_key.encode("utf-8")).hexdigest()[:20]
            from_xy = coordinates.get(from_stop)
            to_xy = coordinates.get(to_stop)
            geometry = None
            shape_id = str(pattern.get("shape_id", ""))
            shape_points = shape_geometries.get(shape_id)
            if shape_points:
                geometry = {"type": "LineString", "coordinates": shape_points}
                geometry_mode = "source_route_shape"
            elif from_xy is not None and to_xy is not None:
                geometry = {"type": "LineString", "coordinates": [list(from_xy), list(to_xy)]}
                geometry_mode = "stop_to_stop_approximation"
            else:
                geometry_mode = "unavailable"
            if geometry is not None:
                features.append(
                    {"type": "Feature", "id": section_id,
                     "properties": {"section_id": section_id, "line_id": line,
                                    "pattern_id": pattern_id, "direction": direction,
                                    "sequence": index, "from_stop_id": from_stop,
                                    "to_stop_id": to_stop, "geometry_mode": geometry_mode},
                     "geometry": geometry}
                )
            catalog_rows.append(
                {"section_id": section_id, "line_id": line, "pattern_id": pattern_id,
                 "direction": direction, "sequence": index, "from_stop_id": from_stop,
                 "to_stop_id": to_stop, "geometry_ref": section_id if geometry else ""}
            )
    geometry_dir = staging / "line_geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(geometry_dir / "sections.csv", (
        "section_id", "line_id", "pattern_id", "direction", "sequence",
        "from_stop_id", "to_stop_id", "geometry_ref"
    ), catalog_rows)
    _write_json(
        geometry_dir / "lines.geojson",
        {"type": "FeatureCollection", "geometry_mode": "stop_to_stop_approximation",
         "features": features},
    )
    catalog_path = geometry_dir / "sections.csv"
    geometry_path = geometry_dir / "lines.geojson"
    return {
        "available": bool(catalog_rows),
        "schema_version": GRAVITY_VIEWER_ROUTE_SECTION_SCHEMA_VERSION,
        "geometry_mode": "source_route_shape" if shape_geometries else "stop_to_stop_approximation",
        "message": ("validated source route-shape geometry" if shape_geometries else
                    "straight stop-to-stop approximation; not a road-following vehicle shape"),
        "catalog_path": "line_geometry/sections.csv",
        "geometry_path": "line_geometry/lines.geojson",
        "catalog_sha256": _sha256(catalog_path),
        "geometry_sha256": _sha256(geometry_path),
        "catalog_size_bytes": catalog_path.stat().st_size,
        "geometry_size_bytes": geometry_path.stat().st_size,
        "patterns": len(patterns),
        "sections": len(catalog_rows),
    }


def _write_route_section_products(staging: Path, geometry: Mapping[str, object]) -> dict[str, object]:
    """Expand saved route legs into directed section contributions and summaries."""
    route_source = staging / "route_flows.csv"
    if not route_source.is_file() or not geometry.get("available"):
        return {
            "available": False,
            "schema_version": GRAVITY_VIEWER_ROUTE_SECTION_SCHEMA_VERSION,
            "message": "route-section data unavailable: no compatible post-fit route export",
        }
    patterns, trip_sequences, trip_patterns = _read_network_patterns(staging)
    section_lookup: dict[tuple[str, str, str, str], str] = {}
    section_meta: dict[str, dict[str, str]] = {}
    for row in _iter_csv_rows(staging / str(geometry["catalog_path"])):
        section_id = str(row["section_id"])
        section_meta[section_id] = dict(row)
        section_lookup[(str(row["line_id"]), str(row["pattern_id"]), str(row["from_stop_id"]), str(row["to_stop_id"]))] = section_id
    contributors_dir = staging / _ROUTE_SECTION_PARTITION_ROOT
    loads_dir = staging / _SECTION_LOAD_PARTITION_ROOT
    contributors_dir.mkdir(parents=True, exist_ok=True)
    loads_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, csv.DictWriter] = {}
    streams: dict[str, Any] = {}
    counts: dict[str, int] = {}
    paths: dict[str, Path] = {}
    section_index: dict[str, dict[str, dict[str, int]]] = {}
    flow_semantics_seen: set[str] = set()
    load_db_path = contributors_dir / ".section_loads.sqlite"
    load_db = sqlite3.connect(load_db_path)
    load_db.execute(
        "CREATE TABLE section_od (bin TEXT, section_id TEXT, origin TEXT, destination TEXT, assigned REAL, "
        "PRIMARY KEY (bin, section_id, origin, destination))"
    )
    try:
        for raw in _iter_csv_rows(route_source):
            row = _route_flow_row(raw)
            flow_semantics_seen.add(str(row["flow_semantics"]))
            leg_rows: list[dict[str, object]] = [row]
            if (not row["boarding_stop_id"] or not row["alighting_stop_id"]) and row["route_legs"]:
                try:
                    encoded_legs = json.loads(row["route_legs"])
                except (TypeError, ValueError, json.JSONDecodeError):
                    encoded_legs = ()
                if isinstance(encoded_legs, list):
                    leg_rows = []
                    for leg_number, leg in enumerate(encoded_legs):
                        if not isinstance(leg, Mapping):
                            continue
                        leg_row = dict(row)
                        for source, target in (("line", "line_id"), ("line_id", "line_id"),
                                               ("trip", "trip_id"), ("trip_id", "trip_id"),
                                               ("boarding_stop", "boarding_stop_id"),
                                               ("boarding_stop_id", "boarding_stop_id"),
                                               ("alighting_stop", "alighting_stop_id"),
                                               ("alighting_stop_id", "alighting_stop_id"),
                                               ("pattern", "pattern_id"), ("pattern_id", "pattern_id"),
                                               ("direction", "direction")):
                            if source in leg:
                                leg_row[target] = str(leg[source])
                        leg_row["route_leg_index"] = str(leg.get("route_leg_index", leg_number))
                        leg_rows.append(leg_row)
            if row["route_path_id"]:
                base_path_id = row["route_path_id"]
            else:
                path_identity = {key: row[key] for key in (
                    "origin_stop_id", "destination_stop_id", "departure_time_bin", "assigned_demand"
                )}
                path_identity.update(
                    {"route_legs": row.get("route_legs", "")} if row.get("route_legs") else
                    {key: row[key] for key in ("line_id", "trip_id", "boarding_stop_id", "alighting_stop_id")}
                )
                base_path_id = "rp-" + hashlib.sha256(canonical_json(path_identity).encode("utf-8")).hexdigest()[:20]
            for leg_row in leg_rows:
                line = str(leg_row["line_id"])
                trip = str(leg_row["trip_id"])
                pattern = str(leg_row["pattern_id"]) or trip_patterns.get((trip, line), "")
                sequence = patterns.get(pattern, {}).get("stops") if pattern else trip_sequences.get(trip)
                if not sequence:
                    if leg_row["flow_semantics"] == "path_probability" and leg_row["boarding_stop_id"] and leg_row["alighting_stop_id"]:
                        raise ValueError("route path cannot be expanded into validated line sections.")
                    continue
                boarding = str(leg_row["boarding_stop_id"])
                alighting = str(leg_row["alighting_stop_id"])
                try:
                    start = sequence.index(boarding)
                    end = sequence.index(alighting, start + 1)
                except ValueError:
                    if leg_row["flow_semantics"] == "path_probability":
                        raise ValueError("route path boarding/alighting stops are not ordered in the validated pattern.")
                    continue
                row = leg_row
                # The stable path identity is retained across all expanded legs.
                path_id = base_path_id
                for index in range(start, end):
                    from_stop, to_stop = sequence[index], sequence[index + 1]
                    section_id = section_lookup.get((line, pattern, from_stop, to_stop))
                    if section_id is None:
                        if row["flow_semantics"] == "path_probability":
                            raise ValueError("route path section identity is absent from the validated line catalog.")
                        continue
                    key = row["departure_time_bin"]
                    if key not in writers:
                        path = contributors_dir / f"{_partition_slug(key)}.csv"
                        stream = path.open("w", newline="", encoding="utf-8")
                        writers[key] = csv.DictWriter(stream, fieldnames=list(_SECTION_CONTRIBUTION_COLUMNS), extrasaction="raise")
                        writers[key].writeheader()
                        streams[key] = stream
                        counts[key] = 0
                        paths[key] = path
                    contribution = {name: row.get(name, "") for name in _SECTION_CONTRIBUTION_COLUMNS}
                    contribution.update({"section_id": section_id, "route_path_id": path_id,
                                         "pattern_id": pattern, "line_id": line,
                                         "assigned_contribution": row["assigned_demand"]})
                    writers[key].writerow(contribution)
                    section_index.setdefault(key, {}).setdefault(section_id, {"row_start": counts[key], "row_count": 0})["row_count"] += 1
                    counts[key] += 1
                    load_db.execute(
                        "INSERT INTO section_od VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(bin, section_id, origin, destination) DO UPDATE SET assigned = assigned + excluded.assigned",
                        (key, section_id, row["origin_stop_id"], row["destination_stop_id"], float(row["assigned_demand"])),
                    )
    finally:
        for stream in streams.values():
            stream.close()
        load_db.commit()
        load_db.close()
    contributor_descriptors = []
    for key, path in sorted(paths.items()):
        contributor_descriptors.append({"key": key, "path": path.relative_to(staging).as_posix(),
                                        "file": path.relative_to(staging).as_posix(),
                                        "row_count": counts[key], "sha256": _sha256(path),
                                        "size_bytes": path.stat().st_size})
    index_path = contributors_dir / "index.json"
    _write_json(index_path, {"schema_version": GRAVITY_VIEWER_ROUTE_SECTION_SCHEMA_VERSION,
                             "partition_field": "departure_time_bin", "sections": section_index})
    load_descriptors = []
    load_connection = sqlite3.connect(load_db_path)
    load_bins = [row[0] for row in load_connection.execute("SELECT DISTINCT bin FROM section_od ORDER BY bin")]
    for key in sorted(load_bins, key=_time_bin_sort_key):
        rows = []
        for section_id, assigned_flow, distinct_od_count in load_connection.execute(
            "SELECT section_id, SUM(assigned), COUNT(*) FROM section_od WHERE bin = ? GROUP BY section_id ORDER BY section_id",
            (key,),
        ):
            meta = section_meta[section_id]
            rows.append({"section_id": section_id, "line_id": meta["line_id"],
                         "pattern_id": meta["pattern_id"], "direction": meta["direction"],
                         "from_stop_id": meta["from_stop_id"], "to_stop_id": meta["to_stop_id"],
                         "assigned_flow": format(float(assigned_flow), ".17g"),
                         "distinct_od_count": int(distinct_od_count)})
        path = loads_dir / f"{_partition_slug(key)}.csv"
        _write_csv(path, _SECTION_LOAD_COLUMNS, rows)
        load_descriptors.append({"key": key, "path": path.relative_to(staging).as_posix(),
                                "file": path.relative_to(staging).as_posix(),
                                "row_count": len(rows), "sha256": _sha256(path),
                                "size_bytes": path.stat().st_size})
    load_connection.close()
    load_db_path.unlink(missing_ok=True)
    return {
        "available": bool(contributor_descriptors),
        "schema_version": GRAVITY_VIEWER_ROUTE_SECTION_SCHEMA_VERSION,
        "cache_format_version": GRAVITY_VIEWER_CACHE_FORMAT_VERSION,
        "interval_semantics": "OD departure-time interval; not vehicle passage time at the section",
        "contributors": {"partition_field": "departure_time_bin", "partitions": contributor_descriptors,
                         "bins": sorted({str(item["key"]) for item in contributor_descriptors}, key=_time_bin_sort_key),
                         "columns": list(_SECTION_CONTRIBUTION_COLUMNS),
                         "index_path": index_path.relative_to(staging).as_posix(),
                         "index_sha256": _sha256(index_path)},
        "section_loads": {"partition_field": "departure_time_bin", "partitions": load_descriptors,
                          "bins": sorted({str(item["key"]) for item in load_descriptors}, key=_time_bin_sort_key),
                          "columns": list(_SECTION_LOAD_COLUMNS)},
        "geometry_path": geometry.get("geometry_path"),
        "catalog_path": geometry.get("catalog_path"),
        "geometry_mode": geometry.get("geometry_mode"),
        "message": (
            "expected OD use of a scheduled section from cached fixed-routing probabilities; "
            "not an enumerated route alternative or an exact transfer-count distribution"
            if flow_semantics_seen == {"service_marginal"}
            else "assigned flow is summed saved path contribution once per traversed section; "
            "transfer legs are passenger-section usages"
        ),
    }


def _write_viewer_partitions(staging: Path) -> dict[str, object]:
    """Create compact per-time-bin indexes for OD, routes, and route sections."""
    od_fields = _csv_fieldnames(staging / "full_od.csv")
    od_descriptors, values = _write_partitioned_table(
        source=staging / "full_od.csv",
        destination=staging / _OD_PARTITION_ROOT,
        partition_field="departure_time_bin",
        fieldnames=od_fields,
    )
    summary: dict[str, object] = {
        "cache_format_version": GRAVITY_VIEWER_CACHE_FORMAT_VERSION,
        "od": {
            "available": True,
            "partition_field": "departure_time_bin",
            "partitions": od_descriptors,
            "origins": sorted(values["origins"]),
            "destinations": sorted(values["destinations"]),
            "bins": sorted({str(item["key"]) for item in od_descriptors}, key=_time_bin_sort_key),
        },
        "route_flows": {
            "available": False,
            "message": "No post-fit route-flow export was supplied.",
        },
        "route_sections": {"available": False, "schema_version": GRAVITY_VIEWER_ROUTE_SECTION_SCHEMA_VERSION,
                           "message": "route-section data unavailable until an explicit route export is supplied."},
    }
    route_source = staging / "route_flows.csv"
    if route_source.is_file():
        route_descriptors, route_values = _write_partitioned_table(
            source=route_source,
            destination=staging / _ROUTE_FLOW_PARTITION_ROOT,
            partition_field="departure_time_bin",
            fieldnames=_ROUTE_FLOW_COLUMNS,
            row_transform=_route_flow_row,
        )
        summary["route_flows"] = {
            "available": bool(route_descriptors),
            "partition_field": "departure_time_bin",
            "partitions": route_descriptors,
            "bins": sorted({str(item["key"]) for item in route_descriptors}, key=_time_bin_sort_key),
            "lines": sorted(route_values["lines"]),
            "trips": sorted(route_values["trips"]),
            "origins": sorted(route_values["origins"]),
            "destinations": sorted(route_values["destinations"]),
            "flow_semantics": sorted(route_values["semantics"]),
            "transfer_counts": sorted(route_values["transfers"], key=int),
            "patterns": sorted(route_values["patterns"]),
            "directions": sorted(route_values["directions"]),
            "transfer_count_definition": (
                "realized route-assignment transfer_count; absent values mean the "
                "routing artifact did not provide realized transfer counts"
            ),
        }
    geometry = _write_line_geometry(staging)
    summary["line_geometry"] = geometry
    summary["route_sections"] = _write_route_section_products(staging, geometry)
    _write_json(staging / _VIEWER_SUMMARY_FILE, summary)
    return summary


def _csv_number(value: object) -> float | None:
    """Parse an optional finite number used by display-only summaries."""
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _csv_number_text(value: float | None) -> str:
    """Return a deterministic CSV representation for an optional number."""
    return "" if value is None else format(value, ".17g")


def _observed_od_values(
    full_od_rows: Sequence[Mapping[str, str]],
    report_payload: Mapping[str, object],
) -> list[float | None]:
    """Return optional observed OD values without pretending counts are OD data.

    The core reports normally contain only modeled OD demand.  A case-specific
    report may additionally persist ``observed_demand`` (or ``observed``) in
    ``full_od.csv`` or an aligned ``observed_od_demand`` array in ``report.json``.
    When neither is present, the display field remains empty rather than
    substituting modeled or prior demand.
    """
    report_values = report_payload.get("observed_od_demand")
    if isinstance(report_values, Sequence) and not isinstance(report_values, (str, bytes)):
        if len(report_values) == len(full_od_rows):
            return [_csv_number(value) for value in report_values]
    values: list[float | None] = []
    for row in full_od_rows:
        value = row.get("observed_demand")
        if value is None or str(value).strip() == "":
            value = row.get("observed")
        values.append(_csv_number(value))
    return values


def _fixed_free_status(value: object) -> str:
    """Normalize report cell status to the viewer's fixed/free vocabulary."""
    status = str(value or "").strip()
    if status.lower().startswith("fixed"):
        return "fixed"
    if status.lower() == "free" or status.lower().endswith("_free"):
        return "free"
    return status


def _write_od_map(
    destination: Path,
    *,
    full_od_rows: Iterable[Mapping[str, str]],
    report_payload: Mapping[str, object],
) -> int:
    """Write the optional display-oriented OD aggregation."""
    rows_iter = iter(full_od_rows)
    first = next(rows_iter, None)
    required = {"origin_stop_id", "destination_stop_id", "departure_time_bin", "estimated_demand"}
    if first is None or not required.issubset(first.keys()):
        return 0
    report_values = report_payload.get("observed_od_demand")
    report_values_available = isinstance(report_values, Sequence) and not isinstance(
        report_values, (str, bytes)
    )
    def all_rows() -> Iterator[Mapping[str, str]]:
        yield first
        yield from rows_iter

    temporary = destination.with_name(f".{destination.name}.tmp")
    count = 0
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(_OD_MAP_COLUMNS), extrasaction="raise")
        writer.writeheader()
        for index, source in enumerate(all_rows()):
            origin = str(source.get("origin_stop_id", ""))
            destination_id = str(source.get("destination_stop_id", ""))
            if not origin or not destination_id:
                continue
            modeled = _csv_number(source.get("estimated_demand"))
            if modeled is None:
                continue
            observed = (
                _csv_number(report_values[index])
                if report_values_available and index < len(report_values)
                else _csv_number(source.get("observed_demand") or source.get("observed"))
            )
            residual = None if observed is None else observed - modeled
            writer.writerow(
                {
                    "origin_stop_id": origin,
                    "destination_stop_id": destination_id,
                    "departure_time_bin": str(source.get("departure_time_bin", "")),
                    "observed_demand": _csv_number_text(observed),
                    "modeled_demand": _csv_number_text(modeled),
                    "residual": _csv_number_text(residual),
                    "fixed_free_status": _fixed_free_status(source.get("cell_status")),
                    "identifiability_class": str(source.get("identifiability_class", "")),
                }
            )
            count += 1
    if not count:
        temporary.unlink(missing_ok=True)
        return 0
    temporary.replace(destination)
    return count

def _write_stop_summary(
    destination: Path,
    *,
    predicted_fields: Sequence[str],
    predicted_rows: Iterable[Mapping[str, str]],
    network_stops: Iterable[Mapping[str, str]] = (),
) -> int:
    """Write display-oriented observed/modelled stop totals."""
    stop_field = "stop" if "stop" in predicted_fields else "stop_id" if "stop_id" in predicted_fields else None
    if stop_field is None:
        return 0
    totals: dict[str, dict[str, float]] = {}
    for row in predicted_rows:
        stop = str(row.get(stop_field, "")).strip()
        if not stop:
            continue
        observed = _csv_number(row.get("observed"))
        modeled = _csv_number(row.get("modeled"))
        if observed is None or modeled is None:
            continue
        item = totals.setdefault(
            stop,
            {"observed_total": 0.0, "modeled_total": 0.0, "incoming_total": 0.0, "outgoing_total": 0.0},
        )
        item["observed_total"] += observed
        item["modeled_total"] += modeled
        measurement_type = str(row.get("measurement_type", "")).strip().lower()
        if measurement_type in {"alighting", "incoming"}:
            item["incoming_total"] += observed
        elif measurement_type in {"boarding", "outgoing"}:
            item["outgoing_total"] += observed
    for stop in network_stops:
        stop_id = str(stop.get("stop_id", "")).strip()
        if stop_id:
            totals.setdefault(
                stop_id,
                {"observed_total": 0.0, "modeled_total": 0.0, "incoming_total": 0.0, "outgoing_total": 0.0},
            )
    if not totals:
        return 0
    rows = []
    for stop_id in sorted(totals):
        item = totals[stop_id]
        rows.append(
            {
                "stop_id": stop_id,
                "observed_total": _csv_number_text(item["observed_total"]),
                "modeled_total": _csv_number_text(item["modeled_total"]),
                "residual": _csv_number_text(item["observed_total"] - item["modeled_total"]),
                "incoming_total": _csv_number_text(item["incoming_total"]),
                "outgoing_total": _csv_number_text(item["outgoing_total"]),
            }
        )
    _write_csv(destination, list(_STOP_SUMMARY_COLUMNS), rows)
    return len(rows)


def _require_columns(path: Path, fieldnames: Sequence[str], required: Sequence[str]) -> None:
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}.")


def _float_column(rows: Sequence[Mapping[str, str]], name: str, path: Path) -> np.ndarray:
    values: list[float] = []
    for row_number, row in enumerate(rows, start=2):
        raw = row.get(name, "")
        if raw == "":
            values.append(np.nan)
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{row_number} has a non-numeric {name!r}.") from error
    return np.asarray(values, dtype=np.float64)


def _integer_column(rows: Sequence[Mapping[str, str]], name: str, path: Path) -> np.ndarray:
    values = _float_column(rows, name, path)
    if np.any(~np.isfinite(values)) or np.any(values != np.floor(values)):
        raise ValueError(f"{path} column {name!r} must contain finite integers.")
    return values.astype(np.int64)


def _nonempty_values(rows: Iterable[Mapping[str, str]], name: str) -> set[str]:
    return {str(row.get(name, "")) for row in rows if str(row.get(name, "")) != ""}


def _first_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str | None:
    return next((name for name in candidates if name in fieldnames), None)


def _report_path(report: GravityDetailedReport, name: str) -> Path:
    try:
        candidate = report.files[name]
    except (KeyError, TypeError):
        candidate = report.output_directory / name
    path = Path(candidate).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"gravity report file is missing: {path}")
    return path


def _result_specification(
    fit_result: GravityEstimationResult,
    explicit: GravityModelSpecification | Mapping[str, object] | None,
) -> tuple[dict[str, object], str]:
    fit_raw = fit_result.model_specification
    raw = explicit if explicit is not None else fit_raw
    if raw is None:
        raise ValueError(
            "fit_result does not contain an exact model specification; supply "
            "model_specification explicitly before exporting a viewer bundle."
        )
    if isinstance(raw, GravityModelSpecification):
        specification = raw.to_dict()
        computed_fingerprint = raw.fingerprint
    elif isinstance(raw, Mapping):
        payload = dict(raw)
        try:
            parsed = GravityModelSpecification.from_dict(payload)
        except (TypeError, ValueError) as error:
            raise ValueError("model_specification is not a valid gravity specification.") from error
        specification = parsed.to_dict()
        if canonical_json(payload) != canonical_json(specification):
            raise ValueError(
                "model_specification is not the canonical serialized specification."
            )
        computed_fingerprint = parsed.fingerprint
    else:
        raise TypeError("model_specification must be a GravityModelSpecification or mapping.")
    if explicit is not None and fit_raw is not None:
        if isinstance(fit_raw, GravityModelSpecification):
            fit_specification = fit_raw.to_dict()
            fit_fingerprint = fit_raw.fingerprint
        elif isinstance(fit_raw, Mapping):
            fit_payload = dict(fit_raw)
            try:
                fit_parsed = GravityModelSpecification.from_dict(fit_payload)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "fit_result model_specification is not a valid gravity specification."
                ) from error
            fit_specification = fit_parsed.to_dict()
            if canonical_json(fit_payload) != canonical_json(fit_specification):
                raise ValueError(
                    "fit_result model_specification is not the canonical serialized specification."
                )
            fit_fingerprint = fit_parsed.fingerprint
        else:
            raise TypeError(
                "fit_result model_specification must be a GravityModelSpecification or mapping."
            )
        if fit_specification != specification or fit_fingerprint != computed_fingerprint:
            raise ValueError(
                "explicit model_specification differs from the fitted result."
            )
    stored = str(fit_result.specification_fingerprint or "")
    if stored and stored != computed_fingerprint:
        raise ValueError(
            "model specification fingerprint differs from the fitted result: "
            f"result={stored!r}, specification={computed_fingerprint!r}."
        )
    return specification, stored or computed_fingerprint


def _source_provenance(
    report_payload: Mapping[str, object],
    fit_result: GravityEstimationResult,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    report_provenance = report_payload.get("provenance", {})
    if not isinstance(report_provenance, Mapping):
        raise ValueError("report.json provenance must be a mapping.")
    metadata_provenance = metadata.get("provenance", {})
    if not isinstance(metadata_provenance, Mapping):
        raise ValueError("bundle metadata provenance must be a mapping.")
    result_fallbacks: dict[str, object] = {
        "model_fingerprint": fit_result.model_fingerprint,
        "specification_fingerprint": fit_result.specification_fingerprint or None,
        "artifact_identity_fingerprint": (
            fit_result.direct_operator_artifact_fingerprint or None
        ),
    }
    result: dict[str, object] = {}
    for field in _CANONICAL_PROVENANCE:
        result[field] = report_provenance.get(
            field, metadata_provenance.get(field, None)
        )
    for field, value in result_fallbacks.items():
        if value is not None:
            result[field] = value
    result["model_fingerprint"] = fit_result.model_fingerprint
    result["specification_fingerprint"] = fit_result.specification_fingerprint or result.get(
        "specification_fingerprint"
    )
    return result


def _validation_provenance(validation_result: object) -> dict[str, object]:
    if isinstance(validation_result, GravityAdequacyReport):
        return {
            "model_fingerprint": validation_result.model_fingerprint,
            "specification_fingerprint": validation_result.specification_fingerprint,
            "model_specification": validation_result.model_specification,
            "report_fingerprint": validation_result.report_fingerprint,
        }
    if isinstance(validation_result, Mapping):
        adequacy = validation_result.get("adequacy")
        if isinstance(adequacy, Mapping):
            merged = dict(validation_result)
            merged.update(adequacy)
            validation_result = merged
        provenance = validation_result.get("provenance", {})
        if provenance is None:
            provenance = {}
        if not isinstance(provenance, Mapping):
            raise ValueError("validation_result provenance must be a mapping.")
        provenance = dict(provenance)
        for name in (
            "model_fingerprint",
            "specification_fingerprint",
            "model_specification",
            "report_fingerprint",
        ):
            if name in validation_result:
                provenance[name] = validation_result[name]
        return dict(provenance)
    values: dict[str, object] = {}
    for name in (
        "model_fingerprint",
        "specification_fingerprint",
        "model_specification",
        "report_fingerprint",
    ):
        value = getattr(validation_result, name, None)
        if value is not None:
            values[name] = value
    return values


def _validate_table_inputs(
    staging: Path,
    fit_result: GravityEstimationResult,
    report_payload: Mapping[str, object],
    validation_result: object,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    fields, full_od = _read_csv(staging / "full_od.csv")
    _require_columns(staging / "full_od.csv", fields, _FULL_OD_COLUMNS)
    od_indices = _integer_column(full_od, "od_index", staging / "full_od.csv")
    if len(np.unique(od_indices)) != len(od_indices):
        raise ValueError("full_od.csv contains duplicate od_index values.")
    estimated = _float_column(full_od, "estimated_demand", staging / "full_od.csv")
    if estimated.size != fit_result.full_od_demand.size:
        raise ValueError("full_od.csv row count does not match fitted full OD demand.")
    np.testing.assert_allclose(estimated, np.asarray(fit_result.full_od_demand), rtol=1e-8, atol=1e-8)
    counts["full_od.csv"] = len(full_od)

    fields, predictions = _read_csv(staging / "predicted_measurements.csv")
    _require_columns(staging / "predicted_measurements.csv", fields, _PREDICTED_COLUMNS)
    row_indices = _integer_column(
        predictions, "row_index", staging / "predicted_measurements.csv"
    )
    if len(np.unique(row_indices)) != len(row_indices):
        raise ValueError("predicted_measurements.csv contains duplicate row_index values.")
    modeled = _float_column(predictions, "modeled", staging / "predicted_measurements.csv")
    if modeled.shape != np.asarray(fit_result.predicted_measurements).shape:
        raise ValueError("predicted_measurements.csv length does not match fitted predictions.")
    np.testing.assert_allclose(
        modeled, np.asarray(fit_result.predicted_measurements), rtol=1e-8, atol=1e-8
    )
    counts["predicted_measurements.csv"] = len(predictions)

    fields, residuals = _read_csv(staging / "residuals.csv")
    _require_columns(staging / "residuals.csv", fields, _RESIDUAL_COLUMNS)
    residual_row_indices = _integer_column(residuals, "row_index", staging / "residuals.csv")
    if not np.array_equal(residual_row_indices, row_indices):
        raise ValueError("residuals.csv row_index values differ from predicted_measurements.csv.")
    residual_values = _float_column(residuals, "residual", staging / "residuals.csv")
    observed = _float_column(residuals, "observed", staging / "residuals.csv")
    residual_modeled = _float_column(residuals, "modeled", staging / "residuals.csv")
    np.testing.assert_allclose(residual_values, observed - residual_modeled, rtol=1e-8, atol=1e-8)
    counts["residuals.csv"] = len(residuals)

    fields, grouped = _read_csv(staging / "grouped_residuals.csv")
    if not grouped:
        raise ValueError("grouped_residuals.csv must contain its header and at least one row.")
    counts["grouped_residuals.csv"] = len(grouped)

    fields, parameters = _read_csv(staging / "parameters.csv")
    _require_columns(staging / "parameters.csv", fields, _PARAMETER_COLUMNS)
    parameter_indices = _integer_column(parameters, "parameter_index", staging / "parameters.csv")
    if not np.array_equal(parameter_indices, np.arange(parameter_indices.size)):
        raise ValueError("parameters.csv parameter_index values must be consecutive.")
    raw = _float_column(parameters, "raw_value", staging / "parameters.csv")
    physical = _float_column(parameters, "physical_value", staging / "parameters.csv")
    np.testing.assert_allclose(raw, np.asarray(fit_result.raw_parameters), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        physical, np.asarray(fit_result.physical_parameters), rtol=1e-8, atol=1e-8
    )
    counts["parameters.csv"] = len(parameters)

    report_model = report_payload.get("model_fingerprint")
    if report_model != fit_result.model_fingerprint:
        raise ValueError(
            "report.json model_fingerprint differs from the fitted result: "
            f"report={report_model!r}, fit={fit_result.model_fingerprint!r}."
        )
    validation_provenance = _validation_provenance(validation_result)
    validation_model = validation_provenance.get("model_fingerprint")
    if validation_model is not None and validation_model != fit_result.model_fingerprint:
        raise ValueError(
            "validation provenance model_fingerprint differs from the fitted result: "
            f"validation={validation_model!r}, fit={fit_result.model_fingerprint!r}."
        )
    fit_specification, fit_specification_fingerprint = _result_specification(
        fit_result, None
    )
    report_specification = report_payload.get("model_specification")
    if not isinstance(report_specification, Mapping):
        raise ValueError("report.json is missing model_specification.")
    try:
        parsed_report_specification = GravityModelSpecification.from_dict(
            dict(report_specification)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("report.json model_specification is invalid.") from error
    if canonical_json(dict(report_specification)) != canonical_json(
        parsed_report_specification.to_dict()
    ):
        raise ValueError("report.json model_specification is not canonical.")
    report_specification_fingerprint = report_payload.get(
        "specification_fingerprint"
    )
    if report_specification_fingerprint != parsed_report_specification.fingerprint:
        raise ValueError(
            "report.json specification_fingerprint does not match its "
            "serialized model_specification."
        )
    if (
        report_specification_fingerprint != fit_specification_fingerprint
        or canonical_json(dict(report_specification))
        != canonical_json(fit_specification)
    ):
        raise ValueError(
            "report.json model specification differs from the fitted result."
        )
    validation_specification = validation_provenance.get("model_specification")
    validation_specification_fingerprint = validation_provenance.get(
        "specification_fingerprint"
    )
    if not isinstance(validation_specification, Mapping):
        raise ValueError("validation provenance is missing model_specification.")
    if validation_specification_fingerprint != fit_specification_fingerprint:
        raise ValueError(
            "validation provenance specification_fingerprint differs from the "
            "fitted result."
        )
    if canonical_json(dict(validation_specification)) != canonical_json(
        fit_specification
    ):
        raise ValueError(
            "validation provenance model_specification differs from the fitted result."
        )
    return counts


def _validate_network(
    staging: Path,
    table_counts: Mapping[str, int],
) -> dict[str, int]:
    network_dir = staging / "network"
    if not network_dir.exists():
        return {}
    files = sorted(path.name for path in network_dir.iterdir() if path.is_file())
    unknown = [name for name in files if name not in _NETWORK_FILES]
    if unknown:
        raise ValueError(f"unsupported network snapshot files: {', '.join(unknown)}.")
    if "stops.csv" not in files:
        raise ValueError("network snapshot must include stops.csv.")
    stop_fields = _csv_fieldnames(network_dir / "stops.csv")
    stops = list(_iter_csv_rows(network_dir / "stops.csv"))
    _require_columns(network_dir / "stops.csv", stop_fields, ("stop_id", "name", "lat", "lon"))
    stop_ids = [str(row["stop_id"]) for row in stops]
    if len(set(stop_ids)) != len(stop_ids):
        raise ValueError("network/stops.csv contains duplicate stop_id values.")
    lat = _float_column(stops, "lat", network_dir / "stops.csv")
    lon = _float_column(stops, "lon", network_dir / "stops.csv")
    if (
        np.any(~np.isfinite(lat))
        or np.any(~np.isfinite(lon))
        or np.any((lat < -90) | (lat > 90))
        or np.any((lon < -180) | (lon > 180))
    ):
        raise ValueError("network/stops.csv contains invalid latitude/longitude values.")
    stop_set = set(stop_ids)

    report_stops = _nonempty_values(
        _iter_csv_rows(staging / "full_od.csv"), "origin_stop_id"
    ) | _nonempty_values(_iter_csv_rows(staging / "full_od.csv"), "destination_stop_id")
    report_stops |= _nonempty_values(
        _iter_csv_rows(staging / "predicted_measurements.csv"), "stop"
    )
    missing_stops = sorted(report_stops - stop_set)
    if missing_stops:
        raise ValueError(
            "report references stop IDs absent from network/stops.csv: "
            + ", ".join(missing_stops[:10])
        )

    counts = {"network/stops.csv": len(stops)}
    line_ids: set[str] = set()
    if "lines.csv" in files:
        fields = _csv_fieldnames(network_dir / "lines.csv")
        line_column = _first_column(fields, ("line_id", "route_id", "line", "id"))
        if line_column is None:
            raise ValueError("network/lines.csv must contain line_id or route_id.")
        line_ids = _nonempty_values(_iter_csv_rows(network_dir / "lines.csv"), line_column)
        line_count = _csv_row_count(network_dir / "lines.csv")
        if len(line_ids) != line_count:
            raise ValueError("network/lines.csv contains duplicate or empty line identifiers.")
        counts["network/lines.csv"] = line_count
        measurement_lines = _nonempty_values(
            _iter_csv_rows(staging / "predicted_measurements.csv"), "line"
        )
        missing_lines = sorted(measurement_lines - line_ids)
        if missing_lines:
            raise ValueError(
                "predicted measurements reference line IDs absent from network/lines.csv: "
                + ", ".join(missing_lines[:10])
            )
    trip_ids: set[str] = set()
    if "trips.csv" in files:
        fields = _csv_fieldnames(network_dir / "trips.csv")
        trip_column = _first_column(fields, ("trip_id", "trip", "id"))
        if trip_column is None:
            raise ValueError("network/trips.csv must contain trip_id.")
        trip_ids = _nonempty_values(_iter_csv_rows(network_dir / "trips.csv"), trip_column)
        trip_count = _csv_row_count(network_dir / "trips.csv")
        if len(trip_ids) != trip_count:
            raise ValueError("network/trips.csv contains duplicate or empty trip identifiers.")
        if line_ids:
            line_column = _first_column(fields, ("line_id", "route_id", "line"))
            if line_column is not None:
                missing = sorted(
                    _nonempty_values(_iter_csv_rows(network_dir / "trips.csv"), line_column)
                    - line_ids
                )
                if missing:
                    raise ValueError(
                        "network/trips.csv references line IDs absent from lines.csv: "
                        + ", ".join(missing[:10])
                    )
        counts["network/trips.csv"] = trip_count
    if "stop_times.csv" in files:
        fields = _csv_fieldnames(network_dir / "stop_times.csv")
        trip_column = _first_column(fields, ("trip_id", "trip"))
        stop_column = _first_column(fields, ("stop_id", "stop"))
        if trip_column is None or stop_column is None:
            raise ValueError("network/stop_times.csv must contain trip_id and stop_id.")
        if trip_ids:
            missing = sorted(
                _nonempty_values(_iter_csv_rows(network_dir / "stop_times.csv"), trip_column)
                - trip_ids
            )
            if missing:
                raise ValueError(
                    "network/stop_times.csv references unknown trip IDs: "
                    + ", ".join(missing[:10])
                )
        missing = sorted(
            _nonempty_values(_iter_csv_rows(network_dir / "stop_times.csv"), stop_column)
            - stop_set
        )
        if missing:
            raise ValueError(
                "network/stop_times.csv references unknown stop IDs: "
                + ", ".join(missing[:10])
            )
        counts["network/stop_times.csv"] = _csv_row_count(network_dir / "stop_times.csv")
    return counts


def _validate_route_flow_network(root: Path, summary: Mapping[str, object]) -> None:
    """Validate sparse route references against the optional network snapshot."""
    network_dir = root / "network"
    route_summary = summary.get("route_flows", {})
    if not network_dir.is_dir() or not isinstance(route_summary, Mapping) or not route_summary.get("available"):
        return
    line_ids: set[str] = set()
    trip_ids: set[str] = set()
    stop_ids: set[str] = set()
    if (network_dir / "stops.csv").is_file():
        stop_ids = _nonempty_values(_iter_csv_rows(network_dir / "stops.csv"), "stop_id")
    if (network_dir / "lines.csv").is_file():
        fields = _csv_fieldnames(network_dir / "lines.csv")
        line_column = _first_column(fields, ("line_id", "route_id", "line", "id"))
        if line_column is not None:
            line_ids = _nonempty_values(_iter_csv_rows(network_dir / "lines.csv"), line_column)
    if (network_dir / "trips.csv").is_file():
        fields = _csv_fieldnames(network_dir / "trips.csv")
        trip_column = _first_column(fields, ("trip_id", "trip", "id"))
        if trip_column is not None:
            trip_ids = _nonempty_values(_iter_csv_rows(network_dir / "trips.csv"), trip_column)
    for descriptor in route_summary.get("partitions", ()):
        if not isinstance(descriptor, Mapping):
            continue
        path = root / str(descriptor.get("path") or descriptor.get("file"))
        for row in _iter_csv_rows(path):
            for field, known, label in (
                ("origin_stop_id", stop_ids, "origin"),
                ("destination_stop_id", stop_ids, "destination"),
                ("boarding_stop_id", stop_ids, "boarding"),
                ("alighting_stop_id", stop_ids, "alighting"),
                ("line_id", line_ids, "line"),
                ("trip_id", trip_ids, "trip"),
            ):
                value = str(row.get(field, ""))
                if value and known and value not in known:
                    raise ValueError(f"route-flow {label} ID {value!r} is absent from the network snapshot.")


def _file_records(root: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "bundle_manifest.json":
            continue
        record: dict[str, object] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                record["row_count"] = max(0, sum(1 for _ in stream) - 1)
        records[relative] = record
    return records


def _validate_bundle_directory(root: Path, manifest: Mapping[str, object]) -> None:
    if manifest.get("schema_version") != GRAVITY_VIEWER_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported gravity viewer bundle schema version.")
    if manifest.get("bundle_type") != GRAVITY_VIEWER_BUNDLE_TYPE:
        raise ValueError("unexpected gravity viewer bundle type.")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("bundle manifest files are missing or invalid.")
    required = (*_REPORT_FILES, *_TABLE_FILES, _MODEL_SPECIFICATION_FILE)
    for name in required:
        if name not in files:
            raise ValueError(f"bundle manifest is missing required file {name!r}.")
    for name, record in files.items():
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise ValueError("bundle manifest file records are invalid.")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"bundle manifest contains unsafe file path {name!r}.")
        path = root / relative
        if not path.is_file():
            raise ValueError(f"bundle file is missing: {name}")
        expected_size = record.get("size_bytes")
        if expected_size != path.stat().st_size:
            raise ValueError(
                f"bundle file size mismatch for {name!r}: "
                f"manifest={expected_size!r}, actual={path.stat().st_size!r}."
            )
        expected_digest = record.get("sha256")
        actual_digest = _sha256(path)
        if expected_digest != actual_digest:
            raise ValueError(
                f"bundle file checksum mismatch for {name!r}: "
                f"manifest={expected_digest!r}, actual={actual_digest!r}."
            )
        if "row_count" in record and path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                actual_rows = max(0, sum(1 for _ in stream) - 1)
            if record.get("row_count") != actual_rows:
                raise ValueError(
                    f"bundle file row count mismatch for {name!r}: "
                    f"manifest={record.get('row_count')!r}, actual={actual_rows!r}."
                )
    report_payload = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if not isinstance(report_payload, Mapping):
        raise ValueError("bundle report.json must contain an object.")
    spec_payload = json.loads((root / _MODEL_SPECIFICATION_FILE).read_text(encoding="utf-8"))
    if not isinstance(spec_payload, Mapping) or spec_payload.get("artifact_type") != "gravity_model_specification":
        raise ValueError("bundle model_specification.json has an unexpected artifact type.")
    specification = spec_payload.get("specification")
    if not isinstance(specification, Mapping):
        raise ValueError("bundle model specification is missing.")
    try:
        parsed_spec = GravityModelSpecification.from_dict(dict(specification))
    except (TypeError, ValueError) as error:
        raise ValueError("bundle model specification is invalid.") from error
    spec_fingerprint = parsed_spec.fingerprint
    if spec_payload.get("specification_fingerprint") != spec_fingerprint:
        raise ValueError("bundle model specification fingerprint mismatch.")
    manifest_spec = manifest.get("model_specification")
    if not isinstance(manifest_spec, Mapping) or manifest_spec.get("fingerprint") != spec_fingerprint:
        raise ValueError("bundle manifest model specification provenance mismatch.")
    if manifest.get("specification_fingerprint") != spec_fingerprint:
        raise ValueError("bundle specification_fingerprint does not match model specification.")
    report_specification = report_payload.get("model_specification")
    if not isinstance(report_specification, Mapping):
        raise ValueError("bundle report.json is missing model_specification.")
    try:
        parsed_report_specification = GravityModelSpecification.from_dict(
            dict(report_specification)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("bundle report model specification is invalid.") from error
    if canonical_json(dict(report_specification)) != canonical_json(
        parsed_report_specification.to_dict()
    ):
        raise ValueError("bundle report model specification is not canonical.")
    if report_payload.get("specification_fingerprint") != spec_fingerprint:
        raise ValueError("bundle report specification provenance mismatch.")
    if parsed_report_specification.fingerprint != spec_fingerprint:
        raise ValueError("bundle report model specification fingerprint mismatch.")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("bundle provenance is missing or invalid.")
    missing_provenance = [
        field for field in _CANONICAL_PROVENANCE if field not in provenance
    ]
    if missing_provenance:
        raise ValueError(
            "bundle provenance is missing fields: "
            + ", ".join(missing_provenance)
        )
    report_model = report_payload.get("model_fingerprint")
    if report_model != provenance.get("model_fingerprint"):
        raise ValueError(
            "bundle report/model provenance mismatch: "
            f"report={report_model!r}, bundle={provenance.get('model_fingerprint')!r}."
        )
    report_provenance = manifest.get("report_provenance", {})
    if not isinstance(report_provenance, Mapping):
        raise ValueError("bundle report provenance is missing or invalid.")
    if (
        report_provenance.get("report_fingerprint") is not None
        and report_payload.get("report_fingerprint") != report_provenance.get("report_fingerprint")
    ):
        raise ValueError(
            "bundle report_fingerprint mismatch: "
            f"report={report_payload.get('report_fingerprint')!r}, "
            f"manifest={report_provenance.get('report_fingerprint')!r}."
        )
    fit_provenance = manifest.get("fit_provenance", {})
    if not isinstance(fit_provenance, Mapping):
        raise ValueError("bundle fit provenance is missing or invalid.")
    if fit_provenance.get("model_fingerprint") != provenance.get("model_fingerprint"):
        raise ValueError(
            "bundle fit/model provenance mismatch: "
            f"fit={fit_provenance.get('model_fingerprint')!r}, "
            f"bundle={provenance.get('model_fingerprint')!r}."
        )
    if fit_provenance.get("specification_fingerprint") != spec_fingerprint:
        raise ValueError(
            "bundle fit/specification provenance mismatch: "
            f"fit={fit_provenance.get('specification_fingerprint')!r}, "
            f"specification={spec_fingerprint!r}."
        )
    validation_provenance = manifest.get("validation_provenance", {})
    if not isinstance(validation_provenance, Mapping):
        raise ValueError("bundle validation provenance is missing or invalid.")
    validation_model = validation_provenance.get("model_fingerprint")
    if validation_model is not None and validation_model != provenance.get("model_fingerprint"):
        raise ValueError(
            "bundle validation/model provenance mismatch: "
            f"validation={validation_model!r}, bundle={provenance.get('model_fingerprint')!r}."
        )
    if validation_provenance.get("specification_fingerprint") != spec_fingerprint:
        raise ValueError(
            "bundle validation/specification provenance mismatch: "
            f"validation={validation_provenance.get('specification_fingerprint')!r}, "
            f"specification={spec_fingerprint!r}."
        )
    validation_specification = validation_provenance.get("model_specification")
    if not isinstance(validation_specification, Mapping) or dict(
        validation_specification
    ) != dict(specification):
        raise ValueError("bundle validation model specification provenance mismatch.")
    report_model_provenance = report_provenance.get("model_fingerprint")
    if report_model_provenance is not None and report_model_provenance != provenance.get("model_fingerprint"):
        raise ValueError(
            "bundle report/model provenance mismatch: "
            f"report={report_model_provenance!r}, bundle={provenance.get('model_fingerprint')!r}."
        )
    if report_provenance.get("specification_fingerprint") != spec_fingerprint:
        raise ValueError("bundle report/specification provenance mismatch.")
    report_specification_provenance = report_provenance.get("model_specification")
    if not isinstance(report_specification_provenance, Mapping) or dict(
        report_specification_provenance
    ) != dict(specification):
        raise ValueError("bundle report model specification provenance mismatch.")
    counts = manifest.get("row_counts", {})
    if not isinstance(counts, Mapping):
        raise ValueError("bundle row_counts are missing or invalid.")
    for name in _TABLE_FILES:
        fields = _csv_fieldnames(root / name)
        actual_rows = _csv_row_count(root / name)
        expected = counts.get(name)
        if expected != actual_rows:
            raise ValueError(
                f"bundle row count mismatch for {name!r}: manifest={expected!r}, actual={actual_rows!r}."
            )
        if name == "full_od.csv":
            _require_columns(root / name, fields, _FULL_OD_COLUMNS)
        elif name == "predicted_measurements.csv":
            _require_columns(root / name, fields, _PREDICTED_COLUMNS)
        elif name == "residuals.csv":
            _require_columns(root / name, fields, _RESIDUAL_COLUMNS)
        elif name == "parameters.csv":
            _require_columns(root / name, fields, _PARAMETER_COLUMNS)
    for name, columns in (
        ("od_map.csv", _OD_MAP_COLUMNS),
        ("stop_summary.csv", _STOP_SUMMARY_COLUMNS),
    ):
        if not (root / name).is_file():
            continue
        fields = _csv_fieldnames(root / name)
        actual_rows = _csv_row_count(root / name)
        expected = counts.get(name)
        if expected != actual_rows:
            raise ValueError(
                f"bundle row count mismatch for {name!r}: "
                f"manifest={expected!r}, actual={actual_rows!r}."
            )
        _require_columns(root / name, fields, columns)
    summary_path = root / _VIEWER_SUMMARY_FILE
    if summary_path.is_file():
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary_payload, Mapping):
            raise ValueError("viewer_summary.json must contain an object.")
        if summary_payload.get("cache_format_version") not in (1, GRAVITY_VIEWER_CACHE_FORMAT_VERSION):
            raise ValueError("unsupported viewer cache format version.")
        for kind, expected_columns in (("od", _FULL_OD_COLUMNS), ("route_flows", _ROUTE_FLOW_REQUIRED_COLUMNS)):
            kind_payload = summary_payload.get(kind, {})
            if not isinstance(kind_payload, Mapping) or not kind_payload.get("available"):
                continue
            partitions = kind_payload.get("partitions", ())
            if not isinstance(partitions, Sequence) or isinstance(partitions, (str, bytes)):
                raise ValueError(f"viewer summary {kind!r} partitions are invalid.")
            total = 0
            for descriptor in partitions:
                if not isinstance(descriptor, Mapping):
                    raise ValueError(f"viewer summary {kind!r} contains an invalid partition.")
                relative = Path(str(descriptor.get("path") or descriptor.get("file") or ""))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("viewer summary contains an unsafe partition path.")
                path = root / relative
                if not path.is_file():
                    raise ValueError(f"viewer summary partition is missing: {relative}")
                _require_columns(path, _csv_fieldnames(path), expected_columns)
                count = _csv_row_count(path)
                if descriptor.get("row_count") != count:
                    raise ValueError(f"viewer summary row count mismatch for {relative!s}.")
                total += count
            if kind == "od" and total != counts.get("full_od.csv"):
                raise ValueError("viewer OD partition rows do not match full_od.csv.")
        section_payload = summary_payload.get("route_sections", {})
        if isinstance(section_payload, Mapping) and section_payload.get("available"):
            if section_payload.get("schema_version") != GRAVITY_VIEWER_ROUTE_SECTION_SCHEMA_VERSION:
                raise ValueError("unsupported viewer route-section schema version.")
            for group_name, expected_columns in (
                ("contributors", _SECTION_CONTRIBUTION_COLUMNS),
                ("section_loads", _SECTION_LOAD_COLUMNS),
            ):
                group = section_payload.get(group_name, {})
                if not isinstance(group, Mapping):
                    raise ValueError(f"viewer summary route_sections {group_name!r} is invalid.")
                for descriptor in group.get("partitions", ()):
                    relative = Path(str(descriptor.get("path") or descriptor.get("file") or ""))
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("viewer route-section partition path is unsafe.")
                    path = root / relative
                    if not path.is_file():
                        raise ValueError(f"viewer route-section partition is missing: {relative}")
                    _require_columns(path, _csv_fieldnames(path), expected_columns)
                    if descriptor.get("row_count") != _csv_row_count(path):
                        raise ValueError(f"viewer route-section row count mismatch for {relative!s}.")
                    if group_name == "contributors":
                        expected_assignment = manifest.get("assignment_fingerprint")
                        if expected_assignment:
                            for row in _iter_csv_rows(path):
                                supplied = str(row.get("assignment_fingerprint", ""))
                                if supplied and supplied != str(expected_assignment):
                                    raise ValueError("viewer route-section assignment fingerprint mismatch.")
        _validate_route_flow_network(root, summary_payload)
    identifiability = manifest.get("identifiability", {})
    if not isinstance(identifiability, Mapping):
        raise ValueError("bundle identifiability metadata is invalid.")
    if bool(identifiability.get("available")):
        ident_dir = root / "identifiability"
        if not (ident_dir / "identifiability.json").is_file() or not (ident_dir / "identifiability.npz").is_file():
            raise ValueError("bundle identifiability metadata claims an absent artifact.")
        diagnostic = read_gravity_od_identifiability(ident_dir)
        if diagnostic.provenance.get("model_fingerprint") != provenance.get("model_fingerprint"):
            raise ValueError("bundle identifiability/model provenance mismatch.")
    network_counts = _validate_network(root, counts)
    for name, count in network_counts.items():
        if counts.get(name) != count:
            raise ValueError(
                f"bundle row count mismatch for {name!r}: manifest={counts.get(name)!r}, actual={count!r}."
            )


def _write_route_flow_source(
    destination: Path,
    route_flow_rows: Iterable[Mapping[str, object]] | str | Path,
    *,
    expected_assignment_fingerprint: str | None = None,
) -> int:
    """Normalize an externally prepared post-fit route-flow export."""
    if isinstance(route_flow_rows, (str, Path)):
        rows: Iterable[Mapping[str, object]] = _iter_csv_rows(Path(route_flow_rows))
    else:
        rows = route_flow_rows
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(_ROUTE_FLOW_COLUMNS), extrasaction="raise")
        writer.writeheader()
        for raw in rows:
            normalized = _route_flow_row(raw)
            supplied_fingerprint = normalized.get("assignment_fingerprint", "")
            if (
                expected_assignment_fingerprint
                and supplied_fingerprint
                and supplied_fingerprint != expected_assignment_fingerprint
            ):
                raise ValueError(
                    "route-flow assignment fingerprint differs from the fitted bundle identity."
                )
            writer.writerow(normalized)
            count += 1
    temporary.replace(destination)
    return count


@dataclass(frozen=True, slots=True)
class GravityViewerBundle:
    """Validated bundle handle exposed to an independent graphical viewer."""

    path: Path
    manifest: Mapping[str, object]

    def validate(self) -> None:
        """Revalidate checksums, joins, provenance, and row counts."""
        _validate_bundle_directory(self.path, self.manifest)

    @property
    def model_specification(self) -> Mapping[str, object]:
        payload = json.loads(
            (self.path / _MODEL_SPECIFICATION_FILE).read_text(encoding="utf-8")
        )
        specification = payload.get("specification")
        if not isinstance(specification, Mapping):
            raise ValueError("bundle model specification is missing.")
        return specification

    @property
    def table_names(self) -> tuple[str, ...]:
        return (*_TABLE_FILES, *tuple(name for name in _OPTIONAL_DERIVED_FILES if (self.path / name).is_file()))

    def table_path(self, name: str) -> Path:
        if name not in _TABLE_FILES and name not in _OPTIONAL_DERIVED_FILES:
            raise KeyError(f"unknown gravity viewer table {name!r}.")
        path = self.path / name
        if not path.is_file():
            raise FileNotFoundError(f"bundle table is not present: {name}")
        return path

    def network_path(self, name: str) -> Path:
        if name not in _NETWORK_FILES:
            raise KeyError(f"unknown network snapshot file {name!r}.")
        path = self.path / "network" / name
        if not path.is_file():
            raise FileNotFoundError(f"network snapshot file is not present: {name}")
        return path

    @property
    def viewer_summary(self) -> Mapping[str, object]:
        """Return the small partition/index summary, if present."""
        path = self.path / _VIEWER_SUMMARY_FILE
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("viewer_summary.json must contain an object.")
        return payload

    @property
    def route_flow_available(self) -> bool:
        route_summary = self.viewer_summary.get("route_flows", {})
        return isinstance(route_summary, Mapping) and bool(route_summary.get("available"))

    @property
    def transfer_count_definition(self) -> str | None:
        route_summary = self.viewer_summary.get("route_flows", {})
        if not isinstance(route_summary, Mapping):
            return None
        value = route_summary.get("transfer_count_definition")
        return None if value is None else str(value)

    def iter_table(
        self,
        name: str,
        *,
        columns: Sequence[str] | None = None,
        chunksize: int = 100_000,
        filters: Mapping[str, object] | None = None,
    ) -> Iterator[Any]:
        """Yield pandas chunks, optionally filtered without loading the full table."""
        if chunksize <= 0:
            raise ValueError("chunksize must be positive.")
        path = self.table_path(name)
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover - pandas is a core dependency
            raise RuntimeError("pandas is required for chunked viewer access.") from error
        filters = {} if filters is None else dict(filters)
        for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
            for field, expected in filters.items():
                if field not in chunk.columns:
                    raise ValueError(f"table {name!r} has no filter column {field!r}.")
                if isinstance(expected, (set, frozenset, tuple, list)):
                    chunk = chunk[chunk[field].isin(expected)]
                else:
                    chunk = chunk[chunk[field] == expected]
            if not chunk.empty:
                yield chunk

    def iter_partition(
        self,
        partition: str,
        *,
        columns: Sequence[str] | None = None,
        chunksize: int = 100_000,
        filters: Mapping[str, object] | None = None,
    ) -> Iterator[Any]:
        """Yield one indexed partition using bounded pandas chunks."""
        if chunksize <= 0:
            raise ValueError("chunksize must be positive.")
        relative = Path(partition)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("partition path must stay inside the bundle.")
        path = self.path / relative
        if not path.is_file() or not relative.as_posix().startswith(f"{_PARTITION_ROOT}/"):
            raise FileNotFoundError(f"bundle partition is not present: {partition}")
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("pandas is required for chunked viewer access.") from error
        filters = {} if filters is None else dict(filters)
        for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
            for field, expected in filters.items():
                if field not in chunk.columns:
                    raise ValueError(f"partition has no filter column {field!r}.")
                if isinstance(expected, (set, frozenset, tuple, list)):
                    chunk = chunk[chunk[field].isin(expected)]
                else:
                    chunk = chunk[chunk[field] == expected]
            if not chunk.empty:
                yield chunk

    def _partition_descriptors(self, kind: str) -> tuple[Mapping[str, object], ...]:
        summary = self.viewer_summary.get(kind, {})
        if not isinstance(summary, Mapping):
            return ()
        partitions = summary.get("partitions", ())
        if not isinstance(partitions, Sequence) or isinstance(partitions, (str, bytes)):
            return ()
        return tuple(item for item in partitions if isinstance(item, Mapping))

    def iter_route_flows(
        self,
        *,
        departure_time_bin: str | None = None,
        origin: str | None = None,
        destination: str | None = None,
        line: str | None = None,
        trip: str | None = None,
        transfer_count: int | None = None,
        chunksize: int = 100_000,
    ) -> Iterator[Any]:
        """Stream sparse post-fit route flows in either query direction."""
        descriptors = self._partition_descriptors("route_flows")
        filters: dict[str, object] = {}
        if origin is not None:
            filters["origin_stop_id"] = origin
        if destination is not None:
            filters["destination_stop_id"] = destination
        if line is not None:
            filters["line_id"] = line
        if trip is not None:
            filters["trip_id"] = trip
        if transfer_count is not None:
            filters["transfer_count"] = transfer_count
        for descriptor in descriptors:
            if departure_time_bin is not None and descriptor.get("key") != departure_time_bin:
                continue
            path = descriptor.get("path") or descriptor.get("file")
            if path is None:
                continue
            yield from self.iter_partition(
                str(path),
                columns=_ROUTE_FLOW_COLUMNS,
                chunksize=chunksize,
                filters=filters,
            )

    def query_route_flows(
        self,
        *,
        departure_time_bin: str | None = None,
        origin: str | None = None,
        destination: str | None = None,
        line: str | None = None,
        trip: str | None = None,
        transfer_count: int | None = None,
        top_n: int | None = None,
        chunksize: int = 100_000,
    ) -> Any:
        """Return a bounded route-flow query for origin-, destination-, or service-first views."""
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("pandas is required for viewer access.") from error
        chunks: list[Any] = []
        for chunk in self.iter_route_flows(
            departure_time_bin=departure_time_bin,
            origin=origin,
            destination=destination,
            line=line,
            trip=trip,
            transfer_count=transfer_count,
            chunksize=chunksize,
        ):
            if top_n is not None:
                if top_n <= 0:
                    raise ValueError("top_n must be positive when provided.")
                chunk["assigned_demand"] = pd.to_numeric(chunk["assigned_demand"])
                chunk = chunk.nlargest(top_n, "assigned_demand")
            chunks.append(chunk)
        if not chunks:
            return pd.DataFrame(columns=_ROUTE_FLOW_COLUMNS)
        result = pd.concat(chunks, ignore_index=True)
        if top_n is not None:
            result["assigned_demand"] = pd.to_numeric(result["assigned_demand"])
            result = result.nlargest(top_n, "assigned_demand")
        return result

    def aggregate_route_flows(
        self,
        by: str | Sequence[str] = ("line_id",),
        **filters: object,
    ) -> Any:
        """Aggregate expected assigned demand while preserving passenger-leg semantics."""
        group_by = (by,) if isinstance(by, str) else tuple(by)
        if not group_by:
            raise ValueError("at least one route-flow aggregation column is required.")
        totals: dict[tuple[object, ...], float] = {}
        for chunk in self.iter_route_flows(**filters):
            chunk["assigned_demand"] = chunk["assigned_demand"].astype(float)
            grouped = chunk.groupby(list(group_by), dropna=False, sort=False)["assigned_demand"].sum()
            for key, value in grouped.items():
                normalized = key if isinstance(key, tuple) else (key,)
                totals[normalized] = totals.get(normalized, 0.0) + float(value)
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("pandas is required for viewer access.") from error
        rows = [dict(zip(group_by, key, strict=True), assigned_demand=value) for key, value in sorted(totals.items(), key=str)]
        return pd.DataFrame(rows, columns=[*group_by, "assigned_demand"])

    def query_od_routes(
        self,
        *,
        origin: str,
        destination: str,
        departure_time_bin: str | None = None,
        top_n: int | None = None,
    ) -> Any:
        return self.query_route_flows(
            origin=origin,
            destination=destination,
            departure_time_bin=departure_time_bin,
            top_n=top_n,
        )

    def query_service_flows(
        self,
        *,
        line: str | None = None,
        trip: str | None = None,
        departure_time_bin: str | None = None,
        top_n: int | None = None,
    ) -> Any:
        if line is None and trip is None:
            raise ValueError("line or trip is required for a service-flow query.")
        return self.query_route_flows(
            line=line,
            trip=trip,
            departure_time_bin=departure_time_bin,
            top_n=top_n,
        )

    @property
    def route_section_available(self) -> bool:
        payload = self.viewer_summary.get("route_sections", {})
        return isinstance(payload, Mapping) and bool(payload.get("available"))

    def _section_group_descriptors(self, group_name: str) -> tuple[Mapping[str, object], ...]:
        payload = self.viewer_summary.get("route_sections", {})
        group = payload.get(group_name, {}) if isinstance(payload, Mapping) else {}
        partitions = group.get("partitions", ()) if isinstance(group, Mapping) else ()
        if not isinstance(partitions, Sequence) or isinstance(partitions, (str, bytes)):
            return ()
        return tuple(item for item in partitions if isinstance(item, Mapping))

    def iter_section_od(
        self,
        section_id: str,
        *,
        departure_time_bin: str,
        line: str | None = None,
        pattern: str | None = None,
        direction: str | None = None,
        chunksize: int = 100_000,
    ) -> Iterator[Any]:
        """Stream every saved OD/path contribution traversing one directed section."""
        if not self.route_section_available:
            return
        filters: dict[str, object] = {"section_id": section_id}
        if line is not None:
            filters["line_id"] = line
        if pattern is not None:
            filters["pattern_id"] = pattern
        if direction is not None:
            filters["direction"] = direction
        descriptors = self._section_group_descriptors("contributors")
        for descriptor in descriptors:
            if str(descriptor.get("key")) != str(departure_time_bin):
                continue
            path = descriptor.get("path") or descriptor.get("file")
            if path:
                yield from self.iter_partition(
                    str(path), columns=_SECTION_CONTRIBUTION_COLUMNS,
                    chunksize=chunksize, filters=filters
                )

    def query_section_od(
        self,
        section_id: str,
        *,
        departure_time_bin: str,
        line: str | None = None,
        pattern: str | None = None,
        direction: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        aggregate_by_od: bool = False,
        chunksize: int = 100_000,
    ) -> Any:
        """Return all (or one page of) saved contributions for a directed section."""
        if page is not None and page < 0:
            raise ValueError("page must be non-negative")
        if page_size is not None and page_size <= 0:
            raise ValueError("page_size must be positive")
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("pandas is required for viewer access.") from error
        chunks = list(self.iter_section_od(
            section_id, departure_time_bin=departure_time_bin, line=line,
            pattern=pattern, direction=direction, chunksize=chunksize,
        ))
        if not chunks:
            result = pd.DataFrame(columns=_SECTION_CONTRIBUTION_COLUMNS)
        else:
            result = pd.concat(chunks, ignore_index=True)
        if aggregate_by_od and not result.empty:
            numeric = pd.to_numeric(result["assigned_demand"], errors="coerce").fillna(0.0)
            result = result.assign(assigned_contribution=numeric)
            group = ["origin_stop_id", "destination_stop_id", "departure_time_bin"]
            result = result.groupby(group, as_index=False, sort=False).agg(
                assigned_contribution=("assigned_contribution", "sum"),
                path_count=("route_path_id", "nunique"),
                transfer_count=("transfer_count", "first"),
            )
            od = self.read_table(
                "full_od.csv",
                columns=("origin_stop_id", "destination_stop_id", "departure_time_bin", "estimated_demand"),
                filters={"departure_time_bin": departure_time_bin},
                chunksize=chunksize,
            )
            if not od.empty:
                od["estimated_demand"] = pd.to_numeric(od["estimated_demand"], errors="coerce")
                result = result.merge(od, on=group, how="left").rename(
                    columns={"estimated_demand": "total_od_demand"}
                )
            else:
                result["total_od_demand"] = np.nan
        if page_size is not None:
            start = 0 if page is None else page * page_size
            result = result.iloc[start:start + page_size].reset_index(drop=True)
        return result

    def query_section_load(
        self,
        section_id: str,
        *,
        departure_time_bin: str,
        line: str | None = None,
        pattern: str | None = None,
        direction: str | None = None,
    ) -> Any:
        """Return the aggregate assigned passenger-section usage for a section."""
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("pandas is required for viewer access.") from error
        filters: dict[str, object] = {"section_id": section_id}
        if line is not None:
            filters["line_id"] = line
        if pattern is not None:
            filters["pattern_id"] = pattern
        if direction is not None:
            filters["direction"] = direction
        for descriptor in self._section_group_descriptors("section_loads"):
            if str(descriptor.get("key")) != str(departure_time_bin):
                continue
            path = descriptor.get("path") or descriptor.get("file")
            if path:
                chunks = list(self.iter_partition(str(path), columns=_SECTION_LOAD_COLUMNS,
                                                  filters=filters))
                return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=_SECTION_LOAD_COLUMNS)
        return pd.DataFrame(columns=_SECTION_LOAD_COLUMNS)

    def read_table(
        self,
        name: str,
        *,
        columns: Sequence[str] | None = None,
        filters: Mapping[str, object] | None = None,
        chunksize: int | None = None,
    ) -> Any:
        """Read a table; filtered reads default to bounded chunks."""
        if chunksize is not None or filters:
            size = 100_000 if chunksize is None else chunksize
            chunks = list(self.iter_table(name, columns=columns, chunksize=size, filters=filters))
            try:
                import pandas as pd
            except ImportError as error:  # pragma: no cover
                raise RuntimeError("pandas is required for viewer access.") from error
            return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("pandas is required for viewer access.") from error
        return pd.read_csv(self.table_path(name), usecols=columns)

    def query_od(
        self,
        *,
        origin: str | None = None,
        destination: str | None = None,
        departure_time_bin: str | None = None,
        identifiability_class: str | None = None,
        minimum_flow: float | None = None,
        top_n: int | None = None,
        chunksize: int = 100_000,
    ) -> Any:
        """Filter the OD table and optionally return its top-N flows."""
        filters: dict[str, object] = {}
        if origin is not None:
            filters["origin_stop_id"] = origin
        if destination is not None:
            filters["destination_stop_id"] = destination
        if departure_time_bin is not None:
            filters["departure_time_bin"] = departure_time_bin
        if identifiability_class is not None:
            filters["identifiability_class"] = identifiability_class
        result = self.read_table("full_od.csv", filters=filters, chunksize=chunksize)
        if minimum_flow is not None:
            result = result[result["estimated_demand"] >= minimum_flow]
        if not result.empty:
            result["demand_share"] = 0.0
            if origin is not None or destination is None:
                group_by = ["origin_stop_id", "departure_time_bin"]
            else:
                group_by = ["destination_stop_id", "departure_time_bin"]
            totals = result.groupby(group_by, dropna=False)["estimated_demand"].transform("sum")
            result["demand_share"] = np.divide(
                result["estimated_demand"], totals, out=np.zeros_like(totals), where=totals != 0
            )
        if top_n is not None:
            if top_n <= 0:
                raise ValueError("top_n must be positive when provided.")
            result = result.nlargest(top_n, "estimated_demand")
        return result

    def aggregate_measurements(
        self,
        by: str | Sequence[str] = "line",
        *,
        chunksize: int = 100_000,
    ) -> Any:
        """Aggregate observed/modelled measurements by one or more columns."""
        group_by = (by,) if isinstance(by, str) else tuple(by)
        if not group_by:
            raise ValueError("at least one aggregation column is required.")
        chunks = list(
            self.iter_table(
                "predicted_measurements.csv",
                columns=(*group_by, "observed", "modeled"),
                chunksize=chunksize,
            )
        )
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("pandas is required for viewer access.") from error
        if not chunks:
            return pd.DataFrame(columns=[*group_by, "observed", "modeled"])
        combined = pd.concat(chunks, ignore_index=True)
        return (
            combined.groupby(list(group_by), dropna=False, sort=True)[["observed", "modeled"]]
            .sum()
            .reset_index()
        )


def write_gravity_viewer_bundle(
    *,
    output_directory: str | Path,
    fit_result: GravityEstimationResult,
    validation_result: object,
    report: GravityDetailedReport,
    identifiability: GravityODIdentifiability | None = None,
    network_files: Mapping[str, str | Path] | None = None,
    metadata: Mapping[str, object] | None = None,
    require_identifiability: bool = False,
    model_specification: GravityModelSpecification | Mapping[str, object] | None = None,
    route_flow_rows: Iterable[Mapping[str, object]] | str | Path | None = None,
) -> GravityViewerBundle:
    """Export validated fit/report data for an independent graphical viewer."""
    if not isinstance(fit_result, GravityEstimationResult):
        raise TypeError("fit_result must be a GravityEstimationResult.")
    if not isinstance(report, GravityDetailedReport):
        raise TypeError("report must be a GravityDetailedReport.")
    if validation_result is None:
        raise ValueError("validation_result is required for a viewer bundle.")
    if require_identifiability and identifiability is None:
        raise ValueError("OD identifiability diagnostics are required for this bundle.")
    metadata = {} if metadata is None else dict(metadata)
    specification, specification_fingerprint = _result_specification(
        fit_result, model_specification
    )
    destination = Path(output_directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing viewer bundle directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name in (*_REPORT_FILES, *_TABLE_FILES):
            shutil.copy2(_report_path(report, name), staging / name)
        for name in _OPTIONAL_REPORT_FILES:
            source = report.files.get(name) if isinstance(report.files, Mapping) else None
            if source is not None and Path(source).is_file():
                shutil.copy2(source, staging / name)

        report_payload = json.loads((staging / "report.json").read_text(encoding="utf-8"))
        if not isinstance(report_payload, Mapping):
            raise ValueError("report.json must contain an object.")
        table_counts = _validate_table_inputs(
            staging, fit_result, report_payload, validation_result
        )

        _write_json(
            staging / _MODEL_SPECIFICATION_FILE,
            {
                "schema_version": _MODEL_SPECIFICATION_SCHEMA_VERSION,
                "artifact_type": "gravity_model_specification",
                "specification_fingerprint": specification_fingerprint,
                "specification": specification,
            },
        )

        identifiability_provenance: Mapping[str, object] | None = None
        if identifiability is not None:
            if identifiability.num_cells != table_counts["full_od.csv"]:
                raise ValueError("identifiability length does not match full_od.csv.")
            if identifiability.total_hessian_dimension != fit_result.raw_parameters.size:
                raise ValueError("identifiability parameter dimension does not match fit_result.")
            if identifiability.provenance.get("model_fingerprint") != fit_result.model_fingerprint:
                raise ValueError("identifiability model_fingerprint does not match fit_result.")
            write_gravity_od_identifiability(identifiability, staging / "identifiability")
            restored = read_gravity_od_identifiability(staging / "identifiability")
            identifiability_provenance = dict(restored.provenance)

        network_counts: dict[str, int] = {}
        if network_files is not None:
            if not isinstance(network_files, Mapping):
                raise TypeError("network_files must be a mapping of names to paths.")
            if "stops.csv" not in network_files:
                raise ValueError("network_files must explicitly include stops.csv.")
            network_dir = staging / "network"
            network_dir.mkdir()
            for name, source_value in network_files.items():
                if name not in _NETWORK_FILES:
                    raise ValueError(f"unsupported network snapshot file {name!r}.")
                source = Path(source_value).expanduser().resolve()
                if not source.is_file():
                    raise FileNotFoundError(f"network snapshot file is missing: {source}")
                shutil.copy2(source, network_dir / name)
            network_counts = _validate_network(staging, table_counts)
            table_counts.update(network_counts)

        # These are intentionally derived, display-oriented summaries.  They
        # never participate in model identity and are omitted when the source
        # report does not contain enough information to construct them.
        if _write_od_map(
            staging / "od_map.csv",
            full_od_rows=_iter_csv_rows(staging / "full_od.csv"),
            report_payload=report_payload,
        ):
            table_counts["od_map.csv"] = _csv_row_count(staging / "od_map.csv")

        predicted_fields = _csv_fieldnames(staging / "predicted_measurements.csv")
        predicted_rows = _iter_csv_rows(staging / "predicted_measurements.csv")
        network_stops: Iterable[Mapping[str, str]] = ()
        stops_path = staging / "network" / "stops.csv"
        if stops_path.is_file():
            network_stops = _iter_csv_rows(stops_path)
        if _write_stop_summary(
            staging / "stop_summary.csv",
            predicted_fields=predicted_fields,
            predicted_rows=predicted_rows,
            network_stops=network_stops,
        ):
            table_counts["stop_summary.csv"] = _csv_row_count(staging / "stop_summary.csv")

        if route_flow_rows is not None:
            report_provenance = report_payload.get("provenance", {})
            expected_assignment = (
                report_provenance.get("assignment_fingerprint")
                if isinstance(report_provenance, Mapping)
                else None
            )
            if expected_assignment is None:
                metadata_provenance = metadata.get("provenance", {})
                expected_assignment = (
                    metadata_provenance.get("assignment_fingerprint")
                    if isinstance(metadata_provenance, Mapping)
                    else None
                )
            _write_route_flow_source(
                staging / "route_flows.csv", route_flow_rows,
                expected_assignment_fingerprint=(
                    None if expected_assignment is None else str(expected_assignment)
                ),
            )
        viewer_summary = _write_viewer_partitions(staging)
        route_source = staging / "route_flows.csv"
        if route_source.exists():
            route_source.unlink()

        provenance = _source_provenance(report_payload, fit_result, metadata)
        provenance["specification_fingerprint"] = specification_fingerprint
        validation_prov = _validation_provenance(validation_result)
        for field, value in validation_prov.items():
            if field in provenance and provenance[field] is not None and value is not None and provenance[field] != value:
                raise ValueError(
                    f"validation provenance field {field!r} differs from fit/report: "
                    f"validation={value!r}, fit={provenance[field]!r}."
                )

        ident_summary: dict[str, object]
        if identifiability is None:
            ident_summary = {"available": False, "message": _IDENTIFIABILITY_MESSAGE}
        else:
            ident_summary = {
                "available": True,
                "path": "identifiability",
                "schema_version": 1,
                "provenance": dict(identifiability_provenance or {}),
                "effective_hessian_rank": identifiability.effective_hessian_rank,
                "total_hessian_dimension": identifiability.total_hessian_dimension,
                "classification_counts": identifiability.classification_counts,
            }
        route_summary = viewer_summary["route_flows"]
        route_flow_fingerprint = hashlib.sha256(
            canonical_json(route_summary).encode("utf-8")
        ).hexdigest()
        bundle_metadata = dict(metadata)
        bundle_metadata.pop("provenance", None)
        coordinate_convention = {
            "coordinate_system": str(
                bundle_metadata.pop("coordinate_system", "latitude_longitude")
            ),
            "x_column": str(bundle_metadata.pop("x_column", "lon")),
            "y_column": str(bundle_metadata.pop("y_column", "lat")),
        }
        time_zone = str(bundle_metadata.pop("time_zone", "unspecified"))
        manifest: dict[str, object] = {
            "schema_version": GRAVITY_VIEWER_BUNDLE_SCHEMA_VERSION,
            "bundle_type": GRAVITY_VIEWER_BUNDLE_TYPE,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_specification": {
                "available": True,
                "path": _MODEL_SPECIFICATION_FILE,
                "fingerprint": specification_fingerprint,
            },
            "provenance": provenance,
            "model_fingerprint": fit_result.model_fingerprint,
            "specification_fingerprint": specification_fingerprint,
            "fit_provenance": {
                "model_fingerprint": fit_result.model_fingerprint,
                "specification_fingerprint": specification_fingerprint,
                "optimizer": fit_result.optimizer,
                "status": fit_result.status,
                "acceptance": fit_result.acceptance,
            },
            "validation_provenance": dict(validation_prov),
            "report_provenance": {
                "report_fingerprint": report_payload.get("report_fingerprint", report.report_fingerprint),
                "model_fingerprint": report_payload.get("model_fingerprint"),
                "specification_fingerprint": report_payload.get("specification_fingerprint"),
                "model_specification": report_payload.get("model_specification"),
                "provenance": report_payload.get("provenance", {}),
            },
            "identifiability": ident_summary,
            "artifact_identity_fingerprint": provenance.get("artifact_identity_fingerprint"),
            "assignment_fingerprint": provenance.get("assignment_fingerprint"),
            "binding_fingerprint": provenance.get("binding_fingerprint"),
            "canonical_index_fingerprint": provenance.get("canonical_index_fingerprint"),
            "od_layout_fingerprint": provenance.get("od_layout_fingerprint"),
            "compact_layout_fingerprint": provenance.get("compact_layout_fingerprint"),
            "gravity_features_fingerprint": provenance.get("gravity_features_fingerprint"),
            "package_revision": provenance.get("package_revision"),
            "time_zone": time_zone,
            "coordinate_convention": coordinate_convention,
            "metadata": bundle_metadata,
            "row_counts": table_counts,
            "partitions": {
                "od": viewer_summary["od"]["partitions"],
                "route_flows": viewer_summary["route_flows"].get("partitions", []),
                "route_sections": viewer_summary.get("route_sections", {}),
                "cache_format_version": GRAVITY_VIEWER_CACHE_FORMAT_VERSION,
            },
            "viewer_summary": {
                "path": _VIEWER_SUMMARY_FILE,
                "cache_format_version": GRAVITY_VIEWER_CACHE_FORMAT_VERSION,
                "od_partitions": len(viewer_summary["od"]["partitions"]),
                "route_flows_available": bool(viewer_summary["route_flows"]["available"]),
                "route_sections_available": bool(viewer_summary.get("route_sections", {}).get("available")),
            },
            "route_flow_provenance": {
                "available": bool(viewer_summary["route_flows"]["available"]),
                "route_flow_fingerprint": route_flow_fingerprint,
                "assignment_fingerprint": provenance.get("assignment_fingerprint"),
                "routing_assignment_fingerprint": provenance.get("assignment_fingerprint"),
                "fit_fingerprint": fit_result.model_fingerprint,
                "specification_fingerprint": specification_fingerprint,
                "package_revision": provenance.get("package_revision"),
                "transfer_count_definition": viewer_summary["route_flows"].get(
                    "transfer_count_definition"
                ),
                "flow_semantics": viewer_summary["route_flows"].get("flow_semantics", []),
                "partition_row_counts": {
                    str(item.get("path")): item.get("row_count")
                    for item in viewer_summary["route_flows"].get("partitions", [])
                },
            },
            "route_section_provenance": {
                "available": bool(viewer_summary.get("route_sections", {}).get("available")),
                "schema_version": GRAVITY_VIEWER_ROUTE_SECTION_SCHEMA_VERSION,
                "cache_format_version": GRAVITY_VIEWER_CACHE_FORMAT_VERSION,
                "interval_semantics": viewer_summary.get("route_sections", {}).get("interval_semantics"),
                "geometry_path": viewer_summary.get("route_sections", {}).get("geometry_path"),
                "catalog_path": viewer_summary.get("route_sections", {}).get("catalog_path"),
                "message": viewer_summary.get("route_sections", {}).get("message"),
                "assignment_fingerprint": provenance.get("assignment_fingerprint"),
            },
        }
        manifest["files"] = _file_records(staging)
        _write_json(staging / "bundle_manifest.json", manifest)
        # Validate the staged bundle before it becomes visible to another process.
        staged_manifest = json.loads((staging / "bundle_manifest.json").read_text(encoding="utf-8"))
        _validate_bundle_directory(staging, staged_manifest)
        os.replace(staging, destination)
        return GravityViewerBundle(destination, staged_manifest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def write_persisted_gravity_viewer_bundle(
    *,
    output_directory: str | Path,
    fit_manifest: Mapping[str, object],
    validation_manifest: Mapping[str, object],
    observations: object,
    od_layout: ODParameterLayout,
    metadata: GravityValidationMetadata | None = None,
    likelihood: GravityLikelihood | str = "negative_binomial",
    identifiability: GravityODIdentifiability | None = None,
    network_files: Mapping[str, str | Path] | None = None,
    bundle_metadata: Mapping[str, object] | None = None,
    require_identifiability: bool = False,
    model_specification: GravityModelSpecification | Mapping[str, object] | None = None,
    report_output_directory: str | Path | None = None,
    route_flow_rows: Iterable[Mapping[str, object]] | str | Path | None = None,
) -> GravityViewerBundle:
    """Export a viewer bundle directly from persisted fit/validation artifacts.

    This convenience entry point restores the typed fit result, generates the
    detailed report through the persisted-data reporting contract, and passes
    the report to :func:`write_gravity_viewer_bundle`. It never loads a case
    context, activates routing, evaluates an objective, or invokes an
    optimizer. Unless ``report_output_directory`` is supplied, the temporary
    detailed report is removed after the bundle is written.
    """
    if not isinstance(fit_manifest, Mapping):
        raise TypeError("fit_manifest must be a mapping.")
    if not isinstance(validation_manifest, Mapping):
        raise TypeError("validation_manifest must be a mapping.")
    raw_result = fit_manifest.get("result")
    if not isinstance(raw_result, Mapping):
        raise ValueError("fit_manifest['result'] must be a mapping.")

    from .reporting import write_persisted_gravity_detailed_report

    result = GravityEstimationResult.from_dict(raw_result)
    destination = Path(output_directory).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_report: Path | None = None
    if report_output_directory is None:
        report_root = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.report-",
                dir=destination.parent,
            )
        )
        temporary_report = report_root
    else:
        report_root = Path(report_output_directory).expanduser().resolve()

    try:
        report = write_persisted_gravity_detailed_report(
            fit_manifest=fit_manifest,
            validation_manifest=validation_manifest,
            observations=observations,
            od_layout=od_layout,
            metadata=metadata,
            likelihood=likelihood,
            output_directory=report_root,
            identifiability=identifiability,
            require_identifiability=require_identifiability,
        )
        return write_gravity_viewer_bundle(
            output_directory=destination,
            fit_result=result,
            validation_result=validation_manifest,
            report=report,
            identifiability=identifiability,
            network_files=network_files,
            metadata=bundle_metadata,
            require_identifiability=require_identifiability,
            model_specification=model_specification,
            route_flow_rows=route_flow_rows,
        )
    finally:
        if temporary_report is not None:
            shutil.rmtree(temporary_report, ignore_errors=True)


def read_gravity_viewer_bundle(path: str | Path) -> GravityViewerBundle:
    """Read and validate a self-contained gravity viewer bundle."""
    root = Path(path).expanduser().resolve()
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"gravity viewer bundle manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("gravity viewer bundle manifest must contain an object.")
    bundle = GravityViewerBundle(root, dict(payload))
    bundle.validate()
    return bundle


__all__ = [
    "GRAVITY_VIEWER_CACHE_FORMAT_VERSION",
    "GRAVITY_VIEWER_BUNDLE_SCHEMA_VERSION",
    "GRAVITY_VIEWER_BUNDLE_TYPE",
    "GRAVITY_VIEWER_ROUTE_SECTION_SCHEMA_VERSION",
    "GravityViewerBundle",
    "read_gravity_viewer_bundle",
    "write_gravity_viewer_bundle",
    "write_persisted_gravity_viewer_bundle",
]
