"""Compact, restartable OD-to-section products.

The historical route-flow export stores one row for every saved path/leg.  This
module provides the smaller product used by the Model 6 viewer: one exact row
per departure interval, directed section, OD pair, line and trip.  The module
is deliberately independent from routing; it can consume a completed route
export or a stream produced by an exporter.

CSV is the always-available format.  ``format='auto'`` selects Parquet when
PyArrow is installed and otherwise uses CSV.  Parquet is optional so importing
the gravity package does not add a binary dependency for users that only use
the estimator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from public_transportation import __version__


SECTION_USAGE_SCHEMA_VERSION = 1
SECTION_USAGE_FORMAT_VERSION = 1
SECTION_USAGE_ALGORITHM_VERSION = "od-section-usage-aggregate-v1"
SECTION_USAGE_COLUMNS = (
    "departure_time_bin",
    "section_id",
    "origin_stop_id",
    "destination_stop_id",
    "line_id",
    "trip_id",
    "assigned_contribution",
)
SECTION_USAGE_KEY_COLUMNS = SECTION_USAGE_COLUMNS[:-1]
SECTION_USAGE_SEMANTICS = "service_marginal"


class SectionUsageStorageBudgetExceeded(RuntimeError):
    """Raised after a durable dataset exceeds its configured storage budget."""


class SectionUsageCheckpointError(ValueError):
    """Raised when a section-usage manifest or source checkpoint is unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _slug(value: object) -> str:
    text = str(value).strip() or "unassigned"
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in text)
    return f"{safe[:64] or 'unassigned'}-{hashlib.sha256(text.encode()).hexdigest()[:10]}"


def _identity_payload(
    identity: Mapping[str, object] | None,
    *,
    output_format: str,
    partition_field: str,
    shard_rows: int,
    grain: str = "trip",
) -> dict[str, object]:
    result = {
        "schema_version": SECTION_USAGE_SCHEMA_VERSION,
        "format_version": SECTION_USAGE_FORMAT_VERSION,
        "algorithm_version": SECTION_USAGE_ALGORITHM_VERSION,
        "package_revision": __version__,
        "grain": f"section_usage_{grain}",
        "flow_semantics": SECTION_USAGE_SEMANTICS,
        "format": output_format,
        "partition_field": partition_field,
        "shard_rows": int(shard_rows),
        "source_identity": {} if identity is None else dict(identity),
    }
    return result


def _normalise_compact_row(
    raw: Mapping[str, object],
    *,
    section_resolver: Callable[[Mapping[str, object]], str] | None = None,
) -> dict[str, object]:
    """Normalize a route-flow or compact row to the seven-column grain."""
    def value(*names: str) -> object:
        for name in names:
            if name in raw and raw[name] is not None:
                return raw[name]
        return ""

    section = value("section_id")
    if section_resolver is not None and not str(section).strip():
        section = section_resolver(raw)
    if not str(section).strip():
        raise ValueError(
            "section_id is required for compact section usage; provide a section_resolver "
            "when migrating a route-level checkpoint."
        )
    required = {
        "departure_time_bin": value("departure_time_bin", "departure", "interval"),
        "origin_stop_id": value("origin_stop_id", "origin"),
        "destination_stop_id": value("destination_stop_id", "destination"),
        "line_id": value("line_id", "line"),
        "trip_id": value("trip_id", "trip"),
    }
    missing = [name for name, item in required.items() if not str(item).strip()]
    if missing:
        raise ValueError("section usage row is missing required fields: " + ", ".join(missing))
    amount = value("assigned_contribution", "assigned_demand", "flow")
    try:
        numeric = float(amount)
    except (TypeError, ValueError) as error:
        raise ValueError("assigned_contribution must be numeric.") from error
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError("assigned_contribution must be finite and non-negative.")
    return {
        "departure_time_bin": str(required["departure_time_bin"]),
        "section_id": str(section),
        "origin_stop_id": str(required["origin_stop_id"]),
        "destination_stop_id": str(required["destination_stop_id"]),
        "line_id": str(required["line_id"]),
        "trip_id": str(required["trip_id"]),
        "assigned_contribution": numeric,
    }


def aggregate_section_usage_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    section_resolver: Callable[[Mapping[str, object]], str] | None = None,
) -> list[dict[str, object]]:
    """Aggregate a finite stream exactly at the compact section grain."""
    totals: dict[tuple[str, ...], float] = {}
    for raw in rows:
        row = _normalise_compact_row(raw, section_resolver=section_resolver)
        key = tuple(str(row[name]) for name in SECTION_USAGE_KEY_COLUMNS)
        totals[key] = totals.get(key, 0.0) + float(row["assigned_contribution"])
    return [
        {**dict(zip(SECTION_USAGE_KEY_COLUMNS, key, strict=True)), "assigned_contribution": value}
        for key, value in sorted(totals.items())
    ]


def aggregate_line_usage_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Return an exact line-level product by summing compact trip rows."""
    totals: dict[tuple[str, ...], float] = {}
    for raw in rows:
        row = _normalise_compact_row(raw)
        key = (
            row["departure_time_bin"], row["section_id"], row["origin_stop_id"],
            row["destination_stop_id"], row["line_id"],
        )
        totals[key] = totals.get(key, 0.0) + float(row["assigned_contribution"])
    columns = SECTION_USAGE_KEY_COLUMNS[:-1]
    return [
        {**dict(zip(columns, key, strict=True)), "trip_id": "", "assigned_contribution": value}
        for key, value in sorted(totals.items())
    ]


def _pyarrow() -> Any:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "Parquet output requires the optional 'pyarrow' dependency; use format='csv' "
            "or install public_transportation[parquet]."
        ) from error
    return pa, pq


def _write_csv_shard(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(SECTION_USAGE_COLUMNS), extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row[name] for name in SECTION_USAGE_COLUMNS})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_parquet_shard(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    pa, pq = _pyarrow()
    table = pa.Table.from_pylist(
        [{name: row[name] for name in SECTION_USAGE_COLUMNS} for row in rows]
    )
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        use_dictionary=list(SECTION_USAGE_KEY_COLUMNS),
        row_group_size=len(rows),
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _source_rows_from_checkpoint(
    checkpoint: Path,
    *,
    section_resolver: Callable[[Mapping[str, object]], str] | None,
) -> Iterator[Mapping[str, object]]:
    manifest_path = checkpoint / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SectionUsageCheckpointError(f"cannot read OD-section checkpoint: {manifest_path}") from error
    if not isinstance(manifest, Mapping):
        raise SectionUsageCheckpointError("OD-section checkpoint manifest must be an object.")
    units = manifest.get("completed_units")
    if not isinstance(units, Mapping):
        raise SectionUsageCheckpointError("OD-section checkpoint has no completed_units manifest.")
    if manifest.get("status") != "completed":
        raise SectionUsageCheckpointError(
            "only a completed OD-section checkpoint can be migrated; interrupted work is not an approved dataset."
        )
    for key in sorted(units, key=lambda item: int(item)):
        record = units[key]
        if not isinstance(record, Mapping) or record.get("status") != "completed":
            raise SectionUsageCheckpointError(f"checkpoint unit {key!r} is not complete.")
        if record.get("unit_type") == "legacy_group":
            paths = (Path(str(record.get("path", ""))),)
        else:
            chunks = record.get("chunks", ())
            if not isinstance(chunks, Sequence):
                raise SectionUsageCheckpointError(f"checkpoint unit {key!r} chunks are corrupt.")
            paths = tuple(checkpoint / str(chunk.get("path", "")) for chunk in chunks if isinstance(chunk, Mapping))
        for path in paths:
            if not path.is_absolute():
                path = checkpoint / path
            expected_sha = None
            if record.get("unit_type") == "legacy_group":
                expected_sha = record.get("sha256")
            else:
                for chunk in record.get("chunks", ()):
                    if isinstance(chunk, Mapping) and str(checkpoint / str(chunk.get("path", ""))) == str(path):
                        expected_sha = chunk.get("sha256")
                        break
            if not path.is_file() or expected_sha != _sha256(path):
                raise SectionUsageCheckpointError(f"checkpoint chunk is missing or corrupt: {path}")
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None:
                    raise SectionUsageCheckpointError(f"checkpoint chunk has no header: {path}")
                yield from (
                    _normalise_compact_row(row, section_resolver=section_resolver)
                    for row in reader
                )


def write_section_usage_dataset(
    rows: Iterable[Mapping[str, object]],
    destination: str | Path,
    *,
    source_identity: Mapping[str, object] | None = None,
    format: str = "auto",
    partition_field: str = "departure_time_bin",
    shard_rows: int = 100_000,
    storage_budget_bytes: int | None = None,
    section_resolver: Callable[[Mapping[str, object]], str] | None = None,
    grain: str = "trip",
) -> dict[str, object]:
    """Write an exact, partitioned section product without full-table memory.

    A small SQLite aggregate is used as an external-memory sort/merge.  It
    keeps duplicate-key aggregation exact while bounded shards are written to
    the final dataset.  A completed ``manifest.json`` is published last.
    """
    if partition_field != "departure_time_bin":
        raise ValueError("section usage datasets are partitioned by departure_time_bin.")
    if grain not in {"trip", "line"}:
        raise ValueError("grain must be trip or line.")
    if shard_rows <= 0:
        raise ValueError("shard_rows must be positive.")
    requested = str(format).lower()
    if requested not in {"auto", "csv", "parquet"}:
        raise ValueError("format must be auto, csv, or parquet.")
    if requested == "parquet":
        _pyarrow()
    actual_format = requested
    if requested == "auto":
        try:
            _pyarrow()
        except RuntimeError:
            actual_format = "csv"
        else:
            actual_format = "parquet"
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists():
        raise FileExistsError(f"section usage destination already exists: {destination_path}")
    destination_path.mkdir(parents=True)
    partition_root = destination_path / "partitions"
    partition_root.mkdir()
    database_path = destination_path / ".aggregate.sqlite"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE usage (departure TEXT NOT NULL, section TEXT NOT NULL, origin TEXT NOT NULL, "
        "destination TEXT NOT NULL, line TEXT NOT NULL, trip TEXT NOT NULL, amount REAL NOT NULL, "
        "PRIMARY KEY (departure, section, origin, destination, line, trip))"
    )
    rows_seen = 0
    descriptors: list[dict[str, object]] = []
    total_rows = 0
    total_bytes = 0
    try:
        for raw in rows:
            row = _normalise_compact_row(raw, section_resolver=section_resolver)
            if grain == "line":
                row["trip_id"] = ""
            connection.execute(
                "INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO UPDATE SET amount = amount + excluded.amount",
                tuple(row[name] for name in SECTION_USAGE_COLUMNS),
            )
            rows_seen += 1
        connection.commit()
        current_bin: str | None = None
        shard_index = 0
        buffer: list[dict[str, object]] = []

        def flush() -> None:
            nonlocal buffer, shard_index, total_rows, total_bytes
            if not buffer or current_bin is None:
                return
            suffix = "parquet" if actual_format == "parquet" else "csv"
            path = partition_root / f"{_slug(current_bin)}-{shard_index:06d}.{suffix}"
            if actual_format == "parquet":
                _write_parquet_shard(path, buffer)
            else:
                _write_csv_shard(path, buffer)
            size = path.stat().st_size
            total_rows += len(buffer)
            total_bytes += size
            descriptors.append({
                "key": current_bin,
                "path": path.relative_to(destination_path).as_posix(),
                "file": path.relative_to(destination_path).as_posix(),
                "row_count": len(buffer),
                "size_bytes": size,
                "sha256": _sha256(path),
            })
            if storage_budget_bytes is not None and total_bytes > storage_budget_bytes:
                raise SectionUsageStorageBudgetExceeded(
                    f"section usage storage budget exceeded after {total_rows} rows "
                    f"({total_bytes} bytes > {storage_budget_bytes})."
                )
            buffer = []
            shard_index += 1

        cursor = connection.execute(
            "SELECT departure, section, origin, destination, line, trip, amount "
            "FROM usage ORDER BY departure, section, line, trip, origin, destination"
        )
        for departure, section, origin, destination_id, line, trip, amount in cursor:
            if current_bin != str(departure):
                flush()
                current_bin = str(departure)
                shard_index = 0
            buffer.append({
                "departure_time_bin": str(departure),
                "section_id": str(section),
                "origin_stop_id": str(origin),
                "destination_stop_id": str(destination_id),
                "line_id": str(line),
                "trip_id": str(trip),
                "assigned_contribution": float(amount),
            })
            if len(buffer) >= shard_rows:
                flush()
        flush()
        identity_payload = _identity_payload(
            source_identity,
            output_format=actual_format,
            partition_field=partition_field,
            shard_rows=shard_rows,
            grain=grain,
        )
        manifest = {
            **identity_payload,
            "status": "completed",
            "columns": list(SECTION_USAGE_COLUMNS),
            "partitions": descriptors,
            "rows_seen": rows_seen,
            "rows_written": total_rows,
            "bytes_written": total_bytes,
            "compression": "zstd" if actual_format == "parquet" else "none",
            "lines": [row[0] for row in connection.execute("SELECT DISTINCT line FROM usage ORDER BY line")],
            "trips": [row[0] for row in connection.execute("SELECT DISTINCT trip FROM usage ORDER BY trip")],
            "orphan_temporary_files": [],
        }
        _atomic_json(destination_path / "manifest.json", manifest)
        return manifest
    except SectionUsageStorageBudgetExceeded as error:
        partial = {
            "schema_version": SECTION_USAGE_SCHEMA_VERSION,
            "format_version": SECTION_USAGE_FORMAT_VERSION,
            "algorithm_version": SECTION_USAGE_ALGORITHM_VERSION,
            "status": "storage_budget_exceeded",
            "rows_seen": rows_seen,
            "rows_written": total_rows,
            "bytes_written": total_bytes,
            "partitions": descriptors,
            "storage_budget_bytes": storage_budget_bytes,
            "message": str(error),
            "orphan_temporary_files": [
                path.relative_to(destination_path).as_posix()
                for path in destination_path.rglob(".*")
                if path.is_file()
            ],
        }
        _atomic_json(destination_path / "manifest.json", partial)
        raise
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)


def migrate_od_section_checkpoint(
    checkpoint_directory: str | Path,
    destination: str | Path,
    *,
    source_identity: Mapping[str, object] | None = None,
    format: str = "auto",
    shard_rows: int = 100_000,
    storage_budget_bytes: int | None = None,
    section_resolver: Callable[[Mapping[str, object]], str] | None = None,
    grain: str = "trip",
) -> dict[str, object]:
    """Stream only manifest-referenced completed chunks into a compact dataset."""
    checkpoint = Path(checkpoint_directory).expanduser().resolve()
    manifest_path = checkpoint / "manifest.json"
    try:
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SectionUsageCheckpointError(f"cannot read OD-section checkpoint: {manifest_path}") from error
    if not isinstance(source_manifest, Mapping):
        raise SectionUsageCheckpointError("OD-section checkpoint manifest must be an object.")
    migration_identity = {
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_manifest_sha256": _sha256(manifest_path),
        "source_checkpoint_provenance": source_manifest.get("provenance", {}),
    }
    if source_identity is not None:
        migration_identity.update(dict(source_identity))
    return write_section_usage_dataset(
        _source_rows_from_checkpoint(checkpoint, section_resolver=section_resolver),
        destination,
        source_identity=migration_identity,
        format=format,
        shard_rows=shard_rows,
        storage_budget_bytes=storage_budget_bytes,
        grain=grain,
    )


def write_line_usage_dataset(
    rows: Iterable[Mapping[str, object]],
    destination: str | Path,
    *,
    source_identity: Mapping[str, object] | None = None,
    format: str = "auto",
    shard_rows: int = 100_000,
    storage_budget_bytes: int | None = None,
) -> dict[str, object]:
    """Write the exact line-level aggregate (trip IDs are intentionally blank)."""
    return write_section_usage_dataset(
        rows,
        destination,
        source_identity=source_identity,
        format=format,
        shard_rows=shard_rows,
        storage_budget_bytes=storage_budget_bytes,
        grain="line",
    )


def read_section_usage_manifest(dataset: str | Path) -> dict[str, object]:
    root = Path(dataset).expanduser().resolve()
    try:
        payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SectionUsageCheckpointError(f"cannot read section usage manifest: {root}") from error
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        raise SectionUsageCheckpointError("section usage dataset is not complete.")
    if payload.get("schema_version") != SECTION_USAGE_SCHEMA_VERSION:
        raise SectionUsageCheckpointError("unsupported section usage schema version.")
    if tuple(payload.get("columns", ())) != SECTION_USAGE_COLUMNS:
        raise SectionUsageCheckpointError("section usage columns do not match the supported grain.")
    for descriptor in payload.get("partitions", ()):
        if not isinstance(descriptor, Mapping):
            raise SectionUsageCheckpointError("section usage partition descriptor is corrupt.")
        relative = Path(str(descriptor.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise SectionUsageCheckpointError("section usage partition path is unsafe.")
        path = root / relative
        if not path.is_file() or descriptor.get("sha256") != _sha256(path):
            raise SectionUsageCheckpointError(f"section usage partition is missing or corrupt: {relative}")
    return payload


def iter_section_usage_rows(
    dataset: str | Path,
    *,
    departure_time_bin: str | None = None,
    section_id: str | None = None,
    line_id: str | None = None,
    trip_id: str | None = None,
    batch_size: int = 100_000,
) -> Iterator[list[dict[str, object]]]:
    """Yield bounded filtered batches, selecting only eligible partitions."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    root = Path(dataset).expanduser().resolve()
    manifest = read_section_usage_manifest(root)
    for descriptor in manifest.get("partitions", ()):
        if departure_time_bin is not None and str(descriptor.get("key")) != str(departure_time_bin):
            continue
        path = root / str(descriptor["path"])
        batch: list[dict[str, object]] = []
        if path.suffix == ".parquet":
            _, pq = _pyarrow()
            parquet_file = pq.ParquetFile(path)
            iterator = (
                row
                for record_batch in parquet_file.iter_batches(
                    batch_size=batch_size, columns=list(SECTION_USAGE_COLUMNS)
                )
                for row in record_batch.to_pylist()
            )
        else:
            with path.open(newline="", encoding="utf-8") as stream:
                iterator = csv.DictReader(stream)
                for raw in iterator:
                    row = dict(raw)
                    if section_id is not None and row["section_id"] != str(section_id):
                        continue
                    if line_id is not None and row["line_id"] != str(line_id):
                        continue
                    if trip_id is not None and row["trip_id"] != str(trip_id):
                        continue
                    row["assigned_contribution"] = float(row["assigned_contribution"])
                    batch.append(row)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch
            continue
        for row in iterator:
            if section_id is not None and row["section_id"] != str(section_id):
                continue
            if line_id is not None and row["line_id"] != str(line_id):
                continue
            if trip_id is not None and row["trip_id"] != str(trip_id):
                continue
            row["assigned_contribution"] = float(row["assigned_contribution"])
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


__all__ = [
    "SECTION_USAGE_ALGORITHM_VERSION",
    "SECTION_USAGE_COLUMNS",
    "SECTION_USAGE_FORMAT_VERSION",
    "SECTION_USAGE_KEY_COLUMNS",
    "SECTION_USAGE_SCHEMA_VERSION",
    "SectionUsageCheckpointError",
    "SectionUsageStorageBudgetExceeded",
    "aggregate_line_usage_rows",
    "aggregate_section_usage_rows",
    "iter_section_usage_rows",
    "migrate_od_section_checkpoint",
    "read_section_usage_manifest",
    "write_section_usage_dataset",
    "write_line_usage_dataset",
]
