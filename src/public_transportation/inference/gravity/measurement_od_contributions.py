"""Sparse OD-to-measurement contribution products for gravity viewer bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .estimator import GravityEstimationResult
from .observation_model import GravityJourneyLatentClassModel

if TYPE_CHECKING:
    from public_transportation.inference.temporal_assignment_persistence import (
        TemporalBlockArtifactReader,
    )


_SCHEMA_VERSION = 1
_WRITE_BATCH_SIZE = 32_768
_MAX_PARTITION_ROWS = 100_000
_CONTRIBUTION_COLUMNS = (
    "measurement_index",
    "od_index",
    "departure_time_bin",
    "operator_coefficient",
    "linear_flow_contribution",
    "modeled_mean_contribution",
)
_COMPONENT_COLUMNS = (
    "measurement_index",
    "fixed_offset_contribution",
    "additive_contribution",
    "mean_floor_adjustment",
    "modeled_mean",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "interval"
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{normalized[:64]}-{suffix}"


def _journey_scale(result: GravityEstimationResult) -> float:
    payload = result.measurement_model
    if payload is None:
        return 1.0
    model = GravityJourneyLatentClassModel.from_dict(payload)
    if not model.parameter_names:
        raw = np.empty(0, dtype=np.float64)
    else:
        name_to_position = {
            name: index for index, name in enumerate(result.parameter_names)
        }
        missing = [name for name in model.parameter_names if name not in name_to_position]
        if missing:
            raise ValueError(
                "fitted measurement-model parameters are missing from parameter_names."
            )
        raw = np.asarray(
            [
                result.raw_parameters[name_to_position[name]]
                for name in model.parameter_names
            ],
            dtype=np.float64,
        )
    scale = float(np.asarray(model.aggregate_scale(raw, dtype=np.float64)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("fitted measurement-model aggregate scale is invalid.")
    return scale


def write_measurement_od_contributions_dataset(
    *,
    destination: str | Path,
    artifact: TemporalBlockArtifactReader,
    fit_result: GravityEstimationResult,
    rho: float | None = None,
) -> dict[str, object]:
    """Stream a validated sparse operator into CSV partitions for the viewer.

    A missing saved observation-scale vector requires an explicit ``rho``.
    No scale is inferred from the predicted values. The temporary SQLite
    accumulator deduplicates split temporal blocks on disk and is removed after
    partitioned CSV files are written.
    """
    target = Path(destination)
    row_count = artifact.number_of_measurements
    if row_count != fit_result.predicted_measurements.size:
        raise ValueError("mapping measurement dimension differs from fitted predictions.")
    if artifact.number_of_demand_cells != fit_result.free_od_demand.size:
        raise ValueError("mapping demand dimension differs from fitted free OD demand.")
    demand = np.asarray(fit_result.free_od_demand, dtype=np.float64)
    modeled = np.asarray(fit_result.predicted_measurements, dtype=np.float64)
    if not np.all(np.isfinite(demand)) or not np.all(np.isfinite(modeled)):
        raise ValueError("fitted OD demand and modeled measurements must be finite.")

    if fit_result.observation_scales is None:
        if rho is None:
            return {
                "available": False,
                "message": (
                    "The fit does not persist its measurement-scale vector; pass rho "
                    "to generate exact modeled-mean contributions."
                ),
                "schema_version": _SCHEMA_VERSION,
            }
        scale = float(rho)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("rho must be finite and positive.")
        observation_scales = np.full(row_count, scale, dtype=np.float64)
        observation_scale_source = "explicit_rho"
    else:
        if rho is not None:
            raise ValueError("rho must be omitted when the fit stores observation_scales.")
        observation_scales = np.asarray(fit_result.observation_scales, dtype=np.float64)
        if observation_scales.shape != (row_count,) or not np.all(
            np.isfinite(observation_scales)
        ) or np.any(observation_scales <= 0.0):
            raise ValueError("fitted observation_scales are invalid.")
        observation_scale_source = "fit_result.observation_scales"

    journey_scale = _journey_scale(fit_result)
    additive_vectors = tuple(
        np.asarray(value, dtype=np.float64)
        for value in fit_result.additive_measurement_contributions
    )
    if any(
        value.shape != (row_count,) or not np.all(np.isfinite(value))
        for value in additive_vectors
    ):
        raise ValueError("fitted additive measurement contributions are invalid.")
    additive = (
        np.sum(np.stack(additive_vectors), axis=0)
        if additive_vectors
        else np.zeros(row_count, dtype=np.float64)
    )

    target.mkdir(parents=True, exist_ok=False)
    partition_root = target / "bins"
    partition_root.mkdir()
    working_directory = target / ".working"
    working_directory.mkdir()
    working_database = working_directory / "contributions.sqlite"
    raw_od = np.zeros(row_count, dtype=np.float64)
    departure_interval_by_column = [""] * artifact.number_of_demand_cells
    full_index_by_column = [-1] * artifact.number_of_demand_cells
    for cell in artifact.canonical_index.demand_cells:
        if cell.operator_column is not None:
            departure_interval_by_column[cell.operator_column] = cell.departure_interval_id
            full_index_by_column[cell.operator_column] = cell.full_index
    if any(not value for value in departure_interval_by_column) or any(
        value < 0 for value in full_index_by_column
    ):
        raise ValueError("canonical free-OD departure intervals are incomplete.")

    connection = sqlite3.connect(working_database)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE contributions ("
            "measurement_index INTEGER NOT NULL, od_index INTEGER NOT NULL, "
            "departure_time_bin TEXT NOT NULL, operator_coefficient REAL NOT NULL, "
            "linear_flow_contribution REAL NOT NULL, "
            "modeled_mean_contribution REAL NOT NULL, "
            "PRIMARY KEY (measurement_index, od_index)) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE INDEX contributions_by_interval "
            "ON contributions (departure_time_bin, measurement_index, od_index)"
        )
        for block in artifact.iter_blocks():
            for start in range(0, block.nonzero_entries, _WRITE_BATCH_SIZE):
                stop = min(block.nonzero_entries, start + _WRITE_BATCH_SIZE)
                rows = np.asarray(block.row_indices[start:stop], dtype=np.int64)
                columns = np.asarray(block.column_indices[start:stop], dtype=np.int64)
                coefficients = np.asarray(block.values[start:stop], dtype=np.float64)
                flows = coefficients * demand[columns]
                np.add.at(raw_od, rows, flows)
                means = flows * journey_scale * observation_scales[rows]
                connection.executemany(
                    "INSERT INTO contributions VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(measurement_index, od_index) DO UPDATE SET "
                    "operator_coefficient=operator_coefficient+excluded.operator_coefficient, "
                    "linear_flow_contribution=linear_flow_contribution+excluded.linear_flow_contribution, "
                    "modeled_mean_contribution=modeled_mean_contribution+excluded.modeled_mean_contribution",
                    [
                        (
                            int(row),
                            int(full_index_by_column[int(column)]),
                            departure_interval_by_column[int(column)],
                            float(coefficient),
                            float(flow),
                            float(mean),
                        )
                        for row, column, coefficient, flow, mean in zip(
                            rows,
                            columns,
                            coefficients,
                            flows,
                            means,
                            strict=True,
                        )
                    ],
                )

        persisted_od = fit_result.od_measurement_contribution
        if persisted_od is not None:
            persisted_od_array = np.asarray(persisted_od, dtype=np.float64)
            if persisted_od_array.shape != (row_count,) or not np.allclose(
                raw_od, persisted_od_array, rtol=1e-6, atol=1e-8
            ):
                raise ValueError(
                    "mapping OD contribution differs from the fitted OD measurement component."
                )

        if fit_result.latent_measurements is None:
            if additive_vectors:
                raise ValueError(
                    "fit is missing latent measurements required to separate fixed and additive components."
                )
            fixed_offset = np.asarray(artifact.fixed_measurement_offset, dtype=np.float64)
        else:
            latent = np.asarray(fit_result.latent_measurements, dtype=np.float64)
            if latent.shape != (row_count,) or not np.all(np.isfinite(latent)):
                raise ValueError("fitted latent measurements are invalid.")
            fixed_offset = latent - journey_scale * (raw_od + additive)
            if not np.allclose(
                fixed_offset,
                np.asarray(artifact.fixed_measurement_offset, dtype=np.float64),
                rtol=1e-6,
                atol=1e-8,
            ):
                raise ValueError("fitted fixed offset differs from the mapping artifact.")

        fixed_mean = observation_scales * fixed_offset
        additive_mean = observation_scales * journey_scale * additive
        od_mean = observation_scales * journey_scale * raw_od
        floor_adjustment = modeled - fixed_mean - additive_mean - od_mean
        tolerance = 1e-6 + 1e-6 * np.abs(modeled)
        if np.any(floor_adjustment < -tolerance):
            raise ValueError(
                "OD, fixed, and additive components exceed the saved modeled mean."
            )
        floor_adjustment = np.maximum(floor_adjustment, 0.0)
        if not np.allclose(
            fixed_mean + additive_mean + od_mean + floor_adjustment,
            modeled,
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError("measurement component totals do not reproduce fitted predictions.")

        connection.execute(
            "CREATE TABLE components (measurement_index INTEGER PRIMARY KEY, "
            "fixed_offset_contribution REAL NOT NULL, additive_contribution REAL NOT NULL, "
            "mean_floor_adjustment REAL NOT NULL, modeled_mean REAL NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO components VALUES (?, ?, ?, ?, ?)",
            [
                (
                    index,
                    float(fixed_mean[index]),
                    float(additive_mean[index]),
                    float(floor_adjustment[index]),
                    float(modeled[index]),
                )
                for index in range(row_count)
            ],
        )
        connection.commit()

        bins = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT departure_time_bin FROM contributions "
                "ORDER BY departure_time_bin"
            )
        )
        descriptors: list[dict[str, object]] = []
        total_rows = 0
        for departure_bin in bins:
            cursor = connection.execute(
                "SELECT measurement_index, od_index, departure_time_bin, "
                "operator_coefficient, linear_flow_contribution, "
                "modeled_mean_contribution FROM contributions "
                "WHERE departure_time_bin=? ORDER BY measurement_index, od_index",
                (departure_bin,),
            )
            stream = None
            writer = None
            output = None
            chunk_count = 0
            chunk_index = 0

            def finish_partition() -> None:
                nonlocal stream, writer, output, chunk_count, chunk_index
                if stream is None or output is None:
                    return
                stream.close()
                descriptors.append(
                    {
                        "key": departure_bin,
                        "chunk_index": chunk_index,
                        "path": output.relative_to(target.parent.parent).as_posix(),
                        "row_count": chunk_count,
                        "size_bytes": output.stat().st_size,
                        "sha256": _sha256(output),
                    }
                )
                chunk_index += 1
                stream = None
                writer = None
                output = None
                chunk_count = 0

            while batch := cursor.fetchmany(50_000):
                for row in batch:
                    if stream is None or chunk_count >= _MAX_PARTITION_ROWS:
                        finish_partition()
                        output = partition_root / (
                            f"departure-{_slug(departure_bin)}-{chunk_index:05d}.csv"
                        )
                        stream = output.open("w", newline="", encoding="utf-8")
                        writer = csv.DictWriter(
                            stream, fieldnames=_CONTRIBUTION_COLUMNS
                        )
                        writer.writeheader()
                    assert writer is not None
                    writer.writerow(dict(zip(_CONTRIBUTION_COLUMNS, row, strict=True)))
                    chunk_count += 1
                    total_rows += 1
            finish_partition()

        component_path = target / "measurement_components.csv"
        with component_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=_COMPONENT_COLUMNS)
            writer.writeheader()
            cursor = connection.execute(
                "SELECT measurement_index, fixed_offset_contribution, "
                "additive_contribution, mean_floor_adjustment, modeled_mean "
                "FROM components ORDER BY measurement_index"
            )
            while batch := cursor.fetchmany(50_000):
                writer.writerows(
                    dict(zip(_COMPONENT_COLUMNS, row, strict=True)) for row in batch
                )
        connection.close()
        connection = None
    finally:
        if connection is not None:
            connection.close()
        shutil.rmtree(working_directory, ignore_errors=True)

    fixed_positive_cells = sum(
        cell.role == "fixed_positive"
        for cell in artifact.canonical_index.demand_cells
    )
    manifest: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "dataset_type": "gravity_measurement_od_contributions",
        "available": True,
        "fit_fingerprint": fit_result.model_fingerprint,
        "assignment_fingerprint": artifact.identity.timetable_fingerprint,
        "mapping_artifact_fingerprint": artifact.identity.fingerprint,
        "mapping_manifest_sha256": artifact.manifest_sha256,
        "canonical_index_fingerprint": artifact.canonical_index.artifact_fingerprint,
        "binding_fingerprint": artifact.canonical_index.binding_fingerprint,
        "od_layout_fingerprint": artifact.canonical_index.source_od_layout_fingerprint,
        "number_of_measurements": row_count,
        "number_of_free_od_cells": len(demand),
        "number_of_physical_od_cells": artifact.canonical_index.number_of_physical_demand_cells,
        "nonzero_pairs": total_rows,
        "journey_aggregate_scale": journey_scale,
        "observation_scale_source": observation_scale_source,
        "explicit_rho": None if observation_scale_source != "explicit_rho" else float(rho),
        "partition_field": "departure_time_bin",
        "contribution_columns": list(_CONTRIBUTION_COLUMNS),
        "component_columns": list(_COMPONENT_COLUMNS),
        "component_table": {
            "path": component_path.relative_to(target.parent.parent).as_posix(),
            "row_count": row_count,
            "size_bytes": component_path.stat().st_size,
            "sha256": _sha256(component_path),
        },
        "fixed_positive_od_attribution_available": fixed_positive_cells == 0,
        "fixed_positive_od_attribution_message": (
            None
            if fixed_positive_cells == 0
            else (
                "Fixed-positive OD demand is aggregated in the mapping's fixed offset; "
                "the artifact has no per-cell coefficients for those cells."
            )
        ),
        "partitions": descriptors,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_relative = manifest_path.relative_to(target.parent.parent).as_posix()
    dataset_relative = target.relative_to(target.parent.parent).as_posix()
    return {
        "available": True,
        "schema_version": _SCHEMA_VERSION,
        "dataset_path": dataset_relative,
        "dataset_manifest": manifest_relative,
        "dataset_manifest_sha256": _sha256(manifest_path),
        "component_table": manifest["component_table"],
        "number_of_measurements": row_count,
        "number_of_free_od_cells": len(demand),
        "nonzero_pairs": total_rows,
        "partitions": descriptors,
        "fixed_positive_od_attribution_available": fixed_positive_cells == 0,
        "fixed_positive_od_attribution_message": manifest[
            "fixed_positive_od_attribution_message"
        ],
        "fit_fingerprint": fit_result.model_fingerprint,
        "assignment_fingerprint": artifact.identity.timetable_fingerprint,
        "mapping_artifact_fingerprint": artifact.identity.fingerprint,
        "canonical_index_fingerprint": artifact.canonical_index.artifact_fingerprint,
        "binding_fingerprint": artifact.canonical_index.binding_fingerprint,
        "od_layout_fingerprint": artifact.canonical_index.source_od_layout_fingerprint,
        "observation_scale_source": observation_scale_source,
        "explicit_rho": None if observation_scale_source != "explicit_rho" else float(rho),
    }


def validate_dataset(root: str | Path, payload: Mapping[str, object]) -> None:
    """Fail closed on corrupt contribution manifests or partitioned CSV files."""
    if payload.get("schema_version") != _SCHEMA_VERSION or payload.get("available") is not True:
        raise ValueError("measurement OD contribution dataset metadata is invalid.")
    bundle_root = Path(root)
    dataset_relative = Path(str(payload.get("dataset_path", "")))
    manifest_relative = Path(str(payload.get("dataset_manifest", "")))
    if (
        not dataset_relative.parts
        or dataset_relative.is_absolute()
        or ".." in dataset_relative.parts
        or manifest_relative.is_absolute()
        or ".." in manifest_relative.parts
        or manifest_relative != dataset_relative / "manifest.json"
    ):
        raise ValueError("measurement OD contribution dataset path is unsafe.")
    dataset_root = bundle_root / dataset_relative
    manifest_path = bundle_root / manifest_relative
    if not dataset_root.is_dir() or not manifest_path.is_file():
        raise ValueError("measurement OD contribution dataset is missing.")
    if _sha256(manifest_path) != payload.get("dataset_manifest_sha256"):
        raise ValueError("measurement OD contribution manifest checksum mismatch.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("measurement OD contribution manifest is corrupt.") from error
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("measurement OD contribution manifest schema is invalid.")
    if (
        manifest.get("fit_fingerprint") != payload.get("fit_fingerprint")
        or manifest.get("assignment_fingerprint")
        != payload.get("assignment_fingerprint")
        or manifest.get("mapping_artifact_fingerprint")
        != payload.get("mapping_artifact_fingerprint")
        or manifest.get("canonical_index_fingerprint")
        != payload.get("canonical_index_fingerprint")
        or manifest.get("binding_fingerprint") != payload.get("binding_fingerprint")
        or manifest.get("od_layout_fingerprint") != payload.get("od_layout_fingerprint")
        or manifest.get("observation_scale_source")
        != payload.get("observation_scale_source")
        or manifest.get("explicit_rho") != payload.get("explicit_rho")
    ):
        raise ValueError("measurement OD contribution provenance differs from its manifest.")
    descriptors = manifest.get("partitions")
    if not isinstance(descriptors, list):
        raise ValueError("measurement OD contribution partitions are invalid.")
    total = 0
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise ValueError("measurement OD contribution partition descriptor is invalid.")
        relative = Path(str(descriptor.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("measurement OD contribution partition path is unsafe.")
        path = bundle_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != descriptor.get("size_bytes")
            or _sha256(path) != descriptor.get("sha256")
        ):
            raise ValueError("measurement OD contribution partition checksum mismatch.")
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != _CONTRIBUTION_COLUMNS:
                raise ValueError("measurement OD contribution partition columns are invalid.")
            count = 0
            for row in reader:
                try:
                    measurement_index = int(row["measurement_index"])
                    od_index = int(row["od_index"])
                    numeric_values = (
                        float(row["operator_coefficient"]),
                        float(row["linear_flow_contribution"]),
                        float(row["modeled_mean_contribution"]),
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError("measurement OD contribution row is invalid.") from error
                if (
                    measurement_index < 0
                    or measurement_index >= int(manifest.get("number_of_measurements", -1))
                    or od_index < 0
                    or od_index >= int(manifest.get("number_of_physical_od_cells", -1))
                    or row["departure_time_bin"] != descriptor.get("key")
                    or not np.all(np.isfinite(numeric_values))
                ):
                    raise ValueError("measurement OD contribution row is outside its partition.")
                count += 1
        if count != descriptor.get("row_count"):
            raise ValueError("measurement OD contribution partition row count mismatch.")
        total += count
    if total != manifest.get("nonzero_pairs") or total != payload.get("nonzero_pairs"):
        raise ValueError("measurement OD contribution total row count mismatch.")

    component = manifest.get("component_table")
    if not isinstance(component, Mapping):
        raise ValueError("measurement OD component table descriptor is invalid.")
    relative = Path(str(component.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("measurement OD component table path is unsafe.")
    component_path = bundle_root / relative
    if (
        not component_path.is_file()
        or component_path.stat().st_size != component.get("size_bytes")
        or _sha256(component_path) != component.get("sha256")
    ):
        raise ValueError("measurement OD component table checksum mismatch.")
    with component_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _COMPONENT_COLUMNS:
            raise ValueError("measurement OD component table columns are invalid.")
        count = 0
        for row in reader:
            try:
                index = int(row["measurement_index"])
                values = [
                    float(row[name])
                    for name in _COMPONENT_COLUMNS[1:]
                ]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("measurement OD component row is invalid.") from error
            if (
                index != count
                or not np.all(np.isfinite(values))
                or index >= int(manifest.get("number_of_measurements", -1))
            ):
                raise ValueError("measurement OD component row is invalid.")
            count += 1
    if count != component.get("row_count") or count != manifest.get("number_of_measurements"):
        raise ValueError("measurement OD component table row count mismatch.")
