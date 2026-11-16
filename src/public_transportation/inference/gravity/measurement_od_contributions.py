"""Sparse OD-to-measurement contribution products for gravity viewer bundles."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .estimator import GravityEstimationResult
from .observation_model import (
    GravityJourneyLatentClassModel,
    GravityObservationModel,
)

if TYPE_CHECKING:
    from public_transportation.inference.temporal_assignment_persistence import (
        TemporalBlockArtifactReader,
    )


_SCHEMA_VERSION = 1
_WRITE_BATCH_SIZE = 32_768
_MAX_PARTITION_ROWS = 100_000
_COMPONENT_PARTITION_ROWS = 500
_ROUTED_OD_ONLY_SCALE_FLAGS = {
    "scales_routed_od_contribution": True,
    "scales_assignment_and_boundary_flow_contributions": False,
    "scales_additive_boundary_flow_contributions": False,
    "scales_fixed_operator_offsets": False,
}
_ROUTED_OD_ONLY_SCALE_SCOPES = frozenset(
    {"routed_od_only", "routed_od_contribution_only", "routed_od_contribution"}
)
_ROUTED_OD_COMPONENT_RECONCILIATION_TOLERANCE = {"rtol": 5e-6, "atol": 5e-6}
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


def _query_scale_inputs(
    *,
    artifact: TemporalBlockArtifactReader,
    fit_result: GravityEstimationResult,
    rho: float | None,
) -> dict[str, object] | None:
    """Validate the fitted scales needed by both the query and its summary."""
    row_count = artifact.number_of_measurements
    modeled = np.asarray(fit_result.predicted_measurements, dtype=np.float64)
    if modeled.shape != (row_count,):
        raise ValueError("mapping measurement dimension differs from fitted predictions.")
    if artifact.number_of_demand_cells != fit_result.free_od_demand.size:
        raise ValueError("mapping demand dimension differs from fitted free OD demand.")
    demand = np.asarray(fit_result.free_od_demand, dtype=np.float64)
    if not np.all(np.isfinite(demand)) or not np.all(np.isfinite(modeled)):
        raise ValueError("fitted OD demand and modeled measurements must be finite.")

    aggregate_flow_scale, scale_contract = _measurement_scale_contract(fit_result)
    if fit_result.observation_scales is None:
        if scale_contract in {"rowwise_observation_scale", "routed_od_only"}:
            raise ValueError(
                f"{scale_contract} results require persisted observation_scales."
            )
        if rho is None:
            return None
        observation_scale = float(rho)
        if not np.isfinite(observation_scale) or observation_scale <= 0.0:
            raise ValueError("rho must be finite and positive.")
        observation_scales = np.full(row_count, observation_scale, dtype=np.float64)
        observation_scale_source = "explicit_rho"
    else:
        if rho is not None:
            raise ValueError("rho must be omitted when the fit stores observation_scales.")
        observation_scales = np.asarray(fit_result.observation_scales, dtype=np.float64)
        if (
            observation_scales.shape != (row_count,)
            or not np.all(np.isfinite(observation_scales))
            or np.any(observation_scales <= 0.0)
        ):
            raise ValueError("fitted observation_scales are invalid.")
        observation_scale_source = "fit_result.observation_scales"

    additive_vectors = tuple(
        np.asarray(value, dtype=np.float64) for value in fit_result.additive_measurement_contributions
    )
    if any(value.shape != (row_count,) or not np.all(np.isfinite(value)) for value in additive_vectors):
        raise ValueError("fitted additive measurement contributions are invalid.")
    additive = np.sum(np.stack(additive_vectors), axis=0) if additive_vectors else np.zeros(row_count, dtype=np.float64)
    if not np.all(np.isfinite(additive)):
        raise ValueError("summed fitted additive measurement contributions are non-finite.")
    saved_od = None
    if fit_result.od_measurement_contribution is not None:
        saved_od = np.asarray(fit_result.od_measurement_contribution, dtype=np.float64)
        if saved_od.shape != (row_count,) or not np.all(np.isfinite(saved_od)):
            raise ValueError("fitted OD measurement contributions are invalid.")
    latent = None
    if fit_result.latent_measurements is not None:
        latent = np.asarray(fit_result.latent_measurements, dtype=np.float64)
        if latent.shape != (row_count,) or not np.all(np.isfinite(latent)):
            raise ValueError("fitted latent measurements are invalid.")
    elif additive_vectors or scale_contract == "routed_od_only":
        raise ValueError("fit is missing latent measurements required to separate fixed and additive components.")

    artifact_fixed = np.asarray(artifact.fixed_measurement_offset, dtype=np.float64)
    if artifact_fixed.shape != (row_count,) or not np.all(np.isfinite(artifact_fixed)):
        raise ValueError("mapping fixed measurement offset is invalid.")

    od_mean_multipliers: np.ndarray | None
    if scale_contract == "routed_od_only":
        if aggregate_flow_scale is not None:
            raise ValueError("routed-OD-only scale contract must not declare a scalar aggregate flow scale.")
        assert latent is not None
        effective_od_latent = latent - additive - artifact_fixed
        if not np.all(np.isfinite(effective_od_latent)):
            raise ValueError("routed-OD-only fitted latent component is non-finite.")
        od_mean_multipliers = None
        if saved_od is not None:
            zero_od = saved_od == 0.0
            q_tolerance = 1e-6 + 1e-6 * np.abs(latent)
            if np.any(zero_od & (np.abs(effective_od_latent) > q_tolerance)):
                raise ValueError(
                    "routed-OD-only fitted component is nonzero for a measurement with zero raw OD contribution."
                )
            od_mean_multipliers = np.zeros(row_count, dtype=np.float64)
            nonzero_od = ~zero_od
            od_mean_multipliers[nonzero_od] = (
                observation_scales[nonzero_od]
                * effective_od_latent[nonzero_od]
                / saved_od[nonzero_od]
            )
            if not np.all(np.isfinite(od_mean_multipliers)):
                raise ValueError("routed-OD-only modeled contribution scale is non-finite.")
            od_mean = observation_scales * effective_od_latent
            od_mean[zero_od] = 0.0
            fixed_mean = observation_scales * artifact_fixed
            additive_mean = observation_scales * additive
            modeled_without_floor = fixed_mean + additive_mean + od_mean
            floor_adjustment = modeled - modeled_without_floor
            if np.any(
                (floor_adjustment < 0.0)
                & ~np.isclose(
                    modeled_without_floor,
                    modeled,
                    **_ROUTED_OD_COMPONENT_RECONCILIATION_TOLERANCE,
                )
            ):
                raise ValueError("OD, fixed, and additive components exceed the saved modeled mean.")
            floor_adjustment = np.maximum(floor_adjustment, 0.0)
            if not np.allclose(
                modeled_without_floor + floor_adjustment,
                modeled,
                **_ROUTED_OD_COMPONENT_RECONCILIATION_TOLERANCE,
            ):
                raise ValueError("measurement component totals do not reproduce fitted predictions.")
    else:
        if aggregate_flow_scale is None:
            raise ValueError("supported measurement-scale contract is missing its aggregate flow scale.")
        od_mean_multipliers = observation_scales * aggregate_flow_scale
        if not np.all(np.isfinite(od_mean_multipliers)):
            raise ValueError("fitted modeled contribution scale is non-finite.")
        if saved_od is not None:
            if latent is None:
                fixed_offset = artifact_fixed
            else:
                fixed_offset = latent - aggregate_flow_scale * (saved_od + additive)
                if not np.allclose(fixed_offset, artifact_fixed, rtol=1e-6, atol=1e-8):
                    raise ValueError("fitted fixed offset differs from the mapping artifact.")
            fixed_mean = observation_scales * fixed_offset
            additive_mean = observation_scales * aggregate_flow_scale * additive
            od_mean = observation_scales * aggregate_flow_scale * saved_od
            floor_adjustment = modeled - fixed_mean - additive_mean - od_mean
            tolerance = 1e-6 + 1e-6 * np.abs(modeled)
            if np.any(floor_adjustment < -tolerance):
                raise ValueError("OD, fixed, and additive components exceed the saved modeled mean.")
            floor_adjustment = np.maximum(floor_adjustment, 0.0)
            if not np.allclose(
                fixed_mean + additive_mean + od_mean + floor_adjustment,
                modeled,
                rtol=1e-6,
                atol=1e-8,
            ):
                raise ValueError("measurement component totals do not reproduce fitted predictions.")

    return {
        "demand": demand,
        "modeled": modeled,
        "observation_scales": observation_scales,
        "observation_scale_source": observation_scale_source,
        "aggregate_flow_scale": aggregate_flow_scale,
        "measurement_scale_contract": scale_contract,
        "scale_component_scope": (
            "routed_od_only" if scale_contract == "routed_od_only" else "all_latent_components"
        ),
        "od_mean_multipliers": od_mean_multipliers,
        "additive": additive,
    }


def measurement_od_query_descriptor(
    *,
    artifact: TemporalBlockArtifactReader,
    fit_result: GravityEstimationResult,
    fit_provenance: Mapping[str, object],
    runtime_package_revision: str,
    rho: float | None = None,
) -> dict[str, object]:
    """Describe an on-demand mapping source without expanding its sparse entries."""
    scales = _query_scale_inputs(artifact=artifact, fit_result=fit_result, rho=rho)
    base: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "query_descriptor_version": 2,
        "available": scales is not None,
        "query_mode": "local_service",
        "query_path": "/api/measurement-od-contributions",
        "reverse_query_path": "/api/od-measurement-contributions",
        "reverse_query_available": scales is not None and scales.get("od_mean_multipliers") is not None,
        "fit_fingerprint": fit_result.model_fingerprint,
        "assignment_fingerprint": fit_provenance.get("assignment_fingerprint"),
        "mapping_fingerprint": fit_provenance.get("mapping_fingerprint"),
        "artifact_timetable_fingerprint": artifact.identity.timetable_fingerprint,
        "artifact_measurement_mapping_fingerprint": artifact.identity.measurement_mapping_fingerprint,
        "mapping_artifact_fingerprint": artifact.identity.fingerprint,
        "mapping_manifest_sha256": artifact.manifest_sha256,
        "mapping_artifact_schema_version": artifact.artifact_schema_version,
        "runtime_package_revision": runtime_package_revision,
        "fit_source_package_revision": fit_provenance.get("package_revision"),
        "mapping_source_package_revision": artifact.source_package_revision,
        "canonical_index_fingerprint": artifact.canonical_index.artifact_fingerprint,
        "binding_fingerprint": artifact.canonical_index.binding_fingerprint,
        "od_layout_fingerprint": artifact.canonical_index.source_od_layout_fingerprint,
        "number_of_measurements": artifact.number_of_measurements,
        "number_of_free_od_cells": artifact.number_of_demand_cells,
        "number_of_physical_od_cells": artifact.canonical_index.number_of_physical_demand_cells,
        "nonzero_pairs": artifact.nonzero_entries,
        "query_block_scope": "canonical_measurement_interval",
        "block_checksum_policy": "verify_each_selected_block_before_use",
        "fixed_positive_od_attribution_available": not any(
            cell.role == "fixed_positive" for cell in artifact.canonical_index.demand_cells
        ),
        "fixed_positive_od_attribution_message": None,
    }
    if not base["fixed_positive_od_attribution_available"]:
        base["fixed_positive_od_attribution_message"] = (
            "Fixed-positive OD demand is aggregated in the mapping's fixed offset; "
            "the artifact has no per-cell coefficients for those cells."
        )
    if scales is None:
        base["message"] = (
            "The fit does not persist its measurement-scale vector; pass rho when "
            "building the bundle to enable exact on-demand modeled-mean contributions."
        )
        return base
    base.update(
        {
            "aggregate_flow_scale": scales["aggregate_flow_scale"],
            "measurement_scale_contract": scales["measurement_scale_contract"],
            "scale_component_scope": scales["scale_component_scope"],
            "observation_scale_source": scales["observation_scale_source"],
            "explicit_rho": float(rho) if scales["observation_scale_source"] == "explicit_rho" else None,
        }
    )
    if scales["measurement_scale_contract"] not in {"rowwise_observation_scale", "routed_od_only"}:
        # Retain this legacy field where its old semantics remain accurate.
        base["journey_aggregate_scale"] = scales["aggregate_flow_scale"]
    if scales["measurement_scale_contract"] == "routed_od_only":
        base["scale_component_semantics"] = {
            "routed_od": "recovered_from_saved_latent_measurement_and_allocated_proportionally_to_raw_od_flow",
            "additive": "multiplied_only_by_outer_observation_scale",
            "fixed_operator_offset": "multiplied_only_by_outer_observation_scale",
            "mean_floor": "residual_to_saved_modeled_mean",
        }
        if not base["reverse_query_available"]:
            base["reverse_query_message"] = (
                "Reverse OD-to-measurement contributions require the persisted raw OD measurement component."
            )
    else:
        base["scale_component_semantics"] = {
            "routed_od": "multiplied_by_aggregate_flow_scale_and_observation_scale",
            "additive": "multiplied_by_aggregate_flow_scale_and_observation_scale",
            "fixed_operator_offset": "multiplied_by_aggregate_flow_scale_and_observation_scale",
            "mean_floor": "residual_to_saved_modeled_mean",
        }
    return base


class MeasurementODContributionQuery:
    """Bounded, row-selective queries over a persisted temporal mapping."""

    def __init__(
        self,
        *,
        artifact: TemporalBlockArtifactReader,
        fit_result: GravityEstimationResult,
        rho: float | None = None,
        max_cached_pages: int = 16,
        max_page_size: int = 500,
    ) -> None:
        if max_cached_pages <= 0 or max_page_size <= 0:
            raise ValueError("query cache and page bounds must be positive.")
        scales = _query_scale_inputs(artifact=artifact, fit_result=fit_result, rho=rho)
        if scales is None:
            raise ValueError(
                "the fit has no persisted observation scale; pass rho for exact modeled-mean contributions."
            )
        self.artifact = artifact
        self.fit_result = fit_result
        self.rho = rho
        self.max_cached_pages = int(max_cached_pages)
        self.max_page_size = int(max_page_size)
        self._scales = scales
        self._cache: OrderedDict[tuple[object, ...], dict[str, object]] = OrderedDict()
        self._namespace = (fit_result.model_fingerprint, artifact.identity.fingerprint)
        self.cache_hits = 0
        self.cache_misses = 0

        index = artifact.canonical_index
        self._full_index_by_column = np.full(artifact.number_of_demand_cells, -1, dtype=np.int64)
        self._operator_column_by_full = np.full(index.number_of_physical_demand_cells, -1, dtype=np.int64)
        self._departure_code_by_column = np.full(artifact.number_of_demand_cells, -1, dtype=np.int32)
        self._departure_intervals: list[str] = []
        departure_codes: dict[str, int] = {}
        for cell in index.demand_cells:
            if cell.operator_column is None:
                continue
            column = int(cell.operator_column)
            if self._full_index_by_column[column] != -1:
                raise ValueError("canonical free-OD operator column is duplicated.")
            code = departure_codes.get(cell.departure_interval_id)
            if code is None:
                code = len(self._departure_intervals)
                departure_codes[cell.departure_interval_id] = code
                self._departure_intervals.append(cell.departure_interval_id)
            self._full_index_by_column[column] = cell.full_index
            self._operator_column_by_full[cell.full_index] = column
            self._departure_code_by_column[column] = code
        if (
            np.any(self._full_index_by_column < 0)
            or np.any(self._departure_code_by_column < 0)
            or not np.array_equal(
                self._operator_column_by_full[self._full_index_by_column],
                np.arange(artifact.number_of_demand_cells, dtype=np.int64),
            )
        ):
            raise ValueError("canonical free-OD operator columns are incomplete.")
        self._measurement_by_index = index.measurements
        self._cell_by_full_index = index.demand_cells

    @property
    def cache_namespace(self) -> tuple[str, str]:
        """Fit and mapping fingerprints used to isolate recent page entries."""
        return self._namespace

    def clear_cache(self) -> None:
        self._cache.clear()

    def query(
        self,
        measurement_index: int,
        *,
        page: int = 0,
        page_size: int = 100,
    ) -> dict[str, object]:
        if isinstance(measurement_index, bool) or not isinstance(measurement_index, (int, np.integer)):
            raise TypeError("measurement_index must be an integer.")
        measurement_index = int(measurement_index)
        if measurement_index < 0 or measurement_index >= len(self._measurement_by_index):
            raise IndexError("measurement_index is outside the canonical mapping.")
        if page < 0 or page_size <= 0 or page_size > self.max_page_size:
            raise ValueError(f"page must be non-negative and page_size must be between 1 and {self.max_page_size}.")
        if (page + 1) * page_size > 5_000:
            raise ValueError("requested ranked page exceeds the bounded contribution-query window.")
        cache_key = (
            *self._namespace,
            "measurement",
            measurement_index,
            page,
            page_size,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            self._cache.move_to_end(cache_key)
            return copy.deepcopy(cached)

        self.cache_misses += 1
        measurement = self._measurement_by_index[measurement_index]
        result = self._query_measurement_interval_row(
            measurement_index=measurement_index,
            measurement_interval_id=measurement.interval_id,
            page=page,
            page_size=page_size,
        )
        self._cache[cache_key] = result
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.max_cached_pages:
            self._cache.popitem(last=False)
        return copy.deepcopy(result)

    def query_od(
        self,
        od_index: int,
        *,
        page: int = 0,
        page_size: int = 100,
    ) -> dict[str, object]:
        """Return affected measurement rows for one physical OD-time cell."""
        if isinstance(od_index, bool) or not isinstance(od_index, (int, np.integer)):
            raise TypeError("od_index must be an integer.")
        od_index = int(od_index)
        cells = self.artifact.canonical_index.demand_cells
        if od_index < 0 or od_index >= len(cells):
            raise IndexError("od_index is outside the canonical mapping.")
        if page < 0 or page_size <= 0 or page_size > self.max_page_size:
            raise ValueError(f"page must be non-negative and page_size must be between 1 and {self.max_page_size}.")
        if (page + 1) * page_size > 5_000:
            raise ValueError("requested ranked page exceeds the bounded contribution-query window.")
        cache_key = (*self._namespace, "od", od_index, page, page_size)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            self._cache.move_to_end(cache_key)
            return copy.deepcopy(cached)
        self.cache_misses += 1

        cell = cells[od_index]
        column = cell.operator_column
        if column is None:
            result: dict[str, object] = {
                "schema_version": _SCHEMA_VERSION,
                "fit_fingerprint": self._namespace[0],
                "mapping_artifact_fingerprint": self._namespace[1],
                "od_index": od_index,
                "page": page,
                "page_size": page_size,
                "total": 0,
                "rows": [],
            }
        else:
            result = self._query_od_interval(
                od_index=od_index,
                operator_column=column,
                departure_interval_id=cell.departure_interval_id,
                page=page,
                page_size=page_size,
            )
        self._cache[cache_key] = result
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.max_cached_pages:
            self._cache.popitem(last=False)
        return copy.deepcopy(result)

    def _query_od_interval(
        self,
        *,
        od_index: int,
        operator_column: int,
        departure_interval_id: str,
        page: int,
        page_size: int,
    ) -> dict[str, object]:
        od_mean_multipliers = self._scales.get("od_mean_multipliers")
        if od_mean_multipliers is None:
            raise ValueError(
                "reverse OD-to-measurement contributions are unavailable because the fit does not persist its raw OD measurement component."
            )
        od_mean_multipliers = np.asarray(od_mean_multipliers, dtype=np.float64)
        offset = page * page_size
        with tempfile.TemporaryDirectory(prefix="gravity-od-measurement-query-") as temporary:
            connection = sqlite3.connect(Path(temporary) / "selected_od.sqlite")
            try:
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("PRAGMA synchronous=OFF")
                connection.execute("PRAGMA temp_store=FILE")
                connection.execute("PRAGMA cache_size=-4096")
                connection.execute(
                    "CREATE TABLE contributions ("
                    "measurement_index INTEGER PRIMARY KEY, operator_coefficient REAL NOT NULL, "
                    "linear_flow_contribution REAL NOT NULL, "
                    "modeled_mean_contribution REAL NOT NULL) WITHOUT ROWID"
                )
                demand = float(np.asarray(self._scales["demand"])[operator_column])
                for block in self.artifact.iter_blocks_for_departure_interval(departure_interval_id):
                    for start in range(0, block.nonzero_entries, _WRITE_BATCH_SIZE):
                        stop = min(block.nonzero_entries, start + _WRITE_BATCH_SIZE)
                        block_columns = block.column_indices[start:stop]
                        selected = np.flatnonzero(block_columns == operator_column)
                        if not selected.size:
                            continue
                        rows = np.asarray(block.row_indices[start:stop][selected], dtype=np.int64)
                        coefficients = np.asarray(block.values[start:stop][selected], dtype=np.float64)
                        flows = coefficients * demand
                        means = flows * od_mean_multipliers[rows]
                        if not np.all(np.isfinite(flows)) or not np.all(np.isfinite(means)):
                            raise ValueError("selected OD contributions are non-finite.")
                        connection.executemany(
                            "INSERT INTO contributions VALUES (?, ?, ?, ?) "
                            "ON CONFLICT(measurement_index) DO UPDATE SET "
                            "operator_coefficient=operator_coefficient+excluded.operator_coefficient, "
                            "linear_flow_contribution=linear_flow_contribution+excluded.linear_flow_contribution, "
                            "modeled_mean_contribution=modeled_mean_contribution+excluded.modeled_mean_contribution",
                            [
                                (int(row), float(coefficient), float(flow), float(mean))
                                for row, coefficient, flow, mean in zip(rows, coefficients, flows, means, strict=True)
                            ],
                        )
                connection.commit()
                total = int(connection.execute("SELECT COUNT(*) FROM contributions").fetchone()[0])
                rows = connection.execute(
                    "SELECT measurement_index, operator_coefficient, linear_flow_contribution, modeled_mean_contribution "
                    "FROM contributions ORDER BY ABS(modeled_mean_contribution) DESC, measurement_index ASC LIMIT ? OFFSET ?",
                    (page_size, offset),
                ).fetchall()
            finally:
                connection.close()
        return {
            "schema_version": _SCHEMA_VERSION,
            "fit_fingerprint": self._namespace[0],
            "mapping_artifact_fingerprint": self._namespace[1],
            "od_index": od_index,
            "page": page,
            "page_size": page_size,
            "total": total,
            "measurement_scale_contract": self._scales["measurement_scale_contract"],
            "scale_component_scope": self._scales["scale_component_scope"],
            "rows": [
                {
                    "measurement_index": int(measurement_index),
                    "od_index": od_index,
                    "departure_time_bin": departure_interval_id,
                    "operator_coefficient": float(coefficient),
                    "linear_flow_contribution": float(flow),
                    "modeled_mean_contribution": float(mean),
                }
                for measurement_index, coefficient, flow, mean in rows
            ],
        }

    def _query_measurement_interval_row(
        self,
        *,
        measurement_index: int,
        measurement_interval_id: str,
        page: int,
        page_size: int,
    ) -> dict[str, object]:
        offset = page * page_size
        with tempfile.TemporaryDirectory(prefix="gravity-measurement-query-") as temporary:
            database_path = Path(temporary) / "selected_measurement.sqlite"
            connection = sqlite3.connect(database_path)
            try:
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("PRAGMA synchronous=OFF")
                connection.execute("PRAGMA temp_store=FILE")
                connection.execute("PRAGMA cache_size=-4096")
                connection.execute(
                    "CREATE TABLE contributions ("
                    "od_index INTEGER PRIMARY KEY, operator_coefficient REAL NOT NULL, "
                    "linear_flow_contribution REAL NOT NULL, "
                    "modeled_mean_contribution REAL NOT NULL) WITHOUT ROWID"
                )
                demand = np.asarray(self._scales["demand"])
                od_mean_multipliers = self._scales.get("od_mean_multipliers")
                if od_mean_multipliers is None:
                    initial_mean = 0.0
                else:
                    initial_mean = float(np.asarray(od_mean_multipliers)[measurement_index])
                for block in self.artifact.iter_blocks_for_measurement_interval(measurement_interval_id):
                    for start in range(0, block.nonzero_entries, _WRITE_BATCH_SIZE):
                        stop = min(block.nonzero_entries, start + _WRITE_BATCH_SIZE)
                        block_rows = block.row_indices[start:stop]
                        selected = np.flatnonzero(block_rows == measurement_index)
                        if not selected.size:
                            continue
                        columns = np.asarray(block.column_indices[start:stop][selected], dtype=np.int64)
                        coefficients = np.asarray(block.values[start:stop][selected], dtype=np.float64)
                        flows = coefficients * demand[columns]
                        means = flows * initial_mean
                        if not np.all(np.isfinite(flows)) or not np.all(np.isfinite(means)):
                            raise ValueError("selected measurement contributions are non-finite.")
                        rows_to_write = [
                            (
                                int(self._full_index_by_column[int(column)]),
                                float(coefficient),
                                float(flow),
                                float(mean),
                            )
                            for column, coefficient, flow, mean in zip(columns, coefficients, flows, means, strict=True)
                        ]
                        connection.executemany(
                            "INSERT INTO contributions VALUES (?, ?, ?, ?) "
                            "ON CONFLICT(od_index) DO UPDATE SET "
                            "operator_coefficient=operator_coefficient+excluded.operator_coefficient, "
                            "linear_flow_contribution=linear_flow_contribution+excluded.linear_flow_contribution, "
                            "modeled_mean_contribution=modeled_mean_contribution+excluded.modeled_mean_contribution",
                            rows_to_write,
                        )
                connection.commit()
                total = int(connection.execute("SELECT COUNT(*) FROM contributions").fetchone()[0])
                raw_od = float(
                    connection.execute(
                        "SELECT COALESCE(SUM(linear_flow_contribution), 0.0) FROM contributions"
                    ).fetchone()[0]
                )
                components = self._components_for_row(measurement_index, raw_od)
                if self._scales["measurement_scale_contract"] == "routed_od_only":
                    if raw_od == 0.0:
                        od_mean_multiplier = 0.0
                    else:
                        od_mean_multiplier = float(components["od_contribution"]) / raw_od
                    connection.execute(
                        "UPDATE contributions SET modeled_mean_contribution = linear_flow_contribution * ?",
                        (od_mean_multiplier,),
                    )
                    connection.commit()
                rows = connection.execute(
                    "SELECT od_index, operator_coefficient, linear_flow_contribution, modeled_mean_contribution "
                    "FROM contributions ORDER BY ABS(modeled_mean_contribution) DESC, od_index ASC LIMIT ? OFFSET ?",
                    (page_size, offset),
                ).fetchall()
            finally:
                connection.close()

        demand = np.asarray(self._scales["demand"])
        contributors: list[dict[str, object]] = []
        for od_index, coefficient, linear_flow, modeled_mean in rows:
            cell = self._cell_by_full_index[int(od_index)]
            column = cell.operator_column
            if column is None:
                raise ValueError("a sparse mapping entry resolved to a non-free OD cell.")
            departure_code = int(self._departure_code_by_column[column])
            contributors.append(
                {
                    "measurement_index": measurement_index,
                    "od_index": int(od_index),
                    "departure_time_bin": self._departure_intervals[departure_code],
                    "operator_coefficient": float(coefficient),
                    "linear_flow_contribution": float(linear_flow),
                    "modeled_mean_contribution": float(modeled_mean),
                    "origin_stop_id": cell.origin_id,
                    "destination_stop_id": cell.destination_id,
                    "estimated_demand": float(demand[column]),
                }
            )
        return {
            "schema_version": _SCHEMA_VERSION,
            "fit_fingerprint": self._namespace[0],
            "mapping_artifact_fingerprint": self._namespace[1],
            "measurement_index": measurement_index,
            "page": page,
            "page_size": page_size,
            "total": total,
            "measurement_scale_contract": self._scales["measurement_scale_contract"],
            "scale_component_scope": self._scales["scale_component_scope"],
            "modeled_mean": components["modeled_mean"],
            "rows": contributors,
            "components": components,
        }

    def _components_for_row(self, measurement_index: int, raw_od: float) -> dict[str, float]:
        result = self.fit_result
        expected_od = result.od_measurement_contribution
        if expected_od is not None:
            expected = np.asarray(expected_od, dtype=np.float64)
            if expected.shape != (self.artifact.number_of_measurements,) or not np.isclose(
                raw_od, float(expected[measurement_index]), rtol=1e-6, atol=1e-8
            ):
                raise ValueError("mapping OD contribution differs from the fitted OD measurement component.")
        additive = float(np.asarray(self._scales["additive"])[measurement_index])
        scale = float(np.asarray(self._scales["observation_scales"])[measurement_index])
        modeled = float(np.asarray(self._scales["modeled"])[measurement_index])
        if self._scales["measurement_scale_contract"] == "routed_od_only":
            if result.latent_measurements is None:
                raise ValueError("routed-OD-only decomposition requires persisted latent measurements.")
            latent = np.asarray(result.latent_measurements, dtype=np.float64)
            fixed_offset = float(np.asarray(self.artifact.fixed_measurement_offset)[measurement_index])
            effective_od_latent = float(latent[measurement_index]) - additive - fixed_offset
            if raw_od == 0.0:
                if not np.isclose(effective_od_latent, 0.0, rtol=1e-6, atol=1e-8):
                    raise ValueError(
                        "routed-OD-only fitted component is nonzero for a measurement with zero raw OD contribution."
                    )
                od_mean = 0.0
            else:
                od_mean = scale * effective_od_latent
            fixed_mean = scale * fixed_offset
            additive_mean = scale * additive
        elif result.latent_measurements is None:
            if result.additive_measurement_contributions:
                raise ValueError(
                    "fit is missing latent measurements required to separate fixed and additive components."
                )
            fixed_offset = float(self.artifact.fixed_measurement_offset[measurement_index])
            aggregate_flow_scale = float(self._scales["aggregate_flow_scale"])
            fixed_mean = scale * fixed_offset
            additive_mean = scale * aggregate_flow_scale * additive
            od_mean = scale * aggregate_flow_scale * raw_od
        else:
            latent = np.asarray(result.latent_measurements, dtype=np.float64)
            aggregate_flow_scale = float(self._scales["aggregate_flow_scale"])
            fixed_offset = float(
                latent[measurement_index]
                - float(self._scales["aggregate_flow_scale"]) * (raw_od + additive)
            )
            if not np.isclose(
                fixed_offset,
                float(self.artifact.fixed_measurement_offset[measurement_index]),
                rtol=1e-6,
                atol=1e-8,
            ):
                raise ValueError("fitted fixed offset differs from the mapping artifact.")
            fixed_mean = scale * fixed_offset
            additive_mean = scale * aggregate_flow_scale * additive
            od_mean = scale * aggregate_flow_scale * raw_od
        modeled_without_floor = fixed_mean + additive_mean + od_mean
        floor = modeled - modeled_without_floor
        if self._scales["measurement_scale_contract"] == "routed_od_only":
            if floor < 0.0 and not np.isclose(
                modeled_without_floor,
                modeled,
                **_ROUTED_OD_COMPONENT_RECONCILIATION_TOLERANCE,
            ):
                raise ValueError("OD, fixed, and additive components exceed the saved modeled mean.")
            floor = max(floor, 0.0)
            if not np.isclose(
                modeled_without_floor + floor,
                modeled,
                **_ROUTED_OD_COMPONENT_RECONCILIATION_TOLERANCE,
            ):
                raise ValueError("measurement component totals do not reproduce fitted predictions.")
        else:
            tolerance = 1e-6 + 1e-6 * abs(modeled)
            if floor < -tolerance:
                raise ValueError("OD, fixed, and additive components exceed the saved modeled mean.")
            floor = max(floor, 0.0)
            if not np.isclose(fixed_mean + additive_mean + od_mean + floor, modeled, rtol=1e-6, atol=1e-8):
                raise ValueError("measurement component totals do not reproduce fitted predictions.")
        return {
            "fixed_offset_contribution": fixed_mean,
            "additive_contribution": additive_mean,
            "od_contribution": od_mean,
            "mean_floor_adjustment": floor,
            "modeled_mean": modeled,
        }


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


def _measurement_scale_contract(
    result: GravityEstimationResult,
) -> tuple[float | None, str]:
    """Resolve a supported generic scale contract without interpreting opaque models."""
    payload = result.measurement_model
    if payload is None:
        return 1.0, "unit_flow_scale"
    if not isinstance(payload, Mapping):
        raise ValueError("fitted measurement-scale contract must be an object.")

    declared_scope_fields = tuple(
        field
        for field in ("scale_scope", "scale_application_scope", "scope")
        if field in payload
    )
    routed_flags_present = any(
        field in payload for field in _ROUTED_OD_ONLY_SCALE_FLAGS
    )
    if declared_scope_fields or routed_flags_present:
        if not declared_scope_fields:
            raise ValueError(
                "routed-OD measurement-scale contract is missing its declared scale scope."
            )
        declared_scopes = tuple(payload[field] for field in declared_scope_fields)
        if any(scope != declared_scopes[0] for scope in declared_scopes[1:]):
            raise ValueError("fitted measurement-scale scope declarations contradict each other.")
        scope = declared_scopes[0]
        if not isinstance(scope, str) or scope not in _ROUTED_OD_ONLY_SCALE_SCOPES:
            raise ValueError(f"unsupported fitted measurement-scale scope: {scope!r}.")
        missing_flags = tuple(
            field for field in _ROUTED_OD_ONLY_SCALE_FLAGS if field not in payload
        )
        if missing_flags:
            raise ValueError(
                "routed-OD measurement-scale contract is incomplete; missing component flags: "
                + ", ".join(sorted(missing_flags))
                + "."
            )
        contradictory = tuple(
            field
            for field, expected in _ROUTED_OD_ONLY_SCALE_FLAGS.items()
            if not isinstance(payload[field], bool) or payload[field] is not expected
        )
        if contradictory:
            raise ValueError(
                "routed-OD measurement-scale scope contradicts its component flags: "
                + ", ".join(sorted(contradictory))
                + "."
            )
        return None, "routed_od_only"

    if payload.get("schema_version", 1) != 1:
        raise ValueError("fitted measurement-scale contract schema is unsupported.")

    payload_type = payload.get("type")
    if payload_type == "journey_latent_class" or (
        payload_type is None and "num_classes" in payload
    ):
        try:
            model = GravityJourneyLatentClassModel.from_dict(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "fitted journey latent-class scale contract is incomplete or invalid."
            ) from error
        contract = "journey_latent_class"
    elif (
        isinstance(payload_type, str)
        and payload_type in {"rowwise_measurement_scale", "gravity_observation"}
    ) or (
        payload_type is None
        and {
            "measurement_types",
            "boarding_scale",
            "alighting_scale",
        }.issubset(payload)
    ):
        if not {
            "measurement_types",
            "boarding_scale",
            "alighting_scale",
        }.issubset(payload):
            raise ValueError(
                "row-wise measurement-scale contract is missing its scale specification."
            )
        if result.observation_scales is None:
            raise ValueError(
                "row-wise measurement-scale results require persisted observation_scales."
            )
        try:
            observation_model = GravityObservationModel.from_dict(payload)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "row-wise measurement-scale contract is incomplete or invalid."
            ) from error
        row_count = int(np.asarray(result.predicted_measurements).size)
        if (
            observation_model.measurement_types is not None
            and len(observation_model.measurement_types) != row_count
        ):
            raise ValueError(
                "row-wise measurement-scale types do not match the fitted measurement dimension."
            )
        stored_scales = np.asarray(result.observation_scales, dtype=np.float64)
        if observation_model.measurement_types is None:
            row_scales = np.ones(row_count, dtype=np.float64)
        else:
            row_scales = np.asarray(
                [
                    observation_model.boarding_scale
                    if measurement_type == "boarding"
                    else observation_model.alighting_scale
                    for measurement_type in observation_model.measurement_types
                ],
                dtype=np.float64,
            )
        if (
            stored_scales.shape != (row_count,)
            or not np.all(np.isfinite(stored_scales))
            or np.any(stored_scales <= 0.0)
        ):
            raise ValueError("fitted observation_scales are invalid.")
        if row_count:
            normalized_scales = stored_scales / row_scales
            if not np.allclose(
                normalized_scales,
                np.full(row_count, normalized_scales[0]),
                rtol=1e-6,
                atol=1e-8,
            ):
                raise ValueError(
                    "persisted observation_scales are inconsistent with the row-wise measurement-scale contract."
                )
        return 1.0, "rowwise_observation_scale"
    else:
        raise ValueError(
            f"unsupported fitted measurement-scale contract type: {payload_type!r}."
        )

    if not model.parameter_names:
        raw = np.empty(0, dtype=np.float64)
    else:
        name_to_position = {name: index for index, name in enumerate(result.parameter_names)}
        missing = [name for name in model.parameter_names if name not in name_to_position]
        if missing:
            raise ValueError("fitted measurement-model parameters are missing from parameter_names.")
        raw = np.asarray(
            [result.raw_parameters[name_to_position[name]] for name in model.parameter_names],
            dtype=np.float64,
        )
    scale = float(np.asarray(model.aggregate_scale(raw, dtype=np.float64)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("fitted measurement-model aggregate scale is invalid.")
    return scale, contract


def write_measurement_od_contributions_dataset(
    *,
    destination: str | Path,
    artifact: TemporalBlockArtifactReader,
    fit_result: GravityEstimationResult,
    rho: float | None = None,
) -> dict[str, object]:
    """Stream a validated sparse operator into CSV partitions for the viewer.

    A missing saved observation-scale vector requires an explicit ``rho`` for
    scalar-scale fits; row-wise scale contracts require their persisted vector.
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
    aggregate_flow_scale, scale_contract = _measurement_scale_contract(fit_result)
    if aggregate_flow_scale is None:
        raise ValueError(
            "routed-OD-only scale semantics require bounded on-demand queries and cannot be written as a materialized contribution dataset."
        )

    if fit_result.observation_scales is None:
        if scale_contract == "rowwise_observation_scale":
            raise ValueError(
                "row-wise measurement-scale results require persisted observation_scales."
            )
        if rho is None:
            return {
                "available": False,
                "message": ("The fit does not persist its measurement-scale vector; pass rho to generate exact modeled-mean contributions."),
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
        if observation_scales.shape != (row_count,) or not np.all(np.isfinite(observation_scales)) or np.any(observation_scales <= 0.0):
            raise ValueError("fitted observation_scales are invalid.")
        observation_scale_source = "fit_result.observation_scales"

    additive_vectors = tuple(np.asarray(value, dtype=np.float64) for value in fit_result.additive_measurement_contributions)
    if any(value.shape != (row_count,) or not np.all(np.isfinite(value)) for value in additive_vectors):
        raise ValueError("fitted additive measurement contributions are invalid.")
    additive = np.sum(np.stack(additive_vectors), axis=0) if additive_vectors else np.zeros(row_count, dtype=np.float64)

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
    if any(not value for value in departure_interval_by_column) or any(value < 0 for value in full_index_by_column):
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
        connection.execute("CREATE INDEX contributions_by_interval ON contributions (departure_time_bin, measurement_index, od_index)")
        for block in artifact.iter_blocks():
            for start in range(0, block.nonzero_entries, _WRITE_BATCH_SIZE):
                stop = min(block.nonzero_entries, start + _WRITE_BATCH_SIZE)
                rows = np.asarray(block.row_indices[start:stop], dtype=np.int64)
                columns = np.asarray(block.column_indices[start:stop], dtype=np.int64)
                coefficients = np.asarray(block.values[start:stop], dtype=np.float64)
                flows = coefficients * demand[columns]
                np.add.at(raw_od, rows, flows)
                means = flows * aggregate_flow_scale * observation_scales[rows]
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
            if persisted_od_array.shape != (row_count,) or not np.allclose(raw_od, persisted_od_array, rtol=1e-6, atol=1e-8):
                raise ValueError("mapping OD contribution differs from the fitted OD measurement component.")

        if fit_result.latent_measurements is None:
            if additive_vectors:
                raise ValueError("fit is missing latent measurements required to separate fixed and additive components.")
            fixed_offset = np.asarray(artifact.fixed_measurement_offset, dtype=np.float64)
        else:
            latent = np.asarray(fit_result.latent_measurements, dtype=np.float64)
            if latent.shape != (row_count,) or not np.all(np.isfinite(latent)):
                raise ValueError("fitted latent measurements are invalid.")
            fixed_offset = latent - aggregate_flow_scale * (raw_od + additive)
            if not np.allclose(
                fixed_offset,
                np.asarray(artifact.fixed_measurement_offset, dtype=np.float64),
                rtol=1e-6,
                atol=1e-8,
            ):
                raise ValueError("fitted fixed offset differs from the mapping artifact.")

        fixed_mean = observation_scales * fixed_offset
        additive_mean = observation_scales * aggregate_flow_scale * additive
        od_mean = observation_scales * aggregate_flow_scale * raw_od
        floor_adjustment = modeled - fixed_mean - additive_mean - od_mean
        tolerance = 1e-6 + 1e-6 * np.abs(modeled)
        if np.any(floor_adjustment < -tolerance):
            raise ValueError("OD, fixed, and additive components exceed the saved modeled mean.")
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

        bins = tuple(str(row[0]) for row in connection.execute("SELECT DISTINCT departure_time_bin FROM contributions ORDER BY departure_time_bin"))
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
                        output = partition_root / (f"departure-{_slug(departure_bin)}-{chunk_index:05d}.csv")
                        stream = output.open("w", newline="", encoding="utf-8")
                        writer = csv.DictWriter(stream, fieldnames=_CONTRIBUTION_COLUMNS)
                        writer.writeheader()
                    assert writer is not None
                    writer.writerow(dict(zip(_CONTRIBUTION_COLUMNS, row, strict=True)))
                    chunk_count += 1
                    total_rows += 1
            finish_partition()

        component_root = target / "components"
        component_root.mkdir()
        component_partitions: list[dict[str, object]] = []
        cursor = connection.execute(
            "SELECT measurement_index, fixed_offset_contribution, "
            "additive_contribution, mean_floor_adjustment, modeled_mean "
            "FROM components ORDER BY measurement_index"
        )
        while batch := cursor.fetchmany(_COMPONENT_PARTITION_ROWS):
            start_index = int(batch[0][0])
            output = component_root / f"measurements-{start_index:09d}.csv"
            with output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=_COMPONENT_COLUMNS)
                writer.writeheader()
                writer.writerows(dict(zip(_COMPONENT_COLUMNS, row, strict=True)) for row in batch)
            component_partitions.append(
                {
                    "start_index": start_index,
                    "row_count": len(batch),
                    "path": output.relative_to(target.parent.parent).as_posix(),
                    "size_bytes": output.stat().st_size,
                    "sha256": _sha256(output),
                }
            )
        connection.close()
        connection = None
    finally:
        if connection is not None:
            connection.close()
        shutil.rmtree(working_directory, ignore_errors=True)

    fixed_positive_cells = sum(cell.role == "fixed_positive" for cell in artifact.canonical_index.demand_cells)
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
        "aggregate_flow_scale": aggregate_flow_scale,
        "measurement_scale_contract": scale_contract,
        "observation_scale_source": observation_scale_source,
        "explicit_rho": None if observation_scale_source != "explicit_rho" else float(rho),
        "partition_field": "departure_time_bin",
        "contribution_columns": list(_CONTRIBUTION_COLUMNS),
        "component_columns": list(_COMPONENT_COLUMNS),
        "component_table": {
            "partition_rows": _COMPONENT_PARTITION_ROWS,
            "partitions": component_partitions,
            "row_count": row_count,
        },
        "fixed_positive_od_attribution_available": fixed_positive_cells == 0,
        "fixed_positive_od_attribution_message": (
            None
            if fixed_positive_cells == 0
            else ("Fixed-positive OD demand is aggregated in the mapping's fixed offset; the artifact has no per-cell coefficients for those cells.")
        ),
        "partitions": descriptors,
    }
    if scale_contract != "rowwise_observation_scale":
        manifest["journey_aggregate_scale"] = aggregate_flow_scale
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
        "fixed_positive_od_attribution_message": manifest["fixed_positive_od_attribution_message"],
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
        or manifest.get("assignment_fingerprint") != payload.get("assignment_fingerprint")
        or manifest.get("mapping_artifact_fingerprint") != payload.get("mapping_artifact_fingerprint")
        or manifest.get("canonical_index_fingerprint") != payload.get("canonical_index_fingerprint")
        or manifest.get("binding_fingerprint") != payload.get("binding_fingerprint")
        or manifest.get("od_layout_fingerprint") != payload.get("od_layout_fingerprint")
        or manifest.get("observation_scale_source") != payload.get("observation_scale_source")
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
        if not path.is_file() or path.stat().st_size != descriptor.get("size_bytes") or _sha256(path) != descriptor.get("sha256"):
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
    component_partitions = component.get("partitions")
    if isinstance(component_partitions, list):
        partition_limit = component.get("partition_rows")
        if partition_limit != _COMPONENT_PARTITION_ROWS:
            raise ValueError("measurement OD component partition bound is invalid.")
        expected_index = 0
        for descriptor in component_partitions:
            if not isinstance(descriptor, Mapping):
                raise ValueError("measurement OD component partition descriptor is invalid.")
            relative = Path(str(descriptor.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("measurement OD component partition path is unsafe.")
            component_path = bundle_root / relative
            if (
                not component_path.is_file()
                or component_path.stat().st_size != descriptor.get("size_bytes")
                or _sha256(component_path) != descriptor.get("sha256")
            ):
                raise ValueError("measurement OD component partition checksum mismatch.")
            with component_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != _COMPONENT_COLUMNS:
                    raise ValueError("measurement OD component table columns are invalid.")
                count = 0
                for row in reader:
                    try:
                        index = int(row["measurement_index"])
                        values = [float(row[name]) for name in _COMPONENT_COLUMNS[1:]]
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError("measurement OD component row is invalid.") from error
                    if index != expected_index + count or not np.all(np.isfinite(values)):
                        raise ValueError("measurement OD component row is invalid.")
                    count += 1
            if count != descriptor.get("row_count") or count <= 0 or count > partition_limit:
                raise ValueError("measurement OD component partition row count mismatch.")
            if descriptor.get("start_index") != expected_index:
                raise ValueError("measurement OD component partition index is invalid.")
            expected_index += count
        count = expected_index
    else:
        relative = Path(str(component.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("measurement OD component table path is unsafe.")
        component_path = bundle_root / relative
        if not component_path.is_file() or component_path.stat().st_size != component.get("size_bytes") or _sha256(component_path) != component.get("sha256"):
            raise ValueError("measurement OD component table checksum mismatch.")
        with component_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != _COMPONENT_COLUMNS:
                raise ValueError("measurement OD component table columns are invalid.")
            count = 0
            for row in reader:
                try:
                    index = int(row["measurement_index"])
                    values = [float(row[name]) for name in _COMPONENT_COLUMNS[1:]]
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError("measurement OD component row is invalid.") from error
                if index != count or not np.all(np.isfinite(values)) or index >= int(manifest.get("number_of_measurements", -1)):
                    raise ValueError("measurement OD component row is invalid.")
                count += 1
    if count != component.get("row_count") or count != manifest.get("number_of_measurements"):
        raise ValueError("measurement OD component table row count mismatch.")
    payload_component = payload.get("component_table")
    if not isinstance(payload_component, Mapping) or payload_component.get("row_count") != component.get("row_count"):
        raise ValueError("measurement OD component table metadata differs from its manifest.")
