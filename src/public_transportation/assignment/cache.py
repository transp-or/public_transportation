"""Portable persistent cache for demand-independent assignment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.preprocessing.structural_zeros.scenario_fingerprint import (
    scenario_fingerprint_payload_json,
)
from public_transportation.version import __version__

from .config import AssignmentConfig
from .costs import CostParts
from .jax_graph_types import JaxGraph
from .build_od_groups import ODGroups

ASSIGNMENT_CACHE_SCHEMA_VERSION = 1
AssignmentCachePolicy = Literal["off", "auto", "refresh", "readonly"]

_GRAPH_ARRAYS = (
    "tail", "head", "topo_order", "topo_order_rev", "node_time",
    "node_stop_index", "node_time_s", "node_kind", "node_trip_index",
    "out_start", "out_links_csr", "out_links", "out_mask", "link_type",
    "travel_time", "capacity", "link_trip_index", "node_time_bin_index",
    "node_bin_start_min", "node_bin_end_min",
)
_OD_ARRAYS = (
    "od_origin_node", "od_dest_node", "group_start", "group_dest_node",
    "group_od_index", "group_od_index_padded", "group_od_mask", "group_link_mask",
)
_COST_ARRAYS = (
    "base_cost", "is_access", "is_ride", "is_transfer", "is_egress", "is_dwell",
)


@dataclass(frozen=True, slots=True)
class AssignmentCacheMetrics:
    status: str
    cache_hit: bool
    cache_load_seconds: float
    validation_seconds: float
    host_reconstruction_seconds: float
    device_transfer_seconds: float
    preparation_seconds_when_built: float
    stored_bytes: int
    schema_version: int
    cache_key: str | None
    fingerprint_seconds: float
    preparation_stages: dict[str, float]
    logical_bytes: int = 0
    num_nodes: int = 0
    num_links: int = 0
    num_od: int = 0
    num_groups: int = 0
    array_summary: dict[str, dict[str, Any]] | None = None
    npz_decompression_seconds: float = 0.0


def _assignment_scenario_payload(scenario: Any) -> dict[str, Any]:
    payload = json.loads(scenario_fingerprint_payload_json(scenario))
    # Assignment topology depends on OD keys and time bins, never demand magnitudes.
    for record in payload["demand"]:
        record.pop("demand", None)
        record.pop("value", None)
    return payload


def assignment_cache_provenance(
    *, scenario: Any, config: AssignmentConfig, numeric_dtype: str = "float32"
) -> tuple[str, str]:
    """Return canonical provenance JSON and its SHA-256 cache key."""
    payload = {
        "schema_version": ASSIGNMENT_CACHE_SCHEMA_VERSION,
        "package_version": __version__,
        "scenario": _assignment_scenario_payload(scenario),
        "assignment_config": asdict(config),
        "numeric_dtype": numeric_dtype,
        "sentinel_conventions": {
            "centroid_in_time": "-infinity sentinel",
            "centroid_out_time": "+infinity sentinel",
            "missing_index": -1,
            "padded_link_index": 0,
        },
        "indexing_rules": {
            "stops": "sorted stop_id",
            "trips": "timetable order",
            "links": "canonical builder order",
            "od": "scenario demand-record order",
            "destination_groups": "stable destination-node sort",
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def assignment_cache_path(
    *, cache_directory: str | os.PathLike[str], cache_key: str
) -> Path:
    return Path(cache_directory).expanduser().resolve() / f"assignment-{cache_key}.npz"


def _arrays_from_artifacts(artifacts: Any) -> dict[str, np.ndarray]:
    arrays = {
        **{f"graph__{name}": np.asarray(getattr(artifacts.graph, name)) for name in _GRAPH_ARRAYS},
        **{f"od__{name}": np.asarray(getattr(artifacts.od_groups, name)) for name in _OD_ARRAYS},
        **{f"cost__{name}": np.asarray(getattr(artifacts.cost_parts, name)) for name in _COST_ARRAYS},
    }
    return arrays


def assignment_artifact_summary(artifacts: Any) -> tuple[int, dict[str, dict[str, Any]]]:
    arrays = _arrays_from_artifacts(artifacts)
    return (
        sum(int(value.nbytes) for value in arrays.values()),
        {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": int(value.nbytes),
            }
            for name, value in sorted(arrays.items())
        },
    )


def _manifest(*, artifacts: Any, provenance_json: str, cache_key: str) -> dict[str, Any]:
    arrays = _arrays_from_artifacts(artifacts)
    return {
        "schema_version": ASSIGNMENT_CACHE_SCHEMA_VERSION,
        "package_version": __version__,
        "cache_key": cache_key,
        "provenance_payload_json": provenance_json,
        "num_nodes": int(artifacts.graph.num_nodes),
        "num_links": int(artifacts.graph.num_links),
        "num_od": int(artifacts.od_groups.num_od),
        "graph_labels": {
            "node_stop_id": list(artifacts.graph.node_stop_id),
            "node_stop_name": list(artifacts.graph.node_stop_name),
            "trip_id": list(artifacts.graph.trip_id),
            "trip_line_ref": list(artifacts.graph.trip_line_ref),
        },
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(arrays.items())
        },
    }


def _write(*, path: Path, artifacts: Any, provenance_json: str, cache_key: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _arrays_from_artifacts(artifacts)
    metadata = json.dumps(
        _manifest(artifacts=artifacts, provenance_json=provenance_json, cache_key=cache_key),
        sort_keys=True,
        separators=(",", ":"),
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, metadata=np.asarray(metadata), **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path.stat().st_size


def _validated_host_arrays(
    *, path: Path, provenance_json: str, cache_key: str
) -> tuple[dict[str, Any], dict[str, np.ndarray], float, float, float]:
    started = perf_counter()
    with np.load(path, allow_pickle=False) as archive:
        load_seconds = perf_counter() - started
        validation_started = perf_counter()
        metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
        if metadata.get("schema_version") != ASSIGNMENT_CACHE_SCHEMA_VERSION:
            raise ValueError("Assignment cache schema version mismatch.")
        if metadata.get("package_version") != __version__:
            raise ValueError("Assignment cache package version mismatch.")
        if metadata.get("cache_key") != cache_key:
            raise ValueError("Assignment cache key mismatch.")
        if metadata.get("provenance_payload_json") != provenance_json:
            raise ValueError("Assignment cache provenance mismatch.")
        manifest = metadata.get("arrays")
        if not isinstance(manifest, dict):
            raise ValueError("Assignment cache array manifest is missing.")
        arrays: dict[str, np.ndarray] = {}
        decompression_seconds = 0.0
        for name, expected in manifest.items():
            if name not in archive.files:
                raise ValueError(f"Assignment cache array is missing: {name}.")
            decompression_started = perf_counter()
            value = np.array(archive[name], copy=True)
            decompression_seconds += perf_counter() - decompression_started
            if list(value.shape) != expected.get("shape"):
                raise ValueError(f"Assignment cache shape mismatch for {name}.")
            if str(value.dtype) != expected.get("dtype"):
                raise ValueError(f"Assignment cache dtype mismatch for {name}.")
            arrays[name] = value
        expected_names = {"metadata", *manifest}
        if set(archive.files) != expected_names:
            raise ValueError("Assignment cache contains unrecognized arrays.")
        _validate_structure(metadata=metadata, arrays=arrays)
        validation_seconds = perf_counter() - validation_started
    return metadata, arrays, load_seconds, validation_seconds, decompression_seconds


def _validate_structure(
    *, metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    num_nodes = int(metadata["num_nodes"])
    num_links = int(metadata["num_links"])
    num_od = int(metadata["num_od"])
    if min(num_nodes, num_links, num_od) <= 0:
        raise ValueError("Assignment cache counts must be positive.")
    link_names = (
        "tail", "head", "out_links_csr", "link_type", "travel_time",
        "capacity", "link_trip_index",
    )
    node_names = (
        "topo_order", "topo_order_rev", "node_time", "node_stop_index",
        "node_time_s", "node_kind", "node_trip_index", "node_time_bin_index",
        "node_bin_start_min", "node_bin_end_min",
    )
    for name in link_names:
        if arrays[f"graph__{name}"].shape != (num_links,):
            raise ValueError(f"Assignment cache link shape mismatch for {name}.")
    for name in node_names:
        if arrays[f"graph__{name}"].shape != (num_nodes,):
            raise ValueError(f"Assignment cache node shape mismatch for {name}.")
    if arrays["graph__out_start"].shape != (num_nodes + 1,):
        raise ValueError("Assignment cache CSR pointer shape mismatch.")
    padded = arrays["graph__out_links"]
    if padded.ndim != 2 or padded.shape[0] != num_nodes:
        raise ValueError("Assignment cache padded adjacency shape mismatch.")
    if arrays["graph__out_mask"].shape != padded.shape:
        raise ValueError("Assignment cache padded adjacency mask mismatch.")
    for name in ("od_origin_node", "od_dest_node", "group_od_index"):
        if arrays[f"od__{name}"].shape != (num_od,):
            raise ValueError(f"Assignment cache OD shape mismatch for {name}.")
    num_groups = arrays["od__group_dest_node"].shape[0]
    if arrays["od__group_start"].shape != (num_groups + 1,):
        raise ValueError("Assignment cache group pointer shape mismatch.")
    if arrays["od__group_link_mask"].shape != (num_groups, num_links):
        raise ValueError("Assignment cache destination mask shape mismatch.")
    if arrays["od__group_od_index_padded"].shape != arrays["od__group_od_mask"].shape:
        raise ValueError("Assignment cache padded OD mask shape mismatch.")
    if arrays["od__group_od_index_padded"].shape[0] != num_groups:
        raise ValueError("Assignment cache padded OD group count mismatch.")
    for name in _COST_ARRAYS:
        if arrays[f"cost__{name}"].shape != (num_links,):
            raise ValueError(f"Assignment cache cost shape mismatch for {name}.")
    integer_names = (
        "graph__tail", "graph__head", "graph__topo_order", "graph__topo_order_rev",
        "graph__node_stop_index", "graph__node_time_s", "graph__node_kind",
        "graph__node_trip_index", "graph__out_start", "graph__out_links_csr",
        "graph__out_links", "graph__link_type", "graph__link_trip_index",
        "od__od_origin_node", "od__od_dest_node", "od__group_start",
        "od__group_dest_node", "od__group_od_index", "od__group_od_index_padded",
    )
    if any(not np.issubdtype(arrays[name].dtype, np.integer) for name in integer_names):
        raise ValueError("Assignment cache indexing arrays must have integer dtype.")
    boolean_names = (
        "graph__out_mask", "od__group_od_mask", "od__group_link_mask",
        "cost__is_access", "cost__is_ride", "cost__is_transfer",
        "cost__is_egress", "cost__is_dwell",
    )
    if any(arrays[name].dtype != np.dtype(bool) for name in boolean_names):
        raise ValueError("Assignment cache mask arrays must have boolean dtype.")


def _reconstruct(
    *, metadata: dict[str, Any], arrays: dict[str, np.ndarray], config: AssignmentConfig
) -> tuple[Any, float, float]:
    from .assign import AssignmentArtifacts

    host_started = perf_counter()
    labels = metadata["graph_labels"]
    graph_kwargs = {name: arrays[f"graph__{name}"] for name in _GRAPH_ARRAYS}
    od_kwargs = {name: arrays[f"od__{name}"] for name in _OD_ARRAYS}
    cost_kwargs = {name: arrays[f"cost__{name}"] for name in _COST_ARRAYS}
    host_seconds = perf_counter() - host_started
    device_started = perf_counter()
    graph = JaxGraph(
        num_nodes=int(metadata["num_nodes"]),
        num_links=int(metadata["num_links"]),
        **{name: jnp.asarray(value) for name, value in graph_kwargs.items()},
        node_stop_id=tuple(labels["node_stop_id"]),
        node_stop_name=tuple(labels["node_stop_name"]),
        trip_id=tuple(labels["trip_id"]),
        trip_line_ref=tuple(labels["trip_line_ref"]),
    )
    od_groups = ODGroups(
        num_od=int(metadata["num_od"]),
        **{name: jnp.asarray(value) for name, value in od_kwargs.items()},
    )
    cost_parts = CostParts(**{name: jnp.asarray(value) for name, value in cost_kwargs.items()})
    jax.block_until_ready((graph, od_groups, cost_parts))
    device_seconds = perf_counter() - device_started
    return AssignmentArtifacts(graph, od_groups, cost_parts, config), host_seconds, device_seconds


def load_or_prepare_assignment(
    *,
    scenario: Any,
    config: AssignmentConfig,
    cache_directory: str | os.PathLike[str],
    policy: AssignmentCachePolicy = "auto",
    timetable_index: Any | None = None,
) -> Any:
    """Load, build, refresh, or read an explicit assignment cache artifact."""
    if policy not in ("off", "auto", "refresh", "readonly"):
        raise ValueError("Assignment cache policy must be off, auto, refresh, or readonly.")
    from .assign import _prepare_assignment_uncached

    if policy == "off":
        return _prepare_assignment_uncached(
            scenario=scenario, config=config, timetable_index=timetable_index
        )
    fingerprint_started = perf_counter()
    provenance_json, cache_key = assignment_cache_provenance(scenario=scenario, config=config)
    fingerprint_seconds = perf_counter() - fingerprint_started
    path = assignment_cache_path(cache_directory=cache_directory, cache_key=cache_key)
    invalid = False
    if policy != "refresh" and path.exists():
        try:
            (
                metadata, arrays, load_seconds, validation_seconds,
                decompression_seconds,
            ) = _validated_host_arrays(
                path=path, provenance_json=provenance_json, cache_key=cache_key
            )
            artifacts, host_seconds, device_seconds = _reconstruct(
                metadata=metadata, arrays=arrays, config=config
            )
            logical_bytes = sum(int(value.nbytes) for value in arrays.values())
            metrics = AssignmentCacheMetrics(
                status="hit", cache_hit=True, cache_load_seconds=load_seconds,
                validation_seconds=validation_seconds,
                host_reconstruction_seconds=host_seconds,
                device_transfer_seconds=device_seconds,
                preparation_seconds_when_built=0.0, stored_bytes=path.stat().st_size,
                schema_version=ASSIGNMENT_CACHE_SCHEMA_VERSION, cache_key=cache_key,
                fingerprint_seconds=fingerprint_seconds, preparation_stages={},
                logical_bytes=logical_bytes, num_nodes=int(metadata["num_nodes"]),
                num_links=int(metadata["num_links"]), num_od=int(metadata["num_od"]),
                num_groups=int(arrays["od__group_dest_node"].shape[0]),
                array_summary={
                    name: {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": int(value.nbytes)}
                    for name, value in sorted(arrays.items())
                },
                npz_decompression_seconds=decompression_seconds,
            )
            return artifacts.__class__(
                artifacts.graph, artifacts.od_groups, artifacts.cost_parts, artifacts.config,
                metrics, provenance_json, cache_key,
            )
        except (
            OSError, EOFError, zipfile.BadZipFile, ValueError, KeyError, TypeError,
            json.JSONDecodeError,
        ):
            invalid = True
            if policy == "readonly":
                raise ValueError("Invalid read-only assignment cache entry.")
    elif policy == "readonly":
        raise FileNotFoundError(f"Read-only assignment cache entry not found: {path}")
    built = _prepare_assignment_uncached(
        scenario=scenario, config=config, timetable_index=timetable_index
    )
    stored_bytes = _write(
        path=path, artifacts=built, provenance_json=provenance_json, cache_key=cache_key
    )
    original = built.cache_metrics
    logical_bytes, array_summary = assignment_artifact_summary(built)
    metrics = AssignmentCacheMetrics(
        status="invalid_rebuilt" if invalid else ("refresh" if policy == "refresh" else "miss"),
        cache_hit=False, cache_load_seconds=0.0, validation_seconds=0.0,
        host_reconstruction_seconds=0.0, device_transfer_seconds=0.0,
        preparation_seconds_when_built=original.preparation_seconds_when_built,
        stored_bytes=stored_bytes, schema_version=ASSIGNMENT_CACHE_SCHEMA_VERSION,
        cache_key=cache_key, fingerprint_seconds=fingerprint_seconds,
        preparation_stages=original.preparation_stages,
        logical_bytes=logical_bytes, num_nodes=int(built.graph.num_nodes),
        num_links=int(built.graph.num_links), num_od=int(built.od_groups.num_od),
        num_groups=int(built.od_groups.group_dest_node.shape[0]),
        array_summary=array_summary,
    )
    return built.__class__(
        built.graph, built.od_groups, built.cost_parts, built.config,
        metrics, provenance_json, cache_key,
    )
