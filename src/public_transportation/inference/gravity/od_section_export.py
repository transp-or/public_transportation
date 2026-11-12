"""Restartable export of expected OD use of scheduled ride sections.

The fixed-routing cache stores one outgoing-link probability vector per
destination group.  This module replays those probabilities for one OD cell at
a time and writes sparse ``service_marginal`` rows for the scheduled ride
links.  It deliberately does not reconstruct passenger paths or run routing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from public_transportation import __version__
from public_transportation.assignment.graph_sentinels import LINK_TYPE_RIDE

from ..compact_od_assignment_layout import CompactODAssignmentLayout
from ..assignment_adapter import AssignmentInputs
from ..sharded_fixed_routing import (
    ShardedFixedRoutingInputs,
    load_fixed_routing_shard,
    sharded_fixed_routing_identity,
    validate_sharded_fixed_routing_compatibility,
)
from ..block_coordinate._canonical import fingerprint
from .viewer_bundle import _ROUTE_FLOW_COLUMNS, _route_flow_row


ODSectionProgress = Callable[[Mapping[str, object]], None]
EXPORT_SCHEMA_VERSION = 1
PLAN_CHECKPOINT_SCHEMA_VERSION = 1
PLAN_ALGORITHM_VERSION = "od-section-plan-pilot-v1"
PLAN_CHECKPOINT_FILE = "plan_checkpoint.json"


class ODSectionPlanInterrupted(RuntimeError):
    """Raised after a pilot checkpoint has been durably saved for interruption."""


class ODSectionPlanCheckpointError(ValueError):
    """Raised when a pilot checkpoint is corrupt, stale, or unsafe to resume."""


class _PlanCheckpointLock:
    """Process lock whose kernel ownership is released if a job is killed."""

    def __init__(self, directory: Path) -> None:
        self.path = directory / ".plan.lock"
        self._file_descriptor: int | None = None

    def __enter__(self) -> "_PlanCheckpointLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            os.close(descriptor)
            raise RuntimeError(
                f"another OD-section plan process already owns checkpoint {self.path}"
            ) from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        self._file_descriptor = descriptor
        return self

    def __exit__(self, *_: object) -> None:
        if self._file_descriptor is None:
            return
        try:
            import fcntl

            fcntl.flock(self._file_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._file_descriptor)
            self._file_descriptor = None


class _PlanSignalGuard:
    """Turn SIGTERM into a checkpointed interruption at a safe group boundary."""

    def __init__(self) -> None:
        self.requested = False
        self._previous: Any = None
        self._installed = False

    def __enter__(self) -> "_PlanSignalGuard":
        try:
            self._previous = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._handle)
            self._installed = True
        except (ValueError, OSError):
            # Worker threads cannot install signal handlers. The checkpoint is
            # still durable after every group and remains restartable.
            self._installed = False
        return self

    def _handle(self, _signum: int, _frame: Any) -> None:
        self.requested = True

    def check(self) -> None:
        if self.requested:
            raise ODSectionPlanInterrupted("SIGTERM received; pilot checkpoint is resumable.")

    def __exit__(self, *_: object) -> None:
        if self._installed:
            signal.signal(signal.SIGTERM, self._previous)


@dataclass(frozen=True, slots=True)
class ODSectionCell:
    """Human-readable identity of one canonical OD/departure cell."""

    origin_stop_id: str
    destination_stop_id: str
    departure_time_bin: str

    def __post_init__(self) -> None:
        for name in ("origin_stop_id", "destination_stop_id", "departure_time_bin"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be nonempty.")
        object.__setattr__(self, "origin_stop_id", str(self.origin_stop_id))
        object.__setattr__(self, "destination_stop_id", str(self.destination_stop_id))
        object.__setattr__(self, "departure_time_bin", str(self.departure_time_bin))


@dataclass(frozen=True, slots=True)
class RideLinkSectionMetadata:
    """Metadata identifying one adjacent-stop scheduled ride link."""

    line_id: str
    trip_id: str
    boarding_stop_id: str
    alighting_stop_id: str
    pattern_id: str = ""
    direction: str = ""

    def __post_init__(self) -> None:
        for name in ("line_id", "trip_id", "boarding_stop_id", "alighting_stop_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be nonempty for a ride link.")
        for name in ("line_id", "trip_id", "boarding_stop_id", "alighting_stop_id", "pattern_id", "direction"):
            object.__setattr__(self, name, str(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class ODSectionExportPlan:
    """Dry-run summary; a full-job walltime is never recommended without a pilot."""

    available: bool
    num_od: int
    num_destination_groups: int
    num_shards: int
    ride_links: int
    potential_rows: int
    estimated_bytes: int
    estimated_memory_bytes: int
    pilot_groups: int
    effective_pilot_groups: int
    pilot_seconds: float | None
    pilot_rows: int | None
    pilot_peak_memory_bytes: int | None
    pilot_required: bool
    estimated_full_seconds: float | None
    recommendation: str
    routing_preparation_fingerprint: str
    fit_assignment_fingerprint: str
    fitted_demand_fingerprint: str
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _demand_fingerprint(demand: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(demand, dtype=np.float64)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _bytes_fingerprint(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _json_normalize(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_normalize(value.tolist())
    if isinstance(value, np.generic):
        return _json_normalize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("checkpoint identity cannot contain non-finite values.")
    return value


def _checkpoint_identity(
    *,
    assignment_inputs: AssignmentInputs,
    routing: ShardedFixedRoutingInputs,
    demand: np.ndarray,
    cells: tuple[ODSectionCell, ...],
    compact_layout: CompactODAssignmentLayout | None,
    fit_assignment_fingerprint: str,
    requested_pilot_groups: int,
    checkpoint_batch_size: int,
    bytes_per_row: int,
    external_identity: Mapping[str, object] | None,
) -> tuple[dict[str, object], str]:
    routing_identity = sharded_fixed_routing_identity(
        inputs=assignment_inputs,
        theta=routing.theta,
        shard_partition=routing.shard_partition,
    )
    group_order_fingerprint = _bytes_fingerprint(
        np.asarray(assignment_inputs.group_dest_node),
        np.asarray(assignment_inputs.group_od_index_padded),
        np.asarray(assignment_inputs.group_od_mask),
    )
    cell_identity = tuple(
        (cell.origin_stop_id, cell.destination_stop_id, cell.departure_time_bin)
        for cell in cells
    )
    payload: dict[str, object] = {
        "checkpoint_schema_version": PLAN_CHECKPOINT_SCHEMA_VERSION,
        "algorithm_version": PLAN_ALGORITHM_VERSION,
        "package_revision": __version__,
        "fit_assignment_fingerprint": fit_assignment_fingerprint,
        "fitted_demand_fingerprint": _demand_fingerprint(demand),
        "od_cell_fingerprint": fingerprint(cell_identity),
        "compact_layout_fingerprint": (
            None if compact_layout is None else compact_layout.fingerprint
        ),
        "graph_fingerprint": routing_identity["graph_fingerprint"],
        "routing_preparation_fingerprint": routing.provenance.preparation_fingerprint,
        "routing_cache_directory": str(routing.provenance.cache_directory),
        "routing_schema_version": routing.provenance.schema_version,
        "routing_shard_partition": routing_identity["shard_partition"],
        "group_order_fingerprint": group_order_fingerprint,
        "pilot_settings": {
            "requested_pilot_groups": int(requested_pilot_groups),
            "checkpoint_batch_size": int(checkpoint_batch_size),
            "bytes_per_row": int(bytes_per_row),
        },
        "external_identity": {} if external_identity is None else dict(external_identity),
    }
    normalized = _json_normalize(payload)
    assert isinstance(normalized, dict)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return normalized, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def _as_od_cell(value: ODSectionCell | Mapping[str, object]) -> ODSectionCell:
    if isinstance(value, ODSectionCell):
        return value
    aliases = {
        "origin": "origin_stop_id",
        "destination": "destination_stop_id",
        "departure_interval_id": "departure_time_bin",
        "departure": "departure_time_bin",
    }
    data = dict(value)
    for source, target in aliases.items():
        if target not in data and source in data:
            data[target] = data[source]
    return ODSectionCell(
        origin_stop_id=str(data["origin_stop_id"]),
        destination_stop_id=str(data["destination_stop_id"]),
        departure_time_bin=str(data["departure_time_bin"]),
    )


def _as_link_metadata(
    value: RideLinkSectionMetadata | Mapping[str, object],
) -> RideLinkSectionMetadata:
    if isinstance(value, RideLinkSectionMetadata):
        return value
    data = dict(value)
    aliases = {
        "line": "line_id",
        "trip": "trip_id",
        "boarding_stop": "boarding_stop_id",
        "alighting_stop": "alighting_stop_id",
        "pattern": "pattern_id",
        "direction_id": "direction",
    }
    for source, target in aliases.items():
        if target not in data and source in data:
            data[target] = data[source]
    return RideLinkSectionMetadata(
        line_id=str(data["line_id"]),
        trip_id=str(data["trip_id"]),
        boarding_stop_id=str(data["boarding_stop_id"]),
        alighting_stop_id=str(data["alighting_stop_id"]),
        pattern_id=str(data.get("pattern_id", "") or ""),
        direction=str(data.get("direction", "") or ""),
    )


def _node_stop_ids(graph: Any, supplied: Mapping[int, str] | Sequence[str] | None) -> tuple[str, ...]:
    if supplied is None:
        result = tuple(getattr(graph, "node_stop_id", ()))
    elif isinstance(supplied, Mapping):
        result_list = [""] * int(graph.num_nodes)
        for node, stop_id in supplied.items():
            result_list[int(node)] = str(stop_id)
        result = tuple(result_list)
    else:
        result = tuple(str(value) for value in supplied)
    if len(result) == 0 or any(not value for value in result):
        raise ValueError("node_stop_ids must provide a nonempty stop ID for every graph stop index.")
    return result


def _graph_link_metadata(graph: Any, link: int, node_stop_ids: tuple[str, ...]) -> RideLinkSectionMetadata:
    trip_index = int(np.asarray(graph.link_trip_index)[link])
    trip_ids = tuple(getattr(graph, "trip_id", ()))
    line_refs = tuple(getattr(graph, "trip_line_ref", ()))
    if trip_index < 0 or trip_index >= len(trip_ids) or trip_index >= len(line_refs):
        raise ValueError(f"ride link {link} has no valid trip and line metadata.")
    stop_index = np.asarray(graph.node_stop_index)
    tail = int(np.asarray(graph.tail)[link])
    head = int(np.asarray(graph.head)[link])
    boarding_index = int(stop_index[tail])
    alighting_index = int(stop_index[head])
    if not (0 <= boarding_index < len(node_stop_ids) and 0 <= alighting_index < len(node_stop_ids)):
        raise ValueError(f"ride link {link} has invalid boarding/alighting stop indices.")
    return RideLinkSectionMetadata(
        line_id=str(line_refs[trip_index]),
        trip_id=str(trip_ids[trip_index]),
        boarding_stop_id=str(node_stop_ids[boarding_index]),
        alighting_stop_id=str(node_stop_ids[alighting_index]),
    )


def _normalise_link_metadata(
    graph: Any,
    ride_link_metadata: Mapping[int, RideLinkSectionMetadata | Mapping[str, object]] | None,
    ride_links: np.ndarray,
    node_stop_ids: tuple[str, ...],
) -> dict[int, RideLinkSectionMetadata]:
    supplied = {} if ride_link_metadata is None else dict(ride_link_metadata)
    result: dict[int, RideLinkSectionMetadata] = {}
    for link in ride_links.tolist():
        raw = supplied.get(int(link))
        result[int(link)] = (
            _graph_link_metadata(graph, int(link), node_stop_ids) if raw is None else _as_link_metadata(raw)
        )
    unknown = set(supplied) - set(result)
    if unknown:
        raise ValueError(f"ride_link_metadata contains non-ride or unknown links: {sorted(unknown)}")
    return result


def _normalise_demand(
    fitted_demand: Sequence[float] | np.ndarray,
    *,
    od_cells: Sequence[ODSectionCell | Mapping[str, object]],
    compact_layout: CompactODAssignmentLayout | None,
    expected_active: int,
) -> tuple[np.ndarray, tuple[ODSectionCell, ...]]:
    demand = np.asarray(fitted_demand, dtype=np.float64)
    if demand.ndim != 1 or not np.all(np.isfinite(demand)) or np.any(demand < 0.0):
        raise ValueError("fitted_demand must be a finite, non-negative one-dimensional vector.")
    cells = tuple(_as_od_cell(item) for item in od_cells)
    if compact_layout is None:
        if len(cells) != demand.size or demand.size != expected_active:
            raise ValueError("OD layout does not match the compact fitted-demand vector.")
        return demand, cells
    if len(cells) != compact_layout.num_od_total:
        raise ValueError("canonical OD cell count does not match compact layout.")
    if demand.size == compact_layout.num_od_total:
        demand = demand[np.asarray(compact_layout.active_full_indices, dtype=np.int64)]
    elif demand.size != compact_layout.num_active:
        raise ValueError("fitted_demand length does not match canonical or compact OD layout.")
    if demand.size != expected_active:
        raise ValueError("compact OD layout does not match assignment inputs.")
    return demand, tuple(cells[index] for index in compact_layout.active_full_indices)


def _validate_assignment_layout(inputs: AssignmentInputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origins = np.asarray(inputs.od_origin_node, dtype=np.int64)
    group_indices = np.asarray(inputs.group_od_index_padded, dtype=np.int64)
    group_mask = np.asarray(inputs.group_od_mask, dtype=bool)
    if origins.ndim != 1:
        raise ValueError("assignment OD origins must be one-dimensional.")
    if group_indices.ndim != 2 or group_mask.shape != group_indices.shape:
        raise ValueError("assignment group OD arrays have inconsistent shapes.")
    valid = group_indices[group_mask]
    expected = np.arange(origins.size, dtype=np.int64)
    if valid.size != expected.size or not np.array_equal(np.sort(valid), expected):
        raise ValueError("assignment OD groups do not cover the canonical compact OD ordering exactly once.")
    if np.any(origins < 0) or np.any(origins >= int(inputs.graph.num_nodes)):
        raise ValueError("assignment OD origins contain invalid graph nodes.")
    return origins, group_indices, group_mask


def _validate_od_cell_binding(
    *,
    inputs: AssignmentInputs,
    origins: np.ndarray,
    group_indices: np.ndarray,
    group_mask: np.ndarray,
    cells: tuple[ODSectionCell, ...],
    node_stop_ids: tuple[str, ...],
) -> None:
    """Check that human-readable OD cells bind to the assignment graph order."""
    keys = [(cell.origin_stop_id, cell.destination_stop_id, cell.departure_time_bin) for cell in cells]
    if len(set(keys)) != len(keys):
        raise ValueError("OD cells contain duplicate canonical origin/destination/departure keys.")
    stop_index = np.asarray(inputs.graph.node_stop_index, dtype=np.int64)

    def node_stop(node: int) -> str:
        index = int(stop_index[node])
        if index < 0 or index >= len(node_stop_ids):
            raise ValueError("assignment graph contains an invalid node stop index.")
        return str(node_stop_ids[index])

    destination_nodes = np.asarray(inputs.group_dest_node, dtype=np.int64)
    for group in range(group_indices.shape[0]):
        destination_stop = node_stop(int(destination_nodes[group]))
        for od_index in group_indices[group][group_mask[group]].tolist():
            cell = cells[int(od_index)]
            if cell.origin_stop_id != node_stop(int(origins[int(od_index)])):
                raise ValueError("OD origin stop does not match canonical assignment ordering.")
            if cell.destination_stop_id != destination_stop:
                raise ValueError("OD destination stop does not match canonical assignment ordering.")


def _validate_routing_cache(
    *, inputs: AssignmentInputs, routing: ShardedFixedRoutingInputs
) -> str:
    validate_sharded_fixed_routing_compatibility(inputs=inputs, routing=routing)
    identity = sharded_fixed_routing_identity(
        inputs=inputs, theta=routing.theta, shard_partition=routing.shard_partition
    )
    expected = fingerprint(identity)
    if expected != routing.provenance.preparation_fingerprint:
        raise ValueError("routing cache preparation fingerprint does not match assignment inputs.")
    # Load every shard once before consuming fitted demand.  The loader checks
    # descriptor identity, payload shape, dtype, and scientific content hash.
    for _ in _iter_validated_shards(routing):
        pass
    return expected


def _iter_validated_shards(routing: ShardedFixedRoutingInputs):
    from ..sharded_fixed_routing import iter_fixed_routing_shards

    yield from iter_fixed_routing_shards(routing)


def _ride_links(graph: Any) -> np.ndarray:
    link_type = np.asarray(graph.link_type)
    return np.flatnonzero(link_type == LINK_TYPE_RIDE).astype(np.int64)


def _group_ride_probabilities(
    *, graph: Any, probability: np.ndarray, enabled: np.ndarray, origin_node: int
) -> dict[int, float]:
    """Propagate unit mass through one group without retaining OD×link state."""
    node_mass = np.zeros(int(graph.num_nodes), dtype=np.float64)
    node_mass[int(origin_node)] = 1.0
    head = np.asarray(graph.head, dtype=np.int64)
    out_links = np.asarray(graph.out_links, dtype=np.int64)
    out_mask = np.asarray(graph.out_mask, dtype=bool)
    ride = np.asarray(graph.link_type) == LINK_TYPE_RIDE
    result: dict[int, float] = {}
    for node in np.asarray(graph.topo_order, dtype=np.int64):
        mass = float(node_mass[int(node)])
        if mass == 0.0:
            continue
        for link in out_links[int(node)][out_mask[int(node)]]:
            link_index = int(link)
            if not bool(enabled[link_index]):
                continue
            flow = mass * float(probability[link_index])
            if flow == 0.0:
                continue
            if bool(ride[link_index]):
                result[link_index] = result.get(link_index, 0.0) + flow
            node_mass[int(head[link_index])] += flow
    return result


def _plan_from_dict(payload: Mapping[str, object]) -> ODSectionExportPlan:
    fields = {
        "available",
        "num_od",
        "num_destination_groups",
        "num_shards",
        "ride_links",
        "potential_rows",
        "estimated_bytes",
        "estimated_memory_bytes",
        "pilot_groups",
        "effective_pilot_groups",
        "pilot_seconds",
        "pilot_rows",
        "pilot_peak_memory_bytes",
        "pilot_required",
        "estimated_full_seconds",
        "recommendation",
        "routing_preparation_fingerprint",
        "fit_assignment_fingerprint",
        "fitted_demand_fingerprint",
        "reason",
    }
    if not fields.issubset(payload):
        raise ODSectionPlanCheckpointError("completed pilot checkpoint has an incomplete plan payload.")
    return ODSectionExportPlan(**{key: payload[key] for key in fields})  # type: ignore[arg-type]


def _checkpoint_manifest_path(directory: Path) -> Path:
    return directory / PLAN_CHECKPOINT_FILE


def _read_plan_checkpoint(
    *,
    path: Path,
    identity_payload: Mapping[str, object],
    identity_fingerprint: str,
    resume: bool,
) -> dict[str, object]:
    if not path.exists():
        if resume:
            raise FileNotFoundError(f"--resume requested but checkpoint is missing: {path}")
        return {
            "schema_version": PLAN_CHECKPOINT_SCHEMA_VERSION,
            "algorithm_version": PLAN_ALGORITHM_VERSION,
            "identity": dict(identity_payload),
            "identity_fingerprint": identity_fingerprint,
            "status": "running",
            "total_groups": 0,
            "requested_pilot_groups": 0,
            "effective_pilot_groups": 0,
            "completed_groups": [],
            "group_records": {},
            "pilot_rows": 0,
            "pilot_elapsed_seconds": 0.0,
        }
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ODSectionPlanCheckpointError(f"corrupt pilot checkpoint: {path}") from error
    if not isinstance(checkpoint, dict):
        raise ODSectionPlanCheckpointError("pilot checkpoint must contain a JSON object.")
    if checkpoint.get("schema_version") != PLAN_CHECKPOINT_SCHEMA_VERSION:
        raise ODSectionPlanCheckpointError("unsupported pilot checkpoint schema version.")
    if checkpoint.get("algorithm_version") != PLAN_ALGORITHM_VERSION:
        raise ODSectionPlanCheckpointError("pilot checkpoint algorithm version differs.")
    if checkpoint.get("identity_fingerprint") != identity_fingerprint or checkpoint.get("identity") != dict(identity_payload):
        raise ODSectionPlanCheckpointError("pilot checkpoint identity differs from current inputs.")
    completed_values = checkpoint.get("completed_groups", [])
    records_for_consistency = checkpoint.get("group_records", {})
    if not isinstance(completed_values, list) or not isinstance(records_for_consistency, dict):
        raise ODSectionPlanCheckpointError("pilot checkpoint completion metadata is corrupt.")
    if {str(value) for value in completed_values} != set(records_for_consistency):
        raise ODSectionPlanCheckpointError("pilot checkpoint completed groups do not match group records.")
    if checkpoint.get("status") == "completed":
        if not isinstance(checkpoint.get("plan"), Mapping):
            raise ODSectionPlanCheckpointError("completed pilot checkpoint has no saved plan.")
        records = checkpoint.get("group_records")
        if not isinstance(records, dict):
            raise ODSectionPlanCheckpointError("completed pilot checkpoint group records are corrupt.")
        expected_completed = int(checkpoint.get("effective_pilot_groups", -1))
        if expected_completed < 0 or set(records) != {str(index) for index in range(expected_completed)}:
            raise ODSectionPlanCheckpointError("completed pilot checkpoint is missing destination groups.")
        for raw_record in records.values():
            if not isinstance(raw_record, Mapping):
                raise ODSectionPlanCheckpointError("completed pilot checkpoint contains an invalid group record.")
            group_path = path.parent / str(raw_record.get("checkpoint_path", ""))
            if not group_path.is_file() or raw_record.get("sha256") != _sha256(group_path):
                raise ODSectionPlanCheckpointError(
                    f"pilot group checkpoint is missing or corrupt: {group_path}"
                )
        return checkpoint
    if not resume:
        raise ODSectionPlanCheckpointError(
            "pilot checkpoint already exists; pass --resume to continue it explicitly."
        )
    if checkpoint.get("status") not in {"running", "interrupted"}:
        raise ODSectionPlanCheckpointError("pilot checkpoint is not resumable.")
    if not isinstance(checkpoint.get("group_records"), dict):
        raise ODSectionPlanCheckpointError("pilot checkpoint group records are corrupt.")
    return checkpoint


def _save_group_checkpoint(
    *,
    directory: Path,
    group: int,
    shard_index: int,
    row_count: int,
    duration_seconds: float,
) -> Path:
    path = directory / "groups" / f"group-{group:08d}.json"
    _atomic_json(
        path,
        {
            "schema_version": PLAN_CHECKPOINT_SCHEMA_VERSION,
            "algorithm_version": PLAN_ALGORITHM_VERSION,
            "group_index": int(group),
            "shard_index": int(shard_index),
            "row_count": int(row_count),
            "duration_seconds": float(duration_seconds),
        },
    )
    return path


def _emit_plan_progress(
    progress: ODSectionProgress | None,
    *,
    status: str,
    completed_groups: int,
    total_groups: int,
    effective_pilot_groups: int,
    current_group: int | None,
    shard_index: int | None,
    elapsed_seconds: float,
    group_duration_seconds: float | None = None,
    resumed: bool = False,
) -> None:
    if progress is None:
        return
    throughput = None if elapsed_seconds <= 0.0 else completed_groups / elapsed_seconds
    remaining = None
    if throughput and completed_groups < effective_pilot_groups:
        remaining = (effective_pilot_groups - completed_groups) / throughput
    progress(
        {
            "status": status,
            "completed_groups": int(completed_groups),
            "total_groups": int(total_groups),
            "effective_pilot_groups": int(effective_pilot_groups),
            "pilot_covers_all_groups": bool(effective_pilot_groups == total_groups),
            "current_group": current_group,
            "shard_index": shard_index,
            "elapsed_seconds": float(elapsed_seconds),
            "group_duration_seconds": group_duration_seconds,
            "throughput_groups_per_second": throughput,
            "estimated_remaining_seconds": remaining,
            "resumed": resumed,
        }
    )


def plan_od_section_marginal_export(
    *,
    assignment_inputs: AssignmentInputs,
    routing: ShardedFixedRoutingInputs,
    fitted_demand: Sequence[float] | np.ndarray,
    od_cells: Sequence[ODSectionCell | Mapping[str, object]],
    compact_layout: CompactODAssignmentLayout | None = None,
    fit_assignment_fingerprint: str | None = None,
    node_stop_ids: Mapping[int, str] | Sequence[str] | None = None,
    ride_link_metadata: Mapping[int, RideLinkSectionMetadata | Mapping[str, object]] | None = None,
    bytes_per_row: int = 260,
    pilot_groups: int = 0,
    checkpoint_directory: str | Path | None = None,
    resume: bool = False,
    checkpoint_batch_size: int = 1,
    checkpoint_identity: Mapping[str, object] | None = None,
    progress: ODSectionProgress | None = None,
) -> ODSectionExportPlan:
    """Validate inputs and estimate work without preparing routing.

    When ``checkpoint_directory`` is supplied, pilot groups are committed one
    at a time to an atomic checkpoint. ``resume=True`` is required to reuse or
    continue an existing checkpoint; its identity is bound to all scientific
    inputs and pilot settings.
    """
    if bytes_per_row <= 0:
        raise ValueError("bytes_per_row must be positive.")
    if pilot_groups < 0:
        raise ValueError("pilot_groups must be non-negative.")
    if checkpoint_batch_size <= 0:
        raise ValueError("checkpoint_batch_size must be positive.")
    preparation_fingerprint = _validate_routing_cache(inputs=assignment_inputs, routing=routing)
    origins, group_indices, group_mask = _validate_assignment_layout(assignment_inputs)
    stop_ids = _node_stop_ids(assignment_inputs.graph, node_stop_ids)
    demand, cells = _normalise_demand(
        fitted_demand,
        od_cells=od_cells,
        compact_layout=compact_layout,
        expected_active=origins.size,
    )
    _validate_od_cell_binding(
        inputs=assignment_inputs,
        origins=origins,
        group_indices=group_indices,
        group_mask=group_mask,
        cells=cells,
        node_stop_ids=stop_ids,
    )
    ride_links = _ride_links(assignment_inputs.graph)
    metadata = _normalise_link_metadata(assignment_inputs.graph, ride_link_metadata, ride_links, stop_ids)
    del metadata
    potential_rows = 0
    for group in range(group_indices.shape[0]):
        count = int(np.count_nonzero(group_mask[group]))
        # This is a safe upper bound: masked links or zero probabilities reduce
        # the emitted rows, while no OD×link matrix is materialized.
        potential_rows += count * int(ride_links.size)
    fit_fingerprint = str(fit_assignment_fingerprint or routing.assignment_fingerprint)
    if not fit_fingerprint:
        raise ValueError("fit_assignment_fingerprint must be provided or present in routing identity.")
    total_groups = int(group_indices.shape[0])
    effective_pilot_groups = min(int(pilot_groups), total_groups)
    identity_payload, identity_fingerprint = _checkpoint_identity(
        assignment_inputs=assignment_inputs,
        routing=routing,
        demand=demand,
        cells=cells,
        compact_layout=compact_layout,
        fit_assignment_fingerprint=fit_fingerprint,
        requested_pilot_groups=int(pilot_groups),
        checkpoint_batch_size=int(checkpoint_batch_size),
        bytes_per_row=int(bytes_per_row),
        external_identity=checkpoint_identity,
    )
    memory_estimate = int(
        assignment_inputs.graph.num_nodes * 8 + assignment_inputs.graph.num_links * 16
    )

    def make_plan(pilot_seconds: float | None, pilot_rows: int | None) -> ODSectionExportPlan:
        representative = bool(
            effective_pilot_groups
            and effective_pilot_groups >= max(1, int(np.ceil(total_groups * 0.1)))
        )
        estimated_full_seconds = (
            None
            if pilot_seconds is None or effective_pilot_groups == 0
            else float(pilot_seconds) * total_groups / effective_pilot_groups
        )
        return ODSectionExportPlan(
            available=True,
            num_od=int(origins.size),
            num_destination_groups=total_groups,
            num_shards=int(routing.num_shards),
            ride_links=int(ride_links.size),
            potential_rows=int(potential_rows),
            estimated_bytes=int(potential_rows * bytes_per_row),
            estimated_memory_bytes=memory_estimate,
            pilot_groups=int(pilot_groups),
            effective_pilot_groups=effective_pilot_groups,
            pilot_seconds=pilot_seconds,
            pilot_rows=pilot_rows,
            pilot_peak_memory_bytes=memory_estimate if pilot_seconds is not None else None,
            pilot_required=not representative,
            estimated_full_seconds=estimated_full_seconds if representative else None,
            recommendation=(
                "Pilot is representative enough to scale cautiously from measured timing; verify server memory before scheduling."
                if representative
                else "Run a representative pilot and record timing/memory before scheduling the full export."
            ),
            routing_preparation_fingerprint=preparation_fingerprint,
            fit_assignment_fingerprint=fit_fingerprint,
            fitted_demand_fingerprint=_demand_fingerprint(demand),
        )

    def run_pilot(
        *,
        directory: Path | None,
        initial_checkpoint: dict[str, object] | None,
    ) -> ODSectionExportPlan:
        state = initial_checkpoint
        manifest_path = None if directory is None else _checkpoint_manifest_path(directory)
        group_records: dict[str, dict[str, object]] = {}
        completed: set[int] = set()
        pilot_rows = 0
        pilot_seconds = 0.0
        resumed = False
        if state is not None:
            resumed = bool(state.get("completed_groups")) or state.get("status") in {"interrupted", "completed"}
            if state.get("status") == "completed":
                plan_payload = state.get("plan")
                assert isinstance(plan_payload, Mapping)
                plan = _plan_from_dict(plan_payload)
                _emit_plan_progress(
                    progress,
                    status="reused",
                    completed_groups=plan.effective_pilot_groups,
                    total_groups=total_groups,
                    effective_pilot_groups=effective_pilot_groups,
                    current_group=None,
                    shard_index=None,
                    elapsed_seconds=float(plan.pilot_seconds or 0.0),
                    resumed=True,
                )
                return plan
            raw_records = state.get("group_records", {})
            assert isinstance(raw_records, dict)
            for key, raw_record in raw_records.items():
                if not isinstance(raw_record, Mapping):
                    raise ODSectionPlanCheckpointError("pilot checkpoint contains an invalid group record.")
                group = int(key)
                if group < 0 or group >= effective_pilot_groups:
                    raise ODSectionPlanCheckpointError("pilot checkpoint contains an out-of-range group.")
                group_path = directory / str(raw_record.get("checkpoint_path", ""))
                if not group_path.is_file() or raw_record.get("sha256") != _sha256(group_path):
                    raise ODSectionPlanCheckpointError(
                        f"pilot group checkpoint is missing or corrupt: {group_path}"
                    )
                try:
                    group_payload = json.loads(group_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    raise ODSectionPlanCheckpointError(
                        f"pilot group checkpoint is corrupt: {group_path}"
                    ) from error
                if not isinstance(group_payload, Mapping) or int(group_payload.get("group_index", -1)) != group:
                    raise ODSectionPlanCheckpointError("pilot group checkpoint identity is corrupt.")
                group_records[str(group)] = dict(raw_record)
                completed.add(group)
                pilot_rows += int(raw_record.get("row_count", 0))
                pilot_seconds += float(raw_record.get("duration_seconds", 0.0))
            if int(state.get("pilot_rows", pilot_rows)) != pilot_rows or not np.isclose(
                float(state.get("pilot_elapsed_seconds", pilot_seconds)), pilot_seconds
            ):
                raise ODSectionPlanCheckpointError("pilot checkpoint cumulative totals are corrupt.")
        else:
            state = {
                "schema_version": PLAN_CHECKPOINT_SCHEMA_VERSION,
                "algorithm_version": PLAN_ALGORITHM_VERSION,
                "identity": identity_payload,
                "identity_fingerprint": identity_fingerprint,
                "status": "running",
                "total_groups": total_groups,
                "requested_pilot_groups": int(pilot_groups),
                "effective_pilot_groups": effective_pilot_groups,
                "completed_groups": [],
                "group_records": {},
                "pilot_rows": 0,
                "pilot_elapsed_seconds": 0.0,
            }
            if directory is not None and manifest_path is not None:
                _atomic_json(manifest_path, state)

        _emit_plan_progress(
            progress,
            status="started",
            completed_groups=len(completed),
            total_groups=total_groups,
            effective_pilot_groups=effective_pilot_groups,
            current_group=None,
            shard_index=None,
            elapsed_seconds=pilot_seconds,
            resumed=resumed,
        )
        if effective_pilot_groups == 0:
            plan = make_plan(None, None)
            if directory is not None and manifest_path is not None:
                state.update({"status": "completed", "plan": plan.to_dict()})
                _atomic_json(manifest_path, state)
            return plan

        guard = _PlanSignalGuard()
        try:
            with guard:
                guard.check()
                for shard in _iter_validated_shards(routing):
                    for local, group in enumerate(shard.descriptor.destination_group_indices):
                        if group >= effective_pilot_groups:
                            break
                        if group in completed:
                            continue
                        started = perf_counter()
                        row_count = 0
                        probabilities = np.asarray(shard.group_link_probability[local])
                        enabled = np.asarray(shard.effective_group_link_mask[local], dtype=bool)
                        for od_index in group_indices[group][group_mask[group]].tolist():
                            contributions = _group_ride_probabilities(
                                graph=assignment_inputs.graph,
                                probability=probabilities,
                                enabled=enabled,
                                origin_node=int(origins[od_index]),
                            )
                            row_count += sum(
                                float(demand[od_index]) * value > 0.0
                                for value in contributions.values()
                            )
                        duration = perf_counter() - started
                        pilot_rows += int(row_count)
                        pilot_seconds += float(duration)
                        if directory is not None and manifest_path is not None:
                            group_path = _save_group_checkpoint(
                                directory=directory,
                                group=group,
                                shard_index=shard.descriptor.shard_index,
                                row_count=row_count,
                                duration_seconds=duration,
                            )
                            group_records[str(group)] = {
                                "group_index": group,
                                "shard_index": shard.descriptor.shard_index,
                                "row_count": int(row_count),
                                "duration_seconds": float(duration),
                                "checkpoint_path": str(group_path.relative_to(directory)),
                                "sha256": _sha256(group_path),
                            }
                            completed.add(group)
                            state.update(
                                {
                                    "status": "running",
                                    "completed_groups": sorted(completed),
                                    "group_records": group_records,
                                    "pilot_rows": int(pilot_rows),
                                    "pilot_elapsed_seconds": float(pilot_seconds),
                                }
                            )
                            _atomic_json(manifest_path, state)
                        else:
                            completed.add(group)
                        _emit_plan_progress(
                            progress,
                            status="running",
                            completed_groups=len(completed),
                            total_groups=total_groups,
                            effective_pilot_groups=effective_pilot_groups,
                            current_group=group,
                            shard_index=shard.descriptor.shard_index,
                            elapsed_seconds=pilot_seconds,
                            group_duration_seconds=duration,
                            resumed=resumed,
                        )
                        guard.check()
                    if len(completed) >= effective_pilot_groups:
                        break
                guard.check()
                plan = make_plan(pilot_seconds, pilot_rows)
                if directory is not None and manifest_path is not None:
                    state.update({"status": "completed", "plan": plan.to_dict()})
                    _atomic_json(manifest_path, state)
                _emit_plan_progress(
                    progress,
                    status="completed",
                    completed_groups=len(completed),
                    total_groups=total_groups,
                    effective_pilot_groups=effective_pilot_groups,
                    current_group=None,
                    shard_index=None,
                    elapsed_seconds=pilot_seconds,
                    resumed=resumed,
                )
                return plan
        except BaseException as error:
            if directory is not None and manifest_path is not None and state is not None:
                state.update(
                    {
                        "status": "interrupted"
                        if guard.requested or isinstance(error, (ODSectionPlanInterrupted, KeyboardInterrupt))
                        else "failed",
                        "completed_groups": sorted(completed),
                        "group_records": group_records,
                        "pilot_rows": int(pilot_rows),
                        "pilot_elapsed_seconds": float(pilot_seconds),
                        "last_error": str(error),
                    }
                )
                _atomic_json(manifest_path, state)
            raise

    if checkpoint_directory is None:
        return run_pilot(directory=None, initial_checkpoint=None)
    directory = Path(checkpoint_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with _PlanCheckpointLock(directory):
        manifest_path = _checkpoint_manifest_path(directory)
        checkpoint = _read_plan_checkpoint(
            path=manifest_path,
            identity_payload=identity_payload,
            identity_fingerprint=identity_fingerprint,
            resume=resume,
        )
        if checkpoint.get("status") != "completed" and (
            int(checkpoint.get("total_groups", total_groups)) not in {0, total_groups}
            or int(checkpoint.get("effective_pilot_groups", effective_pilot_groups))
            not in {0, effective_pilot_groups}
        ):
            raise ODSectionPlanCheckpointError("pilot checkpoint group-count settings differ from current inputs.")
        if not manifest_path.exists():
            checkpoint.update(
                {
                    "total_groups": total_groups,
                    "requested_pilot_groups": int(pilot_groups),
                    "effective_pilot_groups": effective_pilot_groups,
                }
            )
            _atomic_json(manifest_path, checkpoint)
        return run_pilot(directory=directory, initial_checkpoint=checkpoint)


def export_od_section_marginals(
    *,
    assignment_inputs: AssignmentInputs,
    routing: ShardedFixedRoutingInputs,
    fitted_demand: Sequence[float] | np.ndarray,
    od_cells: Sequence[ODSectionCell | Mapping[str, object]],
    output_path: str | Path,
    checkpoint_directory: str | Path | None = None,
    compact_layout: CompactODAssignmentLayout | None = None,
    fit_assignment_fingerprint: str | None = None,
    node_stop_ids: Mapping[int, str] | Sequence[str] | None = None,
    ride_link_metadata: Mapping[int, RideLinkSectionMetadata | Mapping[str, object]] | None = None,
    zero_tolerance: float = 0.0,
    progress: ODSectionProgress | None = None,
) -> dict[str, object]:
    """Export sparse expected OD use of ride sections with group checkpoints."""
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative.")
    destination = Path(output_path).expanduser().resolve()
    checkpoint = Path(checkpoint_directory or destination.with_name(f".{destination.name}.checkpoint")).expanduser().resolve()
    state_path = checkpoint / "manifest.json"
    units_directory = checkpoint / "units"
    preparation_fingerprint = _validate_routing_cache(inputs=assignment_inputs, routing=routing)
    origins, group_indices, group_mask = _validate_assignment_layout(assignment_inputs)
    stop_ids = _node_stop_ids(assignment_inputs.graph, node_stop_ids)
    demand, cells = _normalise_demand(
        fitted_demand,
        od_cells=od_cells,
        compact_layout=compact_layout,
        expected_active=origins.size,
    )
    _validate_od_cell_binding(
        inputs=assignment_inputs,
        origins=origins,
        group_indices=group_indices,
        group_mask=group_mask,
        cells=cells,
        node_stop_ids=stop_ids,
    )
    fit_fingerprint = str(fit_assignment_fingerprint or routing.assignment_fingerprint)
    if not fit_fingerprint:
        raise ValueError("fit_assignment_fingerprint must be provided or present in routing identity.")
    ride_links = _ride_links(assignment_inputs.graph)
    metadata = _normalise_link_metadata(assignment_inputs.graph, ride_link_metadata, ride_links, stop_ids)
    provenance = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "routing_preparation_fingerprint": preparation_fingerprint,
        "fit_assignment_fingerprint": fit_fingerprint,
        "fitted_demand_fingerprint": _demand_fingerprint(demand),
        "routing_cache_directory": str(routing.provenance.cache_directory),
        "routing_cache_identity": f"{preparation_fingerprint}:{routing.provenance.schema_version}",
        "num_od": int(origins.size),
        "num_destination_groups": int(group_indices.shape[0]),
        "num_ride_links": int(ride_links.size),
    }
    state: dict[str, object] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("OD section export checkpoint is corrupt.") from error
        if state.get("provenance") != provenance:
            raise ValueError("OD section export checkpoint provenance differs from current inputs.")
    else:
        state = {"status": "running", "provenance": provenance, "completed_units": {}, "rows_written": 0}
        _atomic_json(state_path, state)
    completed = dict(state.get("completed_units", {}))
    if destination.exists() and state.get("status") != "completed":
        raise FileExistsError("output_path already exists for an incomplete OD section export.")
    units_directory.mkdir(parents=True, exist_ok=True)
    total_groups = int(group_indices.shape[0])
    rows_written = sum(int(item.get("rows", 0)) for item in completed.values() if isinstance(item, Mapping))
    started = perf_counter()
    for descriptor in routing.shard_partition:
        # At most one routing shard is retained while processing its groups.
        shard = load_fixed_routing_shard(routing=routing, descriptor=descriptor)
        for group in descriptor.destination_group_indices:
            unit_key = str(group)
            unit_path = units_directory / f"group-{group:08d}.csv"
            record = completed.get(unit_key)
            if isinstance(record, Mapping) and unit_path.exists() and record.get("sha256") == _sha256(unit_path):
                continue
            local = group - descriptor.group_start
            valid_indices = group_indices[group][group_mask[group]]
            descriptor_fd, temporary_name = tempfile.mkstemp(prefix=f".{unit_path.name}.", dir=units_directory)
            unit_rows = 0
            try:
                with os.fdopen(descriptor_fd, "w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(_ROUTE_FLOW_COLUMNS), extrasaction="raise")
                    writer.writeheader()
                    for od_index in valid_indices.tolist():
                        contributions = _group_ride_probabilities(
                            graph=assignment_inputs.graph,
                            probability=np.asarray(shard.group_link_probability[local]),
                            enabled=np.asarray(shard.effective_group_link_mask[local], dtype=bool),
                            origin_node=int(origins[od_index]),
                        )
                        for link, probability in contributions.items():
                            value = float(demand[od_index]) * probability
                            if value <= zero_tolerance:
                                continue
                            link_meta = metadata[link]
                            writer.writerow(_route_flow_row({
                                "origin_stop_id": cells[od_index].origin_stop_id,
                                "destination_stop_id": cells[od_index].destination_stop_id,
                                "departure_time_bin": cells[od_index].departure_time_bin,
                                "line_id": link_meta.line_id,
                                "trip_id": link_meta.trip_id,
                                "boarding_stop_id": link_meta.boarding_stop_id,
                                "alighting_stop_id": link_meta.alighting_stop_id,
                                "assigned_demand": value,
                                "flow_semantics": "service_marginal",
                                "pattern_id": link_meta.pattern_id,
                                "direction": link_meta.direction,
                                "assignment_fingerprint": fit_fingerprint,
                            }))
                            unit_rows += 1
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, unit_path)
            except BaseException:
                Path(temporary_name).unlink(missing_ok=True)
                raise
            completed[unit_key] = {"path": str(unit_path), "rows": unit_rows, "sha256": _sha256(unit_path)}
            rows_written = sum(int(item.get("rows", 0)) for item in completed.values() if isinstance(item, Mapping))
            state.update({"status": "running", "completed_units": completed, "rows_written": rows_written})
            _atomic_json(state_path, state)
            if progress is not None:
                progress({"group": group, "groups_completed": len(completed), "total_groups": total_groups, "rows_written": rows_written, "elapsed_seconds": perf_counter() - started})
    temporary_output = destination.with_name(f".{destination.name}.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_output.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(_ROUTE_FLOW_COLUMNS), extrasaction="raise")
            writer.writeheader()
            for group in range(total_groups):
                unit_path = units_directory / f"group-{group:08d}.csv"
                if not unit_path.exists() or _sha256(unit_path) != completed[str(group)]["sha256"]:
                    raise ValueError(f"OD section export unit {group} is missing or corrupt.")
                with unit_path.open(newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    for row in reader:
                        writer.writerow(row)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_output, destination)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise
    final = {
        **provenance,
        "status": "completed",
        "output_path": str(destination),
        "output_sha256": _sha256(destination),
        "rows_written": rows_written,
        "completed_units": completed,
    }
    _atomic_json(state_path, final)
    if progress is not None:
        progress({"status": "completed", "groups_completed": total_groups, "total_groups": total_groups, "rows_written": rows_written, "elapsed_seconds": perf_counter() - started})
    return final


# Naming aliases matching the existing gravity route-flow export facade.
export_gravity_od_section_marginals = export_od_section_marginals
plan_gravity_od_section_marginal_export = plan_od_section_marginal_export


__all__ = [
    "EXPORT_SCHEMA_VERSION",
    "PLAN_ALGORITHM_VERSION",
    "PLAN_CHECKPOINT_SCHEMA_VERSION",
    "ODSectionCell",
    "ODSectionExportPlan",
    "ODSectionPlanCheckpointError",
    "ODSectionPlanInterrupted",
    "RideLinkSectionMetadata",
    "export_od_section_marginals",
    "export_gravity_od_section_marginals",
    "plan_od_section_marginal_export",
    "plan_gravity_od_section_marginal_export",
]
