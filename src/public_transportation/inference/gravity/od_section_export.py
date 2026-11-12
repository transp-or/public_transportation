"""Restartable export of expected OD use of scheduled ride sections.

The fixed-routing cache stores one outgoing-link probability vector per
destination group.  This module replays those probabilities in bounded OD
batches and writes sparse ``service_marginal`` rows for the scheduled ride
links.  It deliberately does not reconstruct passenger paths or run routing.
The scalar cell-by-cell traversal remains available as a diagnostic reference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import tempfile
from collections import deque
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
EXPORT_CHECKPOINT_SCHEMA_VERSION = 2
EXPORT_CHECKPOINT_ALGORITHM_VERSION = "od-section-export-chunked-v2"


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
    *,
    graph: Any,
    probability: np.ndarray,
    enabled: np.ndarray,
    origin_node: int,
    progress: Callable[[Mapping[str, object]], None] | None = None,
    progress_interval_seconds: float = 60.0,
) -> dict[int, float]:
    """Propagate unit mass through one group without retaining OD×link state."""
    node_mass = np.zeros(int(graph.num_nodes), dtype=np.float64)
    node_mass[int(origin_node)] = 1.0
    head = np.asarray(graph.head, dtype=np.int64)
    out_links = np.asarray(graph.out_links, dtype=np.int64)
    out_mask = np.asarray(graph.out_mask, dtype=bool)
    ride = np.asarray(graph.link_type) == LINK_TYPE_RIDE
    result: dict[int, float] = {}
    started = perf_counter()
    last_progress = started
    nodes_visited = 0
    for node in np.asarray(graph.topo_order, dtype=np.int64):
        nodes_visited += 1
        mass = float(node_mass[int(node)])
        if mass != 0.0:
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
        now = perf_counter()
        if progress is not None and (
            progress_interval_seconds <= 0.0
            or now - last_progress >= progress_interval_seconds
        ):
            progress(
                {
                    "status": "od_heartbeat",
                    "durable_checkpoint": False,
                    "nodes_visited": int(nodes_visited),
                    "total_nodes": int(graph.num_nodes),
                    "od_elapsed_seconds": float(now - started),
                }
            )
            last_progress = now
    return result


@dataclass(slots=True)
class _VectorizedRouteRunner:
    """Batch the same forward reachability pass used by linear-map construction."""

    batch_size: int
    edge_block_size: int
    section_links: np.ndarray
    reach_kernel: Any
    gather_kernel: Any
    graph_arrays: tuple[Any, ...]
    tail: Any
    jax: Any
    jnp: Any

    # Match the default OD dispatch size and support-edge block used by the
    # linear-mapping builder.  The cap keeps the retained reachability matrix
    # bounded even when an output checkpoint chunk is configured larger.
    MAX_BATCH_SIZE = 128
    EDGE_BLOCK_SIZE = 2048

    @classmethod
    def build(cls, *, graph: Any, batch_size: int) -> "_VectorizedRouteRunner":
        # Keep the implementation shared with the sharded linear-mapping
        # builder.  The kernels perform one graph traversal for a batch of OD
        # origins, then gather only the ride links needed by the exporter.
        from ..fixed_routing_sharded_builder import (
            _make_edge_gather_kernel,
            _make_forward_reachability_kernel,
        )

        import jax
        import jax.numpy as jnp

        link_type = np.asarray(graph.link_type)
        ride = link_type == LINK_TYPE_RIDE
        out_links = np.asarray(graph.out_links, dtype=np.int64)
        out_mask = np.asarray(graph.out_mask, dtype=bool)
        topo_order = np.asarray(graph.topo_order, dtype=np.int64)
        ordered: list[int] = []
        seen: set[int] = set()
        for node in topo_order:
            for link in out_links[int(node)][out_mask[int(node)]]:
                link_index = int(link)
                if ride[link_index] and link_index not in seen:
                    seen.add(link_index)
                    ordered.append(link_index)
        return cls(
            batch_size=int(batch_size),
            edge_block_size=cls.EDGE_BLOCK_SIZE,
            section_links=np.asarray(ordered, dtype=np.int32),
            reach_kernel=_make_forward_reachability_kernel(
                chunk_size=int(batch_size), num_nodes=int(graph.num_nodes)
            ),
            gather_kernel=_make_edge_gather_kernel(
                chunk_size=int(batch_size), edge_block_size=cls.EDGE_BLOCK_SIZE
            ),
            graph_arrays=(
                jnp.asarray(graph.topo_order),
                jnp.asarray(graph.out_links),
                jnp.asarray(graph.out_mask),
                jnp.asarray(graph.head),
            ),
            tail=jnp.asarray(graph.tail),
            jax=jax,
            jnp=jnp,
        )

    def __call__(
        self,
        *,
        origin_nodes: np.ndarray,
        probability: np.ndarray,
        enabled: np.ndarray,
    ) -> list[dict[int, float]]:
        count = int(origin_nodes.size)
        if count <= 0 or count > self.batch_size:
            raise ValueError("vectorized route batch has an invalid size")
        selected = self.section_links[np.asarray(enabled[self.section_links], dtype=bool)]
        results: list[dict[int, float]] = [dict() for _ in range(count)]
        if selected.size == 0:
            return results
        padded_origins = np.zeros(self.batch_size, dtype=np.int32)
        padded_origins[:count] = np.asarray(origin_nodes, dtype=np.int32)
        valid = np.zeros(self.batch_size, dtype=bool)
        valid[:count] = True
        probability_device = self.jnp.asarray(probability)
        enabled_device = self.jnp.asarray(enabled)
        reach = self.reach_kernel(
            self.jnp.asarray(padded_origins),
            self.jnp.asarray(valid),
            probability_device,
            enabled_device,
            *self.graph_arrays,
        )
        blocks: list[tuple[Any, np.ndarray]] = []
        devices: list[Any] = []
        for start in range(0, selected.size, self.edge_block_size):
            links = selected[start : start + self.edge_block_size]
            padded_links = np.zeros(self.edge_block_size, dtype=np.int32)
            padded_links[: links.size] = links
            link_mask = np.zeros(self.edge_block_size, dtype=bool)
            link_mask[: links.size] = True
            devices.append(
                self.gather_kernel(
                    reach,
                    probability_device,
                    enabled_device,
                    self.jnp.asarray(padded_links),
                    self.jnp.asarray(link_mask),
                    self.tail,
                )
            )
            blocks.append((devices[-1], links))
        self.jax.block_until_ready(devices)
        for device, links in blocks:
            values = np.asarray(device)[:count, : links.size]
            for local, result in enumerate(results):
                nonzero = np.flatnonzero(values[local])
                for position in nonzero:
                    result[int(links[position])] = float(values[local, position])
        return results


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


def _write_chunk_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> str:
    """Write one bounded chunk durably and return its checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(_ROUTE_FLOW_COLUMNS), extrasaction="raise"
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return _sha256(path)


def _export_eta(
    *,
    durations: deque[float],
    completed_cells: int,
    total_cells: int,
    elapsed_seconds: float,
) -> tuple[float | None, float | None, str, str]:
    if not durations or completed_cells <= 0:
        return None, None, "unavailable", "no completed OD-cell timing observations"
    # A rolling median avoids a single unusually expensive cell creating false
    # precision while still adapting to the current network and hardware.
    observations = np.asarray(tuple(durations), dtype=np.float64)
    seconds_per_cell = float(np.median(observations))
    throughput = 1.0 / seconds_per_cell if seconds_per_cell > 0.0 else None
    remaining = max(0, total_cells - completed_cells)
    eta = None if throughput is None else remaining / throughput
    if len(durations) < 3:
        confidence = "low"
        reason = "fewer than three completed OD-cell observations"
    elif len(durations) < 10:
        confidence = "medium"
        reason = "rolling median from a limited OD-cell sample"
    else:
        confidence = "high"
        reason = "rolling median from at least ten OD-cell observations"
    return throughput, eta, confidence, reason


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
    od_chunk_size: int = 256,
    progress_interval_seconds: float = 60.0,
    use_vectorized_route_kernel: bool = True,
    progress: ODSectionProgress | None = None,
) -> dict[str, object]:
    """Export sparse expected OD use of ride sections with chunk checkpoints."""
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative.")
    if od_chunk_size <= 0:
        raise ValueError("od_chunk_size must be positive.")
    if not np.isfinite(progress_interval_seconds) or progress_interval_seconds < 0.0:
        raise ValueError("progress_interval_seconds must be finite and non-negative.")
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
    total_groups = int(group_indices.shape[0])
    total_active_cells = int(np.count_nonzero(group_mask))
    cell_durations: deque[float] = deque(maxlen=64)
    routing_identity = sharded_fixed_routing_identity(
        inputs=assignment_inputs,
        theta=routing.theta,
        shard_partition=routing.shard_partition,
    )
    od_cell_fingerprint = fingerprint(
        tuple(
            (cell.origin_stop_id, cell.destination_stop_id, cell.departure_time_bin)
            for cell in cells
        )
    )
    provenance = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "checkpoint_schema_version": EXPORT_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_algorithm_version": EXPORT_CHECKPOINT_ALGORITHM_VERSION,
        "package_revision": __version__,
        "graph_fingerprint": routing_identity["graph_fingerprint"],
        "routing_preparation_fingerprint": preparation_fingerprint,
        "fit_assignment_fingerprint": fit_fingerprint,
        "fitted_demand_fingerprint": _demand_fingerprint(demand),
        "od_cell_fingerprint": od_cell_fingerprint,
        "compact_layout_fingerprint": (
            None if compact_layout is None else compact_layout.fingerprint
        ),
        "routing_cache_directory": str(routing.provenance.cache_directory),
        "routing_cache_identity": f"{preparation_fingerprint}:{routing.provenance.schema_version}",
        "num_od": int(origins.size),
        "num_destination_groups": total_groups,
        "num_ride_links": int(ride_links.size),
        "flow_semantics": "service_marginal",
        "output_columns": list(_ROUTE_FLOW_COLUMNS),
        "zero_tolerance": float(zero_tolerance),
        "od_chunk_size": int(od_chunk_size),
    }

    def _progress_event(
        *,
        status: str,
        group: int | None,
        shard_index: int | None,
        completed_groups: int,
        completed_group_cells: int | None,
        total_group_cells: int | None,
        completed_cells: int,
        rows: int,
        elapsed: float,
        cell: int | None = None,
        durable: bool = False,
        chunk_index: int | None = None,
        resumed: bool = False,
        **extra: object,
    ) -> None:
        if progress is None:
            return
        throughput, eta, confidence, reason = _export_eta(
            durations=cell_durations,
            completed_cells=completed_cells,
            total_cells=total_active_cells,
            elapsed_seconds=elapsed,
        )
        progress(
            {
                "status": status,
                "current_group": group,
                "group": group,
                "shard_index": shard_index,
                "completed_groups": int(completed_groups),
                "total_groups": total_groups,
                "completed_group_od_cells": completed_group_cells,
                "total_group_od_cells": total_group_cells,
                "completed_od_cells": int(completed_cells),
                "total_od_cells": total_active_cells,
                "current_od_cell": cell,
                "rows_written": int(rows),
                "elapsed_seconds": float(elapsed),
                "cells_per_second": throughput,
                "estimated_remaining_seconds": eta,
                "eta_confidence": confidence,
                "eta_reason": reason,
                "durable_checkpoint": durable,
                "chunk_index": chunk_index,
                "resumed": resumed,
                **extra,
            }
        )

    def _read_or_initialize_state() -> tuple[dict[str, object], bool]:
        if not state_path.exists():
            state = {
                "status": "running",
                "provenance": provenance,
                "completed_units": {},
                "completed_groups": [],
                "rows_written": 0,
                "completed_od_cells": 0,
                "total_od_cells": total_active_cells,
                "elapsed_seconds": 0.0,
            }
            _atomic_json(state_path, state)
            return state, False
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("OD section export checkpoint is corrupt.") from error
        if not isinstance(state, dict):
            raise ValueError("OD section export checkpoint must contain a JSON object.")
        legacy_keys = {
            "schema_version",
            "routing_preparation_fingerprint",
            "fit_assignment_fingerprint",
            "fitted_demand_fingerprint",
            "routing_cache_directory",
            "routing_cache_identity",
            "num_od",
            "num_destination_groups",
            "num_ride_links",
        }
        saved_provenance = state.get("provenance")
        if not isinstance(saved_provenance, Mapping):
            # Schema-1 completed manifests stamped provenance at the top level.
            # Accept that shape only when every legacy identity field is present;
            # all other missing-provenance manifests fail closed.
            legacy_top_level = {
                key: state[key] for key in legacy_keys if key in state
            }
            if set(legacy_top_level) != legacy_keys:
                raise ValueError("OD section export checkpoint has no provenance.")
            saved_provenance = legacy_top_level
        migrated = False
        if dict(saved_provenance) != provenance:
            legacy_compatible = (
                set(saved_provenance).issubset(legacy_keys)
                and all(provenance.get(key) == saved_provenance.get(key) for key in saved_provenance)
                and zero_tolerance == 0.0
            )
            if not legacy_compatible:
                raise ValueError("OD section export checkpoint provenance differs from current inputs.")
            migrated = True
            legacy_units = state.get("completed_units", {})
            if not isinstance(legacy_units, dict):
                raise ValueError("legacy OD section export checkpoint units are corrupt.")
            migrated_units: dict[str, dict[str, object]] = {}
            for key, record in legacy_units.items():
                if not isinstance(record, Mapping):
                    raise ValueError("legacy OD section export checkpoint contains an invalid group record.")
                path = Path(str(record.get("path", ""))).expanduser()
                if not path.is_absolute():
                    path = checkpoint / path
                if not path.is_file() or record.get("sha256") != _sha256(path):
                    raise ValueError(f"legacy OD section group checkpoint is missing or corrupt: {path}")
                try:
                    group = int(key)
                except (TypeError, ValueError) as error:
                    raise ValueError("legacy OD section export contains an invalid group index.") from error
                if group < 0 or group >= total_groups:
                    raise ValueError(f"legacy OD section export group index is out of range: {group}")
                active_count = int(np.count_nonzero(group_mask[group]))
                migrated_units[str(group)] = {
                    "unit_type": "legacy_group",
                    "status": "completed",
                    "path": str(path),
                    "rows": int(record.get("rows", 0)),
                    "sha256": str(record["sha256"]),
                    "completed_od_cells": active_count,
                    "total_od_cells": active_count,
                    "duration_seconds": float(record.get("duration_seconds", 0.0)),
                }
            state["completed_units"] = migrated_units
            state["completed_groups"] = sorted(int(key) for key in migrated_units)
            state["rows_written"] = sum(int(item["rows"]) for item in migrated_units.values())
            state["completed_od_cells"] = sum(int(item["completed_od_cells"]) for item in migrated_units.values())
            state["total_od_cells"] = total_active_cells
            state["provenance"] = provenance
            state["migration"] = "verified_group_units_to_chunked_checkpoint"
            _atomic_json(state_path, state)
        return state, migrated

    def _validate_chunk_record(record: Mapping[str, object], *, group: int) -> tuple[int, int, int]:
        path = checkpoint / str(record.get("path", ""))
        if not path.is_file() or record.get("sha256") != _sha256(path):
            raise ValueError(f"OD section export chunk is missing or corrupt: {path}")
        return (
            int(record.get("od_start", -1)),
            int(record.get("od_stop", -1)),
            int(record.get("rows", 0)),
        )

    def _run() -> dict[str, object]:
        state, migrated = _read_or_initialize_state()
        if destination.exists() and state.get("status") != "completed":
            raise FileExistsError("output_path already exists for an incomplete OD section export.")
        units_directory.mkdir(parents=True, exist_ok=True)
        completed_units = state.get("completed_units", {})
        if not isinstance(completed_units, dict):
            raise ValueError("OD section export checkpoint units are corrupt.")
        completed_groups = {
            int(key)
            for key, record in completed_units.items()
            if isinstance(record, Mapping) and record.get("status") == "completed"
        }
        completed_cells = int(state.get("completed_od_cells", 0))
        rows_written = int(state.get("rows_written", 0))
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
        run_started = perf_counter()
        for record in completed_units.values():
            if isinstance(record, Mapping):
                chunks = record.get("chunks", [])
                if isinstance(chunks, list):
                    for chunk in chunks:
                        if isinstance(chunk, Mapping):
                            od_start, od_stop, _ = _validate_chunk_record(chunk, group=-1)
                            if od_stop > od_start:
                                cell_durations.append(
                                    float(chunk.get("duration_seconds", 0.0)) / (od_stop - od_start)
                                )
        resumed = bool(completed_units) or migrated
        vectorized_runner = None
        if use_vectorized_route_kernel:
            vectorized_runner = _VectorizedRouteRunner.build(
                graph=assignment_inputs.graph,
                batch_size=min(int(od_chunk_size), _VectorizedRouteRunner.MAX_BATCH_SIZE),
            )
        guard = _PlanSignalGuard()
        try:
            with guard:
                for descriptor in routing.shard_partition:
                    shard = load_fixed_routing_shard(routing=routing, descriptor=descriptor)
                    for local, group in enumerate(descriptor.destination_group_indices):
                        local_indices = np.asarray(group_indices[group][group_mask[group]], dtype=np.int64)
                        total_group_cells = int(local_indices.size)
                        unit_key = str(group)
                        record = completed_units.get(unit_key)
                        chunks: list[dict[str, object]] = []
                        next_cell = 0
                        group_rows = 0
                        group_elapsed = 0.0
                        if isinstance(record, Mapping) and record.get("unit_type") == "legacy_group":
                            path = Path(str(record["path"])).expanduser()
                            if not path.is_absolute():
                                path = checkpoint / path
                            if not path.is_file() or record.get("sha256") != _sha256(path):
                                raise ValueError(f"legacy OD section group checkpoint is missing or corrupt: {path}")
                            completed_groups.add(group)
                            continue
                        if isinstance(record, Mapping):
                            raw_chunks = record.get("chunks", [])
                            if not isinstance(raw_chunks, list):
                                raise ValueError(f"OD section export chunks are corrupt for group {group}.")
                            expected_start = 0
                            for raw_chunk in raw_chunks:
                                if not isinstance(raw_chunk, Mapping):
                                    raise ValueError(f"OD section export chunk record is corrupt for group {group}.")
                                od_start, od_stop, rows = _validate_chunk_record(raw_chunk, group=group)
                                if od_start != expected_start or od_stop <= od_start or od_stop > total_group_cells:
                                    raise ValueError(f"OD section export chunk ranges are corrupt for group {group}.")
                                expected_start = od_stop
                                chunk = dict(raw_chunk)
                                chunks.append(chunk)
                                next_cell = od_stop
                                group_rows += rows
                                group_elapsed += float(chunk.get("duration_seconds", 0.0))
                                cell_durations.append(float(chunk.get("duration_seconds", 0.0)) / (od_stop - od_start))
                            if int(record.get("completed_od_cells", next_cell)) != next_cell:
                                raise ValueError(f"OD section export completed cell count is corrupt for group {group}.")
                            if str(record.get("status", "running")) == "completed":
                                if next_cell != total_group_cells:
                                    raise ValueError(f"OD section export completed group {group} has missing chunks.")
                                completed_groups.add(group)
                                continue
                        _progress_event(
                            status="group_started",
                            group=group,
                            shard_index=descriptor.shard_index,
                            completed_groups=len(completed_groups),
                            completed_group_cells=next_cell,
                            total_group_cells=total_group_cells,
                            completed_cells=completed_cells,
                            rows=rows_written,
                            elapsed=elapsed_before + perf_counter() - run_started,
                            resumed=resumed,
                        )
                        probabilities = np.asarray(shard.group_link_probability[local])
                        enabled = np.asarray(shard.effective_group_link_mask[local], dtype=bool)

                        def append_cell_rows(
                            *,
                            od_index: int,
                            contributions: Mapping[int, float],
                            rows: list[dict[str, object]],
                        ) -> None:
                            for link, probability in contributions.items():
                                value = float(demand[od_index]) * probability
                                if value <= zero_tolerance:
                                    continue
                                link_meta = metadata[link]
                                rows.append(_route_flow_row({
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

                        while next_cell < total_group_cells:
                            chunk_start = next_cell
                            chunk_rows: list[dict[str, object]] = []
                            chunk_started = perf_counter()
                            chunk_cells = min(total_group_cells, chunk_start + int(od_chunk_size))
                            if vectorized_runner is not None:
                                while next_cell < chunk_cells:
                                    batch_start = next_cell
                                    batch_stop = min(
                                        chunk_cells,
                                        batch_start + vectorized_runner.batch_size,
                                    )
                                    batch_indices = np.asarray(
                                        local_indices[batch_start:batch_stop],
                                        dtype=np.int64,
                                    )
                                    batch_started = perf_counter()
                                    batch_contributions = vectorized_runner(
                                        origin_nodes=np.asarray(origins[batch_indices], dtype=np.int32),
                                        probability=probabilities,
                                        enabled=enabled,
                                    )
                                    batch_duration = perf_counter() - batch_started
                                    for local_batch, od_index in enumerate(batch_indices):
                                        append_cell_rows(
                                            od_index=int(od_index),
                                            contributions=batch_contributions[local_batch],
                                            rows=chunk_rows,
                                        )
                                    count = batch_stop - batch_start
                                    cell_durations.extend([batch_duration / count] * count)
                                    completed_cells += count
                                    next_cell = batch_stop
                                    guard.check()
                                    _progress_event(
                                        status="od_batch_completed",
                                        group=group,
                                        shard_index=descriptor.shard_index,
                                        completed_groups=len(completed_groups),
                                        completed_group_cells=next_cell,
                                        total_group_cells=total_group_cells,
                                        completed_cells=completed_cells,
                                        rows=rows_written + len(chunk_rows),
                                        elapsed=elapsed_before + perf_counter() - run_started,
                                        cell=next_cell - 1,
                                        current_od_index=int(batch_indices[-1]),
                                        resumed=resumed,
                                        batch_size=count,
                                        vectorized_route_kernel=True,
                                    )
                            else:
                                while next_cell < chunk_cells:
                                    od_index = int(local_indices[next_cell])
                                    od_started = perf_counter()

                                    def on_od_heartbeat(event: Mapping[str, object]) -> None:
                                        guard.check()
                                        heartbeat_fields = {
                                            key: value
                                            for key, value in event.items()
                                            if key not in {"status", "durable_checkpoint"}
                                        }
                                        _progress_event(
                                            status="od_heartbeat",
                                            group=group,
                                            shard_index=descriptor.shard_index,
                                            completed_groups=len(completed_groups),
                                            completed_group_cells=next_cell,
                                            total_group_cells=total_group_cells,
                                            completed_cells=completed_cells,
                                            rows=rows_written,
                                            elapsed=elapsed_before + perf_counter() - run_started,
                                            cell=next_cell,
                                            resumed=resumed,
                                            current_od_index=od_index,
                                            **heartbeat_fields,
                                        )

                                    contributions = _group_ride_probabilities(
                                        graph=assignment_inputs.graph,
                                        probability=probabilities,
                                        enabled=enabled,
                                        origin_node=int(origins[od_index]),
                                        progress=on_od_heartbeat,
                                        progress_interval_seconds=progress_interval_seconds,
                                    )
                                    append_cell_rows(
                                        od_index=od_index,
                                        contributions=contributions,
                                        rows=chunk_rows,
                                    )
                                    cell_duration = perf_counter() - od_started
                                    cell_durations.append(cell_duration)
                                    completed_cells += 1
                                    next_cell += 1
                                    guard.check()
                            chunk_duration = perf_counter() - chunk_started
                            chunk_path = units_directory / f"group-{group:08d}" / f"chunk-{len(chunks):08d}.csv"
                            chunk_sha = _write_chunk_csv(chunk_path, chunk_rows)
                            chunk_record = {
                                "chunk_index": len(chunks),
                                "od_start": chunk_start,
                                "od_stop": chunk_cells,
                                "rows": len(chunk_rows),
                                "duration_seconds": float(chunk_duration),
                                "path": str(chunk_path.relative_to(checkpoint)),
                                "sha256": chunk_sha,
                            }
                            chunks.append(chunk_record)
                            group_rows += len(chunk_rows)
                            rows_written += len(chunk_rows)
                            group_elapsed += chunk_duration
                            completed_units[unit_key] = {
                                "unit_type": "group_chunks",
                                "status": "running",
                                "group_index": group,
                                "shard_index": descriptor.shard_index,
                                "completed_od_cells": chunk_cells,
                                "total_od_cells": total_group_cells,
                                "rows": group_rows,
                                "duration_seconds": group_elapsed,
                                "chunks": chunks,
                            }
                            state.update({
                                "status": "running",
                                "completed_units": completed_units,
                                "completed_groups": sorted(completed_groups),
                                "rows_written": rows_written,
                                "completed_od_cells": completed_cells,
                                "total_od_cells": total_active_cells,
                                "elapsed_seconds": elapsed_before + perf_counter() - run_started,
                            })
                            _atomic_json(state_path, state)
                            _progress_event(
                                status="chunk_checkpointed",
                                group=group,
                                shard_index=descriptor.shard_index,
                                completed_groups=len(completed_groups),
                                completed_group_cells=chunk_cells,
                                total_group_cells=total_group_cells,
                                completed_cells=completed_cells,
                                rows=rows_written,
                                elapsed=float(state["elapsed_seconds"]),
                                cell=chunk_cells - 1,
                                current_od_index=int(local_indices[chunk_cells - 1]),
                                durable=True,
                                chunk_index=len(chunks) - 1,
                                resumed=resumed,
                            )
                        completed_groups.add(group)
                        if unit_key not in completed_units:
                            completed_units[unit_key] = {
                                "unit_type": "group_chunks",
                                "status": "running",
                                "group_index": group,
                                "shard_index": descriptor.shard_index,
                                "completed_od_cells": 0,
                                "total_od_cells": total_group_cells,
                                "rows": 0,
                                "duration_seconds": 0.0,
                                "chunks": [],
                            }
                        completed_units[unit_key]["status"] = "completed"
                        state.update({
                            "status": "running",
                            "completed_units": completed_units,
                            "completed_groups": sorted(completed_groups),
                            "rows_written": rows_written,
                            "completed_od_cells": completed_cells,
                            "elapsed_seconds": elapsed_before + perf_counter() - run_started,
                        })
                        _atomic_json(state_path, state)
                        _progress_event(
                            status="group_completed",
                            group=group,
                            shard_index=descriptor.shard_index,
                            completed_groups=len(completed_groups),
                            completed_group_cells=total_group_cells,
                            total_group_cells=total_group_cells,
                            completed_cells=completed_cells,
                            rows=rows_written,
                            elapsed=float(state["elapsed_seconds"]),
                            durable=True,
                            resumed=resumed,
                        )
                    if len(completed_groups) >= total_groups:
                        break
                guard.check()
                temporary_output = destination.with_name(f".{destination.name}.tmp")
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with temporary_output.open("w", newline="", encoding="utf-8") as output:
                        writer = csv.DictWriter(output, fieldnames=list(_ROUTE_FLOW_COLUMNS), extrasaction="raise")
                        writer.writeheader()
                        for group in range(total_groups):
                            record = completed_units.get(str(group))
                            if not isinstance(record, Mapping) or record.get("status") != "completed":
                                raise ValueError(f"OD section export group {group} is incomplete.")
                            if record.get("unit_type") == "legacy_group":
                                legacy_path = Path(str(record["path"])).expanduser()
                                if not legacy_path.is_absolute():
                                    legacy_path = checkpoint / legacy_path
                                paths = (legacy_path,)
                            else:
                                paths = tuple(
                                    checkpoint / str(chunk["path"])
                                    for chunk in record.get("chunks", [])
                                    if isinstance(chunk, Mapping)
                                )
                            for path in paths:
                                with path.open(newline="", encoding="utf-8") as stream:
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
                    "provenance": provenance,
                    "status": "completed",
                    "output_path": str(destination),
                    "output_sha256": _sha256(destination),
                    "rows_written": rows_written,
                    "completed_units": completed_units,
                    "completed_groups": sorted(completed_groups),
                    "completed_od_cells": completed_cells,
                }
                if "migration" in state:
                    final["migration"] = state["migration"]
                state.update(final)
                _atomic_json(state_path, final)
                _progress_event(
                    status="completed",
                    group=None,
                    shard_index=None,
                    completed_groups=len(completed_groups),
                    completed_group_cells=None,
                    total_group_cells=None,
                    completed_cells=completed_cells,
                    rows=rows_written,
                    elapsed=float(state.get("elapsed_seconds", elapsed_before + perf_counter() - run_started)),
                    durable=True,
                    resumed=resumed,
                )
                return final
        except BaseException as error:
            if state.get("status") == "completed":
                raise
            state.update({
                "status": "interrupted"
                if guard.requested or isinstance(error, (ODSectionPlanInterrupted, KeyboardInterrupt, InterruptedError))
                else "failed",
                "completed_units": completed_units,
                "completed_groups": sorted(completed_groups),
                "rows_written": rows_written,
                "completed_od_cells": completed_cells,
                "total_od_cells": total_active_cells,
                "elapsed_seconds": elapsed_before + perf_counter() - run_started,
                "last_error": str(error),
            })
            _atomic_json(state_path, state)
            if progress is not None:
                _progress_event(
                    status=state["status"],
                    group=None,
                    shard_index=None,
                    completed_groups=len(completed_groups),
                    completed_group_cells=None,
                    total_group_cells=None,
                    completed_cells=completed_cells,
                    rows=rows_written,
                    elapsed=float(state["elapsed_seconds"]),
                    durable=True,
                    resumed=resumed,
                    error=str(error),
                )
            raise

    checkpoint.mkdir(parents=True, exist_ok=True)
    with _PlanCheckpointLock(checkpoint):
        return _run()


# Naming aliases matching the existing gravity route-flow export facade.
export_gravity_od_section_marginals = export_od_section_marginals
plan_gravity_od_section_marginal_export = plan_od_section_marginal_export


__all__ = [
    "EXPORT_SCHEMA_VERSION",
    "PLAN_ALGORITHM_VERSION",
    "PLAN_CHECKPOINT_SCHEMA_VERSION",
    "EXPORT_CHECKPOINT_SCHEMA_VERSION",
    "EXPORT_CHECKPOINT_ALGORITHM_VERSION",
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
