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
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

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
) -> ODSectionExportPlan:
    """Validate inputs and estimate work without writing rows or preparing routing."""
    if bytes_per_row <= 0:
        raise ValueError("bytes_per_row must be positive.")
    if pilot_groups < 0:
        raise ValueError("pilot_groups must be non-negative.")
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
    pilot_seconds: float | None = None
    pilot_rows: int | None = None
    if pilot_groups:
        pilot_limit = min(int(pilot_groups), int(group_indices.shape[0]))
        pilot_started = perf_counter()
        pilot_rows = 0
        completed_groups = 0
        for shard in _iter_validated_shards(routing):
            for local, group in enumerate(shard.descriptor.destination_group_indices):
                if group >= pilot_limit:
                    break
                probabilities = np.asarray(shard.group_link_probability[local])
                enabled = np.asarray(shard.effective_group_link_mask[local], dtype=bool)
                for od_index in group_indices[group][group_mask[group]].tolist():
                    contributions = _group_ride_probabilities(
                        graph=assignment_inputs.graph,
                        probability=probabilities,
                        enabled=enabled,
                        origin_node=int(origins[od_index]),
                    )
                    pilot_rows += sum(
                        float(demand[od_index]) * value > 0.0
                        for value in contributions.values()
                    )
                completed_groups += 1
            if completed_groups >= pilot_limit:
                break
        pilot_seconds = perf_counter() - pilot_started
    representative = bool(pilot_groups and pilot_groups >= max(1, int(np.ceil(group_indices.shape[0] * 0.1))))
    estimated_full_seconds = (
        None
        if pilot_seconds is None or pilot_groups == 0
        else float(pilot_seconds) * int(group_indices.shape[0]) / max(1, min(int(pilot_groups), int(group_indices.shape[0])))
    )
    return ODSectionExportPlan(
        available=True,
        num_od=int(origins.size),
        num_destination_groups=int(group_indices.shape[0]),
        num_shards=int(routing.num_shards),
        ride_links=int(ride_links.size),
        potential_rows=int(potential_rows),
        estimated_bytes=int(potential_rows * bytes_per_row),
        estimated_memory_bytes=int(assignment_inputs.graph.num_nodes * 8 + assignment_inputs.graph.num_links * 16),
        pilot_groups=int(pilot_groups),
        pilot_seconds=pilot_seconds,
        pilot_rows=pilot_rows,
        pilot_peak_memory_bytes=int(assignment_inputs.graph.num_nodes * 8 + assignment_inputs.graph.num_links * 16) if pilot_seconds is not None else None,
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
    "ODSectionCell",
    "ODSectionExportPlan",
    "RideLinkSectionMetadata",
    "export_od_section_marginals",
    "export_gravity_od_section_marginals",
    "plan_od_section_marginal_export",
    "plan_gravity_od_section_marginal_export",
]
