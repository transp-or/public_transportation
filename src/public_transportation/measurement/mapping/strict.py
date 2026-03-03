from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.assignment.graph_sentinels import (
    NODE_KIND_EVENT_ARR,
    NODE_KIND_EVENT_DEP,
)
from public_transportation.measurement.schema import MeasurementTable, MeasurementType

from .spec import AggregationSpec, MappingEntry, MappingInfo, MappingSpecResult
from .indexing import (
    AssignmentMappingIndex,
    build_assignment_mapping_index,
    links_for_boarding,
    links_for_alighting,
)


def build_mapping_spec_strict(
    *,
    id_manager: AssignmentIDManager,
    table: MeasurementTable,
    include_link_lists_for_report: bool = False,
) -> MappingSpecResult:
    """Strictly map MeasurementTable records to assignment links and return an aggregation spec.

    Structural only:
    - Does NOT use assignment.link_flow (no predictions computed here).
    - Returns (y_obs, spec, info) reusable inside JAX likelihood.

    Strictness:
    - Each record must match exactly one timetable event node.
    - Contributing links must be non-empty.
    """
    idx: AssignmentMappingIndex = build_assignment_mapping_index(id_manager)

    entries: list[MappingEntry] = []
    y_obs: list[float] = []

    meas_index_flat: list[int] = []
    link_index_flat: list[int] = []

    for i, r in enumerate(table.records):
        # Basic checks
        if r.value < 0.0:
            raise ValueError(f"Record {i}: value must be nonnegative, got {r.value}")

        stop_id = str(r.stop_id)
        try:
            s_idx = idx.stop_index_by_id[stop_id]
        except KeyError as e:
            raise ValueError(f"Unknown stop_id in measurements: {stop_id!r}") from e

        t_s = int(r.time.seconds_from_midnight)
        if t_s < 0 or t_s >= 24 * 3600:
            raise ValueError(
                f"Record {i}: time out of range (00:00:00–23:59:59): {r.time.to_string(include_seconds=True)!r}"
            )

        # Determine candidate trip indices
        trip_indices: list[int]
        if r.trip_id is not None:
            tid = str(r.trip_id)
            try:
                trip_indices = [idx.trip_index_by_id[tid]]
            except KeyError as e:
                raise ValueError(f"Unknown trip_id in measurements: {tid!r}") from e
        else:
            # MeasurementRecord guarantees at least one of trip_id/line_id exists
            assert r.line_id is not None
            lid = str(r.line_id)
            try:
                trip_indices = list(idx.trip_indices_by_line_id[lid])
            except KeyError as e:
                raise ValueError(f"Unknown line_id in measurements: {lid!r}") from e

        # Determine node_kind needed
        if r.measurement_type == MeasurementType.BOARDING:
            need_kind = int(NODE_KIND_EVENT_DEP)
            mtype_str = "boarding"
        elif r.measurement_type == MeasurementType.ALIGHTING:
            need_kind = int(NODE_KIND_EVENT_ARR)
            mtype_str = "alighting"
        else:
            raise ValueError(
                f"Record {i}: unsupported measurement_type={r.measurement_type.value!r} "
                "(supported for strict mapping: boarding, alighting)"
            )

        # Strictly find the unique event node
        candidates: list[int] = []
        for ti in trip_indices:
            key = (need_kind, int(s_idx), int(ti), int(t_s))
            node = idx.event_node_index.get(key)
            if node is not None:
                candidates.append(int(node))

        if len(candidates) == 0:
            raise ValueError(
                f"Record {i}: no matching event node found for "
                f"(type={mtype_str}, stop_id={stop_id}, time={r.time.to_string(include_seconds=True)}, "
                f"trip_id={r.trip_id}, line_id={r.line_id})."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Record {i}: ambiguous mapping: multiple event nodes match "
                f"(type={mtype_str}, stop_id={stop_id}, time={r.time.to_string(include_seconds=True)}, "
                f"trip_id={r.trip_id}, line_id={r.line_id}): {candidates}"
            )
        event_node = candidates[0]

        # Contributing links
        if r.measurement_type == MeasurementType.BOARDING:
            link_ids = links_for_boarding(id_manager, event_node)
        else:
            link_ids = links_for_alighting(id_manager, event_node)

        if link_ids.size == 0:
            raise ValueError(
                f"Record {i}: matched event node {event_node} but found no contributing links for "
                f"measurement_type={mtype_str!r}."
            )

        link_ids_list = link_ids.tolist()
        meas_index_flat.extend([int(i)] * len(link_ids_list))
        link_index_flat.extend(int(lid) for lid in link_ids_list)

        matched_links = tuple(int(lid) for lid in link_ids_list) if include_link_lists_for_report else None

        y_obs.append(float(r.value))
        time_hms = r.time.to_string(include_seconds=True)

        entries.append(
            MappingEntry(
                row_index=i,
                measurement_type=mtype_str,
                method_id=str(r.method_id),
                stop_id=stop_id,
                time_hms=time_hms,
                trip_id=str(r.trip_id) if r.trip_id is not None else None,
                line_id=str(r.line_id) if r.line_id is not None else None,
                observed_value=float(r.value),
                predicted_value=float("nan"),
                matched_event_node=int(event_node),
                matched_link_indices=matched_links,
            )
        )

    info = MappingInfo(entries=tuple(entries), fingerprint=str(id_manager.fingerprint))

    spec = AggregationSpec(
        num_measurements=len(y_obs),
        measurement_index=np.ascontiguousarray(np.asarray(meas_index_flat, dtype=np.int32)),
        link_index=np.ascontiguousarray(np.asarray(link_index_flat, dtype=np.int32)),
    )

    return MappingSpecResult(
        y_obs=jnp.asarray(y_obs),
        spec=spec,
        info=info,
    )