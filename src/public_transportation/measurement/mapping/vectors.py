from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from public_transportation.assignment.assign import AssignmentResult
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.measurement.schema import MeasurementTable

from .spec import MeasurementVectorsResult, MappingInfo, MappingEntry
from .strict import build_mapping_spec_strict
from .apply import apply_mapping_spec


def build_measurement_vectors(
    *,
    assignment: AssignmentResult,
    id_manager: AssignmentIDManager,
    table: MeasurementTable,
    include_link_lists_for_report: bool = False,
    enrich_predictions: bool = False,
) -> MeasurementVectorsResult:
    """Build aligned vectors (y_obs, y_pred) from MeasurementTable (strict mapping)."""
    msr = build_mapping_spec_strict(
        id_manager=id_manager,
        table=table,
        include_link_lists_for_report=include_link_lists_for_report,
    )

    y_pred = apply_mapping_spec(link_flow=assignment.link_flow, spec=msr.spec)

    if enrich_predictions:
        y_pred_np = np.asarray(y_pred)
        new_entries: list[MappingEntry] = []
        for e in msr.info.entries:
            new_entries.append(
                MappingEntry(
                    row_index=e.row_index,
                    measurement_type=e.measurement_type,
                    method_id=e.method_id,
                    stop_id=e.stop_id,
                    time_hms=e.time_hms,
                    trip_id=e.trip_id,
                    line_id=e.line_id,
                    observed_value=e.observed_value,
                    predicted_value=float(y_pred_np[e.row_index]),
                    matched_event_node=e.matched_event_node,
                    matched_link_indices=e.matched_link_indices,
                )
            )
        info = MappingInfo(entries=tuple(new_entries), fingerprint=msr.info.fingerprint)
    else:
        info = msr.info

    return MeasurementVectorsResult(
        y_obs=msr.y_obs,
        y_pred=jnp.asarray(y_pred),
        spec=msr.spec,
        info=info,
    )