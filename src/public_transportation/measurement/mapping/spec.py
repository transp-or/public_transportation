from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Mapping info (human/report side)
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MappingEntry:
    """One measurement mapped to assignment objects (strict, structural).

    - matched_event_node is the unique timetable event node matched for this record.
    - matched_link_indices is REPORT ONLY (may be None unless requested).
    - predicted_value is NaN in the structural mapping and can be filled later.
    """
    row_index: int
    measurement_type: str
    method_id: str
    stop_id: str
    time_hms: str
    trip_id: str | None
    line_id: str | None

    observed_value: float
    predicted_value: float

    matched_event_node: int
    matched_link_indices: tuple[int, ...] | None  # REPORT ONLY


@dataclass(frozen=True, slots=True)
class MappingInfo:
    """Metadata for debugging/reporting and mismatch detection."""
    entries: tuple[MappingEntry, ...]
    fingerprint: str  # copied from AssignmentIDManager


# -----------------------------------------------------------------------------
# JAX-safe aggregation spec
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AggregationSpec:
    """JAX-safe recipe to compute predicted measurements from link_flow.

    Interpretation:
      y_pred[m] = sum_{k: measurement_index[k]==m} link_flow[link_index[k]]
    """
    num_measurements: int
    measurement_index: np.ndarray  # (K,), int32
    link_index: np.ndarray         # (K,), int32


# -----------------------------------------------------------------------------
# Result containers (avoid tuple soup)
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MappingSpecResult:
    y_obs: jnp.ndarray
    spec: AggregationSpec
    info: MappingInfo


@dataclass(frozen=True, slots=True)
class MeasurementVectorsResult:
    y_obs: jnp.ndarray
    y_pred: jnp.ndarray
    spec: AggregationSpec
    info: MappingInfo