"""Compact direct aggregation for one- or two-link event measurements."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

from public_transportation.measurement.mapping.spec import AggregationSpec


@dataclass(frozen=True, slots=True)
class EventAlignedAggregationSpec:
    """One primary link per measurement plus sparse secondary contributions."""

    primary_link_index: np.ndarray
    secondary_measurement_index: np.ndarray
    secondary_link_index: np.ndarray

    @property
    def num_measurements(self) -> int:
        return int(self.primary_link_index.size)


def build_event_aligned_aggregation_spec(
    spec: AggregationSpec,
) -> EventAlignedAggregationSpec:
    """Convert a strict mapping when every measurement uses one or two links."""
    measurement = np.asarray(spec.measurement_index, dtype=np.int64)
    link = np.asarray(spec.link_index, dtype=np.int32)
    if measurement.ndim != 1 or link.ndim != 1 or measurement.shape != link.shape:
        raise ValueError("Aggregation indices must be matching one-dimensional arrays.")
    if np.any(measurement < 0) or np.any(measurement >= spec.num_measurements):
        raise ValueError("Aggregation measurement index is out of range.")
    order = np.argsort(measurement, kind="stable")
    sorted_measurement = measurement[order]
    sorted_link = link[order]
    counts = np.bincount(sorted_measurement, minlength=spec.num_measurements)
    if np.any(counts == 0):
        raise ValueError("Every event-aligned measurement must map to at least one link.")
    if np.any(counts > 2):
        raise ValueError("Event-aligned aggregation supports at most two links per measurement.")
    start = np.empty((spec.num_measurements + 1,), dtype=np.int64)
    start[0] = 0
    np.cumsum(counts, out=start[1:])
    primary = sorted_link[start[:-1]]
    secondary_measurement = np.flatnonzero(counts == 2).astype(np.int32)
    secondary_link = sorted_link[start[secondary_measurement] + 1]
    return EventAlignedAggregationSpec(
        primary_link_index=np.ascontiguousarray(primary, dtype=np.int32),
        secondary_measurement_index=np.ascontiguousarray(
            secondary_measurement, dtype=np.int32
        ),
        secondary_link_index=np.ascontiguousarray(secondary_link, dtype=np.int32),
    )


@jax.jit
def predict_measurements_event_aligned(
    link_flow: jax.Array,
    primary_link_index: jax.Array,
    secondary_measurement_index: jax.Array,
    secondary_link_index: jax.Array,
) -> jax.Array:
    """Gather primary event flows and add the sparse secondary access flows."""
    prediction = link_flow[primary_link_index]
    return prediction.at[secondary_measurement_index].add(
        link_flow[secondary_link_index]
    )
