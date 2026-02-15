"""
Build JAX-friendly OD groupings to support batched assignment.

The Dial-style loading described in the document is most efficient when we:
- group OD demand records by (destination, desired time interval),
- compute one backward value function per group,
- propagate multiple origins in parallel in the forward pass.

This module provides utilities that convert the domain-layer Demand object
into compact index arrays used by JAX code.

Important
---------
This is a *structural* builder only. It does not compute flows.
It produces stable integer arrays so that assignment can be jit-compiled.

The OD demand *values* (flows) are not stored here: they are the parameters
estimated during inference and will be provided as a vector aligned with
the `od_index` array produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import jax.numpy as jnp

from .jax_graph_types import JaxOD

if TYPE_CHECKING:  # pragma: no cover
    from public_transportation.domain import Scenario


Array = jnp.ndarray


@dataclass(frozen=True, slots=True)
class ODGroups:
    """
    Grouping of OD records for batched evaluation.

    All arrays refer to OD-record indices in the canonical order used to build
    the demand vector.

    Attributes
    ----------
    num_od : int
        Total number of OD records.

    od_origin_node : Array[int] shape (num_od,)
        Origin centroid node index for each OD record.

    od_dest_node : Array[int] shape (num_od,)
        Destination centroid node index for each OD record.

    od_a_min : Array[float] shape (num_od,)
        Desired departure interval lower bound, in minutes.

    od_b_min : Array[float] shape (num_od,)
        Desired departure interval upper bound, in minutes.

    group_start : Array[int] shape (num_groups + 1,)
        CSR-style pointers into `group_od_index`.

    group_dest_node : Array[int] shape (num_groups,)
        Destination node for each group.

    group_time_bin : Array[int] shape (num_groups,)
        Time-bin index for each group (as defined in Scenario.time_bins).

    group_od_index : Array[int] shape (num_od,)
        Concatenation of OD indices, grouped by (dest, time_bin).
    """

    num_od: int

    od_origin_node: Array
    od_dest_node: Array
    od_a_min: Array
    od_b_min: Array

    group_start: Array
    group_dest_node: Array
    group_time_bin: Array
    group_od_index: Array


def _to_minutes(seconds: float | int) -> float:
    """
    Convert seconds to minutes.

    :param seconds: Value in seconds.
    :return: Value in minutes.
    """
    return float(seconds) / 60.0


def build_od_groups(scenario: "Scenario") -> ODGroups:
    """
    Build OD record arrays and groupings from a Scenario.

    Assumptions about the domain structures
    ---------------------------------------
    - `scenario.demand.records` is an iterable of demand records.
    - Each demand record has:
        - origin_stop_id
        - dest_stop_id
        - time_bin_id  (or time_bin_index)
        - demand       (not used here; the value is provided later as parameters)

    - `scenario.time_bins` provides lower/upper bounds per bin, in seconds:
        - start_time
        - end_time

    Output alignment
    ----------------
    The returned OD ordering is exactly the iteration order of `scenario.demand.records`.
    The inference code should provide an OD vector aligned with this ordering.

    :param scenario: Domain scenario with demand and time bins.
    :return: ODGroups with JAX arrays.
    """
    if scenario.demand is None:
        raise ValueError("Scenario has no demand.")
    if scenario.time_bins is None or len(scenario.time_bins) == 0:
        raise ValueError("Scenario has no time bins.")

    # Stop -> centroid node id mapping (centroids are indexed like stops order in build_time_expanded)
    # Here we reproduce the same convention: sorted stop_ids.
    if isinstance(scenario.stops, dict):
        stop_ids_sorted = sorted(scenario.stops.keys())
    else:
        stop_ids_sorted = sorted(getattr(s, "stop_id", getattr(s, "id")) for s in scenario.stops)

    centroid_index = {sid: i for i, sid in enumerate(stop_ids_sorted)}

    # Time bins: map id/index -> (a,b) in minutes
    # We accept either 'time_bin_id' matching some attribute, or index-based.
    bin_bounds_min: dict[int, tuple[float, float]] = {}

    for idx, tb in enumerate(scenario.time_bins):
        a = _to_minutes(getattr(tb, "start_time"))
        b = _to_minutes(getattr(tb, "end_time"))
        bin_bounds_min[int(idx)] = (a, b)

    # Read OD records
    records = list(getattr(scenario.demand, "records"))
    num_od = len(records)
    if num_od == 0:
        raise ValueError("Demand has zero records.")

    od_origin_node = np.empty(num_od, dtype=int)
    od_dest_node = np.empty(num_od, dtype=int)
    od_a_min = np.empty(num_od, dtype=float)
    od_b_min = np.empty(num_od, dtype=float)
    od_time_bin = np.empty(num_od, dtype=int)

    for k, r in enumerate(records):
        o = str(getattr(r, "origin_stop_id"))
        d = str(getattr(r, "dest_stop_id"))
        tb = getattr(r, "time_bin_index", None)
        if tb is None:
            tb = getattr(r, "time_bin_id", None)
        if tb is None:
            raise ValueError("Demand record missing time_bin_index/time_bin_id.")
        tb = int(tb)

        if o not in centroid_index:
            raise ValueError(f"Unknown origin_stop_id in demand: {o}")
        if d not in centroid_index:
            raise ValueError(f"Unknown dest_stop_id in demand: {d}")
        if tb not in bin_bounds_min:
            raise ValueError(f"Unknown time bin index in demand: {tb}")

        od_origin_node[k] = int(centroid_index[o])
        od_dest_node[k] = int(centroid_index[d])
        a, b = bin_bounds_min[tb]
        od_a_min[k] = float(a)
        od_b_min[k] = float(b)
        od_time_bin[k] = tb

    # Group by (dest_node, time_bin)
    keys = np.stack([od_dest_node, od_time_bin], axis=1)
    # Stable lexicographic sort for deterministic ordering
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys_sorted = keys[order]

    # Identify group boundaries
    # A new group starts when key changes
    change = np.ones(num_od, dtype=bool)
    change[1:] = np.any(keys_sorted[1:] != keys_sorted[:-1], axis=1)
    group_starts = np.nonzero(change)[0]
    num_groups = int(group_starts.shape[0])

    group_start = np.empty(num_groups + 1, dtype=int)
    group_start[:-1] = group_starts
    group_start[-1] = num_od

    group_dest_node = np.empty(num_groups, dtype=int)
    group_time_bin = np.empty(num_groups, dtype=int)
    for g in range(num_groups):
        i0 = group_start[g]
        group_dest_node[g] = int(keys_sorted[i0, 0])
        group_time_bin[g] = int(keys_sorted[i0, 1])

    group_od_index = order.astype(int)

    return ODGroups(
        num_od=num_od,
        od_origin_node=jnp.asarray(od_origin_node),
        od_dest_node=jnp.asarray(od_dest_node),
        od_a_min=jnp.asarray(od_a_min),
        od_b_min=jnp.asarray(od_b_min),
        group_start=jnp.asarray(group_start),
        group_dest_node=jnp.asarray(group_dest_node),
        group_time_bin=jnp.asarray(group_time_bin),
        group_od_index=jnp.asarray(group_od_index),
    )