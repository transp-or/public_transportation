"""
Build a JAX-compatible time-expanded graph from a domain-layer Scenario.

This module converts the flexible, user-facing domain representation
(stops, timetable, time bins) into a static, array-based representation that
is suitable for fast JAX evaluation.

Key design choices (current version)
------------------------------------
- Centroid-in nodes: one per (stop, time-bin), not time-tagged (conceptually -infty).
- Centroid-out nodes: one per stop, non time-tagged (conceptually +infty).
- Event nodes are trip-specific and split into ARR/DEP: (stop_id, arrival_s, trip_id) and (stop_id, departure_s, trip_id).
- Access links: (s_o, t)^{in} -> (s_o, dep_time, trip) only if dep_time lies in
  [time_bin.start_s - max_access_deviation_s, time_bin.end_s + max_access_deviation_s].
- Transfer links: (s, arr_time, trip1) -> (s, dep_time, trip2) only if
  - line(trip1) != line(trip2)
  - 0 < dep_time - arr_time <= max_transfer_wait
  (Not restricted to consecutive events: bounded waiting controls graph size.)
- Ride links: between consecutive stops within each trip, using DEP->ARR.
- Egress links: from each ARR(stop, arr_time, trip) -> centroid-out(stop), with zero cost.
  Destination-gating is handled later by masking.
- Dwell/continue links: (s, arrival_s, trip) -> (s, departure_s, trip) for the same trip/stop_time (arr<dep).
  If raw input has arr==dep, the builder auto-regularizes dep := arr + min_dwell_s (from AssignmentConfig) and emits a warning.

Notes
-----
- The builder expects stop_times to provide departure and arrival times in seconds-from-midnight.
- All physical times are stored in minutes (float).
- The resulting JaxGraph includes:
  - node_time (minutes) used for topological ordering,
  - node_time_s (seconds-from-midnight) for event nodes; CENTROID_TIME_S for centroids,
  - node_kind: NODE_KIND_* codes (centroid-in, event-arr, event-dep, centroid-out),
  - node_stop_index: ...
  - node_time_bin_index: time-bin index for centroid-in nodes; -1 for all other nodes,
  - node_bin_start_min / node_bin_end_min: departure-interval bounds (minutes) attached to centroid-in nodes; NaN for all other nodes.
  - link_type: LINK_TYPE_* codes (ride, transfer, access, egress, dwell),
  - link_trip_index for ride and event-derived links; -1 for links not tied to a trip.

This file is intended to be *self-contained* and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, TYPE_CHECKING

import numpy as np
import warnings
import jax
import jax.numpy as jnp

from .config import AssignmentConfig
from .jax_graph_types import JaxGraph
from .graph_sentinels import (
    CENTROID_IN_TIME_MIN,
    CENTROID_OUT_TIME_MIN,
    CENTROID_TIME_S,
    NODE_KIND_CENTROID_IN,
    NODE_KIND_EVENT_ARR,
    NODE_KIND_EVENT_DEP,
    NODE_KIND_CENTROID_OUT,
    LINK_TYPE_RIDE,
    LINK_TYPE_TRANSFER,
    LINK_TYPE_ACCESS,
    LINK_TYPE_EGRESS,
    LINK_TYPE_DWELL,
)



if TYPE_CHECKING:  # pragma: no cover
    from public_transportation.domain import Scenario


# =============================================================================
# Helpers
# =============================================================================


def _to_minutes(seconds: float | int) -> float:
    """
    Convert seconds to minutes.

    :param seconds: Time in seconds.
    :return: Time in minutes.
    """
    return float(seconds) / 60.0


def _require(cond: bool, msg: str) -> None:
    """
    Raise a ValueError if condition is false.

    :param cond: Condition to check.
    :param msg: Error message.
    :return: None
    :raises ValueError: if cond is False.
    """
    if not cond:
        raise ValueError(msg)


def _get_stop_id(st: Any) -> str:
    """
    Extract stop_id from a StopTime-like object.

    :param st: StopTime-like object.
    :return: stop_id
    """
    if hasattr(st, "stop_id"):
        return str(getattr(st, "stop_id"))
    raise AttributeError("StopTime-like object must provide `stop_id`.")


def _get_trip_id(st: Any) -> str:
    """
    Extract trip_id from a StopTime-like object.

    :param st: StopTime-like object.
    :return: trip_id
    """
    if hasattr(st, "trip_id"):
        return str(getattr(st, "trip_id"))
    raise AttributeError("StopTime-like object must provide `trip_id`.")


def _get_stop_sequence(st: Any) -> int:
    """
    Extract stop sequence from a StopTime-like object.

    :param st: StopTime-like object.
    :return: stop_sequence
    """
    if hasattr(st, "stop_sequence"):
        return int(getattr(st, "stop_sequence"))
    if hasattr(st, "sequence"):
        return int(getattr(st, "sequence"))
    raise AttributeError("StopTime-like object must provide `stop_sequence` or `sequence`.")


def _get_departure_seconds(st: Any) -> int:
    """
    Return departure time in seconds-from-midnight for a StopTime-like object.

    :param st: StopTime-like object.
    :return: departure seconds-from-midnight
    """
    if hasattr(st, "departure_time"):
        return int(getattr(st, "departure_time"))
    if hasattr(st, "departure_s"):
        return int(getattr(st, "departure_s"))
    if hasattr(st, "departure"):
        dep = getattr(st, "departure")
        if hasattr(dep, "seconds_from_midnight"):
            return int(getattr(dep, "seconds_from_midnight"))
    raise AttributeError(
        "StopTime-like object must provide `departure_time` (seconds) "
        "or `departure_s` or `departure.seconds_from_midnight`."
    )


def _get_arrival_seconds(st: Any) -> int:
    """Return arrival time in seconds-from-midnight for a StopTime-like object.

    :param st: StopTime-like object.
    :return: arrival seconds-from-midnight
    """
    if hasattr(st, "arrival_time"):
        return int(getattr(st, "arrival_time"))
    if hasattr(st, "arrival_s"):
        return int(getattr(st, "arrival_s"))
    if hasattr(st, "arrival"):
        arr = getattr(st, "arrival")
        if hasattr(arr, "seconds_from_midnight"):
            return int(getattr(arr, "seconds_from_midnight"))
    raise AttributeError(
        "StopTime-like object must provide `arrival_time` (seconds) "
        "or `arrival_s` or `arrival.seconds_from_midnight`."
    )

# -----------------------------------------------------------------------------
# Departure regularization helper
# -----------------------------------------------------------------------------
def _regularize_departure_seconds(*, arr_s: int, dep_s: int, min_dwell_s: int) -> int:
    """
    If dep_s == arr_s, return arr_s + min_dwell_s, else return dep_s unchanged.
    No warnings are emitted here.
    """
    if dep_s == arr_s:
        return arr_s + min_dwell_s
    return dep_s


def _sorted_stop_times_for_trip(stop_times: Iterable[Any]) -> list[Any]:
    """
    Sort stop_times of a trip by stop_sequence, then by departure time.

    :param stop_times: StopTime-like objects.
    :return: Sorted list.
    """
    sts = list(stop_times)
    sts.sort(key=lambda st: (_get_stop_sequence(st), _get_departure_seconds(st)))
    return sts


def _trip_capacity(tr: Any) -> float:
    """
    Extract capacity from a Trip-like object.

    :param tr: Trip-like object.
    :return: capacity (float), inf if not defined.
    """
    cap = getattr(tr, "capacity", None)
    return float(cap) if cap is not None else float(np.inf)


def _trip_line_id(tr: Any) -> str:
    """Extract line identifier from a Trip-like object.

    The domain-layer `Trip` uses `line_ref` to reference `Line.line_id`.
    For compatibility, we also accept `line_id` or `line.line_id` if present.

    A line identifier is required to enforce *inter-line only* transfer links.

    :param tr: Trip-like object.
    :return: Line identifier.
    :raises AttributeError: if no line identifier is available.
    :raises ValueError: if a line identifier is present but empty.
    """
    # Preferred (domain API)
    if hasattr(tr, "line_ref"):
        ref = getattr(tr, "line_ref")
        if ref is None:
            raise AttributeError(
                "Trip.line_ref is None; a line identifier is required to build inter-line transfer links."
            )
        ref_str = str(ref).strip()
        if not ref_str:
            raise ValueError(
                "Trip.line_ref is empty; provide a non-empty line_ref to build inter-line transfer links."
            )
        return ref_str

    # Backward/alternate conventions
    if hasattr(tr, "line_id"):
        lid = str(getattr(tr, "line_id")).strip()
        if not lid:
            raise ValueError(
                "Trip.line_id is empty; provide a non-empty line_id to build inter-line transfer links."
            )
        return lid
    if hasattr(tr, "line"):
        line = getattr(tr, "line")
        if hasattr(line, "line_id"):
            lid = str(getattr(line, "line_id")).strip()
            if not lid:
                raise ValueError(
                    "Trip.line.line_id is empty; provide a non-empty line_id to build inter-line transfer links."
                )
            return lid

    raise AttributeError(
        "Trip-like object must provide `line_ref` (preferred), or `line_id`, or `line.line_id` to build inter-line transfer links."
    )




def _iter_time_bins(scenario: "Scenario") -> list[Any]:
    """
    Return the list of time bins from scenario.

    The Scenario is expected to expose time bins (as per your remark).
    We accept a few minimal naming conventions to reduce friction.

    :param scenario: Scenario object.
    :return: list of time bin objects.
    """
    for attr in ("time_bins", "timebin", "timebin_set", "departure_time_bins"):
        if hasattr(scenario, attr):
            tbs = getattr(scenario, attr)
            if tbs is None:
                continue
            return list(tbs)
    raise AttributeError("Scenario must expose time bins (e.g., `scenario.time_bins`).")


def _time_bin_start_s(tb: Any) -> int:
    """Return the start of a time bin in seconds-from-midnight."""
    if hasattr(tb, "start_s"):
        return int(getattr(tb, "start_s"))
    if hasattr(tb, "start") and hasattr(getattr(tb, "start"), "seconds_from_midnight"):
        return int(getattr(getattr(tb, "start"), "seconds_from_midnight"))
    raise AttributeError("TimeBin-like object must provide `start_s` or `start.seconds_from_midnight`.")


def _time_bin_end_s(tb: Any) -> int:
    """Return the end of a time bin in seconds-from-midnight."""
    if hasattr(tb, "end_s"):
        return int(getattr(tb, "end_s"))
    if hasattr(tb, "end") and hasattr(getattr(tb, "end"), "seconds_from_midnight"):
        return int(getattr(getattr(tb, "end"), "seconds_from_midnight"))
    raise AttributeError("TimeBin-like object must provide `end_s` or `end.seconds_from_midnight`.")


# =============================================================================
# Internal node indexing
# =============================================================================


@dataclass(slots=True)
class _NodeIndex:
    """
    Internal node indexing helpers.
    :param centroid_in_index: Mapping (stop_id, time_bin_idx) -> centroid-in node index.
    :param centroid_out_index: Mapping stop_id -> centroid-out node index.
    :param event_arr_index: Mapping (stop_id, arr_s, trip_id) -> ARR event node index.
    :param event_dep_index: Mapping (stop_id, dep_s, trip_id) -> DEP event node index.
    :param event_trip_index: Array mapping event node index -> trip index (for line lookup).
    :param node_time_min: Array of node times (minutes) used to define topological order.
    :param node_time_s: Array of node times (seconds-from-midnight) for event nodes; `CENTROID_TIME_S` for centroids.
    :param node_stop_index: Array mapping each node to its stop index (sorted stop_id convention).
    :param node_time_bin_index: Array mapping each node to its time-bin index for centroid-in nodes; -1 otherwise.
    :param node_bin_start_min: Array giving the start of the departure interval (minutes) for centroid-in nodes; NaN otherwise.
    :param node_bin_end_min: Array giving the end of the departure interval (minutes) for centroid-in nodes; NaN otherwise.
    :param node_kind: Integer codes from `graph_sentinels`:
                      `NODE_KIND_CENTROID_IN`, `NODE_KIND_EVENT_ARR`,
                      `NODE_KIND_EVENT_DEP`, `NODE_KIND_CENTROID_OUT`.
    :param stop_ids: Tuple of sorted stop ids.
    :param time_bins: Tuple of time bins in scenario order (for mapping).
    """
    centroid_in_index: dict[tuple[str, int], int]
    centroid_out_index: dict[str, int]
    event_arr_index: dict[tuple[str, int, str], int]
    event_dep_index: dict[tuple[str, int, str], int]
    event_trip_index: np.ndarray

    node_time_min: np.ndarray
    node_time_s: np.ndarray
    node_stop_index: np.ndarray
    node_time_bin_index: np.ndarray
    node_bin_start_min: np.ndarray
    node_bin_end_min: np.ndarray
    node_kind: np.ndarray

    stop_ids: tuple[str, ...]
    time_bins: tuple[Any, ...]


def _build_nodes(
    scenario: "Scenario",
    *,
    config: AssignmentConfig,
) -> _NodeIndex:
    """
    Build nodes:
      - centroid-in nodes for each (stop, time_bin)
      - ARR/DEP event nodes for each (stop, arrival_s/departure_s, trip)
      - centroid-out nodes for each stop

    :param scenario: Domain scenario.
    :param config: Assignment configuration.
    :return: Node index structure.
    """
    _require(scenario.timetable is not None, "Scenario has no timetable.")
    _require(hasattr(scenario.timetable, "stop_times"), "Scenario timetable has no stop_times.")
    _require(hasattr(scenario.timetable, "trips"), "Scenario timetable has no trips.")
    _require(
        hasattr(config, "min_dwell_s"),
        "AssignmentConfig must define `min_dwell_s` (seconds) to enforce strictly positive dwell.",
    )
    min_dwell_s = int(getattr(config, "min_dwell_s"))
    _require(min_dwell_s > 0, "AssignmentConfig.min_dwell_s must be > 0.")

    # Stops
    if isinstance(scenario.stops, dict):
        stop_ids = [str(k) for k in scenario.stops.keys()]
    else:
        def _stop_id(obj: object) -> str:
            if hasattr(obj, "stop_id"):
                return str(getattr(obj, "stop_id"))
            if hasattr(obj, "id"):
                return str(getattr(obj, "id"))
            raise AttributeError("Stop-like object must provide `stop_id` or `id`.")
        stop_ids = [_stop_id(s) for s in scenario.stops]

    stop_ids_sorted = sorted(stop_ids)
    stop_index = {sid: i for i, sid in enumerate(stop_ids_sorted)}

    # Time bins
    time_bins = _iter_time_bins(scenario)
    _require(len(time_bins) > 0, "Scenario exposes no time bins.")
    time_bins_tuple = tuple(time_bins)


    centroid_in_index: dict[tuple[str, int], int] = {}
    centroid_out_index: dict[str, int] = {}
    event_arr_index: dict[tuple[str, int, str], int] = {}
    event_dep_index: dict[tuple[str, int, str], int] = {}

    node_time_min: list[float] = []
    node_time_s: list[int] = []
    node_stop_index: list[int] = []
    node_kind: list[int] = []
    node_time_bin_index: list[int] = []
    node_bin_start_min: list[float] = []
    node_bin_end_min: list[float] = []

    # Precompute time-bin interval bounds (minutes) for centroid-in nodes.
    bin_start_min_by_t = [ _to_minutes(_time_bin_start_s(tb)) for tb in time_bins_tuple ]
    bin_end_min_by_t = [ _to_minutes(_time_bin_end_s(tb)) for tb in time_bins_tuple ]

    # 1) centroid-in nodes: (stop, time_bin)
    next_node = 0
    for sid in stop_ids_sorted:
        sidx = int(stop_index[sid])
        for t_idx in range(len(time_bins_tuple)):
            centroid_in_index[(sid, t_idx)] = next_node
            next_node += 1
            # Centroid-in nodes are NOT time-tagged; they appear first in the topological order.
            # The time-bin index is used only to decide which departure events are connected by access links
            # and to compute schedule-deviation costs.
            node_time_min.append(CENTROID_IN_TIME_MIN)
            node_time_s.append(CENTROID_TIME_S)
            node_stop_index.append(sidx)
            node_kind.append(NODE_KIND_CENTROID_IN)
            node_time_bin_index.append(int(t_idx))
            node_bin_start_min.append(float(bin_start_min_by_t[int(t_idx)]))
            node_bin_end_min.append(float(bin_end_min_by_t[int(t_idx)]))

    # 2) event nodes: split into ARR and DEP per stop_time record.
    #    This ensures (i) correct semantics (arrive -> possibly transfer -> depart)
    #    and (ii) each event remains trip-specific, hence line-specific.

    # We store node->trip_index for all event nodes (ARR and DEP).
    event_trip_index_list: list[tuple[int, int]] = []  # (node_idx, trip_index)

    trip_ids = [str(getattr(tr, "trip_id")) for tr in scenario.timetable.trips]
    trip_index_by_id = {tid: i for i, tid in enumerate(trip_ids)}
    regularized_equal_times: list[tuple[str, str, int]] = []  # (trip_id, stop_id, arrival_s)

    for st in scenario.timetable.stop_times:
        stop_id = _get_stop_id(st)
        arr_s = _get_arrival_seconds(st)
        dep_s = _get_departure_seconds(st)
        trip_id = _get_trip_id(st)

        _require(stop_id in stop_index, f"Unknown stop_id in stop_times: {stop_id}")
        _require(trip_id in trip_index_by_id, f"Unknown trip_id in stop_times: {trip_id}")

        tr_idx = int(trip_index_by_id[trip_id])

        # Strict dwell policy: if arr==dep in raw input, auto-regularize departure.
        dep_s_reg = _regularize_departure_seconds(arr_s=int(arr_s), dep_s=int(dep_s), min_dwell_s=int(min_dwell_s))
        if dep_s_reg != int(dep_s):
            regularized_equal_times.append((str(trip_id), str(stop_id), int(arr_s)))
        _require(
            int(dep_s_reg) > int(arr_s),
            f"StopTime has departure not after arrival for trip {trip_id} at stop {stop_id}: arr={arr_s}, dep={dep_s_reg}",
        )

        # ARR node
        key_a = (stop_id, int(arr_s), trip_id)
        if key_a not in event_arr_index:
            event_arr_index[key_a] = next_node
            node_idx = next_node
            next_node += 1

            node_time_min.append(_to_minutes(arr_s))
            node_time_s.append(int(arr_s))
            node_stop_index.append(int(stop_index[stop_id]))
            node_kind.append(NODE_KIND_EVENT_ARR)
            node_time_bin_index.append(-1)
            node_bin_start_min.append(float("nan"))
            node_bin_end_min.append(float("nan"))

            event_trip_index_list.append((node_idx, tr_idx))

        # DEP node
        key_d = (stop_id, int(dep_s_reg), trip_id)
        if key_d not in event_dep_index:
            event_dep_index[key_d] = next_node
            node_idx = next_node
            next_node += 1

            node_time_min.append(_to_minutes(dep_s_reg))
            node_time_s.append(int(dep_s_reg))
            node_stop_index.append(int(stop_index[stop_id]))
            node_kind.append(NODE_KIND_EVENT_DEP)
            node_time_bin_index.append(-1)
            node_bin_start_min.append(float("nan"))
            node_bin_end_min.append(float("nan"))

            event_trip_index_list.append((node_idx, tr_idx))

    if regularized_equal_times:
        examples = ", ".join(
            [f"{trip}@{stop}(arr={arr_s}s)" for (trip, stop, arr_s) in regularized_equal_times[:5]]
        )
        more = "" if len(regularized_equal_times) <= 5 else f" (+{len(regularized_equal_times) - 5} more)"
        warnings.warn(
            "Detected stop_times with arrival == departure; auto-regularized departure := arrival + min_dwell_s="
            f"{min_dwell_s}s. Examples: {examples}{more}",
            RuntimeWarning,
            stacklevel=2,
        )

    # 3) centroid-out nodes: one per stop, placed at +infty surrogate
    for sid in stop_ids_sorted:
        centroid_out_index[sid] = next_node
        next_node += 1

        node_time_min.append(CENTROID_OUT_TIME_MIN)
        node_time_s.append(CENTROID_TIME_S)
        node_stop_index.append(int(stop_index[sid]))
        node_kind.append(NODE_KIND_CENTROID_OUT)
        node_time_bin_index.append(-1)
        node_bin_start_min.append(float("nan"))
        node_bin_end_min.append(float("nan"))

    # Build event_trip_index array aligned with node indices:
    # -1 for centroid nodes, trip index for ARR/DEP event nodes.
    num_nodes = next_node
    event_trip_index = -np.ones(num_nodes, dtype=int)
    for node_idx, tr_idx in event_trip_index_list:
        event_trip_index[int(node_idx)] = int(tr_idx)

    return _NodeIndex(
        centroid_in_index=centroid_in_index,
        centroid_out_index=centroid_out_index,
        event_arr_index=event_arr_index,
        event_dep_index=event_dep_index,
        event_trip_index=event_trip_index,
        node_time_min=np.asarray(node_time_min, dtype=float),
        node_time_s=np.asarray(node_time_s, dtype=int),
        node_stop_index=np.asarray(node_stop_index, dtype=int),
        node_time_bin_index=np.asarray(node_time_bin_index, dtype=int),
        node_bin_start_min=np.asarray(node_bin_start_min, dtype=float),
        node_bin_end_min=np.asarray(node_bin_end_min, dtype=float),
        node_kind=np.asarray(node_kind, dtype=int),
        stop_ids=tuple(stop_ids_sorted),
        time_bins=time_bins_tuple,
    )


# =============================================================================
# Links
# =============================================================================


def _build_links(
    scenario: "Scenario",
    nodes: _NodeIndex,
    *,
    config: AssignmentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build link arrays (tail, head, link_type, travel_time_min, capacity, link_trip_index).

    Link type codes:
      LINK_TYPE_RIDE
      LINK_TYPE_TRANSFER
      LINK_TYPE_ACCESS
      LINK_TYPE_EGRESS
      LINK_TYPE_DWELL

    :param scenario: Domain scenario.
    :param nodes: Node index structure.
    :param config: Assignment config (for thresholds).
    :return: (tail, head, link_type, travel_time_min, capacity, link_trip_index)
    """
    _require(scenario.timetable is not None, "Scenario has no timetable.")
    _require(hasattr(scenario.timetable, "stop_times"), "Scenario timetable has no stop_times.")
    _require(hasattr(scenario.timetable, "trips"), "Scenario timetable has no trips.")

    # Thresholds (minutes) controlling graph complexity
    _require(
        hasattr(config, "max_access_deviation_min"),
        "AssignmentConfig must define `max_access_deviation_min` (minutes).",
    )
    _require(
        hasattr(config, "max_transfer_wait_min"),
        "AssignmentConfig must define `max_transfer_wait_min` (minutes).",
    )
    max_access_min = float(getattr(config, "max_access_deviation_min"))
    max_transfer_min = float(getattr(config, "max_transfer_wait_min"))

    _require(
        hasattr(config, "min_dwell_s"),
        "AssignmentConfig must define `min_dwell_s` (seconds) to enforce strictly positive dwell.",
    )
    min_dwell_s = int(getattr(config, "min_dwell_s"))
    _require(min_dwell_s > 0, "AssignmentConfig.min_dwell_s must be > 0.")

    max_access_s = int(round(max_access_min * 60.0))
    max_transfer_s = int(round(max_transfer_min * 60.0))

    # Trip metadata
    trips = list(scenario.timetable.trips)
    trip_ids = [str(getattr(tr, "trip_id")) for tr in trips]
    trip_index_by_id = {tid: i for i, tid in enumerate(trip_ids)}
    line_id_by_trip_index = [ _trip_line_id(tr) for tr in trips ]
    cap_by_trip_id = {str(getattr(tr, "trip_id")): _trip_capacity(tr) for tr in trips}

    # Time-bin bounds in seconds-from-midnight (closed interval semantics)
    start_s_by_t = np.asarray([_time_bin_start_s(tb) for tb in nodes.time_bins], dtype=int)
    end_s_by_t = np.asarray([_time_bin_end_s(tb) for tb in nodes.time_bins], dtype=int)

    tail: list[int] = []
    head: list[int] = []
    link_type: list[int] = []
    travel_time_min: list[float] = []
    capacity: list[float] = []
    link_trip_index: list[int] = []

    # -------------------------------------------------------------------------
    # Group ARR and DEP event nodes by stop for efficiency
    arr_events_by_stop: dict[str, list[tuple[int, int, int, str, str]]] = {}
    dep_events_by_stop: dict[str, list[tuple[int, int, int, str, str]]] = {}
    # items: (time_s, node_idx, trip_index, trip_id, line_id)
    for (stop_id, arr_s, trip_id), ev_node in sorted(nodes.event_arr_index.items()):
        tr_idx = int(trip_index_by_id[trip_id])
        arr_events_by_stop.setdefault(stop_id, []).append(
            (int(arr_s), int(ev_node), tr_idx, trip_id, line_id_by_trip_index[tr_idx])
        )
    for (stop_id, dep_s, trip_id), ev_node in sorted(nodes.event_dep_index.items()):
        tr_idx = int(trip_index_by_id[trip_id])
        dep_events_by_stop.setdefault(stop_id, []).append(
            (int(dep_s), int(ev_node), tr_idx, trip_id, line_id_by_trip_index[tr_idx])
        )

    # -------------------------------------------------------------------------
    # Access links: (s, t)^{in} -> DEP(s, dep_s, trip) within access window
    # Only to SAME origin stop.
    # -------------------------------------------------------------------------
    for stop_id in sorted(dep_events_by_stop.keys()):
        items = dep_events_by_stop[stop_id]
        # Sort events by time for this stop
        items_sorted = sorted(items, key=lambda z: z[0])
        dep_s_arr = np.asarray([it[0] for it in items_sorted], dtype=int)

        for t_idx, (start_s, end_s) in enumerate(zip(start_s_by_t, end_s_by_t)):
            c_in = nodes.centroid_in_index.get((stop_id, int(t_idx)))
            _require(c_in is not None, f"Missing centroid-in node for stop={stop_id}, t={t_idx}")

            # Find events in [start_s - max_access_s, end_s + max_access_s]
            # (i.e., allow boarding within the time-bin, plus a symmetric tolerance).
            lo = int(start_s) - max_access_s
            hi = int(end_s) + max_access_s

            # Use numpy search to restrict range
            a = int(np.searchsorted(dep_s_arr, lo, side="left"))
            b = int(np.searchsorted(dep_s_arr, hi, side="right"))

            for dep_s, dep_node, tr_idx, trip_id, line_id in items_sorted[a:b]:
                # Physical travel time on access link is 0; generalized cost computed later
                tail.append(int(c_in))
                head.append(int(dep_node))
                link_type.append(LINK_TYPE_ACCESS)
                travel_time_min.append(0.0)
                capacity.append(np.inf)
                link_trip_index.append(int(tr_idx))  # keep trip index for possible debugging/line-dependent rules

    # -------------------------------------------------------------------------
    # Egress links: ARR(stop, arr_s, trip) -> centroid-out(stop)
    # Global egress for all stops; destination gating later.
    # -------------------------------------------------------------------------
    for (stop_id, arr_s, trip_id), ev_node in sorted(nodes.event_arr_index.items()):
        c_out = nodes.centroid_out_index.get(stop_id)
        _require(c_out is not None, f"Missing centroid-out node for stop={stop_id}")
        tail.append(int(ev_node))
        head.append(int(c_out))
        link_type.append(LINK_TYPE_EGRESS)
        travel_time_min.append(0.0)
        capacity.append(np.inf)
        link_trip_index.append(-1)

    # -------------------------------------------------------------------------
    # Ride links: consecutive stop_times within each trip, DEP(a) -> ARR(b)
    # -------------------------------------------------------------------------
    st_by_trip: dict[str, list[Any]] = {}
    for st in scenario.timetable.stop_times:
        st_by_trip.setdefault(_get_trip_id(st), []).append(st)

    for trip_id in sorted(st_by_trip.keys()):
        sts = st_by_trip[trip_id]
        sts_sorted = _sorted_stop_times_for_trip(sts)
        tr_idx = int(trip_index_by_id.get(trip_id, -1))
        _require(tr_idx >= 0, f"Trip {trip_id} not found in timetable.trips")

        for a_st, b_st in zip(sts_sorted[:-1], sts_sorted[1:]):
            stop_a = _get_stop_id(a_st)
            stop_b = _get_stop_id(b_st)
            arr_a = _get_arrival_seconds(a_st)
            dep_a_raw = _get_departure_seconds(a_st)
            dep_a = _regularize_departure_seconds(arr_s=int(arr_a), dep_s=int(dep_a_raw), min_dwell_s=int(min_dwell_s))
            arr_b = _get_arrival_seconds(b_st)

            _require(int(arr_b) > int(dep_a), f"Non-increasing ride times in trip {trip_id}: {stop_a}@{dep_a} -> {stop_b}@{arr_b}")

            na = nodes.event_dep_index[(stop_a, int(dep_a), trip_id)]
            nb = nodes.event_arr_index[(stop_b, int(arr_b), trip_id)]

            tail.append(int(na))
            head.append(int(nb))
            link_type.append(LINK_TYPE_RIDE)
            travel_time_min.append(_to_minutes(int(arr_b) - int(dep_a)))
            capacity.append(float(cap_by_trip_id.get(trip_id, np.inf)))
            link_trip_index.append(tr_idx)

    # -------------------------------------------------------------------------
    # Dwell/continue links: ARR(stop, arr_s, trip) -> DEP(stop, dep_s, trip)
    # This represents staying on the same vehicle (or dwell) at a stop.
    # -------------------------------------------------------------------------
    stop_times_sorted_for_dwell = sorted(
        scenario.timetable.stop_times,
        key=lambda st: (
            _get_trip_id(st),
            _get_stop_id(st),
            _get_arrival_seconds(st),
            _get_departure_seconds(st),
        ),
    )
    for st in stop_times_sorted_for_dwell:
        stop_id = _get_stop_id(st)
        arr_s = _get_arrival_seconds(st)
        dep_s = _get_departure_seconds(st)
        trip_id = _get_trip_id(st)

        dep_s_reg = _regularize_departure_seconds(arr_s=int(arr_s), dep_s=int(dep_s), min_dwell_s=int(min_dwell_s))
        _require(int(dep_s_reg) > int(arr_s), f"StopTime dep not after arr for trip {trip_id} at stop {stop_id}")

        n_arr = nodes.event_arr_index[(stop_id, int(arr_s), trip_id)]
        n_dep = nodes.event_dep_index[(stop_id, int(dep_s_reg), trip_id)]

        tail.append(int(n_arr))
        head.append(int(n_dep))
        link_type.append(LINK_TYPE_DWELL)
        travel_time_min.append(_to_minutes(int(dep_s_reg) - int(arr_s)))
        capacity.append(np.inf)
        link_trip_index.append(int(trip_index_by_id[trip_id]))

    # -------------------------------------------------------------------------
    # Transfer links: inter-line, bounded waiting
    # For each stop, consider all pairs ARR->DEP within max_transfer_s where line differs.
    # -------------------------------------------------------------------------
    common_stops = sorted(set(arr_events_by_stop.keys()).intersection(dep_events_by_stop.keys()))
    for stop_id in common_stops:
        arr_items = sorted(arr_events_by_stop[stop_id], key=lambda z: z[0])
        dep_items = sorted(dep_events_by_stop[stop_id], key=lambda z: z[0])
        dep_times = np.asarray([it[0] for it in dep_items], dtype=int)
        n_arr = len(arr_items)
        n_dep = len(dep_items)
        if n_arr == 0 or n_dep == 0:
            continue
        for i, (t_arr, n_arr_idx, tr_arr, trip_id_arr, line_arr) in enumerate(arr_items):
            # Find all dep events with dep in (t_arr, t_arr + max_transfer_s]
            a = int(np.searchsorted(dep_times, t_arr, side="right"))
            b = int(np.searchsorted(dep_times, t_arr + max_transfer_s, side="right"))
            for j in range(a, b):
                dep_time, n_dep_idx, tr_dep, trip_id_dep, line_dep = dep_items[j]
                if line_dep == line_arr:
                    continue
                wait_s = int(dep_time - t_arr)
                if wait_s <= 0:
                    continue
                tail.append(int(n_arr_idx))
                head.append(int(n_dep_idx))
                link_type.append(LINK_TYPE_TRANSFER)
                travel_time_min.append(_to_minutes(wait_s))
                capacity.append(np.inf)
                link_trip_index.append(-1)

    return (
        np.asarray(tail, dtype=int),
        np.asarray(head, dtype=int),
        np.asarray(link_type, dtype=int),
        np.asarray(travel_time_min, dtype=float),
        np.asarray(capacity, dtype=float),
        np.asarray(link_trip_index, dtype=int),
    )


# =============================================================================
# Adjacency builders
# =============================================================================


def _build_csr_outgoing(num_nodes: int, tail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build CSR-like outgoing adjacency from tail indices.

    :param num_nodes: Number of nodes.
    :param tail: Tail node index for each link, shape (num_links,).
    :return: (out_start, out_links) arrays.
    """
    num_links = int(tail.shape[0])
    order = np.argsort(tail, kind="mergesort")
    out_links = order.astype(int)

    deg = np.zeros(num_nodes, dtype=int)
    np.add.at(deg, tail, 1)

    out_start = np.zeros(num_nodes + 1, dtype=int)
    out_start[1:] = np.cumsum(deg)

    _require(int(out_start[-1]) == num_links, "CSR construction error: out_start[-1] != num_links.")
    return out_start, out_links


def _build_padded_outgoing(
    *,
    num_nodes: int,
    out_start: np.ndarray,
    out_links_csr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build padded outgoing adjacency arrays from CSR.

    :param num_nodes: Number of nodes.
    :param out_start: CSR row pointers, shape (num_nodes+1,).
    :param out_links_csr: CSR link indices, shape (num_links,).
    :return: (out_links, out_mask) padded arrays.
    """
    deg = (out_start[1:] - out_start[:-1]).astype(int)
    max_out = int(deg.max()) if num_nodes > 0 else 0

    out_links = -np.ones((num_nodes, max_out), dtype=int)
    out_mask = np.zeros((num_nodes, max_out), dtype=bool)

    for i in range(num_nodes):
        a = int(out_start[i])
        b = int(out_start[i + 1])
        k = b - a
        if k <= 0:
            continue
        out_links[i, :k] = out_links_csr[a:b]
        out_mask[i, :k] = True

    return out_links, out_mask


# =============================================================================
# Public builder
# =============================================================================


def build_jax_graph(
    scenario: "Scenario",
    *,
    config: AssignmentConfig,
    profile: dict[str, float] | None = None,
) -> JaxGraph:
    """
    Build a static, JAX-compatible time-expanded graph from a validated Scenario.

    :param scenario: Domain Scenario containing stops, timetable, and time bins.
    :param config: Assignment configuration.
    :return: JaxGraph with immutable arrays.
    """
    from time import perf_counter

    def record(name: str, started: float) -> None:
        if profile is not None:
            profile[name] = perf_counter() - started

    started = perf_counter()
    config.validate()
    record("graph_configuration_validation", started)

    started = perf_counter()
    nodes = _build_nodes(scenario, config=config)
    record("stop_trip_line_timetable_time_bin_indexing_and_nodes", started)
    started = perf_counter()
    tail, head, link_type, travel_time_min, capacity, link_trip_index = _build_links(
        scenario, nodes, config=config
    )
    record("link_creation_and_concatenation", started)

    num_nodes = int(nodes.node_time_min.shape[0])
    num_links = int(tail.shape[0])

    _require(num_nodes > 0, "Built graph has no nodes.")
    _require(num_links > 0, "Built graph has no links. Check inputs and thresholds.")

    # Topological order (DAG order):
    # The dynamic programming requires that for every link (i -> j), i appears before j.
    # With ARR/DEP event splitting we have links such as:
    #   centroid-in(-infty) -> DEP,  DEP -> ARR (ride),  ARR -> DEP (dwell/continue),  ARR -> centroid-out(+infty) (egress)
    # Centroid-in nodes are placed at -infty (not time-tagged), so ordering remains consistent even
    # when boarding before the desired departure time.
    #
    # Phase / tie-break order (node_kind):
    #   NODE_KIND_CENTROID_IN  <  NODE_KIND_EVENT_ARR  <  NODE_KIND_EVENT_DEP  <  NODE_KIND_CENTROID_OUT
    #
    # We enforce this robustly by lexicographic sort on (time, phase).
    started = perf_counter()
    kind = nodes.node_kind.astype(int)
    time = nodes.node_time_min.astype(float)
    topo_order = np.lexsort((kind, time)).astype(int)  # primary: time, secondary: kind
    topo_order_rev = topo_order[::-1].copy()
    record("link_and_node_sorting_canonicalization", started)

    # CSR outgoing adjacency
    started = perf_counter()
    out_start, out_links_csr = _build_csr_outgoing(num_nodes, tail)
    record("csr_adjacency_construction", started)

    # Padded outgoing adjacency
    started = perf_counter()
    out_links_pad, out_mask = _build_padded_outgoing(
        num_nodes=num_nodes,
        out_start=out_start,
        out_links_csr=out_links_csr,
    )
    record("mask_and_padded_array_construction", started)

    # Python-side metadata
    started = perf_counter()
    trips = list(scenario.timetable.trips)
    trip_ids = tuple([str(getattr(tr, "trip_id")) for tr in trips])
    trip_line_ref = tuple([_trip_line_id(tr) for tr in trips])
    stop_ids_sorted = nodes.stop_ids
    record("graph_metadata_and_diagnostic_construction", started)

    started = perf_counter()
    graph = JaxGraph(
        num_nodes=num_nodes,
        num_links=num_links,
        tail=jnp.asarray(tail),
        head=jnp.asarray(head),
        topo_order=jnp.asarray(topo_order),
        topo_order_rev=jnp.asarray(topo_order_rev),
        node_time=jnp.asarray(nodes.node_time_min),
        node_stop_index=jnp.asarray(nodes.node_stop_index),
        node_time_bin_index=jnp.asarray(nodes.node_time_bin_index),
        node_bin_start_min=jnp.asarray(nodes.node_bin_start_min),
        node_bin_end_min=jnp.asarray(nodes.node_bin_end_min),
        node_time_s=jnp.asarray(nodes.node_time_s),
        node_kind=jnp.asarray(nodes.node_kind),
        node_trip_index=jnp.asarray(nodes.event_trip_index),
        out_start=jnp.asarray(out_start),
        out_links_csr=jnp.asarray(out_links_csr),
        out_links=jnp.asarray(out_links_pad),
        out_mask=jnp.asarray(out_mask),
        link_type=jnp.asarray(link_type),
        travel_time=jnp.asarray(travel_time_min),
        capacity=jnp.asarray(capacity),
        link_trip_index=jnp.asarray(link_trip_index),
        node_stop_id=stop_ids_sorted,
        trip_id=trip_ids,
        trip_line_ref=trip_line_ref,
    )
    jax.block_until_ready(graph)
    record("graph_numpy_to_jax_device_transfer_and_synchronization", started)
    return graph
