"""On-demand itinerary queries over the assignment model's route-choice state.

This module enumerates individual paths only for a selected OD/time query. It
does not persist path lists; dense and sharded routing states are both read a
destination group at a time.
"""

from __future__ import annotations

import heapq
import math
from time import monotonic
from typing import Callable, Sequence

import numpy as np

from public_transportation.inference.assignment_contract import (
    CanonicalTimeInterval,
    fixed_routing_route_choice_fingerprint,
)
from public_transportation.assignment.graph_sentinels import (
    LINK_TYPE_DWELL,
    LINK_TYPE_RIDE,
    NODE_KIND_CENTROID_IN,
    NODE_KIND_CENTROID_OUT,
)
from public_transportation.inference.assignment_adapter import FixedRoutingInputs
from public_transportation.inference.sharded_fixed_routing import (
    ShardedFixedRoutingInputs,
    fixed_routing_descriptor_for_group,
    load_fixed_routing_shard,
)


class RoutePathQueryCancelled(RuntimeError):
    """Raised when the viewer cancels a route-path query."""


class RouteChoicePathQuery:
    """Enumerate paths induced by persisted link-choice probabilities.

    The route-choice source is tied to the fitted mapping identity. The
    returned probability is conditional on the selected origin, destination,
    and departure interval, and is the product of the model's per-link
    transition probabilities along the complete time-expanded path.
    """

    def __init__(
        self,
        *,
        graph: object,
        group_dest_node: object,
        time_intervals: Sequence[CanonicalTimeInterval],
        mapping_artifact_fingerprint: str,
        fit_fingerprint: str,
        route_choice_fingerprint: str,
        probability_for_group: Callable[[int], tuple[object, object]],
        number_of_groups: int,
        max_paths: int = 200,
        max_expansions: int = 100_000,
        max_seconds: float = 5.0,
    ) -> None:
        if (
            not mapping_artifact_fingerprint
            or not fit_fingerprint
            or not route_choice_fingerprint
        ):
            raise ValueError("route-path query identities must not be empty.")
        if (
            max_paths <= 0
            or max_expansions <= 0
            or not math.isfinite(max_seconds)
            or max_seconds <= 0
        ):
            raise ValueError("route-path query bounds must be positive and finite.")
        self.graph = graph
        self.mapping_artifact_fingerprint = str(mapping_artifact_fingerprint)
        self.fit_fingerprint = str(fit_fingerprint)
        self.route_choice_fingerprint = str(route_choice_fingerprint)
        self._probability_for_group = probability_for_group
        self.max_paths = int(max_paths)
        self.max_expansions = int(max_expansions)
        self.max_seconds = float(max_seconds)

        self._tail = np.asarray(graph.tail, dtype=np.int64)
        self._head = np.asarray(graph.head, dtype=np.int64)
        self._node_kind = np.asarray(graph.node_kind, dtype=np.int64)
        self._node_stop_index = np.asarray(graph.node_stop_index, dtype=np.int64)
        self._node_time_bin_index = np.asarray(
            graph.node_time_bin_index, dtype=np.int64
        )
        self._node_time_s = np.asarray(graph.node_time_s, dtype=np.int64)
        self._node_trip_index = np.asarray(graph.node_trip_index, dtype=np.int64)
        self._link_type = np.asarray(graph.link_type, dtype=np.int64)
        self._link_trip_index = np.asarray(graph.link_trip_index, dtype=np.int64)
        self._topological_order = np.asarray(graph.topo_order, dtype=np.int64)
        self._stop_ids = tuple(str(item) for item in graph.node_stop_id)
        self._stop_names = tuple(
            str(item) for item in getattr(graph, "node_stop_name", ())
        )
        self._trip_ids = tuple(str(item) for item in getattr(graph, "trip_id", ()))
        self._trip_lines = tuple(
            str(item) for item in getattr(graph, "trip_line_ref", ())
        )
        self._group_dest_node = np.asarray(group_dest_node, dtype=np.int64)
        self._number_of_groups = int(number_of_groups)
        self.time_intervals = tuple(time_intervals)
        self._interval_index = {
            interval.interval_id: index
            for index, interval in enumerate(self.time_intervals)
        }
        self._out_start = np.asarray(graph.out_start, dtype=np.int64)
        self._out_links = np.asarray(graph.out_links_csr, dtype=np.int64)
        self._validate_graph()

        stop_indices = {stop_id: index for index, stop_id in enumerate(self._stop_ids)}
        if len(stop_indices) != len(self._stop_ids):
            raise ValueError("route-choice graph contains duplicate stop identifiers.")
        self._stop_index_by_id = stop_indices
        origins: dict[tuple[int, int], int] = {}
        for node in np.flatnonzero(self._node_kind == NODE_KIND_CENTROID_IN):
            key = (
                int(self._node_stop_index[node]),
                int(self._node_time_bin_index[node]),
            )
            if key in origins:
                raise ValueError(
                    "route-choice graph has duplicate OD-origin/time-bin nodes."
                )
            origins[key] = int(node)
        self._origin_node_by_stop_and_bin = origins

        destinations: dict[str, int] = {}
        for group, node in enumerate(self._group_dest_node):
            if node < 0 or node >= self._node_kind.size:
                raise ValueError(
                    "route-choice destination group references an invalid node."
                )
            if self._node_kind[node] != NODE_KIND_CENTROID_OUT:
                raise ValueError(
                    "route-choice destination group must end at a destination centroid."
                )
            stop_index = int(self._node_stop_index[node])
            if stop_index < 0 or stop_index >= len(self._stop_ids):
                raise ValueError(
                    "route-choice destination group has an invalid stop index."
                )
            stop_id = self._stop_ids[stop_index]
            if stop_id in destinations:
                raise ValueError(
                    "route-choice source contains duplicate destination groups for a stop."
                )
            destinations[stop_id] = group
        self._group_by_destination = destinations

    @classmethod
    def from_fixed_routing(
        cls,
        *,
        routing: FixedRoutingInputs | ShardedFixedRoutingInputs,
        time_intervals: Sequence[CanonicalTimeInterval],
        mapping_artifact_fingerprint: str,
        fit_fingerprint: str,
        **bounds: object,
    ) -> "RouteChoicePathQuery":
        """Create a bounded query over an already prepared dense or sharded cache."""
        if isinstance(routing, ShardedFixedRoutingInputs):
            group_dest_node = np.asarray(
                routing.destination_group_identifiers, dtype=np.int64
            )

            def probability_for_group(group: int) -> tuple[np.ndarray, np.ndarray]:
                descriptor = fixed_routing_descriptor_for_group(routing, group)
                shard = load_fixed_routing_shard(routing=routing, descriptor=descriptor)
                local = group - descriptor.group_start
                return shard.group_link_probability[
                    local
                ], shard.effective_group_link_mask[local]

            group_count = routing.num_destination_groups
            graph = routing.graph
            theta = float(routing.theta)
        else:
            group_dest_node = np.asarray(routing.group_dest_node, dtype=np.int64)

            def probability_for_group(group: int) -> tuple[np.ndarray, np.ndarray]:
                return (
                    np.asarray(routing.group_link_probability[group]),
                    np.asarray(routing.effective_group_link_mask[group]),
                )

            group_count = group_dest_node.size
            graph = routing.graph
            theta = float(np.asarray(routing.theta))
        expected_route_fingerprint = fixed_routing_route_choice_fingerprint(theta)
        source_route_fingerprint = getattr(routing, "route_choice_fingerprint", None)
        if (
            source_route_fingerprint is not None
            and source_route_fingerprint != expected_route_fingerprint
        ):
            raise ValueError("route-choice cache theta fingerprint is inconsistent.")
        result = cls(
            graph=graph,
            group_dest_node=group_dest_node,
            time_intervals=time_intervals,
            mapping_artifact_fingerprint=mapping_artifact_fingerprint,
            fit_fingerprint=fit_fingerprint,
            route_choice_fingerprint=expected_route_fingerprint,
            probability_for_group=probability_for_group,
            number_of_groups=group_count,
            **bounds,
        )
        return result

    def validate_identity(
        self,
        *,
        mapping_artifact_fingerprint: str,
        fit_fingerprint: str,
        route_choice_fingerprint: str,
        time_intervals: Sequence[CanonicalTimeInterval],
    ) -> None:
        if self.mapping_artifact_fingerprint != mapping_artifact_fingerprint:
            raise ValueError(
                "route-choice cache mapping identity differs from the selected viewer bundle."
            )
        if self.fit_fingerprint != fit_fingerprint:
            raise ValueError(
                "route-choice cache fit identity differs from the selected viewer bundle."
            )
        if self.route_choice_fingerprint != route_choice_fingerprint:
            raise ValueError(
                "route-choice cache fingerprint differs from the selected mapping artifact."
            )
        if tuple(self.time_intervals) != tuple(time_intervals):
            raise ValueError(
                "route-choice cache departure intervals differ from the selected mapping artifact."
            )

    def _validate_graph(self) -> None:
        node_count = self._node_kind.size
        link_count = self._tail.size
        if (
            self._head.shape != (link_count,)
            or self._link_type.shape != (link_count,)
            or self._link_trip_index.shape != (link_count,)
        ):
            raise ValueError("route-choice graph link arrays have invalid dimensions.")
        if any(
            value.shape != (node_count,)
            for value in (
                self._node_stop_index,
                self._node_time_bin_index,
                self._node_time_s,
                self._node_trip_index,
            )
        ):
            raise ValueError("route-choice graph node arrays have invalid dimensions.")
        if self._group_dest_node.shape != (self._number_of_groups,):
            raise ValueError("route-choice destination-group dimensions are invalid.")
        if (
            self._out_start.shape != (node_count + 1,)
            or self._out_start[0] != 0
            or self._out_start[-1] != link_count
            or np.any(np.diff(self._out_start) < 0)
        ):
            raise ValueError("route-choice graph outgoing-link index is invalid.")
        if self._out_links.shape != (link_count,) or not np.array_equal(
            np.sort(self._out_links), np.arange(link_count)
        ):
            raise ValueError("route-choice graph outgoing-link order is invalid.")
        if (
            np.any(self._tail < 0)
            or np.any(self._tail >= node_count)
            or np.any(self._head < 0)
            or np.any(self._head >= node_count)
        ):
            raise ValueError("route-choice graph contains out-of-range links.")
        if (
            len(self._stop_ids) == 0
            or np.any(self._node_stop_index < 0)
            or np.any(self._node_stop_index >= len(self._stop_ids))
        ):
            raise ValueError("route-choice graph stop references are invalid.")
        if self._stop_names and len(self._stop_names) != len(self._stop_ids):
            raise ValueError("route-choice stop labels are misaligned.")
        if self._trip_lines and len(self._trip_lines) != len(self._trip_ids):
            raise ValueError("route-choice trip labels are misaligned.")
        if np.any(self._link_trip_index >= len(self._trip_ids)):
            raise ValueError("route-choice graph link references an invalid trip.")
        topological_rank = np.full(node_count, -1, dtype=np.int64)
        if (
            self._topological_order.shape != (node_count,)
            or np.any(self._topological_order < 0)
            or np.any(self._topological_order >= node_count)
            or np.unique(self._topological_order).size != node_count
        ):
            raise ValueError("route-choice graph topological order is invalid.")
        topological_rank[self._topological_order] = np.arange(node_count)
        if np.any(topological_rank[self._tail] >= topological_rank[self._head]):
            raise ValueError(
                "route-choice graph is not acyclic in its saved topological order."
            )
        if len(self._interval_index) != len(self.time_intervals):
            raise ValueError(
                "route-choice departure intervals contain duplicate identifiers."
            )

    def query(
        self,
        *,
        origin_stop_id: str,
        destination_stop_id: str,
        departure_interval_id: str,
        minimum_probability: float = 0.01,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        probability_threshold = float(minimum_probability)
        if (
            not math.isfinite(probability_threshold)
            or not 0.0 < probability_threshold <= 1.0
        ):
            raise ValueError(
                "minimum_probability must be greater than 0 and at most 1."
            )
        if origin_stop_id == destination_stop_id:
            raise ValueError("origin and destination must be different stops.")
        try:
            origin_stop_index = self._stop_index_by_id[origin_stop_id]
            departure_bin = self._interval_index[departure_interval_id]
            destination_group = self._group_by_destination[destination_stop_id]
        except KeyError as error:
            raise ValueError(
                f"route-choice source does not contain the selected stop or interval: {error.args[0]!r}."
            ) from error
        try:
            origin_node = self._origin_node_by_stop_and_bin[
                (origin_stop_index, departure_bin)
            ]
        except KeyError as error:
            raise ValueError(
                "no modeled origin node exists for the selected departure interval."
            ) from error
        destination_node = int(self._group_dest_node[destination_group])
        link_probability_raw, enabled_raw = self._probability_for_group(
            destination_group
        )
        link_probability = np.asarray(link_probability_raw, dtype=np.float64)
        enabled = np.asarray(enabled_raw, dtype=bool)
        if (
            link_probability.shape != self._tail.shape
            or enabled.shape != self._tail.shape
        ):
            raise ValueError(
                "selected destination routing probabilities have invalid dimensions."
            )
        if (
            not np.all(np.isfinite(link_probability))
            or np.any(link_probability < 0.0)
            or np.any(link_probability > 1.0)
        ):
            raise ValueError("selected destination routing probabilities are invalid.")
        if np.any((~enabled) & (link_probability != 0.0)):
            raise ValueError("disabled route-choice links have nonzero probabilities.")

        threshold_log = math.log(probability_threshold)
        # A partial path's probability is an upper bound on every completion,
        # because all remaining transition probabilities are at most one.
        # Best-first expansion therefore returns the highest-probability
        # complete paths first, even when the query hits its path-count bound.
        pending: list[tuple[float, int, int, int]] = []
        # Store path prefixes as parent pointers, not copied tuples, so a long
        # or highly branching DAG cannot multiply memory by prefix length.
        path_parent = [-1]
        path_link = [-1]
        serial = 0
        heapq.heappush(pending, (0.0, serial, origin_node, 0))
        found: list[tuple[float, tuple[int, ...]]] = []
        expansions = 0
        truncated = False
        deadline = monotonic() + self.max_seconds
        while pending:
            if cancel_check is not None and cancel_check():
                raise RoutePathQueryCancelled(
                    "Path query cancelled because the selection changed."
                )
            if monotonic() >= deadline or expansions >= self.max_expansions:
                truncated = True
                break
            negative_log_probability, _serial, node, path_state = heapq.heappop(pending)
            log_probability = -negative_log_probability
            if node == destination_node:
                links: list[int] = []
                current_state = path_state
                while current_state > 0:
                    links.append(path_link[current_state])
                    current_state = path_parent[current_state]
                found.append((log_probability, tuple(reversed(links))))
                if len(found) > self.max_paths:
                    found.pop()
                    truncated = True
                    break
                continue
            first = int(self._out_start[node])
            last = int(self._out_start[node + 1])
            for link in self._out_links[first:last]:
                link = int(link)
                if not enabled[link] or link_probability[link] <= 0.0:
                    continue
                extended_log_probability = log_probability + math.log(
                    float(link_probability[link])
                )
                if extended_log_probability < threshold_log:
                    continue
                expansions += 1
                if expansions >= self.max_expansions:
                    truncated = True
                    break
                serial += 1
                path_parent.append(path_state)
                path_link.append(link)
                heapq.heappush(
                    pending,
                    (
                        -extended_log_probability,
                        serial,
                        int(self._head[link]),
                        len(path_parent) - 1,
                    ),
                )
        found.sort(key=lambda item: item[0], reverse=True)

        total_probability = self._total_reach_probability(
            origin_node,
            destination_node,
            link_probability,
            enabled,
            cancel_check=cancel_check,
            deadline=deadline,
        )
        if total_probability is None:
            truncated = True
        paths = [
            self._format_path(
                origin_stop_id, destination_stop_id, math.exp(logp), links
            )
            for logp, links in found
        ]
        listed_mass = float(sum(float(path["probability"]) for path in paths))
        return {
            "origin_stop_id": origin_stop_id,
            "destination_stop_id": destination_stop_id,
            "departure_interval_id": departure_interval_id,
            "minimum_probability": probability_threshold,
            "paths": paths,
            "total_path_probability": total_probability,
            "listed_probability_mass": listed_mass,
            "unlisted_probability_mass": (
                max(0.0, total_probability - listed_mass)
                if total_probability is not None
                else None
            ),
            "truncated": truncated,
            "expansions": expansions,
            "fit_fingerprint": self.fit_fingerprint,
            "mapping_artifact_fingerprint": self.mapping_artifact_fingerprint,
            "route_choice_fingerprint": self.route_choice_fingerprint,
        }

    def _total_reach_probability(
        self,
        origin_node: int,
        destination_node: int,
        probabilities: np.ndarray,
        enabled: np.ndarray,
        *,
        cancel_check: Callable[[], bool] | None,
        deadline: float,
    ) -> float | None:
        mass = np.zeros(self._node_kind.size, dtype=np.float64)
        mass[origin_node] = 1.0
        for node in self._topological_order:
            if cancel_check is not None and cancel_check():
                raise RoutePathQueryCancelled(
                    "Path query cancelled because the selection changed."
                )
            if monotonic() >= deadline:
                return None
            source_mass = mass[int(node)]
            if source_mass == 0.0:
                continue
            for link in self._out_links[
                self._out_start[node] : self._out_start[node + 1]
            ]:
                link = int(link)
                if enabled[link] and probabilities[link] > 0.0:
                    mass[self._head[link]] += source_mass * probabilities[link]
        return float(mass[destination_node])

    def _format_path(
        self,
        origin_stop_id: str,
        destination_stop_id: str,
        probability: float,
        links: tuple[int, ...],
    ) -> dict[str, object]:
        legs: list[dict[str, object]] = []
        current: dict[str, object] | None = None
        stop_sequence = [origin_stop_id]
        transfers = 0
        previous_trip: int | None = None
        for link in links:
            tail_node = int(self._tail[link])
            head_node = int(self._head[link])
            tail_stop_index = int(self._node_stop_index[tail_node])
            head_stop_index = int(self._node_stop_index[head_node])
            tail_stop = self._stop_ids[tail_stop_index]
            head_stop = self._stop_ids[head_stop_index]
            if stop_sequence[-1] != head_stop:
                stop_sequence.append(head_stop)
            if self._link_type[link] == LINK_TYPE_RIDE:
                trip_index = int(self._link_trip_index[link])
                if current is None or int(current["trip_index"]) != trip_index:
                    if current is not None:
                        legs.append(current)
                    if previous_trip is not None and previous_trip != trip_index:
                        transfers += 1
                    trip_id = (
                        self._trip_ids[trip_index]
                        if 0 <= trip_index < len(self._trip_ids)
                        else ""
                    )
                    line_id = (
                        self._trip_lines[trip_index]
                        if 0 <= trip_index < len(self._trip_lines)
                        else ""
                    )
                    current = {
                        "trip_index": trip_index,
                        "trip_id": trip_id,
                        "line_id": line_id,
                        "from_stop_id": tail_stop,
                        "to_stop_id": head_stop,
                        "departure_seconds": int(self._node_time_s[tail_node]),
                        "arrival_seconds": int(self._node_time_s[head_node]),
                        "stops": [tail_stop, head_stop],
                    }
                else:
                    current["to_stop_id"] = head_stop
                    current["arrival_seconds"] = int(self._node_time_s[head_node])
                    current_stops = current["stops"]
                    assert isinstance(current_stops, list)
                    if current_stops[-1] != head_stop:
                        current_stops.append(head_stop)
                previous_trip = trip_index
            elif self._link_type[link] not in (LINK_TYPE_DWELL,):
                if current is not None:
                    legs.append(current)
                    current = None
        if current is not None:
            legs.append(current)
        for leg in legs:
            leg.pop("trip_index", None)
            leg["departure_time"] = _format_service_time(
                int(leg.pop("departure_seconds"))
            )
            leg["arrival_time"] = _format_service_time(int(leg.pop("arrival_seconds")))
        if stop_sequence[-1] != destination_stop_id:
            stop_sequence.append(destination_stop_id)
        return {
            "probability": float(probability),
            "legs": legs,
            "stops": stop_sequence,
            "transfers": transfers,
        }


def _format_service_time(seconds: int) -> str:
    if seconds < 0:
        return ""
    hours, remainder = divmod(seconds, 3_600)
    minutes, _seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}"
