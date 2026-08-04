"""Contracts, instrumentation, and microshard planning for partial routing.

This module deliberately does not execute routing work.  It establishes the
stable, serializable foundation used by the future persistent-worker backend
without changing the current exact numerical path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from threading import get_ident
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from .sharded_matrix_free_operator import ShardedOperatorProgress


PARALLEL_PARTIAL_EXECUTION_SCHEMA_VERSION = 1


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class RoutingWorkUnit:
    """Smallest independently schedulable additive routing contribution."""

    work_id: str
    destination_group_indices: tuple[int, ...]
    predicted_cost: float
    routing_bytes: int
    active_od_cells: int
    support_entries: int
    measurement_support: int
    stratum: str = "default"

    def __post_init__(self) -> None:
        if not self.work_id:
            raise ValueError("work_id must not be empty.")
        if not self.destination_group_indices:
            raise ValueError("a routing work unit must contain at least one group.")
        if len(set(self.destination_group_indices)) != len(
            self.destination_group_indices
        ):
            raise ValueError("destination groups must not be duplicated within a unit.")
        if any(value < 0 for value in self.destination_group_indices):
            raise ValueError("destination-group indices must be nonnegative.")
        if not np.isfinite(self.predicted_cost) or self.predicted_cost <= 0.0:
            raise ValueError("predicted_cost must be positive and finite.")
        for name in (
            "routing_bytes",
            "active_od_cells",
            "support_entries",
            "measurement_support",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if not self.stratum:
            raise ValueError("stratum must not be empty.")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["destination_group_indices"] = list(self.destination_group_indices)
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RoutingWorkUnit:
        values = dict(payload)
        values["destination_group_indices"] = tuple(
            int(value) for value in values["destination_group_indices"]  # type: ignore[union-attr]
        )
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RoutingMicroshardPlan:
    """Complete deterministic partition of source routing work."""

    problem_fingerprint: str
    microshards: tuple[RoutingWorkUnit, ...]
    source_work_ids: tuple[str, ...]
    plan_fingerprint: str
    schema_version: int = PARALLEL_PARTIAL_EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PARALLEL_PARTIAL_EXECUTION_SCHEMA_VERSION:
            raise ValueError("unsupported parallel partial-execution schema version.")
        if not self.problem_fingerprint or not self.plan_fingerprint:
            raise ValueError("plan fingerprints must not be empty.")
        ids = tuple(item.work_id for item in self.microshards)
        if len(ids) != len(set(ids)):
            raise ValueError("microshard identities must be unique.")
        if len(self.source_work_ids) != len(set(self.source_work_ids)):
            raise ValueError("source work identities must be unique.")
        groups = [
            group
            for microshard in self.microshards
            for group in microshard.destination_group_indices
        ]
        if len(groups) != len(set(groups)):
            raise ValueError("microshards must not duplicate destination groups.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "problem_fingerprint": self.problem_fingerprint,
            "microshards": [item.to_dict() for item in self.microshards],
            "source_work_ids": list(self.source_work_ids),
            "plan_fingerprint": self.plan_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RoutingMicroshardPlan:
        return cls(
            schema_version=int(payload["schema_version"]),
            problem_fingerprint=str(payload["problem_fingerprint"]),
            microshards=tuple(
                RoutingWorkUnit.from_dict(item)
                for item in payload["microshards"]  # type: ignore[union-attr]
            ),
            source_work_ids=tuple(
                str(item) for item in payload["source_work_ids"]  # type: ignore[union-attr]
            ),
            plan_fingerprint=str(payload["plan_fingerprint"]),
        )


@dataclass(frozen=True, slots=True)
class PartialExecutionBatch:
    batch_id: str
    work_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.batch_id or not self.work_ids:
            raise ValueError("an execution batch requires an identity and work.")
        if len(self.work_ids) != len(set(self.work_ids)):
            raise ValueError("a batch must not duplicate work identities.")


@dataclass(frozen=True, slots=True)
class PartialExecutionPlan:
    """Serializable selected-work contract for matched forward and reverse use."""

    problem_fingerprint: str
    microshard_plan_fingerprint: str
    requested_effort_percent: float
    realized_effort_percent: float
    selected_work_ids: tuple[str, ...]
    batches: tuple[PartialExecutionBatch, ...]
    selection_seed: int
    execution_fingerprint: str
    schema_version: int = PARALLEL_PARTIAL_EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PARALLEL_PARTIAL_EXECUTION_SCHEMA_VERSION:
            raise ValueError("unsupported parallel partial-execution schema version.")
        for name in ("requested_effort_percent", "realized_effort_percent"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 < value <= 100.0:
                raise ValueError(f"{name} must lie in (0, 100].")
        selected = set(self.selected_work_ids)
        flattened = [work for batch in self.batches for work in batch.work_ids]
        if len(self.selected_work_ids) != len(selected):
            raise ValueError("selected work identities must be unique.")
        if len(flattened) != len(set(flattened)) or set(flattened) != selected:
            raise ValueError("execution batches must partition selected work exactly.")
        if not self.execution_fingerprint:
            raise ValueError("execution_fingerprint must not be empty.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "problem_fingerprint": self.problem_fingerprint,
            "microshard_plan_fingerprint": self.microshard_plan_fingerprint,
            "requested_effort_percent": self.requested_effort_percent,
            "realized_effort_percent": self.realized_effort_percent,
            "selected_work_ids": list(self.selected_work_ids),
            "batches": [
                {"batch_id": item.batch_id, "work_ids": list(item.work_ids)}
                for item in self.batches
            ],
            "selection_seed": self.selection_seed,
            "execution_fingerprint": self.execution_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PartialExecutionPlan:
        return cls(
            schema_version=int(payload["schema_version"]),
            problem_fingerprint=str(payload["problem_fingerprint"]),
            microshard_plan_fingerprint=str(payload["microshard_plan_fingerprint"]),
            requested_effort_percent=float(payload["requested_effort_percent"]),
            realized_effort_percent=float(payload["realized_effort_percent"]),
            selected_work_ids=tuple(
                str(item) for item in payload["selected_work_ids"]  # type: ignore[union-attr]
            ),
            batches=tuple(
                PartialExecutionBatch(
                    batch_id=str(item["batch_id"]),
                    work_ids=tuple(str(value) for value in item["work_ids"]),
                )
                for item in payload["batches"]  # type: ignore[union-attr]
            ),
            selection_seed=int(payload["selection_seed"]),
            execution_fingerprint=str(payload["execution_fingerprint"]),
        )


@dataclass(frozen=True, slots=True)
class RoutingCostModel:
    """Documented metadata-only predictor used to balance routing work."""

    group_weight: float = 1.0
    support_weight: float = 1.0
    active_od_weight: float = 1.0
    routing_byte_weight: float = 1.0 / (1024.0 * 1024.0)
    measurement_weight: float = 1.0

    def __post_init__(self) -> None:
        values = asdict(self).values()
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("routing cost-model weights must be finite and nonnegative.")
        if not any(value > 0.0 for value in values):
            raise ValueError("at least one routing cost-model weight must be positive.")


@dataclass(frozen=True, slots=True)
class FixedBudgetRoutingSelection:
    """Nested stratified selection over approximately cost-balanced microshards."""

    microshard_plan_fingerprint: str
    requested_effort_percent: float
    realized_effort_percent: float
    selected_work_ids: tuple[str, ...]
    inclusion_probabilities: tuple[float, ...]
    expansion_weights: tuple[float, ...]
    selection_seed: int
    selection_fingerprint: str
    exact: bool
    schema_version: int = PARALLEL_PARTIAL_EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not 0.0 < self.requested_effort_percent <= 100.0:
            raise ValueError("requested_effort_percent must lie in (0, 100].")
        if not 0.0 < self.realized_effort_percent <= 100.0 + 1.0e-10:
            raise ValueError("realized_effort_percent must lie in (0, 100].")
        size = len(self.selected_work_ids)
        if size == 0 or len(set(self.selected_work_ids)) != size:
            raise ValueError("selected work identities must be nonempty and unique.")
        if len(self.inclusion_probabilities) != size or len(self.expansion_weights) != size:
            raise ValueError("selection probabilities and weights must align with work.")
        if any(not 0.0 < value <= 1.0 for value in self.inclusion_probabilities):
            raise ValueError("inclusion probabilities must lie in (0, 1].")
        if any(not np.isfinite(value) or value < 1.0 for value in self.expansion_weights):
            raise ValueError("expansion weights must be finite and at least one.")
        if self.exact != (self.requested_effort_percent == 100.0):
            raise ValueError("exact must agree with requested effort 100.")

    @property
    def weight_by_work_id(self) -> dict[str, float]:
        return dict(zip(self.selected_work_ids, self.expansion_weights, strict=True))

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["selected_work_ids"] = list(self.selected_work_ids)
        result["inclusion_probabilities"] = list(self.inclusion_probabilities)
        result["expansion_weights"] = list(self.expansion_weights)
        return result


def plan_fixed_budget_routing_selection(
    microshard_plan: RoutingMicroshardPlan,
    *,
    effort_percent: float,
    seed: int,
) -> FixedBudgetRoutingSelection:
    """Select a deterministic prefix per stratum with exact finite-population weights.

    Microshards are balanced by predicted cost before this step. Within each
    stratum, requesting effort ``p`` selects ``ceil(p*n)`` of ``n`` units from a
    stable random ordering. Every unit therefore has first-order inclusion
    probability ``k/n`` under the seed design and expansion weight ``n/k``.
    Prefixes are nested as effort increases for an unchanged seed and plan.
    """
    if not np.isfinite(effort_percent) or not 0.0 < effort_percent <= 100.0:
        raise ValueError("effort_percent must be finite and lie in (0, 100].")
    strata: dict[str, list[RoutingWorkUnit]] = {}
    for item in microshard_plan.microshards:
        strata.setdefault(item.stratum, []).append(item)
    selected: list[tuple[RoutingWorkUnit, float]] = []
    fraction = effort_percent / 100.0
    for stratum in sorted(strata):
        values = sorted(
            strata[stratum],
            key=lambda item: hashlib.sha256(
                json.dumps(
                    (
                        microshard_plan.plan_fingerprint,
                        seed,
                        stratum,
                        item.work_id,
                    ),
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        )
        count = len(values) if effort_percent == 100.0 else max(1, int(np.ceil(fraction * len(values))))
        probability = count / len(values)
        selected.extend((item, probability) for item in values[:count])
    selected.sort(key=lambda pair: pair[0].work_id)
    total_cost = sum(item.predicted_cost for item in microshard_plan.microshards)
    selected_cost = sum(item.predicted_cost for item, _ in selected)
    payload = {
        "schema_version": PARALLEL_PARTIAL_EXECUTION_SCHEMA_VERSION,
        "microshard_plan_fingerprint": microshard_plan.plan_fingerprint,
        "effort_percent": effort_percent,
        "seed": seed,
        "selected": [item.work_id for item, _ in selected],
    }
    return FixedBudgetRoutingSelection(
        microshard_plan_fingerprint=microshard_plan.plan_fingerprint,
        requested_effort_percent=float(effort_percent),
        realized_effort_percent=100.0 * selected_cost / total_cost,
        selected_work_ids=tuple(item.work_id for item, _ in selected),
        inclusion_probabilities=tuple(probability for _, probability in selected),
        expansion_weights=tuple(1.0 / probability for _, probability in selected),
        selection_seed=int(seed),
        selection_fingerprint=_fingerprint(payload),
        exact=effort_percent == 100.0,
    )


def routing_group_work_units(
    operator,
    *,
    cost_model: RoutingCostModel = RoutingCostModel(),
    stratum_by_group: Mapping[int, str] | None = None,
) -> tuple[RoutingWorkUnit, ...]:
    """Describe one metadata-only atomic work unit per destination group."""
    inputs = operator.inputs
    links = int(operator.routing.num_links)
    bytes_per_group = links * (operator.dtype.itemsize + np.dtype(bool).itemsize)
    group_od_mask = np.asarray(inputs.group_od_mask)
    source_link_mask = np.asarray(inputs.group_link_mask)
    measurement_links = np.asarray(operator.spec.link_index)
    result = []
    for group in range(operator.routing.num_destination_groups):
        active_od = int(np.count_nonzero(group_od_mask[group]))
        support = int(np.count_nonzero(source_link_mask[group]))
        measurement_support = int(
            np.count_nonzero(source_link_mask[group, measurement_links])
        )
        cost = (
            cost_model.group_weight
            + cost_model.support_weight * support
            + cost_model.active_od_weight * active_od
            + cost_model.routing_byte_weight * bytes_per_group
            + cost_model.measurement_weight * measurement_support
        )
        result.append(
            RoutingWorkUnit(
                work_id=f"group-{group}",
                destination_group_indices=(group,),
                predicted_cost=float(cost),
                routing_bytes=bytes_per_group,
                active_od_cells=active_od,
                support_entries=support,
                measurement_support=measurement_support,
                stratum=(
                    "default"
                    if stratum_by_group is None
                    else str(stratum_by_group.get(group, "default"))
                ),
            )
        )
    return tuple(result)


def build_balanced_microshard_plan(
    work_units: Sequence[RoutingWorkUnit],
    *,
    target_microshards: int,
    problem_fingerprint: str,
) -> RoutingMicroshardPlan:
    """Build a deterministic longest-processing-time balanced partition."""
    units = tuple(work_units)
    if not units:
        raise ValueError("work_units must not be empty.")
    if target_microshards <= 0:
        raise ValueError("target_microshards must be positive.")
    if not problem_fingerprint:
        raise ValueError("problem_fingerprint must not be empty.")
    source_ids = tuple(item.work_id for item in units)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source work identities must be unique.")
    source_groups = [group for item in units for group in item.destination_group_indices]
    if len(source_groups) != len(set(source_groups)):
        raise ValueError("source work units must not overlap destination groups.")

    count = min(target_microshards, len(units))
    buckets: list[list[RoutingWorkUnit]] = [[] for _ in range(count)]
    costs = [0.0] * count
    ordered = sorted(units, key=lambda item: (-item.predicted_cost, item.work_id))
    for item in ordered:
        target = min(range(count), key=lambda index: (costs[index], index))
        buckets[target].append(item)
        costs[target] += item.predicted_cost

    microshards = []
    for index, bucket in enumerate(buckets):
        members = sorted(bucket, key=lambda item: item.work_id)
        strata = sorted({item.stratum for item in members})
        microshards.append(
            RoutingWorkUnit(
                work_id=f"microshard-{index:05d}",
                destination_group_indices=tuple(
                    sorted(
                        group
                        for item in members
                        for group in item.destination_group_indices
                    )
                ),
                predicted_cost=sum(item.predicted_cost for item in members),
                routing_bytes=sum(item.routing_bytes for item in members),
                active_od_cells=sum(item.active_od_cells for item in members),
                support_entries=sum(item.support_entries for item in members),
                measurement_support=sum(item.measurement_support for item in members),
                stratum="+".join(strata),
            )
        )
    identity = {
        "schema_version": PARALLEL_PARTIAL_EXECUTION_SCHEMA_VERSION,
        "problem_fingerprint": problem_fingerprint,
        "source": [item.to_dict() for item in sorted(units, key=lambda item: item.work_id)],
        "microshards": [item.to_dict() for item in microshards],
    }
    return RoutingMicroshardPlan(
        problem_fingerprint=problem_fingerprint,
        microshards=tuple(microshards),
        source_work_ids=source_ids,
        plan_fingerprint=_fingerprint(identity),
    )


@dataclass(frozen=True, slots=True)
class RoutingWorkObservation:
    operation: str
    shard_indices: tuple[int, ...]
    destination_groups: int
    predicted_routing_bytes: int
    load_and_assembly_seconds: float
    transfer_and_execution_seconds: float
    accumulation_seconds: float
    total_seconds: float
    cache_hits_delta: int
    cache_misses_delta: int
    execution_lane: str


@dataclass(slots=True)
class _PendingObservation:
    started: float
    loaded: float | None = None
    executed: float | None = None
    cache_hits: int = 0
    cache_misses: int = 0


class ShardedWorkInstrumentation:
    """Progress callback collecting per-batch timings without numerical changes."""

    def __init__(self, routing) -> None:
        self._descriptors = {
            item.shard_index: item for item in routing.shard_partition
        }
        self._links = int(routing.num_links)
        self._dtype = np.dtype(routing.probability_dtype)
        self._pending: dict[tuple[str, tuple[int, ...]], _PendingObservation] = {}
        self._last_completed = 0.0
        self._last_cache_hits = 0
        self._last_cache_misses = 0
        self.observations: list[RoutingWorkObservation] = []

    def __call__(self, progress: ShardedOperatorProgress) -> None:
        now = perf_counter()
        key = (progress.operation, progress.current_shard_indices)
        if progress.phase == "product_started":
            self._last_completed = now
            self._last_cache_hits = progress.cache_hits
            self._last_cache_misses = progress.cache_misses
        elif progress.phase == "batch_loaded":
            self._pending[key] = _PendingObservation(
                started=self._last_completed,
                loaded=now,
                cache_hits=progress.cache_hits,
                cache_misses=progress.cache_misses,
            )
        elif progress.phase == "batch_executed" and key in self._pending:
            self._pending[key].executed = now
        elif progress.phase == "batch_accumulated" and key in self._pending:
            pending = self._pending.pop(key)
            loaded = pending.loaded or pending.started
            executed = pending.executed or loaded
            descriptors = [self._descriptors[index] for index in key[1]]
            groups = sum(item.num_groups for item in descriptors)
            bytes_per_entry = self._dtype.itemsize + np.dtype(bool).itemsize
            self.observations.append(
                RoutingWorkObservation(
                    operation=progress.operation,
                    shard_indices=key[1],
                    destination_groups=groups,
                    predicted_routing_bytes=groups * self._links * bytes_per_entry,
                    load_and_assembly_seconds=max(0.0, loaded - pending.started),
                    transfer_and_execution_seconds=max(0.0, executed - loaded),
                    accumulation_seconds=max(0.0, now - executed),
                    total_seconds=max(0.0, now - pending.started),
                    cache_hits_delta=max(0, progress.cache_hits - self._last_cache_hits),
                    cache_misses_delta=max(
                        0, progress.cache_misses - self._last_cache_misses
                    ),
                    execution_lane=f"thread-{get_ident()}",
                )
            )
            self._last_completed = now
            self._last_cache_hits = progress.cache_hits
            self._last_cache_misses = progress.cache_misses

    def report(self) -> dict[str, object]:
        observations = [asdict(item) for item in self.observations]
        return {
            "schema_version": PARALLEL_PARTIAL_EXECUTION_SCHEMA_VERSION,
            "observations": observations,
            "totals": {
                "batches": len(observations),
                "destination_groups": sum(
                    item.destination_groups for item in self.observations
                ),
                "predicted_routing_bytes": sum(
                    item.predicted_routing_bytes for item in self.observations
                ),
                "wall_observation_seconds": sum(
                    item.total_seconds for item in self.observations
                ),
                "cache_hits": sum(item.cache_hits_delta for item in self.observations),
                "cache_misses": sum(
                    item.cache_misses_delta for item in self.observations
                ),
            },
        }
