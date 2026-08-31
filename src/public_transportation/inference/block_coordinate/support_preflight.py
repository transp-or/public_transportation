"""Bounded, resumable exact-support preflight for fixed-routing OD blocks."""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import resource
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import perf_counter

import numpy as np

from public_transportation.assignment.dial_dp import prepare_destination_routing
from public_transportation.inference.assignment_adapter import (
    AssignmentInputs,
    _routing_inputs_for_destination,
)
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)
from public_transportation.inference.construction_control import (
    estimate_completed_unit_eta,
)

from ._canonical import canonical_json, fingerprint
from .partition import ODBlockPartition

SUPPORT_PREFLIGHT_SCHEMA_VERSION = 3
LEGACY_SUPPORT_PREFLIGHT_SCHEMA_VERSION = 2


class SupportPreflightMode(str, Enum):
    STRUCTURAL = "structural"
    SAMPLED_EXACT_SUPPORT = "sampled_exact_support"
    STREAMING_EXACT_SUPPORT = "streaming_exact_support"
    EXACT_MATERIALIZED_PLAN = "exact_materialized_plan"


class SupportPreflightStatus(str, Enum):
    COMPLETED = "completed"
    STOPPED_TIME = "stopped_time_budget"
    STOPPED_RSS = "stopped_rss_budget"
    STOPPED_TEMPORARY = "stopped_temporary_budget"
    STOPPED_RETAINED = "stopped_retained_state_budget"
    INTERRUPTED = "interrupted_partial"
    RESOURCE_GUARD = "resource_guard_triggered"
    NUMERICAL_FAILURE = "numerical_failure"


class SupportPreflightStopLocation(str, Enum):
    COMPLETED = "completed"
    BEFORE_GROUP = "before_group"
    INSIDE_GROUP = "inside_group"
    INSIDE_CHUNK = "inside_chunk"


@dataclass(frozen=True, slots=True)
class SupportPreflightBudget:
    maximum_elapsed_seconds: float = 3600.0
    maximum_process_rss_bytes: int = 16 * 1024**3
    maximum_temporary_bytes: int = 512 * 1024**2
    maximum_retained_support_bytes: int = 64 * 1024**2
    maximum_support_rows_per_block: int = 100_000
    maximum_nonzeros_per_block: int = 10_000_000
    maximum_block_operator_bytes: int = 512 * 1024**2

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum_elapsed_seconds) or self.maximum_elapsed_seconds <= 0:
            raise ValueError("maximum_elapsed_seconds must be finite and positive.")
        for name in (
            "maximum_process_rss_bytes",
            "maximum_temporary_bytes",
            "maximum_retained_support_bytes",
            "maximum_support_rows_per_block",
            "maximum_nonzeros_per_block",
            "maximum_block_operator_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")


@dataclass(frozen=True, slots=True)
class SupportPreflightInvocationPolicy:
    """Operational policy that may change compatibly between invocations."""

    budget: SupportPreflightBudget
    checkpoint_interval_groups: int
    checkpoint_interval_seconds: float
    progress_interval_groups: int
    retain_partial_results: bool
    construction_workers: int
    threads_per_worker: int

    @property
    def fingerprint(self) -> str:
        return fingerprint(self)


@dataclass(frozen=True, slots=True)
class SupportPreflightConfig:
    mode: SupportPreflightMode = SupportPreflightMode.SAMPLED_EXACT_SUPPORT
    destination_group_ids: tuple[int, ...] | None = None
    sample_count: int = 6
    sampling_seed: int = 0
    origin_chunk_size: int = 32
    probability_tolerance: float = 0.0
    checkpoint_directory: Path | None = None
    checkpoint_interval_groups: int = 1
    checkpoint_interval_seconds: float = 60.0
    progress_interval_groups: int = 1
    retain_partial_results: bool = True
    persist_selected_block_support: bool = False
    construction_workers: int = 1
    threads_per_worker: int = 1
    authorize_exact_materialized_plan: bool = False
    budget: SupportPreflightBudget = field(default_factory=SupportPreflightBudget)
    fingerprint: str = field(init=False)
    semantics_fingerprint: str = field(init=False)
    policy_fingerprint: str = field(init=False)
    invocation_policy: SupportPreflightInvocationPolicy = field(init=False)

    def __post_init__(self) -> None:
        try:
            mode = SupportPreflightMode(self.mode)
        except ValueError as error:
            raise ValueError("invalid support-preflight mode.") from error
        groups = self.destination_group_ids
        if groups is not None:
            groups = tuple(int(value) for value in groups)
            if any(value < 0 for value in groups) or len(set(groups)) != len(groups):
                raise ValueError("destination_group_ids must be unique and non-negative.")
        for name in (
            "sample_count",
            "origin_chunk_size",
            "checkpoint_interval_groups",
            "progress_interval_groups",
            "construction_workers",
            "threads_per_worker",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not math.isfinite(self.checkpoint_interval_seconds) or self.checkpoint_interval_seconds <= 0:
            raise ValueError("checkpoint_interval_seconds must be finite and positive.")
        if not math.isfinite(self.probability_tolerance) or self.probability_tolerance < 0:
            raise ValueError("probability_tolerance must be finite and non-negative.")
        if mode is SupportPreflightMode.EXACT_MATERIALIZED_PLAN and not self.authorize_exact_materialized_plan:
            raise ValueError("exact_materialized_plan requires explicit authorization.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "destination_group_ids", groups)
        semantic_payload = {
            "schema_version": SUPPORT_PREFLIGHT_SCHEMA_VERSION,
            "persisted_support_representation": "group-block-summary-v1",
            "mode": mode,
            "destination_group_ids": groups,
            "sample_count": self.sample_count,
            "sampling_seed": self.sampling_seed,
            "origin_chunk_size": self.origin_chunk_size,
            "probability_tolerance": self.probability_tolerance,
            "persist_selected_block_support": self.persist_selected_block_support,
            "authorize_exact_materialized_plan": self.authorize_exact_materialized_plan,
        }
        semantic_identity = fingerprint(semantic_payload)
        invocation_policy = SupportPreflightInvocationPolicy(
            budget=self.budget,
            checkpoint_interval_groups=self.checkpoint_interval_groups,
            checkpoint_interval_seconds=self.checkpoint_interval_seconds,
            progress_interval_groups=self.progress_interval_groups,
            retain_partial_results=self.retain_partial_results,
            construction_workers=self.construction_workers,
            threads_per_worker=self.threads_per_worker,
        )
        object.__setattr__(self, "fingerprint", semantic_identity)
        object.__setattr__(self, "semantics_fingerprint", semantic_identity)
        object.__setattr__(self, "policy_fingerprint", invocation_policy.fingerprint)
        object.__setattr__(self, "invocation_policy", invocation_policy)


@dataclass(frozen=True, slots=True)
class SupportPreflightFingerprints:
    scenario: str
    assignment_inputs: str
    od_layout: str
    fixed_demand: str
    measurement_mapping: str
    routing: str
    partition: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} fingerprint must be nonempty.")

    @property
    def fingerprint(self) -> str:
        return fingerprint(self)


@dataclass(frozen=True, slots=True)
class DestinationSupportSummary:
    group_id: int
    free_columns: int
    positive_fixed_columns: int
    measurement_support_rows: int
    exact_nonzeros: int
    unique_support_patterns: int
    support_discovery_seconds: float
    estimated_temporary_bytes: int
    rss_bytes: int


@dataclass(frozen=True, slots=True)
class BlockSupportSummary:
    block_id: str
    group_id: int
    free_columns: int
    measurement_support_rows: int
    exact_nonzeros: int
    estimated_operator_bytes: int
    exceeds_support_rows: bool
    exceeds_nonzeros: bool
    exceeds_operator_bytes: bool


@dataclass(frozen=True, slots=True)
class SupportPreflightProgress:
    processed_groups: int
    total_groups: int
    processed_free_columns: int
    elapsed_seconds: float
    current_rss_bytes: int
    peak_rss_bytes: int
    retained_state_bytes: int
    current_group_id: int | None
    current_invocation_elapsed_seconds: float = 0.0
    previous_invocations_elapsed_seconds: float = 0.0
    invocation_count: int = 1
    schema_version: int = 1
    status: str = "running"
    completed_units: int | None = None
    total_units: int | None = None
    recent_unit_seconds: float | None = None
    predicted_remaining_seconds: float | None = None
    eta_confidence: str = "unavailable"
    eta_reason: str | None = None
    estimated_completion_at_utc: str | None = None
    eta_lower_seconds: float | None = None
    eta_upper_seconds: float | None = None
    throughput_units_per_second: float | None = None
    checkpoint_location: str | None = None


@dataclass(frozen=True, slots=True)
class SampledSupportExtrapolation:
    """Conservative observed-range extrapolation for incomplete sampled coverage."""

    sampled_groups: int
    population_groups: int
    support_seconds_range: tuple[float, float]
    nonzero_count_range: tuple[int, int]
    cache_bytes_range: tuple[int, int]
    storage_shards_range: tuple[int, int]
    worker_memory_range: tuple[int, int]
    largest_observed_block_bytes: int
    largest_block_risk: str


@dataclass(frozen=True, slots=True)
class SupportPreflightResult:
    status: SupportPreflightStatus
    reason: str
    mode: SupportPreflightMode
    total_destination_groups: int
    selected_destination_groups: tuple[int, ...]
    full_network_coverage: bool
    completed_destination_groups: tuple[int, ...]
    pending_destination_groups: tuple[int, ...]
    processed_free_columns: int
    elapsed_seconds: float
    current_rss_bytes: int
    peak_rss_bytes: int
    temporary_high_water_bytes: int
    retained_state_bytes: int
    destination_summaries: tuple[DestinationSupportSummary, ...]
    block_summaries: tuple[BlockSupportSummary, ...]
    predicted_storage_shards: int
    predicted_cache_bytes: int
    predicted_construction_dispatches: int
    largest_observed_block_operator_bytes: int
    rss_enforcement_available: bool
    assumptions: tuple[str, ...]
    extrapolation: SampledSupportExtrapolation | None
    fingerprints: SupportPreflightFingerprints
    config_fingerprint: str
    policy_fingerprint: str = "legacy"
    semantic_config_fingerprint: str = ""
    invocation_policy: SupportPreflightInvocationPolicy | None = None
    cumulative_elapsed_seconds: float = 0.0
    previous_invocations_elapsed_seconds: float = 0.0
    current_invocation_elapsed_seconds: float = 0.0
    invocation_count: int = 1
    invocation_allowance_seconds: float = 0.0
    invocation_allowance_overshoot_seconds: float = 0.0
    stop_location: SupportPreflightStopLocation = SupportPreflightStopLocation.COMPLETED
    stop_group_id: int | None = None
    discarded_partial_group_seconds: float = 0.0
    schema_version: int = SUPPORT_PREFLIGHT_SCHEMA_VERSION

    @property
    def complete(self) -> bool:
        return self.status is SupportPreflightStatus.COMPLETED


ProgressCallback = Callable[[SupportPreflightProgress], None]
ResourceObserver = Callable[[], int]
Clock = Callable[[], float]


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(payload) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_path(config: SupportPreflightConfig) -> Path | None:
    return None if config.checkpoint_directory is None else config.checkpoint_directory / "support-preflight.json"


def _result_from_payload(payload: dict) -> SupportPreflightResult:
    schema_version = payload.get("schema_version")
    if schema_version == 1:
        raise ValueError(
            "support-preflight checkpoint schema 1 cannot be resumed safely; "
            "start a new deterministic preflight."
        )
    if schema_version not in {LEGACY_SUPPORT_PREFLIGHT_SCHEMA_VERSION, SUPPORT_PREFLIGHT_SCHEMA_VERSION}:
        raise ValueError(
            "unsupported support-preflight checkpoint schema "
            f"{schema_version!r}; expected 2 or {SUPPORT_PREFLIGHT_SCHEMA_VERSION}."
        )
    cumulative_elapsed = float(payload["elapsed_seconds"])
    current_elapsed = float(
        payload.get("current_invocation_elapsed_seconds", cumulative_elapsed)
    )
    previous_elapsed = float(
        payload.get(
            "previous_invocations_elapsed_seconds",
            max(0.0, cumulative_elapsed - current_elapsed),
        )
    )
    policy_payload = payload.get("invocation_policy")
    invocation_policy = None
    if policy_payload is not None:
        invocation_policy = SupportPreflightInvocationPolicy(
            budget=SupportPreflightBudget(**policy_payload["budget"]),
            checkpoint_interval_groups=policy_payload["checkpoint_interval_groups"],
            checkpoint_interval_seconds=policy_payload["checkpoint_interval_seconds"],
            progress_interval_groups=policy_payload["progress_interval_groups"],
            retain_partial_results=policy_payload["retain_partial_results"],
            construction_workers=policy_payload["construction_workers"],
            threads_per_worker=policy_payload["threads_per_worker"],
        )
    result = SupportPreflightResult(
        status=SupportPreflightStatus(payload["status"]),
        reason=payload["reason"],
        mode=SupportPreflightMode(payload["mode"]),
        total_destination_groups=payload["total_destination_groups"],
        selected_destination_groups=tuple(payload["selected_destination_groups"]),
        full_network_coverage=payload["full_network_coverage"],
        completed_destination_groups=tuple(payload["completed_destination_groups"]),
        pending_destination_groups=tuple(payload["pending_destination_groups"]),
        processed_free_columns=payload["processed_free_columns"],
        elapsed_seconds=cumulative_elapsed,
        current_rss_bytes=payload["current_rss_bytes"],
        peak_rss_bytes=payload["peak_rss_bytes"],
        temporary_high_water_bytes=payload["temporary_high_water_bytes"],
        retained_state_bytes=payload["retained_state_bytes"],
        destination_summaries=tuple(DestinationSupportSummary(**item) for item in payload["destination_summaries"]),
        block_summaries=tuple(BlockSupportSummary(**item) for item in payload["block_summaries"]),
        predicted_storage_shards=payload["predicted_storage_shards"],
        predicted_cache_bytes=payload["predicted_cache_bytes"],
        predicted_construction_dispatches=payload["predicted_construction_dispatches"],
        largest_observed_block_operator_bytes=payload["largest_observed_block_operator_bytes"],
        rss_enforcement_available=payload["rss_enforcement_available"],
        assumptions=tuple(payload["assumptions"]),
        extrapolation=(
            None
            if payload["extrapolation"] is None
            else SampledSupportExtrapolation(**payload["extrapolation"])
        ),
        fingerprints=SupportPreflightFingerprints(**payload["fingerprints"]),
        config_fingerprint=payload["config_fingerprint"],
        policy_fingerprint=payload.get("policy_fingerprint") or "legacy",
        semantic_config_fingerprint=payload.get(
            "semantic_config_fingerprint", payload["config_fingerprint"]
        ),
        invocation_policy=invocation_policy,
        cumulative_elapsed_seconds=float(
            payload.get("cumulative_elapsed_seconds", cumulative_elapsed)
        ),
        previous_invocations_elapsed_seconds=previous_elapsed,
        current_invocation_elapsed_seconds=current_elapsed,
        invocation_count=int(payload.get("invocation_count", 1)),
        invocation_allowance_seconds=float(
            payload.get("invocation_allowance_seconds", 0.0)
        ),
        invocation_allowance_overshoot_seconds=float(
            payload.get("invocation_allowance_overshoot_seconds", 0.0)
        ),
        stop_location=SupportPreflightStopLocation(
            payload.get(
                "stop_location",
                "completed"
                if payload["status"] == SupportPreflightStatus.COMPLETED.value
                else "before_group",
            )
        ),
        stop_group_id=payload.get("stop_group_id"),
        discarded_partial_group_seconds=float(
            payload.get("discarded_partial_group_seconds", 0.0)
        ),
    )
    elapsed_values = (
        result.elapsed_seconds,
        result.cumulative_elapsed_seconds,
        result.previous_invocations_elapsed_seconds,
        result.current_invocation_elapsed_seconds,
        result.invocation_allowance_seconds,
        result.invocation_allowance_overshoot_seconds,
        result.discarded_partial_group_seconds,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in elapsed_values):
        raise ValueError("support-preflight checkpoint elapsed-time fields are invalid.")
    if not math.isclose(
        result.previous_invocations_elapsed_seconds
        + result.current_invocation_elapsed_seconds,
        result.cumulative_elapsed_seconds,
        rel_tol=1.0e-9,
        abs_tol=1.0e-6,
    ):
        raise ValueError("support-preflight checkpoint elapsed-time accounting is inconsistent.")
    return result


def _legacy_config_fingerprint(config: SupportPreflightConfig) -> str:
    """Reproduce the pre-v2 fingerprint for an exact, safe legacy resume."""
    return fingerprint(
        {
            "mode": config.mode,
            "destination_group_ids": config.destination_group_ids,
            "sample_count": config.sample_count,
            "sampling_seed": config.sampling_seed,
            "origin_chunk_size": config.origin_chunk_size,
            "probability_tolerance": config.probability_tolerance,
            "checkpoint_interval_groups": config.checkpoint_interval_groups,
            "checkpoint_interval_seconds": config.checkpoint_interval_seconds,
            "progress_interval_groups": config.progress_interval_groups,
            "retain_partial_results": config.retain_partial_results,
            "persist_selected_block_support": config.persist_selected_block_support,
            "construction_workers": config.construction_workers,
            "threads_per_worker": config.threads_per_worker,
            "authorize_exact_materialized_plan": config.authorize_exact_materialized_plan,
            "budget": config.budget,
        }
    )


def _validate_resume_policy(
    result: SupportPreflightResult, config: SupportPreflightConfig
) -> None:
    budget = config.budget
    if result.retained_state_bytes > budget.maximum_retained_support_bytes:
        raise ValueError(
            "support-preflight checkpoint retained state exceeds the new retained-state limit."
        )
    for summary in result.block_summaries:
        if summary.measurement_support_rows > budget.maximum_support_rows_per_block:
            raise ValueError(
                "support-preflight checkpoint contains a block above the new support-row limit."
            )
        if summary.exact_nonzeros > budget.maximum_nonzeros_per_block:
            raise ValueError(
                "support-preflight checkpoint contains a block above the new nonzero limit."
            )
        if summary.estimated_operator_bytes > budget.maximum_block_operator_bytes:
            raise ValueError(
                "support-preflight checkpoint contains a block above the new operator-size limit."
            )


def load_support_preflight_checkpoint(
    directory: Path | str,
    *,
    fingerprints: SupportPreflightFingerprints,
    config: SupportPreflightConfig,
) -> SupportPreflightResult:
    path = Path(directory) / "support-preflight.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("support-preflight checkpoint is missing or corrupt.") from error
    result = _result_from_payload(payload)
    if result.fingerprints != fingerprints:
        raise ValueError("support-preflight checkpoint fingerprints are incompatible.")
    if result.policy_fingerprint == "legacy":
        if result.config_fingerprint != _legacy_config_fingerprint(config):
            raise ValueError(
                "legacy support-preflight checkpoint configuration cannot be safely "
                "migrated after configuration or budget changes; resume once with the "
                "original configuration to write the current schema."
            )
    elif result.semantic_config_fingerprint != config.semantics_fingerprint:
        raise ValueError("support-preflight checkpoint semantic configuration is incompatible.")
    _validate_resume_policy(result, config)
    return result


def _selected_groups(
    free_counts: np.ndarray,
    config: SupportPreflightConfig,
) -> tuple[int, ...]:
    nonempty = np.flatnonzero(free_counts > 0)
    if config.destination_group_ids is not None:
        if any(value >= free_counts.size for value in config.destination_group_ids):
            raise ValueError("destination_group_ids contain an out-of-range group.")
        return tuple(config.destination_group_ids)
    if config.mode in {SupportPreflightMode.STRUCTURAL, SupportPreflightMode.STREAMING_EXACT_SUPPORT, SupportPreflightMode.EXACT_MATERIALIZED_PLAN}:
        return tuple(int(value) for value in nonempty)
    if nonempty.size <= config.sample_count:
        return tuple(int(value) for value in nonempty)
    ordered = nonempty[np.argsort(free_counts[nonempty], kind="stable")]
    anchors = {int(ordered[0]), int(ordered[-1])}
    for quantile in (0.5, 0.95):
        anchors.add(int(ordered[round((ordered.size - 1) * quantile)]))
    rng = np.random.default_rng(config.sampling_seed)
    remaining = np.asarray([value for value in nonempty if int(value) not in anchors])
    needed = max(0, config.sample_count - len(anchors))
    if needed and remaining.size:
        anchors.update(int(value) for value in rng.choice(remaining, min(needed, remaining.size), replace=False))
    return tuple(sorted(anchors))


def _reachability(
    origins: np.ndarray,
    *,
    enabled: np.ndarray,
    topo: np.ndarray,
    out_links: np.ndarray,
    out_mask: np.ndarray,
    head: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    reachable = np.zeros((origins.size, num_nodes), dtype=bool)
    reachable[np.arange(origins.size), origins] = True
    for node in topo:
        active = reachable[:, node]
        if not np.any(active):
            continue
        links = out_links[node][out_mask[node]]
        links = links[enabled[links]]
        if links.size:
            reachable[np.ix_(active, np.unique(head[links]))] = True
    return reachable


def run_support_preflight(
    *,
    inputs: AssignmentInputs,
    theta: float,
    spec: object,
    compact_layout: CompactODAssignmentLayout,
    partition: ODBlockPartition | None,
    fingerprints: SupportPreflightFingerprints,
    config: SupportPreflightConfig,
    resume: bool = False,
    resource_observer: ResourceObserver = _rss_bytes,
    progress_callback: ProgressCallback | None = None,
    clock: Clock = perf_counter,
) -> SupportPreflightResult:
    """Discover exact support one destination at a time without global support state."""
    if config.mode is SupportPreflightMode.EXACT_MATERIALIZED_PLAN:
        raise ValueError("the materialized planner is intentionally separate from bounded streaming preflight.")
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be finite and positive.")
    num_groups = int(inputs.group_dest_node.shape[0])
    num_active = int(inputs.od_origin_node.shape[0])
    if compact_layout.num_active != num_active:
        raise ValueError("compact layout does not match assignment inputs.")
    free_column = np.full(num_active, -1, dtype=np.int64)
    free_active = np.asarray(compact_layout.free_compact_indices, dtype=np.int64)
    free_column[free_active] = np.arange(compact_layout.num_free, dtype=np.int64)
    fixed_active = np.asarray(compact_layout.fixed_compact_indices, dtype=np.int64)
    fixed_values = np.asarray(compact_layout.fixed_compact_values)
    positive_fixed = set(int(value) for value in fixed_active[fixed_values > 0])
    selected_active = (free_column >= 0)
    if positive_fixed:
        selected_active[list(positive_fixed)] = True
    group_indices = np.asarray(inputs.group_od_index_padded)
    group_masks = np.asarray(inputs.group_od_mask)
    free_counts = np.asarray([
        np.count_nonzero(free_column[group_indices[g][group_masks[g]]] >= 0)
        for g in range(num_groups)
    ], dtype=np.int64)
    selected_groups = _selected_groups(free_counts, config)
    all_nonempty_groups = tuple(int(value) for value in np.flatnonzero(free_counts > 0))
    block_by_column: dict[int, str] = {}
    block_size: dict[str, int] = {}
    if partition is not None:
        for block in partition.blocks:
            block_size[block.block_id] = block.num_free_variables
            for column in block.free_column_indices:
                block_by_column[int(column)] = block.block_id

    destination_summaries: list[DestinationSupportSummary] = []
    block_summaries: list[BlockSupportSummary] = []
    completed: list[int] = []
    elapsed_offset = 0.0
    invocation_count = 1
    discarded_partial_group_seconds = 0.0
    peak_rss = resource_observer()
    temporary_high = retained_bytes = 0
    checkpoint = _checkpoint_path(config)
    if resume:
        if checkpoint is None:
            raise ValueError("resume requires checkpoint_directory.")
        previous = load_support_preflight_checkpoint(checkpoint.parent, fingerprints=fingerprints, config=config)
        if previous.selected_destination_groups != selected_groups:
            raise ValueError("checkpoint pending-group order is incompatible.")
        destination_summaries.extend(previous.destination_summaries)
        block_summaries.extend(previous.block_summaries)
        completed.extend(previous.completed_destination_groups)
        elapsed_offset = previous.cumulative_elapsed_seconds or previous.elapsed_seconds
        invocation_count = previous.invocation_count + 1
        discarded_partial_group_seconds = previous.discarded_partial_group_seconds
        peak_rss = max(peak_rss, previous.peak_rss_bytes)
        temporary_high = previous.temporary_high_water_bytes

    pending = [value for value in selected_groups if value not in set(completed)]
    mapping_links = np.asarray(getattr(spec, "link_index"), dtype=np.int64)
    mapping_rows = np.asarray(getattr(spec, "measurement_index"), dtype=np.int64)
    if mapping_links.shape != mapping_rows.shape:
        raise ValueError("measurement mapping arrays have inconsistent shapes.")
    origins = np.asarray(inputs.od_origin_node, dtype=np.int64)
    graph = inputs.graph
    topo = np.asarray(graph.topo_order, dtype=np.int64)
    out_links = np.asarray(graph.out_links, dtype=np.int64)
    out_mask = np.asarray(graph.out_mask, dtype=bool)
    head = np.asarray(graph.head, dtype=np.int64)
    tail = np.asarray(graph.tail, dtype=np.int64)
    num_nodes = int(graph.num_nodes)
    start = clock()
    last_checkpoint = start
    recent_group_seconds: deque[float] = deque(maxlen=32)
    status = SupportPreflightStatus.COMPLETED
    reason = "selected destination groups completed"
    current_group: int | None = None
    stop_location = SupportPreflightStopLocation.COMPLETED
    stop_group_id: int | None = None

    def emit_progress(
        *,
        current_group_id: int | None,
        recent_unit_seconds: float | None,
        force_status: SupportPreflightStatus | None = None,
    ) -> None:
        if progress_callback is None:
            return
        completed_groups = len(completed)
        eta = estimate_completed_unit_eta(
            recent_group_seconds,
            completed_units=completed_groups,
            total_units=len(selected_groups),
            parallelism=config.construction_workers,
            elapsed_seconds=max(0.0, clock() - start),
        )
        event_status = (status if force_status is None else force_status).value
        event = SupportPreflightProgress(
            completed_groups,
            len(selected_groups),
            sum(item.free_columns for item in destination_summaries),
            elapsed_offset + max(0.0, clock() - start),
            resource_observer(),
            peak_rss,
            retained_bytes,
            current_group_id,
            max(0.0, clock() - start),
            elapsed_offset,
            invocation_count,
            status=event_status,
            completed_units=completed_groups,
            total_units=len(selected_groups),
            recent_unit_seconds=recent_unit_seconds,
            predicted_remaining_seconds=eta.predicted_remaining_seconds,
            eta_confidence=eta.eta_confidence,
            eta_reason=eta.eta_reason,
            estimated_completion_at_utc=eta.estimated_completion_at_utc,
            eta_lower_seconds=eta.eta_lower_seconds,
            eta_upper_seconds=eta.eta_upper_seconds,
            throughput_units_per_second=eta.throughput_units_per_second,
            checkpoint_location=None if checkpoint is None else str(checkpoint),
        )
        try:
            progress_callback(event)
        except Exception:
            # Progress is observability only; preserve support decisions when
            # a consumer or log sink is unavailable.
            return

    def build_result() -> SupportPreflightResult:
        invocation_elapsed = clock() - start
        elapsed = elapsed_offset + invocation_elapsed
        current_rss = resource_observer()
        predicted_cache = sum(item.estimated_operator_bytes for item in block_summaries)
        extrapolation = None
        if (
            config.mode is SupportPreflightMode.SAMPLED_EXACT_SUPPORT
            and destination_summaries
            and selected_groups != all_nonempty_groups
        ):
            population = len(all_nonempty_groups)
            times = [item.support_discovery_seconds for item in destination_summaries]
            nonzeros = [item.exact_nonzeros for item in destination_summaries]
            sampled = len(destination_summaries)
            scale = population / sampled
            per_group_shards = len(block_summaries) / sampled if sampled else 0.0
            worker = [item.estimated_operator_bytes for item in block_summaries] or [0]
            extrapolation = SampledSupportExtrapolation(
                sampled_groups=sampled,
                population_groups=population,
                support_seconds_range=(min(times) * population, max(times) * population),
                nonzero_count_range=(min(nonzeros) * population, max(nonzeros) * population),
                cache_bytes_range=(int(predicted_cache * scale * 0.5), int(predicted_cache * scale * 2.0)),
                storage_shards_range=(max(1, int(per_group_shards * population * 0.5)), max(1, math.ceil(per_group_shards * population * 2.0))),
                worker_memory_range=(min(worker), max(worker)),
                largest_observed_block_bytes=max(worker),
                largest_block_risk="unobserved groups may exceed the sampled maximum",
            )
        return SupportPreflightResult(
            status=status,
            reason=reason,
            mode=config.mode,
            total_destination_groups=num_groups,
            selected_destination_groups=selected_groups,
            full_network_coverage=selected_groups == all_nonempty_groups,
            completed_destination_groups=tuple(completed),
            pending_destination_groups=tuple(value for value in selected_groups if value not in set(completed)),
            processed_free_columns=sum(item.free_columns for item in destination_summaries),
            elapsed_seconds=elapsed,
            current_rss_bytes=current_rss,
            peak_rss_bytes=max(peak_rss, current_rss),
            temporary_high_water_bytes=temporary_high,
            retained_state_bytes=retained_bytes,
            destination_summaries=tuple(destination_summaries),
            block_summaries=tuple(block_summaries),
            predicted_storage_shards=len(block_summaries),
            predicted_cache_bytes=predicted_cache,
            predicted_construction_dispatches=len(block_summaries),
            largest_observed_block_operator_bytes=max((item.estimated_operator_bytes for item in block_summaries), default=0),
            rss_enforcement_available=True,
            assumptions=("sample extrapolation is descriptive; no linear-scaling claim",),
            extrapolation=extrapolation,
            fingerprints=fingerprints,
            config_fingerprint=config.fingerprint,
            semantic_config_fingerprint=config.semantics_fingerprint,
            policy_fingerprint=config.policy_fingerprint,
            invocation_policy=config.invocation_policy,
            cumulative_elapsed_seconds=elapsed,
            previous_invocations_elapsed_seconds=elapsed_offset,
            current_invocation_elapsed_seconds=invocation_elapsed,
            invocation_count=invocation_count,
            invocation_allowance_seconds=config.budget.maximum_elapsed_seconds,
            invocation_allowance_overshoot_seconds=max(
                0.0, invocation_elapsed - config.budget.maximum_elapsed_seconds
            ),
            stop_location=stop_location,
            stop_group_id=stop_group_id,
            discarded_partial_group_seconds=discarded_partial_group_seconds,
        )

    def persist() -> SupportPreflightResult:
        nonlocal retained_bytes
        retained_bytes = len(
            canonical_json((destination_summaries, block_summaries)).encode("utf-8")
        )
        result = build_result()
        if checkpoint is not None:
            _atomic_json(checkpoint, result)
        return result

    final_result: SupportPreflightResult | None = None
    try:
        for current_group in pending:
            group_start = clock()
            elapsed = clock() - start
            rss = resource_observer()
            peak_rss = max(peak_rss, rss)
            if elapsed >= config.budget.maximum_elapsed_seconds:
                status, reason = SupportPreflightStatus.STOPPED_TIME, "elapsed-time budget reached before next group"
                stop_location = SupportPreflightStopLocation.BEFORE_GROUP
                stop_group_id = current_group
                break
            if rss >= config.budget.maximum_process_rss_bytes:
                status, reason = SupportPreflightStatus.STOPPED_RSS, "RSS budget reached before next group"
                stop_location = SupportPreflightStopLocation.BEFORE_GROUP
                stop_group_id = current_group
                break
            active = group_indices[current_group][group_masks[current_group]]
            active = active[selected_active[active]].astype(np.int64, copy=False)
            estimated_temporary = config.origin_chunk_size * (num_nodes + mapping_links.size)
            estimated_temporary += int(graph.num_links) * (np.dtype(np.float32).itemsize + np.dtype(bool).itemsize)
            if estimated_temporary > config.budget.maximum_temporary_bytes:
                status, reason = SupportPreflightStatus.RESOURCE_GUARD, f"group {current_group} temporary estimate exceeds budget"
                stop_location = SupportPreflightStopLocation.BEFORE_GROUP
                stop_group_id = current_group
                break
            if rss + estimated_temporary > config.budget.maximum_process_rss_bytes:
                status, reason = (
                    SupportPreflightStatus.RESOURCE_GUARD,
                    f"group {current_group} temporary estimate exceeds remaining RSS budget",
                )
                stop_location = SupportPreflightStopLocation.BEFORE_GROUP
                stop_group_id = current_group
                break
            if config.mode is SupportPreflightMode.STRUCTURAL:
                destination_summaries.append(DestinationSupportSummary(current_group, int(free_counts[current_group]), sum(int(value) in positive_fixed for value in active), 0, 0, 0, 0.0, estimated_temporary, rss))
                completed.append(current_group)
            else:
                enabled_jax, cost = _routing_inputs_for_destination(
                    graph=graph,
                    base_link_cost=inputs.base_link_cost,
                    group_link_mask=inputs.group_link_mask[current_group],
                    dest_node=inputs.group_dest_node[current_group],
                )
                routing = prepare_destination_routing(
                    graph=graph,
                    link_cost=cost,
                    enabled_link_mask=enabled_jax,
                    dest_node=inputs.group_dest_node[current_group],
                    theta=theta,
                )
                enabled = np.asarray(routing.enabled_link_mask)
                probabilities = np.asarray(routing.link_prob)
                eligible = enabled[mapping_links] & (probabilities[mapping_links] > config.probability_tolerance)
                group_rows: set[int] = set()
                patterns: set[tuple[int, ...]] = set()
                exact_nonzeros = 0
                block_rows: dict[str, set[int]] = {}
                block_nnz: dict[str, int] = {}
                group_stop = False
                for first in range(0, active.size, config.origin_chunk_size):
                    chunk = active[first : first + config.origin_chunk_size]
                    reachable = _reachability(origins[chunk], enabled=enabled, topo=topo, out_links=out_links, out_mask=out_mask, head=head, num_nodes=num_nodes)
                    mapped = reachable[:, tail[mapping_links]] & eligible[None, :]
                    for local, active_index in enumerate(chunk):
                        rows = tuple(int(value) for value in np.unique(mapping_rows[mapped[local]]))
                        patterns.add(rows)
                        group_rows.update(rows)
                        exact_nonzeros += len(rows)
                        column = int(free_column[active_index])
                        block_id = block_by_column.get(column)
                        if block_id is not None and column >= 0:
                            block_rows.setdefault(block_id, set()).update(rows)
                            block_nnz[block_id] = block_nnz.get(block_id, 0) + len(rows)
                    del reachable, mapped
                    chunk_elapsed = clock() - start
                    chunk_rss = resource_observer()
                    peak_rss = max(peak_rss, chunk_rss)
                    if chunk_elapsed >= config.budget.maximum_elapsed_seconds:
                        status, reason = (
                            SupportPreflightStatus.STOPPED_TIME,
                            f"elapsed-time budget reached within group {current_group}",
                        )
                        stop_location = SupportPreflightStopLocation.INSIDE_CHUNK
                        stop_group_id = current_group
                        group_stop = True
                        break
                    if chunk_rss >= config.budget.maximum_process_rss_bytes:
                        status, reason = (
                            SupportPreflightStatus.STOPPED_RSS,
                            f"RSS budget reached within group {current_group}",
                        )
                        stop_location = SupportPreflightStopLocation.INSIDE_CHUNK
                        stop_group_id = current_group
                        group_stop = True
                        break
                if group_stop:
                    discarded_partial_group_seconds += clock() - group_start
                    del routing, enabled_jax, cost, enabled, probabilities, eligible
                    del group_rows, patterns, block_rows, block_nnz
                    gc.collect()
                    break
                for block_id in sorted(block_rows):
                    row_count = len(block_rows[block_id])
                    nnz = block_nnz[block_id]
                    one_orientation = nnz * (
                        np.dtype(np.float64).itemsize
                        + np.dtype(np.int32).itemsize
                    ) + (row_count + 1) * np.dtype(np.int32).itemsize
                    operator_bytes = 2 * one_orientation
                    block_summaries.append(BlockSupportSummary(block_id, current_group, block_size[block_id], row_count, nnz, operator_bytes, row_count > config.budget.maximum_support_rows_per_block, nnz > config.budget.maximum_nonzeros_per_block, operator_bytes > config.budget.maximum_block_operator_bytes))
                destination_summaries.append(DestinationSupportSummary(current_group, int(free_counts[current_group]), sum(int(value) in positive_fixed for value in active), len(group_rows), exact_nonzeros, len(patterns), clock() - group_start, estimated_temporary, resource_observer()))
                completed.append(current_group)
                temporary_high = max(temporary_high, estimated_temporary)
                del routing, enabled_jax, cost, enabled, probabilities, eligible, group_rows, patterns, block_rows, block_nnz
                gc.collect()
            recent_group_seconds.append(max(0.0, clock() - group_start))
            retained_bytes = len(canonical_json((destination_summaries, block_summaries)).encode("utf-8"))
            if retained_bytes > config.budget.maximum_retained_support_bytes:
                status, reason = SupportPreflightStatus.STOPPED_RETAINED, "retained summary budget reached"
                stop_location = SupportPreflightStopLocation.BEFORE_GROUP
                stop_group_id = next(
                    (value for value in selected_groups if value not in set(completed)),
                    None,
                )
                break
            now = clock()
            if checkpoint is not None and (len(completed) % config.checkpoint_interval_groups == 0 or now - last_checkpoint >= config.checkpoint_interval_seconds):
                persist()
                last_checkpoint = now
            if len(completed) % config.progress_interval_groups == 0:
                emit_progress(
                    current_group_id=current_group,
                    recent_unit_seconds=recent_group_seconds[-1] if recent_group_seconds else None,
                )
    except KeyboardInterrupt:
        status, reason = SupportPreflightStatus.INTERRUPTED, "keyboard interrupt"
        stop_location = (
            SupportPreflightStopLocation.INSIDE_GROUP
            if current_group is not None
            else SupportPreflightStopLocation.BEFORE_GROUP
        )
        stop_group_id = current_group
    except (FloatingPointError, ArithmeticError) as error:
        status, reason = SupportPreflightStatus.NUMERICAL_FAILURE, str(error)
    finally:
        if config.retain_partial_results or status in {
            SupportPreflightStatus.STOPPED_TIME,
            SupportPreflightStatus.INTERRUPTED,
        }:
            final_result = persist()
        if status is not SupportPreflightStatus.INTERRUPTED:
            emit_progress(
                current_group_id=stop_group_id,
                recent_unit_seconds=recent_group_seconds[-1] if recent_group_seconds else None,
                force_status=status,
            )
    return build_result() if final_result is None else final_result


@dataclass(frozen=True, slots=True)
class PilotAuthorization:
    accepted: bool
    reasons: tuple[str, ...]
    maximum_variables_per_block: int | None = None
    maximum_nonzeros_per_block: int | None = None
    maximum_support_rows: int | None = None
    per_worker_memory_ceiling_bytes: int | None = None
    construction_workers: int | None = None
    solver_workers: int | None = None
    threads_per_worker: int | None = None
    maximum_block_updates: int | None = None
    checkpoint_frequency: int | None = None
    exact_diagnostic_frequency: int | None = None
    elapsed_time_limit_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class SelectedBlockPilotAuthorization:
    """Authorization restricted to one deterministic, preflighted block schedule."""

    accepted: bool
    requested_block_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    maximum_updates: int
    maximum_elapsed_seconds: float


def authorize_selected_block_pilot(
    result: SupportPreflightResult,
    *,
    requested_block_ids: tuple[str, ...],
    support_artifacts: tuple[object, ...],
    maximum_support_rows: int,
    maximum_nonzeros: int,
    maximum_operator_bytes: int,
    maximum_temporary_bytes: int,
    maximum_updates: int,
    maximum_elapsed_seconds: float,
) -> SelectedBlockPilotAuthorization:
    """Authorize only named blocks with complete groups and persisted exact support."""
    requested = tuple(str(value).strip() for value in requested_block_ids)
    reasons: list[str] = []
    if not requested or any(not value for value in requested):
        reasons.append("requested block schedule is empty or invalid")
    if len(requested) != len(set(requested)):
        reasons.append("requested block schedule contains duplicate block IDs")
    for name, value in (
        ("maximum_support_rows", maximum_support_rows),
        ("maximum_nonzeros", maximum_nonzeros),
        ("maximum_operator_bytes", maximum_operator_bytes),
        ("maximum_temporary_bytes", maximum_temporary_bytes),
        ("maximum_updates", maximum_updates),
    ):
        if value <= 0:
            reasons.append(f"{name} must be positive")
    if not math.isfinite(maximum_elapsed_seconds) or maximum_elapsed_seconds <= 0:
        reasons.append("maximum_elapsed_seconds must be finite and positive")
    summaries = {item.block_id: item for item in result.block_summaries}
    artifacts = {str(getattr(item, "block_id", "")): item for item in support_artifacts}
    completed_groups = set(result.completed_destination_groups)
    for block_id in requested:
        summary = summaries.get(block_id)
        artifact = artifacts.get(block_id)
        if summary is None:
            reasons.append(f"block {block_id!r} has no preflight summary")
            continue
        if summary.group_id not in completed_groups:
            reasons.append(f"block {block_id!r} belongs to an incomplete support group")
        if artifact is None:
            reasons.append(f"block {block_id!r} has no exact-support artifact")
        else:
            if int(getattr(artifact, "exact_nonzeros", -1)) != summary.exact_nonzeros:
                reasons.append(f"block {block_id!r} support artifact is incompatible")
            path = Path(getattr(artifact, "path", ""))
            if not path.is_file():
                reasons.append(f"block {block_id!r} support artifact is not persisted")
        if summary.measurement_support_rows > maximum_support_rows:
            reasons.append(f"block {block_id!r} exceeds the support-row ceiling")
        if summary.exact_nonzeros > maximum_nonzeros:
            reasons.append(f"block {block_id!r} exceeds the nonzero ceiling")
        if summary.estimated_operator_bytes > maximum_operator_bytes:
            reasons.append(f"block {block_id!r} exceeds the operator-size ceiling")
        if summary.estimated_operator_bytes > maximum_temporary_bytes:
            reasons.append(f"block {block_id!r} exceeds the temporary-memory ceiling")
    return SelectedBlockPilotAuthorization(
        accepted=not reasons,
        requested_block_ids=requested,
        reasons=tuple(reasons) if reasons else ("all requested blocks are explicitly authorized",),
        maximum_updates=maximum_updates,
        maximum_elapsed_seconds=maximum_elapsed_seconds,
    )


def authorize_block_coordinate_pilot(
    result: SupportPreflightResult,
    *,
    maximum_updates: int = 100,
    safety_fraction: float = 0.8,
) -> PilotAuthorization:
    """Return a conservative pilot configuration without starting estimation."""
    if not 0 < safety_fraction < 1:
        raise ValueError("safety_fraction must be in (0, 1).")
    reasons: list[str] = []
    if not result.complete:
        reasons.append("support preflight is incomplete")
    if not result.full_network_coverage:
        reasons.append("support preflight does not cover every nonempty destination group")
    unsafe = [item for item in result.block_summaries if item.exceeds_support_rows or item.exceeds_nonzeros or item.exceeds_operator_bytes]
    if unsafe:
        reasons.append(f"{len(unsafe)} observed blocks exceed configured limits")
    if not result.block_summaries:
        reasons.append("no exact block-support measurements are available")
    if reasons:
        return PilotAuthorization(False, tuple(reasons))
    maximum_variables = max(item.free_columns for item in result.block_summaries)
    maximum_nonzeros = max(item.exact_nonzeros for item in result.block_summaries)
    maximum_rows = max(item.measurement_support_rows for item in result.block_summaries)
    return PilotAuthorization(True, ("complete bounded preflight is within observed limits",), maximum_variables, maximum_nonzeros, maximum_rows, int(result.largest_observed_block_operator_bytes / safety_fraction), 1, 1, 1, maximum_updates, 10, maximum_updates, min(3600.0, result.elapsed_seconds * 2 + 60.0))
