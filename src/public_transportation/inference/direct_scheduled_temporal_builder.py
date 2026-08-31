"""Direct, resumable scheduled construction of canonical temporal blocks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from collections import deque
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Callable, Literal, Mapping
import uuid

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.measurement.mapping import MappingInfo

from .assignment_adapter import (
    AssignmentInputs,
    FixedRoutingInputs,
    validate_fixed_routing_compatibility,
)
from .assignment_contract import (
    AssignmentArtifactIdentity,
    AssignmentCompatibilityError,
    CanonicalAssignmentIndex,
    fixed_routing_route_choice_fingerprint,
)
from .compact_od_assignment_layout import CompactODAssignmentLayout
from .fixed_routing_measurement_operator import (
    assignment_inputs_fingerprint,
    measurement_mapping_fingerprint,
)
from .fixed_routing_sharded_builder import (
    GroupSupportTimingCallback,
    ShardedConstructionConfig,
    ShardedConstructionResult,
    prepare_sharded_fixed_routing_measurement_operator,
)
from .sharded_sparse_operator import load_sparse_shard, shard_path
from .sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
    FixedRoutingShardProgress,
    ShardedFixedRoutingInputs,
    plan_fixed_routing_shards,
    prepare_fixed_routing_sharded,
)
from .temporal_assignment_blocks import (
    TemporalBlockAssignmentOperator,
    TemporalBlockConstructionDiagnostics,
    TemporalBlockKey,
    TemporalSparseBlock,
)
from .temporal_assignment_sparse_backend import CSRCSCTemporalAssignmentOperator
from .temporal_assignment_persistence import (
    PREFLIGHT_ADOPTION_SCHEMA_VERSION,
    TEMPORAL_OPERATOR_VALIDATOR_VERSION,
    _atomic_json_write,
    _materialize_operator_cache_from_operator,
    load_temporal_block_operator,
    save_temporal_block_operator,
    temporal_block_cache_path,
)
from .measurement_operator_protocol import GravityOperatorCapabilities
from .measurement_support_preflight import (
    PositiveBoardingPreflightContext,
    audit_positive_boarding_support,
    boarding_access_supported_measurement_rows,
    enforce_positive_boarding_support,
    load_positive_boarding_support_cache,
    write_positive_boarding_support_cache,
)
from .construction_control import (
    ConstructionDeadline,
    ConstructionDeadlineStop,
    ConstructionETA,
    ConstructionPhase,
    ConstructionProgressReporter,
    ConstructionTermination,
    deadline_stop,
    estimate_completed_unit_eta,
)

DirectTemporalProgressCallback = Callable[[dict[str, object]], None]
DirectScheduledActivationMode = Literal["off", "auto", "direct"]
DirectFixedRoutingSource = FixedRoutingInputs | ShardedFixedRoutingInputs
TEMPORAL_FRAGMENT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DirectScheduledTemporalConstructionResult:
    operator: TemporalBlockAssignmentOperator
    checkpoint_directory: Path
    artifact_directory: Path | None
    source: ShardedConstructionResult | None
    temporal_artifact_reused: bool
    finalization_seconds: float


@dataclass(frozen=True, slots=True)
class DirectScheduledActivationDecision:
    """Auditable decision made before potentially expensive construction."""

    mode: DirectScheduledActivationMode
    activated: bool
    cache_reused: bool
    reason: str
    expected_evaluations: int
    break_even_evaluations: float | None


@dataclass(frozen=True, slots=True)
class DirectScheduledActivationResult:
    """Production selection result consumable by gravity estimation."""

    operator: DirectScheduledGravityOperator | None
    decision: DirectScheduledActivationDecision
    construction: DirectScheduledTemporalConstructionResult | None
    termination: ConstructionTermination | None = None


@dataclass(frozen=True, slots=True)
class _DirectScheduledMetrics:
    stored_bytes: int
    peak_construction_bytes: int


@dataclass(frozen=True, slots=True)
class DirectScheduledGravityOperator:
    """Expose temporal blocks through the established gravity protocol."""

    operator: TemporalBlockAssignmentOperator
    theta: float
    _execution_backend: CSRCSCTemporalAssignmentOperator = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        value = float(self.theta)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("theta must be positive and finite.")
        if (
            fixed_routing_route_choice_fingerprint(value)
            != self.operator.identity.route_choice_fingerprint
        ):
            raise AssignmentCompatibilityError(
                "temporal operator and fixed theta are incompatible."
            )
        object.__setattr__(self, "theta", value)
        # The persisted temporal blocks are the provenance/source artifact.
        # Estimation executes through one CSR/CSC forward/adjoint pair so JAX
        # does not trace a scatter-update loop over every temporal block.
        object.__setattr__(
            self,
            "_execution_backend",
            CSRCSCTemporalAssignmentOperator(self.operator),
        )

    @property
    def num_free_od(self) -> int:
        return self.operator.number_of_demand_cells

    @property
    def num_measurements(self) -> int:
        return self.operator.number_of_measurements

    @property
    def compact_layout_fingerprint(self) -> str | None:
        return self.operator.canonical_index.source_compact_layout_fingerprint

    @property
    def fixed_measurement_offset(self) -> object:
        return self.operator.fixed_measurement_offset

    @property
    def representation(self) -> str:
        return "direct_scheduled_temporal_blocks_csr_csc"

    @property
    def is_matrix_free(self) -> bool:
        return False

    @property
    def assignment_fingerprint(self) -> str:
        return self.operator.identity.timetable_fingerprint

    @property
    def graph_fingerprint(self) -> str:
        return self.operator.identity.network_fingerprint

    @property
    def mapping_fingerprint(self) -> str:
        return self.operator.identity.measurement_mapping_fingerprint

    @property
    def dtype(self) -> np.dtype:
        return self.operator.dtype

    @property
    def metrics(self) -> _DirectScheduledMetrics:
        stored = int(self.operator.fixed_measurement_offset.nbytes) + sum(
            block.row_indices.nbytes
            + block.column_indices.nbytes
            + block.values.nbytes
            for block in self.operator.blocks
        )
        return _DirectScheduledMetrics(stored_bytes=stored, peak_construction_bytes=0)

    @property
    def product_capabilities(self) -> GravityOperatorCapabilities:
        return GravityOperatorCapabilities(matmat=True)

    def jax_matvec(self, vector: jax.Array) -> jax.Array:
        return self._execution_backend.jax_matvec(vector)

    def jax_rmatvec(self, vector: jax.Array) -> jax.Array:
        return self._execution_backend.jax_rmatvec(vector)

    def jax_matmat(self, matrix: jax.Array) -> jax.Array:
        value = jnp.asarray(matrix, dtype=self.dtype)
        if value.ndim != 2 or value.shape[0] != self.num_free_od:
            raise ValueError(
                f"matrix must have shape ({self.num_free_od}, k), got {value.shape}."
            )
        return jax.vmap(self.jax_matvec, in_axes=1, out_axes=1)(value)


def _fragment_path(directory: Path, shard_key: str) -> Path:
    return directory / "temporal_fragments" / f"{shard_key}.npz"


def _fragment_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _routing_checkpoint_path(checkpoint_directory: Path) -> Path:
    return checkpoint_directory / "routing.npz"


def _save_routing_checkpoint(
    *,
    checkpoint_directory: Path,
    routing: FixedRoutingInputs,
    identity: AssignmentArtifactIdentity,
) -> Path:
    arrays = {
        "effective_group_link_mask": np.asarray(
            routing.effective_group_link_mask, dtype=bool
        ),
        "group_link_probability": np.asarray(routing.group_link_probability),
    }
    metadata = {
        "schema_version": 1,
        "identity_fingerprint": identity.fingerprint,
        "content_hash": _fragment_hash(arrays),
    }
    destination = _routing_checkpoint_path(checkpoint_directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as stream:
            np.savez(stream, metadata=np.asarray(json.dumps(metadata)), **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(temporary, allow_pickle=False) as archive:
            persisted = {
                name: np.asarray(archive[name])
                for name in archive.files
                if name != "metadata"
            }
            persisted_metadata = json.loads(str(archive["metadata"]))
        if persisted_metadata != metadata or _fragment_hash(persisted) != metadata[
            "content_hash"
        ]:
            raise ValueError("staged routing checkpoint failed validation.")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _load_routing_checkpoint(
    *,
    checkpoint_directory: Path,
    inputs: AssignmentInputs,
    identity: AssignmentArtifactIdentity,
    theta: float,
) -> FixedRoutingInputs:
    path = _routing_checkpoint_path(checkpoint_directory)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "metadata"
        }
    if metadata.get("schema_version") != 1:
        raise ValueError("routing checkpoint schema is incompatible.")
    if metadata.get("identity_fingerprint") != identity.fingerprint:
        raise ValueError("routing checkpoint identity is incompatible.")
    if metadata.get("content_hash") != _fragment_hash(arrays):
        raise ValueError("routing checkpoint content hash is invalid.")
    routing = FixedRoutingInputs(
        theta=jnp.asarray(theta, dtype=inputs.base_link_cost.dtype).reshape(()),
        graph=inputs.graph,
        source_base_link_cost=inputs.base_link_cost,
        group_dest_node=inputs.group_dest_node,
        source_group_link_mask=inputs.group_link_mask,
        effective_group_link_mask=jnp.asarray(arrays["effective_group_link_mask"]),
        group_link_probability=jnp.asarray(arrays["group_link_probability"]),
        num_nodes=int(inputs.graph.num_nodes),
        num_links=int(inputs.graph.num_links),
    )
    validate_fixed_routing_compatibility(inputs=inputs, routing=routing)
    if identity.route_choice_fingerprint != fixed_routing_route_choice_fingerprint(
        theta
    ):
        raise AssignmentCompatibilityError(
            "routing checkpoint theta is incompatible."
        )
    return routing


def _save_temporal_fragment(
    *,
    directory: Path,
    shard_key: str,
    identity: AssignmentArtifactIdentity,
    blocks: tuple[TemporalSparseBlock, ...],
    offset_rows: np.ndarray,
    offset_values: np.ndarray,
) -> Path:
    block_lengths = np.asarray(
        [block.nonzero_entries for block in blocks], dtype=np.int64
    )
    block_offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(block_lengths))
    )
    arrays = {
        "block_offsets": block_offsets,
        "measurement_intervals": np.asarray(
            [block.key.measurement_interval_id for block in blocks]
        ),
        "departure_intervals": np.asarray(
            [block.key.departure_interval_id for block in blocks]
        ),
        "rows": (
            np.concatenate([block.row_indices for block in blocks])
            if blocks
            else np.empty(0, dtype=np.int32)
        ),
        "columns": (
            np.concatenate([block.column_indices for block in blocks])
            if blocks
            else np.empty(0, dtype=np.int32)
        ),
        "values": (
            np.concatenate([block.values for block in blocks])
            if blocks
            else np.empty(0, dtype=np.dtype(identity.numeric_dtype))
        ),
        "offset_rows": np.asarray(offset_rows, dtype=np.int32),
        "offset_values": np.asarray(
            offset_values, dtype=np.dtype(identity.numeric_dtype)
        ),
    }
    metadata = {
        "schema_version": TEMPORAL_FRAGMENT_SCHEMA_VERSION,
        "identity_fingerprint": identity.fingerprint,
        "source_shard": shard_key,
        "content_hash": _fragment_hash(arrays),
    }
    destination = _fragment_path(directory, shard_key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as stream:
            np.savez(stream, metadata=np.asarray(json.dumps(metadata)), **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _load_temporal_fragment(
            Path(temporary),
            shard_key=shard_key,
            identity=identity,
            number_of_measurements=blocks[0].number_of_measurements if blocks else 0,
            number_of_demand_cells=blocks[0].number_of_demand_cells if blocks else 0,
        )
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _load_temporal_fragment(
    path: Path,
    *,
    shard_key: str,
    identity: AssignmentArtifactIdentity,
    number_of_measurements: int,
    number_of_demand_cells: int,
) -> tuple[tuple[TemporalSparseBlock, ...], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "metadata"
        }
    if metadata.get("schema_version") != TEMPORAL_FRAGMENT_SCHEMA_VERSION:
        raise ValueError("temporal fragment schema is incompatible.")
    if metadata.get("identity_fingerprint") != identity.fingerprint:
        raise ValueError("temporal fragment identity is incompatible.")
    if metadata.get("source_shard") != shard_key:
        raise ValueError("temporal fragment source shard is incompatible.")
    if metadata.get("content_hash") != _fragment_hash(arrays):
        raise ValueError("temporal fragment content hash is invalid.")
    offsets = arrays["block_offsets"]
    blocks = tuple(
        TemporalSparseBlock(
            key=TemporalBlockKey(str(measurement), str(departure)),
            row_indices=arrays["rows"][offsets[index] : offsets[index + 1]],
            column_indices=arrays["columns"][offsets[index] : offsets[index + 1]],
            values=arrays["values"][offsets[index] : offsets[index + 1]],
            number_of_measurements=number_of_measurements,
            number_of_demand_cells=number_of_demand_cells,
        )
        for index, (measurement, departure) in enumerate(
            zip(
                arrays["measurement_intervals"],
                arrays["departure_intervals"],
                strict=True,
            )
        )
    )
    return blocks, arrays["offset_rows"], arrays["offset_values"]


def _validate_inputs(
    *,
    inputs: AssignmentInputs,
    routing: DirectFixedRoutingSource,
    spec,
    canonical_index: CanonicalAssignmentIndex,
    identity: AssignmentArtifactIdentity,
) -> None:
    _validate_identity_inputs(
        inputs=inputs,
        spec=spec,
        canonical_index=canonical_index,
        identity=identity,
    )
    if identity.route_choice_fingerprint != fixed_routing_route_choice_fingerprint(
        float(np.asarray(routing.theta))
    ):
        raise AssignmentCompatibilityError(
            "direct temporal route-choice fingerprint is incompatible."
        )


def _validate_identity_inputs(
    *,
    inputs: AssignmentInputs,
    spec,
    canonical_index: CanonicalAssignmentIndex,
    identity: AssignmentArtifactIdentity,
) -> None:
    if identity.canonical_index_fingerprint != canonical_index.artifact_fingerprint:
        raise AssignmentCompatibilityError(
            "direct temporal identity and canonical index are incompatible."
        )
    assignment = assignment_inputs_fingerprint(inputs)
    if identity.network_fingerprint != assignment:
        raise AssignmentCompatibilityError(
            "direct temporal network fingerprint is incompatible."
        )
    if identity.timetable_fingerprint != assignment:
        raise AssignmentCompatibilityError(
            "direct temporal timetable fingerprint is incompatible."
        )
    if identity.measurement_mapping_fingerprint != measurement_mapping_fingerprint(
        spec
    ):
        raise AssignmentCompatibilityError(
            "direct temporal measurement mapping is incompatible."
        )
    if np.dtype(identity.numeric_dtype) != np.dtype(inputs.base_link_cost.dtype):
        raise AssignmentCompatibilityError(
            "direct temporal numeric dtype is incompatible."
        )


def _positive_boarding_context(
    *,
    checkpoint_root: str | Path,
    inputs: AssignmentInputs,
    spec,
    canonical_index: CanonicalAssignmentIndex,
    identity: AssignmentArtifactIdentity,
    observations: object,
    mapping_info: MappingInfo | None,
    fixed_zero_reasons_by_full_index: Mapping[int, str] | None,
) -> PositiveBoardingPreflightContext:
    return PositiveBoardingPreflightContext(
        canonical_index=canonical_index,
        observations=observations,
        report_path=(
            Path(checkpoint_root)
            / identity.fingerprint
            / "positive_boarding_support_preflight.json"
        ),
        mapping_info=mapping_info,
        fixed_zero_reasons_by_full_index=fixed_zero_reasons_by_full_index,
        canonical_supported_measurement_rows=boarding_access_supported_measurement_rows(
            active_origin_nodes=inputs.od_origin_node,
            graph_link_tails=inputs.graph.tail,
            measurement_index=spec.measurement_index,
            link_index=spec.link_index,
        ),
    )


def _preflight_manifest_candidate(
    *, checkpoint_root: str | Path, artifact_root: str | Path | None
) -> Path | None:
    """Locate the durable stage manifest without assuming a private layout."""
    candidates = [Path(checkpoint_root).parent / "manifests" / "preflight.json"]
    if artifact_root is not None:
        candidates.append(Path(artifact_root).parent / "manifests" / "preflight.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _operator_cache_directory(checkpoint_directory: Path) -> Path:
    return checkpoint_directory / "temporal_operator_cache"


def _run_origin_positive_boarding_preflight(
    *,
    context: PositiveBoardingPreflightContext,
    reporter: ConstructionProgressReporter,
) -> None:
    reporter.emit(
        phase=ConstructionPhase.MEASUREMENT_SUPPORT_PREFLIGHT,
        status="started",
        force=True,
        completed_units=0,
        total_units=2,
        checkpoint_location=(
            None if context.report_path is None else str(context.report_path)
        ),
        details={"preflight_stage": "canonical_origin_support"},
    )
    report = audit_positive_boarding_support(
        canonical_index=context.canonical_index,
        observations=context.observations,
        supported_measurement_rows=context.canonical_supported_measurement_rows,
        mapping_info=context.mapping_info,
        fixed_zero_reasons_by_full_index=context.fixed_zero_reasons_by_full_index,
    )
    enforce_positive_boarding_support(report, report_path=context.report_path)
    reporter.emit(
        phase=ConstructionPhase.MEASUREMENT_SUPPORT_PREFLIGHT,
        status="running",
        force=True,
        completed_units=1,
        total_units=2,
        checkpoint_location=(
            None if context.report_path is None else str(context.report_path)
        ),
        details={
            "preflight_stage": "canonical_origin_support",
            "positive_boarding_rows": report.positive_boarding_rows,
            "unsupported_positive_boarding_rows": 0,
        },
    )


def _validate_cached_positive_boarding_support(
    *,
    operator: TemporalBlockAssignmentOperator,
    context: PositiveBoardingPreflightContext,
    reporter: ConstructionProgressReporter,
    artifact_directory: Path | None = None,
    operator_cache_directory: Path | None = None,
) -> ConstructionETA | None:
    if artifact_directory is not None and operator_cache_directory is not None:
        artifact_manifest = artifact_directory / "manifest.json"
        operator_manifest = operator_cache_directory / "manifest.json"
        if artifact_manifest.is_file() and operator_manifest.is_file():
            cached_support = load_positive_boarding_support_cache(
                directory=context.report_path.parent / "positive_boarding_support_cache"
                if context.report_path is not None
                else operator_cache_directory.parent / "positive_boarding_support_cache",
                artifact_identity_fingerprint=operator.identity.fingerprint,
                artifact_manifest_sha256=_file_sha256(artifact_manifest),
                operator_cache_manifest_sha256=_file_sha256(operator_manifest),
                canonical_index=context.canonical_index,
                observations=context.observations,
                mapping_info=context.mapping_info,
                fixed_zero_reasons_by_full_index=context.fixed_zero_reasons_by_full_index,
            )
            if cached_support is not None:
                _, cached_report = cached_support
                enforce_positive_boarding_support(
                    cached_report, report_path=context.report_path
                )
                eta = estimate_completed_unit_eta(
                    (), completed_units=1, total_units=1, parallelism=1
                )
                reporter.emit(
                    phase=ConstructionPhase.CACHE_VALIDATION,
                    status="completed",
                    force=True,
                    completed_units=1,
                    total_units=1,
                    current_unit="positive_boarding_support_cache",
                    checkpoint_location=str(operator_cache_directory.parent),
                    predicted_remaining_seconds=0.0,
                    eta_confidence="high",
                    eta_lower_seconds=0.0,
                    eta_upper_seconds=0.0,
                    throughput_units_per_second=1.0,
                    cache_hits=1,
                    cache_misses=0,
                    details={"cache_validation_stage": "positive_boarding_support_cache"},
                )
                reporter.emit(
                    phase=ConstructionPhase.MEASUREMENT_SUPPORT_PREFLIGHT,
                    status="completed",
                    force=True,
                    completed_units=2,
                    total_units=2,
                    checkpoint_location=(
                        None if context.report_path is None else str(context.report_path)
                    ),
                    details={
                        "preflight_stage": "realized_operator_support",
                        "support_cache_hit": True,
                        "positive_boarding_rows": cached_report.positive_boarding_rows,
                        "unsupported_positive_boarding_rows": 0,
                    },
                )
                return eta
    total_blocks = len(operator.blocks)
    reporting_enabled = reporter.sink is not None
    recent_durations: deque[float] = deque(maxlen=16)
    phase_started_at = perf_counter() if reporting_enabled else None
    if reporting_enabled:
        reporter.emit(
            phase=ConstructionPhase.CACHE_VALIDATION,
            status="running",
            force=True,
            completed_units=0,
            total_units=total_blocks,
            current_unit=("block-000000.npz" if total_blocks else None),
            details={"cache_validation_stage": "realized_operator_support"},
        )
    supported: set[int] = set()
    for block_position, block in enumerate(operator.blocks):
        block_started_at = perf_counter() if reporting_enabled else None
        supported.update(int(row) for row in np.unique(block.row_indices))
        if reporting_enabled:
            assert block_started_at is not None
            assert phase_started_at is not None
            duration = max(0.0, perf_counter() - block_started_at)
            recent_durations.append(duration)
            phase_elapsed_seconds = max(
                np.finfo(float).eps, perf_counter() - phase_started_at
            )
            eta = estimate_completed_unit_eta(
                recent_durations,
                completed_units=block_position + 1,
                total_units=total_blocks,
                parallelism=1,
                elapsed_seconds=phase_elapsed_seconds,
            )
            reporter.emit(
                phase=ConstructionPhase.CACHE_VALIDATION,
                status="running",
                completed_units=block_position + 1,
                total_units=total_blocks,
                current_unit=f"block-{block_position:06d}.npz",
                recent_unit_seconds=duration,
                predicted_remaining_seconds=eta.predicted_remaining_seconds,
                eta_confidence=eta.eta_confidence,
                estimated_completion_at_utc=eta.estimated_completion_at_utc,
                eta_reason=eta.eta_reason,
                eta_lower_seconds=eta.eta_lower_seconds,
                eta_upper_seconds=eta.eta_upper_seconds,
                throughput_units_per_second=eta.throughput_units_per_second,
                details={"cache_validation_stage": "realized_operator_support"},
            )
    supported.update(
        int(row) for row in np.flatnonzero(operator.fixed_measurement_offset != 0.0)
    )
    report = audit_positive_boarding_support(
        canonical_index=context.canonical_index,
        observations=context.observations,
        supported_measurement_rows=np.asarray(sorted(supported), dtype=np.int64),
        stage="realized_operator_support",
        mapping_info=context.mapping_info,
        fixed_zero_reasons_by_full_index=context.fixed_zero_reasons_by_full_index,
    )
    enforce_positive_boarding_support(report, report_path=context.report_path)
    if artifact_directory is not None and operator_cache_directory is not None:
        artifact_manifest = artifact_directory / "manifest.json"
        operator_manifest = operator_cache_directory / "manifest.json"
        if artifact_manifest.is_file() and operator_manifest.is_file():
            write_positive_boarding_support_cache(
                directory=(
                    context.report_path.parent / "positive_boarding_support_cache"
                    if context.report_path is not None
                    else operator_cache_directory.parent / "positive_boarding_support_cache"
                ),
                report=report,
                supported_measurement_rows=np.asarray(sorted(supported), dtype=np.int64),
                artifact_identity_fingerprint=operator.identity.fingerprint,
                artifact_manifest_sha256=_file_sha256(artifact_manifest),
                operator_cache_manifest_sha256=_file_sha256(operator_manifest),
                canonical_index=context.canonical_index,
                observations=context.observations,
                mapping_info=context.mapping_info,
                fixed_zero_reasons_by_full_index=context.fixed_zero_reasons_by_full_index,
            )
    final_eta: ConstructionETA | None = None
    if reporting_enabled:
        assert phase_started_at is not None
        final_eta = estimate_completed_unit_eta(
            recent_durations,
            completed_units=total_blocks,
            total_units=total_blocks,
            parallelism=1,
            elapsed_seconds=max(0.0, perf_counter() - phase_started_at),
        )
        reporter.emit(
            phase=ConstructionPhase.CACHE_VALIDATION,
            status="completed",
            force=True,
            completed_units=total_blocks,
            total_units=total_blocks,
            recent_unit_seconds=(recent_durations[-1] if recent_durations else None),
            predicted_remaining_seconds=final_eta.predicted_remaining_seconds,
            eta_confidence=final_eta.eta_confidence,
            estimated_completion_at_utc=final_eta.estimated_completion_at_utc,
            eta_reason=final_eta.eta_reason,
            eta_lower_seconds=final_eta.eta_lower_seconds,
            eta_upper_seconds=final_eta.eta_upper_seconds,
            throughput_units_per_second=final_eta.throughput_units_per_second,
            details={"cache_validation_stage": "realized_operator_support"},
        )
    reporter.emit(
        phase=ConstructionPhase.MEASUREMENT_SUPPORT_PREFLIGHT,
        status="completed",
        force=True,
        completed_units=2,
        total_units=2,
        checkpoint_location=(
            None if context.report_path is None else str(context.report_path)
        ),
        details={
            "preflight_stage": "realized_operator_support",
            "positive_boarding_rows": report.positive_boarding_rows,
            "unsupported_positive_boarding_rows": 0,
        },
    )
    return final_eta


def _finalize_temporal_blocks(
    *,
    construction: ShardedConstructionResult,
    canonical_index: CanonicalAssignmentIndex,
    identity: AssignmentArtifactIdentity,
    deadline: ConstructionDeadline | None = None,
    reporter: ConstructionProgressReporter | None = None,
    checkpoint_directory: Path | None = None,
) -> TemporalBlockAssignmentOperator:
    measurement_intervals = tuple(
        item.interval_id for item in canonical_index.measurements
    )
    departure_by_column = {
        int(cell.operator_column): cell.departure_interval_id
        for cell in canonical_index.demand_cells
        if cell.operator_column is not None
    }
    blocks: list[TemporalSparseBlock] = []
    offset = np.zeros(
        canonical_index.number_of_measurements, dtype=np.dtype(identity.numeric_dtype)
    )
    retained_l1 = 0.0
    # Only the recent tail is needed for deadline prediction and progress ETA;
    # retaining every shard duration would make reporting memory grow with the
    # number of temporal shards.
    recent_seconds: deque[float] = deque(maxlen=32)
    expected = construction.manifest.expected_shards
    reporting_enabled = reporter is not None and reporter.sink is not None
    phase_started_at = perf_counter() if reporting_enabled else None
    if checkpoint_directory is not None:
        fragments = checkpoint_directory / "temporal_fragments"
        if fragments.exists():
            for abandoned in fragments.glob(".*.tmp"):
                abandoned.unlink(missing_ok=True)
    for shard_position, shard_identity in enumerate(expected):
        predicted = (
            float(np.mean(tuple(recent_seconds)[-3:])) if recent_seconds else None
        )
        if deadline is not None and not deadline.may_start(predicted):
            raise deadline_stop(
                deadline,
                phase=ConstructionPhase.TEMPORAL_BLOCK_ASSEMBLY,
                reason="next temporal shard cannot be assembled within the safe deadline",
                completed_units=shard_position,
                total_units=len(expected),
                next_resumable_position=shard_identity.key,
                checkpoint_location=(
                    None
                    if checkpoint_directory is None
                    else str(checkpoint_directory)
                ),
                checkpoint_reusable=True,
                predicted_next_seconds=predicted,
            )
        shard_started = deadline.clock() if deadline is not None else perf_counter()
        fragment = (
            None
            if checkpoint_directory is None
            else _fragment_path(checkpoint_directory, shard_identity.key)
        )
        fragment_reused = False
        local_blocks = None
        if fragment is not None and fragment.exists():
            try:
                local_blocks, offset_rows, offset_values = _load_temporal_fragment(
                    fragment,
                    shard_key=shard_identity.key,
                    identity=identity,
                    number_of_measurements=canonical_index.number_of_measurements,
                    number_of_demand_cells=canonical_index.number_of_demand_cells,
                )
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                quarantine = fragment.with_name(
                    f"{fragment.name}.invalid-{uuid.uuid4().hex}"
                )
                os.replace(fragment, quarantine)
            else:
                fragment_reused = True
        if local_blocks is None:
            loaded = load_sparse_shard(
                shard_path(construction.directory, shard_identity),
                expected_provenance_hash=construction.manifest.provenance_hash,
            )
            coo = loaded.matrix.tocoo()
            global_rows = loaded.row_indices[coo.row]
            local_triplets: dict[
                TemporalBlockKey, tuple[list[int], list[int], list[float]]
            ] = {}
            for row, column, value in zip(
                global_rows, coo.col, coo.data, strict=True
            ):
                key = TemporalBlockKey(
                    measurement_intervals[int(row)],
                    departure_by_column[int(column)],
                )
                rows, columns, values = local_triplets.setdefault(
                    key, ([], [], [])
                )
                rows.append(int(row))
                columns.append(int(column))
                values.append(float(value))
            local_blocks = tuple(
                TemporalSparseBlock(
                    key=key,
                    row_indices=np.asarray(values[0], dtype=np.int32),
                    column_indices=np.asarray(values[1], dtype=np.int32),
                    values=np.asarray(
                        values[2], dtype=np.dtype(identity.numeric_dtype)
                    ),
                    number_of_measurements=canonical_index.number_of_measurements,
                    number_of_demand_cells=canonical_index.number_of_demand_cells,
                )
                for key, values in sorted(local_triplets.items())
            )
            offset_rows = loaded.row_indices[loaded.fixed_offset_indices]
            offset_values = loaded.fixed_offset_values
            if checkpoint_directory is not None:
                _save_temporal_fragment(
                    directory=checkpoint_directory,
                    shard_key=shard_identity.key,
                    identity=identity,
                    blocks=local_blocks,
                    offset_rows=offset_rows,
                    offset_values=offset_values,
                )
        blocks.extend(local_blocks)
        retained_l1 += sum(
            float(np.sum(np.abs(block.values))) for block in local_blocks
        )
        if offset_rows.size:
            np.add.at(offset, offset_rows, offset_values)
        now = deadline.clock() if deadline is not None else perf_counter()
        recent_seconds.append(max(0.0, now - shard_started))
        if reporter is not None:
            eta = None
            if reporting_enabled:
                assert phase_started_at is not None
                eta = estimate_completed_unit_eta(
                    recent_seconds,
                    completed_units=shard_position + 1,
                    total_units=len(expected),
                    parallelism=1,
                    elapsed_seconds=max(0.0, perf_counter() - phase_started_at),
                )
            reporter.emit(
                phase=ConstructionPhase.TEMPORAL_BLOCK_ASSEMBLY,
                status="running",
                force=(shard_position == 0 or shard_position + 1 == len(expected)),
                completed_units=shard_position + 1,
                total_units=len(expected),
                current_unit=shard_identity.key,
                recent_unit_seconds=recent_seconds[-1],
                predicted_remaining_seconds=(
                    None if eta is None else eta.predicted_remaining_seconds
                ),
                eta_confidence=("unavailable" if eta is None else eta.eta_confidence),
                estimated_completion_at_utc=(
                    None if eta is None else eta.estimated_completion_at_utc
                ),
                eta_reason=(None if eta is None else eta.eta_reason),
                eta_lower_seconds=(None if eta is None else eta.eta_lower_seconds),
                eta_upper_seconds=(None if eta is None else eta.eta_upper_seconds),
                throughput_units_per_second=(
                    None if eta is None else eta.throughput_units_per_second
                ),
                checkpoint_location=(
                    None
                    if checkpoint_directory is None
                    else str(checkpoint_directory)
                ),
                cache_hits=int(fragment_reused),
                cache_misses=int(not fragment_reused),
            )
        if deadline is not None and deadline.expired:
            raise deadline_stop(
                deadline,
                phase=ConstructionPhase.TEMPORAL_BLOCK_ASSEMBLY,
                reason="deadline expired after assembling a temporal shard",
                completed_units=shard_position + 1,
                total_units=len(expected),
                next_resumable_position=(
                    expected[shard_position + 1].key
                    if shard_position + 1 < len(expected)
                    else None
                ),
                checkpoint_location=(
                    None
                    if checkpoint_directory is None
                    else str(checkpoint_directory)
                ),
                checkpoint_reusable=True,
                predicted_next_seconds=recent_seconds[-1],
            )
    block_tuple = tuple(blocks)
    nonzeros = sum(block.nonzero_entries for block in block_tuple)
    source = construction
    return TemporalBlockAssignmentOperator(
        canonical_index=canonical_index,
        identity=identity,
        blocks=block_tuple,
        fixed_measurement_offset=offset,
        diagnostics=TemporalBlockConstructionDiagnostics(
            construction_seconds=source.total_seconds,
            nonzero_entries=nonzeros,
            retained_l1_mass=retained_l1,
            removed_l1_mass=0.0,
            zero_tolerance=float(source.manifest.provenance["zero_tolerance"]),
            columns_processed=canonical_index.number_of_demand_cells,
            compilation_count=int(source.compilation_seconds > 0.0),
            compilation_seconds=source.compilation_seconds,
            execution_seconds=source.dispatch_seconds + source.synchronization_seconds,
            device_transfer_seconds=source.transfer_seconds,
            num_chunks=source.dispatch_count,
            chunk_shape=(
                source.manifest.od_chunk_size,
                source.plan.maximum_shard_measurements,
            ),
        ),
    )


def prepare_direct_scheduled_temporal_operator(
    *,
    checkpoint_root: str | Path,
    artifact_root: str | Path | None,
    inputs: AssignmentInputs,
    routing: DirectFixedRoutingSource,
    spec,
    compact_layout: CompactODAssignmentLayout,
    canonical_index: CanonicalAssignmentIndex,
    observations: object,
    identity: AssignmentArtifactIdentity,
    assignment_fingerprint: str,
    od_layout_fingerprint: str,
    config: ShardedConstructionConfig | None = None,
    progress: DirectTemporalProgressCallback | None = None,
    deadline: ConstructionDeadline | None = None,
    reporter: ConstructionProgressReporter | None = None,
    support_timing_callback: GroupSupportTimingCallback | None = None,
    measurement_info: MappingInfo | None = None,
    fixed_zero_reasons_by_full_index: Mapping[int, str] | None = None,
) -> DirectScheduledTemporalConstructionResult:
    """Build or resume direct measurement shards, then publish temporal blocks."""
    legacy_progress = progress if deadline is None and reporter is None else None
    control = ConstructionDeadline.unlimited() if deadline is None else deadline
    events = (
        ConstructionProgressReporter(control, None if legacy_progress else progress)
        if reporter is None
        else reporter
    )
    _validate_inputs(
        inputs=inputs,
        routing=routing,
        spec=spec,
        canonical_index=canonical_index,
        identity=identity,
    )
    preflight = _positive_boarding_context(
        checkpoint_root=checkpoint_root,
        inputs=inputs,
        spec=spec,
        canonical_index=canonical_index,
        identity=identity,
        observations=observations,
        mapping_info=measurement_info,
        fixed_zero_reasons_by_full_index=fixed_zero_reasons_by_full_index,
    )
    _run_origin_positive_boarding_preflight(context=preflight, reporter=events)
    checkpoint_directory = Path(checkpoint_root) / identity.fingerprint
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    operator_cache_directory = _operator_cache_directory(checkpoint_directory)
    preflight_manifest = _preflight_manifest_candidate(
        checkpoint_root=checkpoint_root, artifact_root=artifact_root
    )
    artifact_directory = (
        None
        if artifact_root is None
        else temporal_block_cache_path(artifact_root, identity)
    )
    if artifact_directory is not None and artifact_directory.exists():
        try:
            operator = load_temporal_block_operator(
                artifact_directory,
                expected_identity=identity,
                expected_canonical_index=canonical_index,
                reporter=events,
                validated_cache_directory=operator_cache_directory,
                preflight_manifest=preflight_manifest,
            )
        except (AssignmentCompatibilityError, ValueError, KeyError, OSError):
            quarantine = artifact_directory.with_name(
                f"{artifact_directory.name}.invalid-{uuid.uuid4().hex}"
            )
            os.replace(artifact_directory, quarantine)
        else:
            _validate_cached_positive_boarding_support(
                operator=operator,
                context=preflight,
                reporter=events,
                artifact_directory=artifact_directory,
                operator_cache_directory=operator_cache_directory,
            )
            return DirectScheduledTemporalConstructionResult(
                operator=operator,
                checkpoint_directory=checkpoint_directory,
                artifact_directory=artifact_directory,
                source=None,
                temporal_artifact_reused=True,
                finalization_seconds=0.0,
            )
    if not control.may_start():
        raise deadline_stop(
            control,
            phase=ConstructionPhase.SUPPORT_DISCOVERY,
            reason="deadline reached before support discovery and planning",
            checkpoint_location=str(checkpoint_directory),
            artifact_location=(
                None if artifact_directory is None else str(artifact_directory)
            ),
            checkpoint_reusable=checkpoint_directory.exists(),
        )
    source = prepare_sharded_fixed_routing_measurement_operator(
        directory=checkpoint_directory,
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact_layout,
        assignment_fingerprint=assignment_fingerprint,
        od_layout_fingerprint=od_layout_fingerprint,
        config=config,
        progress=legacy_progress,
        deadline=None if legacy_progress is not None else control,
        reporter=None if legacy_progress is not None else events,
        scientific_identity={
            "canonical_index_fingerprint": identity.canonical_index_fingerprint,
            "temporal_discretization_fingerprint": (
                identity.temporal_discretization_fingerprint
            ),
            "route_choice_fingerprint": identity.route_choice_fingerprint,
            "departure_choice_fingerprint": identity.departure_choice_fingerprint,
            "feasibility_fingerprint": identity.feasibility_fingerprint,
            "coefficient_policy_fingerprint": (
                identity.coefficient_policy_fingerprint
            ),
            "numeric_dtype": identity.numeric_dtype,
            "assignment_contract_schema_version": identity.schema_version,
        },
        positive_boarding_preflight=preflight,
        support_timing_callback=support_timing_callback,
    )
    predicted_finalization = (
        source.total_seconds / max(1, source.plan.num_shards)
        * source.plan.num_shards
        * 0.1
    )
    if not control.may_start(predicted_finalization):
        raise deadline_stop(
            control,
            phase=ConstructionPhase.TEMPORAL_BLOCK_ASSEMBLY,
            reason="temporal-block assembly cannot start within the safe deadline",
            completed_units=source.plan.num_shards,
            total_units=source.plan.num_shards,
            checkpoint_location=str(checkpoint_directory),
            artifact_location=(
                None if artifact_directory is None else str(artifact_directory)
            ),
            checkpoint_reusable=True,
            predicted_next_seconds=predicted_finalization,
        )
    events.emit(
        phase=ConstructionPhase.TEMPORAL_BLOCK_ASSEMBLY,
        status="started",
        force=True,
        completed_units=0,
        total_units=source.plan.num_shards,
        checkpoint_location=str(checkpoint_directory),
    )
    finalization_started = perf_counter()
    operator = _finalize_temporal_blocks(
        construction=source,
        canonical_index=canonical_index,
        identity=identity,
        deadline=control,
        reporter=events,
        checkpoint_directory=checkpoint_directory,
    )
    if artifact_directory is not None:
        if not control.may_start():
            raise deadline_stop(
                control,
                phase=ConstructionPhase.PERSISTENCE,
                reason="deadline reached before final artifact persistence",
                completed_units=source.plan.num_shards,
                total_units=source.plan.num_shards,
                checkpoint_location=str(checkpoint_directory),
                artifact_location=str(artifact_directory),
                checkpoint_reusable=True,
            )
        save_temporal_block_operator(
            artifact_directory,
            operator,
            deadline=control,
            reporter=events,
            checkpoint_location=checkpoint_directory,
        )
        # The freshly validated in-memory operator is the authoritative source
        # for the first compact-cache materialization.  Persisting this cache
        # here means that a subsequent process can open packed arrays directly
        # instead of re-reading and re-validating every source block.
        artifact_manifest_path = artifact_directory / "manifest.json"
        artifact_manifest = json.loads(
            artifact_manifest_path.read_text(encoding="utf-8")
        )
        operator_cache_directory.mkdir(parents=True, exist_ok=True)
        certificate_path = operator_cache_directory / "validation_certificate.json"
        _atomic_json_write(
            certificate_path,
            {
                "complete": True,
                "schema_version": PREFLIGHT_ADOPTION_SCHEMA_VERSION,
                "validator_version": TEMPORAL_OPERATOR_VALIDATOR_VERSION,
                "provenance": {
                    "source": "full_validation",
                    "preflight_manifest_sha256": None,
                    "preflight_completed_phase": None,
                    "legacy_completed_preflight": False,
                },
                "artifact_identity_fingerprint": identity.fingerprint,
                "artifact_manifest_sha256": _file_sha256(artifact_manifest_path),
                "canonical_index_fingerprint": canonical_index.artifact_fingerprint,
                "binding_fingerprint": canonical_index.binding_fingerprint,
                "number_of_blocks": len(artifact_manifest.get("blocks", [])),
                "fixed_measurement_offset_hash": artifact_manifest.get(
                    "fixed_measurement_offset_hash"
                ),
            },
        )
        _materialize_operator_cache_from_operator(
            operator=operator,
            cache_directory=operator_cache_directory,
            artifact_manifest_sha256=_file_sha256(artifact_manifest_path),
            certificate_path=certificate_path,
            provenance_source="full_validation",
            reporter=events,
        )
    finalization_seconds = max(0.0, perf_counter() - finalization_started)
    if control.expired:
        raise deadline_stop(
            control,
            phase=ConstructionPhase.PERSISTENCE,
            reason="deadline expired during indivisible finalization or persistence",
            completed_units=source.plan.num_shards,
            total_units=source.plan.num_shards,
            checkpoint_location=str(checkpoint_directory),
            artifact_location=(
                None if artifact_directory is None else str(artifact_directory)
            ),
            checkpoint_reusable=True,
        )
    events.emit(
        phase=ConstructionPhase.COMPLETED,
        status="completed",
        force=True,
        completed_units=source.plan.num_shards,
        total_units=source.plan.num_shards,
        checkpoint_location=str(checkpoint_directory),
    )
    return DirectScheduledTemporalConstructionResult(
        operator=operator,
        checkpoint_directory=checkpoint_directory,
        artifact_directory=artifact_directory,
        source=source,
        temporal_artifact_reused=False,
        finalization_seconds=finalization_seconds,
    )


def activate_direct_scheduled_temporal_operator(
    *,
    mode: DirectScheduledActivationMode,
    expected_evaluations: int,
    construction_seconds: float | None,
    reference_evaluation_seconds: float,
    operator_evaluation_seconds: float,
    checkpoint_root: str | Path,
    artifact_root: str | Path,
    inputs: AssignmentInputs,
    routing_factory: Callable[[], FixedRoutingInputs],
    theta: float,
    spec,
    compact_layout: CompactODAssignmentLayout,
    canonical_index: CanonicalAssignmentIndex,
    observations: object,
    identity: AssignmentArtifactIdentity,
    assignment_fingerprint: str,
    od_layout_fingerprint: str,
    config: ShardedConstructionConfig | None = None,
    progress: DirectTemporalProgressCallback | None = None,
    support_timing_callback: GroupSupportTimingCallback | None = None,
    deadline: ConstructionDeadline | None = None,
    time_budget_seconds: float | None = None,
    safety_margin_seconds: float = 0.0,
    progress_interval_seconds: float = 1.0,
    predicted_routing_seconds: float | None = None,
    bounded_routing_factory: Callable[
        [ConstructionDeadline], FixedRoutingInputs
    ]
    | None = None,
    routing_preparation_config: FixedRoutingPreparationConfig | None = None,
    measurement_info: MappingInfo | None = None,
    fixed_zero_reasons_by_full_index: Mapping[int, str] | None = None,
) -> DirectScheduledActivationResult:
    """Reuse a valid artifact or build only when the requested policy permits it.

    Cache validation precedes the cost decision and routing construction.  Thus a
    fresh process can activate a valid artifact even when no construction-time
    estimate is available, while a declined automatic decision retains the
    caller's existing loader without paying routing-preparation cost.
    """
    if deadline is not None and time_budget_seconds is not None:
        raise ValueError("provide deadline or time_budget_seconds, not both.")
    control = (
        ConstructionDeadline.from_budget(
            time_budget_seconds, safety_margin_seconds=safety_margin_seconds
        )
        if deadline is None
        else deadline
    )
    reporter = ConstructionProgressReporter(
        control, progress, minimum_interval_seconds=progress_interval_seconds
    )
    if mode not in ("off", "auto", "direct"):
        raise ValueError("mode must be 'off', 'auto', or 'direct'.")
    if expected_evaluations < 0:
        raise ValueError("expected_evaluations must be nonnegative.")
    for name, value in (
        ("reference_evaluation_seconds", reference_evaluation_seconds),
        ("operator_evaluation_seconds", operator_evaluation_seconds),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if construction_seconds is not None and (
        not np.isfinite(construction_seconds) or construction_seconds < 0.0
    ):
        raise ValueError("construction_seconds must be finite and nonnegative.")
    _validate_identity_inputs(
        inputs=inputs,
        spec=spec,
        canonical_index=canonical_index,
        identity=identity,
    )
    if (
        identity.route_choice_fingerprint
        != fixed_routing_route_choice_fingerprint(theta)
    ):
        raise AssignmentCompatibilityError(
            "direct temporal activation theta is incompatible."
        )
    checkpoint_directory = Path(checkpoint_root) / identity.fingerprint
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    operator_cache_directory = _operator_cache_directory(checkpoint_directory)
    preflight_manifest = _preflight_manifest_candidate(
        checkpoint_root=checkpoint_root, artifact_root=artifact_root
    )
    if mode == "off":
        decision = DirectScheduledActivationDecision(
            mode=mode,
            activated=False,
            cache_reused=False,
            reason="explicitly disabled",
            expected_evaluations=expected_evaluations,
            break_even_evaluations=None,
        )
        return DirectScheduledActivationResult(None, decision, None)

    preflight = _positive_boarding_context(
        checkpoint_root=checkpoint_root,
        inputs=inputs,
        spec=spec,
        canonical_index=canonical_index,
        identity=identity,
        observations=observations,
        mapping_info=measurement_info,
        fixed_zero_reasons_by_full_index=fixed_zero_reasons_by_full_index,
    )
    _run_origin_positive_boarding_preflight(context=preflight, reporter=reporter)

    artifact_directory = temporal_block_cache_path(artifact_root, identity)
    reporter.emit(
        phase=ConstructionPhase.CACHE_VALIDATION,
        status="started",
        force=True,
        current_unit=str(artifact_directory),
    )
    if artifact_directory.exists():
        try:
            cached = load_temporal_block_operator(
                artifact_directory,
                expected_identity=identity,
                expected_canonical_index=canonical_index,
                reporter=reporter,
                validated_cache_directory=operator_cache_directory,
                preflight_manifest=preflight_manifest,
            )
        except (AssignmentCompatibilityError, ValueError, KeyError, OSError):
            cached = None
        if cached is not None:
            cache_eta = _validate_cached_positive_boarding_support(
                operator=cached,
                context=preflight,
                reporter=reporter,
                artifact_directory=artifact_directory,
                operator_cache_directory=operator_cache_directory,
            )
            reporter.emit(
                phase=ConstructionPhase.CACHE_VALIDATION,
                status="completed",
                force=True,
                completed_units=len(cached.blocks),
                total_units=len(cached.blocks),
                current_unit=(
                    None
                    if not cached.blocks
                    else f"block-{len(cached.blocks) - 1:06d}.npz"
                ),
                predicted_remaining_seconds=(
                    None
                    if cache_eta is None
                    else cache_eta.predicted_remaining_seconds
                ),
                eta_confidence=(
                    "unavailable" if cache_eta is None else cache_eta.eta_confidence
                ),
                estimated_completion_at_utc=(
                    None
                    if cache_eta is None
                    else cache_eta.estimated_completion_at_utc
                ),
                eta_reason=None if cache_eta is None else cache_eta.eta_reason,
                eta_lower_seconds=(
                    None if cache_eta is None else cache_eta.eta_lower_seconds
                ),
                eta_upper_seconds=(
                    None if cache_eta is None else cache_eta.eta_upper_seconds
                ),
                throughput_units_per_second=(
                    None
                    if cache_eta is None
                    else cache_eta.throughput_units_per_second
                ),
                cache_hits=1,
                cache_misses=0,
                details={"cache_validation_stage": "realized_operator_support"},
            )
            decision = DirectScheduledActivationDecision(
                mode=mode,
                activated=True,
                cache_reused=True,
                reason="valid persistent artifact",
                expected_evaluations=expected_evaluations,
                break_even_evaluations=0.0,
            )
            return DirectScheduledActivationResult(
                DirectScheduledGravityOperator(cached, theta), decision, None
            )
    reporter.emit(
        phase=ConstructionPhase.CACHE_VALIDATION,
        status="completed",
        force=True,
        cache_hits=0,
        cache_misses=1,
    )

    saving = reference_evaluation_seconds - operator_evaluation_seconds
    break_even = (
        None
        if construction_seconds is None or saving <= 0.0
        else construction_seconds / saving
    )
    justified = (
        mode == "direct"
        or (
            construction_seconds is not None
            and saving > 0.0
            and expected_evaluations * saving > construction_seconds
        )
    )
    if not justified:
        reason = (
            "construction cost is unknown"
            if construction_seconds is None
            else "expected end-to-end saving does not exceed construction cost"
        )
        decision = DirectScheduledActivationDecision(
            mode=mode,
            activated=False,
            cache_reused=False,
            reason=reason,
            expected_evaluations=expected_evaluations,
            break_even_evaluations=break_even,
        )
        return DirectScheduledActivationResult(None, decision, None)

    checkpoint_directory = Path(checkpoint_root) / identity.fingerprint
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    for abandoned in checkpoint_directory.glob(".routing.npz.*.tmp"):
        abandoned.unlink(missing_ok=True)
    routing: DirectFixedRoutingSource | None = None
    routing_checkpoint = _routing_checkpoint_path(checkpoint_directory)
    if routing_preparation_config is None and routing_checkpoint.exists():
        try:
            routing = _load_routing_checkpoint(
                checkpoint_directory=checkpoint_directory,
                inputs=inputs,
                identity=identity,
                theta=theta,
            )
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            quarantine = routing_checkpoint.with_name(
                f"{routing_checkpoint.name}.invalid-{uuid.uuid4().hex}"
            )
            os.replace(routing_checkpoint, quarantine)
    routing_reused = routing is not None
    if routing is None and not control.may_start(predicted_routing_seconds):
        stopped = deadline_stop(
            control,
            phase=ConstructionPhase.ROUTING_PREPARATION,
            reason="routing preparation cannot start within the safe deadline",
            checkpoint_location=str(checkpoint_directory),
            artifact_location=str(artifact_directory),
            checkpoint_reusable=False,
            predicted_next_seconds=predicted_routing_seconds,
        ).termination
        reporter.terminal(stopped)
        decision = DirectScheduledActivationDecision(
            mode=mode,
            activated=False,
            cache_reused=False,
            reason=stopped.reason,
            expected_evaluations=expected_evaluations,
            break_even_evaluations=break_even,
        )
        return DirectScheduledActivationResult(None, decision, None, stopped)

    if routing is None and routing_preparation_config is not None:
        routing_plan = plan_fixed_routing_shards(
            inputs=inputs, config=routing_preparation_config
        )
        routing_directory = (
            checkpoint_directory / "routing" / routing_plan.plan_fingerprint
        )
        effective_routing_config = replace(
            routing_preparation_config,
            checkpoint_directory=routing_directory,
            cache_directory=routing_directory / "batches",
        )

        def routing_progress(event: FixedRoutingShardProgress) -> None:
            eta = estimate_completed_unit_eta(
                (),
                completed_units=event.completed_shards,
                total_units=event.total_shards,
                parallelism=max(1, event.admitted_worker_count),
                elapsed_seconds=event.elapsed_seconds,
            )
            if event.eta_confidence != "unavailable":
                eta_confidence = event.eta_confidence
                predicted_remaining = event.estimated_remaining_seconds
                completion_at = event.estimated_completion_at_utc
                eta_reason = event.eta_reason
            else:
                eta_confidence = eta.eta_confidence
                predicted_remaining = eta.predicted_remaining_seconds
                completion_at = eta.estimated_completion_at_utc
                eta_reason = eta.eta_reason
            reporter.emit(
                phase=ConstructionPhase.ROUTING_PREPARATION,
                status=event.status,
                force=event.phase in {
                    "planning_cache_scan",
                    "planning",
                    "terminal",
                    "shard_persistence",
                    "shard_persisted",
                    "batch_persistence",
                    "trace",
                    "lowering",
                    "compilation",
                    "batch_execution",
                    "synchronization",
                    "host_transfer",
                },
                completed_units=event.completed_shards,
                total_units=event.total_shards,
                current_unit=(
                    None
                    if event.shard_index is None
                    else f"routing-shard-{event.shard_index:06d}"
                ),
                recent_unit_seconds=event.recent_shard_seconds,
                predicted_remaining_seconds=predicted_remaining,
                eta_confidence=eta_confidence,
                estimated_completion_at_utc=completion_at,
                eta_reason=eta_reason,
                checkpoint_location=str(routing_directory),
                cache_hits=event.cache_hits,
                cache_misses=event.cache_misses,
                peak_resident_memory_bytes=event.peak_rss_bytes,
                work_stack=[
                    {
                        "name": "destination_groups",
                        "completed_units": event.completed_groups,
                        "total_units": event.total_groups,
                        "current_unit": (
                            None
                            if event.shard_index is None
                            else f"destination-group-shard-{event.shard_index:06d}"
                        ),
                        "status": event.status,
                    },
                    {
                        "name": "routing_shards",
                        "completed_units": event.completed_shards,
                        "total_units": event.total_shards,
                        "current_unit": (
                            None
                            if event.shard_index is None
                            else f"routing-shard-{event.shard_index:06d}"
                        ),
                        "status": event.status,
                    },
                ],
                active_units=[
                    f"routing-shard-{index:06d}"
                    for index in event.current_shard_indices
                ],
                queued_units=event.queued_shards,
                active_workers=event.active_workers,
                requested_workers=effective_routing_config.construction_workers,
                current_unit_elapsed_seconds=event.recent_shard_seconds,
                current_unit_predicted_remaining_seconds=(
                    event.predicted_next_shard_seconds
                ),
                completed_weight=float(event.completed_groups),
                total_weight=float(event.total_groups),
                eta_lower_seconds=getattr(event, "eta_lower_seconds", None),
                eta_upper_seconds=getattr(event, "eta_upper_seconds", None),
                reused_units=event.cache_hits,
                rebuilt_units=event.cache_misses,
                checkpoint_reusable=event.status in {"completed", "cache_reused"},
                details={
                    "routing_phase": event.phase,
                    "routing_status": event.status,
                    "completed_destination_groups": event.completed_groups,
                    "total_destination_groups": event.total_groups,
                    "remaining_routing_shards": event.remaining_shards,
                    "predicted_next_shard_seconds": (
                        event.predicted_next_shard_seconds
                    ),
                    "batch_shard_indices": list(event.batch_shard_indices),
                    "resident_routing_batches": min(
                        event.buffered_shards + event.active_workers,
                        effective_routing_config.resident_shard_limit,
                    ),
                    "eta_reason": eta_reason,
                },
            )

        routing_result = prepare_fixed_routing_sharded(
            inputs=inputs,
            theta=theta,
            config=effective_routing_config,
            absolute_deadline=control.absolute_deadline,
            progress=routing_progress,
            clock=control.clock,
        )
        routing = routing_result.routing
        routing_seconds = routing_result.elapsed_seconds
        routing_reused = (
            routing_result.status == "completed"
            and routing_result.cache_hits == routing_result.routing.num_shards
        )
        if routing_result.status == "deadline_reached":
            next_shard = routing_result.completed_shards
            stopped = deadline_stop(
                control,
                phase=ConstructionPhase.ROUTING_PREPARATION,
                reason=(
                    "deadline reached during sharded routing preparation: "
                    f"{routing_result.deadline_phase}"
                ),
                completed_units=routing_result.completed_shards,
                total_units=routing_result.routing.num_shards,
                next_resumable_position=f"routing-shard-{next_shard:06d}",
                checkpoint_location=str(routing_directory),
                artifact_location=str(artifact_directory),
                checkpoint_reusable=True,
                predicted_next_seconds=(
                    routing_result.predicted_next_shard_seconds
                ),
            ).termination
            reporter.terminal(stopped)
            decision = DirectScheduledActivationDecision(
                mode=mode,
                activated=False,
                cache_reused=False,
                reason=stopped.reason,
                expected_evaluations=expected_evaluations,
                break_even_evaluations=break_even,
            )
            return DirectScheduledActivationResult(None, decision, None, stopped)
        if routing_result.status != "completed":
            raise MemoryError(
                "sharded routing preparation stopped under its resource policy: "
                f"{routing_result.status}"
            )
    elif routing is None:
        routing_started = control.clock()
        with reporter.heartbeat_scope(
            current_unit="routing_factory",
            details={
                "routing_phase": "routing_factory",
                "eta_reason": "opaque routing factory has no shard callback",
            },
        ):
            reporter.emit(
                phase=ConstructionPhase.ROUTING_PREPARATION,
                status="started",
                force=True,
                predicted_remaining_seconds=None,
                eta_confidence="unavailable",
                eta_reason="opaque routing factory has no shard callback",
                cache_hits=0,
                cache_misses=1,
                details={"routing_phase": "routing_factory"},
            )
            routing = (
                bounded_routing_factory(control)
                if bounded_routing_factory is not None
                else routing_factory()
            )
        routing_seconds = max(0.0, control.clock() - routing_started)
        _save_routing_checkpoint(
            checkpoint_directory=checkpoint_directory,
            routing=routing,
            identity=identity,
        )
    elif routing_preparation_config is None:
        routing_seconds = 0.0
    if control.expired:
        stopped = deadline_stop(
            control,
            phase=ConstructionPhase.ROUTING_PREPARATION,
            reason="deadline expired during indivisible routing preparation",
            checkpoint_location=str(checkpoint_directory),
            artifact_location=str(artifact_directory),
            checkpoint_reusable=True,
            predicted_next_seconds=predicted_routing_seconds,
        ).termination
        reporter.terminal(stopped)
        decision = DirectScheduledActivationDecision(
            mode=mode,
            activated=False,
            cache_reused=False,
            reason=stopped.reason,
            expected_evaluations=expected_evaluations,
            break_even_evaluations=break_even,
        )
        return DirectScheduledActivationResult(None, decision, None, stopped)
    reporter.emit(
        phase=ConstructionPhase.ROUTING_PREPARATION,
        status="completed",
        force=True,
        recent_unit_seconds=routing_seconds,
        cache_hits=int(routing_reused),
        cache_misses=int(not routing_reused),
    )
    try:
        construction = prepare_direct_scheduled_temporal_operator(
            checkpoint_root=checkpoint_root,
            artifact_root=artifact_root,
            inputs=inputs,
            routing=routing,
            spec=spec,
            compact_layout=compact_layout,
            canonical_index=canonical_index,
            observations=observations,
            identity=identity,
            assignment_fingerprint=assignment_fingerprint,
            od_layout_fingerprint=od_layout_fingerprint,
            config=config,
            progress=progress,
            deadline=control,
            reporter=reporter,
            support_timing_callback=support_timing_callback,
            measurement_info=measurement_info,
            fixed_zero_reasons_by_full_index=fixed_zero_reasons_by_full_index,
        )
    except ConstructionDeadlineStop as error:
        reporter.terminal(error.termination)
        decision = DirectScheduledActivationDecision(
            mode=mode,
            activated=False,
            cache_reused=False,
            reason=error.termination.reason,
            expected_evaluations=expected_evaluations,
            break_even_evaluations=break_even,
        )
        return DirectScheduledActivationResult(
            None, decision, None, error.termination
        )
    decision = DirectScheduledActivationDecision(
        mode=mode,
        activated=True,
        cache_reused=construction.temporal_artifact_reused,
        reason=("explicit direct construction" if mode == "direct" else "positive expected net saving"),
        expected_evaluations=expected_evaluations,
        break_even_evaluations=break_even,
    )
    return DirectScheduledActivationResult(
        DirectScheduledGravityOperator(construction.operator, theta),
        decision,
        construction,
    )
