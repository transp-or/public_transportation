"""Progressive-fidelity contracts and deterministic routing-work plans."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Callable, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation import __version__

from ..construction_control import estimate_completed_unit_eta
from .demand import generate_gravity_demand
from .objective import (
    GravityObjectiveEvaluation,
    GravityObjectiveProblem,
    _evaluation_from_mean,
    _objective_from_mean,
    gravity_value_and_gradient_adjoint,
)


class GravityFidelityStrategy(str, Enum):
    """Stable selection strategy for expensive routing work."""

    STRATIFIED_NESTED = "stratified_nested"


def _array_digest(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GravityFidelityAnchor:
    """Reusable complete or documented-fidelity predicted-count anchor."""

    raw_parameters: np.ndarray
    measurement_mean: np.ndarray
    problem_identity: str
    operator_identity: str
    routing_cache_identity: str
    fidelity_percent: float
    exact: bool
    measurement_coverage_fraction: float
    created_at_utc: str
    schema_version: int = 1
    package_version: str = __version__
    parameter_digest: str = ""
    anchor_identity: str = ""

    def __post_init__(self) -> None:
        raw = np.array(self.raw_parameters, copy=True)
        mean = np.array(self.measurement_mean, copy=True)
        if raw.ndim != 1 or raw.dtype.kind not in "fiu" or not np.all(np.isfinite(raw)):
            raise ValueError("anchor raw_parameters must be a finite one-dimensional array.")
        if mean.ndim != 1 or mean.dtype.kind not in "fiu" or not np.all(np.isfinite(mean)):
            raise ValueError("anchor measurement_mean must be a finite one-dimensional array.")
        if np.any(mean < 0.0):
            raise ValueError("anchor measurement_mean must be nonnegative.")
        if not self.problem_identity or not self.operator_identity or not self.routing_cache_identity:
            raise ValueError("anchor identities must not be empty.")
        if not np.isfinite(self.fidelity_percent) or not 1.0 <= self.fidelity_percent <= 100.0:
            raise ValueError("anchor fidelity_percent must lie in [1, 100].")
        if not 0.0 <= self.measurement_coverage_fraction <= 1.0:
            raise ValueError("anchor measurement coverage must lie in [0, 1].")
        raw.setflags(write=False)
        mean.setflags(write=False)
        object.__setattr__(self, "raw_parameters", raw)
        object.__setattr__(self, "measurement_mean", mean)
        parameter_digest = _array_digest(raw)
        if self.parameter_digest and self.parameter_digest != parameter_digest:
            raise ValueError("anchor parameter digest mismatch.")
        object.__setattr__(self, "parameter_digest", parameter_digest)
        payload = {
            "schema_version": self.schema_version,
            "parameter_digest": parameter_digest,
            "measurement_digest": _array_digest(mean),
            "problem_identity": self.problem_identity,
            "operator_identity": self.operator_identity,
            "routing_cache_identity": self.routing_cache_identity,
            "fidelity_percent": float(self.fidelity_percent),
            "exact": bool(self.exact),
            "coverage": float(self.measurement_coverage_fraction),
            "created_at_utc": self.created_at_utc,
            "package_version": self.package_version,
        }
        identity = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if self.anchor_identity and self.anchor_identity != identity:
            raise ValueError("anchor identity mismatch.")
        object.__setattr__(self, "anchor_identity", identity)


@dataclass(frozen=True, slots=True)
class GravityFidelityRequest:
    """Requested fraction of expensive routing work, from 1 through 100."""

    effort_percent: float = 100.0
    seed: int = 0
    strategy: GravityFidelityStrategy = GravityFidelityStrategy.STRATIFIED_NESTED
    anchor: GravityFidelityAnchor | None = None
    quality_groups: int = 4

    def __post_init__(self) -> None:
        effort = float(self.effort_percent)
        if not np.isfinite(effort) or not 1.0 <= effort <= 100.0:
            raise ValueError("effort_percent must be finite and lie in [1, 100].")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if not isinstance(self.strategy, GravityFidelityStrategy):
            raise TypeError("strategy must be a GravityFidelityStrategy.")
        if self.anchor is not None and not isinstance(self.anchor, GravityFidelityAnchor):
            raise TypeError("anchor must be a GravityFidelityAnchor or None.")
        if (
            isinstance(self.quality_groups, bool)
            or not isinstance(self.quality_groups, int)
            or self.quality_groups <= 0
        ):
            raise ValueError("quality_groups must be a positive integer.")
        object.__setattr__(self, "effort_percent", effort)


@dataclass(frozen=True, slots=True)
class GravityFidelityShard:
    """Immutable work metadata; constructing a plan never loads shard arrays."""

    shard_id: str
    support_entries: int
    routing_bytes: int
    stratum: str = "all"

    def __post_init__(self) -> None:
        if not self.shard_id:
            raise ValueError("shard_id must not be empty.")
        if self.support_entries <= 0:
            raise ValueError("support_entries must be positive.")
        if self.routing_bytes < 0:
            raise ValueError("routing_bytes must be nonnegative.")
        if not self.stratum:
            raise ValueError("stratum must not be empty.")


@runtime_checkable
class GravityFidelityShardProduct(Protocol):
    """Additive routing product for one fidelity shard."""

    @property
    def shard_id(self) -> str: ...

    def jax_matvec(self, vector: jax.Array) -> jax.Array: ...

    def jax_rmatvec(self, vector: jax.Array) -> jax.Array: ...


@dataclass(frozen=True, slots=True)
class _ArrayFidelityShardProduct:
    shard_id: str
    matrix: jax.Array

    def jax_matvec(self, vector: jax.Array) -> jax.Array:
        return self.matrix @ vector

    def jax_rmatvec(self, vector: jax.Array) -> jax.Array:
        return self.matrix.T @ vector


@dataclass(frozen=True, slots=True)
class _PersistedFidelityShardProduct:
    shard_id: str
    shard_index: int
    operator: object

    def jax_matvec(self, vector: jax.Array) -> jax.Array:
        return self.operator.jax_matvec_fidelity_shard(self.shard_index, vector)

    def jax_rmatvec(self, vector: jax.Array) -> jax.Array:
        return self.operator.jax_rmatvec_fidelity_shard(self.shard_index, vector)


@dataclass(frozen=True, slots=True)
class GravityFidelityContext:
    """Stable operator identity and shard metadata used by the selector."""

    problem_identity: str
    shards: tuple[GravityFidelityShard, ...]
    shard_products: tuple[GravityFidelityShardProduct, ...] | None = None

    def __post_init__(self) -> None:
        if not self.problem_identity:
            raise ValueError("problem_identity must not be empty.")
        if not self.shards:
            raise ValueError("at least one fidelity shard is required.")
        identifiers = tuple(shard.shard_id for shard in self.shards)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("fidelity shard identifiers must be unique.")
        if self.shard_products is not None:
            product_identifiers = tuple(item.shard_id for item in self.shard_products)
            if product_identifiers != identifiers:
                raise ValueError(
                    "fidelity shard products must match metadata order and identifiers."
                )

    def product_by_id(self, shard_id: str) -> GravityFidelityShardProduct:
        if self.shard_products is None:
            raise ValueError("fidelity context does not provide shard products.")
        for product in self.shard_products:
            if product.shard_id == shard_id:
                return product
        raise KeyError(shard_id)


@dataclass(frozen=True, slots=True)
class GravityFidelityPlan:
    """Deterministic nested subset approaching a requested support fraction."""

    requested_effort_percent: float
    effective_effort_percent: float
    selected_shard_ids: tuple[str, ...]
    inclusion_probabilities: tuple[float, ...]
    expansion_weights: tuple[float, ...]
    ordered_shard_ids: tuple[str, ...]
    selected_support_entries: int
    total_support_entries: int
    selected_routing_bytes: int
    total_routing_bytes: int
    selection_seed: int
    selection_identity: str
    nested: bool
    exact: bool


@dataclass(frozen=True, slots=True)
class GravityFidelityDiagnostics:
    requested_effort_percent: float
    effective_effort_percent: float
    selected_shard_count: int
    total_shard_count: int
    selected_support_entries: int
    total_support_entries: int
    selected_routing_bytes: int
    total_routing_bytes: int
    selection_seed: int
    selection_identity: str
    nested: bool
    evaluation_seconds: float
    forward_seconds: float | None
    reverse_seconds: float | None
    quality_groups_completed: int
    anchor_used: bool
    anchor_identity: str | None
    anchor_parameter_distance: float | None
    exact: bool


@dataclass(frozen=True, slots=True)
class GravityApproximationQuality:
    exact: bool
    quality_score: float
    reliability: str
    objective_standard_error: float | None
    objective_relative_standard_error: float | None
    gradient_error_norm_estimate: float | None
    gradient_relative_error_estimate: float | None
    gradient_cosine_lower_estimate: float | None
    predicted_count_relative_error_estimate: float | None
    measurement_coverage_fraction: float
    effective_sample_size: float
    estimator: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must lie in [0, 1].")
        if not 0.0 <= self.measurement_coverage_fraction <= 1.0:
            raise ValueError("measurement_coverage_fraction must lie in [0, 1].")
        if self.effective_sample_size < 0.0:
            raise ValueError("effective_sample_size must be nonnegative.")


@dataclass(frozen=True, slots=True)
class GravityObjectiveGradientResult:
    evaluation: GravityObjectiveEvaluation
    gradient: jax.Array
    fidelity: GravityFidelityDiagnostics
    quality: GravityApproximationQuality


@dataclass(frozen=True, slots=True)
class GravityFidelityProgress:
    phase: str
    requested_effort_percent: float
    effective_effort_percent: float
    selected_shards: int
    completed_shards: int
    elapsed_seconds: float
    predicted_remaining_seconds: float | None
    quality_groups_completed: int
    anchor_used: bool
    deadline_remaining_seconds: float | None
    partial_result: bool
    schema_version: int = 1
    status: str = "running"
    recent_unit_seconds: float | None = None
    eta_confidence: str = "unavailable"
    eta_reason: str | None = None
    estimated_completion_at_utc: str | None = None
    eta_lower_seconds: float | None = None
    eta_upper_seconds: float | None = None
    throughput_units_per_second: float | None = None


@dataclass(frozen=True, slots=True)
class GravityFidelityExecution:
    checkpoint_path: Path | None = None
    resume: bool = False
    absolute_deadline: float | None = None
    progress: Callable[[GravityFidelityProgress], None] | None = None
    cancellation_requested: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_path is not None:
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        if self.resume and self.checkpoint_path is None:
            raise ValueError("resume requires checkpoint_path.")
        if self.absolute_deadline is not None and not np.isfinite(self.absolute_deadline):
            raise ValueError("absolute_deadline must be finite when provided.")


class GravityFidelityEvaluationInterrupted(RuntimeError):
    """A sampled evaluation stopped safely without publishing a result."""

    def __init__(self, phase: str, completed_shards: int, reason: str) -> None:
        super().__init__(
            f"progressive gravity evaluation interrupted during {phase} "
            f"after {completed_shards} shards: {reason}"
        )
        self.phase = phase
        self.completed_shards = completed_shards
        self.reason = reason


def _checkpoint_write(
    path: Path,
    *,
    metadata: dict[str, object],
    arrays: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(
            temporary,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            **{name: np.asarray(value) for name, value in arrays.items()},
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_read(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
                if name != "metadata"
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read progressive-fidelity checkpoint {path}.") from error
    return metadata, arrays


def gravity_fidelity_problem_identity(problem: GravityObjectiveProblem) -> str:
    """Return a stable routing/objective identity without Python object hashes."""
    operator = problem.operator
    payload = {
        "assignment": operator.assignment_fingerprint,
        "graph": operator.graph_fingerprint,
        "mapping": operator.mapping_fingerprint,
        "layout": operator.compact_layout_fingerprint,
        "features": problem.features.fingerprint,
        "parameters": problem.parameter_layout.fingerprint,
        "likelihood": problem.likelihood.value,
        "rho": float(problem.rho),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _operator_identity(problem: GravityObjectiveProblem) -> str:
    operator = problem.operator
    payload = (
        operator.assignment_fingerprint,
        operator.graph_fingerprint,
        operator.mapping_fingerprint,
        operator.compact_layout_fingerprint,
        operator.num_free_od,
        operator.num_measurements,
    )
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _routing_cache_identity(problem: GravityObjectiveProblem) -> str:
    operator = problem.operator
    payload = (
        operator.assignment_fingerprint,
        operator.graph_fingerprint,
        float(operator.theta),
        operator.representation,
    )
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def build_gravity_fidelity_anchor(
    raw_parameters: object,
    *,
    problem: GravityObjectiveProblem,
    result: GravityObjectiveGradientResult,
) -> GravityFidelityAnchor:
    """Create a reusable anchor from an already completed fidelity result."""
    raw = np.asarray(raw_parameters)
    if raw.shape != np.asarray(result.gradient).shape:
        raise ValueError("anchor parameter shape must match the completed gradient.")
    return GravityFidelityAnchor(
        raw_parameters=raw,
        measurement_mean=np.asarray(result.evaluation.measurement_mean),
        problem_identity=gravity_fidelity_problem_identity(problem),
        operator_identity=_operator_identity(problem),
        routing_cache_identity=_routing_cache_identity(problem),
        fidelity_percent=result.fidelity.effective_effort_percent,
        exact=result.fidelity.exact,
        measurement_coverage_fraction=result.quality.measurement_coverage_fraction,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def _validate_anchor(
    anchor: GravityFidelityAnchor, *, problem: GravityObjectiveProblem, raw: jax.Array
) -> None:
    if anchor.problem_identity != gravity_fidelity_problem_identity(problem):
        raise ValueError("anchor and gravity problem identities differ.")
    if anchor.operator_identity != _operator_identity(problem):
        raise ValueError("anchor operator identity mismatch.")
    if anchor.routing_cache_identity != _routing_cache_identity(problem):
        raise ValueError("anchor routing-cache identity mismatch.")
    if anchor.raw_parameters.shape != raw.shape:
        raise ValueError("anchor parameter shape mismatch.")
    if anchor.measurement_mean.shape != (problem.operator.num_measurements,):
        raise ValueError("anchor measurement shape mismatch.")


def _stable_rank(identity: str, seed: int, shard: GravityFidelityShard) -> str:
    encoded = json.dumps(
        (identity, seed, shard.stratum, shard.shard_id),
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_uniform(identity: str, seed: int, shard: GravityFidelityShard) -> float:
    rank = _stable_rank(identity, seed, shard)
    return (int(rank[:16], 16) + 0.5) / float(2**64)


def _stratified_order(
    context: GravityFidelityContext, seed: int
) -> tuple[GravityFidelityShard, ...]:
    strata: dict[str, list[GravityFidelityShard]] = {}
    for shard in context.shards:
        strata.setdefault(shard.stratum, []).append(shard)
    for values in strata.values():
        values.sort(key=lambda shard: _stable_rank(context.problem_identity, seed, shard))
    stratum_order = sorted(
        strata,
        key=lambda value: hashlib.sha256(
            json.dumps((context.problem_identity, seed, value)).encode()
        ).hexdigest(),
    )
    ordered: list[GravityFidelityShard] = []
    position = 0
    while True:
        added = False
        for name in stratum_order:
            values = strata[name]
            if position < len(values):
                ordered.append(values[position])
                added = True
        if not added:
            break
        position += 1
    return tuple(ordered)


def plan_gravity_fidelity(
    fidelity: GravityFidelityRequest, *, context: GravityFidelityContext
) -> GravityFidelityPlan:
    """Select nested stratified Bernoulli work with known inclusion probabilities.

    Every shard in a stratum is included when its stable uniform value is at
    most the requested effort fraction. If that would leave the stratum empty,
    its minimum-uniform shard is included. For a stratum of size ``n`` and
    threshold ``p``, each shard therefore has inclusion probability
    ``p + (1-p)**n / n``. The inverse is its Horvitz--Thompson expansion weight.
    """
    ordered = _stratified_order(context, fidelity.seed)
    total_support = sum(shard.support_entries for shard in ordered)
    total_bytes = sum(shard.routing_bytes for shard in ordered)
    if fidelity.effort_percent == 100.0:
        selected = ordered
        probability_by_id = {shard.shard_id: 1.0 for shard in ordered}
    else:
        threshold = fidelity.effort_percent / 100.0
        strata: dict[str, list[GravityFidelityShard]] = {}
        for shard in ordered:
            strata.setdefault(shard.stratum, []).append(shard)
        selected_ids: set[str] = set()
        probability_by_id: dict[str, float] = {}
        for values in strata.values():
            uniforms = {
                shard.shard_id: _stable_uniform(
                    context.problem_identity, fidelity.seed, shard
                )
                for shard in values
            }
            chosen = [shard for shard in values if uniforms[shard.shard_id] <= threshold]
            if not chosen:
                chosen = [min(values, key=lambda shard: uniforms[shard.shard_id])]
            selected_ids.update(shard.shard_id for shard in chosen)
            inclusion = threshold + (1.0 - threshold) ** len(values) / len(values)
            probability_by_id.update(
                {shard.shard_id: inclusion for shard in values}
            )
        selected = tuple(shard for shard in ordered if shard.shard_id in selected_ids)
    selected_support = sum(shard.support_entries for shard in selected)
    selected_bytes = sum(shard.routing_bytes for shard in selected)
    identity_payload = {
        "schema_version": 1,
        "problem_identity": context.problem_identity,
        "seed": fidelity.seed,
        "strategy": fidelity.strategy.value,
        "ordered": [shard.shard_id for shard in ordered],
        "selected": [shard.shard_id for shard in selected],
    }
    selection_identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return GravityFidelityPlan(
        requested_effort_percent=fidelity.effort_percent,
        effective_effort_percent=100.0 * selected_support / total_support,
        selected_shard_ids=tuple(shard.shard_id for shard in selected),
        inclusion_probabilities=tuple(
            probability_by_id[shard.shard_id] for shard in selected
        ),
        expansion_weights=tuple(
            1.0 / probability_by_id[shard.shard_id] for shard in selected
        ),
        ordered_shard_ids=tuple(shard.shard_id for shard in ordered),
        selected_support_entries=selected_support,
        total_support_entries=total_support,
        selected_routing_bytes=selected_bytes,
        total_routing_bytes=total_bytes,
        selection_seed=fidelity.seed,
        selection_identity=selection_identity,
        nested=True,
        exact=fidelity.effort_percent == 100.0,
    )


def _single_work_context(problem: GravityObjectiveProblem) -> GravityFidelityContext:
    operator = problem.operator
    metrics = operator.metrics
    support = int(getattr(metrics, "nonzero_entries", 0))
    if support <= 0:
        support = max(1, operator.num_free_od * operator.num_measurements)
    return GravityFidelityContext(
        problem_identity=gravity_fidelity_problem_identity(problem),
        shards=(
            GravityFidelityShard(
                shard_id="complete-operator",
                support_entries=support,
                routing_bytes=max(0, int(metrics.stored_bytes)),
            ),
        ),
    )


def build_gravity_fidelity_context(
    problem: GravityObjectiveProblem,
    *,
    maximum_dense_shards: int = 32,
) -> GravityFidelityContext:
    """Build additive shard products without densifying sparse operators.

    Persisted sharded operators expose load-free predicted-work metadata and
    bounded one-shard products. Already-dense public reference operators are
    partitioned by OD columns for testing and small examples. Other sparse
    representations require an explicit caller-supplied context.
    """
    if maximum_dense_shards <= 0:
        raise ValueError("maximum_dense_shards must be positive.")
    operator = problem.operator
    identity = gravity_fidelity_problem_identity(problem)
    statistics = getattr(operator, "fidelity_shard_statistics", None)
    if callable(statistics):
        records = statistics()
        shards = tuple(
            GravityFidelityShard(
                shard_id=str(record["shard_id"]),
                support_entries=int(record["support_entries"]),
                routing_bytes=int(record["routing_bytes"]),
                stratum=str(record["stratum"]),
            )
            for record in records
        )
        products = tuple(
            _PersistedFidelityShardProduct(
                shard_id=str(record["shard_id"]),
                shard_index=int(record["shard_index"]),
                operator=operator,
            )
            for record in records
        )
        return GravityFidelityContext(identity, shards, products)
    if operator.representation != "dense" or not hasattr(operator, "matrix"):
        raise ValueError(
            "automatic progressive fidelity supports persisted sharded or "
            "already-dense operators; provide an explicit additive context "
            "for other representations."
        )
    matrix = jnp.asarray(operator.matrix)
    count = min(maximum_dense_shards, operator.num_free_od)
    column_groups = np.array_split(np.arange(operator.num_free_od), count)
    metadata: list[GravityFidelityShard] = []
    products: list[GravityFidelityShardProduct] = []
    for index, columns in enumerate(column_groups):
        shard_matrix = jnp.zeros_like(matrix).at[:, columns].set(matrix[:, columns])
        identifier = f"dense-columns-{index}"
        support = max(1, int(np.count_nonzero(np.asarray(shard_matrix))))
        metadata.append(
            GravityFidelityShard(
                identifier,
                support_entries=support,
                routing_bytes=int(shard_matrix.nbytes),
                stratum="od-columns",
            )
        )
        products.append(_ArrayFidelityShardProduct(identifier, shard_matrix))
    return GravityFidelityContext(identity, tuple(metadata), tuple(products))


def _quality_group_assignment(
    plan: GravityFidelityPlan,
    fidelity: GravityFidelityRequest,
    context: GravityFidelityContext,
) -> tuple[dict[str, int], int]:
    """Assign selected shards reproducibly to nonempty replicate groups."""
    count = min(fidelity.quality_groups, len(plan.selected_shard_ids))
    if count == 0:
        return {}, 0
    metadata = {shard.shard_id: shard for shard in context.shards}
    ordered = sorted(
        plan.selected_shard_ids,
        key=lambda identifier: hashlib.sha256(
            json.dumps(
                (
                    context.problem_identity,
                    fidelity.seed,
                    "quality-group",
                    metadata[identifier].stratum,
                    identifier,
                ),
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    )
    return {identifier: index % count for index, identifier in enumerate(ordered)}, count


def _relative(error: float, reference: float) -> float:
    return error / max(abs(reference), np.finfo(float).eps)


def _estimated_quality(
    *,
    raw: jax.Array,
    problem: GravityObjectiveProblem,
    evaluation: GravityObjectiveEvaluation,
    gradient: jax.Array,
    group_routed: list[jax.Array],
    group_demand_cotangents: list[jax.Array],
    demand_pullback,
    direct_gradient: jax.Array,
    base_measurement_mean: jax.Array,
    coverage: float,
    plan: GravityFidelityPlan,
) -> GravityApproximationQuality:
    """Estimate sampling uncertainty from shard-level replicate groups.

    Group forward estimates are full Horvitz--Thompson totals and are used to
    reevaluate the nonlinear objective. Group gradient estimates reuse the main
    likelihood cotangent, so their dispersion estimates the linearized gradient
    error around the reported approximate mean. These are diagnostics, not
    mathematical bounds.
    """
    count = len(group_routed)
    warnings: list[str] = []
    if count < 2:
        warnings.append(
            "Fewer than two nonempty quality groups; approximation error cannot be estimated."
        )
        if coverage < 1.0:
            warnings.append("Selected routing work does not cover every calibration measurement.")
        return GravityApproximationQuality(
            exact=False,
            quality_score=0.0,
            reliability="insufficient_sample",
            objective_standard_error=None,
            objective_relative_standard_error=None,
            gradient_error_norm_estimate=None,
            gradient_relative_error_estimate=None,
            gradient_cosine_lower_estimate=None,
            predicted_count_relative_error_estimate=None,
            measurement_coverage_fraction=coverage,
            effective_sample_size=float(count),
            estimator="stratified_replicate_groups_linearized_gradient",
            warnings=tuple(warnings),
        )

    rho = jnp.asarray(problem.rho, dtype=evaluation.measurement_mean.dtype)
    group_means = [
        jnp.maximum(base_measurement_mean + rho * routed, problem.mean_floor)
        for routed in group_routed
    ]
    group_objectives = np.asarray(
        [float(_objective_from_mean(mean, raw, problem)) for mean in group_means]
    )
    objective_se = float(np.std(group_objectives, ddof=1) / np.sqrt(count))
    objective_rse = _relative(objective_se, float(evaluation.objective))

    means = np.stack([np.asarray(mean) for mean in group_means])
    mean_se = np.std(means, axis=0, ddof=1) / np.sqrt(count)
    predicted_relative = float(np.linalg.norm(mean_se)) / max(
        float(np.linalg.norm(np.asarray(evaluation.measurement_mean))),
        np.finfo(float).eps,
    )

    group_gradients = np.stack(
        [
            np.asarray(demand_pullback(cotangent)[0] + direct_gradient)
            for cotangent in group_demand_cotangents
        ]
    )
    gradient_se = np.std(group_gradients, axis=0, ddof=1) / np.sqrt(count)
    gradient_error = float(np.linalg.norm(gradient_se))
    gradient_norm = float(np.linalg.norm(np.asarray(gradient)))
    gradient_relative = gradient_error / max(gradient_norm, np.finfo(float).eps)
    main_gradient = np.asarray(gradient)
    cosines = []
    for group_gradient in group_gradients:
        denominator = np.linalg.norm(main_gradient) * np.linalg.norm(group_gradient)
        if denominator > np.finfo(float).eps:
            cosines.append(float(np.dot(main_gradient, group_gradient) / denominator))
    cosine_lower = None if not cosines else float(np.clip(min(cosines), -1.0, 1.0))

    if coverage < 1.0:
        warnings.append("Selected routing work does not cover every calibration measurement.")
    if predicted_relative > 0.5:
        warnings.append("Predicted-count uncertainty estimate is large.")
    if gradient_relative > 0.5:
        warnings.append("Gradient uncertainty estimate is large.")
    direction_factor = 0.0 if cosine_lower is None else max(0.0, cosine_lower)
    support_factor = np.sqrt(plan.effective_effort_percent / 100.0)
    score = float(
        np.clip(
            support_factor
            * coverage
            * (1.0 / (1.0 + objective_rse))
            * (1.0 / (1.0 + gradient_relative))
            * np.sqrt(direction_factor),
            0.0,
            1.0,
        )
    )
    reliability = "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"
    return GravityApproximationQuality(
        exact=False,
        quality_score=score,
        reliability=reliability,
        objective_standard_error=objective_se,
        objective_relative_standard_error=objective_rse,
        gradient_error_norm_estimate=gradient_error,
        gradient_relative_error_estimate=gradient_relative,
        gradient_cosine_lower_estimate=cosine_lower,
        predicted_count_relative_error_estimate=predicted_relative,
        measurement_coverage_fraction=coverage,
        effective_sample_size=float(count),
        estimator="stratified_replicate_groups_linearized_gradient",
        warnings=tuple(warnings),
    )


def gravity_value_and_gradient_progressive(
    raw_parameters: object,
    *,
    problem: GravityObjectiveProblem,
    fidelity: GravityFidelityRequest,
    context: GravityFidelityContext | None = None,
    execution: GravityFidelityExecution | None = None,
) -> GravityObjectiveGradientResult:
    """Evaluate an exact or sampled additive-routing objective and its gradient.

    Below 100 percent, selected shard contributions are expanded with their
    Horvitz--Thompson weights to estimate the complete predicted-count vector.
    The nonlinear likelihood is evaluated once at that approximate complete
    vector. The reverse calculation uses the identical selected shards and
    weights, so the returned gradient belongs to this documented approximate
    objective. Fixed offsets, regularization, parameter transformations, and
    direct likelihood terms are always evaluated exactly.
    """
    selected_context = context or (
        _single_work_context(problem)
        if fidelity.effort_percent == 100.0
        else build_gravity_fidelity_context(problem)
    )
    expected_identity = gravity_fidelity_problem_identity(problem)
    if selected_context.problem_identity != expected_identity:
        raise ValueError("fidelity context and gravity problem identities differ.")
    plan = plan_gravity_fidelity(fidelity, context=selected_context)
    execution = execution or GravityFidelityExecution()
    started = perf_counter()
    if fidelity.effort_percent == 100.0:
        evaluation, gradient = gravity_value_and_gradient_adjoint(
            raw_parameters, problem=problem
        )
        elapsed = max(0.0, perf_counter() - started)
        forward_seconds = None
        reverse_seconds = None
        coverage = 1.0
        quality_groups_completed = 0
        quality = GravityApproximationQuality(
            exact=True,
            quality_score=1.0,
            reliability="exact",
            objective_standard_error=0.0,
            objective_relative_standard_error=0.0,
            gradient_error_norm_estimate=0.0,
            gradient_relative_error_estimate=0.0,
            gradient_cosine_lower_estimate=1.0,
            predicted_count_relative_error_estimate=0.0,
            measurement_coverage_fraction=1.0,
            effective_sample_size=float(len(plan.selected_shard_ids)),
            estimator="complete_adjoint",
        )
    else:
        if selected_context.shard_products is None:
            raise ValueError(
                "sub-100 progressive evaluation requires additive shard products."
            )
        raw = jnp.asarray(raw_parameters)

        def demand_function(value: jax.Array) -> jax.Array:
            return generate_gravity_demand(
                value,
                features=problem.features,
                parameter_layout=problem.parameter_layout,
            ).demand

        demand, demand_pullback = jax.vjp(demand_function, raw)
        anchor = fidelity.anchor
        if anchor is None:
            routed_demand = demand
            base_measurement_mean = (
                jnp.asarray(problem.rho, dtype=demand.dtype)
                * jnp.asarray(
                    problem.operator.fixed_measurement_offset, dtype=demand.dtype
                )
            )
        else:
            _validate_anchor(anchor, problem=problem, raw=raw)
            anchor_demand = demand_function(
                jnp.asarray(anchor.raw_parameters, dtype=raw.dtype)
            )
            routed_demand = demand - anchor_demand
            base_measurement_mean = jnp.asarray(
                anchor.measurement_mean, dtype=demand.dtype
            )
        quality_assignment, quality_groups_completed = _quality_group_assignment(
            plan, fidelity, selected_context
        )
        checkpoint_identity = {
            "schema_version": 1,
            "package_version": __version__,
            "problem_identity": expected_identity,
            "parameter_digest": _array_digest(raw),
            "requested_effort_percent": fidelity.effort_percent,
            "seed": fidelity.seed,
            "selection_identity": plan.selection_identity,
            "selected_shard_ids": list(plan.selected_shard_ids),
            "expansion_weights": list(plan.expansion_weights),
            "anchor_identity": None if anchor is None else anchor.anchor_identity,
            "quality_groups": quality_groups_completed,
        }

        phase_durations: dict[str, deque[float]] = {
            "forward": deque(maxlen=32),
            "reverse": deque(maxlen=32),
        }

        def emit(phase: str, completed: int, *, partial: bool) -> None:
            if execution.progress is None:
                return
            elapsed_now = max(0.0, perf_counter() - started)
            phase_name = "reverse" if phase.startswith("reverse") else "forward"
            eta = estimate_completed_unit_eta(
                phase_durations[phase_name],
                completed_units=completed,
                total_units=len(plan.selected_shard_ids),
                parallelism=1,
                elapsed_seconds=elapsed_now,
            )
            status = "completed" if phase == "completed" else "running"
            try:
                execution.progress(
                    GravityFidelityProgress(
                        phase=phase,
                        requested_effort_percent=fidelity.effort_percent,
                        effective_effort_percent=plan.effective_effort_percent,
                        selected_shards=len(plan.selected_shard_ids),
                        completed_shards=completed,
                        elapsed_seconds=elapsed_now,
                        predicted_remaining_seconds=eta.predicted_remaining_seconds,
                        quality_groups_completed=quality_groups_completed,
                        anchor_used=anchor is not None,
                        deadline_remaining_seconds=(
                            None
                            if execution.absolute_deadline is None
                            else max(
                                0.0,
                                execution.absolute_deadline - perf_counter(),
                            )
                        ),
                        partial_result=partial,
                        status=status,
                        recent_unit_seconds=(
                            phase_durations[phase_name][-1]
                            if phase_durations[phase_name]
                            else None
                        ),
                        eta_confidence=eta.eta_confidence,
                        eta_reason=eta.eta_reason,
                        estimated_completion_at_utc=eta.estimated_completion_at_utc,
                        eta_lower_seconds=eta.eta_lower_seconds,
                        eta_upper_seconds=eta.eta_upper_seconds,
                        throughput_units_per_second=eta.throughput_units_per_second,
                    )
                )
            except OSError:
                # Progress sinks are observability-only.  Preserve the
                # scientific result when a durable log is unavailable.
                return

        def interrupted(phase: str, completed: int) -> str | None:
            if (
                execution.cancellation_requested is not None
                and execution.cancellation_requested()
            ):
                return "cancelled"
            if (
                execution.absolute_deadline is not None
                and perf_counter() >= execution.absolute_deadline
            ):
                return "deadline_reached"
            return None

        forward_started = perf_counter()
        routed = jnp.zeros((problem.operator.num_measurements,), dtype=demand.dtype)
        group_routed = [jnp.zeros_like(routed) for _ in range(quality_groups_completed)]
        represented = np.zeros(problem.operator.num_measurements, dtype=bool)
        completed_forward = 0
        completed_reverse = 0
        resumed_arrays: dict[str, np.ndarray] = {}
        resumed_stage = "forward"
        if execution.resume:
            assert execution.checkpoint_path is not None
            metadata, resumed_arrays = _checkpoint_read(execution.checkpoint_path)
            for name, expected in checkpoint_identity.items():
                if metadata.get(name) != expected:
                    raise ValueError(
                        f"progressive-fidelity checkpoint {name} mismatch."
                    )
            resumed_stage = str(metadata.get("stage", "forward"))
            completed_forward = int(metadata.get("completed_forward", 0))
            completed_reverse = int(metadata.get("completed_reverse", 0))
            routed = jnp.asarray(resumed_arrays["routed"], dtype=demand.dtype)
            represented = np.asarray(resumed_arrays["represented"], dtype=bool)
            group_routed = [
                jnp.asarray(value, dtype=demand.dtype)
                for value in resumed_arrays["group_routed"]
            ]
            emit("resumed", completed_forward, partial=True)

        def save_checkpoint(
            stage: str,
            *,
            demand_cotangent_value: jax.Array | None = None,
            group_cotangents_value: list[jax.Array] | None = None,
        ) -> None:
            if execution.checkpoint_path is None:
                return
            metadata = {
                **checkpoint_identity,
                "stage": stage,
                "completed_forward": completed_forward,
                "completed_reverse": completed_reverse,
            }
            arrays: dict[str, object] = {
                "routed": routed,
                "represented": represented,
                "group_routed": jnp.stack(group_routed),
            }
            if demand_cotangent_value is not None:
                arrays["demand_cotangent"] = demand_cotangent_value
            if group_cotangents_value is not None:
                arrays["group_demand_cotangents"] = jnp.stack(
                    group_cotangents_value
                )
            _checkpoint_write(execution.checkpoint_path, metadata=metadata, arrays=arrays)

        emit("forward_started", completed_forward, partial=True)
        forward_items = tuple(
            zip(plan.selected_shard_ids, plan.expansion_weights, strict=True)
        )
        if resumed_stage == "forward":
            forward_start = completed_forward
        else:
            forward_start = len(forward_items)
        for item_index in range(forward_start, len(forward_items)):
            shard_started = perf_counter() if execution.progress is not None else 0.0
            reason = interrupted("forward", completed_forward)
            if reason is not None:
                save_checkpoint("forward")
                emit("forward_interrupted", completed_forward, partial=True)
                raise GravityFidelityEvaluationInterrupted(
                    "forward", completed_forward, reason
                )
            shard_id, weight = forward_items[item_index]
            contribution = selected_context.product_by_id(shard_id).jax_matvec(
                routed_demand
            )
            weighted = jnp.asarray(weight, dtype=demand.dtype) * contribution
            routed = routed + weighted
            group = quality_assignment[shard_id]
            group_routed[group] = (
                group_routed[group]
                + jnp.asarray(quality_groups_completed, dtype=demand.dtype) * weighted
            )
            represented |= np.asarray(contribution) != 0.0
            completed_forward += 1
            if execution.progress is not None:
                phase_durations["forward"].append(
                    max(0.0, perf_counter() - shard_started)
                )
            save_checkpoint("forward")
            emit("forward_shard_completed", completed_forward, partial=True)
        jax.block_until_ready(routed)
        forward_seconds = max(0.0, perf_counter() - forward_started)
        rho = jnp.asarray(problem.rho, dtype=demand.dtype)
        mean_unfloored = base_measurement_mean + rho * routed
        mean = jnp.maximum(mean_unfloored, problem.mean_floor)
        mean_gradient = jax.grad(
            lambda value: _objective_from_mean(value, raw, problem)
        )(mean)
        active_mean = (mean_unfloored > problem.mean_floor).astype(mean.dtype)
        measurement_cotangent = active_mean * mean_gradient
        reverse_started = perf_counter()
        demand_cotangent = jnp.zeros_like(demand)
        group_demand_cotangents = [
            jnp.zeros_like(demand) for _ in range(quality_groups_completed)
        ]
        if resumed_stage in {"reverse", "completed"}:
            demand_cotangent = jnp.asarray(
                resumed_arrays["demand_cotangent"], dtype=demand.dtype
            )
            group_demand_cotangents = [
                jnp.asarray(value, dtype=demand.dtype)
                for value in resumed_arrays["group_demand_cotangents"]
            ]
        else:
            completed_reverse = 0
            save_checkpoint(
                "reverse",
                demand_cotangent_value=demand_cotangent,
                group_cotangents_value=group_demand_cotangents,
            )
        emit("reverse_started", completed_reverse, partial=True)
        reverse_items = forward_items
        for item_index in range(completed_reverse, len(reverse_items)):
            shard_started = perf_counter() if execution.progress is not None else 0.0
            reason = interrupted("reverse", completed_reverse)
            if reason is not None:
                save_checkpoint(
                    "reverse",
                    demand_cotangent_value=demand_cotangent,
                    group_cotangents_value=group_demand_cotangents,
                )
                emit("reverse_interrupted", completed_reverse, partial=True)
                raise GravityFidelityEvaluationInterrupted(
                    "reverse", completed_reverse, reason
                )
            shard_id, weight = reverse_items[item_index]
            contribution = selected_context.product_by_id(shard_id).jax_rmatvec(
                measurement_cotangent
            )
            weighted = rho * jnp.asarray(weight, dtype=demand.dtype) * contribution
            demand_cotangent = demand_cotangent + weighted
            group = quality_assignment[shard_id]
            group_demand_cotangents[group] = (
                group_demand_cotangents[group]
                + jnp.asarray(quality_groups_completed, dtype=demand.dtype) * weighted
            )
            completed_reverse += 1
            if execution.progress is not None:
                phase_durations["reverse"].append(
                    max(0.0, perf_counter() - shard_started)
                )
            save_checkpoint(
                "reverse",
                demand_cotangent_value=demand_cotangent,
                group_cotangents_value=group_demand_cotangents,
            )
            emit("reverse_shard_completed", completed_reverse, partial=True)
        demand_gradient = demand_pullback(demand_cotangent)[0]
        direct_gradient = jax.grad(
            lambda parameters: _objective_from_mean(mean, parameters, problem)
        )(raw)
        gradient = demand_gradient + direct_gradient
        jax.block_until_ready(gradient)
        reverse_seconds = max(0.0, perf_counter() - reverse_started)
        evaluation = _evaluation_from_mean(
            raw, mean=mean, demand=demand, problem=problem
        )
        elapsed = max(0.0, perf_counter() - started)
        calibration = np.asarray(problem.calibration_mask, dtype=bool)
        coverage = float(np.count_nonzero(represented & calibration)) / float(
            np.count_nonzero(calibration)
        )
        if anchor is not None:
            coverage = max(coverage, anchor.measurement_coverage_fraction)
        quality = _estimated_quality(
            raw=raw,
            problem=problem,
            evaluation=evaluation,
            gradient=gradient,
            group_routed=group_routed,
            group_demand_cotangents=group_demand_cotangents,
            demand_pullback=demand_pullback,
            direct_gradient=direct_gradient,
            base_measurement_mean=base_measurement_mean,
            coverage=coverage,
            plan=plan,
        )
        save_checkpoint(
            "completed",
            demand_cotangent_value=demand_cotangent,
            group_cotangents_value=group_demand_cotangents,
        )
        emit("completed", len(plan.selected_shard_ids), partial=False)
    diagnostics = GravityFidelityDiagnostics(
        requested_effort_percent=fidelity.effort_percent,
        effective_effort_percent=plan.effective_effort_percent,
        selected_shard_count=len(plan.selected_shard_ids),
        total_shard_count=len(plan.ordered_shard_ids),
        selected_support_entries=plan.selected_support_entries,
        total_support_entries=plan.total_support_entries,
        selected_routing_bytes=plan.selected_routing_bytes,
        total_routing_bytes=plan.total_routing_bytes,
        selection_seed=fidelity.seed,
        selection_identity=plan.selection_identity,
        nested=True,
        evaluation_seconds=elapsed,
        forward_seconds=forward_seconds,
        reverse_seconds=reverse_seconds,
        quality_groups_completed=quality_groups_completed,
        anchor_used=fidelity.anchor is not None and fidelity.effort_percent < 100.0,
        anchor_identity=(
            None
            if fidelity.anchor is None or fidelity.effort_percent == 100.0
            else fidelity.anchor.anchor_identity
        ),
        anchor_parameter_distance=(
            None
            if fidelity.anchor is None or fidelity.effort_percent == 100.0
            else float(
                np.linalg.norm(
                    np.asarray(raw_parameters)
                    - np.asarray(fidelity.anchor.raw_parameters)
                )
            )
        ),
        exact=fidelity.effort_percent == 100.0,
    )
    return GravityObjectiveGradientResult(evaluation, gradient, diagnostics, quality)
