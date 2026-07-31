"""Resumable sparse shards exposed as one bounded-memory linear operator."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np
from scipy import sparse

Array = np.ndarray
SHARDED_OPERATOR_SCHEMA_VERSION = 4


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _content_hash(*arrays: Array) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _atomic_json(payload: object, path: Path) -> tuple[float, int]:
    encoded = (_canonical_json(payload) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    start = perf_counter()
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return perf_counter() - start, len(encoded)


@dataclass(frozen=True, slots=True)
class SparseShardIdentity:
    group: int
    measurement_block: int
    first_measurement_position: int
    measurement_count: int
    support_pattern: int = 0
    storage_shard: int | None = None

    @property
    def key(self) -> str:
        if self.storage_shard is not None:
            return f"storage-{self.storage_shard:06d}"
        return (
            f"group-{self.group:06d}-pattern-{self.support_pattern:06d}-"
            f"block-{self.measurement_block:06d}-"
            f"rows-{self.first_measurement_position:09d}-{self.measurement_count:06d}"
        )


@dataclass(frozen=True, slots=True)
class SparseShardMetrics:
    candidate_entries: int
    realized_entries: int
    discarded_entries: int
    construction_seconds: float = 0.0
    canonicalization_seconds: float = 0.0
    csr_construction_seconds: float = 0.0
    serialization_seconds: float = 0.0
    filesystem_seconds: float = 0.0
    disk_bytes: int = 0


@dataclass(frozen=True, slots=True)
class SparseShardMetadata:
    identity: SparseShardIdentity
    num_measurements: int
    num_free_od: int
    dtype: str
    zero_tolerance: float
    provenance_hash: str
    content_hash: str
    nonzero_entries: int
    fixed_offset_nonzeros: int
    metrics: SparseShardMetrics
    schema_version: int = SHARDED_OPERATOR_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ShardedOperatorManifest:
    num_measurements: int
    num_free_od: int
    dtype: str
    provenance: dict[str, object]
    expected_shards: tuple[SparseShardIdentity, ...]
    completed_shards: tuple[str, ...]
    aggregate_nonzeros: int
    complete: bool
    measurement_block_size: int
    od_chunk_size: int
    plan_summary: dict[str, object] | None = None
    schema_version: int = SHARDED_OPERATOR_SCHEMA_VERSION

    @property
    def provenance_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.provenance).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedSparseShard:
    metadata: SparseShardMetadata
    row_indices: Array
    matrix: sparse.csr_array
    transpose: sparse.csc_array
    fixed_offset_indices: Array
    fixed_offset_values: Array


def shard_path(directory: Path, identity: SparseShardIdentity) -> Path:
    return Path(directory) / "shards" / f"{identity.key}.npz"


def manifest_path(directory: Path) -> Path:
    return Path(directory) / "manifest.json"


def save_sparse_shard(
    *,
    directory: Path,
    identity: SparseShardIdentity,
    row_indices: object,
    matrix: sparse.spmatrix | sparse.sparray,
    fixed_offset: object,
    num_measurements: int,
    num_free_od: int,
    dtype: object,
    zero_tolerance: float,
    provenance_hash: str,
    metrics: SparseShardMetrics,
    compressed: bool = False,
) -> SparseShardMetadata:
    """Canonicalize and atomically persist one independent CSR shard."""
    canonical_start = perf_counter()
    rows = np.asarray(row_indices, dtype=np.int64)
    if rows.ndim != 1 or rows.size != identity.measurement_count:
        raise ValueError("shard row mapping has an invalid shape.")
    if rows.size and (
        np.any(rows < 0)
        or np.any(rows >= num_measurements)
        or np.unique(rows).size != rows.size
        or np.any(rows[1:] <= rows[:-1])
    ):
        raise ValueError("shard rows must be unique, sorted, and in bounds.")
    csr_start = perf_counter()
    csr = sparse.csr_array(matrix, shape=(rows.size, num_free_od), dtype=dtype)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    csr.sort_indices()
    csr_seconds = perf_counter() - csr_start
    if not np.all(np.isfinite(csr.data)) or np.any(csr.data < 0.0):
        raise ValueError("shard data must be finite and non-negative.")
    if zero_tolerance:
        csr.data[np.abs(csr.data) <= zero_tolerance] = 0.0
        csr.eliminate_zeros()
    offset = np.asarray(fixed_offset, dtype=dtype)
    if offset.shape != (rows.size,) or not np.all(np.isfinite(offset)):
        raise ValueError("shard fixed offset is invalid.")
    offset_positions = np.flatnonzero(np.abs(offset) > zero_tolerance).astype(np.int64)
    offset_values = offset[offset_positions]
    content_hash = _content_hash(
        rows,
        csr.data,
        csr.indices,
        csr.indptr,
        offset_positions,
        offset_values,
    )
    canonical_seconds = perf_counter() - canonical_start
    metadata = SparseShardMetadata(
        identity=identity,
        num_measurements=num_measurements,
        num_free_od=num_free_od,
        dtype=str(np.dtype(dtype)),
        zero_tolerance=float(zero_tolerance),
        provenance_hash=provenance_hash,
        content_hash=content_hash,
        nonzero_entries=int(csr.nnz),
        fixed_offset_nonzeros=int(offset_positions.size),
        metrics=SparseShardMetrics(
            candidate_entries=metrics.candidate_entries,
            realized_entries=int(csr.nnz),
            discarded_entries=metrics.candidate_entries - int(csr.nnz),
            construction_seconds=metrics.construction_seconds,
            canonicalization_seconds=canonical_seconds,
            csr_construction_seconds=csr_seconds,
        ),
    )
    destination = shard_path(directory, identity)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    serialization_start = perf_counter()
    try:
        with open(temporary, "wb") as stream:
            writer = np.savez_compressed if compressed else np.savez
            writer(
                stream,
                metadata=np.asarray(_canonical_json(asdict(metadata))),
                rows=rows,
                data=csr.data,
                indices=csr.indices,
                indptr=csr.indptr,
                offset_indices=offset_positions,
                offset_values=offset_values,
            )
        serialization_seconds = perf_counter() - serialization_start
        filesystem_start = perf_counter()
        os.replace(temporary, destination)
        filesystem_seconds = perf_counter() - filesystem_start
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    complete_metrics = SparseShardMetrics(
        candidate_entries=metadata.metrics.candidate_entries,
        realized_entries=metadata.metrics.realized_entries,
        discarded_entries=metadata.metrics.discarded_entries,
        construction_seconds=metadata.metrics.construction_seconds,
        canonicalization_seconds=metadata.metrics.canonicalization_seconds,
        csr_construction_seconds=metadata.metrics.csr_construction_seconds,
        serialization_seconds=serialization_seconds,
        filesystem_seconds=filesystem_seconds,
        disk_bytes=destination.stat().st_size,
    )
    # Timing from the atomic write is returned to the caller. It is deliberately
    # not persisted via a second full NPZ rewrite: one durable write per shard is
    # the important operational invariant.
    return replace(metadata, metrics=complete_metrics)


def _metadata_from_payload(payload: dict[str, object]) -> SparseShardMetadata:
    return SparseShardMetadata(
        identity=SparseShardIdentity(**payload["identity"]),
        metrics=SparseShardMetrics(**payload["metrics"]),
        **{key: value for key, value in payload.items() if key not in {"identity", "metrics"}},
    )


def load_sparse_shard(
    path: Path, *, expected_provenance_hash: str | None = None
) -> LoadedSparseShard:
    """Load and fully validate one shard, including its numerical content hash."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = _metadata_from_payload(json.loads(str(archive["metadata"])))
        if metadata.schema_version != SHARDED_OPERATOR_SCHEMA_VERSION:
            raise ValueError("unsupported sparse shard schema version.")
        if expected_provenance_hash is not None and (
            metadata.provenance_hash != expected_provenance_hash
        ):
            raise ValueError("sparse shard provenance mismatch.")
        rows = np.asarray(archive["rows"], dtype=np.int64)
        data = np.asarray(archive["data"])
        indices = np.asarray(archive["indices"])
        indptr = np.asarray(archive["indptr"])
        offset_indices = np.asarray(archive["offset_indices"], dtype=np.int64)
        offset_values = np.asarray(archive["offset_values"])
    if rows.shape != (metadata.identity.measurement_count,):
        raise ValueError("stored shard row mapping shape mismatch.")
    matrix = sparse.csr_array(
        (data, indices, indptr),
        shape=(rows.size, metadata.num_free_od),
    )
    if matrix.nnz != metadata.nonzero_entries:
        raise ValueError("stored shard nonzero count mismatch.")
    actual_hash = _content_hash(
        rows, data, indices, indptr, offset_indices, offset_values
    )
    if actual_hash != metadata.content_hash:
        raise ValueError("stored sparse shard content hash mismatch.")
    if offset_indices.size and (
        np.any(offset_indices < 0) or np.any(offset_indices >= rows.size)
    ):
        raise ValueError("stored shard offset indices are out of bounds.")
    transpose = sparse.csc_array(matrix, copy=True)
    return LoadedSparseShard(
        metadata=metadata,
        row_indices=rows,
        matrix=matrix,
        transpose=transpose,
        fixed_offset_indices=offset_indices,
        fixed_offset_values=offset_values,
    )


def save_sharded_operator_manifest(
    manifest: ShardedOperatorManifest, directory: Path
) -> tuple[float, int]:
    payload = asdict(manifest)
    payload["provenance_hash"] = manifest.provenance_hash
    return _atomic_json(payload, manifest_path(directory))


def load_sharded_operator_manifest(directory: Path) -> ShardedOperatorManifest:
    payload = json.loads(manifest_path(directory).read_text(encoding="utf-8"))
    stored_hash = payload.pop("provenance_hash")
    expected = tuple(SparseShardIdentity(**item) for item in payload.pop("expected_shards"))
    manifest = ShardedOperatorManifest(expected_shards=expected, **payload)
    if manifest.schema_version != SHARDED_OPERATOR_SCHEMA_VERSION:
        raise ValueError("unsupported sharded operator manifest schema version.")
    if manifest.provenance_hash != stored_hash:
        raise ValueError("sharded operator manifest provenance hash mismatch.")
    expected_keys = {item.key for item in manifest.expected_shards}
    completed = set(manifest.completed_shards)
    if not completed <= expected_keys:
        raise ValueError("manifest contains an unexpected completed shard.")
    if manifest.complete != (completed == expected_keys):
        raise ValueError("manifest completion flag is inconsistent.")
    return manifest


class ShardedSparseLinearOperator:
    """Logical sparse operator with eager or bounded LRU shard loading."""

    def __init__(
        self,
        directory: Path,
        *,
        max_cached_shards: int | None = None,
        memory_budget_bytes: int | None = None,
        merge_eager: bool = True,
    ):
        loading_start = perf_counter()
        self.directory = Path(directory)
        self.manifest = load_sharded_operator_manifest(self.directory)
        if not self.manifest.complete:
            raise ValueError("sharded operator cache is incomplete.")
        if max_cached_shards is not None and max_cached_shards <= 0:
            raise ValueError("max_cached_shards must be positive when provided.")
        if memory_budget_bytes is not None and memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive when provided.")
        shard_disk_bytes = [
            shard_path(self.directory, identity).stat().st_size
            for identity in self.manifest.expected_shards
        ]
        if max_cached_shards is None and memory_budget_bytes is not None:
            estimated_resident_bytes = sum(shard_disk_bytes) * 2
            if estimated_resident_bytes > memory_budget_bytes and shard_disk_bytes:
                average_resident = max(1, estimated_resident_bytes // len(shard_disk_bytes))
                max_cached_shards = max(
                    1,
                    min(
                        len(shard_disk_bytes),
                        memory_budget_bytes // average_resident,
                    ),
                )
        self.max_cached_shards = max_cached_shards
        self.loading_policy = "eager" if max_cached_shards is None else "lru"
        self._cache: OrderedDict[str, LoadedSparseShard] = OrderedDict()
        self._lock = RLock()
        self.matvec_count = 0
        self.rmatvec_count = 0
        self.shard_load_count = 0
        self.shard_cache_hit_count = 0
        self.shard_eviction_count = 0
        self.bytes_read = 0
        self.file_open_count = 0
        self.sparse_matrix_calls = 0
        self.merge_seconds = 0.0
        self.merged_csr_seconds = 0.0
        self.merged_transpose_seconds = 0.0
        self.merged_storage_bytes = 0
        self._merged_matrix: sparse.csr_array | None = None
        self._merged_transpose: sparse.csc_array | None = None
        if max_cached_shards is None:
            for identity in self.manifest.expected_shards:
                self._get(identity)
        offset = np.zeros(self.manifest.num_measurements, dtype=self.dtype)
        for identity in self.manifest.expected_shards:
            shard = self._get(identity)
            if shard.fixed_offset_indices.size:
                np.add.at(
                    offset,
                    shard.row_indices[shard.fixed_offset_indices],
                    shard.fixed_offset_values,
                )
        offset.setflags(write=False)
        self.fixed_measurement_offset = offset
        if self.loading_policy == "eager" and merge_eager:
            merge_start = perf_counter()
            row_parts: list[Array] = []
            column_parts: list[Array] = []
            data_parts: list[Array] = []
            for identity in self.manifest.expected_shards:
                shard = self._get(identity)
                local = sparse.coo_array(shard.matrix)
                row_parts.append(shard.row_indices[local.row])
                column_parts.append(local.col)
                data_parts.append(local.data)
            csr_start = perf_counter()
            matrix = sparse.coo_array(
                (
                    np.concatenate(data_parts) if data_parts else np.empty(0, self.dtype),
                    (
                        np.concatenate(row_parts) if row_parts else np.empty(0, np.int64),
                        np.concatenate(column_parts) if column_parts else np.empty(0, np.int64),
                    ),
                ),
                shape=self.shape,
                dtype=self.dtype,
            ).tocsr()
            matrix.sum_duplicates()
            matrix.sort_indices()
            self.merged_csr_seconds = perf_counter() - csr_start
            transpose_start = perf_counter()
            transpose = sparse.csc_array(matrix, copy=True)
            self.merged_transpose_seconds = perf_counter() - transpose_start
            self._merged_matrix = matrix
            self._merged_transpose = transpose
            self.merged_storage_bytes = sum(
                array.nbytes
                for array in (
                    matrix.data,
                    matrix.indices,
                    matrix.indptr,
                    transpose.data,
                    transpose.indices,
                    transpose.indptr,
                )
            )
            self.merge_seconds = perf_counter() - merge_start
        self.loading_seconds = perf_counter() - loading_start

    @property
    def shape(self) -> tuple[int, int]:
        return self.manifest.num_measurements, self.manifest.num_free_od

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.manifest.dtype)

    @property
    def uses_merged_operator(self) -> bool:
        return self._merged_matrix is not None

    def _get(self, identity: SparseShardIdentity) -> LoadedSparseShard:
        with self._lock:
            cached = self._cache.pop(identity.key, None)
            if cached is not None:
                self.shard_cache_hit_count += 1
                self._cache[identity.key] = cached
                return cached
            loaded = load_sparse_shard(
                shard_path(self.directory, identity),
                expected_provenance_hash=self.manifest.provenance_hash,
            )
            self.shard_load_count += 1
            self.file_open_count += 1
            self.bytes_read += shard_path(self.directory, identity).stat().st_size
            self._cache[identity.key] = loaded
            if self.max_cached_shards is not None:
                while len(self._cache) > self.max_cached_shards:
                    self._cache.popitem(last=False)
                    self.shard_eviction_count += 1
            return loaded

    def matvec(self, vector: object) -> Array:
        value = np.asarray(vector, dtype=self.dtype)
        if value.shape != (self.shape[1],) or not np.all(np.isfinite(value)):
            raise ValueError(f"forward vector must be finite with shape ({self.shape[1]},).")
        result = np.zeros(self.shape[0], dtype=np.result_type(value.dtype, self.dtype))
        if self._merged_matrix is not None:
            result = np.asarray(self._merged_matrix @ value)
            self.sparse_matrix_calls += 1
            with self._lock:
                self.matvec_count += 1
            return result
        for identity in self.manifest.expected_shards:
            shard = self._get(identity)
            np.add.at(result, shard.row_indices, np.asarray(shard.matrix @ value))
            self.sparse_matrix_calls += 1
        with self._lock:
            self.matvec_count += 1
        return result

    def rmatvec(self, vector: object) -> Array:
        value = np.asarray(vector, dtype=self.dtype)
        if value.shape != (self.shape[0],) or not np.all(np.isfinite(value)):
            raise ValueError(f"transpose vector must be finite with shape ({self.shape[0]},).")
        result = np.zeros(self.shape[1], dtype=np.result_type(value.dtype, self.dtype))
        if self._merged_transpose is not None:
            result = np.asarray(self._merged_transpose.T @ value)
            self.sparse_matrix_calls += 1
            with self._lock:
                self.rmatvec_count += 1
            return result
        for identity in self.manifest.expected_shards:
            shard = self._get(identity)
            result += np.asarray(shard.transpose.T @ value[shard.row_indices])
            self.sparse_matrix_calls += 1
        with self._lock:
            self.rmatvec_count += 1
        return result
