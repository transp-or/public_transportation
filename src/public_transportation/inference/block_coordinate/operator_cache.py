"""Fingerprint-safe persistence and bounded retention of OD-block operators."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock, RLock
from time import perf_counter

import numpy as np
from scipy import sparse

from public_transportation import __version__
from public_transportation.inference.linear_operator import (
    DenseLinearOperator,
    LinearOperatorProtocol,
    SparseLinearOperator,
)

from ._canonical import fingerprint
from .blocks import ODBlock
from .operator import BlockLinearOperatorProtocol, SparseBlockLinearOperator

BLOCK_OPERATOR_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BlockOperatorCacheProvenance:
    """Authoritative identities affecting a fixed-routing block operator."""

    assignment_inputs: str
    od_layout: str
    fixed_demand_layout: str
    measurement_mapping: str
    routing_parameters: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} fingerprint must be nonempty.")


@dataclass(frozen=True, slots=True)
class BlockOperatorCacheConfig:
    cache_directory: Path
    maximum_retained_blocks: int = 2
    maximum_retained_bytes: int | None = None
    zero_tolerance: float = 0.0
    storage_dtype: str = "float64"

    def __post_init__(self) -> None:
        directory = Path(self.cache_directory).expanduser()
        if self.maximum_retained_blocks < 0:
            raise ValueError("maximum_retained_blocks must be non-negative.")
        if self.maximum_retained_bytes is not None and self.maximum_retained_bytes <= 0:
            raise ValueError("maximum_retained_bytes must be positive when provided.")
        if not np.isfinite(self.zero_tolerance) or self.zero_tolerance < 0.0:
            raise ValueError("zero_tolerance must be finite and non-negative.")
        dtype = np.dtype(self.storage_dtype)
        if dtype.kind != "f":
            raise TypeError("storage_dtype must be a real floating-point dtype.")
        object.__setattr__(self, "cache_directory", directory)
        object.__setattr__(self, "storage_dtype", str(dtype))

    @property
    def construction_fingerprint(self) -> str:
        return fingerprint(
            {
                "schema_version": BLOCK_OPERATOR_CACHE_SCHEMA_VERSION,
                "package_version": __version__,
                "zero_tolerance": self.zero_tolerance,
                "storage_dtype": self.storage_dtype,
            }
        )


@dataclass(frozen=True, slots=True)
class BlockOperatorPreparationMetrics:
    cache_hit: bool
    cache_lookup_seconds: float
    cache_load_seconds: float
    construction_seconds: float
    persistence_seconds: float
    nonzero_entries: int
    retained_bytes: int
    disk_bytes: int


@dataclass(frozen=True, slots=True)
class BlockOperatorProductMetrics:
    matvec_count: int
    rmatvec_count: int
    matvec_seconds: float
    rmatvec_seconds: float


@dataclass(frozen=True, slots=True)
class BlockOperatorFactoryMetrics:
    memory_cache_hits: int
    disk_cache_hits: int
    cold_builds: int
    evictions: int
    explicit_releases: int


class CachedBlockLinearOperator:
    """Measured block handle that can explicitly release its sparse storage."""

    def __init__(
        self,
        *,
        block: ODBlock,
        cache_key: str,
        operator: SparseBlockLinearOperator,
        preparation_metrics: BlockOperatorPreparationMetrics,
    ) -> None:
        self.block = block
        self.cache_key = cache_key
        self._operator: SparseBlockLinearOperator | None = operator
        self.preparation_metrics = preparation_metrics
        self._matvec_count = 0
        self._rmatvec_count = 0
        self._matvec_seconds = 0.0
        self._rmatvec_seconds = 0.0

    def _require_operator(self) -> SparseBlockLinearOperator:
        if self._operator is None:
            raise RuntimeError("block operator storage has been released.")
        return self._operator

    @property
    def shape(self) -> tuple[int, int]:
        return self._require_operator().shape

    @property
    def dtype(self) -> np.dtype:
        return self._require_operator().dtype

    @property
    def num_measurements(self) -> int:
        return self._require_operator().num_measurements

    @property
    def num_local_variables(self) -> int:
        return self._require_operator().num_local_variables

    @property
    def measurement_support_indices(self) -> tuple[int, ...]:
        return self._require_operator().measurement_support_indices

    @property
    def released(self) -> bool:
        return self._operator is None

    @property
    def retained_bytes(self) -> int:
        if self._operator is None:
            return 0
        operator = self._operator
        return int(
            operator.matrix.data.nbytes
            + operator.matrix.indices.nbytes
            + operator.matrix.indptr.nbytes
            + operator.transpose_matrix.data.nbytes
            + operator.transpose_matrix.indices.nbytes
            + operator.transpose_matrix.indptr.nbytes
        )

    @property
    def product_metrics(self) -> BlockOperatorProductMetrics:
        return BlockOperatorProductMetrics(
            self._matvec_count,
            self._rmatvec_count,
            self._matvec_seconds,
            self._rmatvec_seconds,
        )

    def matvec(self, local_vector: object) -> np.ndarray:
        started = perf_counter()
        value = self._require_operator().matvec(local_vector)
        self._matvec_seconds += perf_counter() - started
        self._matvec_count += 1
        return value

    def rmatvec(self, measurement_vector: object) -> np.ndarray:
        started = perf_counter()
        value = self._require_operator().rmatvec(measurement_vector)
        self._rmatvec_seconds += perf_counter() - started
        self._rmatvec_count += 1
        return value

    def release(self) -> None:
        self._operator = None


def _content_hash(matrix: sparse.csr_array) -> str:
    digest = hashlib.sha256()
    for value in (matrix.data, matrix.indices, matrix.indptr):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_save(path: Path, metadata: dict, matrix: sparse.csr_array) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(
                stream,
                metadata=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                ),
                data=matrix.data,
                indices=matrix.indices,
                indptr=matrix.indptr,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path.stat().st_size


def _load(path: Path, *, expected_key: str) -> tuple[SparseBlockLinearOperator, int]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            data = np.asarray(archive["data"])
            indices = np.asarray(archive["indices"])
            indptr = np.asarray(archive["indptr"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid block-operator cache file {path}.") from error
    if metadata.get("schema_version") != BLOCK_OPERATOR_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported block-operator cache schema version.")
    if metadata.get("cache_key") != expected_key:
        raise ValueError("block-operator cache fingerprint mismatch.")
    shape = tuple(metadata.get("shape", ()))
    if len(shape) != 2:
        raise ValueError("cached block-operator shape is invalid.")
    matrix = sparse.csr_array((data, indices, indptr), shape=shape)
    if _content_hash(matrix) != metadata.get("content_hash"):
        raise ValueError("cached block-operator content hash mismatch.")
    operator = SparseBlockLinearOperator(matrix)
    if list(operator.measurement_support_indices) != metadata.get(
        "measurement_support_indices"
    ):
        raise ValueError("cached block-operator measurement support mismatch.")
    return operator, path.stat().st_size


class FixedRoutingBlockOperatorFactory:
    """Build, persist, retain, and release compact fixed-routing OD blocks."""

    def __init__(
        self,
        *,
        complete_operator: LinearOperatorProtocol,
        provenance: BlockOperatorCacheProvenance,
        config: BlockOperatorCacheConfig,
    ) -> None:
        self.complete_operator = complete_operator
        self.provenance = provenance
        self.config = config
        self._retained: OrderedDict[str, CachedBlockLinearOperator] = OrderedDict()
        self._lock = RLock()
        self._key_locks: dict[str, Lock] = {}
        self._memory_cache_hits = 0
        self._disk_cache_hits = 0
        self._cold_builds = 0
        self._evictions = 0
        self._explicit_releases = 0
        if not isinstance(complete_operator, LinearOperatorProtocol):
            raise TypeError("complete_operator must satisfy LinearOperatorProtocol.")
        if complete_operator.shape[0] <= 0 or complete_operator.shape[1] <= 0:
            raise ValueError("complete_operator dimensions must be strictly positive.")

    def cache_key(self, block: ODBlock) -> str:
        return fingerprint(
            {
                "schema_version": BLOCK_OPERATOR_CACHE_SCHEMA_VERSION,
                "package_version": __version__,
                "provenance": asdict(self.provenance),
                "block_fingerprint": block.fingerprint,
                "construction_configuration": self.config.construction_fingerprint,
                "complete_shape": self.complete_operator.shape,
                "complete_dtype": str(self.complete_operator.dtype),
            }
        )

    def cache_path(self, block: ODBlock) -> Path:
        return self.config.cache_directory / f"block-{self.cache_key(block)}.npz"

    @property
    def retained_block_count(self) -> int:
        return len(self._retained)

    @property
    def retained_bytes(self) -> int:
        return sum(operator.retained_bytes for operator in self._retained.values())

    @property
    def metrics(self) -> BlockOperatorFactoryMetrics:
        return BlockOperatorFactoryMetrics(
            memory_cache_hits=self._memory_cache_hits,
            disk_cache_hits=self._disk_cache_hits,
            cold_builds=self._cold_builds,
            evictions=self._evictions,
            explicit_releases=self._explicit_releases,
        )

    def _build(self, block: ODBlock) -> SparseBlockLinearOperator:
        columns = np.asarray(block.free_column_indices, dtype=np.intp)
        if columns[-1] >= self.complete_operator.shape[1]:
            raise ValueError("block contains a column outside the complete operator.")
        if isinstance(self.complete_operator, SparseLinearOperator):
            matrix = self.complete_operator.matrix[:, columns]
        elif isinstance(self.complete_operator, DenseLinearOperator):
            matrix = sparse.csr_array(self.complete_operator.matrix[:, columns])
        else:
            matrix = np.empty(
                (self.complete_operator.shape[0], columns.size),
                dtype=self.config.storage_dtype,
            )
            basis = np.zeros(
                self.complete_operator.shape[1], dtype=self.complete_operator.dtype
            )
            for local, column in enumerate(columns):
                basis[column] = 1.0
                matrix[:, local] = self.complete_operator.matvec(basis)
                basis[column] = 0.0
            matrix = sparse.csr_array(matrix)
        matrix = sparse.csr_array(
            matrix, dtype=np.dtype(self.config.storage_dtype), copy=True
        )
        if self.config.zero_tolerance:
            matrix.data[np.abs(matrix.data) <= self.config.zero_tolerance] = 0.0
        matrix.eliminate_zeros()
        operator = SparseBlockLinearOperator(matrix)
        if block.measurement_support_indices is not None and (
            operator.measurement_support_indices
            != block.measurement_support_indices
        ):
            raise ValueError(
                f"block {block.block_id!r} declared measurement support does not "
                "match its constructed operator."
            )
        return operator

    def _retain(self, key: str, operator: CachedBlockLinearOperator) -> None:
        if self.config.maximum_retained_blocks == 0:
            return
        self._retained[key] = operator
        self._retained.move_to_end(key)
        while self._retained and (
            len(self._retained) > self.config.maximum_retained_blocks
            or (
                self.config.maximum_retained_bytes is not None
                and self.retained_bytes > self.config.maximum_retained_bytes
            )
        ):
            _, evicted = self._retained.popitem(last=False)
            self._evictions += 1
            if evicted is not operator or len(self._retained) > 0:
                evicted.release()

    def get(self, block: ODBlock) -> CachedBlockLinearOperator:
        key = self.cache_key(block)
        with self._lock:
            key_lock = self._key_locks.setdefault(key, Lock())
        with key_lock:
            with self._lock:
                retained = self._retained.get(key)
                if retained is not None and not retained.released:
                    self._memory_cache_hits += 1
                    self._retained.move_to_end(key)
                    return retained
            path = self.cache_path(block)
            lookup_started = perf_counter()
            exists = path.is_file()
            lookup_seconds = perf_counter() - lookup_started
            load_seconds = construction_seconds = persistence_seconds = 0.0
            if exists:
                started = perf_counter()
                operator, disk_bytes = _load(path, expected_key=key)
                load_seconds = perf_counter() - started
                cache_hit = True
                with self._lock:
                    self._disk_cache_hits += 1
            else:
                started = perf_counter()
                operator = self._build(block)
                construction_seconds = perf_counter() - started
                metadata = {
                    "schema_version": BLOCK_OPERATOR_CACHE_SCHEMA_VERSION,
                    "package_version": __version__,
                    "cache_key": key,
                    "block_id": block.block_id,
                    "block_fingerprint": block.fingerprint,
                    "provenance": asdict(self.provenance),
                    "construction_configuration": self.config.construction_fingerprint,
                    "shape": list(operator.shape),
                    "dtype": str(operator.dtype),
                    "measurement_support_indices": list(
                        operator.measurement_support_indices
                    ),
                    "content_hash": _content_hash(operator.matrix),
                }
                started = perf_counter()
                disk_bytes = _atomic_save(path, metadata, operator.matrix)
                persistence_seconds = perf_counter() - started
                cache_hit = False
                with self._lock:
                    self._cold_builds += 1
            retained_bytes = int(
                operator.matrix.data.nbytes
                + operator.matrix.indices.nbytes
                + operator.matrix.indptr.nbytes
                + operator.transpose_matrix.data.nbytes
                + operator.transpose_matrix.indices.nbytes
                + operator.transpose_matrix.indptr.nbytes
            )
            handle = CachedBlockLinearOperator(
                block=block,
                cache_key=key,
                operator=operator,
                preparation_metrics=BlockOperatorPreparationMetrics(
                    cache_hit=cache_hit,
                    cache_lookup_seconds=lookup_seconds,
                    cache_load_seconds=load_seconds,
                    construction_seconds=construction_seconds,
                    persistence_seconds=persistence_seconds,
                    nonzero_entries=int(operator.matrix.nnz),
                    retained_bytes=retained_bytes,
                    disk_bytes=disk_bytes,
                ),
            )
            with self._lock:
                self._retain(key, handle)
            return handle

    def __call__(self, block: ODBlock) -> BlockLinearOperatorProtocol:
        return self.get(block)

    def release(self, block: ODBlock) -> bool:
        key = self.cache_key(block)
        with self._lock:
            operator = self._retained.pop(key, None)
            if operator is None:
                return False
            operator.release()
            self._explicit_releases += 1
            return True

    def release_all(self) -> None:
        with self._lock:
            for operator in self._retained.values():
                operator.release()
                self._explicit_releases += 1
            self._retained.clear()
