"""Deterministic pre-execution splitting of resource-unsafe OD blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Sequence

from .blocks import ODBlock
from .checkpoint import BlockCoordinateFingerprints
from .partition import ODBlockPartition
from .resources import BlockPreflightSample

SplitTrigger = Literal["memory", "runtime", "memory_and_runtime"]


class BlockResourceGuardError(RuntimeError):
    """Raised when an indivisible block cannot satisfy an accepted limit."""


@dataclass(frozen=True, slots=True)
class AdaptiveBlockSplitConfig:
    maximum_worker_memory_bytes: int | None = None
    maximum_block_runtime_seconds: float | None = None
    minimum_free_variables_per_block: int = 1
    maximum_split_depth: int = 32

    def __post_init__(self) -> None:
        if self.maximum_worker_memory_bytes is not None and self.maximum_worker_memory_bytes <= 0:
            raise ValueError("maximum_worker_memory_bytes must be positive when provided.")
        if self.maximum_block_runtime_seconds is not None and (
            not math.isfinite(self.maximum_block_runtime_seconds)
            or self.maximum_block_runtime_seconds <= 0.0
        ):
            raise ValueError("maximum_block_runtime_seconds must be finite and positive.")
        if self.minimum_free_variables_per_block <= 0:
            raise ValueError("minimum_free_variables_per_block must be positive.")
        if self.maximum_split_depth <= 0:
            raise ValueError("maximum_split_depth must be positive.")
        if (
            self.maximum_worker_memory_bytes is None
            and self.maximum_block_runtime_seconds is None
        ):
            raise ValueError("adaptive splitting requires a memory or runtime limit.")


@dataclass(frozen=True, slots=True)
class BlockResourceCostModel:
    worker_bytes_per_variable: float
    runtime_seconds_per_variable: float
    uncertainty_factor: float = 1.25

    def __post_init__(self) -> None:
        for name in ("worker_bytes_per_variable", "runtime_seconds_per_variable"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not math.isfinite(self.uncertainty_factor) or self.uncertainty_factor < 1.0:
            raise ValueError("uncertainty_factor must be finite and at least one.")

    @classmethod
    def from_preflight_samples(
        cls,
        samples: Sequence[BlockPreflightSample],
        *,
        uncertainty_factor: float = 1.25,
    ) -> BlockResourceCostModel:
        if not samples:
            raise ValueError("at least one preflight sample is required.")
        return cls(
            worker_bytes_per_variable=max(
                sample.worker_peak_bytes / sample.variables for sample in samples
            ),
            runtime_seconds_per_variable=max(
                (
                    sample.construction_seconds
                    + sample.matvec_seconds
                    + sample.rmatvec_seconds
                )
                / sample.variables
                for sample in samples
            ),
            uncertainty_factor=uncertainty_factor,
        )

    def estimate(self, block: ODBlock) -> tuple[int, float]:
        variables = block.num_free_variables
        return (
            math.ceil(
                variables * self.worker_bytes_per_variable * self.uncertainty_factor
            ),
            variables
            * self.runtime_seconds_per_variable
            * self.uncertainty_factor,
        )


@dataclass(frozen=True, slots=True)
class AdaptiveBlockSplitRecord:
    parent_block_id: str
    child_block_ids: tuple[str, str]
    trigger: SplitTrigger
    estimated_parent_memory_bytes: int
    estimated_parent_runtime_seconds: float


@dataclass(frozen=True, slots=True)
class AdaptiveBlockSplitResult:
    partition: ODBlockPartition
    original_partition_fingerprint: str
    records: tuple[AdaptiveBlockSplitRecord, ...]
    maximum_estimated_worker_memory_bytes: int
    maximum_estimated_block_runtime_seconds: float

    @property
    def changed(self) -> bool:
        return bool(self.records)


def _trigger(
    memory_bytes: int,
    runtime_seconds: float,
    config: AdaptiveBlockSplitConfig,
) -> SplitTrigger | None:
    memory = (
        config.maximum_worker_memory_bytes is not None
        and memory_bytes > config.maximum_worker_memory_bytes
    )
    runtime = (
        config.maximum_block_runtime_seconds is not None
        and runtime_seconds > config.maximum_block_runtime_seconds
    )
    if memory and runtime:
        return "memory_and_runtime"
    if memory:
        return "memory"
    if runtime:
        return "runtime"
    return None


def _split_block(block: ODBlock, *, depth: int) -> tuple[ODBlock, ODBlock]:
    midpoint = block.num_free_variables // 2
    slices = (slice(0, midpoint), slice(midpoint, block.num_free_variables))
    children: list[ODBlock] = []
    total_nonzeros = block.estimated_nonzeros
    if total_nonzeros is None:
        nonzeros: tuple[int | None, int | None] = (None, None)
    else:
        first_nonzeros = total_nonzeros * midpoint // block.num_free_variables
        nonzeros = (first_nonzeros, total_nonzeros - first_nonzeros)
    for branch, block_slice in enumerate(slices):
        children.append(
            ODBlock(
                block_id=f"{block.block_id}::split-{depth:02d}-{branch}",
                free_column_indices=block.free_column_indices[block_slice],
                active_od_indices=block.active_od_indices[block_slice],
                destination_group_indices=block.destination_group_indices,
                time_bin_ids=block.time_bin_ids,
                estimated_nonzeros=nonzeros[branch],
                # The parent support is a safe conservative support for each child.
                measurement_support_indices=block.measurement_support_indices,
            )
        )
    return children[0], children[1]


def split_partition_for_resource_limits(
    partition: ODBlockPartition,
    *,
    cost_model: BlockResourceCostModel,
    config: AdaptiveBlockSplitConfig,
) -> AdaptiveBlockSplitResult:
    """Recursively bisect unsafe blocks before any full block is constructed."""
    pending = [(block, 0) for block in partition.blocks]
    safe: list[ODBlock] = []
    records: list[AdaptiveBlockSplitRecord] = []
    while pending:
        block, depth = pending.pop(0)
        memory, runtime = cost_model.estimate(block)
        trigger = _trigger(memory, runtime, config)
        if trigger is None:
            safe.append(block)
            continue
        if (
            block.num_free_variables < 2 * config.minimum_free_variables_per_block
            or depth >= config.maximum_split_depth
        ):
            raise BlockResourceGuardError(
                f"block {block.block_id!r} remains unsafe at "
                f"{block.num_free_variables} variable(s): estimated memory={memory} "
                f"bytes, runtime={runtime:.6g} seconds."
            )
        children = _split_block(block, depth=depth + 1)
        if any(
            child.num_free_variables < config.minimum_free_variables_per_block
            for child in children
        ):
            raise BlockResourceGuardError(
                f"block {block.block_id!r} cannot be split without violating the minimum size."
            )
        records.append(
            AdaptiveBlockSplitRecord(
                parent_block_id=block.block_id,
                child_block_ids=(children[0].block_id, children[1].block_id),
                trigger=trigger,
                estimated_parent_memory_bytes=memory,
                estimated_parent_runtime_seconds=runtime,
            )
        )
        pending[0:0] = [(children[0], depth + 1), (children[1], depth + 1)]

    revised = ODBlockPartition(
        blocks=tuple(safe),
        num_free_variables=partition.num_free_variables,
    )
    free_columns = [
        column for block in revised.blocks for column in block.free_column_indices
    ]
    if sorted(free_columns) != list(range(partition.num_free_variables)) or len(
        free_columns
    ) != len(set(free_columns)):
        raise AssertionError("adaptive splitting did not preserve exact free-column coverage.")
    estimates = [cost_model.estimate(block) for block in revised.blocks]
    return AdaptiveBlockSplitResult(
        partition=revised,
        original_partition_fingerprint=partition.fingerprint,
        records=tuple(records),
        maximum_estimated_worker_memory_bytes=max(value[0] for value in estimates),
        maximum_estimated_block_runtime_seconds=max(value[1] for value in estimates),
    )


def fingerprints_for_adapted_partition(
    fingerprints: BlockCoordinateFingerprints,
    result: AdaptiveBlockSplitResult,
) -> BlockCoordinateFingerprints:
    """Bind checkpoints to the revised partition and reject obsolete schedules."""
    if fingerprints.partition != result.original_partition_fingerprint:
        raise ValueError("fingerprints do not describe the original partition.")
    return replace(fingerprints, partition=result.partition.fingerprint)
