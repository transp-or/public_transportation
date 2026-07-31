"""Deterministic destination/time OD block partitioning."""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import index

import numpy as np

from public_transportation.inference.assignment_adapter import AssignmentInputs
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout

from ._canonical import fingerprint
from .blocks import ODBlock
from .config import BlockSizingConfig


def _index_tuple(value: object, *, name: str) -> tuple[int, ...]:
    try:
        result = tuple(index(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain integers.") from error
    return result


def require_measurements_for_block_estimation(num_measurements: int) -> None:
    """Reject estimation explicitly when a structural example has no measurements."""
    if num_measurements <= 0:
        raise ValueError(
            "block-coordinate MAP estimation requires at least one measurement; "
            "this scenario is applicable only to loading and partitioning."
        )


@dataclass(frozen=True, slots=True)
class ODBlockPartition:
    blocks: tuple[ODBlock, ...]
    num_free_variables: int
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        if self.num_free_variables < 0:
            raise ValueError("num_free_variables must be non-negative.")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(
            self,
            "fingerprint",
            fingerprint(
                {
                    "version": 1,
                    "num_free_variables": self.num_free_variables,
                    "block_fingerprints": tuple(block.fingerprint for block in blocks),
                }
            ),
        )

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)


def validate_block_partition(
    blocks: object,
    *,
    free_to_active_indices: object,
    frozen_active_indices: object = (),
    sizing: BlockSizingConfig | None = None,
) -> ODBlockPartition:
    """Validate exact, unique coverage of the canonical free-column layout."""
    normalized = tuple(blocks)  # type: ignore[arg-type]
    if any(not isinstance(block, ODBlock) for block in normalized):
        raise TypeError("blocks must contain ODBlock instances.")
    free_to_active = _index_tuple(
        free_to_active_indices, name="free_to_active_indices"
    )
    if free_to_active != tuple(sorted(free_to_active)) or len(set(free_to_active)) != len(
        free_to_active
    ) or any(value < 0 for value in free_to_active):
        raise ValueError(
            "free_to_active_indices must be unique, ascending, and non-negative."
        )
    frozen = set(_index_tuple(frozen_active_indices, name="frozen_active_indices"))
    if frozen & set(free_to_active):
        raise ValueError("free and frozen active indices must be disjoint.")
    columns = [column for block in normalized for column in block.free_column_indices]
    if len(columns) != len(set(columns)):
        raise ValueError("free columns must not appear in more than one block.")
    expected = set(range(len(free_to_active)))
    if set(columns) != expected:
        missing = sorted(expected - set(columns))
        extra = sorted(set(columns) - expected)
        raise ValueError(
            f"blocks must cover every free column exactly once; missing={missing}, extra={extra}."
        )
    active_by_column = dict(enumerate(free_to_active))
    for block in normalized:
        expected_active = tuple(
            sorted(active_by_column[column] for column in block.free_column_indices)
        )
        if block.active_od_indices != expected_active:
            raise ValueError(
                f"block {block.block_id!r} active OD indices do not match its free columns."
            )
        if frozen & set(block.active_od_indices):
            raise ValueError(f"block {block.block_id!r} contains a frozen active OD index.")
        if sizing is not None:
            maximum_variables = sizing.maximum_free_variables_per_block
            if maximum_variables is not None and block.num_free_variables > maximum_variables:
                raise ValueError(
                    f"block {block.block_id!r} exceeds the maximum free-variable ceiling."
                )
            maximum_nonzeros = sizing.maximum_operator_nonzeros_per_block
            if maximum_nonzeros is not None:
                if block.estimated_nonzeros is None:
                    raise ValueError(
                        "a nonzero ceiling requires estimated_nonzeros for every block."
                    )
                if block.estimated_nonzeros > maximum_nonzeros:
                    raise ValueError(
                        f"block {block.block_id!r} exceeds the operator-nonzero ceiling."
                    )
    return ODBlockPartition(
        blocks=normalized,
        num_free_variables=len(free_to_active),
    )


def _fits(
    columns: list[int],
    *,
    next_nonzeros: int | None,
    current_nonzeros: int,
    sizing: BlockSizingConfig,
) -> bool:
    maximum_variables = sizing.maximum_free_variables_per_block
    if maximum_variables is not None and len(columns) + 1 > maximum_variables:
        return False
    maximum_nonzeros = sizing.maximum_operator_nonzeros_per_block
    if maximum_nonzeros is not None:
        if next_nonzeros is None:
            raise ValueError(
                "maximum_operator_nonzeros_per_block requires per-column estimates."
            )
        if current_nonzeros + next_nonzeros > maximum_nonzeros:
            return False
    return True


def partition_od_blocks(
    *,
    free_to_active_indices: object,
    destination_group_by_free_column: object,
    time_bin_by_free_column: object,
    sizing: BlockSizingConfig,
    estimated_nonzeros_by_free_column: object | None = None,
    measurement_support_by_free_column: object | None = None,
    merge_small_compatible: bool = False,
) -> ODBlockPartition:
    """Partition deterministically by destination group, time bin, then column."""
    free_to_active = _index_tuple(
        free_to_active_indices, name="free_to_active_indices"
    )
    groups = _index_tuple(
        destination_group_by_free_column,
        name="destination_group_by_free_column",
    )
    time_bins = tuple(str(value) for value in time_bin_by_free_column)  # type: ignore[arg-type]
    size = len(free_to_active)
    if len(groups) != size or len(time_bins) != size:
        raise ValueError("partition metadata must have one entry per free column.")
    if any(group < 0 for group in groups) or any(not value for value in time_bins):
        raise ValueError("destination groups and time bins must be valid.")
    estimates = (
        None
        if estimated_nonzeros_by_free_column is None
        else _index_tuple(
            estimated_nonzeros_by_free_column,
            name="estimated_nonzeros_by_free_column",
        )
    )
    if estimates is not None and (
        len(estimates) != size or any(value < 0 for value in estimates)
    ):
        raise ValueError("nonzero estimates must be non-negative and match free columns.")
    supports = (
        None
        if measurement_support_by_free_column is None
        else tuple(
            tuple(sorted(set(_index_tuple(value, name="measurement support"))))
            for value in measurement_support_by_free_column  # type: ignore[arg-type]
        )
    )
    if supports is not None and len(supports) != size:
        raise ValueError("measurement supports must match free columns.")
    if (
        sizing.maximum_operator_nonzeros_per_block is not None
        and estimates is None
    ):
        raise ValueError("a nonzero ceiling requires per-column estimates.")
    if size and (
        sizing.maximum_free_variables_per_block is None
        and sizing.maximum_operator_nonzeros_per_block is None
    ):
        raise ValueError(
            "partitioning requires a variable or nonzero ceiling; a memory-only "
            "configuration requires an accepted resource recommendation first."
        )

    grouped: dict[tuple[int, str], list[int]] = {}
    for column, key in enumerate(zip(groups, time_bins, strict=True)):
        grouped.setdefault(key, []).append(column)
    raw: list[tuple[tuple[int, str], list[int]]] = []
    for key in sorted(grouped):
        current: list[int] = []
        current_nonzeros = 0
        for column in grouped[key]:
            next_nonzeros = None if estimates is None else estimates[column]
            if not _fits(
                current,
                next_nonzeros=next_nonzeros,
                current_nonzeros=current_nonzeros,
                sizing=sizing,
            ):
                if not current:
                    raise ValueError(
                        f"free column {column} exceeds a hard block ceiling by itself."
                    )
                raw.append((key, current))
                current = []
                current_nonzeros = 0
                if not _fits(
                    current,
                    next_nonzeros=next_nonzeros,
                    current_nonzeros=current_nonzeros,
                    sizing=sizing,
                ):
                    raise ValueError(
                        f"free column {column} exceeds a hard block ceiling by itself."
                    )
            current.append(column)
            current_nonzeros += 0 if next_nonzeros is None else next_nonzeros
        if current:
            raw.append((key, current))

    if merge_small_compatible and raw:
        merged: list[tuple[tuple[int, str], list[int]]] = []
        for key, columns in raw:
            if merged and merged[-1][0][1] == key[1]:
                previous_key, previous_columns = merged[-1]
                combined_nonzeros = sum(estimates[column] for column in previous_columns) if estimates else 0
                can_merge = all(
                    _fits(
                        previous_columns + columns[:offset],
                        next_nonzeros=None if estimates is None else estimates[column],
                        current_nonzeros=combined_nonzeros
                        + (sum(estimates[item] for item in columns[:offset]) if estimates else 0),
                        sizing=sizing,
                    )
                    for offset, column in enumerate(columns)
                )
                if can_merge:
                    merged[-1] = ((min(previous_key[0], key[0]), key[1]), previous_columns + columns)
                    continue
            merged.append((key, columns))
        raw = merged

    blocks: list[ODBlock] = []
    for sequence, (key, columns) in enumerate(raw):
        block_support = (
            None
            if supports is None
            else tuple(sorted({row for column in columns for row in supports[column]}))
        )
        blocks.append(
            ODBlock(
                block_id=f"block-{sequence:06d}",
                free_column_indices=tuple(columns),
                active_od_indices=tuple(sorted(free_to_active[column] for column in columns)),
                destination_group_indices=tuple(sorted({groups[column] for column in columns})),
                time_bin_ids=tuple(sorted({time_bins[column] for column in columns})),
                estimated_nonzeros=(
                    None if estimates is None else sum(estimates[column] for column in columns)
                ),
                measurement_support_indices=block_support,
            )
        )
    return validate_block_partition(
        blocks,
        free_to_active_indices=free_to_active,
        sizing=sizing,
    )


def partition_assignment_od_blocks(
    *,
    inputs: AssignmentInputs,
    parameter_layout: ODParameterLayout,
    compact_layout: CompactODAssignmentLayout,
    sizing: BlockSizingConfig,
    estimated_nonzeros_by_free_column: object | None = None,
    measurement_support_by_free_column: object | None = None,
    merge_small_compatible: bool = False,
) -> ODBlockPartition:
    """Adapt canonical assignment/layout metadata to the default partitioner."""
    if compact_layout.num_free != parameter_layout.num_free:
        raise ValueError("parameter and compact layouts have inconsistent free dimensions.")
    if compact_layout.num_active != int(inputs.od_origin_node.shape[0]):
        raise ValueError("compact layout and assignment inputs have inconsistent active dimensions.")
    group_by_active = np.full(compact_layout.num_active, -1, dtype=np.int64)
    padded = np.asarray(inputs.group_od_index_padded, dtype=np.int64)
    mask = np.asarray(inputs.group_od_mask, dtype=bool)
    for group in range(padded.shape[0]):
        active = padded[group][mask[group]]
        if np.any(group_by_active[active] >= 0):
            raise ValueError("an active OD index appears in multiple destination groups.")
        group_by_active[active] = group
    free_active = tuple(int(value) for value in compact_layout.free_compact_indices)
    if any(group_by_active[index] < 0 for index in free_active):
        raise ValueError("a free OD index has no destination group.")
    time_bins = tuple(
        parameter_layout.od_keys[full_index][2]
        for full_index in parameter_layout.free_od_indices
    )
    return partition_od_blocks(
        free_to_active_indices=free_active,
        destination_group_by_free_column=tuple(
            int(group_by_active[index]) for index in free_active
        ),
        time_bin_by_free_column=time_bins,
        sizing=sizing,
        estimated_nonzeros_by_free_column=estimated_nonzeros_by_free_column,
        measurement_support_by_free_column=measurement_support_by_free_column,
        merge_small_compatible=merge_small_compatible,
    )
