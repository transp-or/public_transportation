"""Exact sparse-column equivalence for reduced measurement responses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _immutable_int(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be a one-dimensional integer array.")
    result = np.array(array, dtype=np.int64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ResponseEquivalence:
    """Canonical partition of free cells having exactly equal response columns."""

    class_by_cell: np.ndarray
    representative_cell_indices: np.ndarray
    member_indptr: np.ndarray
    member_cell_indices: np.ndarray

    def __post_init__(self) -> None:
        class_by_cell = _immutable_int(self.class_by_cell, "class_by_cell")
        representatives = _immutable_int(
            self.representative_cell_indices, "representative_cell_indices"
        )
        indptr = _immutable_int(self.member_indptr, "member_indptr")
        members = _immutable_int(self.member_cell_indices, "member_cell_indices")
        classes = representatives.size
        if indptr.size != classes + 1 or indptr[0] != 0 or indptr[-1] != members.size:
            raise ValueError("member_indptr does not delimit every member index.")
        if np.any(indptr[1:] < indptr[:-1]):
            raise ValueError("member_indptr must be nondecreasing.")
        if class_by_cell.size != members.size:
            raise ValueError("every free cell must occur in one equivalence class.")
        if members.size and not np.array_equal(np.sort(members), np.arange(members.size)):
            raise ValueError("member indices must partition the free cells.")
        if class_by_cell.size and (
            np.any(class_by_cell < 0) or np.any(class_by_cell >= classes)
        ):
            raise ValueError("class_by_cell contains an invalid class index.")
        for class_index in range(classes):
            selected = members[indptr[class_index] : indptr[class_index + 1]]
            if selected.size == 0 or representatives[class_index] != selected[0]:
                raise ValueError("the representative must be the first class member.")
            if np.any(class_by_cell[selected] != class_index):
                raise ValueError("class membership arrays are inconsistent.")
        object.__setattr__(self, "class_by_cell", class_by_cell)
        object.__setattr__(self, "representative_cell_indices", representatives)
        object.__setattr__(self, "member_indptr", indptr)
        object.__setattr__(self, "member_cell_indices", members)

    @property
    def number_of_cells(self) -> int:
        return int(self.class_by_cell.size)

    @property
    def number_of_classes(self) -> int:
        return int(self.representative_cell_indices.size)

    @property
    def compression_ratio(self) -> float:
        if self.number_of_cells == 0:
            return 1.0
        return self.number_of_classes / self.number_of_cells


def build_response_equivalence(
    *,
    number_of_cells: int,
    measurement_index: np.ndarray,
    cell_index: np.ndarray,
    values: np.ndarray,
) -> ResponseEquivalence:
    """Group sparse columns only when row indices and float64 bytes are equal."""
    if isinstance(number_of_cells, bool) or number_of_cells < 0:
        raise ValueError("number_of_cells must be non-negative.")
    rows = np.asarray(measurement_index, dtype=np.int64)
    columns = np.asarray(cell_index, dtype=np.int64)
    coefficients = np.asarray(values, dtype=np.float64)
    if rows.ndim != 1 or columns.ndim != 1 or coefficients.ndim != 1:
        raise ValueError("sparse response arrays must be one-dimensional.")
    if not (rows.size == columns.size == coefficients.size):
        raise ValueError("sparse response arrays must have equal length.")
    if columns.size and (np.any(columns < 0) or np.any(columns >= number_of_cells)):
        raise ValueError("cell_index contains an invalid free-cell index.")
    entries_by_cell: list[list[tuple[int, bytes]]] = [
        [] for _ in range(number_of_cells)
    ]
    for row, cell, coefficient in zip(
        rows, columns, coefficients, strict=True
    ):
        entries_by_cell[int(cell)].append(
            (int(row), np.float64(coefficient).tobytes())
        )
    signatures: dict[tuple[tuple[int, bytes], ...], list[int]] = {}
    for cell, entries in enumerate(entries_by_cell):
        signature = tuple(sorted(entries))
        signatures.setdefault(signature, []).append(cell)
    groups = sorted(signatures.values(), key=lambda group: group[0])
    class_by_cell = np.empty(number_of_cells, dtype=np.int64)
    representatives: list[int] = []
    members: list[int] = []
    indptr = [0]
    for class_index, group in enumerate(groups):
        representatives.append(group[0])
        members.extend(group)
        class_by_cell[group] = class_index
        indptr.append(len(members))
    return ResponseEquivalence(
        class_by_cell=class_by_cell,
        representative_cell_indices=np.asarray(representatives, dtype=np.int64),
        member_indptr=np.asarray(indptr, dtype=np.int64),
        member_cell_indices=np.asarray(members, dtype=np.int64),
    )
