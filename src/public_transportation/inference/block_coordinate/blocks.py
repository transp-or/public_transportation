"""Immutable OD-block descriptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import index

from ._canonical import fingerprint


def _integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    try:
        result = tuple(index(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be an iterable of integers.") from error
    if any(item < 0 for item in result):
        raise ValueError(f"{name} must contain non-negative values.")
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique and ascending.")
    return result


@dataclass(frozen=True, slots=True)
class ODBlock:
    """Deterministic subset of free OD coordinates using the complete network."""

    block_id: str
    free_column_indices: tuple[int, ...]
    active_od_indices: tuple[int, ...]
    destination_group_indices: tuple[int, ...]
    time_bin_ids: tuple[str, ...]
    estimated_nonzeros: int | None = None
    measurement_support_indices: tuple[int, ...] | None = None
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        block_id = str(self.block_id).strip()
        if not block_id:
            raise ValueError("block_id must be nonempty.")
        free = _integer_tuple(self.free_column_indices, name="free_column_indices")
        active = _integer_tuple(self.active_od_indices, name="active_od_indices")
        groups = _integer_tuple(
            self.destination_group_indices, name="destination_group_indices"
        )
        if not free:
            raise ValueError("a block must contain at least one free column.")
        if len(active) != len(free):
            raise ValueError("active_od_indices must correspond one-to-one with free columns.")
        time_bins = tuple(str(value).strip() for value in self.time_bin_ids)
        if any(not value for value in time_bins) or len(set(time_bins)) != len(time_bins):
            raise ValueError("time_bin_ids must be nonempty and unique.")
        if self.estimated_nonzeros is not None and self.estimated_nonzeros < 0:
            raise ValueError("estimated_nonzeros must be non-negative when provided.")
        support = (
            None
            if self.measurement_support_indices is None
            else _integer_tuple(
                self.measurement_support_indices,
                name="measurement_support_indices",
            )
        )
        payload = {
            "version": 1,
            "block_id": block_id,
            "free_column_indices": free,
            "active_od_indices": active,
            "destination_group_indices": groups,
            "time_bin_ids": time_bins,
            "estimated_nonzeros": self.estimated_nonzeros,
            "measurement_support_indices": support,
        }
        object.__setattr__(self, "block_id", block_id)
        object.__setattr__(self, "free_column_indices", free)
        object.__setattr__(self, "active_od_indices", active)
        object.__setattr__(self, "destination_group_indices", groups)
        object.__setattr__(self, "time_bin_ids", time_bins)
        object.__setattr__(self, "measurement_support_indices", support)
        object.__setattr__(self, "fingerprint", fingerprint(payload))

    @property
    def num_free_variables(self) -> int:
        return len(self.free_column_indices)
