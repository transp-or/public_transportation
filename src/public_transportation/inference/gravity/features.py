"""Validated immutable inputs for reduced-dimensional gravity demand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from public_transportation.inference.block_coordinate._canonical import fingerprint
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)


def _immutable_vector(value: object, *, name: str, dtype: np.dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class GravityFeatures:
    """Sparse canonical cell list and externally prepared gravity offsets."""

    origin_index: np.ndarray
    canonical_od_index: np.ndarray
    destination_index: np.ndarray
    departure_time_index: np.ndarray
    origin_time_group_index: np.ndarray
    journey_time: np.ndarray
    transfer_count: np.ndarray
    structural_feasible: np.ndarray
    origin_time_totals: np.ndarray
    destination_attractiveness: np.ndarray
    num_origins: int
    num_destinations: int
    num_departure_times: int
    od_layout_fingerprint: str
    journey_time_scale: float = 1.0
    initial_waiting_time: np.ndarray | None = None
    origin_zone_index: np.ndarray | None = None
    destination_zone_index: np.ndarray | None = None
    time_period_index: np.ndarray | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            "origin_index",
            "canonical_od_index",
            "destination_index",
            "departure_time_index",
            "origin_time_group_index",
            "transfer_count",
        )
        for name in integer_fields:
            source = np.asarray(getattr(self, name))
            if source.dtype.kind not in "iu":
                raise TypeError(f"{name} must contain integers.")
            object.__setattr__(
                self,
                name,
                _immutable_vector(source, name=name, dtype=np.dtype(np.int64)),
            )
        journey_source = np.asarray(self.journey_time)
        totals_source = np.asarray(self.origin_time_totals)
        attractiveness_source = np.asarray(self.destination_attractiveness)
        if journey_source.dtype.kind not in "f":
            raise TypeError("journey_time must use a floating-point dtype.")
        dtype = journey_source.dtype
        for name, source in (
            ("journey_time", journey_source),
            ("origin_time_totals", totals_source),
            ("destination_attractiveness", attractiveness_source),
        ):
            if source.dtype.kind not in "iuf":
                raise TypeError(f"{name} must contain real numeric values.")
            object.__setattr__(
                self, name, _immutable_vector(source, name=name, dtype=dtype)
            )
        feasible = _immutable_vector(
            self.structural_feasible,
            name="structural_feasible",
            dtype=np.dtype(np.bool_),
        )
        object.__setattr__(self, "structural_feasible", feasible)

        cell_count = self.origin_index.size
        cell_fields = (
            "destination_index",
            "departure_time_index",
            "origin_time_group_index",
            "journey_time",
            "transfer_count",
            "structural_feasible",
            "destination_attractiveness",
        )
        for name in cell_fields:
            if getattr(self, name).size != cell_count:
                raise ValueError(f"{name} must contain {cell_count} cells.")
        for name in (
            "initial_waiting_time",
            "origin_zone_index",
            "destination_zone_index",
            "time_period_index",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            source = np.asarray(value)
            if name == "initial_waiting_time":
                if source.dtype.kind not in "iuf":
                    raise TypeError("initial_waiting_time must contain real values.")
                prepared = _immutable_vector(source, name=name, dtype=dtype)
                if not np.all(np.isfinite(prepared)) or np.any(prepared < 0):
                    raise ValueError(
                        "initial_waiting_time must be finite and non-negative."
                    )
            else:
                if source.dtype.kind not in "iu":
                    raise TypeError(f"{name} must contain integers.")
                prepared = _immutable_vector(
                    source, name=name, dtype=np.dtype(np.int64)
                )
                if np.any(prepared < 0):
                    raise ValueError(f"{name} must be non-negative.")
            if prepared.size != cell_count:
                raise ValueError(f"{name} must contain {cell_count} cells.")
            object.__setattr__(self, name, prepared)
        for name in ("num_origins", "num_destinations", "num_departure_times"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name, values, upper in (
            ("origin_index", self.origin_index, self.num_origins),
            ("destination_index", self.destination_index, self.num_destinations),
            (
                "departure_time_index",
                self.departure_time_index,
                self.num_departure_times,
            ),
            (
                "origin_time_group_index",
                self.origin_time_group_index,
                self.origin_time_totals.size,
            ),
        ):
            if np.any(values < 0) or np.any(values >= upper):
                raise ValueError(f"{name} contains an out-of-bounds index.")
        if not np.all(np.isfinite(self.journey_time)) or np.any(self.journey_time < 0):
            raise ValueError("journey_time must be finite and non-negative.")
        if np.any(self.transfer_count < 0):
            raise ValueError("transfer_count must be non-negative.")
        if not np.all(np.isfinite(self.origin_time_totals)) or np.any(
            self.origin_time_totals < 0
        ):
            raise ValueError("origin_time_totals must be finite and non-negative.")
        if not np.all(np.isfinite(self.destination_attractiveness)) or np.any(
            self.destination_attractiveness <= 0
        ):
            raise ValueError(
                "destination_attractiveness must be finite and strictly positive."
            )
        if not np.isfinite(self.journey_time_scale) or self.journey_time_scale <= 0:
            raise ValueError("journey_time_scale must be finite and positive.")
        if not self.od_layout_fingerprint:
            raise ValueError("od_layout_fingerprint must be nonempty.")
        if len(set(self.canonical_od_index.tolist())) != cell_count:
            raise ValueError("canonical_od_index must contain unique indices.")
        if np.any(self.canonical_od_index < 0):
            raise ValueError("canonical_od_index must be non-negative.")
        cells = list(
            zip(
                self.origin_index.tolist(),
                self.destination_index.tolist(),
                self.departure_time_index.tolist(),
                strict=True,
            )
        )
        if len(set(cells)) != cell_count:
            raise ValueError("origin-destination-time cells must be unique.")
        group_pairs: dict[int, tuple[int, int]] = {}
        pair_groups: dict[tuple[int, int], int] = {}
        for origin, departure, group in zip(
            self.origin_index,
            self.departure_time_index,
            self.origin_time_group_index,
            strict=True,
        ):
            pair = (int(origin), int(departure))
            group_value = int(group)
            if group_value in group_pairs and group_pairs[group_value] != pair:
                raise ValueError("an origin-time group cannot combine different pairs.")
            if pair in pair_groups and pair_groups[pair] != group_value:
                raise ValueError("an origin-time pair cannot use multiple groups.")
            group_pairs[group_value] = pair
            pair_groups[pair] = group_value
        feasible_counts = np.bincount(
            self.origin_time_group_index[self.structural_feasible],
            minlength=self.origin_time_totals.size,
        )
        represented_counts = np.bincount(
            self.origin_time_group_index, minlength=self.origin_time_totals.size
        )
        if np.any(represented_counts == 0):
            raise ValueError("every origin-time group must have at least one cell.")
        if np.any(feasible_counts == 0):
            raise ValueError(
                "every origin-time group must have at least one feasible destination."
            )

    @property
    def num_cells(self) -> int:
        return int(self.origin_index.size)

    @property
    def num_origin_time_groups(self) -> int:
        return int(self.origin_time_totals.size)

    @property
    def dtype(self) -> np.dtype:
        return self.journey_time.dtype

    @property
    def fingerprint(self) -> str:
        return fingerprint(
            {
                "schema_version": 1,
                "origin_index": self.origin_index,
                "canonical_od_index": self.canonical_od_index,
                "destination_index": self.destination_index,
                "departure_time_index": self.departure_time_index,
                "origin_time_group_index": self.origin_time_group_index,
                "journey_time": self.journey_time,
                "transfer_count": self.transfer_count,
                "structural_feasible": self.structural_feasible,
                "origin_time_totals": self.origin_time_totals,
                "destination_attractiveness": self.destination_attractiveness,
                "num_origins": self.num_origins,
                "num_destinations": self.num_destinations,
                "num_departure_times": self.num_departure_times,
                "od_layout_fingerprint": self.od_layout_fingerprint,
                "journey_time_scale": self.journey_time_scale,
                "dtype": str(self.dtype),
                "initial_waiting_time": self.initial_waiting_time,
                "origin_zone_index": self.origin_zone_index,
                "destination_zone_index": self.destination_zone_index,
                "time_period_index": self.time_period_index,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "canonical_od_index": self.canonical_od_index.tolist(),
            "origin_index": self.origin_index.tolist(),
            "destination_index": self.destination_index.tolist(),
            "departure_time_index": self.departure_time_index.tolist(),
            "origin_time_group_index": self.origin_time_group_index.tolist(),
            "journey_time": self.journey_time.tolist(),
            "transfer_count": self.transfer_count.tolist(),
            "structural_feasible": self.structural_feasible.tolist(),
            "origin_time_totals": self.origin_time_totals.tolist(),
            "destination_attractiveness": self.destination_attractiveness.tolist(),
            "num_origins": self.num_origins,
            "num_destinations": self.num_destinations,
            "num_departure_times": self.num_departure_times,
            "od_layout_fingerprint": self.od_layout_fingerprint,
            "journey_time_scale": self.journey_time_scale,
            "dtype": str(self.dtype),
            "initial_waiting_time": (
                None
                if self.initial_waiting_time is None
                else self.initial_waiting_time.tolist()
            ),
            "origin_zone_index": (
                None
                if self.origin_zone_index is None
                else self.origin_zone_index.tolist()
            ),
            "destination_zone_index": (
                None
                if self.destination_zone_index is None
                else self.destination_zone_index.tolist()
            ),
            "time_period_index": (
                None
                if self.time_period_index is None
                else self.time_period_index.tolist()
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GravityFeatures:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported gravity feature schema version.")
        dtype = np.dtype(str(payload["dtype"]))
        return cls(
            canonical_od_index=np.asarray(payload["canonical_od_index"]),
            origin_index=np.asarray(payload["origin_index"]),
            destination_index=np.asarray(payload["destination_index"]),
            departure_time_index=np.asarray(payload["departure_time_index"]),
            origin_time_group_index=np.asarray(payload["origin_time_group_index"]),
            journey_time=np.asarray(payload["journey_time"], dtype=dtype),
            transfer_count=np.asarray(payload["transfer_count"]),
            structural_feasible=np.asarray(payload["structural_feasible"]),
            origin_time_totals=np.asarray(payload["origin_time_totals"], dtype=dtype),
            destination_attractiveness=np.asarray(
                payload["destination_attractiveness"], dtype=dtype
            ),
            num_origins=int(cast(Any, payload["num_origins"])),
            num_destinations=int(cast(Any, payload["num_destinations"])),
            num_departure_times=int(cast(Any, payload["num_departure_times"])),
            od_layout_fingerprint=str(payload["od_layout_fingerprint"]),
            journey_time_scale=float(cast(Any, payload["journey_time_scale"])),
            initial_waiting_time=(
                None
                if payload.get("initial_waiting_time") is None
                else np.asarray(payload["initial_waiting_time"], dtype=dtype)
            ),
            origin_zone_index=(
                None
                if payload.get("origin_zone_index") is None
                else np.asarray(payload["origin_zone_index"])
            ),
            destination_zone_index=(
                None
                if payload.get("destination_zone_index") is None
                else np.asarray(payload["destination_zone_index"])
            ),
            time_period_index=(
                None
                if payload.get("time_period_index") is None
                else np.asarray(payload["time_period_index"])
            ),
        )

    def validate_compact_layout(self, layout: CompactODAssignmentLayout) -> None:
        """Verify exact free-cell order against the assignment boundary."""
        if self.od_layout_fingerprint != layout.fingerprint:
            raise ValueError(
                "gravity features and compact OD layout fingerprints differ."
            )
        if tuple(self.canonical_od_index.tolist()) != layout.free_full_indices:
            raise ValueError(
                "gravity canonical cells do not match compact free-cell order."
            )
