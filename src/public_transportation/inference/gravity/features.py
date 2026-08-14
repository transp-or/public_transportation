"""Validated immutable inputs for reduced-dimensional gravity demand."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, cast

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


def _immutable_matrix(value: object, *, name: str, dtype: np.dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {array.shape}.")
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
    destination_time_group_index: np.ndarray | None = None
    zone_pair_index: np.ndarray | None = None
    custom_group_indices: Mapping[str, np.ndarray] = field(default_factory=dict)
    smooth_time_basis: np.ndarray | None = None

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
            "destination_time_group_index",
            "zone_pair_index",
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
        custom: dict[str, np.ndarray] = {}
        for name, value in self.custom_group_indices.items():
            if not name or name in self.available_mapping_names:
                raise ValueError(
                    f"custom group name {name!r} is empty or collides with a built-in mapping."
                )
            source = np.asarray(value)
            if source.dtype.kind not in "iu":
                raise TypeError(f"custom group {name!r} must contain integers.")
            prepared = _immutable_vector(
                source, name=f"custom_group_indices[{name!r}]", dtype=np.dtype(np.int64)
            )
            if prepared.size != cell_count or np.any(prepared < 0):
                raise ValueError(
                    f"custom group {name!r} must contain {cell_count} non-negative cells."
                )
            custom[name] = prepared
        object.__setattr__(self, "custom_group_indices", MappingProxyType(custom))
        if self.smooth_time_basis is not None:
            basis = _immutable_matrix(
                self.smooth_time_basis, name="smooth_time_basis", dtype=dtype
            )
            if basis.shape[0] != cell_count or basis.shape[1] == 0:
                raise ValueError(
                    "smooth_time_basis must contain one row per cell and at least one column."
                )
            if not np.all(np.isfinite(basis)):
                raise ValueError("smooth_time_basis must be finite.")
            object.__setattr__(self, "smooth_time_basis", basis)
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
    def available_mapping_names(self) -> tuple[str, ...]:
        return (
            "origin_index",
            "destination_index",
            "departure_time_index",
            "time_period_index",
            "origin_time_group_index",
            "origin_zone_index",
            "destination_zone_index",
            "destination_time_group_index",
            "zone_pair_index",
            "smooth_time_basis",
        )

    def mapping(self, name: str) -> np.ndarray | None:
        """Return a built-in or named custom cell mapping."""
        if name in self.available_mapping_names:
            return getattr(self, name)
        return self.custom_group_indices.get(name)

    def validate_mapping(
        self,
        name: str,
        *,
        group_count: int,
        constant_within_origin_time: bool = False,
        smooth_basis: bool = False,
    ) -> np.ndarray:
        """Validate one mapping exactly as declared by a model component."""
        values = self.mapping(name)
        if values is None:
            raise ValueError(f"feature mapping {name!r} is required by the model.")
        array = np.asarray(values)
        if smooth_basis:
            if array.ndim != 2 or array.shape != (self.num_cells, group_count):
                raise ValueError(
                    f"smooth basis {name!r} must have shape "
                    f"({self.num_cells}, {group_count})."
                )
            return array
        if array.ndim != 1 or array.shape != (self.num_cells,):
            raise ValueError(f"feature mapping {name!r} must contain {self.num_cells} cells.")
        if array.dtype.kind not in "iu":
            raise TypeError(f"feature mapping {name!r} must contain integers.")
        if not np.array_equal(np.unique(array), np.arange(group_count)):
            raise ValueError(
                f"feature mapping {name!r} must use every contiguous index from "
                f"zero to {group_count - 1}."
            )
        for group in range(group_count):
            if not np.any((array == group) & self.structural_feasible):
                raise ValueError(
                    f"feature mapping {name!r} group {group} has no structurally "
                    "feasible cell."
                )
        if constant_within_origin_time:
            for group in range(self.num_origin_time_groups):
                if np.unique(array[self.origin_time_group_index == group]).size != 1:
                    raise ValueError(
                        f"feature mapping {name!r} must be constant within each "
                        "origin-time group."
                    )
        return array

    @property
    def fingerprint(self) -> str:
        payload = {
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
        if (
            self.destination_time_group_index is not None
            or self.zone_pair_index is not None
            or self.custom_group_indices
            or self.smooth_time_basis is not None
        ):
            payload.update(
                {
                "schema_version": 2,
                "destination_time_group_index": self.destination_time_group_index,
                "zone_pair_index": self.zone_pair_index,
                "custom_group_indices": dict(self.custom_group_indices),
                "smooth_time_basis": self.smooth_time_basis,
                }
            )
        return fingerprint(payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
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
            "destination_time_group_index": (
                None
                if self.destination_time_group_index is None
                else self.destination_time_group_index.tolist()
            ),
            "zone_pair_index": (
                None if self.zone_pair_index is None else self.zone_pair_index.tolist()
            ),
            "custom_group_indices": {
                name: values.tolist()
                for name, values in self.custom_group_indices.items()
            },
            "smooth_time_basis": (
                None
                if self.smooth_time_basis is None
                else self.smooth_time_basis.tolist()
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GravityFeatures:
        if payload.get("schema_version") not in (1, 2):
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
            destination_time_group_index=(
                None
                if payload.get("destination_time_group_index") is None
                else np.asarray(payload["destination_time_group_index"])
            ),
            zone_pair_index=(
                None
                if payload.get("zone_pair_index") is None
                else np.asarray(payload["zone_pair_index"])
            ),
            custom_group_indices={
                str(name): np.asarray(values)
                for name, values in cast(
                    Mapping[str, object], payload.get("custom_group_indices", {})
                ).items()
            },
            smooth_time_basis=(
                None
                if payload.get("smooth_time_basis") is None
                else np.asarray(payload["smooth_time_basis"], dtype=dtype)
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
