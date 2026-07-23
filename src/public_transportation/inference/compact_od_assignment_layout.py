"""Compact OD representation used at the inference-to-assignment boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from public_transportation.inference.od_parameter_layout import ODParameterLayout


@dataclass(frozen=True, slots=True)
class CompactODAssignmentLayout:
    """Immutable mapping from estimation coordinates to active assignment cells.

    Every free OD cell remains active. Frozen cells remain active only when their
    fixed value is strictly positive. Consequently, structurally frozen-zero
    cells have no coordinate in the compact assignment demand vector.

    All ``*_full_*`` indices refer to scenario/assignment OD order. All
    ``*_compact_*`` indices refer to ``active_full_indices`` order.
    """

    num_od_total: int
    active_full_indices: tuple[int, ...]
    removed_zero_full_indices: tuple[int, ...]
    full_to_compact: tuple[int, ...]
    free_full_indices: tuple[int, ...]
    free_compact_indices: tuple[int, ...]
    free_baseline_values: tuple[float, ...]
    fixed_compact_indices: tuple[int, ...]
    fixed_compact_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.num_od_total < 0:
            raise ValueError("num_od_total must be non-negative.")
        if len(self.full_to_compact) != self.num_od_total:
            raise ValueError("full_to_compact length must equal num_od_total.")

        active = self.active_full_indices
        removed = self.removed_zero_full_indices
        if active != tuple(sorted(active)) or len(set(active)) != len(active):
            raise ValueError("active_full_indices must be unique and ascending.")
        if removed != tuple(sorted(removed)) or len(set(removed)) != len(removed):
            raise ValueError("removed_zero_full_indices must be unique and ascending.")
        if set(active) & set(removed):
            raise ValueError("active and removed full indices must be disjoint.")
        if set(active) | set(removed) != set(range(self.num_od_total)):
            raise ValueError("active and removed full indices must partition all OD cells.")

        expected_map = [-1] * self.num_od_total
        for compact_index, full_index in enumerate(active):
            expected_map[full_index] = compact_index
        if self.full_to_compact != tuple(expected_map):
            raise ValueError("full_to_compact is inconsistent with active_full_indices.")

        if len(self.free_full_indices) != len(self.free_compact_indices):
            raise ValueError("free full and compact index arrays must have equal length.")
        if len(self.free_compact_indices) != len(self.free_baseline_values):
            raise ValueError("free indices and baseline values must have equal length.")
        if len(self.fixed_compact_indices) != len(self.fixed_compact_values):
            raise ValueError("fixed compact indices and values must have equal length.")

        compact_range = set(range(len(active)))
        free_compact = set(self.free_compact_indices)
        fixed_compact = set(self.fixed_compact_indices)
        if len(free_compact) != len(self.free_compact_indices):
            raise ValueError("free_compact_indices contains duplicates.")
        if len(fixed_compact) != len(self.fixed_compact_indices):
            raise ValueError("fixed_compact_indices contains duplicates.")
        if free_compact & fixed_compact or free_compact | fixed_compact != compact_range:
            raise ValueError("free and fixed compact indices must partition active cells.")
        if tuple(self.full_to_compact[index] for index in self.free_full_indices) != (
            self.free_compact_indices
        ):
            raise ValueError("free full and compact indices are inconsistent.")
        if any(value <= 0.0 or not np.isfinite(value) for value in self.free_baseline_values):
            raise ValueError("free_baseline_values must be finite and strictly positive.")
        if any(value <= 0.0 or not np.isfinite(value) for value in self.fixed_compact_values):
            raise ValueError("fixed_compact_values must be finite and strictly positive.")

    @property
    def num_active(self) -> int:
        return len(self.active_full_indices)

    @property
    def num_removed_zero(self) -> int:
        return len(self.removed_zero_full_indices)

    @property
    def num_free(self) -> int:
        return len(self.free_compact_indices)

    @property
    def num_fixed_positive(self) -> int:
        return len(self.fixed_compact_indices)

    @property
    def fingerprint_payload_json(self) -> str:
        payload = {
            "version": 1,
            "num_od_total": self.num_od_total,
            "active_full_indices": self.active_full_indices,
            "removed_zero_full_indices": self.removed_zero_full_indices,
            "free_full_indices": self.free_full_indices,
            "free_compact_indices": self.free_compact_indices,
            "free_baseline_values": self.free_baseline_values,
            "fixed_compact_indices": self.fixed_compact_indices,
            "fixed_compact_values": self.fixed_compact_values,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.fingerprint_payload_json.encode("utf-8")).hexdigest()

    def assemble_compact_jax(self, free_log_deviation: object) -> jnp.ndarray:
        """Assemble only the active assignment vector, never the full OD vector."""
        z_free = jnp.asarray(free_log_deviation)
        if z_free.ndim != 1 or z_free.shape[0] != self.num_free:
            raise ValueError(
                f"free_log_deviation must have shape ({self.num_free},), got {z_free.shape}."
            )
        dtype = jnp.result_type(
            z_free.dtype,
            jnp.asarray(self.free_baseline_values).dtype,
            jnp.asarray(self.fixed_compact_values).dtype,
        )
        compact = jnp.zeros((self.num_active,), dtype=dtype)
        if self.num_free:
            free_values = jnp.asarray(self.free_baseline_values, dtype=dtype) * jnp.exp(z_free)
            compact = compact.at[jnp.asarray(self.free_compact_indices)].set(free_values)
        if self.num_fixed_positive:
            compact = compact.at[jnp.asarray(self.fixed_compact_indices)].set(
                jnp.asarray(self.fixed_compact_values, dtype=dtype)
            )
        return compact

    def reconstruct_full_numpy(self, free_log_deviation: object) -> np.ndarray:
        """Reconstruct full OD demand for persistence and reporting only."""
        z_free = np.asarray(free_log_deviation)
        if z_free.ndim not in (1, 2) or z_free.shape[-1] != self.num_free:
            raise ValueError(
                "free_log_deviation must have shape "
                f"({self.num_free},) or (S, {self.num_free}), got {z_free.shape}."
            )
        full = np.zeros(
            z_free.shape[:-1] + (self.num_od_total,),
            dtype=np.result_type(z_free.dtype, np.float64),
        )
        if self.num_free:
            full[..., np.asarray(self.free_full_indices)] = (
                np.asarray(self.free_baseline_values) * np.exp(z_free)
            )
        if self.num_fixed_positive:
            fixed_full_indices = np.asarray(self.active_full_indices)[
                np.asarray(self.fixed_compact_indices)
            ]
            full[..., fixed_full_indices] = np.asarray(self.fixed_compact_values)
        return full


def build_compact_od_assignment_layout(
    *, parameter_layout: ODParameterLayout
) -> CompactODAssignmentLayout:
    """Remove structurally frozen-zero cells from an OD parameter layout."""
    active_full_indices = tuple(
        sorted((*parameter_layout.free_od_indices, *parameter_layout.fixed_positive_indices))
    )
    full_to_compact_list = [-1] * parameter_layout.num_od_total
    for compact_index, full_index in enumerate(active_full_indices):
        full_to_compact_list[full_index] = compact_index

    fixed_value_by_full_index = dict(
        zip(
            parameter_layout.fixed_od_indices,
            parameter_layout.fixed_od_values,
            strict=True,
        )
    )
    free_compact_indices = tuple(
        full_to_compact_list[index] for index in parameter_layout.free_od_indices
    )
    fixed_compact_indices = tuple(
        full_to_compact_list[index] for index in parameter_layout.fixed_positive_indices
    )
    fixed_compact_values = tuple(
        fixed_value_by_full_index[index] for index in parameter_layout.fixed_positive_indices
    )

    return CompactODAssignmentLayout(
        num_od_total=parameter_layout.num_od_total,
        active_full_indices=active_full_indices,
        removed_zero_full_indices=parameter_layout.fixed_zero_indices,
        full_to_compact=tuple(full_to_compact_list),
        free_full_indices=parameter_layout.free_od_indices,
        free_compact_indices=free_compact_indices,
        free_baseline_values=parameter_layout.free_baseline_values,
        fixed_compact_indices=fixed_compact_indices,
        fixed_compact_values=fixed_compact_values,
    )
