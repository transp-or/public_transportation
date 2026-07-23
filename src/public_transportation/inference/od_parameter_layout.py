"""Immutable free/fixed OD parameter layout shared by estimation engines."""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from public_transportation.domain.fixed_demand import FixedODDemand, FixedODKey

if TYPE_CHECKING:  # pragma: no cover
    from public_transportation.domain.scenario import Scenario


@dataclass(frozen=True, slots=True)
class ODParameterLayout:
    """Canonical partition of scenario OD cells into free and fixed cells.

    All indices refer to the iteration order of ``scenario.demand.records``.
    Tuple storage keeps the layout immutable and serialization-friendly; callers
    may convert tuples to NumPy or JAX arrays at engine boundaries.
    """

    num_od_total: int
    od_keys: tuple[FixedODKey, ...]
    free_od_indices: tuple[int, ...]
    fixed_od_indices: tuple[int, ...]
    fixed_od_values: tuple[float, ...]
    free_baseline_values: tuple[float, ...]
    fixed_zero_indices: tuple[int, ...]
    fixed_positive_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.num_od_total < 0:
            raise ValueError("num_od_total must be non-negative.")
        if len(self.od_keys) != self.num_od_total:
            raise ValueError("od_keys length must equal num_od_total.")
        if len(set(self.od_keys)) != len(self.od_keys):
            raise ValueError("od_keys contains duplicates.")
        if len(self.free_od_indices) != len(self.free_baseline_values):
            raise ValueError("free indices and free baseline values must have equal length.")
        if len(self.fixed_od_indices) != len(self.fixed_od_values):
            raise ValueError("fixed indices and fixed values must have equal length.")

        free = set(self.free_od_indices)
        fixed = set(self.fixed_od_indices)
        expected = set(range(self.num_od_total))
        if len(free) != len(self.free_od_indices):
            raise ValueError("free_od_indices contains duplicates.")
        if len(fixed) != len(self.fixed_od_indices):
            raise ValueError("fixed_od_indices contains duplicates.")
        if free & fixed:
            raise ValueError("free and fixed OD indices must be disjoint.")
        if free | fixed != expected:
            raise ValueError("free and fixed OD indices must partition all OD cells.")
        if self.free_od_indices != tuple(sorted(self.free_od_indices)):
            raise ValueError("free_od_indices must be in canonical ascending order.")
        if self.fixed_od_indices != tuple(sorted(self.fixed_od_indices)):
            raise ValueError("fixed_od_indices must be in canonical ascending order.")

        if any(not math.isfinite(value) or value <= 0.0 for value in self.free_baseline_values):
            raise ValueError("free_baseline_values must be finite and strictly positive.")
        if any(not math.isfinite(value) or value < 0.0 for value in self.fixed_od_values):
            raise ValueError("fixed_od_values must be finite and non-negative.")

        fixed_zero = set(self.fixed_zero_indices)
        fixed_positive = set(self.fixed_positive_indices)
        if len(fixed_zero) != len(self.fixed_zero_indices):
            raise ValueError("fixed_zero_indices contains duplicates.")
        if len(fixed_positive) != len(self.fixed_positive_indices):
            raise ValueError("fixed_positive_indices contains duplicates.")
        if fixed_zero & fixed_positive:
            raise ValueError("fixed-zero and fixed-positive indices must be disjoint.")
        if fixed_zero | fixed_positive != fixed:
            raise ValueError("fixed-zero and fixed-positive indices must partition fixed cells.")
        fixed_value_by_index = dict(zip(self.fixed_od_indices, self.fixed_od_values, strict=True))
        expected_zero = {index for index, value in fixed_value_by_index.items() if value == 0.0}
        expected_positive = {index for index, value in fixed_value_by_index.items() if value > 0.0}
        if fixed_zero != expected_zero or fixed_positive != expected_positive:
            raise ValueError(
                "fixed-zero and fixed-positive indices must agree with fixed_od_values."
            )

    @property
    def num_free(self) -> int:
        return len(self.free_od_indices)

    @property
    def num_fixed(self) -> int:
        return len(self.fixed_od_indices)

    @property
    def num_fixed_zero(self) -> int:
        return len(self.fixed_zero_indices)

    @property
    def num_fixed_positive(self) -> int:
        return len(self.fixed_positive_indices)

    def parameter_dim(self, *, estimate_theta: bool) -> int:
        """Return the exact statistical parameter dimension for this layout."""
        return self.num_free + int(bool(estimate_theta))

    @property
    def fingerprint_payload_json(self) -> str:
        """Return the canonical serialized reduced-parameter contract."""
        payload = {
            "version": 1,
            "num_od_total": self.num_od_total,
            "od_keys": self.od_keys,
            "free_od_indices": self.free_od_indices,
            "fixed_od_indices": self.fixed_od_indices,
            "fixed_od_values": self.fixed_od_values,
            "free_baseline_values": self.free_baseline_values,
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @property
    def fingerprint(self) -> str:
        """Return a stable digest of the complete reduced-parameter contract."""
        return hashlib.sha256(self.fingerprint_payload_json.encode("utf-8")).hexdigest()

    def reconstruct_jax(self, free_log_deviation: object) -> jnp.ndarray:
        """Reconstruct full OD demand from free deviations and fixed constants.

        The input has exactly ``num_free`` entries. This full reconstruction is
        intended for reporting, persistence, and equivalence diagnostics. The
        estimator's assignment path uses ``CompactODAssignmentLayout`` instead.
        """
        z_free = jnp.asarray(free_log_deviation)
        if z_free.ndim != 1 or z_free.shape[0] != self.num_free:
            raise ValueError(
                "free_log_deviation must have shape "
                f"({self.num_free},), got {z_free.shape}."
            )
        dtype = jnp.result_type(z_free.dtype, jnp.asarray(self.free_baseline_values).dtype)
        full = jnp.zeros((self.num_od_total,), dtype=dtype)
        if self.num_free:
            free_values = jnp.asarray(self.free_baseline_values, dtype=dtype) * jnp.exp(z_free)
            full = full.at[jnp.asarray(self.free_od_indices, dtype=jnp.int32)].set(free_values)
        if self.num_fixed:
            full = full.at[jnp.asarray(self.fixed_od_indices, dtype=jnp.int32)].set(
                jnp.asarray(self.fixed_od_values, dtype=dtype)
            )
        return full

    def reconstruct_numpy(self, free_log_deviation: object) -> np.ndarray:
        """Vectorized NumPy reconstruction for one or more posterior draws."""
        z_free = np.asarray(free_log_deviation)
        if z_free.ndim not in (1, 2) or z_free.shape[-1] != self.num_free:
            raise ValueError(
                "free_log_deviation must have shape "
                f"({self.num_free},) or (S, {self.num_free}), got {z_free.shape}."
            )
        output_shape = z_free.shape[:-1] + (self.num_od_total,)
        full = np.zeros(output_shape, dtype=np.result_type(z_free.dtype, np.float64))
        if self.num_free:
            full[..., np.asarray(self.free_od_indices)] = (
                np.asarray(self.free_baseline_values) * np.exp(z_free)
            )
        if self.num_fixed:
            full[..., np.asarray(self.fixed_od_indices)] = np.asarray(self.fixed_od_values)
        return full


def assert_od_layout_fingerprint_matches(
    *,
    expected: str,
    got: str,
    context: str = "",
) -> None:
    """Raise a diagnostic error when reduced-layout identities differ."""
    if str(expected) == str(got):
        return
    suffix = f" ({context})" if context else ""
    raise ValueError(
        f"OD parameter layout fingerprint mismatch{suffix}: "
        f"expected={expected}, got={got}. The fixed-demand file, OD ordering, "
        "or free baselines differ."
    )


def _record_key(record: object) -> FixedODKey:
    return (
        str(getattr(record, "origin_stop_id")),
        str(getattr(record, "dest_stop_id")),
        str(getattr(record, "time_bin_id")),
    )


def build_od_parameter_layout(
    *,
    scenario: Scenario,
    fixed_demand: FixedODDemand | None = None,
) -> ODParameterLayout:
    """Build the canonical reduced OD parameter layout.

    Free cells must have a strictly positive finite baseline because the
    multiplicative parameterization cannot move a zero baseline. Frozen cells
    bypass that parameterization and may therefore have any finite nonnegative
    fixed value, independently of their baseline value.
    """
    demand_records = tuple(scenario.demand.records)
    od_keys = tuple(_record_key(record) for record in demand_records)

    first_index_by_key: dict[FixedODKey, int] = {}
    for index, key in enumerate(od_keys):
        if key in first_index_by_key:
            raise ValueError(
                f"scenario.demand.records contains duplicate OD/time-bin key {key!r} "
                f"at indices {first_index_by_key[key]} and {index}."
            )
        first_index_by_key[key] = index

    fixed_records = () if fixed_demand is None else fixed_demand.records
    fixed_by_key: dict[FixedODKey, float] = {}
    for record in fixed_records:
        key = record.key
        value = float(record.fixed_flow)
        if key in fixed_by_key:
            raise ValueError(f"fixed_demand contains duplicate OD/time-bin key {key!r}.")
        if key not in first_index_by_key:
            raise ValueError(
                f"fixed_demand OD/time-bin key {key!r} is not present in scenario.demand.records."
            )
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"fixed demand for key {key!r} must be finite and non-negative, got {value!r}."
            )
        fixed_by_key[key] = value

    free_indices: list[int] = []
    fixed_indices: list[int] = []
    fixed_values: list[float] = []
    free_baselines: list[float] = []
    fixed_zero_indices: list[int] = []
    fixed_positive_indices: list[int] = []

    for index, (record, key) in enumerate(zip(demand_records, od_keys, strict=True)):
        baseline = float(getattr(record, "flow"))
        if not math.isfinite(baseline) or baseline < 0.0:
            raise ValueError(
                f"scenario baseline demand for key {key!r} must be finite and non-negative, "
                f"got {baseline!r}."
            )

        if key in fixed_by_key:
            fixed_value = fixed_by_key[key]
            fixed_indices.append(index)
            fixed_values.append(fixed_value)
            if fixed_value == 0.0:
                fixed_zero_indices.append(index)
            else:
                fixed_positive_indices.append(index)
            continue

        if baseline == 0.0:
            raise ValueError(
                f"free OD/time-bin key {key!r} has a zero baseline; provide a positive "
                "baseline seed or freeze the cell."
            )
        free_indices.append(index)
        free_baselines.append(baseline)

    return ODParameterLayout(
        num_od_total=len(demand_records),
        od_keys=od_keys,
        free_od_indices=tuple(free_indices),
        fixed_od_indices=tuple(fixed_indices),
        fixed_od_values=tuple(fixed_values),
        free_baseline_values=tuple(free_baselines),
        fixed_zero_indices=tuple(fixed_zero_indices),
        fixed_positive_indices=tuple(fixed_positive_indices),
    )
