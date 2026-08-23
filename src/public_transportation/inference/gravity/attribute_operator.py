"""Route-share response operators for optional gravity observations.

The APC measurement operator maps free OD--time demand to boarding and
alighting measurements.  This module defines the analogous, independent
operator for an aggregate journey-attribute histogram.  A column is built
from *route shares*, not from a single mean journey-time feature: if an OD cell
has two assigned paths, both paths contribute to their respective attribute
categories.

The concrete implementation supports a small dense reference representation
and a category-sharded representation.  Both have the same JAX-compatible
forward and adjoint products, so a later routing-preparation backend can emit
shards without changing the gravity objective.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.inference.block_coordinate._canonical import fingerprint

from .aggregate import GravityAggregateHistogram, GravityAggregateObservation

Array = jax.Array
ResponseRepresentation = Literal["dense", "sharded"]


def _immutable_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got {array.shape}.")
    array.setflags(write=False)
    return array


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True, slots=True)
class GravityAttributeResponseProvenance:
    """Identity of the routing inputs used to construct a response operator."""

    od_layout_fingerprint: str
    assignment_fingerprint: str
    graph_fingerprint: str = "unspecified"
    timetable_fingerprint: str = "unspecified"
    feasibility_fingerprint: str = "unspecified"

    def __post_init__(self) -> None:
        for name in (
            "od_layout_fingerprint",
            "assignment_fingerprint",
            "graph_fingerprint",
            "timetable_fingerprint",
            "feasibility_fingerprint",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))

    def to_dict(self) -> dict[str, str]:
        return {
            "od_layout_fingerprint": self.od_layout_fingerprint,
            "assignment_fingerprint": self.assignment_fingerprint,
            "graph_fingerprint": self.graph_fingerprint,
            "timetable_fingerprint": self.timetable_fingerprint,
            "feasibility_fingerprint": self.feasibility_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class GravityRouteShare:
    """One route's contribution for one free OD--time cell.

    ``share`` is a probability/mass in the cell's route-choice distribution.
    ``stratum`` and ``bin_label`` identify the category to which this route is
    assigned after evaluating the route's journey attributes.
    """

    stratum: str
    bin_label: str
    share: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "stratum", _nonempty(self.stratum, name="stratum"))
        object.__setattr__(
            self, "bin_label", _nonempty(self.bin_label, name="bin_label")
        )
        value = float(self.share)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("route share must be finite and non-negative.")
        object.__setattr__(self, "share", value)


@dataclass(frozen=True, slots=True)
class GravityAttributeSupportError(ValueError):
    """Strict preflight failure for positive observations without route support."""

    report: Mapping[str, object]

    def __post_init__(self) -> None:
        ValueError.__init__(self, self.__str__())

    def __str__(self) -> str:
        unsupported = self.report.get("unsupported_positive_mass", 0)
        return (
            "aggregate attribute observations contain unsupported positive bins "
            f"(mass {unsupported})."
        )


@dataclass(frozen=True, slots=True)
class GravityAttributeResponseOperator:
    """Map free OD demand to route-share-weighted attribute-bin mass.

    Categories are flattened in the declared order of ``category_labels``.
    The operator has shape ``(num_categories, num_free_od)`` and includes the
    contribution of fixed positive demand in ``fixed_attribute_offset``.
    Exactly one of ``matrix`` and ``shards`` must be supplied.  Shards are
    contiguous category blocks and are intentionally kept separate so large
    response artifacts need not be assembled into one dense array.
    """

    category_labels: tuple[tuple[str, str], ...]
    num_free_od: int
    fixed_attribute_offset: np.ndarray
    provenance: GravityAttributeResponseProvenance
    attribute: str = "unspecified"
    unit: str = "unspecified"
    matrix: np.ndarray | None = None
    shards: tuple[np.ndarray, ...] = ()
    representation: ResponseRepresentation = "dense"
    route_mass_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "attribute", _nonempty(self.attribute, name="attribute")
        )
        object.__setattr__(self, "unit", _nonempty(self.unit, name="unit"))
        labels = tuple(
            (
                _nonempty(item[0], name="category stratum"),
                _nonempty(item[1], name="category bin"),
            )
            for item in self.category_labels
        )
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("category_labels must be non-empty and unique.")
        object.__setattr__(self, "category_labels", labels)
        if not isinstance(self.num_free_od, (int, np.integer)) or self.num_free_od <= 0:
            raise ValueError("num_free_od must be a positive integer.")
        offset = _immutable_array(
            self.fixed_attribute_offset,
            name="fixed_attribute_offset",
            ndim=1,
        )
        if offset.shape != (len(labels),):
            raise ValueError(
                "fixed_attribute_offset must have one value per category, "
                f"got {offset.shape} for {len(labels)} categories."
            )
        if not np.all(np.isfinite(offset)) or np.any(offset < 0.0):
            raise ValueError("fixed_attribute_offset must be finite and non-negative.")
        object.__setattr__(self, "fixed_attribute_offset", offset)
        if self.representation not in {"dense", "sharded"}:
            raise ValueError("representation must be 'dense' or 'sharded'.")
        tolerance = float(self.route_mass_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("route_mass_tolerance must be finite and non-negative.")
        object.__setattr__(self, "route_mass_tolerance", tolerance)
        has_matrix = self.matrix is not None
        has_shards = bool(self.shards)
        if has_matrix == has_shards:
            raise ValueError("provide exactly one of matrix or shards.")
        if self.representation == "dense" and not has_matrix:
            raise ValueError("dense representation requires matrix.")
        if self.representation == "sharded" and not has_shards:
            raise ValueError("sharded representation requires shards.")
        if has_matrix:
            matrix = _immutable_array(self.matrix, name="matrix", ndim=2)
            if matrix.shape != (len(labels), self.num_free_od):
                raise ValueError(
                    "matrix must have shape "
                    f"({len(labels)}, {self.num_free_od}), got {matrix.shape}."
                )
            arrays = (matrix,)
            object.__setattr__(self, "matrix", matrix)
        else:
            prepared = tuple(
                _immutable_array(item, name=f"shards[{index}]", ndim=2)
                for index, item in enumerate(self.shards)
            )
            if any(item.shape[1] != self.num_free_od for item in prepared):
                raise ValueError("every response shard must have num_free_od columns.")
            if sum(item.shape[0] for item in prepared) != len(labels):
                raise ValueError(
                    "response shards must cover every category exactly once."
                )
            arrays = prepared
            object.__setattr__(self, "shards", prepared)
        for array in arrays:
            if not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError(
                    "attribute response coefficients must be finite and non-negative."
                )
        if not isinstance(self.provenance, GravityAttributeResponseProvenance):
            raise TypeError("provenance must be GravityAttributeResponseProvenance.")

    @property
    def num_measurements(self) -> int:
        """Number of flattened stratum/bin categories."""

        return len(self.category_labels)

    @property
    def fixed_measurement_offset(self) -> np.ndarray:
        """Compatibility alias matching the APC operator protocol."""

        return self.fixed_attribute_offset

    @property
    def is_matrix_free(self) -> bool:
        return self.representation == "sharded"

    @property
    def supported_category_labels(self) -> tuple[tuple[str, str], ...]:
        """Categories supported by free routes or fixed positive demand."""

        supported: set[tuple[str, str]] = set()
        start = 0
        for shard in self._arrays():
            mask = np.any(shard > self.route_mass_tolerance, axis=1)
            stop = start + shard.shape[0]
            supported.update(
                label
                for label, enabled in zip(
                    self.category_labels[start:stop], mask, strict=True
                )
                if enabled
            )
            start = stop
        supported.update(
            label
            for label, value in zip(
                self.category_labels,
                self.fixed_attribute_offset,
                strict=True,
            )
            if value > self.route_mass_tolerance
        )
        return tuple(label for label in self.category_labels if label in supported)

    @property
    def route_share_mass(self) -> np.ndarray:
        """Total route-share mass in each free OD--time column."""

        mass = np.zeros(self.num_free_od, dtype=np.float64)
        for shard in self._arrays():
            mass += shard.sum(axis=0)
        return mass

    @property
    def fingerprint(self) -> str:
        """Identity of the attribute response, separate from the APC operator."""

        payload: dict[str, object] = {
            "schema_version": 1,
            "attribute": self.attribute,
            "unit": self.unit,
            "category_labels": [list(item) for item in self.category_labels],
            "num_free_od": self.num_free_od,
            "fixed_attribute_offset": self.fixed_attribute_offset.tolist(),
            "provenance": self.provenance.to_dict(),
            "representation": self.representation,
            "route_mass_tolerance": self.route_mass_tolerance,
            "arrays": [
                {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": hashlib.sha256(
                        np.ascontiguousarray(array).tobytes()
                    ).hexdigest(),
                }
                for array in self._arrays()
            ],
        }
        return fingerprint(payload)

    def _arrays(self) -> tuple[np.ndarray, ...]:
        return (self.matrix,) if self.matrix is not None else self.shards

    def _check_vector(self, vector: object, *, length: int, name: str) -> Array:
        value = jnp.asarray(vector)
        if value.shape != (length,):
            raise ValueError(f"{name} must have shape ({length},), got {value.shape}.")
        return value

    def jax_matvec(self, vector: Array) -> Array:
        """Return ``B @ vector`` without adding the fixed-demand offset."""

        value = self._check_vector(
            vector, length=self.num_free_od, name="forward vector"
        )
        if self.matrix is not None:
            return jnp.asarray(self.matrix, dtype=value.dtype) @ value
        pieces = [
            jnp.asarray(shard, dtype=value.dtype) @ value for shard in self.shards
        ]
        return jnp.concatenate(pieces, axis=0)

    def jax_rmatvec(self, vector: Array) -> Array:
        """Return ``B.T @ vector``; fixed demand has no adjoint parameter."""

        value = self._check_vector(
            vector, length=self.num_measurements, name="transpose vector"
        )
        if self.matrix is not None:
            return jnp.asarray(self.matrix, dtype=value.dtype).T @ value
        result = jnp.zeros((self.num_free_od,), dtype=value.dtype)
        start = 0
        for shard in self.shards:
            stop = start + shard.shape[0]
            result = (
                result + jnp.asarray(shard, dtype=value.dtype).T @ value[start:stop]
            )
            start = stop
        return result

    def validate_route_share_mass(
        self,
        *,
        expected: Sequence[float] | None = None,
        require_unit_mass: bool = False,
    ) -> np.ndarray:
        """Validate per-cell route-share mass and return it.

        A cell may have less than unit mass when the routing contract
        explicitly permits an unserved/unsupported journey.  Set
        ``require_unit_mass`` when the declared route categories cover every
        feasible path and unit mass is required.
        """

        mass = self.route_share_mass
        if np.any(mass < -self.route_mass_tolerance) or np.any(
            mass > 1.0 + self.route_mass_tolerance
        ):
            raise ValueError(
                "route-share mass must lie in [0, 1] for every free OD cell."
            )
        if expected is not None:
            target = np.asarray(expected, dtype=np.float64)
            if target.shape != mass.shape:
                raise ValueError("expected route-share mass has the wrong dimension.")
            if not np.allclose(mass, target, rtol=0.0, atol=self.route_mass_tolerance):
                raise ValueError(
                    "route-share mass does not match the expected cell mass."
                )
        if require_unit_mass and not np.allclose(
            mass, 1.0, rtol=0.0, atol=self.route_mass_tolerance
        ):
            raise ValueError("every free OD cell must have unit route-share mass.")
        return mass

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "attribute": self.attribute,
            "unit": self.unit,
            "category_labels": [list(item) for item in self.category_labels],
            "num_free_od": self.num_free_od,
            "fixed_attribute_offset": self.fixed_attribute_offset.tolist(),
            "provenance": self.provenance.to_dict(),
            "representation": self.representation,
            "route_mass_tolerance": self.route_mass_tolerance,
            "fingerprint": self.fingerprint,
        }

    def save(self, path: str | Path) -> Path:
        """Persist the response arrays and identity metadata in an NPZ artifact."""

        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        arrays: dict[str, object] = {
            "metadata": np.asarray(metadata),
            "fixed_attribute_offset": np.asarray(self.fixed_attribute_offset),
        }
        if self.matrix is not None:
            arrays["matrix"] = np.asarray(self.matrix)
        else:
            arrays.update(
                {f"shard_{index:06d}": array for index, array in enumerate(self.shards)}
            )
        np.savez_compressed(target, **arrays)
        return target

    @classmethod
    def load(cls, path: str | Path) -> GravityAttributeResponseOperator:
        """Load an artifact and verify its embedded semantic fingerprint."""

        source = Path(path).expanduser()
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            provenance = GravityAttributeResponseProvenance(**metadata["provenance"])
            labels = tuple(tuple(item) for item in metadata["category_labels"])
            representation = metadata["representation"]
            if representation == "dense":
                operator = cls(
                    category_labels=labels,
                    num_free_od=int(metadata["num_free_od"]),
                    fixed_attribute_offset=archive["fixed_attribute_offset"],
                    provenance=provenance,
                    attribute=str(metadata["attribute"]),
                    unit=str(metadata["unit"]),
                    matrix=archive["matrix"],
                    representation="dense",
                    route_mass_tolerance=float(metadata["route_mass_tolerance"]),
                )
            else:
                names = sorted(
                    name for name in archive.files if name.startswith("shard_")
                )
                operator = cls(
                    category_labels=labels,
                    num_free_od=int(metadata["num_free_od"]),
                    fixed_attribute_offset=archive["fixed_attribute_offset"],
                    provenance=provenance,
                    attribute=str(metadata["attribute"]),
                    unit=str(metadata["unit"]),
                    shards=tuple(archive[name] for name in names),
                    representation="sharded",
                    route_mass_tolerance=float(metadata["route_mass_tolerance"]),
                )
        if operator.fingerprint != metadata.get("fingerprint"):
            raise ValueError(
                "attribute response artifact fingerprint does not match its metadata."
            )
        return operator

    @classmethod
    def from_route_shares(
        cls,
        *,
        attribute: str = "unspecified",
        unit: str = "unspecified",
        category_labels: Sequence[tuple[str, str]],
        route_shares: Sequence[Sequence[GravityRouteShare]],
        fixed_attribute_offset: object,
        provenance: GravityAttributeResponseProvenance,
        representation: ResponseRepresentation = "dense",
        shard_size: int = 0,
        route_mass_tolerance: float = 1.0e-10,
    ) -> GravityAttributeResponseOperator:
        """Construct ``B`` from per-cell route-share records.

        ``route_shares[j]`` contains every assigned path for free cell ``j``.
        Contributions with the same category are summed.  No journey-time or
        transfer-count mean is used, so a multi-route cell retains its full
        categorical distribution.
        """

        labels = tuple(tuple(item) for item in category_labels)
        index = {label: position for position, label in enumerate(labels)}
        if not labels:
            raise ValueError("category_labels must not be empty.")
        matrix = np.zeros((len(labels), len(route_shares)), dtype=np.float64)
        for cell, records in enumerate(route_shares):
            total = 0.0
            for raw_record in records:
                record = (
                    raw_record
                    if isinstance(raw_record, GravityRouteShare)
                    else GravityRouteShare(*raw_record)
                    if isinstance(raw_record, (tuple, list)) and len(raw_record) == 3
                    else None
                )
                if not isinstance(record, GravityRouteShare):
                    raise TypeError(
                        "route_shares entries must be GravityRouteShare or (stratum, bin_label, share) values."
                    )
                key = (record.stratum, record.bin_label)
                if key not in index:
                    raise ValueError(f"route share uses undeclared category {key!r}.")
                matrix[index[key], cell] += record.share
                total += record.share
            if total > 1.0 + route_mass_tolerance:
                raise ValueError(f"route shares for free cell {cell} exceed unit mass.")
        kwargs: dict[str, object] = {
            "category_labels": labels,
            "num_free_od": len(route_shares),
            "fixed_attribute_offset": fixed_attribute_offset,
            "provenance": provenance,
            "attribute": attribute,
            "unit": unit,
            "route_mass_tolerance": route_mass_tolerance,
            "representation": representation,
        }
        if representation == "dense":
            kwargs["matrix"] = matrix
        else:
            if not isinstance(shard_size, (int, np.integer)) or shard_size <= 0:
                raise ValueError(
                    "shard_size must be positive for sharded representation."
                )
            kwargs["shards"] = tuple(
                matrix[start : start + shard_size]
                for start in range(0, matrix.shape[0], shard_size)
            )
        return cls(**kwargs)


def validate_aggregate_support(
    operator: GravityAttributeResponseOperator,
    histogram: GravityAggregateHistogram,
    *,
    aggregate: GravityAggregateObservation | None = None,
) -> dict[str, object]:
    """Audit positive histogram bins against route support before estimation.

    Zero observations in an unsupported category are harmless.  A positive
    count in a category with no route contribution is a strict preflight
    failure and is never silently discarded.
    """

    expected_bins = tuple(item.label for item in histogram.bins)
    if (
        operator.attribute != "unspecified"
        and operator.attribute != histogram.attribute
    ):
        raise ValueError(
            "attribute response attribute does not match observed histogram: "
            f"{operator.attribute!r} != {histogram.attribute!r}."
        )
    if operator.unit != "unspecified" and operator.unit != histogram.unit:
        raise ValueError(
            "attribute response unit does not match observed histogram: "
            f"{operator.unit!r} != {histogram.unit!r}."
        )
    operator_bins = {label for _, label in operator.category_labels}
    operator_strata = {stratum for stratum, _ in operator.category_labels}
    supported = set(operator.supported_category_labels)
    supported_mass = 0
    unsupported_mass = 0
    unsupported: list[dict[str, object]] = []
    total = 0
    for stratum in histogram.strata:
        for label, count in zip(expected_bins, stratum.counts, strict=True):
            total += count
            key = (stratum.name, label)
            if key in supported:
                supported_mass += count
            else:
                unsupported_mass += count
                if count > 0:
                    reason = (
                        "stratum absent from route response"
                        if stratum.name not in operator_strata
                        else "attribute bin absent from route response"
                    )
                    unsupported.append(
                        {
                            "stratum": stratum.name,
                            "bin_label": label,
                            "observed_mass": count,
                            "cause": reason,
                        }
                    )
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "supported" if not unsupported else "rejected",
        "attribute": histogram.attribute,
        "unit": histogram.unit,
        "total_observed_mass": total,
        "supported_observed_mass": supported_mass,
        "unsupported_positive_mass": unsupported_mass,
        "unsupported_positive_bins": unsupported,
        "operator_fingerprint": operator.fingerprint,
    }
    if aggregate is not None:
        report["aggregate_fingerprint"] = aggregate.fingerprint
    # Keep diagnostics explicit when the operator's declared bin vocabulary
    # differs from the observed histogram, even if all observed counts are 0.
    report["observed_bin_labels"] = list(expected_bins)
    report["operator_bin_labels"] = sorted(operator_bins)
    if unsupported:
        raise GravityAttributeSupportError(report)
    return report


__all__ = [
    "GravityAttributeResponseOperator",
    "GravityAttributeResponseProvenance",
    "GravityAttributeSupportError",
    "GravityRouteShare",
    "validate_aggregate_support",
]
