"""Transforms and entropy warm starts for minimal reduced gravity."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .features import ConditionalGravityFeatures
from .specification import MinimalGravitySpecification


@dataclass(frozen=True, slots=True)
class MinimalGravityParameterLayout:
    specification: MinimalGravitySpecification

    @property
    def size(self) -> int:
        return self.specification.parameter_count

    @property
    def dispersion_index(self) -> int | None:
        return 2 if self.specification.likelihood == "negative_binomial" else None

    @property
    def production_slice(self) -> slice:
        start = 3 if self.dispersion_index is not None else 2
        return slice(start, start + self.specification.production_basis_columns)

    @property
    def destination_attractiveness_slice(self) -> slice:
        return slice(
            self.production_slice.stop,
            self.production_slice.stop
            + self.specification.destination_attractiveness_basis_columns,
        )

    def raw_parameter_names(
        self,
        production_basis_labels: tuple[str, ...] | None = None,
        destination_attractiveness_basis_labels: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        """Return the canonical raw-coordinate ordering for this layout."""
        names = ["beta_time", "beta_transfer"]
        if self.dispersion_index is not None:
            names.append("dispersion")
        if production_basis_labels is None:
            names.extend(
                f"production_coefficient_{index}"
                for index in range(self.specification.production_basis_columns)
            )
        else:
            if (
                len(production_basis_labels)
                != self.specification.production_basis_columns
            ):
                raise ValueError("production basis labels do not match the layout.")
            if len(set(production_basis_labels)) != len(production_basis_labels):
                raise ValueError("production basis labels must be unique.")
            names.extend(production_basis_labels)
        destination_count = self.specification.destination_attractiveness_basis_columns
        if destination_attractiveness_basis_labels is None:
            destination_labels = tuple(
                f"destination_attractiveness_coefficient_{index}"
                for index in range(destination_count)
            )
        else:
            if (
                len(destination_attractiveness_basis_labels) != destination_count
                or len(set(destination_attractiveness_basis_labels)) != destination_count
            ):
                raise ValueError(
                    "destination-attractiveness basis labels do not match the layout."
                )
            destination_labels = tuple(destination_attractiveness_basis_labels)
        names.extend(destination_labels)
        if len(set(names)) != len(names):
            raise ValueError("raw parameter names must be unique.")
        return tuple(names)


@dataclass(frozen=True, slots=True)
class MinimalGravityParameters:
    beta_time: jax.Array
    beta_transfer: jax.Array
    dispersion: jax.Array | None
    production_coefficients: jax.Array
    destination_attractiveness_coefficients: jax.Array


def transform_minimal_gravity_parameters(
    raw_parameters: object,
    *,
    layout: MinimalGravityParameterLayout,
) -> MinimalGravityParameters:
    raw = jnp.asarray(raw_parameters)
    if raw.ndim != 1 or raw.shape[0] != layout.size:
        raise ValueError(f"raw_parameters must have shape ({layout.size},).")
    floor = jnp.asarray(layout.specification.positivity_floor, dtype=raw.dtype)
    positive = jax.nn.softplus(raw[:2]) + floor
    dispersion = (
        jax.nn.softplus(raw[layout.dispersion_index]) + floor
        if layout.dispersion_index is not None
        else None
    )
    return MinimalGravityParameters(
        beta_time=positive[0],
        beta_transfer=positive[1],
        dispersion=dispersion,
        production_coefficients=raw[layout.production_slice],
        destination_attractiveness_coefficients=raw[
            layout.destination_attractiveness_slice
        ],
    )


def _inverse_softplus(value: float) -> float:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError("positive initial parameters must be finite and positive.")
    return value + math.log(-math.expm1(-value))


def default_minimal_gravity_raw_parameters(
    layout: MinimalGravityParameterLayout,
    *,
    beta_time: float = 1.0,
    beta_transfer: float = 1.0,
    dispersion: float = 50.0,
) -> np.ndarray:
    floor = layout.specification.positivity_floor
    result = np.zeros(layout.size, dtype=np.float64)
    result[0] = _inverse_softplus(beta_time - floor)
    result[1] = _inverse_softplus(beta_transfer - floor)
    if layout.dispersion_index is not None:
        result[layout.dispersion_index] = _inverse_softplus(dispersion - floor)
    return result


def initialize_minimal_gravity_from_entropy(
    *,
    features: ConditionalGravityFeatures,
    entropy_cell_flow: object,
    layout: MinimalGravityParameterLayout,
    production_basis: np.ndarray | None = None,
    dispersion: float = 50.0,
) -> np.ndarray:
    """Project a positive entropy plan onto the minimal gravity utility form."""
    flow = np.asarray(entropy_cell_flow, dtype=np.float64)
    if (
        flow.shape != (features.number_of_cells,)
        or np.any(flow < 0.0)
        or not np.all(np.isfinite(flow))
    ):
        raise ValueError("entropy_cell_flow must align and be finite/non-negative.")
    groups = features.origin_time_group_index
    group_totals = np.bincount(
        groups, weights=flow, minlength=features.number_of_origin_time_groups
    )
    probabilities = np.divide(
        flow,
        group_totals[groups],
        out=np.zeros_like(flow),
        where=group_totals[groups] > 0.0,
    )
    positive = probabilities > 0.0
    if np.count_nonzero(positive) < 2:
        raise ValueError("entropy initialization needs at least two positive cells.")
    group_count = features.number_of_origin_time_groups
    design = np.column_stack(
        (
            -features.journey_time_seconds
            / layout.specification.journey_time_scale_seconds,
            -features.transfer_count,
            np.eye(group_count)[groups],
        )
    )
    target = np.log(probabilities) - np.log(features.destination_attractiveness)
    coefficients = np.linalg.lstsq(design[positive], target[positive], rcond=None)[0]
    beta_time = max(float(coefficients[0]), layout.specification.positivity_floor * 2)
    beta_transfer = max(
        float(coefficients[1]), layout.specification.positivity_floor * 2
    )
    raw = default_minimal_gravity_raw_parameters(
        layout,
        beta_time=beta_time,
        beta_transfer=beta_transfer,
        dispersion=dispersion,
    )
    if layout.specification.production_mode == "estimated_basis":
        if production_basis is None:
            raise ValueError("production_basis is required for estimated productions.")
        basis = np.asarray(production_basis, dtype=np.float64)
        if basis.shape != (
            group_count,
            layout.specification.production_basis_columns,
        ):
            raise ValueError("production_basis has an invalid shape.")
        baseline = features.baseline_productions
        valid = (group_totals > 0.0) & (baseline > 0.0)
        if not np.any(valid):
            raise ValueError("entropy plan cannot initialize production coefficients.")
        raw[layout.production_slice] = np.linalg.lstsq(
            basis[valid], np.log(group_totals[valid] / baseline[valid]), rcond=None
        )[0]
    return raw
