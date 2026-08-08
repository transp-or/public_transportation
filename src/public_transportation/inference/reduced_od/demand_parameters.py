"""Named parameter blocks and warm starts for composable demand models."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from public_transportation.preprocessing.reduced_od.artifacts import canonical_json

from .demand_specification import DemandModelDimensions, DemandModelSpecification


@dataclass(frozen=True, slots=True)
class DemandParameterBlock:
    name: str
    start: int
    stop: int
    shape: tuple[int, ...]
    transform: str = "identity"
    identification: str = "none"
    ridge: float = 0.0

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class DemandParameterLayout:
    specification_fingerprint: str
    dimensions: DemandModelDimensions
    blocks: tuple[DemandParameterBlock, ...]
    schema_version: int = 1

    @property
    def size(self) -> int:
        return self.blocks[-1].stop if self.blocks else 0

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "specification": self.specification_fingerprint,
            "dimensions": asdict(self.dimensions),
            "blocks": [asdict(item) for item in self.blocks],
        }
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()

    @property
    def slices(self) -> Mapping[str, slice]:
        return {item.name: slice(item.start, item.stop) for item in self.blocks}

    @property
    def raw_parameter_names(self) -> tuple[str, ...]:
        """Return stable, unique names for every flattened optimizer coordinate."""
        names: list[str] = []
        for block in self.blocks:
            if block.size == 1:
                names.append(block.name)
                continue
            for flat_index in range(block.size):
                index = np.unravel_index(flat_index, block.shape)
                suffix = ",".join(str(value) for value in index)
                names.append(f"{block.name}[{suffix}]")
        return tuple(names)

    def block(self, name: str) -> DemandParameterBlock:
        matches = [item for item in self.blocks if item.name == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def reconstruct(self, raw_parameters: object) -> dict[str, np.ndarray]:
        raw = np.asarray(raw_parameters)
        if raw.shape != (self.size,):
            raise ValueError(f"raw parameters must have shape ({self.size},).")
        return {
            item.name: raw[item.start : item.stop].reshape(item.shape)
            for item in self.blocks
        }


def build_demand_parameter_layout(
    specification: DemandModelSpecification, dimensions: DemandModelDimensions
) -> DemandParameterLayout:
    if (
        specification.interaction.rank
        and min(dimensions.origin_groups, dimensions.destination_groups) < 2
    ):
        raise ValueError(
            "low-rank interaction requires at least two groups on both margins."
        )
    blocks: list[DemandParameterBlock] = []
    start = 0

    def add(
        name: str,
        shape: tuple[int, ...],
        *,
        transform: str = "identity",
        identification: str = "none",
    ) -> None:
        nonlocal start
        size = int(np.prod(shape, dtype=int))
        if size == 0:
            return
        ridge = float(
            specification.regularization.block_ridge.get(
                name,
                specification.regularization.latent_factor_ridge
                if name.startswith("interaction_")
                else 0.0,
            )
        )
        blocks.append(
            DemandParameterBlock(
                name, start, start + size, shape, transform, identification, ridge
            )
        )
        start += size

    p, og, dg = (
        dimensions.periods,
        dimensions.origin_groups,
        dimensions.destination_groups,
    )
    if specification.production.intercept:
        add("production_intercept", (1,))
    if specification.production.period_effects:
        add("production_period", (p - 1,), identification="sum_to_zero")
    if specification.production.origin_group_effects:
        add("production_origin_group", (og - 1,), identification="sum_to_zero")
    if specification.production.origin_group_period_effects:
        add(
            "production_origin_period",
            (og - 1, p - 1),
            identification="double_sum_to_zero",
        )
    if specification.attraction.global_term:
        add("attraction_global", (1,))
    if specification.attraction.period_effects:
        add("attraction_period", (p - 1,), identification="sum_to_zero")
    if specification.attraction.destination_group_effects:
        add("attraction_destination_group", (dg - 1,), identification="sum_to_zero")
    if specification.attraction.destination_group_period_effects:
        add(
            "attraction_destination_period",
            (dg - 1, p - 1),
            identification="double_sum_to_zero",
        )
    if specification.impedance.travel_time != "none":
        add(
            "impedance_time",
            (p if specification.impedance.travel_time == "period" else 1,),
            transform="softplus",
        )
    if specification.impedance.transfers != "none":
        add(
            "impedance_transfer",
            (p if specification.impedance.transfers == "period" else 1,),
            transform="softplus",
        )
    rank = specification.interaction.rank
    if rank:
        add("interaction_origin", (og, rank), identification="column_centered")
        add("interaction_destination", (dg, rank), identification="column_centered")
    if specification.observation.family in {"negative_binomial", "zinb"}:
        add("observation_dispersion", (1,), transform="softplus")
    if specification.observation.zero_inflation == "intercept":
        add("zero_inflation", (1,))
    elif specification.observation.zero_inflation != "none":
        add("zero_inflation", (dimensions.zero_design_columns,))
    return DemandParameterLayout(specification.fingerprint, dimensions, tuple(blocks))


@dataclass(frozen=True, slots=True)
class WarmStartReport:
    copied: tuple[str, ...]
    expanded: tuple[str, ...]
    initialized: tuple[str, ...]
    dropped: tuple[str, ...]
    transformations: tuple[str, ...]


def warm_start_demand_parameters(
    parent: DemandParameterLayout,
    child: DemandParameterLayout,
    parent_raw: object,
    *,
    latent_scale: float = 1.0e-3,
    zero_inflation_logit: float = -8.0,
    dispersion_raw: float = 4.0,
) -> tuple[np.ndarray, WarmStartReport]:
    source = np.asarray(parent_raw, dtype=np.float64)
    if source.shape != (parent.size,):
        raise ValueError(f"parent_raw must have shape ({parent.size},).")
    result = np.zeros(child.size, dtype=np.float64)
    copied: list[str] = []
    expanded: list[str] = []
    initialized: list[str] = []
    parent_by_name = {item.name: item for item in parent.blocks}
    child_names = {item.name for item in child.blocks}
    for block in child.blocks:
        target = result[block.start : block.stop]
        old = parent_by_name.get(block.name)
        if old is not None:
            values = source[old.start : old.stop]
            width = min(values.size, target.size)
            target[:width] = values[:width]
            (copied if values.size == target.size else expanded).append(block.name)
            if block.name.startswith("interaction_") and target.size > width:
                target[width:] = latent_scale * np.sin(
                    np.arange(target.size - width) + 1.0
                )
        else:
            initialized.append(block.name)
            if block.name == "zero_inflation":
                target[:] = zero_inflation_logit
            elif block.name == "observation_dispersion":
                target[:] = dispersion_raw
            elif block.name.startswith("interaction_"):
                target[:] = latent_scale * np.sin(np.arange(target.size) + 1.0)
    dropped = sorted(set(parent_by_name) - child_names)
    return result, WarmStartReport(
        tuple(copied), tuple(expanded), tuple(initialized), tuple(dropped), ()
    )
