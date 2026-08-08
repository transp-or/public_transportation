"""Composable specification for reduced-dimensional demand models."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping

from public_transportation.preprocessing.reduced_od.artifacts import canonical_json

EffectMode = Literal["none", "global", "period", "group", "group_period"]
ImpedanceMode = Literal["none", "global", "period"]
ObservationFamily = Literal["poisson", "negative_binomial", "zip", "zinb"]
ZeroInflationMode = Literal[
    "none",
    "intercept",
    "measurement_type",
    "period",
    "measurement_type_period",
    "design",
]


@dataclass(frozen=True, slots=True)
class ProductionSpecification:
    intercept: bool = False
    period_effects: bool = False
    origin_group_effects: bool = False
    origin_group_period_effects: bool = False


@dataclass(frozen=True, slots=True)
class AttractionSpecification:
    global_term: bool = False
    period_effects: bool = False
    destination_group_effects: bool = False
    destination_group_period_effects: bool = False


@dataclass(frozen=True, slots=True)
class ImpedanceSpecification:
    travel_time: ImpedanceMode = "global"
    transfers: ImpedanceMode = "global"

    def __post_init__(self) -> None:
        if self.travel_time not in {"none", "global", "period"}:
            raise ValueError("unsupported travel-time scope.")
        if self.transfers not in {"none", "global", "period"}:
            raise ValueError("unsupported transfer scope.")


@dataclass(frozen=True, slots=True)
class InteractionSpecification:
    rank: int = 0
    period_specific: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or self.rank < 0:
            raise ValueError("interaction rank must be a nonnegative integer.")
        if self.period_specific:
            raise NotImplementedError(
                "period-specific latent factors require the shared-factor model first."
            )


@dataclass(frozen=True, slots=True)
class ObservationSpecification:
    family: ObservationFamily = "poisson"
    zero_inflation: ZeroInflationMode = "none"

    def __post_init__(self) -> None:
        if self.family not in {"poisson", "negative_binomial", "zip", "zinb"}:
            raise ValueError("unsupported observation family.")
        inflated = self.family in {"zip", "zinb"}
        if inflated != (self.zero_inflation != "none"):
            raise ValueError(
                "zero-inflation predictors must be active exactly for ZIP or ZINB."
            )


@dataclass(frozen=True, slots=True)
class RegularizationSpecification:
    block_ridge: Mapping[str, float] = field(default_factory=dict)
    latent_factor_ridge: float = 1.0

    def __post_init__(self) -> None:
        values = (*self.block_ridge.values(), self.latent_factor_ridge)
        if any(
            value < 0.0 or not float("-inf") < value < float("inf") for value in values
        ):
            raise ValueError("regularization weights must be finite and nonnegative.")


@dataclass(frozen=True, slots=True)
class DemandModelDimensions:
    periods: int
    origin_groups: int
    destination_groups: int
    zero_design_columns: int = 0

    def __post_init__(self) -> None:
        if min(self.periods, self.origin_groups, self.destination_groups) <= 0:
            raise ValueError("period and group dimensions must be positive.")
        if self.zero_design_columns < 0:
            raise ValueError("zero_design_columns must be nonnegative.")


@dataclass(frozen=True, slots=True)
class DemandModelSpecification:
    production: ProductionSpecification = ProductionSpecification()
    attraction: AttractionSpecification = AttractionSpecification()
    impedance: ImpedanceSpecification = ImpedanceSpecification()
    interaction: InteractionSpecification = InteractionSpecification()
    observation: ObservationSpecification = ObservationSpecification()
    regularization: RegularizationSpecification = RegularizationSpecification()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported demand-model specification schema.")
        if self.production.origin_group_period_effects and not (
            self.production.origin_group_effects and self.production.period_effects
        ):
            raise ValueError(
                "origin-group-by-period effects require both main effects."
            )
        if self.attraction.destination_group_period_effects and not (
            self.attraction.destination_group_effects and self.attraction.period_effects
        ):
            raise ValueError(
                "destination-group-by-period effects require both main effects."
            )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def parameter_counts(self, dimensions: DemandModelDimensions) -> dict[str, int]:
        p = dimensions.periods
        og = dimensions.origin_groups
        dg = dimensions.destination_groups
        production = int(self.production.intercept)
        production += (p - 1) if self.production.period_effects else 0
        production += (og - 1) if self.production.origin_group_effects else 0
        production += (
            (og - 1) * (p - 1) if self.production.origin_group_period_effects else 0
        )
        attraction = int(self.attraction.global_term)
        attraction += (p - 1) if self.attraction.period_effects else 0
        attraction += (dg - 1) if self.attraction.destination_group_effects else 0
        attraction += (
            (dg - 1) * (p - 1)
            if self.attraction.destination_group_period_effects
            else 0
        )
        impedance = (
            p
            if self.impedance.travel_time == "period"
            else int(self.impedance.travel_time == "global")
        )
        impedance += (
            p
            if self.impedance.transfers == "period"
            else int(self.impedance.transfers == "global")
        )
        interaction = self.interaction.rank * (og + dg)
        observation = int(self.observation.family in {"negative_binomial", "zinb"})
        if self.observation.zero_inflation == "intercept":
            observation += 1
        elif self.observation.zero_inflation == "measurement_type":
            observation += dimensions.zero_design_columns
        elif self.observation.zero_inflation == "period":
            observation += dimensions.zero_design_columns
        elif self.observation.zero_inflation in {"measurement_type_period", "design"}:
            observation += dimensions.zero_design_columns
        return {
            "production": production,
            "attraction": attraction,
            "impedance": impedance,
            "interaction": interaction,
            "observation": observation,
        }

    def summary(self, dimensions: DemandModelDimensions) -> str:
        counts = self.parameter_counts(dimensions)
        enabled = ", ".join(f"{key}={value}" for key, value in counts.items() if value)
        return f"DemandModelSpecification({enabled or 'no active parameters'}; total={sum(counts.values())})"


def progressive_model_ladder() -> dict[str, DemandModelSpecification]:
    """Return conservative nested reference specifications M0--M6."""
    m0 = DemandModelSpecification(production=ProductionSpecification(intercept=True))
    m1 = DemandModelSpecification(
        production=ProductionSpecification(intercept=True, period_effects=True),
        impedance=ImpedanceSpecification("period", "period"),
    )
    m2 = DemandModelSpecification(
        production=m1.production,
        impedance=m1.impedance,
        observation=ObservationSpecification("negative_binomial", "none"),
    )
    m3 = DemandModelSpecification(
        production=ProductionSpecification(True, True, True),
        impedance=m1.impedance,
        observation=m2.observation,
    )
    m4 = DemandModelSpecification(
        production=m3.production,
        attraction=AttractionSpecification(destination_group_effects=True),
        impedance=m1.impedance,
        observation=m2.observation,
    )
    result = {"M0": m0, "M1": m1, "M2": m2, "M3": m3, "M4": m4}
    for label, rank in (("M5", 1), ("M6-r2", 2), ("M6-r4", 4), ("M6-r8", 8)):
        result[label] = DemandModelSpecification(
            production=m3.production,
            attraction=m4.attraction,
            impedance=m1.impedance,
            interaction=InteractionSpecification(rank),
            observation=m2.observation,
        )
    return result
