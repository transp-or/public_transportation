"""Compatibility bridge from the established gravity example inputs."""

from __future__ import annotations

import numpy as np

from public_transportation.inference.gravity import GravityFeatures
from public_transportation.inference.measurement_operator_protocol import GravityMeasurementOperator
from public_transportation.preprocessing.reduced_od import ResponseCellKey

from .demand_model import DemandModelProblem
from .demand_parameters import build_demand_parameter_layout
from .demand_specification import DemandModelDimensions, DemandModelSpecification
from .features import ConditionalGravityFeatures
from .response_operator import GravityResponseOperatorAdapter


def build_generic_demand_problem_from_gravity(
    *,
    features: GravityFeatures,
    operator: GravityMeasurementOperator,
    observations: object,
    specification: DemandModelSpecification,
) -> DemandModelProblem:
    """Reuse a legacy gravity feature table and device operator without densifying."""
    periods = (
        features.time_period_index
        if features.time_period_index is not None
        else features.departure_time_index
    )
    triples = list(
        zip(
            features.origin_index.tolist(),
            features.destination_index.tolist(),
            features.departure_time_index.tolist(),
            strict=True,
        )
    )
    order = np.asarray(sorted(range(len(triples)), key=triples.__getitem__), dtype=np.int64)
    triples = [triples[index] for index in order]
    ordered_origins = np.asarray(features.origin_index)[order]
    ordered_periods = np.asarray(periods)[order]
    group_pairs = sorted(
        set(zip(features.origin_index.tolist(), periods.tolist(), strict=True))
    )
    group_lookup = {pair: index for index, pair in enumerate(group_pairs)}
    group_period = np.asarray([period for _, period in group_pairs], dtype=np.int64)
    origin_groups = np.asarray([origin for origin, _ in group_pairs], dtype=np.int64)
    destination_groups = np.asarray(features.destination_index, dtype=np.int64)
    converted = ConditionalGravityFeatures(
        cell_keys=tuple(
            ResponseCellKey(f"O{origin:08d}", f"D{destination:08d}", f"P{period:08d}")
            for origin, destination, period in triples
        ),
        origin_time_group_index=np.asarray(
            [
                group_lookup[(origin, period)]
                for origin, period in zip(
                    ordered_origins, ordered_periods, strict=True
                )
            ]
        ),
        destination_index=np.asarray(features.destination_index)[order],
        journey_time_seconds=(
            np.asarray(features.journey_time)[order]
            * 1800.0
            / features.journey_time_scale
        ),
        transfer_count=np.asarray(features.transfer_count, dtype=np.float64)[order],
        destination_attractiveness=np.asarray(features.destination_attractiveness)[
            order
        ],
        baseline_productions=np.asarray(
            [features.origin_time_totals[index] for index in range(len(group_pairs))]
        ),
        origin_time_group_keys=tuple((f"O{o:08d}", f"P{p:08d}") for o, p in group_pairs),
        destination_ids=tuple(f"D{index:08d}" for index in range(features.num_destinations)),
    )
    dimensions = DemandModelDimensions(
        periods=int(group_period.max()) + 1,
        origin_groups=features.num_origins,
        destination_groups=features.num_destinations,
    )
    layout = build_demand_parameter_layout(specification, dimensions)
    return DemandModelProblem(
        features=converted,
        response_operator=GravityResponseOperatorAdapter(operator, order),
        observations=np.asarray(observations),
        specification=specification,
        parameter_layout=layout,
        group_period_index=group_period,
        origin_group_index=origin_groups,
        cell_destination_group_index=destination_groups[order],
    )
