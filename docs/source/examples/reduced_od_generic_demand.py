"""Inspect a conservative progressive demand-model ladder."""

from public_transportation.inference.reduced_od import (
    DemandModelDimensions,
    build_demand_parameter_layout,
    progressive_model_ladder,
    warm_start_demand_parameters,
)
import numpy as np


dimensions = DemandModelDimensions(
    periods=2,
    origin_groups=3,
    destination_groups=4,
)
ladder = progressive_model_ladder()
for name in ("M0", "M1", "M2", "M3", "M4", "M5"):
    specification = ladder[name]
    layout = build_demand_parameter_layout(specification, dimensions)
    print(name, specification.summary(dimensions), layout.fingerprint)

parent = build_demand_parameter_layout(ladder["M0"], dimensions)
child = build_demand_parameter_layout(ladder["M1"], dimensions)
initial, report = warm_start_demand_parameters(
    parent,
    child,
    np.zeros(parent.size),
)
print(initial.shape, report)
