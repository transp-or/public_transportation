# src/public_transportation/inference/__init__.py
"""
Shared statistical-inference subpackage.

Public API:
- Prior construction (see `priors.py`)
- Model wiring (see `model.py`)
- Likelihood wiring (see `likelihood.py`), which bridges:
    mapper outputs (AggregationSpec, y_obs) -> JAX log-likelihood
"""

from __future__ import annotations

# Re-export core lightweight types/helpers
from .types import *  # noqa: F401,F403

# Re-export priors / model entry points (keep these stable)
from .priors import *  # noqa: F401,F403
from .model import *  # noqa: F401,F403
from .od_parameter_layout import (  # noqa: F401
    ODParameterLayout,
    assert_od_layout_fingerprint_matches,
    build_od_parameter_layout,
)
from .compact_od_assignment_layout import (  # noqa: F401
    CompactODAssignmentLayout,
    build_compact_od_assignment_layout,
)
from .compact_od_groups import compact_od_groups  # noqa: F401
from .fixed_routing_measurement_operator import (  # noqa: F401
    FixedRoutingMeasurementOperator,
    MeasurementOperatorMetrics,
    choose_fixed_measurement_operator,
    fixed_routing_measurement_operator_cache_path,
    load_fixed_routing_measurement_operator,
    load_or_prepare_fixed_routing_measurement_operator,
    predict_measurements_fixed_operator,
    prepare_fixed_routing_measurement_operator,
    save_fixed_routing_measurement_operator,
    validate_fixed_routing_measurement_operator,
)
from .maximum_likelihood_pipeline import (  # noqa: F401
    ODThetaMLProblem,
    build_od_theta_ml_problem,
)
from .complexity import ODParameterComplexity, build_od_parameter_complexity  # noqa: F401
from .runtime_profile import (  # noqa: F401
    ODAssignmentRuntimeProfile,
    build_od_assignment_runtime_profile,
)

# Re-export likelihood utilities used by model construction
from .likelihood import (  # noqa: F401
    PreparedLikelihoodInputs,
    prepare_likelihood_inputs,
    predict_y,
    predict_mu,
    loglikelihood_from_link_flow,
    loglikelihood_from_measurement_mean,
)
from .fixed_routing_group import (  # noqa: F401
    SingleGroupAssignment,
    assemble_single_group_demand,
    build_single_free_group_assignment,
)
from .matrix_free_streaming import (  # noqa: F401
    StreamedDestinationGroup,
    StreamedValueAndGradient,
    replayable_streamed_measurement_value_and_grad,
    streamed_measurement_value_and_grad,
)
