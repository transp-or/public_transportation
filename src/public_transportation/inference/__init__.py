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
)
