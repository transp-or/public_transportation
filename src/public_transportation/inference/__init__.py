# src/public_transportation/inference/__init__.py
"""
Bayesian inference subpackage.

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

# Re-export likelihood utilities used by model construction
from .likelihood import (  # noqa: F401
    PreparedLikelihoodInputs,
    prepare_likelihood_inputs,
    predict_y,
    predict_mu,
    loglikelihood_from_link_flow,
)