"""
Generic estimation framework.

The package separates the statistical model specification from the estimation
engine:

- estimation.common: shared black-box log-likelihood/log-prior utilities.
- estimation.bayesian: variational Bayesian estimation.
- estimation.maximum_likelihood: ML and penalized ML/MAP estimation.

The same log-likelihood and log-prior functions can be used by both engines.
"""

from .common import (
    Array as Array,
    LogLikFn as LogLikFn,
    LogPriorFn as LogPriorFn,
    base_normal_logpdf as base_normal_logpdf,
)
from .bayesian import VIConfig as VIConfig, VIResult as VIResult, run_vi as run_vi
from .maximum_likelihood import (
    MLConfig as MLConfig,
    MLResult as MLResult,
    run_ml as run_ml,
)
