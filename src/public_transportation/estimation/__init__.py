"""
Generic estimation framework.

The package separates the statistical model specification from the estimation
engine:

- estimation.common: shared black-box log-likelihood/log-prior utilities.
- estimation.bayesian: variational Bayesian estimation.
- estimation.maximum_likelihood: ML and penalized ML/MAP estimation.

The same log-likelihood and log-prior functions can be used by both engines.
"""

from .common import Array, LogLikFn, LogPriorFn, base_normal_logpdf
from .bayesian import VIConfig, VIResult, run_vi
from .maximum_likelihood import MLConfig, MLResult, run_ml
