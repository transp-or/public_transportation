# src/public_transportation/bayesian_estimation/config.py
"""
Configuration objects for the generic Variational Inference (VI) engine.

This module contains lightweight, frozen dataclasses describing how
`run_vi` should behave. It is intentionally engine-facing and model-agnostic.

Design principles
-----------------
- No model logic here.
- No JAX code here.
- Only configuration containers.
- Defaults match those of `run_vi` in core_vi.py.

This allows higher-level modules (e.g. inference.pipeline) to expose a clean,
single configuration object instead of forwarding many keyword arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class VIConfig:
    """
    Configuration for the generic SVI-based variational inference engine.

    This mirrors the keyword arguments of `run_vi(...)` and provides
    a stable interface for higher-level inference pipelines.

    Parameters
    ----------
    guide:
        Autoguide choice:
            - "auto_diag"      : mean-field Gaussian (fast, robust default)
            - "auto_lowrank"   : low-rank + diagonal Gaussian
            - "auto_mvn"       : full-covariance Gaussian (expensive)
            - "auto_normal"    : AutoNormal (kept for completeness)

    lowrank_rank:
        Rank for the low-rank guide (used only if guide="auto_lowrank").

    use_base_normal_correction:
        If True, subtract log N(theta; 0, I) inside the black-box model.
        See documentation of `make_blackbox_model`.

    num_steps:
        Number of SVI optimization steps.

    learning_rate:
        Adam learning rate.

    seed:
        Random seed for initialization and posterior sampling.

    num_posterior_draws:
        Number of posterior samples drawn from the variational guide.

    log_every:
        If a logger is provided to `run_vi`, emit progress every
        `log_every` steps.
    """

    guide: Literal["auto_diag", "auto_lowrank", "auto_mvn", "auto_normal"] = "auto_diag"
    lowrank_rank: int | None = None

    use_base_normal_correction: bool = False

    num_steps: int = 5_000
    learning_rate: float = 1e-2
    seed: int = 0

    num_posterior_draws: int = 1_000
    log_every: int = 100