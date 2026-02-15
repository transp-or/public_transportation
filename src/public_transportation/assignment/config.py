"""
Configuration objects for the differentiable public-transport assignment model.

This module contains ONLY user-defined, fixed coefficients that shape the
generalized cost of links and the behavior of the loading algorithm.

Important design choice
-----------------------
We **do not estimate behavioral parameters** here.
The only parameters estimated later in the pipeline are:

- the OD demand vector
- optionally the dispersion parameter `theta` of the logit assignment

All coefficients defined in this file are fixed by the user and treated as
known constants during inference.

All costs are expressed in **minutes of generalized cost**.
Coefficients below therefore act as multipliers converting physical time into
perceived generalized time.

Examples:
- 1 minute in-vehicle time = 1 unit of cost
- 1 minute walking = 1.5 units of cost
- 1 minute transfer = 2 units of cost
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AssignmentConfig:
    """
    Fixed configuration of the assignment model.

    This class stores coefficients used to compute generalized costs.
    None of these parameters are estimated.

    Parameters
    ----------
    beta_in_vehicle : float
        Weight applied to in-vehicle time (usually 1.0).
    beta_transfer : float
        Weight applied to transfer time between vehicles.
    beta_wait : float
        Weight applied to waiting time.
    beta_access : float
        Weight applied to access/egress time (if modeled explicitly).

    beta_early : float
        Penalty per minute for departing *before* the desired time window.
    beta_late : float
        Penalty per minute for departing *after* the desired time window.

    theta_default : float
        Default dispersion parameter used if not provided by the caller.
        This parameter may optionally be estimated later.

    use_capacity_penalty : bool
        Whether to include capacity penalties in link cost.
        First implementation may keep this False.

    capacity_penalty_alpha : float
        Weight of capacity overload penalty.
    capacity_penalty_kappa : float
        Softplus smoothing parameter for overload.
    """

    # --- time perception weights (all in minutes of generalized cost) ---
    beta_in_vehicle: float = 1.0
    beta_transfer: float = 1.5
    beta_wait: float = 1.0
    beta_access: float = 1.0

    # --- schedule deviation penalties ---
    beta_early: float = 2.0
    beta_late: float = 2.0

    # --- logit dispersion ---
    theta_default: float = 5.0

    # --- capacity penalty (optional) ---
    use_capacity_penalty: bool = False
    capacity_penalty_alpha: float = 0.0
    capacity_penalty_kappa: float = 1.0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """
        Validate configuration values.

        Raises
        ------
        ValueError if any coefficient is invalid.
        """

        def _check_nonneg(name: str, value: float) -> None:
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        _check_nonneg("beta_in_vehicle", self.beta_in_vehicle)
        _check_nonneg("beta_transfer", self.beta_transfer)
        _check_nonneg("beta_wait", self.beta_wait)
        _check_nonneg("beta_access", self.beta_access)
        _check_nonneg("beta_early", self.beta_early)
        _check_nonneg("beta_late", self.beta_late)

        if self.theta_default <= 0:
            raise ValueError("theta_default must be strictly positive")

        if self.use_capacity_penalty:
            _check_nonneg("capacity_penalty_alpha", self.capacity_penalty_alpha)
            if self.capacity_penalty_kappa <= 0:
                raise ValueError("capacity_penalty_kappa must be > 0")