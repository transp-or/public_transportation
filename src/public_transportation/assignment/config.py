"""
Configuration objects for the differentiable public-transport assignment model.

This module contains ONLY user-defined, fixed coefficients that shape the
generalized cost of links and the behavior of the loading algorithm.

This configuration includes:
- generalized-cost coefficients,
- graph construction windows (access and transfer thresholds), and
- dwell-time policy parameters (minimum dwell time).

Centroid placement
------------------
Centroid-in nodes are never time-tagged and are placed conceptually at -∞,
appearing first in the topological order. Centroid-out nodes appear last,
conceptually at +∞. The time-bin index is used only to determine which departure
event nodes are connected by access links and for schedule-deviation costs.

Destination-gated egress
------------------------
The assignment may use destination-gated egress links. This can be implemented either by:
- building one global graph with egress links for all stops and masking out egress links that do not
  lead to the destination of the current OD group, or
- building OD-specific graphs.

The default implementation uses a global graph plus a per-group link mask for efficiency.
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

    max_access_deviation_min : float
        Maximum absolute schedule deviation allowed for access links; links outside this window are not built.
    max_transfer_wait_min : float
        Maximum waiting time allowed for transfer links; links beyond this threshold are not built.

    min_dwell_s : int
        Minimum dwell time at stops in seconds. Used to enforce strictly positive dwell by auto-regularizing
        timetable records where arrival_time == departure_time (departure := arrival + min_dwell_s).
        Must be > 0.

    egress_cost : float
        Additive cost on egress links (event -> centroid-out). Default is 0.

    use_global_egress_links : bool
        If True, keep a global graph with egress links for all stops and apply a per-destination
        mask during assignment. If False, builders may generate destination-gated egress per group.

    mask_strategy : str
        Strategy for destination-gated egress masking. Typical values:
        - "link_mask" (per-group boolean mask over links)
        - "egress_by_head" (mask only egress links by their head node)

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

    Notes
    -----
    Centroid-in nodes are not time-tagged and appear first in the topological order (conceptually −∞).
    Centroid-out nodes appear last (conceptually +∞).
    The time-bin index is used only to determine which departure event nodes are connected by access links
    and for schedule-deviation costs.
    """

    # --- time perception weights (all in minutes of generalized cost) ---
    beta_in_vehicle: float = 1.0
    beta_transfer: float = 1.5
    beta_wait: float = 1.0

    # --- graph construction windows (minutes) ---
    max_access_deviation_min: float = 15.0
    max_transfer_wait_min: float = 30.0

    # --- dwell time policy (seconds) ---
    min_dwell_s: int = 1

    # --- egress handling / destination gating ---
    egress_cost: float = 0.0
    use_global_egress_links: bool = True
    mask_strategy: str = "link_mask"

    # --- schedule deviation penalties ---
    beta_early: float = 2.0
    beta_late: float = 2.0

    # --- logit dispersion ---
    theta_default: float = 5.0
    theta_min: float = 0.1

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
        _check_nonneg("egress_cost", self.egress_cost)
        _check_nonneg("beta_early", self.beta_early)
        _check_nonneg("beta_late", self.beta_late)
        _check_nonneg("max_access_deviation_min", self.max_access_deviation_min)
        _check_nonneg("max_transfer_wait_min", self.max_transfer_wait_min)

        if self.min_dwell_s <= 0:
            raise ValueError(f"min_dwell_s must be strictly positive (seconds), got {self.min_dwell_s}")

        if self.mask_strategy not in {"link_mask", "egress_by_head"}:
            raise ValueError(
                f"mask_strategy must be one of {{'link_mask','egress_by_head'}}, got {self.mask_strategy!r}"
            )

        if self.theta_default <= 0:
            raise ValueError("theta_default must be strictly positive")

        if self.use_capacity_penalty:
            _check_nonneg("capacity_penalty_alpha", self.capacity_penalty_alpha)
            if self.capacity_penalty_kappa <= 0:
                raise ValueError("capacity_penalty_kappa must be > 0")