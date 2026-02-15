"""
Exception hierarchy for the assignment subpackage.

This module centralizes errors raised by:
- graph building (time-expanded representation),
- OD grouping,
- cost computation,
- Dial-style dynamic programming,
- high-level assignment orchestration.

We keep a small, readable hierarchy so callers can catch broad categories
(e.g., AssignmentError) while still allowing precise failures.
"""

from __future__ import annotations


class AssignmentError(Exception):
    """Base class for all errors raised by public_transportation.assignment."""


# ---------------------------
# Configuration / validation
# ---------------------------


class AssignmentConfigError(AssignmentError):
    """Raised when AssignmentConfig is invalid or inconsistent."""


class ScenarioCompatibilityError(AssignmentError):
    """
    Raised when the domain Scenario is missing required elements
    or contains incompatible data for assignment.
    """


# ---------------------------
# Graph building
# ---------------------------


class GraphBuildError(AssignmentError):
    """Raised when building the time-expanded graph fails."""


class GraphConsistencyError(GraphBuildError):
    """Raised when the built graph violates expected invariants (e.g., acyclicity)."""


# ---------------------------
# OD grouping
# ---------------------------


class ODGroupingError(AssignmentError):
    """Raised when building OD groups or mapping OD demand fails."""


class ODDemandShapeError(ODGroupingError):
    """Raised when the provided OD demand vector has an unexpected shape or layout."""


# ---------------------------
# Costs
# ---------------------------


class CostComputationError(AssignmentError):
    """Raised when link cost computation fails (e.g., missing time fields)."""


# ---------------------------
# Dial / DP
# ---------------------------


class DialDPError(AssignmentError):
    """Raised when Dial-style dynamic programming fails."""


class NumericalStabilityError(DialDPError):
    """
    Raised when the computation encounters numerical issues
    (NaNs, infs) that cannot be recovered.
    """