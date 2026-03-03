"""public_transportation.inference.types

Lightweight dataclasses and type aliases for the Bayesian inference layer.

This module intentionally contains ONLY types (no computation) and should avoid
importing heavy runtime dependencies.

Design goals
------------
- Keep the inference pipeline explicit: what is fixed data vs. what is random.
- Make it easy to pass data between:
    (i) measurement mapping (MeasurementTable -> AggregationSpec),
    (ii) priors (baseline OD vector f0),
    (iii) assignment (link flows),
    (iv) likelihood (NB on predicted measurements).
- Preserve JAX-compatibility: arrays used inside the traced model must be
  convertible to jax.numpy arrays without Python-side logic.

Notes
-----
- Measurement mapping uses the *new* mapping subpackage:
    public_transportation.measurement.mapping
  (the legacy `measurement.mapper` API is deprecated).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from public_transportation.measurement.mapping import AggregationSpec
from public_transportation.measurement.schema import MeasurementTable


@dataclass(frozen=True, slots=True)
class InferenceData:
    """All fixed inputs required to evaluate the inference model.

    This object is meant to be assembled in Python (outside JAX tracing) and then
    passed into inference builders / pipelines.

    Fields
    ------
    fingerprint:
        Copied from AssignmentIDManager to detect mismatches between:
        - the assignment indexing,
        - the measurement mapping spec, and
        - the baseline OD vector f0.

    scenario:
        Optional provenance object. Not required by the core JAX likelihood; keep
        it only if you need it for reporting.

    assignment_artifacts:
        Precomputed assignment bundle (graph + auxiliary arrays) returned by
        `prepare_assignment(...)`. This must match what the assignment evaluation
        functions expect.

    assignment_config:
        Optional assignment configuration used to generate reports / rerun the
        assignment outside inference.

    f0:
        Baseline OD vector aligned to the assignment OD indexing.

    y_obs:
        Observed measurement vector aligned to the mapping spec ordering.

    mapping_spec:
        Structural aggregation recipe linking assignment link_flow -> y_pred.

    measurements:
        Optional MeasurementTable kept for reporting/validation outside JAX.
        It is NOT used by the traced likelihood.
    """

    # Provenance / consistency
    fingerprint: str
    scenario: Any | None

    # Assignment inputs (opaque to the inference layer)
    assignment_artifacts: Any
    assignment_config: Any | None

    # Baseline OD vector (aligned to assignment OD indexing)
    f0: Any

    # Observed data and mapping to predictions
    y_obs: Any
    mapping_spec: AggregationSpec

    # Optional: keep original measurements for reporting only
    measurements: MeasurementTable | None = None


@dataclass(frozen=True, slots=True)
class ThetaPriorSpec:
    """Numerical specification of a prior for theta (dispersion, in minutes).

    This is intentionally distribution-agnostic. The model-building layer decides
    the final distribution.
    """

    loc: float
    scale: float
    lower: float = 0.0


@dataclass(frozen=True, slots=True)
class ODDeviationPriorSpec:
    """Prior specification for log-deviations z (global sigma).

    The standard v1 parameterization is:
        z_i ~ Normal(0, sigma_z)
        f_i = f0_i * exp(z_i)
    """

    sigma_z: float


@dataclass(frozen=True, slots=True)
class MeasurementModelSpec:
    """Fixed/learned knobs of the measurement model.

    Current v1 measurement likelihood:
      Y_m | mu_m, r ~ NegBinom(mean=mu_m, dispersion=r)
      mu_m = rho * lambda_m

    Where lambda_m is the aggregated (continuous) predicted flow.

    Flags
    -----
    include_rho / include_nb_dispersion:
        Kept for compatibility with earlier code that toggled whether these are
        inferred (True) or treated as fixed (False).

    If a parameter is treated as fixed, its value is taken from `rho` / `nb_dispersion`.
    """

    rho: float = 1.0
    nb_dispersion: float = 50.0

    include_rho: bool = False
    include_nb_dispersion: bool = False


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Configuration knobs for inference model construction.

    This config is separate from AssignmentConfig. It only controls:
    - parametrizations (e.g., log-deviation)
    - prior hyperparameters (sigma_z, theta prior)
    - measurement model toggles/values (rho, NB dispersion)

    It MUST NOT include anything that affects the assignment implementation itself.
    """

    od_prior: ODDeviationPriorSpec
    theta_prior: ThetaPriorSpec
    measurement: MeasurementModelSpec = MeasurementModelSpec()