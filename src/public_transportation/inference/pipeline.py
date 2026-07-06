"""
public_transportation.inference.pipeline

End-to-end Variational Inference (VI) pipeline for OD + theta.

-This module hides inference wiring from scripts:
- parameterization: z -> f = f0 * exp(z), optionally u -> theta = exp(u)
- forward model: assignment link flows + measurement aggregation
- measurement likelihood: Negative Binomial on observed counts
- VI engine: NumPyro SVI via `public_transportation.bayesian_estimation.run_vi`

Notes
-----
- This file performs computation and therefore must NOT pretend to be types-only.
- Mapping construction / IO are out of scope: callers provide y_obs and AggregationSpec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

import numpyro.distributions as dist

from public_transportation.estimation.bayesian import VIConfig, run_vi
from public_transportation.estimation.bayesian.results import VIResult
from public_transportation.measurement.mapping import AggregationSpec
from public_transportation.inference.likelihood import (
    prepare_likelihood_inputs,
    loglikelihood_from_link_flow,
)

from public_transportation.inference.assignment_adapter import build_assignment_inputs
from public_transportation.inference.model import make_forward_inputs, forward_model



@dataclass(frozen=True, slots=True)
class ODThetaEstimationRequest:
    """User-facing inputs for OD+theta VI.

    This dataclass is intentionally compact: the main script should provide data,
    mapping, and configuration; all technical wiring stays inside inference.
    """

    # Provenance / consistency
    fingerprint: str

    # Baseline OD vector (assignment indexing), shape (num_od,)
    f0: jnp.ndarray

    # Observations + structural mapping
    y_obs: jnp.ndarray
    mapping_spec: AggregationSpec

    # Baseline theta used to center the prior on log(theta)
    baseline_theta: float

    # Theta handling
    # - estimate_theta=True: estimate theta through u = log(theta).
    # - estimate_theta=False: estimate only z and use fixed_theta.
    estimate_theta: bool = True
    fixed_theta: float | None = None

    # Inference hyperparameters
    sigma_z: float = 1.0
    sigma_u: float = 0.7
    mu_u_strategy: Literal["center_at_baseline", "fixed"] = "center_at_baseline"
    mu_u_fixed: float | None = None

    # Measurement model constants
    rho: float = 1.0
    nb_dispersion: float = 50.0

    # Numerical safety
    z_clip: float = 6.0
    u_clip: float = 6.0

    # VI engine config
    vi: VIConfig = VIConfig()

    # Assignment evaluation (opaque artifacts from prepare_assignment)
    assignment_artifacts: Any = None

    # Optional logger
    logger: Any | None = None

    # Optional: exact JSON payload used to compute the fingerprint (for richer diagnostics)
    fingerprint_payload_json: str | None = None


@dataclass(frozen=True, slots=True)
class ODThetaInferenceResult:
    """High-level outputs (ready to save/report)."""
    vi: VIResult                     # raw engine output
    num_od: int
    f0: np.ndarray                   # baseline in assignment OD order
    theta_samples: np.ndarray        # shape (S,)
    f_samples: np.ndarray            # shape (S, num_od)

    # optional convenience summaries
    theta_mean: float
    theta_sd: float
    f_mean: np.ndarray               # (num_od,)

    # Theta handling used for this run
    estimate_theta: bool
    fixed_theta: float | None

    # provenance / consistency checks
    fingerprint: str                 # copied from id_manager

    # Optional: exact JSON payload used to compute the fingerprint (for richer diagnostics)
    fingerprint_payload_json: str | None = None



def estimate_od_theta_vi(request: ODThetaEstimationRequest) -> ODThetaInferenceResult:
    """Run OD+theta VI end-to-end.

    The main script should only:
    - prepare scenario + assignment artifacts,
    - prepare measurements + mapping spec,
    - choose configuration,
    - call this function,
    - save outputs and generate reports.

    This function hides:
    - parameterization (z and, optionally, u) and transforms to (f, theta),
    - measurement likelihood wiring,
    - VI engine call,
    - post-processing of posterior samples.
    """
    if request.assignment_artifacts is None:
        raise ValueError("request.assignment_artifacts must be provided (output of prepare_assignment).")

    f0 = jnp.asarray(request.f0)
    if f0.ndim != 1:
        raise ValueError(f"f0 must be 1D, got shape {f0.shape}")

    y_obs = jnp.asarray(request.y_obs)
    if y_obs.ndim != 1:
        raise ValueError(f"y_obs must be 1D, got shape {y_obs.shape}")

    spec = request.mapping_spec
    m = int(spec.num_measurements)
    if int(y_obs.shape[0]) != m:
        raise ValueError(f"y_obs length {int(y_obs.shape[0])} does not match spec.num_measurements {m}")

    estimate_theta = bool(request.estimate_theta)
    fixed_theta: float | None = None
    if estimate_theta:
        if request.fixed_theta is not None:
            raise ValueError("fixed_theta must be None when estimate_theta=True.")
    else:
        if request.fixed_theta is None:
            raise ValueError("fixed_theta must be provided when estimate_theta=False.")
        fixed_theta = float(request.fixed_theta)
        if not np.isfinite(fixed_theta) or fixed_theta <= 0.0:
            raise ValueError(
                f"fixed_theta must be positive and finite when estimate_theta=False, got {fixed_theta!r}"
            )


    # ---- Build assignment inputs (adapter) and forward-model inputs
    assignment_inputs = build_assignment_inputs(artifacts=request.assignment_artifacts)
    forward_inputs = make_forward_inputs(f0=f0, spec=spec)
    prepared = prepare_likelihood_inputs(y_obs=y_obs, spec=spec)

    num_od = int(f0.shape[0])
    dim = num_od + 1 if estimate_theta else num_od

    # ---- Prior centering for u = log(theta)
    if request.mu_u_strategy == "center_at_baseline":
        bt = float(request.baseline_theta)
        if not np.isfinite(bt) or bt <= 0.0:
            raise ValueError(f"baseline_theta must be positive and finite, got {bt!r}")
        mu_u = float(np.log(bt))
    elif request.mu_u_strategy == "fixed":
        if request.mu_u_fixed is None:
            raise ValueError("mu_u_fixed must be provided when mu_u_strategy='fixed'.")
        mu_u = float(request.mu_u_fixed)
    else:
        raise ValueError(f"Unknown mu_u_strategy: {request.mu_u_strategy!r}")

    sigma_z = float(request.sigma_z)
    sigma_u = float(request.sigma_u)
    if not np.isfinite(sigma_z) or sigma_z <= 0.0:
        raise ValueError(f"sigma_z must be positive and finite, got {sigma_z!r}")
    if not np.isfinite(sigma_u) or sigma_u <= 0.0:
        raise ValueError(f"sigma_u must be positive and finite, got {sigma_u!r}")

    rho = jnp.asarray(float(request.rho)).reshape(())
    r_nb = jnp.asarray(float(request.nb_dispersion)).reshape(())

    z_clip = float(request.z_clip)
    u_clip = float(request.u_clip)

    # ---- Data PyTree for the black-box VI engine
    # Keep everything JAX-friendly.
    data = {
        "rho": rho,
        "r_nb": r_nb,
    }

    def _split_theta(theta_vec: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray]:
        """Split the engine parameter vector and return z, u, theta.

        If theta is estimated, the vector is [z, u] and theta = exp(u).
        If theta is fixed, the vector is [z] and theta = fixed_theta.
        """
        z = theta_vec[:num_od]
        if estimate_theta:
            u = theta_vec[num_od]
            u_clipped = jnp.clip(u, -u_clip, u_clip)
            theta = jnp.exp(u_clipped)
            return z, u_clipped, theta

        assert fixed_theta is not None
        theta = jnp.asarray(fixed_theta, dtype=theta_vec.dtype).reshape(())
        return z, None, theta

    def loglik(theta_vec: jnp.ndarray, data_: Any) -> jnp.ndarray:
        # Parameterization: f = f0 * exp(z), theta = exp(u) or fixed theta
        z, _, theta = _split_theta(theta_vec)
        z = jnp.clip(z, -z_clip, z_clip)

        # Forward model: assignment + aggregation + detection
        out = forward_model(
            inputs=forward_inputs,
            z=z,
            theta=theta,
            rho=data_["rho"],
            assignment_inputs=assignment_inputs,
        )

        # Measurement NB log-likelihood on link_flow (aggregation spec is passed here)
        ll = loglikelihood_from_link_flow(
            link_flow=out.link_flow,
            prepared=prepared,
            theta=theta,
            rho=data_["rho"],
            r=data_["r_nb"],
        )

        # Keep failures as -inf (helps VI)
        return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)

    def logprior(theta_vec: jnp.ndarray) -> jnp.ndarray:
        z, u, _ = _split_theta(theta_vec)
        z = jnp.clip(z, -z_clip, z_clip)

        # Absolute log prior. If theta is fixed, there is no u parameter and no
        # theta prior contribution to the estimated parameter vector.
        lp_abs = dist.Normal(0.0, sigma_z).log_prob(z).sum()
        if estimate_theta:
            assert u is not None
            lp_abs = lp_abs + dist.Normal(mu_u, sigma_u).log_prob(u).sum()

        # The VI engine uses a base Normal(0,1) site. If correction is OFF (default),
        # we should provide the increment: log p - log N(0,1).
        if request.vi.use_base_normal_correction:
            return lp_abs

        lp_base = dist.Normal(0.0, 1.0).log_prob(z).sum()
        if estimate_theta:
            assert u is not None
            lp_base = lp_base + dist.Normal(0.0, 1.0).log_prob(u).sum()
        return lp_abs - lp_base

    # ---- Run VI (NumPyro SVI)
    vi_res = run_vi(
        dim=dim,
        data=data,
        loglik=loglik,
        logprior=logprior,
        guide=request.vi.guide,
        lowrank_rank=request.vi.lowrank_rank,
        use_base_normal_correction=request.vi.use_base_normal_correction,
        num_steps=request.vi.num_steps,
        learning_rate=request.vi.learning_rate,
        seed=request.vi.seed,
        num_posterior_draws=request.vi.num_posterior_draws,
        logger=request.logger,
        log_every=request.vi.log_every,
    )

    # ---- Post-process samples
    samples = np.asarray(vi_res.posterior_samples_theta)
    if samples.ndim != 2 or int(samples.shape[1]) != dim:
        raise RuntimeError(f"Unexpected posterior sample shape: {samples.shape}, expected (S, {dim})")

    z_s = samples[:, :num_od]

    if estimate_theta:
        u_s = samples[:, num_od]
        theta_samples = np.exp(u_s)
    else:
        assert fixed_theta is not None
        theta_samples = np.full(samples.shape[0], fixed_theta, dtype=float)
    f0_np = np.asarray(f0)
    f_samples = f0_np[None, :] * np.exp(z_s)

    theta_mean = float(theta_samples.mean())
    theta_sd = float(theta_samples.std(ddof=0))
    f_mean = f_samples.mean(axis=0)

    return ODThetaInferenceResult(
        vi=vi_res,
        num_od=num_od,
        f0=np.asarray(f0_np, dtype=float),
        theta_samples=np.asarray(theta_samples, dtype=float),
        f_samples=np.asarray(f_samples, dtype=float),
        theta_mean=theta_mean,
        theta_sd=theta_sd,
        f_mean=np.asarray(f_mean, dtype=float),
        estimate_theta=estimate_theta,
        fixed_theta=fixed_theta,
        fingerprint=str(request.fingerprint),
        fingerprint_payload_json=(None if request.fingerprint_payload_json is None else str(request.fingerprint_payload_json)),
    )
