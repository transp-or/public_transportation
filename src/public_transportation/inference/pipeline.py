"""
public_transportation.inference.pipeline

End-to-end Variational Inference (VI) pipeline for OD + theta.

-This module hides inference wiring from scripts:
- parameterization: free raw z/u -> smooth bounded values -> full f/theta
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

from public_transportation.inference.assignment_adapter import (
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.model import (
    make_forward_inputs,
    forward_model_from_demand,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout
from public_transportation.inference.parameterization import (
    raw_value_for_effective_center,
    smooth_bound as _smooth_bound,
    smooth_bound_numpy as _smooth_bound_numpy,
)
from public_transportation.inference.runtime_profile import (
    ODAssignmentRuntimeProfile,
    build_od_assignment_runtime_profile,
)


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

    # Optional reduced layout. When provided, only its free cells are latent;
    # frozen cells enter assignment solely as fixed constants.
    od_layout: ODParameterLayout | None = None

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

    # Smooth bounds used for numerical safety. The historical ``*_clip`` names
    # are retained for API compatibility; values are no longer hard-clipped.
    z_clip: float = 6.0
    u_clip: float = 6.0

    # VI engine config
    vi: VIConfig = VIConfig()

    # Assignment evaluation (opaque artifacts from prepare_assignment)
    assignment_artifacts: Any = None

    # Optional fixed-theta ML/MAP acceleration. ``off`` preserves the reference
    # link-flow loader; ``dense`` and ``bcoo`` precompute a direct free-OD to
    # measurement operator. This option is ignored by VI and estimated-theta ML.
    fixed_measurement_operator: Literal["off", "auto", "dense", "bcoo"] = "off"
    fixed_measurement_operator_chunk_size: int = 128
    fixed_measurement_operator_cache_directory: str | None = None
    fixed_measurement_operator_expected_evaluations: int = 0
    fixed_measurement_operator_construction_seconds: float | None = None
    fixed_measurement_operator_reference_seconds: float = 1.94
    fixed_measurement_operator_evaluation_seconds: float = 0.0

    # Optional logger
    logger: Any | None = None

    # Optional: exact JSON payload used to compute the fingerprint (for richer diagnostics)
    fingerprint_payload_json: str | None = None


@dataclass(frozen=True, slots=True)
class ODThetaInferenceResult:
    """High-level outputs (ready to save/report)."""

    vi: VIResult  # raw engine output
    num_od: int
    num_free_od: int
    num_fixed_od: int
    runtime_profile: ODAssignmentRuntimeProfile
    f0: np.ndarray  # baseline in assignment OD order
    theta_samples: np.ndarray  # shape (S,)
    f_samples: np.ndarray  # shape (S, num_od)

    # optional convenience summaries
    theta_mean: float
    theta_sd: float
    f_mean: np.ndarray  # (num_od,)

    # Theta handling used for this run
    estimate_theta: bool
    fixed_theta: float | None

    # provenance / consistency checks
    fingerprint: str  # copied from id_manager
    od_layout_fingerprint: str | None
    od_layout_payload_json: str | None
    compact_layout_fingerprint: str | None
    compact_layout_payload_json: str | None

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
    - compact assignment demand and full-vector result reconstruction,
    - measurement likelihood wiring,
    - VI engine call,
    - post-processing of posterior samples.
    """
    if request.assignment_artifacts is None:
        raise ValueError(
            "request.assignment_artifacts must be provided (output of prepare_assignment)."
        )

    f0 = jnp.asarray(request.f0)
    if f0.ndim != 1:
        raise ValueError(f"f0 must be 1D, got shape {f0.shape}")

    y_obs = jnp.asarray(request.y_obs)
    if y_obs.ndim != 1:
        raise ValueError(f"y_obs must be 1D, got shape {y_obs.shape}")

    spec = request.mapping_spec
    m = int(spec.num_measurements)
    if int(y_obs.shape[0]) != m:
        raise ValueError(
            f"y_obs length {int(y_obs.shape[0])} does not match spec.num_measurements {m}"
        )

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

    num_od = int(f0.shape[0])
    layout = request.od_layout
    if layout is None:
        free_indices = tuple(range(num_od))
        fixed_indices: tuple[int, ...] = ()
        free_f0 = f0
        num_free = num_od
    else:
        if layout.num_od_total != num_od:
            raise ValueError(
                f"od_layout.num_od_total {layout.num_od_total} does not match f0 length {num_od}."
            )
        free_indices = layout.free_od_indices
        fixed_indices = layout.fixed_od_indices
        free_f0 = jnp.asarray(layout.free_baseline_values, dtype=f0.dtype)
        if not np.allclose(
            np.asarray(f0)[np.asarray(free_indices, dtype=int)],
            np.asarray(free_f0),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "od_layout free baselines must exactly match f0 at free OD indices."
            )
        num_free = layout.num_free

    compact_layout = (
        None
        if layout is None
        else build_compact_od_assignment_layout(parameter_layout=layout)
    )

    def assemble_assignment_demand(z_free: jnp.ndarray) -> jnp.ndarray:
        if layout is not None:
            assert compact_layout is not None
            return compact_layout.assemble_compact_jax(z_free)
        return free_f0 * jnp.exp(z_free)

    # ---- Build assignment inputs (adapter) and forward-model inputs
    assignment_inputs = build_assignment_inputs(
        artifacts=request.assignment_artifacts,
        compact_layout=compact_layout,
    )
    fixed_routing = (
        None
        if estimate_theta
        else prepare_fixed_routing(inputs=assignment_inputs, theta=fixed_theta)
    )
    runtime_profile = build_od_assignment_runtime_profile(
        num_od_total=num_od,
        parameter_layout=layout,
        compact_layout=compact_layout,
        artifacts=request.assignment_artifacts,
        assignment_inputs=assignment_inputs,
    )
    forward_inputs = make_forward_inputs(f0=f0, spec=spec)
    prepared = prepare_likelihood_inputs(y_obs=y_obs, spec=spec)

    dim = num_free + 1 if estimate_theta else num_free

    # ---- Requested prior center for the effective u = log(theta)
    if request.mu_u_strategy == "center_at_baseline":
        bt = float(request.baseline_theta)
        if not np.isfinite(bt) or bt <= 0.0:
            raise ValueError(f"baseline_theta must be positive and finite, got {bt!r}")
        effective_mu_u = float(np.log(bt))
    elif request.mu_u_strategy == "fixed":
        if request.mu_u_fixed is None:
            raise ValueError("mu_u_fixed must be provided when mu_u_strategy='fixed'.")
        effective_mu_u = float(request.mu_u_fixed)
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

    z_bound = float(request.z_clip)
    u_bound = float(request.u_clip)
    if not np.isfinite(z_bound) or z_bound <= 0.0:
        raise ValueError(f"z_clip must be positive and finite, got {z_bound!r}")
    if not np.isfinite(u_bound) or u_bound <= 0.0:
        raise ValueError(f"u_clip must be positive and finite, got {u_bound!r}")
    if not np.isfinite(effective_mu_u) or abs(effective_mu_u) >= u_bound:
        raise ValueError(
            "The requested prior center for log(theta) must be finite and strictly "
            f"inside (-u_clip, u_clip); got {effective_mu_u!r} with u_clip={u_bound!r}."
        )

    # Center the raw Gaussian so its smooth transform is exactly the requested
    # center in effective log(theta) space.
    mu_u_raw = raw_value_for_effective_center(effective_mu_u, u_bound)

    # ---- Data PyTree for the black-box VI engine
    # Keep everything JAX-friendly.
    data = {
        "rho": rho,
        "r_nb": r_nb,
    }

    def _split_theta(
        theta_vec: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray]:
        """Split the engine parameter vector and return z, u, theta.

        If theta is estimated, the vector is [z, u] and theta = exp(u).
        If theta is fixed, the vector is [z] and theta = fixed_theta.
        """
        z_raw = theta_vec[:num_free]
        if estimate_theta:
            u_raw = theta_vec[num_free]
            u = _smooth_bound(u_raw, u_bound)
            theta = jnp.exp(u)
            return z_raw, u_raw, theta

        assert fixed_theta is not None
        theta = jnp.asarray(fixed_theta, dtype=theta_vec.dtype).reshape(())
        return z_raw, None, theta

    def loglik(theta_vec: jnp.ndarray, data_: Any) -> jnp.ndarray:
        # Only free OD cells occur in z. Frozen-zero cells are absent from the
        # assignment vector; positive frozen values are compact constants.
        z_raw, _, theta = _split_theta(theta_vec)
        z = _smooth_bound(z_raw, z_bound)
        f = assemble_assignment_demand(z)

        # Forward model: assignment + aggregation + detection
        out = forward_model_from_demand(
            inputs=forward_inputs,
            f=f,
            theta=theta,
            rho=data_["rho"],
            assignment_inputs=assignment_inputs,
            fixed_routing=fixed_routing,
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
        z_raw, u_raw, _ = _split_theta(theta_vec)

        # The Gaussian priors apply to the raw unconstrained variables. Their
        # smooth bounded transforms are used only by the forward model.
        lp_abs = dist.Normal(0.0, sigma_z).log_prob(z_raw).sum()
        if estimate_theta:
            assert u_raw is not None
            lp_abs = lp_abs + dist.Normal(mu_u_raw, sigma_u).log_prob(u_raw).sum()

        # The VI engine uses a base Normal(0,1) site. If correction is OFF (default),
        # we should provide the increment: log p - log N(0,1).
        if request.vi.use_base_normal_correction:
            return lp_abs

        lp_base = dist.Normal(0.0, 1.0).log_prob(z_raw).sum()
        if estimate_theta:
            assert u_raw is not None
            lp_base = lp_base + dist.Normal(0.0, 1.0).log_prob(u_raw).sum()
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
        raise RuntimeError(
            f"Unexpected posterior sample shape: {samples.shape}, expected (S, {dim})"
        )

    z_raw_samples = samples[:, :num_free]
    z_samples = _smooth_bound_numpy(z_raw_samples, z_bound)

    if estimate_theta:
        u_raw_samples = samples[:, num_free]
        u_samples = _smooth_bound_numpy(u_raw_samples, u_bound)
        theta_samples = np.exp(u_samples)
    else:
        assert fixed_theta is not None
        theta_samples = np.full(samples.shape[0], fixed_theta, dtype=float)
    f0_np = np.asarray(f0)
    if layout is None:
        f_samples = f0_np[None, :] * np.exp(z_samples)
    else:
        f_samples = layout.reconstruct_numpy(z_samples)

    theta_mean = float(theta_samples.mean())
    theta_sd = float(theta_samples.std(ddof=0))
    f_mean = f_samples.mean(axis=0)

    return ODThetaInferenceResult(
        vi=vi_res,
        num_od=num_od,
        num_free_od=num_free,
        num_fixed_od=len(fixed_indices),
        runtime_profile=runtime_profile,
        f0=np.asarray(f0_np, dtype=float),
        theta_samples=np.asarray(theta_samples, dtype=float),
        f_samples=np.asarray(f_samples, dtype=float),
        theta_mean=theta_mean,
        theta_sd=theta_sd,
        f_mean=np.asarray(f_mean, dtype=float),
        estimate_theta=estimate_theta,
        fixed_theta=fixed_theta,
        fingerprint=str(request.fingerprint),
        od_layout_fingerprint=(None if layout is None else layout.fingerprint),
        od_layout_payload_json=(
            None if layout is None else layout.fingerprint_payload_json
        ),
        compact_layout_fingerprint=(
            None if compact_layout is None else compact_layout.fingerprint
        ),
        compact_layout_payload_json=(
            None if compact_layout is None else compact_layout.fingerprint_payload_json
        ),
        fingerprint_payload_json=(
            None
            if request.fingerprint_payload_json is None
            else str(request.fingerprint_payload_json)
        ),
    )
