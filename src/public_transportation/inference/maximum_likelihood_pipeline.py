"""Reduced-dimensional ML/MAP problem construction for OD and theta."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from public_transportation.inference.assignment_adapter import build_assignment_inputs
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.likelihood import (
    loglikelihood_from_link_flow,
    prepare_likelihood_inputs,
)
from public_transportation.inference.model import make_forward_inputs, forward_model_from_demand
from public_transportation.inference.parameterization import (
    raw_value_for_effective_center,
    smooth_bound,
    smooth_bound_numpy,
)
from public_transportation.inference.runtime_profile import (
    ODAssignmentRuntimeProfile,
    build_od_assignment_runtime_profile,
)


@dataclass(frozen=True, slots=True)
class ODThetaMLProblem:
    """Engine-ready ML/MAP problem containing only estimable coordinates."""

    dim: int
    num_free_od: int
    num_fixed_od: int
    runtime_profile: ODAssignmentRuntimeProfile
    od_layout_fingerprint: str | None
    od_layout_payload_json: str | None
    compact_layout_fingerprint: str | None
    compact_layout_payload_json: str | None
    data: dict[str, jnp.ndarray]
    loglik: Callable[[jnp.ndarray, Any], jnp.ndarray]
    logprior: Callable[[jnp.ndarray], jnp.ndarray]
    theta0: jnp.ndarray
    decode: Callable[[object], tuple[np.ndarray, float]]


def build_od_theta_ml_problem(request: Any) -> ODThetaMLProblem:
    """Build ML/MAP closures with the same model and layout used by VI."""
    if request.assignment_artifacts is None:
        raise ValueError("request.assignment_artifacts must be provided (output of prepare_assignment).")
    f0 = jnp.asarray(request.f0)
    if f0.ndim != 1:
        raise ValueError(f"f0 must be 1D, got shape {f0.shape}")
    y_obs = jnp.asarray(request.y_obs)
    if y_obs.ndim != 1:
        raise ValueError(f"y_obs must be 1D, got shape {y_obs.shape}")
    if y_obs.shape[0] != request.mapping_spec.num_measurements:
        raise ValueError("y_obs length does not match mapping_spec.num_measurements.")

    num_od = int(f0.shape[0])
    layout = request.od_layout
    if layout is None:
        num_free = num_od
        free_f0 = f0
    else:
        if layout.num_od_total != num_od:
            raise ValueError("od_layout.num_od_total must match f0 length.")
        num_free = layout.num_free
        free_f0 = jnp.asarray(layout.free_baseline_values, dtype=f0.dtype)
        if not np.array_equal(
            np.asarray(f0)[np.asarray(layout.free_od_indices, dtype=int)],
            np.asarray(free_f0),
        ):
            raise ValueError("od_layout free baselines must exactly match f0 at free OD indices.")

    estimate_theta = bool(request.estimate_theta)
    if estimate_theta:
        if request.fixed_theta is not None:
            raise ValueError("fixed_theta must be None when estimate_theta=True.")
        fixed_theta = None
    else:
        if request.fixed_theta is None:
            raise ValueError("fixed_theta must be provided when estimate_theta=False.")
        fixed_theta = float(request.fixed_theta)
        if not np.isfinite(fixed_theta) or fixed_theta <= 0.0:
            raise ValueError("fixed_theta must be positive and finite.")

    z_bound = float(request.z_clip)
    u_bound = float(request.u_clip)
    sigma_z = float(request.sigma_z)
    sigma_u = float(request.sigma_u)
    if not np.isfinite(z_bound) or z_bound <= 0.0:
        raise ValueError("z_clip must be positive and finite.")
    if not np.isfinite(u_bound) or u_bound <= 0.0:
        raise ValueError("u_clip must be positive and finite.")
    if not np.isfinite(sigma_z) or sigma_z <= 0.0:
        raise ValueError("sigma_z must be positive and finite.")
    if not np.isfinite(sigma_u) or sigma_u <= 0.0:
        raise ValueError("sigma_u must be positive and finite.")

    if request.mu_u_strategy == "center_at_baseline":
        baseline_theta = float(request.baseline_theta)
        if not np.isfinite(baseline_theta) or baseline_theta <= 0.0:
            raise ValueError("baseline_theta must be positive and finite.")
        effective_mu_u = float(np.log(baseline_theta))
    elif request.mu_u_strategy == "fixed":
        if request.mu_u_fixed is None:
            raise ValueError("mu_u_fixed must be provided when mu_u_strategy='fixed'.")
        effective_mu_u = float(request.mu_u_fixed)
    else:
        raise ValueError(f"Unknown mu_u_strategy: {request.mu_u_strategy!r}")
    if not np.isfinite(effective_mu_u) or abs(effective_mu_u) >= u_bound:
        raise ValueError("The prior center for log(theta) must lie strictly inside its bound.")
    mu_u_raw = raw_value_for_effective_center(effective_mu_u, u_bound)

    compact_layout = (
        None if layout is None else build_compact_od_assignment_layout(parameter_layout=layout)
    )
    assignment_inputs = build_assignment_inputs(
        artifacts=request.assignment_artifacts,
        compact_layout=compact_layout,
    )
    runtime_profile = build_od_assignment_runtime_profile(
        num_od_total=num_od,
        parameter_layout=layout,
        compact_layout=compact_layout,
        artifacts=request.assignment_artifacts,
        assignment_inputs=assignment_inputs,
    )
    forward_inputs = make_forward_inputs(f0=f0, spec=request.mapping_spec)
    prepared = prepare_likelihood_inputs(y_obs=y_obs, spec=request.mapping_spec)
    data = {
        "rho": jnp.asarray(float(request.rho)).reshape(()),
        "r_nb": jnp.asarray(float(request.nb_dispersion)).reshape(()),
    }
    dim = num_free + int(estimate_theta)

    def split(parameter: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray]:
        z_raw = parameter[:num_free]
        if estimate_theta:
            u_raw = parameter[num_free]
            return z_raw, u_raw, jnp.exp(smooth_bound(u_raw, u_bound))
        assert fixed_theta is not None
        return z_raw, None, jnp.asarray(fixed_theta, dtype=parameter.dtype).reshape(())

    def reconstruct(z: jnp.ndarray) -> jnp.ndarray:
        if layout is None:
            return free_f0 * jnp.exp(z)
        assert compact_layout is not None
        return compact_layout.assemble_compact_jax(z)

    def loglik(parameter: jnp.ndarray, data_: Any) -> jnp.ndarray:
        z_raw, _, theta = split(parameter)
        f = reconstruct(smooth_bound(z_raw, z_bound))
        out = forward_model_from_demand(
            inputs=forward_inputs,
            f=f,
            theta=theta,
            rho=data_["rho"],
            assignment_inputs=assignment_inputs,
        )
        value = loglikelihood_from_link_flow(
            link_flow=out.link_flow,
            prepared=prepared,
            theta=theta,
            rho=data_["rho"],
            r=data_["r_nb"],
        )
        return jnp.where(jnp.isfinite(value), value, -jnp.inf)

    def logprior(parameter: jnp.ndarray) -> jnp.ndarray:
        z_raw, u_raw, _ = split(parameter)
        value = dist.Normal(0.0, sigma_z).log_prob(z_raw).sum()
        if estimate_theta:
            assert u_raw is not None
            value = value + dist.Normal(mu_u_raw, sigma_u).log_prob(u_raw)
        return value

    theta0 = jnp.zeros((dim,), dtype=f0.dtype)
    if estimate_theta:
        theta0 = theta0.at[num_free].set(mu_u_raw)

    def decode(parameter: object) -> tuple[np.ndarray, float]:
        raw = np.asarray(parameter, dtype=float)
        if raw.shape != (dim,):
            raise ValueError(f"parameter must have shape ({dim},), got {raw.shape}.")
        z = smooth_bound_numpy(raw[:num_free], z_bound)
        if layout is None:
            f = np.asarray(f0, dtype=float) * np.exp(z)
        else:
            f = layout.reconstruct_numpy(z)
        if estimate_theta:
            theta = float(np.exp(smooth_bound_numpy(raw[num_free], u_bound)))
        else:
            assert fixed_theta is not None
            theta = fixed_theta
        return np.asarray(f, dtype=float), theta

    return ODThetaMLProblem(
        dim=dim,
        num_free_od=num_free,
        num_fixed_od=(0 if layout is None else layout.num_fixed),
        runtime_profile=runtime_profile,
        od_layout_fingerprint=(None if layout is None else layout.fingerprint),
        od_layout_payload_json=(None if layout is None else layout.fingerprint_payload_json),
        compact_layout_fingerprint=(
            None if compact_layout is None else compact_layout.fingerprint
        ),
        compact_layout_payload_json=(
            None if compact_layout is None else compact_layout.fingerprint_payload_json
        ),
        data=data,
        loglik=loglik,
        logprior=logprior,
        theta0=theta0,
        decode=decode,
    )
