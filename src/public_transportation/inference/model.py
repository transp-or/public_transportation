# src/public_transportation/inference/model.py
"""public_transportation.inference.model

JAX-compliant forward model for Bayesian inference.

This module wires together (deterministically):
  - OD log-deviation parameterization: f = f0 * exp(z)
  - Assignment evaluation (adapter): link_flow = assign_link_flow(inputs=assignment_inputs, f=f, theta=theta)
  - Measurement aggregation (structural mapping): lambda_m = aggregate(link_flow, spec)
  - Detection rate: mu_m = rho * lambda_m

Design constraints
------------------
- Must be JAX-traceable: no Python-side data-dependent control flow in core computations.
- Must not invent new indexing conventions: all alignment is done outside this module.
- Keep responsibilities narrow: this module contains only forward-model plumbing,
  not priors, not inference algorithms, and no IO.

Important
---------
This file must contain exactly ONE implementation of `forward_model`, and it must
return `ForwardModelOutputs` (not a tuple). The inference pipeline relies on
attribute access like `out.link_flow`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from public_transportation.inference.assignment_adapter import AssignmentInputs, assign_link_flow
from public_transportation.measurement.likelihood_jax import predict_measurements_from_link_flow
from public_transportation.measurement.mapping import AggregationSpec

Array = jnp.ndarray


# -----------------------------------------------------------------------------
# Inputs / outputs
# -----------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class ForwardModelInputs:
    """Static inputs needed by the forward model.

    We store the aggregation specification as arrays (JAX-friendly) rather than
    the Python `AggregationSpec` dataclass, to keep traced computations robust.

    Parameters
    ----------
    f0:
        Baseline OD vector aligned to assignment OD indexing, shape (num_od,).
    num_measurements:
        Number of measurements M (Python int).
    measurement_index:
        Shape (K,), int32. For each contribution k, target measurement index m.
    link_index:
        Shape (K,), int32. For each contribution k, contributing link index ℓ.
    """

    f0: Array
    num_measurements: int
    measurement_index: Array  # (K,), int32
    link_index: Array         # (K,), int32

    def tree_flatten(self):
        children = (self.f0, self.measurement_index, self.link_index)
        aux = (int(self.num_measurements),)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (m,) = aux
        f0, measurement_index, link_index = children
        return cls(
            f0=f0,
            num_measurements=int(m),
            measurement_index=measurement_index,
            link_index=link_index,
        )


@dataclass(frozen=True, slots=True)
class ForwardModelOutputs:
    """Outputs of the forward model (used by likelihood)."""

    f: Array         # (num_od,)
    link_flow: Array # (num_links,)
    lambda_m: Array  # (M,)
    mu_m: Array      # (M,)


def make_forward_inputs(*, f0: Array, spec: AggregationSpec) -> ForwardModelInputs:
    """Convert (f0, AggregationSpec) into JAX-ready ForwardModelInputs.

    Responsibility
    --------------
    Pure conversion: no inference logic, no assignment logic, no IO.
    """
    return ForwardModelInputs(
        f0=jnp.asarray(f0),
        num_measurements=int(spec.num_measurements),
        measurement_index=jnp.asarray(spec.measurement_index, dtype=jnp.int32),
        link_index=jnp.asarray(spec.link_index, dtype=jnp.int32),
    )


# -----------------------------------------------------------------------------
# Core JAX-safe building blocks
# -----------------------------------------------------------------------------

@jax.jit
def build_od_from_deviation(*, f0: Array, z: Array) -> Array:
    """Compute OD vector f = f0 * exp(z) (positivity guaranteed)."""
    f0_j = jnp.asarray(f0)
    z_j = jnp.asarray(z, dtype=f0_j.dtype)
    return f0_j * jnp.exp(z_j)


@jax.jit(static_argnames=("num_measurements",))
def predict_measurements(
    *,
    link_flow: Array,
    num_measurements: int,
    measurement_index: Array,
    link_index: Array,
) -> Array:
    """Compute λ (predicted measurement before detection) from link_flow.

    Delegates to the canonical JAX implementation in measurement.likelihood_jax.
    """
    return predict_measurements_from_link_flow(
        link_flow=jnp.asarray(link_flow),
        spec_num_measurements=int(num_measurements),
        spec_measurement_index=jnp.asarray(measurement_index, dtype=jnp.int32),
        spec_link_index=jnp.asarray(link_index, dtype=jnp.int32),
    )


@jax.jit
def apply_detection_rate(*, lambda_m: Array, rho: Array) -> Array:
    """Compute μ = rho * λ with scalar rho in (0, 1]."""
    rho_s = jnp.asarray(rho).reshape(())
    return rho_s * lambda_m


# -----------------------------------------------------------------------------
# Full forward model
# -----------------------------------------------------------------------------

def forward_model(
    *,
    inputs: ForwardModelInputs,
    z: Array,
    theta: Array,
    rho: Array,
    assignment_inputs: AssignmentInputs,
) -> ForwardModelOutputs:
    """Run the full forward model (assignment + aggregation + detection).

    Responsibility
    --------------
    Pure forward evaluation:
      (z, theta, rho) -> (f, link_flow, lambda_m, mu_m)

    No priors. No VI. No IO.
    """
    # f = f0 * exp(z)
    f = build_od_from_deviation(f0=inputs.f0, z=z)

    # link_flow = assignment(f, theta)
    link_flow = assign_link_flow(inputs=assignment_inputs, f=f, theta=theta)

    # lambda_m = aggregation(link_flow)
    lambda_m = predict_measurements(
        link_flow=link_flow,
        num_measurements=inputs.num_measurements,
        measurement_index=inputs.measurement_index,
        link_index=inputs.link_index,
    )

    # mu_m = rho * lambda_m
    mu_m = apply_detection_rate(lambda_m=lambda_m, rho=rho)

    return ForwardModelOutputs(
        f=f,
        link_flow=link_flow,
        lambda_m=lambda_m,
        mu_m=mu_m,
    )