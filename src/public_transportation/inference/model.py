# src/public_transportation/inference/model.py
"""public_transportation.inference.model

JAX-compliant forward model for Bayesian inference.

This module wires together (deterministically):
  - OD log-deviation parameterization: f = f0 * exp(z)
  - Assignment evaluation from either a full or compact demand vector
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

from public_transportation.inference.assignment_adapter import (
    AssignmentInputs,
    FixedRoutingInputs,
    assign_link_flow,
    assign_link_flow_fixed_routing,
)
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
    """Outputs of the forward model used by the likelihood.

    ``assignment_demand`` follows the indexing encoded by ``AssignmentInputs``.
    It is a full OD vector for the legacy/all-free path and a compact active-OD
    vector for reduced estimation. It must not be interpreted as a reporting
    vector without consulting the corresponding layout.
    """

    assignment_demand: Array  # (num_assignment_od,)
    link_flow: Array  # (num_links,)
    lambda_m: Array  # (M,)
    mu_m: Array  # (M,)

    @property
    def f(self) -> Array:
        """Compatibility alias for ``assignment_demand``."""
        return self.assignment_demand


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
    fixed_routing: FixedRoutingInputs | None = None,
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

    return _forward_from_demand(
        inputs=inputs,
        f=f,
        theta=theta,
        rho=rho,
        assignment_inputs=assignment_inputs,
        fixed_routing=fixed_routing,
    )


def forward_model_from_demand(
    *,
    inputs: ForwardModelInputs,
    f: Array,
    theta: Array,
    rho: Array,
    assignment_inputs: AssignmentInputs,
    fixed_routing: FixedRoutingInputs | None = None,
) -> ForwardModelOutputs:
    """Run assignment and measurement prediction from an explicit demand vector.

    The vector may use full OD order or a compact assignment order; its required
    shape is defined by ``assignment_inputs.od_origin_node``.
    """
    return _forward_from_demand(
        inputs=inputs,
        f=f,
        theta=theta,
        rho=rho,
        assignment_inputs=assignment_inputs,
        fixed_routing=fixed_routing,
    )


def _forward_from_demand(
    *,
    inputs: ForwardModelInputs,
    f: Array,
    theta: Array,
    rho: Array,
    assignment_inputs: AssignmentInputs,
    fixed_routing: FixedRoutingInputs | None,
) -> ForwardModelOutputs:
    """Canonical assignment and measurement plumbing for an assignment vector."""
    f_j = jnp.asarray(f)
    expected_shape = assignment_inputs.od_origin_node.shape
    if f_j.ndim != 1 or f_j.shape != expected_shape:
        raise ValueError(f"f must have shape {expected_shape}, got {f_j.shape}.")
    if fixed_routing is None:
        link_flow = assign_link_flow(inputs=assignment_inputs, f=f_j, theta=theta)
    else:
        link_flow = assign_link_flow_fixed_routing(
            inputs=assignment_inputs,
            routing=fixed_routing,
            f=f_j,
        )
    lambda_m = predict_measurements(
        link_flow=link_flow,
        num_measurements=inputs.num_measurements,
        measurement_index=inputs.measurement_index,
        link_index=inputs.link_index,
    )
    mu_m = apply_detection_rate(lambda_m=lambda_m, rho=rho)
    return ForwardModelOutputs(
        assignment_demand=f_j,
        link_flow=link_flow,
        lambda_m=lambda_m,
        mu_m=mu_m,
    )
