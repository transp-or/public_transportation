# src/public_transportation/inference/likelihood.py
"""
Inference-side likelihood wiring (no IO, no PyMC).

This module bridges:
- measurement-side structural mapping (AggregationSpec, y_obs), and
- pure-JAX measurement likelihood implementation in `public_transportation.measurement.likelihood_jax` (NB likelihood + aggregation).

Responsibilities
----------------
- Convert mapper outputs (NumPy / Python objects) into JAX-friendly arrays.
- Provide small, explicit functions to compute:
    * y_pred (aggregated predicted measurements),
    * mu = rho * y_pred (expected counts),
    * total Negative Binomial log-likelihood.

Non-responsibilities
--------------------
- No mapping logic: use `public_transportation.measurement.mapping.build_mapping_spec_strict` (or `build_measurement_vectors`) to obtain (y_obs, AggregationSpec).
- No IO: reading measurement CSV is handled by measurement/io.
- No assignment: producing `link_flow` is handled by the assignment module.
- No PyMC code: model construction belongs in `public_transportation.inference.model`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from public_transportation.measurement.mapping import AggregationSpec
from public_transportation.measurement.likelihood_jax import (
    negbinom_loglikelihood,
    predict_measurements_from_link_flow,
    measurement_loglik_from_link_flow,
)

Array = jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class PreparedLikelihoodInputs:
    """JAX-ready likelihood inputs.

    Attributes
    ----------
    y_obs:
        Observed measurement vector, shape (M,). Stored as a JAX array.
        (Counts are integer-valued in principle, but we accept float arrays
        as long as they represent nonnegative integers.)

    spec_num_measurements:
        Number of measurements M (Python int, used as a static shape).

    spec_measurement_index:
        Flat measurement indices, shape (K,), dtype int32.

    spec_link_index:
        Flat link indices, shape (K,), dtype int32.

    Notes
    -----
    - This object is safe to pass around in Python. When calling JAX-traced
      functions, its fields are used as arrays/scalars only.
    - Construct this once (Python side) and reuse it across likelihood evaluations.
    - `spec_num_measurements` is ultimately used as a static shape argument in the underlying JAX-jitted aggregation.
    """

    y_obs: Array
    spec_num_measurements: int
    spec_measurement_index: Array
    spec_link_index: Array

    def tree_flatten(self):
        # Children are JAX arrays; aux data are Python scalars / static objects.
        children = (
            self.y_obs,
            self.spec_measurement_index,
            self.spec_link_index,
        )
        aux = (int(self.spec_num_measurements),)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (m,) = aux
        y_obs, mi, li = children
        return cls(
            y_obs=y_obs,
            spec_num_measurements=int(m),
            spec_measurement_index=mi,
            spec_link_index=li,
        )


def prepare_likelihood_inputs(
    *,
    y_obs: object,
    spec: AggregationSpec,
) -> PreparedLikelihoodInputs:
    """Prepare JAX-ready likelihood inputs from mapper outputs.

    Parameters
    ----------
    y_obs:
        Observed measurement vector produced by the mapper (often a NumPy array).
        Will be converted to a 1D JAX array.

    spec:
        AggregationSpec produced by the strict mapper (Python-side). Must contain:
          - num_measurements: int
          - measurement_index: np.ndarray int32
          - link_index: np.ndarray int32

    Returns
    -------
    PreparedLikelihoodInputs
        Ready to be used by pure-JAX log-likelihood functions.

    Raises
    ------
    ValueError
        If shapes are inconsistent.
    """
    y_obs_j = jnp.asarray(y_obs)
    if y_obs_j.ndim != 1:
        raise ValueError(f"y_obs must be 1D, got shape {y_obs_j.shape}")

    m = int(spec.num_measurements)
    if int(y_obs_j.shape[0]) != m:
        raise ValueError(
            "spec.num_measurements must match len(y_obs): "
            f"{m} vs {int(y_obs_j.shape[0])}"
        )

    mi = jnp.asarray(spec.measurement_index, dtype=jnp.int32)
    li = jnp.asarray(spec.link_index, dtype=jnp.int32)

    if mi.ndim != 1 or li.ndim != 1:
        raise ValueError(
            "spec.measurement_index and spec.link_index must be 1D: "
            f"{mi.shape} vs {li.shape}"
        )
    if int(mi.shape[0]) != int(li.shape[0]):
        raise ValueError(
            "spec.measurement_index and spec.link_index must have same length K: "
            f"{int(mi.shape[0])} vs {int(li.shape[0])}"
        )

    return PreparedLikelihoodInputs(
        y_obs=y_obs_j,
        spec_num_measurements=m,
        spec_measurement_index=mi,
        spec_link_index=li,
    )


@jax.jit(static_argnames=())
def predict_y(
    *,
    link_flow: Array,
    prepared: PreparedLikelihoodInputs,
) -> Array:
    """Compute predicted measurements y_pred from link_flow via aggregation spec.

    Returns
    -------
    y_pred: Array
        Shape (M,).
    """
    return predict_measurements_from_link_flow(
        link_flow=link_flow,
        spec_num_measurements=prepared.spec_num_measurements,
        spec_measurement_index=prepared.spec_measurement_index,
        spec_link_index=prepared.spec_link_index,
    )


@jax.jit
def predict_mu(
    *,
    link_flow: Array,
    rho: Array,
    prepared: PreparedLikelihoodInputs,
    eps_mu: float = 1e-9,
) -> Array:
    """Compute expected counts mu = rho * y_pred (with epsilon floor).

    Notes
    -----
    - `rho` is a single scalar in (0, 1] in the current design.
    - `eps_mu` prevents mu=0 which can cause log(0) in NB logpmf.
    """
    y_pred = predict_y(link_flow=link_flow, prepared=prepared)
    mu = rho * y_pred
    mu = jnp.maximum(mu, jnp.asarray(eps_mu, dtype=mu.dtype))
    return mu


@jax.jit
def loglikelihood_from_link_flow(
    *,
    link_flow: Array,
    prepared: PreparedLikelihoodInputs,
    theta: Array,
    rho: Array,
    r: Array,
    eps_mu: float = 1e-9,
) -> Array:
    """Total NB log-likelihood for observed measurements given link_flow.

    Parameters
    ----------
    link_flow:
        Assignment link flows, shape (num_links,).

    prepared:
        PreparedLikelihoodInputs holding y_obs and aggregation spec arrays.

    theta:
        Logit dispersion parameter. Included explicitly for model clarity.
        It typically affects `link_flow` upstream through assignment; it is not
        used directly here.

    rho:
        Single scalar detection rate in (0, 1], scaling continuous flows into counts.

    r:
        NB dispersion/shape parameter, r > 0.

    Returns
    -------
    total_loglik: Array
        Scalar.
    """
    ll = measurement_loglik_from_link_flow(
        link_flow=link_flow,
        y_obs=prepared.y_obs,
        theta=theta,
        rho=rho,
        r=r,
        spec_num_measurements=prepared.spec_num_measurements,
        spec_measurement_index=prepared.spec_measurement_index,
        spec_link_index=prepared.spec_link_index,
        eps_mu=eps_mu,
    )
    return ll


@jax.jit
def loglikelihood_from_measurement_mean(
    *,
    mu: Array,
    prepared: PreparedLikelihoodInputs,
    r: Array,
    eps_mu: float = 1e-9,
) -> Array:
    """Evaluate the NB likelihood from an already aggregated measurement mean."""
    mean = jnp.asarray(mu)
    expected_shape = (prepared.spec_num_measurements,)
    if mean.shape != expected_shape:
        raise ValueError(f"mu must have shape {expected_shape}, got {mean.shape}.")
    mean = jnp.maximum(mean, jnp.asarray(eps_mu, dtype=mean.dtype))
    return negbinom_loglikelihood(y_obs=prepared.y_obs, mu=mean, r=r)
