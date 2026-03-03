"""public_transportation.inference.priors

Prior utilities for Bayesian inference.

This module provides **strict**, reflection-free helpers to build prior inputs.

Responsibilities
----------------
- Build the baseline OD vector f0 from `Scenario.demand.records`.
- Ensure f0 is aligned to the **assignment OD indexing convention** defined by
  `AssignmentIDManager`.
- Provide a JAX-safe OD parameterization helper `od_from_log_deviation`.
- Provide a numeric helper for a theta prior scale.

Non-responsibilities
--------------------
- No IO.
- No mapping.
- No inference engine wiring.

Important conventions
---------------------
- The assignment consumes OD vectors in **scenario order**
  (`Scenario.demand.records` iteration order).
- `AssignmentIDManager` stores OD keys in both scenario and canonical order;
  the caller can request either convention.

Strictness
----------
This module fails fast if:
- Scenario has no demand or no time bins.
- Demand records do not match the OD keys frozen in the AssignmentIDManager.
- Any flow is negative.

No reflection rule
------------------
This file intentionally avoids `getattr` / `hasattr`.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Literal

import numpy as np
import jax.numpy as jnp

from public_transportation.assignment.id_manager import AssignmentIDManager

from .types import ODDeviationPriorSpec, ThetaPriorSpec


# -----------------------------------------------------------------------------
# Baseline OD vector f0
# -----------------------------------------------------------------------------

ODConvention = Literal["scenario", "canonical"]


def _build_time_bin_index_by_id(*, scenario: Any) -> dict[str, int]:
    """Build a strict mapping time_bin_id -> time_bin_index from Scenario.

    Assumptions
    -----------
    - `scenario.time_bins` is an iterable of objects exposing `bin_id`.
    - indices are their position in `scenario.time_bins`.

    Raises
    ------
    ValueError
        If time bins are missing or if bin_id is missing/duplicate.
    """
    if scenario.time_bins is None or len(scenario.time_bins) == 0:
        raise ValueError("Scenario has no time bins.")

    out: dict[str, int] = {}
    for idx, tb in enumerate(scenario.time_bins):
        tb_id = tb.bin_id
        if not tb_id:
            raise ValueError("Scenario time bin has empty bin_id.")
        if tb_id in out:
            raise ValueError(f"Duplicate time bin id in scenario.time_bins: {tb_id}")
        out[tb_id] = int(idx)
    return out


def _validate_demand_matches_id_manager(
    *,
    scenario: Any,
    id_manager: AssignmentIDManager,
    time_bin_index_by_id: Mapping[str, int],
) -> None:
    """Validate that Scenario demand records match the ID manager scenario OD keys.

    Responsibility
    --------------
    One thing: verify strict alignment and fail fast with actionable errors.

    Notes
    -----
    - We validate against `id_manager.od_keys_scenario` because that is the OD
      convention consumed by the assignment.
    """
    if scenario.demand is None:
        raise ValueError("Scenario has no demand.")

    records = scenario.demand.records
    if records is None or len(records) == 0:
        raise ValueError("Scenario demand has zero records.")

    if len(records) != id_manager.num_od:
        raise ValueError(
            "Scenario demand record count does not match AssignmentIDManager.num_od: "
            f"{len(records)} vs {id_manager.num_od}."
        )

    # Strict key-by-key match in scenario order
    for k, r in enumerate(records):
        if not r.origin_stop_id or not r.dest_stop_id:
            raise ValueError(f"Demand record {k} has empty origin/destination stop id.")
        if not r.time_bin_id:
            raise ValueError(f"Demand record {k} has empty time_bin_id.")
        try:
            tb_idx = int(time_bin_index_by_id[r.time_bin_id])
        except KeyError as e:
            raise ValueError(f"Unknown time_bin_id in demand record {k}: {r.time_bin_id}") from e

        expected = id_manager.od_keys_scenario[k].as_tuple()
        got = (r.origin_stop_id, r.dest_stop_id, tb_idx)
        if got != expected:
            raise ValueError(
                "Scenario demand is not aligned with AssignmentIDManager OD convention at index "
                f"{k}. Expected {expected}, got {got}."
            )


def build_f0_from_scenario_demand(
    *,
    scenario: Any,
    id_manager: AssignmentIDManager,
    dtype: Any = jnp.float32,
    convention: ODConvention = "scenario",
) -> jnp.ndarray:
    """Build the baseline OD vector f0 from `Scenario.demand.records` (strict).

    Parameters
    ----------
    scenario:
        Scenario instance containing `demand.records` and `time_bins`.
    id_manager:
        AssignmentIDManager built from the same scenario+graph.
    dtype:
        Output dtype.
    convention:
        - "scenario": return f0 in assignment-consumed scenario order.
        - "canonical": return f0 in the stable canonical order.

    Returns
    -------
    f0: jax.numpy.ndarray, shape (num_od,)
        Baseline OD vector aligned to the requested convention.

    Raises
    ------
    ValueError
        If scenario data is missing or inconsistent with the ID manager.
    """
    if convention not in ("scenario", "canonical"):
        raise ValueError(f"Unknown convention: {convention!r}")

    # Build time-bin lookup (strict)
    time_bin_index_by_id = _build_time_bin_index_by_id(scenario=scenario)

    # Validate strict alignment against the ID manager
    _validate_demand_matches_id_manager(
        scenario=scenario,
        id_manager=id_manager,
        time_bin_index_by_id=time_bin_index_by_id,
    )

    # Scenario order is assignment order: f0_scenario[k] = records[k].flow
    records = scenario.demand.records
    f0_scenario = np.empty((id_manager.num_od,), dtype=np.float64)
    for k, r in enumerate(records):
        flow = float(r.flow)
        if flow < 0.0:
            raise ValueError(f"Negative flow in scenario.demand at index {k}: {flow}")
        f0_scenario[k] = flow

    if convention == "scenario":
        return jnp.asarray(f0_scenario, dtype=dtype)

    # canonical
    f0_canonical = id_manager.od_values_scenario_to_canonical(f0_scenario)
    return jnp.asarray(f0_canonical, dtype=dtype)


# -----------------------------------------------------------------------------
# Log-deviation parameterization
# -----------------------------------------------------------------------------


def od_from_log_deviation(*, f0: Any, z: Any) -> jnp.ndarray:
    """Compute OD vector f = f0 * exp(z) (JAX-safe).

    Parameters
    ----------
    f0:
        Baseline OD vector, shape (num_od,).
    z:
        Log-deviations, shape (num_od,).

    Returns
    -------
    f:
        Positive OD vector, shape (num_od,).
    """
    f0_j = jnp.asarray(f0)
    z_j = jnp.asarray(z, dtype=f0_j.dtype)
    return f0_j * jnp.exp(z_j)


# -----------------------------------------------------------------------------
# Theta prior helper (from typical cost)
# -----------------------------------------------------------------------------


def theta_prior_from_typical_cost(
    *,
    typical_cost: float,
    rel_loc: float = 0.25,
    rel_scale: float = 0.25,
    lower: float = 1e-6,
) -> ThetaPriorSpec:
    """Create a simple truncated-normal prior spec for theta based on a cost scale.

    Parameters
    ----------
    typical_cost:
        Positive cost scale (minutes).
    rel_loc:
        Mean as a fraction of typical_cost.
    rel_scale:
        Stddev as a fraction of typical_cost.
    lower:
        Strict lower bound for theta.

    Returns
    -------
    ThetaPriorSpec
    """
    c = float(typical_cost)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError(f"typical_cost must be positive and finite, got {typical_cost!r}")

    loc = float(rel_loc) * c
    scale = float(rel_scale) * c
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("rel_scale must yield a positive, finite scale.")

    return ThetaPriorSpec(loc=loc, scale=scale, lower=float(lower))


def validate_od_deviation_prior(spec: ODDeviationPriorSpec) -> None:
    """Validate ODDeviationPriorSpec."""
    if not np.isfinite(spec.sigma_z) or spec.sigma_z <= 0.0:
        raise ValueError(f"od_prior.sigma_z must be positive and finite, got {spec.sigma_z!r}")