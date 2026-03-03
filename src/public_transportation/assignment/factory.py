from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from public_transportation.domain.scenario import Scenario

from .assign import AssignmentArtifacts, AssignmentResult, assign, prepare_assignment, _assign_core
from .config import AssignmentConfig
from .costs import link_costs
from .id_manager import AssignmentIDManager

Array = jnp.ndarray


@dataclass(frozen=True, slots=True)
class AssignmentFactory:
    """Holds one-time-built objects and exposes JIT-compiled assignment callables."""

    artifacts: AssignmentArtifacts
    id_manager: AssignmentIDManager
    base_link_cost: Array

    # Fast callables
    link_flow_fn: Callable[[Array, float | Array], Array]
    link_flow_and_group_fn: Callable[[Array, float | Array], tuple[Array, Array]]
    def run(
        self,
        *,
        od_values: Array,
        theta: float | Array = None,
        return_group_link_flows: bool = False,
    ) -> AssignmentResult:
        """Run the full assignment and return an `AssignmentResult`.

        This is the *public* high-level interface for scripts, reporting, and debugging.

        Notes:
          - This uses the package's `assign(...)` routine and returns a rich object
            (theta used, link_flow, link_cost, and optionally per-group flows).
          - For inference / PyMC, prefer `self.link_flow_fn(...)` or
            `self.link_flow_and_group_fn(...)` which are JIT compiled and array-only.
        """
        od_values = jnp.asarray(od_values)
        return assign(
            od_values=od_values,
            artifacts=self.artifacts,
            theta=theta,
            return_group_link_flows=return_group_link_flows,
        )

def build_assignment_factory(
    *,
    scenario: Scenario,
    config: AssignmentConfig,
) -> AssignmentFactory:
    """Build artifacts once and return fast JIT-compiled assignment functions.

    Input OD convention:
      - `od_values[k]` corresponds to `scenario.demand.records[k]` (scenario order).

    Output link convention:
      - `link_flow[i]` corresponds to `graph` link index `i` (tail/head/link_type/... arrays).

    The ID manager freezes these conventions (and provides canonical OD keys).
    """

    # -----------------------------
    # One-time preprocessing (Python)
    # -----------------------------
    artifacts = prepare_assignment(scenario, config)

    # Freeze conventions (scenario OD order + graph link order)
    id_manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)

    # Precompute base generalized link costs once
    base_link_cost = link_costs(
        graph=artifacts.graph,
        cost_parts=artifacts.cost_parts,
        config=artifacts.config,
    )
    base_link_cost = jnp.asarray(base_link_cost)

    # Extract ODGroups arrays once (avoid object passing to jit)
    odg = artifacts.od_groups
    group_dest_node = jnp.asarray(odg.group_dest_node)
    group_link_mask = jnp.asarray(odg.group_link_mask)
    od_origin_node = jnp.asarray(odg.od_origin_node)
    group_od_index_padded = jnp.asarray(odg.group_od_index_padded)
    group_od_mask = jnp.asarray(odg.group_od_mask)

    graph = artifacts.graph

    # -----------------------------
    # JIT kernels (array-only inputs)
    # -----------------------------

    @jax.jit
    def _jit_link_flow_only(od_values: Array, theta_arr: Array) -> Array:
        total, _ = _assign_core(
            graph=graph,
            od_values=od_values,
            base_link_cost=base_link_cost,
            theta=theta_arr,
            group_dest_node=group_dest_node,
            group_link_mask=group_link_mask,
            od_origin_node=od_origin_node,
            group_od_index_padded=group_od_index_padded,
            group_od_mask=group_od_mask,
            return_group_link_flows=False,
        )
        return total

    @jax.jit
    def _jit_link_flow_and_group(od_values: Array, theta_arr: Array) -> tuple[Array, Array]:
        total, per_group = _assign_core(
            graph=graph,
            od_values=od_values,
            base_link_cost=base_link_cost,
            theta=theta_arr,
            group_dest_node=group_dest_node,
            group_link_mask=group_link_mask,
            od_origin_node=od_origin_node,
            group_od_index_padded=group_od_index_padded,
            group_od_mask=group_od_mask,
            return_group_link_flows=True,
        )
        # per_group is guaranteed not None here
        return total, per_group  # type: ignore[return-value]

    # -----------------------------
    # Public wrappers (stabilize theta dtype; avoid recompilation on theta)
    # -----------------------------

    def link_flow_fn(od_values: Array, theta: float | Array) -> Array:
        od_values = jnp.asarray(od_values)
        theta_arr = jnp.asarray(theta, dtype=od_values.dtype)
        return _jit_link_flow_only(od_values, theta_arr)

    def link_flow_and_group_fn(od_values: Array, theta: float | Array) -> tuple[Array, Array]:
        od_values = jnp.asarray(od_values)
        theta_arr = jnp.asarray(theta, dtype=od_values.dtype)
        return _jit_link_flow_and_group(od_values, theta_arr)

    return AssignmentFactory(
        artifacts=artifacts,
        id_manager=id_manager,
        base_link_cost=base_link_cost,
        link_flow_fn=link_flow_fn,
        link_flow_and_group_fn=link_flow_and_group_fn,
    )