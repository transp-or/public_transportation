from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import (
    _routing_inputs_for_destination,
    prepare_assignment,
)
from public_transportation.assignment.dial_dp import prepare_destination_routing
from public_transportation.domain import Scenario
from public_transportation.inference.assignment_adapter import (
    FixedRoutingInputs,
    assign_link_flow,
    assign_link_flow_fixed_routing,
    assign_link_flow_fixed_routing_custom_adjoint,
    build_assignment_inputs,
    prepare_fixed_routing,
    validate_fixed_routing_compatibility,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "docs/source/examples/simple_example_02"
NETWORK_FILES = (
    "metadata.json",
    "stops.csv",
    "lines.csv",
    "trips.csv",
    "stop_times.csv",
    "time_bins.csv",
)


@pytest.fixture(scope="module")
def assignment_inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fixed-routing-inputs")
    for name in NETWORK_FILES:
        shutil.copy2(EXAMPLE / "data" / name, directory / name)
    shutil.copy2(EXAMPLE / "pre_processing/results/demand.csv", directory / "demand.csv")
    scenario = Scenario.from_folder(directory, strict=True)
    artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
    return build_assignment_inputs(artifacts=artifacts)


def test_prepare_fixed_routing_shapes_values_and_pytree(assignment_inputs):
    prepared = prepare_fixed_routing(inputs=assignment_inputs, theta=1.0)

    num_groups = int(assignment_inputs.group_dest_node.shape[0])
    num_links = assignment_inputs.graph.num_links
    assert isinstance(prepared, FixedRoutingInputs)
    assert prepared.graph is assignment_inputs.graph
    assert prepared.group_link_probability.shape == (num_groups, num_links)
    assert prepared.effective_group_link_mask.shape == (num_groups, num_links)
    assert np.array_equal(prepared.group_dest_node, assignment_inputs.group_dest_node)
    assert np.array_equal(
        prepared.source_group_link_mask,
        assignment_inputs.group_link_mask,
    )
    assert np.array_equal(prepared.source_base_link_cost, assignment_inputs.base_link_cost)
    probability = np.asarray(prepared.group_link_probability)
    effective = np.asarray(prepared.effective_group_link_mask)
    assert np.all(np.isfinite(probability))
    assert np.all(probability >= 0.0)
    assert np.all(probability[~effective] <= np.exp(-80.0))

    children, treedef = jax.tree_util.tree_flatten(prepared)
    rebuilt = jax.tree_util.tree_unflatten(treedef, children)
    assert isinstance(rebuilt, FixedRoutingInputs)
    assert np.array_equal(rebuilt.group_link_probability, probability)
    validate_fixed_routing_compatibility(inputs=assignment_inputs, routing=rebuilt)


def test_fixed_routing_depends_on_theta_and_is_deterministic(assignment_inputs):
    first = prepare_fixed_routing(inputs=assignment_inputs, theta=0.5)
    repeated = prepare_fixed_routing(inputs=assignment_inputs, theta=0.5)
    different = prepare_fixed_routing(inputs=assignment_inputs, theta=5.0)

    np.testing.assert_array_equal(
        first.group_link_probability,
        repeated.group_link_probability,
    )
    assert not np.allclose(
        np.asarray(first.group_link_probability),
        np.asarray(different.group_link_probability),
    )


def test_prepare_fixed_routing_supports_no_active_destination_groups(assignment_inputs):
    empty = replace(
        assignment_inputs,
        group_dest_node=jnp.empty((0,), dtype=jnp.int32),
        group_link_mask=jnp.empty(
            (0, assignment_inputs.graph.num_links),
            dtype=bool,
        ),
    )
    prepared = prepare_fixed_routing(inputs=empty, theta=1.0)

    assert prepared.group_dest_node.shape == (0,)
    assert prepared.effective_group_link_mask.shape == (
        0,
        assignment_inputs.graph.num_links,
    )
    assert prepared.group_link_probability.shape == (
        0,
        assignment_inputs.graph.num_links,
    )
def test_cached_probabilities_match_independent_destination_routing(assignment_inputs):
    prepared = prepare_fixed_routing(inputs=assignment_inputs, theta=1.0)
    group_index = 0
    destination = assignment_inputs.group_dest_node[group_index]
    enabled, cost = _routing_inputs_for_destination(
        graph=assignment_inputs.graph,
        base_link_cost=assignment_inputs.base_link_cost,
        group_link_mask=assignment_inputs.group_link_mask[group_index],
        dest_node=destination,
    )
    independent = prepare_destination_routing(
        graph=assignment_inputs.graph,
        link_cost=cost,
        enabled_link_mask=enabled,
        dest_node=destination,
        theta=1.0,
    )

    np.testing.assert_array_equal(
        prepared.effective_group_link_mask[group_index],
        independent.enabled_link_mask,
    )
    np.testing.assert_allclose(
        prepared.group_link_probability[group_index],
        independent.link_prob,
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_fixed_routing_rejects_incompatible_assignment_inputs(assignment_inputs):
    prepared = prepare_fixed_routing(inputs=assignment_inputs, theta=1.0)
    changed_cost = replace(
        assignment_inputs,
        base_link_cost=assignment_inputs.base_link_cost.at[0].add(1.0),
    )

    with pytest.raises(ValueError, match="base link costs"):
        validate_fixed_routing_compatibility(inputs=changed_cost, routing=prepared)


@pytest.mark.parametrize("scale", [0.0, 0.4, 1.0, 2.5])
def test_fixed_routing_loading_matches_dynamic_assignment(assignment_inputs, scale):
    theta = 1.0
    routing = prepare_fixed_routing(inputs=assignment_inputs, theta=theta)
    demand = jnp.linspace(
        0.0,
        20.0 * scale,
        assignment_inputs.od_origin_node.shape[0],
        dtype=jnp.float32,
    )

    dynamic = assign_link_flow(
        inputs=assignment_inputs,
        f=demand,
        theta=jnp.asarray(theta),
    )
    cached = assign_link_flow_fixed_routing(
        inputs=assignment_inputs,
        routing=routing,
        f=demand,
    )

    np.testing.assert_allclose(cached, dynamic, rtol=2.0e-6, atol=2.0e-6)


def test_fixed_routing_loading_gradient_matches_dynamic_assignment(assignment_inputs):
    theta = 1.0
    routing = prepare_fixed_routing(inputs=assignment_inputs, theta=theta)
    demand = jnp.linspace(
        0.1,
        12.0,
        assignment_inputs.od_origin_node.shape[0],
        dtype=jnp.float32,
    )

    def dynamic_objective(value):
        flow = assign_link_flow(
            inputs=assignment_inputs,
            f=value,
            theta=jnp.asarray(theta),
        )
        return jnp.square(flow).sum()

    def cached_objective(value):
        flow = assign_link_flow_fixed_routing(
            inputs=assignment_inputs,
            routing=routing,
            f=value,
        )
        return jnp.square(flow).sum()

    dynamic_gradient = jax.grad(dynamic_objective)(demand)
    cached_gradient = jax.grad(cached_objective)(demand)
    np.testing.assert_allclose(
        cached_gradient,
        dynamic_gradient,
        rtol=3.0e-5,
        atol=3.0e-5,
    )


def test_fixed_routing_custom_adjoint_matches_ordinary_autodiff(assignment_inputs):
    routing = prepare_fixed_routing(inputs=assignment_inputs, theta=1.0)
    demand = jnp.linspace(
        0.1,
        12.0,
        assignment_inputs.od_origin_node.shape[0],
        dtype=jnp.float32,
    )
    weight = jnp.sin(
        jnp.arange(assignment_inputs.graph.num_links, dtype=jnp.float32) * 0.13
    )

    def ordinary(value):
        return jnp.vdot(
            assign_link_flow_fixed_routing(
                inputs=assignment_inputs, routing=routing, f=value
            ),
            weight,
        )

    def explicit(value):
        return jnp.vdot(
            assign_link_flow_fixed_routing_custom_adjoint(
                inputs=assignment_inputs, routing=routing, f=value
            ),
            weight,
        )

    ordinary_value, ordinary_gradient = jax.value_and_grad(ordinary)(demand)
    explicit_value, explicit_gradient = jax.value_and_grad(explicit)(demand)
    np.testing.assert_allclose(explicit_value, ordinary_value, rtol=0, atol=0)
    np.testing.assert_allclose(
        explicit_gradient, ordinary_gradient, rtol=2e-5, atol=2e-5
    )


@pytest.mark.parametrize("theta", [0.0, -1.0, np.inf, np.nan])
def test_prepare_fixed_routing_rejects_invalid_theta(assignment_inputs, theta):
    with pytest.raises(ValueError, match="positive and finite"):
        prepare_fixed_routing(inputs=assignment_inputs, theta=theta)
