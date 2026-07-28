from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.domain import Scenario
from public_transportation.inference.assignment_adapter import (
    assign_link_flow_fixed_routing,
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    choose_fixed_measurement_operator,
    fixed_routing_measurement_operator_cache_path,
    load_or_prepare_fixed_routing_measurement_operator,
    measurement_mapping_fingerprint,
    predict_measurements_fixed_operator,
    prepare_fixed_routing_measurement_operator,
    validate_fixed_routing_measurement_operator,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout
from public_transportation.inference.maximum_likelihood_pipeline import (
    build_od_theta_ml_problem,
)
from public_transportation.inference.pipeline import ODThetaEstimationRequest
from public_transportation.measurement.likelihood_jax import (
    predict_measurements_from_link_flow,
)
from public_transportation.measurement.mapping import AggregationSpec

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
def artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("measurement-operator")
    for name in NETWORK_FILES:
        shutil.copy2(EXAMPLE / "data" / name, directory / name)
    shutil.copy2(
        EXAMPLE / "pre_processing/results/demand.csv", directory / "demand.csv"
    )
    scenario = Scenario.from_folder(directory, strict=True)
    return prepare_assignment(scenario=scenario, config=AssignmentConfig())


def _spec(num_links: int) -> AggregationSpec:
    links = np.arange(min(num_links, 8), dtype=np.int32)
    return AggregationSpec(
        num_measurements=3,
        measurement_index=np.arange(links.size, dtype=np.int32) % 3,
        link_index=links,
    )


def _reference(inputs, routing, spec, demand):
    link_flow = assign_link_flow_fixed_routing(
        inputs=inputs,
        routing=routing,
        f=demand,
    )
    return predict_measurements_from_link_flow(
        link_flow,
        spec_num_measurements=spec.num_measurements,
        spec_measurement_index=jnp.asarray(spec.measurement_index),
        spec_link_index=jnp.asarray(spec.link_index),
    )


@pytest.fixture(scope="module")
def all_free_operator(artifacts):
    inputs = build_assignment_inputs(artifacts=artifacts)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    operator = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="simple-example-02",
        representation="dense",
        chunk_size=8,
    )
    return inputs, routing, spec, operator


@pytest.mark.parametrize("scale", [0.0, 0.25, 1.0, 2.0])
def test_dense_operator_matches_fixed_loader_and_scaling(all_free_operator, scale):
    inputs, routing, spec, operator = all_free_operator
    demand = jnp.linspace(0.0, 7.0 * scale, operator.num_free_od)

    direct = predict_measurements_fixed_operator(
        operator=operator,
        free_demand=demand,
        rho=jnp.asarray(1.0),
    )
    reference = _reference(inputs, routing, spec, demand)

    np.testing.assert_allclose(direct, reference, rtol=3e-5, atol=3e-5)


def test_operator_preserves_superposition(all_free_operator):
    _, _, _, operator = all_free_operator
    first = jnp.linspace(0.0, 2.0, operator.num_free_od)
    second = jnp.linspace(1.0, 0.0, operator.num_free_od)

    def predict(demand):
        return predict_measurements_fixed_operator(
            operator=operator, free_demand=demand, rho=jnp.asarray(1.0)
        )

    np.testing.assert_allclose(
        predict(first + second),
        predict(first) + predict(second),
        rtol=2e-6,
        atol=2e-6,
    )


def test_dense_operator_gradient_matches_reference(all_free_operator):
    inputs, routing, spec, operator = all_free_operator
    demand = jnp.linspace(0.1, 3.0, operator.num_free_od)

    direct_gradient = jax.grad(
        lambda value: jnp.square(
            predict_measurements_fixed_operator(
                operator=operator,
                free_demand=value,
                rho=jnp.asarray(0.7),
            )
        ).sum()
    )(demand)
    reference_gradient = jax.grad(
        lambda value: jnp.square(0.7 * _reference(inputs, routing, spec, value)).sum()
    )(demand)

    np.testing.assert_allclose(
        direct_gradient,
        reference_gradient,
        rtol=5e-5,
        atol=5e-5,
    )


def test_sparse_operator_matches_dense(all_free_operator):
    inputs, routing, spec, dense = all_free_operator
    sparse = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="simple-example-02",
        representation="bcoo",
        chunk_size=8,
    )
    demand = jnp.linspace(0.0, 4.0, dense.num_free_od)

    dense_value = predict_measurements_fixed_operator(
        operator=dense, free_demand=demand, rho=jnp.asarray(0.8)
    )
    sparse_value = predict_measurements_fixed_operator(
        operator=sparse, free_demand=demand, rho=jnp.asarray(0.8)
    )
    np.testing.assert_allclose(sparse_value, dense_value, rtol=2e-6, atol=2e-6)
    assert sparse.metrics.stored_bytes <= sparse.metrics.dense_bytes


def test_compact_positive_frozen_flow_becomes_offset(artifacts):
    full_inputs = build_assignment_inputs(artifacts=artifacts)
    num_od = int(full_inputs.od_origin_node.shape[0])
    free_indices = tuple(range(1, num_od))
    layout = ODParameterLayout(
        num_od_total=num_od,
        od_keys=tuple((f"o{i}", "d", "t") for i in range(num_od)),
        free_od_indices=free_indices,
        fixed_od_indices=(0,),
        fixed_od_values=(4.0,),
        free_baseline_values=tuple(1.0 for _ in free_indices),
        fixed_zero_indices=(),
        fixed_positive_indices=(0,),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    operator = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="compact",
        compact_layout=compact,
        od_layout_fingerprint=layout.fingerprint,
        chunk_size=8,
    )
    free_demand = jnp.linspace(1.0, 3.0, layout.num_free)
    active_demand = jnp.zeros((compact.num_active,), dtype=free_demand.dtype)
    active_demand = active_demand.at[jnp.asarray(compact.free_compact_indices)].set(
        free_demand
    )
    active_demand = active_demand.at[jnp.asarray(compact.fixed_compact_indices)].set(
        jnp.asarray(compact.fixed_compact_values)
    )

    direct = predict_measurements_fixed_operator(
        operator=operator, free_demand=free_demand, rho=jnp.asarray(1.0)
    )
    reference = _reference(inputs, routing, spec, active_demand)
    np.testing.assert_allclose(direct, reference, rtol=3e-5, atol=3e-5)
    fixed_only = active_demand.at[jnp.asarray(compact.free_compact_indices)].set(0.0)
    np.testing.assert_allclose(
        operator.fixed_measurement_offset,
        _reference(inputs, routing, spec, fixed_only),
        rtol=3e-5,
        atol=3e-5,
    )


def test_provenance_mismatch_is_rejected(all_free_operator):
    inputs, routing, spec, operator = all_free_operator
    changed_spec = replace(
        spec,
        link_index=np.asarray([1], dtype=np.int32),
        measurement_index=np.asarray([0], dtype=np.int32),
    )
    assert measurement_mapping_fingerprint(changed_spec) != operator.mapping_fingerprint

    with pytest.raises(ValueError, match="mapping fingerprint mismatch"):
        validate_fixed_routing_measurement_operator(
            operator=operator,
            inputs=inputs,
            routing=routing,
            spec=changed_spec,
            assignment_fingerprint="simple-example-02",
            compact_layout=None,
            od_layout_fingerprint=None,
        )


def test_empty_compact_layout_produces_zero_measurements(artifacts):
    full_inputs = build_assignment_inputs(artifacts=artifacts)
    num_od = int(full_inputs.od_origin_node.shape[0])
    layout = ODParameterLayout(
        num_od_total=num_od,
        od_keys=tuple((f"o{i}", "d", "t") for i in range(num_od)),
        free_od_indices=(),
        fixed_od_indices=tuple(range(num_od)),
        fixed_od_values=tuple(0.0 for _ in range(num_od)),
        free_baseline_values=(),
        fixed_zero_indices=tuple(range(num_od)),
        fixed_positive_indices=(),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    operator = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="empty",
        compact_layout=compact,
        od_layout_fingerprint=layout.fingerprint,
    )

    prediction = predict_measurements_fixed_operator(
        operator=operator,
        free_demand=jnp.empty((0,), dtype=jnp.float32),
        rho=jnp.asarray(0.5),
    )
    assert operator.matrix.shape == (spec.num_measurements, 0)
    np.testing.assert_array_equal(prediction, np.zeros(spec.num_measurements))


def test_operator_records_memory_and_construction_metrics(all_free_operator):
    _, _, _, operator = all_free_operator
    metrics = operator.metrics
    assert metrics.construction_seconds >= 0.0
    assert metrics.dense_bytes == operator.num_measurements * operator.num_free_od * 4
    assert metrics.peak_construction_bytes >= 0
    assert 0 <= metrics.nonzero_entries <= metrics.total_entries
    assert metrics.density == pytest.approx(
        0.0
        if metrics.total_entries == 0
        else metrics.nonzero_entries / metrics.total_entries
    )
    assert metrics.compilation_count == 1
    assert metrics.chunk_shape == (8, operator.num_measurements)
    assert metrics.num_chunks > 0
    assert metrics.routing_loading_seconds >= 0.0
    assert metrics.device_synchronization_seconds >= 0.0
    assert metrics.numpy_transfer_seconds >= 0.0


def test_progress_is_reported_for_each_fixed_shape_chunk(artifacts):
    inputs = build_assignment_inputs(artifacts=artifacts)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    events = []
    operator = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=_spec(inputs.graph.num_links),
        assignment_fingerprint="progress",
        chunk_size=7,
        progress=events.append,
    )
    assert len(events) == operator.metrics.num_chunks
    assert all(
        tuple(event["shape"]) == (7, operator.num_measurements) for event in events
    )
    assert events[-1]["chunk"] == events[-1]["chunks"]


@pytest.mark.parametrize("representation", ["dense", "bcoo"])
def test_persistent_cache_reuse_and_invalid_file_rebuild(
    artifacts, tmp_path, representation
):
    inputs = build_assignment_inputs(artifacts=artifacts)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    spec = _spec(inputs.graph.num_links)
    kwargs = dict(
        cache_directory=tmp_path,
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="persistent",
        representation=representation,
        chunk_size=8,
    )
    built = load_or_prepare_fixed_routing_measurement_operator(**kwargs)
    loaded = load_or_prepare_fixed_routing_measurement_operator(**kwargs)
    assert not built.metrics.cache_hit
    assert loaded.metrics.cache_hit
    np.testing.assert_allclose(
        np.asarray(
            loaded.matrix.todense() if representation == "bcoo" else loaded.matrix
        ),
        np.asarray(
            built.matrix.todense() if representation == "bcoo" else built.matrix
        ),
    )
    path = fixed_routing_measurement_operator_cache_path(
        cache_directory=tmp_path,
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint="persistent",
        representation=representation,
    )
    path.write_bytes(b"not a valid cache")
    rebuilt = load_or_prepare_fixed_routing_measurement_operator(**kwargs)
    assert not rebuilt.metrics.cache_hit


def test_auto_activation_policy_respects_cache_and_break_even():
    assert (
        choose_fixed_measurement_operator(
            mode="off", cached=True, expected_evaluations=100, construction_seconds=1.0
        )
        is None
    )
    assert (
        choose_fixed_measurement_operator(
            mode="dense",
            cached=False,
            expected_evaluations=0,
            construction_seconds=None,
        )
        == "dense"
    )
    assert (
        choose_fixed_measurement_operator(
            mode="auto", cached=True, expected_evaluations=0, construction_seconds=None
        )
        == "bcoo"
    )
    assert (
        choose_fixed_measurement_operator(
            mode="auto",
            cached=False,
            expected_evaluations=5,
            construction_seconds=20.0,
            reference_evaluation_seconds=1.94,
        )
        is None
    )
    assert (
        choose_fixed_measurement_operator(
            mode="auto",
            cached=False,
            expected_evaluations=20,
            construction_seconds=20.0,
            reference_evaluation_seconds=1.94,
        )
        == "bcoo"
    )


def test_ml_problem_cache_hit_does_not_prepare_routing(
    artifacts, tmp_path, monkeypatch
):
    inputs = build_assignment_inputs(artifacts=artifacts)
    num_od = int(inputs.od_origin_node.shape[0])
    request = ODThetaEstimationRequest(
        fingerprint="routing-free-cache-hit",
        f0=jnp.ones((num_od,)),
        y_obs=jnp.asarray([2.0, 1.0, 3.0]),
        mapping_spec=_spec(inputs.graph.num_links),
        baseline_theta=1.0,
        estimate_theta=False,
        fixed_theta=1.0,
        assignment_artifacts=artifacts,
        fixed_measurement_operator="bcoo",
        fixed_measurement_operator_cache_directory=tmp_path,
        fixed_measurement_operator_chunk_size=8,
    )
    built = build_od_theta_ml_problem(request)
    assert built.fixed_measurement_operator is not None
    assert not built.fixed_measurement_operator.metrics.cache_hit

    import public_transportation.inference.maximum_likelihood_pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "prepare_fixed_routing",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("routing must not be prepared on a valid cache hit")
        ),
    )
    loaded = build_od_theta_ml_problem(request)
    assert loaded.fixed_measurement_operator is not None
    assert loaded.fixed_measurement_operator.metrics.cache_hit


def test_ml_likelihood_objective_gradient_and_solution_inputs_are_equivalent(artifacts):
    inputs = build_assignment_inputs(artifacts=artifacts)
    spec = _spec(inputs.graph.num_links)
    num_od = int(inputs.od_origin_node.shape[0])
    common = dict(
        fingerprint="simple-example-02",
        f0=jnp.linspace(1.0, 2.0, num_od),
        y_obs=jnp.asarray([2.0, 1.0, 3.0]),
        mapping_spec=spec,
        baseline_theta=1.0,
        estimate_theta=False,
        fixed_theta=1.0,
        rho=0.8,
        nb_dispersion=10.0,
        assignment_artifacts=artifacts,
    )
    reference = build_od_theta_ml_problem(
        ODThetaEstimationRequest(**common, fixed_measurement_operator="off")
    )
    optimized = build_od_theta_ml_problem(
        ODThetaEstimationRequest(**common, fixed_measurement_operator="dense")
    )

    def objective(problem, parameter):
        return -(problem.loglik(parameter, problem.data) + problem.logprior(parameter))

    for parameter in (
        jnp.zeros((num_od,)),
        jnp.linspace(-0.3, 0.4, num_od),
        jnp.linspace(0.2, -0.1, num_od),
    ):
        reference_value, reference_gradient = jax.value_and_grad(
            lambda value: objective(reference, value)
        )(parameter)
        optimized_value, optimized_gradient = jax.value_and_grad(
            lambda value: objective(optimized, value)
        )(parameter)
        np.testing.assert_allclose(
            optimized.loglik(parameter, optimized.data),
            reference.loglik(parameter, reference.data),
            rtol=5e-5,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            optimized_value,
            reference_value,
            rtol=5e-5,
            atol=5e-5,
        )
        np.testing.assert_allclose(
            optimized_gradient,
            reference_gradient,
            rtol=8e-5,
            atol=8e-5,
        )
