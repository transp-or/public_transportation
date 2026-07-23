from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import (
    FixedODDemand,
    FixedODRecord,
    Scenario,
    read_fixed_demand_csv,
)
from public_transportation.estimation.bayesian import VIConfig
from public_transportation.estimation.maximum_likelihood import MLConfig, run_ml
from public_transportation.inference.assignment_adapter import (
    assign_link_flow,
    build_assignment_inputs,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.compact_od_groups import compact_od_groups
from public_transportation.inference.complexity import build_od_parameter_complexity
from public_transportation.inference.likelihood import (
    loglikelihood_from_link_flow,
    predict_y,
    prepare_likelihood_inputs,
)
from public_transportation.inference.maximum_likelihood_pipeline import build_od_theta_ml_problem
from public_transportation.inference.od_parameter_layout import build_od_parameter_layout
from public_transportation.inference.pipeline import ODThetaEstimationRequest, estimate_od_theta_vi
from public_transportation.inference.priors import build_f0_from_scenario_demand
from public_transportation.inference.results_io import (
    ODThetaVIResults,
    load_od_theta_vi_results,
    save_od_theta_vi_results,
)
from public_transportation.measurement import build_mapping_spec_strict, read_measurements_csv


@pytest.mark.skipif(
    os.environ.get("RUN_FROZEN_ESTIMATION_ACCEPTANCE") != "1",
    reason="opt-in real JAX/assignment acceptance test",
)
def test_real_vi_and_ml_use_only_free_od_coordinates(tmp_path):
    jax.clear_caches()
    root = Path(__file__).resolve().parents[2] / "docs/source/examples/simple_example_01"
    for name in (
        "metadata.json",
        "stops.csv",
        "lines.csv",
        "trips.csv",
        "stop_times.csv",
        "time_bins.csv",
    ):
        shutil.copy2(root / "data" / name, tmp_path / name)
    shutil.copy2(root / "pre_processing/results/demand.csv", tmp_path / "demand.csv")

    scenario = Scenario.from_folder(tmp_path)
    fixed = read_fixed_demand_csv(root / "data/fixed_demand.csv", scenario=scenario)
    layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed)
    artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
    id_manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)
    measurements = read_measurements_csv(
        root / "pre_processing/results/measurements_boarding_alighting.csv"
    )
    mapped = build_mapping_spec_strict(
        id_manager=id_manager,
        table=measurements,
        include_link_lists_for_report=False,
    )
    f0 = build_f0_from_scenario_demand(
        scenario=scenario,
        id_manager=id_manager,
        dtype=jnp.float32,
    )
    compact_layout = build_compact_od_assignment_layout(parameter_layout=layout)
    compact_groups = compact_od_groups(
        od_groups=artifacts.od_groups,
        layout=compact_layout,
    )
    full_assignment_inputs = build_assignment_inputs(artifacts=artifacts)
    compact_assignment_inputs = replace(
        full_assignment_inputs,
        group_dest_node=compact_groups.group_dest_node,
        group_link_mask=compact_groups.group_link_mask,
        od_origin_node=compact_groups.od_origin_node,
        group_od_index_padded=compact_groups.group_od_index_padded,
        group_od_mask=compact_groups.group_od_mask,
    )
    z0 = jnp.zeros((layout.num_free,), dtype=jnp.float32)
    full_flow = assign_link_flow(
        inputs=full_assignment_inputs,
        f=layout.reconstruct_jax(z0),
        theta=jnp.asarray(5.0),
    )
    compact_flow = assign_link_flow(
        inputs=compact_assignment_inputs,
        f=compact_layout.assemble_compact_jax(z0),
        theta=jnp.asarray(5.0),
    )
    assert compact_groups.num_od == compact_layout.num_active
    assert np.allclose(compact_flow, full_flow, rtol=1e-6, atol=1e-6)

    prepared = prepare_likelihood_inputs(y_obs=mapped.y_obs, spec=mapped.spec)
    full_prediction = predict_y(link_flow=full_flow, prepared=prepared)
    compact_prediction = predict_y(link_flow=compact_flow, prepared=prepared)
    assert np.allclose(compact_prediction, full_prediction, rtol=1e-6, atol=1e-6)
    likelihood_kwargs = {
        "prepared": prepared,
        "theta": jnp.asarray(5.0),
        "rho": jnp.asarray(1.0),
        "r": jnp.asarray(50.0),
    }
    full_loglik = loglikelihood_from_link_flow(
        link_flow=full_flow,
        **likelihood_kwargs,
    )
    compact_loglik = loglikelihood_from_link_flow(
        link_flow=compact_flow,
        **likelihood_kwargs,
    )
    assert float(compact_loglik) == pytest.approx(float(full_loglik), rel=1e-6, abs=1e-6)

    def full_objective(z):
        flow = assign_link_flow(
            inputs=full_assignment_inputs,
            f=layout.reconstruct_jax(z),
            theta=jnp.asarray(5.0),
        )
        return jnp.square(flow).sum()

    def compact_objective(z):
        flow = assign_link_flow(
            inputs=compact_assignment_inputs,
            f=compact_layout.assemble_compact_jax(z),
            theta=jnp.asarray(5.0),
        )
        return jnp.square(flow).sum()

    assert np.allclose(
        jax.grad(compact_objective)(z0),
        jax.grad(full_objective)(z0),
        rtol=2e-5,
        atol=2e-5,
    )

    def full_loglik_objective(parameter):
        z = parameter[:-1]
        theta = jnp.exp(parameter[-1])
        flow = assign_link_flow(
            inputs=full_assignment_inputs,
            f=layout.reconstruct_jax(z),
            theta=theta,
        )
        return loglikelihood_from_link_flow(
            link_flow=flow,
            prepared=prepared,
            theta=theta,
            rho=jnp.asarray(1.0),
            r=jnp.asarray(50.0),
        )

    def compact_loglik_objective(parameter):
        z = parameter[:-1]
        theta = jnp.exp(parameter[-1])
        flow = assign_link_flow(
            inputs=compact_assignment_inputs,
            f=compact_layout.assemble_compact_jax(z),
            theta=theta,
        )
        return loglikelihood_from_link_flow(
            link_flow=flow,
            prepared=prepared,
            theta=theta,
            rho=jnp.asarray(1.0),
            r=jnp.asarray(50.0),
        )

    probes = (
        jnp.concatenate((z0, jnp.asarray([np.log(5.0)], dtype=z0.dtype))),
        jnp.concatenate(
            (
                jnp.linspace(-0.35, 0.45, layout.num_free, dtype=z0.dtype),
                jnp.asarray([np.log(3.5)], dtype=z0.dtype),
            )
        ),
    )
    for probe in probes:
        compact_value, compact_gradient = jax.value_and_grad(compact_loglik_objective)(probe)
        full_value, full_gradient = jax.value_and_grad(full_loglik_objective)(probe)
        assert float(compact_value) == pytest.approx(
            float(full_value), rel=2e-5, abs=2e-5
        )
        assert np.allclose(compact_gradient, full_gradient, rtol=3e-5, atol=3e-5)

    # Freeze every cell for destination D at zero: its assignment group must
    # disappear while the remaining link-flow result stays equivalent.
    removed_destination_fixed = FixedODDemand(
        records=tuple(
            FixedODRecord(
                record.origin_stop_id,
                record.dest_stop_id,
                record.time_bin_id,
                0.0,
            )
            for record in scenario.demand.records
            if record.dest_stop_id == "D"
        )
    )
    removed_destination_layout = build_od_parameter_layout(
        scenario=scenario,
        fixed_demand=removed_destination_fixed,
    )
    removed_compact_layout = build_compact_od_assignment_layout(
        parameter_layout=removed_destination_layout
    )
    removed_groups = compact_od_groups(
        od_groups=artifacts.od_groups,
        layout=removed_compact_layout,
    )
    assert removed_groups.group_dest_node.shape[0] == (
        artifacts.od_groups.group_dest_node.shape[0] - 1
    )
    removed_inputs = build_assignment_inputs(
        artifacts=artifacts,
        compact_layout=removed_compact_layout,
    )
    removed_z = jnp.linspace(
        -0.2,
        0.3,
        removed_destination_layout.num_free,
        dtype=jnp.float32,
    )
    removed_full_flow = assign_link_flow(
        inputs=full_assignment_inputs,
        f=removed_destination_layout.reconstruct_jax(removed_z),
        theta=jnp.asarray(4.0),
    )
    removed_compact_flow = assign_link_flow(
        inputs=removed_inputs,
        f=removed_compact_layout.assemble_compact_jax(removed_z),
        theta=jnp.asarray(4.0),
    )
    assert np.allclose(removed_compact_flow, removed_full_flow, rtol=1e-6, atol=1e-6)

    request = ODThetaEstimationRequest(
        fingerprint=str(id_manager.fingerprint),
        f0=f0,
        y_obs=jnp.asarray(mapped.y_obs, dtype=jnp.float32),
        mapping_spec=mapped.spec,
        baseline_theta=5.0,
        od_layout=layout,
        estimate_theta=False,
        fixed_theta=5.0,
        assignment_artifacts=artifacts,
        vi=VIConfig(
            guide="auto_normal",
            num_steps=1,
            num_posterior_draws=2,
            log_every=1,
        ),
    )

    vi_result = estimate_od_theta_vi(request)
    assert vi_result.vi.posterior_samples_theta.shape == (2, layout.num_free)
    assert vi_result.od_layout_fingerprint == layout.fingerprint
    assert vi_result.od_layout_payload_json == layout.fingerprint_payload_json
    assert np.array_equal(
        vi_result.f_samples[:, layout.fixed_od_indices],
        np.broadcast_to(layout.fixed_od_values, (2, layout.num_fixed)),
    )

    complexity = build_od_parameter_complexity(
        layout=layout,
        estimate_theta=False,
        guide="auto_normal",
        compute_hessian=True,
    )
    assert complexity.statistical_dim == layout.num_free
    assert complexity.optimizer_vector_size == layout.num_free
    assert complexity.gradient_size == layout.num_free
    assert complexity.hessian_element_count == layout.num_free**2
    assert complexity.assignment_od_vector_size == compact_layout.num_active

    result_path = tmp_path / "vi_results.npz"
    save_od_theta_vi_results(
        path=result_path,
        results=ODThetaVIResults(
            fingerprint=vi_result.fingerprint,
            fingerprint_payload_json=vi_result.fingerprint_payload_json,
            f0=vi_result.f0,
            theta_samples=vi_result.theta_samples,
            f_samples=vi_result.f_samples,
            theta_mean=vi_result.theta_mean,
            theta_sd=vi_result.theta_sd,
            f_mean=vi_result.f_mean,
            vi_losses=vi_result.vi.losses,
            free_od_indices=layout.free_od_indices,
            fixed_od_indices=layout.fixed_od_indices,
            fixed_od_values=layout.fixed_od_values,
            od_layout_fingerprint=vi_result.od_layout_fingerprint,
            od_layout_payload_json=vi_result.od_layout_payload_json,
            compact_layout_fingerprint=vi_result.compact_layout_fingerprint,
            compact_layout_payload_json=vi_result.compact_layout_payload_json,
            runtime_profile=vi_result.runtime_profile,
        ),
    )
    loaded = load_od_theta_vi_results(result_path)
    assert loaded.od_layout_fingerprint == layout.fingerprint
    assert loaded.num_free_od == layout.num_free
    assert loaded.runtime_profile == vi_result.runtime_profile
    assert np.array_equal(
        loaded.f_samples[:, layout.fixed_od_indices],
        np.broadcast_to(layout.fixed_od_values, (2, layout.num_fixed)),
    )

    problem = build_od_theta_ml_problem(request)
    assert problem.dim == layout.num_free
    assert problem.od_layout_fingerprint == layout.fingerprint
    assert problem.od_layout_payload_json == layout.fingerprint_payload_json
    ml_result = run_ml(
        dim=problem.dim,
        data=problem.data,
        loglik=problem.loglik,
        logprior=problem.logprior,
        theta0=problem.theta0,
        config=MLConfig(maxiter=1, prior_weight=0.0, compute_hessian=False),
    )
    f_hat, theta_hat = problem.decode(ml_result.theta_hat)
    assert theta_hat == 5.0
    assert np.array_equal(f_hat[np.asarray(layout.fixed_od_indices)], layout.fixed_od_values)

    # Exercise both complete engines when every OD cell is structurally zero.
    all_zero_fixed = FixedODDemand(
        records=tuple(
            FixedODRecord(
                record.origin_stop_id,
                record.dest_stop_id,
                record.time_bin_id,
                0.0,
            )
            for record in scenario.demand.records
        )
    )
    all_zero_layout = build_od_parameter_layout(
        scenario=scenario,
        fixed_demand=all_zero_fixed,
    )
    all_zero_request = replace(
        request,
        od_layout=all_zero_layout,
        vi=replace(request.vi, num_steps=1, num_posterior_draws=2),
    )
    all_zero_vi = estimate_od_theta_vi(all_zero_request)
    assert all_zero_vi.vi.posterior_samples_theta.shape == (2, 0)
    assert np.array_equal(all_zero_vi.f_samples, np.zeros((2, layout.num_od_total)))
    all_zero_problem = build_od_theta_ml_problem(all_zero_request)
    assert all_zero_problem.dim == 0
    all_zero_ml = run_ml(
        dim=0,
        data=all_zero_problem.data,
        loglik=all_zero_problem.loglik,
        logprior=all_zero_problem.logprior,
        theta0=all_zero_problem.theta0,
        config=MLConfig(maxiter=1, prior_weight=0.0, compute_hessian=False),
    )
    all_zero_f, all_zero_theta = all_zero_problem.decode(all_zero_ml.theta_hat)
    assert np.array_equal(all_zero_f, np.zeros(layout.num_od_total))
    assert all_zero_theta == 5.0
    jax.clear_caches()
