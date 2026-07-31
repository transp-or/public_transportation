"""Cross-method comparison on one controlled packaged transit example.

The test deliberately separates statistical estimators from solver backends.
ML, MAP, and VI share the same negative-binomial model but target different
posterior summaries.  The dense, TRF/LSMR, and block-coordinate paths solve
the same fixed-routing linear least-squares problem and therefore must agree.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from public_transportation.estimation.bayesian import VIConfig
from public_transportation.estimation.maximum_likelihood import MLConfig, run_ml
from public_transportation.inference.block_coordinate import (
    BlockCoordinateFingerprints,
    BlockCoordinateMAPConfig,
    ODBlock,
    run_block_coordinate_map,
    validate_block_partition,
)
from public_transportation.inference.fixed_routing_linear_dense_solver import (
    solve_dense_reference,
)
from public_transportation.inference.fixed_routing_linear_trf_solver import (
    TRFLSMRConfig,
    solve_trf_lsmr,
)
from public_transportation.inference.fixed_routing_linear_regularization import (
    ridge_to_prior,
)
from public_transportation.inference.maximum_likelihood_pipeline import (
    build_od_theta_ml_problem,
)
from public_transportation.inference.pipeline import (
    ODThetaEstimationRequest,
    estimate_od_theta_vi,
)

from test_fixed_routing_linear_examples import _prepare_example


def _block_partition(example):
    """Make two deterministic blocks with exact measurement supports."""
    matrix = np.asarray(example.operator.matrix)
    # The prepared representation is the dense measurement-by-free-OD matrix.
    assert matrix.shape == (
        example.problem.num_measurements,
        example.problem.num_free_od,
    )
    midpoint = max(1, example.problem.num_free_od // 2)
    column_groups = (range(0, midpoint), range(midpoint, example.problem.num_free_od))
    blocks = []
    free_to_active = tuple(example.compact_layout.free_compact_indices)
    for block_number, column_group in enumerate(column_groups):
        columns = tuple(column_group)
        if not columns:
            continue
        support = tuple(np.flatnonzero(np.any(matrix[:, columns] != 0.0, axis=1)))
        blocks.append(
            ODBlock(
                block_id=f"half-{block_number}",
                free_column_indices=columns,
                active_od_indices=tuple(free_to_active[column] for column in columns),
                destination_group_indices=(block_number,),
                time_bin_ids=("all",),
                estimated_nonzeros=int(np.count_nonzero(matrix[:, columns])),
                measurement_support_indices=support,
            )
        )
    return validate_block_partition(blocks, free_to_active_indices=free_to_active)


def _fingerprints(example, partition, config):
    return BlockCoordinateFingerprints(
        scenario=example.name,
        assignment_inputs=str(example.fingerprint),
        od_layout=example.od_layout.fingerprint,
        fixed_demand=example.od_layout.fingerprint,
        measurements=str(example.fingerprint),
        prior="ridge-to-prior-0.1",
        routing=f"fixed-theta-{example.theta:g}",
        partition=partition.fingerprint,
        solver_semantics=config.fingerprint,
    )


def test_all_estimators_share_inputs_and_linear_solvers_agree(tmp_path):
    """Exercise every estimator family without asserting false equivalence."""
    example = _prepare_example(
        example_name="simple_example_01",
        theta=5.0,
        temporary_directory=tmp_path,
    )
    linear_problem = replace(
        example.problem,
        measurement_operator=np.asarray(example.operator.matrix, dtype=np.float64),
        fixed_measurement_offset=np.asarray(
            example.problem.fixed_measurement_offset, dtype=np.float64
        ),
        observations=np.asarray(example.problem.observations, dtype=np.float64),
        observation_weights=np.asarray(
            example.problem.observation_weights, dtype=np.float64
        ),
        prior_demand=np.asarray(example.problem.prior_demand, dtype=np.float64),
        lower_bounds=np.asarray(example.problem.lower_bounds, dtype=np.float64),
        upper_bounds=np.asarray(example.problem.upper_bounds, dtype=np.float64),
        regularization_selection="configured",
        regularization_blocks=(
            ridge_to_prior(example.problem.prior_demand, strength=0.1),
        ),
    )
    full_baseline = example.od_layout.reconstruct_numpy(
        np.zeros(example.od_layout.num_free)
    )
    request = ODThetaEstimationRequest(
        fingerprint=str(example.fingerprint),
        f0=jnp.asarray(full_baseline, dtype=jnp.float32),
        y_obs=jnp.asarray(example.observations, dtype=jnp.float32),
        mapping_spec=example.mapping_spec,
        baseline_theta=example.theta,
        od_layout=example.od_layout,
        estimate_theta=False,
        fixed_theta=example.theta,
        assignment_artifacts=example.artifacts,
        fixed_measurement_operator="dense",
        fixed_measurement_operator_chunk_size=8,
        vi=VIConfig(
            guide="auto_normal",
            num_steps=25,
            learning_rate=1.0e-2,
            seed=1729,
            num_posterior_draws=20,
            log_every=25,
        ),
    )

    nonlinear_problem = build_od_theta_ml_problem(request)
    ml = run_ml(
        dim=nonlinear_problem.dim,
        data=nonlinear_problem.data,
        loglik=nonlinear_problem.loglik,
        logprior=nonlinear_problem.logprior,
        theta0=nonlinear_problem.theta0,
        config=MLConfig(maxiter=300, prior_weight=0.0, compute_hessian=False),
    )
    map_result = run_ml(
        dim=nonlinear_problem.dim,
        data=nonlinear_problem.data,
        loglik=nonlinear_problem.loglik,
        logprior=nonlinear_problem.logprior,
        theta0=nonlinear_problem.theta0,
        config=MLConfig(maxiter=300, prior_weight=1.0, compute_hessian=False),
    )
    vi = estimate_od_theta_vi(request)

    assert nonlinear_problem.dim == example.od_layout.num_free
    assert np.all(np.isfinite(ml.theta_hat))
    assert np.all(np.isfinite(map_result.theta_hat))
    assert np.all(np.isfinite(vi.f_mean))
    assert vi.vi.posterior_samples_theta.shape == (
        request.vi.num_posterior_draws,
        nonlinear_problem.dim,
    )
    # These are different estimands.  Check the optimization implications, not
    # equality between ML, the posterior mode, and a VI posterior mean.
    assert ml.loglikelihood >= map_result.loglikelihood - 2.0e-4
    map_objective_at_ml = -(
        ml.loglikelihood + float(nonlinear_problem.logprior(jnp.asarray(ml.theta_hat)))
    )
    assert map_result.objective_value <= map_objective_at_ml + 2.0e-4

    dense = solve_dense_reference(linear_problem, tolerance=1.0e-12)
    iterative = solve_trf_lsmr(
        linear_problem,
        config=TRFLSMRConfig(
            tolerance=1.0e-12,
            lsmr_tolerance=1.0e-12,
            max_iterations=2_000,
        ),
    )
    partition = _block_partition(example)
    sequential_config = BlockCoordinateMAPConfig(
        maximum_sweeps=100,
        block_solver_max_iterations=100,
        block_solver_tolerance=1.0e-8,
        global_projected_gradient_tolerance=1.0e-6,
        relative_sweep_objective_tolerance=1.0e-10,
        checkpoint_directory=tmp_path / "block-sequential",
    )
    sequential = run_block_coordinate_map(
        problem=linear_problem,
        partition=partition,
        config=sequential_config,
        fingerprints=_fingerprints(
            example, partition, sequential_config
        ),
    )
    interleaved_config = replace(
        sequential_config,
        block_order="interleaved",
        solver_workers=2,
        checkpoint_directory=tmp_path / "block-interleaved",
    )
    interleaved = run_block_coordinate_map(
        problem=linear_problem,
        partition=partition,
        config=interleaved_config,
        fingerprints=_fingerprints(example, partition, interleaved_config),
    )

    assert dense.success
    assert iterative.success
    assert sequential.status == "converged", sequential.message
    assert interleaved.status == "converged", interleaved.message
    np.testing.assert_allclose(iterative.demand, dense.demand, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(
        sequential.latest_free_flow, dense.demand, rtol=3e-4, atol=3e-4
    )
    np.testing.assert_allclose(
        interleaved.latest_free_flow, dense.demand, rtol=3e-4, atol=3e-4
    )
