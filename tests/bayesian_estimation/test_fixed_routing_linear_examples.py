"""Dense linear-operator equivalence on both packaged small examples."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario, read_fixed_demand_csv
from public_transportation.inference.assignment_adapter import (
    assign_link_flow_fixed_routing,
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
    build_compact_od_assignment_layout,
)
from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    build_fixed_routing_linear_problem_from_dense_operator,
    build_fixed_routing_linear_problem_from_operator,
)
from public_transportation.inference.fixed_routing_linear_quality import (
    analyze_linear_estimate_quality,
)
from public_transportation.inference.fixed_routing_linear_recommendation import (
    recommend_linear_regularization,
)
from public_transportation.inference.fixed_routing_linear_objective import (
    evaluate_linear_data_fit,
)
from public_transportation.inference.fixed_routing_linear_dense_solver import (
    solve_dense_reference,
)
from public_transportation.inference.fixed_routing_linear_regularization import (
    build_augmented_linear_least_squares_system,
    evaluate_linear_least_squares,
    ridge_to_prior,
    scaled_ridge_to_prior,
)
from public_transportation.inference.fixed_routing_linear_solver import (
    FixedRoutingLinearSolverConfig,
    benchmark_fixed_routing_linear_solvers,
)
from public_transportation.inference.fixed_routing_linear_scalable_quality import (
    ScalableQualityConfig,
    analyze_linear_estimate_quality_scalable,
)
from public_transportation.inference.fixed_routing_linear_transform import (
    build_solver_variable_least_squares_system,
)
from public_transportation.inference.fixed_routing_linear_trf_solver import (
    TRFLSMRConfig,
    solve_trf_lsmr,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    FixedRoutingMeasurementOperator,
    predict_measurements_fixed_operator,
    prepare_fixed_routing_measurement_operator,
)
from public_transportation.inference.fixed_routing_matrix_free_operator import (
    MatrixFreePreparationDeadlineError,
    MatrixFreeFixedRoutingMeasurementOperator,
)
from public_transportation.inference.linear_operator import SparseLinearOperator
from public_transportation.inference.od_parameter_layout import (
    ODParameterLayout,
    build_od_parameter_layout,
)
from public_transportation.measurement import (
    build_mapping_spec_strict,
    read_measurements_csv,
)
from public_transportation.measurement.likelihood_jax import (
    predict_measurements_from_link_flow,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "docs/source/examples"
NETWORK_FILES = (
    "metadata.json",
    "stops.csv",
    "lines.csv",
    "trips.csv",
    "stop_times.csv",
    "time_bins.csv",
)


@dataclass(frozen=True)
class PreparedExample:
    name: str
    problem: FixedRoutingLinearProblem
    operator: FixedRoutingMeasurementOperator
    inputs: object
    routing: object
    mapping_spec: object
    compact_layout: CompactODAssignmentLayout
    od_layout: ODParameterLayout
    artifacts: object
    observations: np.ndarray
    fingerprint: str
    theta: float


def _prepare_example(
    *, example_name: str, theta: float, temporary_directory: Path
) -> PreparedExample:
    example = EXAMPLES / example_name
    scenario_directory = temporary_directory / example_name
    scenario_directory.mkdir()
    for name in NETWORK_FILES:
        shutil.copy2(example / "data" / name, scenario_directory / name)
    shutil.copy2(
        example / "pre_processing/results/demand.csv",
        scenario_directory / "demand.csv",
    )

    scenario = Scenario.from_folder(scenario_directory, strict=True)
    fixed_demand = read_fixed_demand_csv(
        example / "data/fixed_demand.csv", scenario=scenario
    )
    od_layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed_demand)
    compact_layout = build_compact_od_assignment_layout(parameter_layout=od_layout)
    artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
    id_manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)
    mapping = build_mapping_spec_strict(
        id_manager=id_manager,
        table=read_measurements_csv(
            example / "pre_processing/results/measurements_boarding_alighting.csv"
        ),
        include_link_lists_for_report=False,
    )
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact_layout)
    routing = prepare_fixed_routing(inputs=inputs, theta=theta)
    operator = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=mapping.spec,
        assignment_fingerprint=str(id_manager.fingerprint),
        compact_layout=compact_layout,
        od_layout_fingerprint=od_layout.fingerprint,
        representation="dense",
        chunk_size=16,
    )
    prior = np.asarray(od_layout.free_baseline_values)
    problem = build_fixed_routing_linear_problem_from_dense_operator(
        operator=operator,
        observations=np.asarray(mapping.y_obs),
        observation_weights=np.ones(mapping.spec.num_measurements),
        prior_demand=prior,
        lower_bounds=np.zeros(od_layout.num_free),
        upper_bounds=np.full(od_layout.num_free, np.inf),
        regularization_selection="unspecified",
        free_od_indices=np.asarray(od_layout.free_od_indices),
    )
    return PreparedExample(
        name=example_name,
        problem=problem,
        operator=operator,
        inputs=inputs,
        routing=routing,
        mapping_spec=mapping.spec,
        compact_layout=compact_layout,
        od_layout=od_layout,
        artifacts=artifacts,
        observations=np.asarray(mapping.y_obs),
        fingerprint=str(id_manager.fingerprint),
        theta=float(theta),
    )


@pytest.fixture(
    scope="module", params=[("simple_example_01", 5.0), ("simple_example_02", 1.0)]
)
def prepared_example(request, tmp_path_factory) -> PreparedExample:
    name, theta = request.param
    return _prepare_example(
        example_name=name,
        theta=theta,
        temporary_directory=tmp_path_factory.mktemp("linear-small-examples"),
    )


def _active_demand(example: PreparedExample, free_demand: np.ndarray) -> jnp.ndarray:
    compact = example.compact_layout
    active = jnp.zeros((compact.num_active,), dtype=jnp.float32)
    active = active.at[jnp.asarray(compact.free_compact_indices)].set(free_demand)
    active = active.at[jnp.asarray(compact.fixed_compact_indices)].set(
        jnp.asarray(compact.fixed_compact_values)
    )
    return active


def _assignment_prediction(
    example: PreparedExample, free_demand: np.ndarray
) -> np.ndarray:
    link_flow = assign_link_flow_fixed_routing(
        inputs=example.inputs,
        routing=example.routing,
        f=_active_demand(example, free_demand),
    )
    return np.asarray(
        predict_measurements_from_link_flow(
            link_flow,
            spec_num_measurements=example.mapping_spec.num_measurements,
            spec_measurement_index=jnp.asarray(example.mapping_spec.measurement_index),
            spec_link_index=jnp.asarray(example.mapping_spec.link_index),
        )
    )


def _demand_cases(example: PreparedExample) -> tuple[np.ndarray, ...]:
    num_free = example.problem.num_free_od
    single = np.zeros(num_free, dtype=np.float32)
    single[num_free // 2] = 3.25
    rng = np.random.default_rng(1729)
    return (
        np.zeros(num_free, dtype=np.float32),
        np.asarray(example.problem.prior_demand, dtype=np.float32),
        single,
        rng.uniform(0.0, 20.0, size=num_free).astype(np.float32),
        rng.uniform(0.0, 100.0, size=num_free).astype(np.float32),
    )


def test_dense_linear_operator_matches_fixed_assignment(prepared_example):
    example = prepared_example
    for demand in _demand_cases(example):
        linear_prediction = (
            example.problem.measurement_operator.matvec(demand)
            + example.problem.fixed_measurement_offset
        )
        packaged_prediction = np.asarray(
            predict_measurements_fixed_operator(
                operator=example.operator,
                free_demand=jnp.asarray(demand),
                rho=jnp.asarray(1.0),
            )
        )
        assignment_prediction = _assignment_prediction(example, demand)
        np.testing.assert_allclose(
            linear_prediction, assignment_prediction, rtol=4e-5, atol=4e-5
        )
        np.testing.assert_allclose(
            packaged_prediction, assignment_prediction, rtol=4e-5, atol=4e-5
        )


def test_fixed_offset_is_zero_free_demand_prediction(prepared_example):
    example = prepared_example
    zero = np.zeros(example.problem.num_free_od, dtype=np.float32)
    np.testing.assert_allclose(
        example.problem.fixed_measurement_offset,
        _assignment_prediction(example, zero),
        rtol=4e-5,
        atol=4e-5,
    )


def test_problem_preserves_operator_dimensions_and_provenance(prepared_example):
    example = prepared_example
    assert example.problem.num_measurements == example.operator.num_measurements
    assert example.problem.num_free_od == example.od_layout.num_free
    assert (
        example.problem.provenance.od_layout_fingerprint
        == example.od_layout.fingerprint
    )
    assert (
        example.problem.provenance.assignment_fingerprint
        == example.operator.assignment_fingerprint
    )
    assert (
        example.problem.provenance.mapping_fingerprint
        == example.operator.mapping_fingerprint
    )
    assert example.problem.provenance.routing_parameter == pytest.approx(
        example.operator.theta
    )


def test_linear_problem_boundary_rejects_unsupported_or_inconsistent_operator(
    prepared_example,
):
    example = prepared_example
    kwargs = {
        "observations": example.problem.observations,
        "observation_weights": example.problem.observation_weights,
        "prior_demand": example.problem.prior_demand,
        "lower_bounds": example.problem.lower_bounds,
        "upper_bounds": example.problem.upper_bounds,
    }
    with pytest.raises(ValueError, match="requires a dense operator"):
        build_fixed_routing_linear_problem_from_dense_operator(
            operator=replace(example.operator, representation="bcoo"), **kwargs
        )
    with pytest.raises(ValueError, match="OD layout fingerprint"):
        build_fixed_routing_linear_problem_from_dense_operator(
            operator=replace(example.operator, od_layout_fingerprint=None), **kwargs
        )
    with pytest.raises(ValueError, match="matrix shape disagrees"):
        build_fixed_routing_linear_problem_from_dense_operator(
            operator=replace(
                example.operator,
                matrix=jnp.zeros(
                    (example.operator.num_measurements, 1), dtype=jnp.float32
                ),
            ),
            **kwargs,
        )


def test_native_sparse_problem_matches_dense_forward_and_transpose_products(
    prepared_example,
):
    example = prepared_example
    native = prepare_fixed_routing_measurement_operator(
        inputs=example.inputs,
        routing=example.routing,
        spec=example.mapping_spec,
        assignment_fingerprint=example.operator.assignment_fingerprint,
        compact_layout=example.compact_layout,
        od_layout_fingerprint=example.od_layout.fingerprint,
        representation="bcoo",
        chunk_size=16,
    )
    sparse_problem = build_fixed_routing_linear_problem_from_operator(
        operator=native,
        observations=example.problem.observations,
        observation_weights=example.problem.observation_weights,
        prior_demand=example.problem.prior_demand,
        lower_bounds=example.problem.lower_bounds,
        upper_bounds=example.problem.upper_bounds,
        variable_scales=example.problem.variable_scales,
        free_od_indices=example.problem.free_od_indices,
    )
    assert isinstance(sparse_problem.measurement_operator, SparseLinearOperator)
    assert (
        sparse_problem.measurement_operator.nonzero_entries
        == native.metrics.nonzero_entries
    )
    np.testing.assert_allclose(
        sparse_problem.fixed_measurement_offset,
        example.problem.fixed_measurement_offset,
        rtol=2e-6,
        atol=2e-6,
    )

    for demand in _demand_cases(example):
        np.testing.assert_allclose(
            sparse_problem.measurement_operator.matvec(demand),
            example.problem.measurement_operator.matvec(demand),
            rtol=2e-6,
            atol=2e-6,
        )
    rng = np.random.default_rng(271828)
    measurement_vector = rng.normal(size=example.problem.num_measurements)
    np.testing.assert_allclose(
        sparse_problem.measurement_operator.rmatvec(measurement_vector),
        example.problem.measurement_operator.rmatvec(measurement_vector),
        rtol=2e-6,
        atol=2e-6,
    )


def test_matrix_free_products_match_dense_and_satisfy_adjoint_identity(
    prepared_example,
):
    example = prepared_example
    operator = MatrixFreeFixedRoutingMeasurementOperator(
        inputs=example.inputs,
        routing=example.routing,
        spec=example.mapping_spec,
        compact_layout=example.compact_layout,
    )
    assert operator.shape == example.problem.measurement_operator.shape
    assert not hasattr(operator, "matrix")
    assert operator.diagnostics.forward_compilation_count == 0
    assert operator.diagnostics.transpose_compilation_count == 0

    assert not operator.diagnostics.zero_offset_fast_path
    np.testing.assert_allclose(
        operator.fixed_measurement_offset,
        example.problem.fixed_measurement_offset,
        rtol=4e-5,
        atol=4e-5,
    )
    rng = np.random.default_rng(161803)
    first = rng.uniform(0.0, 20.0, size=operator.shape[1])
    second = rng.uniform(0.0, 20.0, size=operator.shape[1])
    cotangent = rng.normal(size=operator.shape[0])
    np.testing.assert_allclose(
        operator.matvec(first),
        example.problem.measurement_operator.matvec(first),
        rtol=4e-5,
        atol=4e-5,
    )
    assert operator.diagnostics.forward_compilation_count == 1
    assert operator.diagnostics.transpose_compilation_count == 0
    np.testing.assert_allclose(
        operator.rmatvec(cotangent),
        example.problem.measurement_operator.rmatvec(cotangent),
        rtol=4e-5,
        atol=4e-5,
    )
    assert operator.diagnostics.forward_compilation_count == 1
    assert operator.diagnostics.transpose_compilation_count == 1
    np.testing.assert_allclose(
        operator.matvec(1.7 * first - 0.4 * second),
        1.7 * operator.matvec(first) - 0.4 * operator.matvec(second),
        rtol=4e-5,
        atol=4e-5,
    )
    assert np.vdot(operator.matvec(first), cotangent) == pytest.approx(
        np.vdot(first, operator.rmatvec(cotangent)), rel=4e-5, abs=4e-5
    )
    assert operator.diagnostics.forward_compilation_count == 1
    assert operator.diagnostics.transpose_compilation_count == 1


def test_matrix_free_empty_fixed_layout_uses_exact_numpy_zero_fast_path(
    prepared_example, monkeypatch
):
    example = prepared_example
    compact = example.compact_layout
    fixed_by_compact = dict(
        zip(
            compact.fixed_compact_indices,
            compact.fixed_compact_values,
            strict=True,
        )
    )
    all_free = tuple(range(compact.num_active))
    full_by_compact = tuple(compact.active_full_indices)
    old_baseline = dict(
        zip(compact.free_compact_indices, compact.free_baseline_values, strict=True)
    )
    empty_fixed = CompactODAssignmentLayout(
        num_od_total=compact.num_od_total,
        active_full_indices=compact.active_full_indices,
        removed_zero_full_indices=compact.removed_zero_full_indices,
        full_to_compact=compact.full_to_compact,
        free_full_indices=full_by_compact,
        free_compact_indices=all_free,
        free_baseline_values=tuple(
            old_baseline[index]
            if index in old_baseline
            else fixed_by_compact[index]
            for index in all_free
        ),
        fixed_compact_indices=(),
        fixed_compact_values=(),
    )
    jit_calls = 0
    real_jit = jax.jit

    def counting_jit(*args, **kwargs):
        nonlocal jit_calls
        jit_calls += 1
        return real_jit(*args, **kwargs)

    monkeypatch.setattr(jax, "jit", counting_jit)
    operator = MatrixFreeFixedRoutingMeasurementOperator(
        inputs=example.inputs,
        routing=example.routing,
        spec=example.mapping_spec,
        compact_layout=empty_fixed,
    )
    assert empty_fixed.fixed_compact_indices == ()
    assert operator.fixed_measurement_offset.shape == (
        example.mapping_spec.num_measurements,
    )
    assert operator.fixed_measurement_offset.dtype == operator.dtype
    assert not operator.fixed_measurement_offset.flags.writeable
    np.testing.assert_array_equal(operator.fixed_measurement_offset, 0)
    assert jit_calls == 0
    assert operator.diagnostics.zero_offset_fast_path
    assert operator.diagnostics.fixed_positive_cells == 0
    assert operator.diagnostics.forward_compilation_count == 0
    assert operator.diagnostics.transpose_compilation_count == 0

    class Clock:
        value = 0.0

        def __call__(self):
            return self.value

    clock = Clock()
    deadline_operator = MatrixFreeFixedRoutingMeasurementOperator(
        inputs=example.inputs,
        routing=example.routing,
        spec=example.mapping_spec,
        compact_layout=empty_fixed,
        preparation_deadline=10.0,
        clock=clock,
    )
    clock.value = 11.0
    with pytest.raises(MatrixFreePreparationDeadlineError):
        deadline_operator.matvec(np.ones(deadline_operator.shape[1]))
    assert deadline_operator.diagnostics.deadline_exceeded
    assert deadline_operator.diagnostics.forward_compilation_count == 0
    assert jit_calls == 0


def test_matrix_free_expired_deadline_stops_before_numerical_preparation(
    prepared_example,
):
    class Clock:
        value = 10.0

        def __call__(self):
            return self.value

    with pytest.raises(MatrixFreePreparationDeadlineError) as error:
        MatrixFreeFixedRoutingMeasurementOperator(
            inputs=prepared_example.inputs,
            routing=prepared_example.routing,
            spec=prepared_example.mapping_spec,
            compact_layout=prepared_example.compact_layout,
            preparation_deadline=5.0,
            clock=Clock(),
        )
    assert error.value.diagnostics.deadline_exceeded
    assert error.value.diagnostics.forward_compilation_count == 0
    assert error.value.diagnostics.transpose_compilation_count == 0


def test_trf_lsmr_matrix_free_solution_matches_sparse_solution(prepared_example):
    example = prepared_example
    matrix_free = MatrixFreeFixedRoutingMeasurementOperator(
        inputs=example.inputs,
        routing=example.routing,
        spec=example.mapping_spec,
        compact_layout=example.compact_layout,
    )
    prior = example.problem.prior_demand
    block = ridge_to_prior(prior, strength=1.0)
    common = replace(
        example.problem,
        regularization_selection="configured",
        regularization_blocks=(block,),
        variable_scales=np.maximum(prior, 1.0),
    )
    matrix_free_problem = replace(
        common,
        measurement_operator=matrix_free,
        fixed_measurement_offset=matrix_free.fixed_measurement_offset,
    )
    config = TRFLSMRConfig(
        tolerance=1e-9,
        lsmr_tolerance=1e-11,
        active_tolerance=1e-6,
    )
    sparse_result = solve_trf_lsmr(common, config=config)
    matrix_free_result = solve_trf_lsmr(matrix_free_problem, config=config)

    assert matrix_free_result.success
    assert matrix_free_result.matvec_count > 0
    assert matrix_free_result.rmatvec_count > 0
    assert matrix_free_result.evaluation.objective == pytest.approx(
        sparse_result.evaluation.objective, rel=4e-5, abs=4e-5
    )
    np.testing.assert_allclose(
        matrix_free_result.evaluation.data_fit.prediction,
        sparse_result.evaluation.data_fit.prediction,
        rtol=4e-4,
        atol=4e-4,
    )
    np.testing.assert_allclose(
        matrix_free_result.demand,
        sparse_result.demand,
        rtol=6e-4,
        atol=6e-4,
    )


def test_scalable_quality_uses_matrix_free_products_only(prepared_example, monkeypatch):
    example = prepared_example
    matrix_free = MatrixFreeFixedRoutingMeasurementOperator(
        inputs=example.inputs,
        routing=example.routing,
        spec=example.mapping_spec,
        compact_layout=example.compact_layout,
    )
    prior = example.problem.prior_demand
    problem = replace(
        example.problem,
        measurement_operator=matrix_free,
        fixed_measurement_offset=matrix_free.fixed_measurement_offset,
        regularization_selection="configured",
        regularization_blocks=(ridge_to_prior(prior, strength=1.0),),
        variable_scales=np.maximum(prior, 1.0),
    )
    solved = solve_trf_lsmr(
        problem,
        config=TRFLSMRConfig(
            tolerance=1e-9,
            lsmr_tolerance=1e-11,
            active_tolerance=1e-6,
        ),
    )

    def reject_materialization(*args, **kwargs):
        raise AssertionError("scalable diagnostics must not materialize the operator")

    monkeypatch.setattr(
        "public_transportation.inference.linear_operator.materialize_linear_operator",
        reject_materialization,
    )
    quality = analyze_linear_estimate_quality_scalable(
        problem,
        solved.demand,
        config=ScalableQualityConfig(
            smallest_singular_values=2,
            resolution_samples=8,
            random_seed=123,
            active_tolerance=1e-6,
        ),
    )

    assert quality.spectral_converged
    assert quality.resolution_converged_samples == 8
    assert quality.resolution_failed_samples == 0
    assert np.isfinite(quality.effective_data_degrees_of_freedom_estimate)
    assert np.isfinite(quality.effective_data_degrees_of_freedom_standard_error)
    assert len(quality.classifications) == problem.num_free_od


def test_csr_operator_matches_dense_products_and_preserves_problem_contract(
    prepared_example,
):
    example = prepared_example
    dense = example.problem.measurement_operator
    sparse_operator = SparseLinearOperator(dense.matrix)
    rng = np.random.default_rng(314159)

    for demand in _demand_cases(example):
        np.testing.assert_allclose(
            sparse_operator.matvec(demand),
            dense.matvec(demand),
            rtol=2e-6,
            atol=2e-6,
        )
    measurement_vector = rng.normal(size=example.problem.num_measurements)
    np.testing.assert_allclose(
        sparse_operator.rmatvec(measurement_vector),
        dense.rmatvec(measurement_vector),
        rtol=2e-6,
        atol=2e-6,
    )

    sparse_problem = replace(example.problem, measurement_operator=sparse_operator)
    assert sparse_problem.provenance == example.problem.provenance
    np.testing.assert_array_equal(
        sparse_problem.fixed_measurement_offset,
        example.problem.fixed_measurement_offset,
    )
    np.testing.assert_array_equal(
        sparse_problem.prior_demand, example.problem.prior_demand
    )
    assert sparse_problem.num_measurements == example.problem.num_measurements
    assert sparse_problem.num_free_od == example.problem.num_free_od
    for demand in _demand_cases(example):
        dense_evaluation = evaluate_linear_data_fit(example.problem, demand)
        sparse_evaluation = evaluate_linear_data_fit(sparse_problem, demand)
        np.testing.assert_allclose(
            dense_evaluation.prediction,
            _assignment_prediction(example, demand),
            rtol=4e-5,
            atol=4e-5,
        )
        np.testing.assert_allclose(
            sparse_evaluation.raw_residual,
            dense_evaluation.raw_residual,
            rtol=2e-6,
            atol=2e-6,
        )
        assert sparse_evaluation.objective == pytest.approx(
            dense_evaluation.objective, rel=2e-6, abs=2e-6
        )
        np.testing.assert_allclose(
            sparse_evaluation.gradient,
            dense_evaluation.gradient,
            rtol=2e-6,
            atol=2e-6,
        )


def test_small_example_regularization_matches_augmented_system(prepared_example):
    example = prepared_example
    prior = example.problem.prior_demand
    scales = np.maximum(prior, 1.0)
    configured = replace(
        example.problem,
        regularization_selection="configured",
        regularization_blocks=(
            ridge_to_prior(prior, strength=0.5),
            scaled_ridge_to_prior(prior, scales, strength=2.0),
        ),
        variable_scales=scales,
    )
    system = build_augmented_linear_least_squares_system(configured)
    solver_system = build_solver_variable_least_squares_system(configured)

    for demand in _demand_cases(example):
        evaluation = evaluate_linear_least_squares(configured, demand)
        augmented_residual = system.operator.matvec(demand) - system.target
        np.testing.assert_allclose(
            evaluation.augmented_residual,
            augmented_residual,
            rtol=3e-6,
            atol=3e-6,
        )
        assert evaluation.objective == pytest.approx(
            0.5 * augmented_residual @ augmented_residual,
            rel=3e-6,
            abs=3e-6,
        )
        np.testing.assert_allclose(
            evaluation.gradient,
            system.operator.rmatvec(augmented_residual),
            rtol=3e-6,
            atol=3e-6,
        )
        solver_variable = solver_system.transform.solver_variable_from_demand(demand)
        np.testing.assert_allclose(
            solver_system.transform.demand_from_solver_variable(solver_variable),
            demand,
            rtol=2e-7,
            atol=2e-7,
        )
        solver_residual = (
            solver_system.operator.matvec(solver_variable) - solver_system.target
        )
        np.testing.assert_allclose(
            solver_residual,
            evaluation.augmented_residual,
            rtol=3e-6,
            atol=3e-6,
        )
        np.testing.assert_allclose(
            solver_system.operator.rmatvec(solver_residual),
            solver_system.transform.solver_gradient_from_physical(evaluation.gradient),
            rtol=1e-5,
            atol=2e-5,
        )
        assert np.all(solver_variable >= solver_system.lower_bounds - 1e-7)
        assert np.all(solver_variable <= solver_system.upper_bounds + 1e-7)


def test_small_example_dense_reference_solver(prepared_example):
    example = prepared_example
    prior = example.problem.prior_demand
    configured = replace(
        example.problem,
        regularization_selection="configured",
        regularization_blocks=(ridge_to_prior(prior, strength=1.0),),
        variable_scales=np.maximum(prior, 1.0),
    )
    prior_evaluation = evaluate_linear_least_squares(configured, prior)
    result = solve_dense_reference(configured, tolerance=1e-9)

    assert result.success
    assert result.method == "bvls"
    assert result.evaluation.objective <= prior_evaluation.objective
    assert result.kkt.feasibility_inf_norm <= 1e-9
    assert result.kkt.projected_gradient_inf_norm <= 2e-4
    assert np.all(result.demand >= configured.lower_bounds - 1e-9)
    assert np.all(result.demand <= configured.upper_bounds + 1e-9)
    np.testing.assert_allclose(
        configured.prior_demand + configured.variable_scales * result.solver_variable,
        result.demand,
        rtol=1e-12,
        atol=1e-12,
    )


def test_small_example_trf_lsmr_matches_dense_reference(prepared_example):
    example = prepared_example
    prior = example.problem.prior_demand
    configured = replace(
        example.problem,
        regularization_selection="configured",
        regularization_blocks=(ridge_to_prior(prior, strength=1.0),),
        variable_scales=np.maximum(prior, 1.0),
    )
    reference = solve_dense_reference(configured, tolerance=1e-10)
    iterative = solve_trf_lsmr(
        configured,
        config=TRFLSMRConfig(
            tolerance=1e-9,
            lsmr_tolerance=1e-11,
            active_tolerance=1e-6,
        ),
    )

    assert iterative.success
    assert iterative.matvec_count > 0
    assert iterative.rmatvec_count > 0
    assert iterative.evaluation.objective == pytest.approx(
        reference.evaluation.objective, rel=2e-5, abs=2e-5
    )
    np.testing.assert_allclose(
        iterative.evaluation.data_fit.prediction,
        reference.evaluation.data_fit.prediction,
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        iterative.demand,
        reference.demand,
        rtol=3e-4,
        atol=3e-4,
    )
    assert iterative.kkt.feasibility_inf_norm <= 1e-8
    assert iterative.kkt.projected_gradient_inf_norm <= 5e-4

    quality = analyze_linear_estimate_quality(
        configured,
        iterative.demand,
        active_tolerance=1e-6,
    )
    assert quality.combined_nullity == 0
    assert quality.measurement_rank <= quality.free_indices.size
    assert quality.measurement_nullity == (
        quality.free_indices.size - quality.measurement_rank
    )
    assert quality.resolution_closure_inf_norm <= 1e-9
    assert 0.0 <= quality.effective_data_degrees_of_freedom <= quality.free_indices.size
    assert len(quality.classifications) == configured.num_free_od
    assert np.all(np.isfinite(quality.data_resolution_score[quality.free_indices]))
    assert np.all(
        np.isfinite(quality.regularization_reliance_score[quality.free_indices])
    )
    assert np.all(quality.data_mode_fractions >= 0.0)
    assert np.all(quality.data_mode_fractions <= 1.0)


def test_small_example_receives_nonbinding_regularization_choices(prepared_example):
    example = prepared_example
    assert example.problem.regularization_selection == "unspecified"
    recommendation = recommend_linear_regularization(example.problem)

    assert recommendation.status in {
        "none_is_reasonable",
        "regularization_recommended",
        "regularization_required_for_uniqueness",
        "scaling_recommended",
    }
    assert tuple(option.name for option in recommendation.options) == (
        "none",
        "ridge_to_prior",
        "scaled_ridge_to_prior",
    )
    assert recommendation.automatic_selection_applied is False
    assert example.problem.regularization_selection == "unspecified"
    assert len(recommendation.quality.classifications) == example.problem.num_free_od


def test_solver_benchmark_contract_on_both_packaged_examples(prepared_example):
    configured = replace(
        prepared_example.problem,
        regularization_selection="none",
        regularization_blocks=(),
    )
    records = benchmark_fixed_routing_linear_solvers(
        configured,
        configs=(
            FixedRoutingLinearSolverConfig(backend="dense_reference"),
            FixedRoutingLinearSolverConfig(
                backend="trf_lsmr",
                trf_lsmr=TRFLSMRConfig(
                    tolerance=1e-9,
                    lsmr_tolerance=1e-11,
                    active_tolerance=1e-6,
                ),
            ),
        ),
    )

    assert all(record.success for record in records)
    assert records[0].matvec_count is None
    assert records[1].matvec_count > 0
    assert records[1].rmatvec_count > 0
    assert records[1].objective == pytest.approx(
        records[0].objective, rel=3e-5, abs=3e-5
    )
    assert records[1].feasibility_inf_norm <= 1e-8
    assert records[1].projected_gradient_inf_norm <= 2e-3
