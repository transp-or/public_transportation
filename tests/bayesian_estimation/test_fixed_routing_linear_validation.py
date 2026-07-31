from __future__ import annotations

import numpy as np

from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.fixed_routing_linear_validation import (
    validate_fixed_routing_forward_equivalence,
    validate_noise_free_linear_recovery,
)


def _problem(matrix: np.ndarray, offset: np.ndarray) -> FixedRoutingLinearProblem:
    return FixedRoutingLinearProblem(
        measurement_operator=matrix,
        fixed_measurement_offset=offset,
        observations=np.ones(matrix.shape[0]),
        observation_weights=np.ones(matrix.shape[0]),
        prior_demand=np.ones(matrix.shape[1]),
        lower_bounds=np.zeros(matrix.shape[1]),
        upper_bounds=np.full(matrix.shape[1], np.inf),
        provenance=FixedRoutingLinearProvenance(
            od_layout_fingerprint="od",
            assignment_fingerprint="assignment",
            mapping_fingerprint="mapping",
            routing_parameter=1.0,
        ),
        regularization_selection="none",
    )


def test_forward_equivalence_checks_multiple_vectors():
    matrix = np.array([[1.0, 0.5], [0.0, 2.0], [1.0, 1.0]])
    offset = np.array([2.0, 0.0, 1.0])
    problem = _problem(matrix, offset)

    validation = validate_fixed_routing_forward_equivalence(
        problem,
        {"zero": np.zeros(2), "other": np.array([3.0, 4.0])},
        lambda demand: matrix @ demand + offset,
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-12,
    )

    assert validation.passed
    assert validation.worst_max_abs_difference == 0.0
    assert [case.name for case in validation.cases] == ["zero", "other"]


def test_forward_equivalence_reports_a_mismatch():
    matrix = np.eye(2)
    problem = _problem(matrix, np.zeros(2))

    validation = validate_fixed_routing_forward_equivalence(
        problem,
        {"truth": np.array([2.0, 3.0])},
        lambda demand: matrix @ demand + np.array([0.0, 0.01]),
        absolute_tolerance=1.0e-6,
        relative_tolerance=1.0e-6,
    )

    assert not validation.passed
    np.testing.assert_allclose(validation.worst_max_abs_difference, 0.01)


def test_noise_free_full_rank_problem_recovers_truth():
    problem = _problem(
        np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        np.array([4.0, 1.0, 0.0]),
    )
    truth = np.array([5.0, 7.0])

    validation = validate_noise_free_linear_recovery(problem, truth)

    assert validation.result.success
    assert validation.measurement_rank == 2
    assert validation.measurement_nullity == 0
    assert validation.measurement_residual_inf_norm <= 1.0e-11
    assert validation.identifiable_error_norm <= 1.0e-11
    assert validation.null_space_error_norm <= 1.0e-11


def test_noise_free_rank_deficient_error_is_confined_to_null_space():
    problem = _problem(
        np.array([[1.0, 1.0], [2.0, 2.0]]),
        np.zeros(2),
    )
    truth = np.array([2.0, 8.0])

    validation = validate_noise_free_linear_recovery(problem, truth)

    assert validation.measurement_rank == 1
    assert validation.measurement_nullity == 1
    assert validation.measurement_residual_inf_norm <= 1.0e-10
    assert validation.identifiable_error_norm <= 1.0e-10
    assert validation.null_space_error_norm > 1.0
