from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
    LinearRegularizationBlock,
)


def provenance() -> FixedRoutingLinearProvenance:
    return FixedRoutingLinearProvenance(
        od_layout_fingerprint="od-layout",
        assignment_fingerprint="assignment",
        mapping_fingerprint="mapping",
        routing_parameter=1.0,
    )


def valid_problem(**overrides) -> FixedRoutingLinearProblem:
    values = {
        "measurement_operator": np.array([[1.0, 0.0], [0.25, 0.75]]),
        "fixed_measurement_offset": np.array([0.0, 2.0]),
        "observations": np.array([5.0, 7.0]),
        "observation_weights": np.array([1.0, 4.0]),
        "prior_demand": np.array([3.0, 4.0]),
        "lower_bounds": np.array([0.0, 0.0]),
        "upper_bounds": np.array([np.inf, 10.0]),
        "provenance": provenance(),
    }
    values.update(overrides)
    return FixedRoutingLinearProblem(**values)


def test_valid_problem_is_normalized_and_immutable():
    matrix = [[1, 0], [0, 1]]
    problem = valid_problem(measurement_operator=matrix)

    assert problem.num_measurements == 2
    assert problem.num_free_od == 2
    assert problem.measurement_operator.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(problem.variable_scales, np.ones(2))
    np.testing.assert_array_equal(problem.free_od_indices, np.arange(2))
    assert not problem.measurement_operator.matrix.flags.writeable
    assert not problem.variable_scales.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        problem.measurement_operator.matrix[0, 0] = 2.0
    with pytest.raises(FrozenInstanceError):
        problem.observations = np.zeros(2)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("measurement_operator", np.ones(2), "two-dimensional"),
        ("measurement_operator", np.empty((0, 2)), "at least one observation"),
        ("measurement_operator", np.empty((2, 0)), "at least one free OD"),
        ("fixed_measurement_offset", np.zeros(3), "shape \\(2,\\)"),
        ("observations", np.zeros((2, 1)), "shape \\(2,\\)"),
        ("observation_weights", np.ones(3), "shape \\(2,\\)"),
        ("prior_demand", np.ones(3), "shape \\(2,\\)"),
        ("lower_bounds", np.zeros(3), "shape \\(2,\\)"),
        ("upper_bounds", np.ones(3), "shape \\(2,\\)"),
        ("variable_scales", np.ones(3), "shape \\(2,\\)"),
        ("free_od_indices", np.arange(3), "shape \\(2,\\)"),
    ],
)
def test_problem_rejects_invalid_shapes(field, value, message):
    with pytest.raises(ValueError, match=message):
        valid_problem(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("measurement_operator", [[1.0, np.nan], [0.0, 1.0]], "must be finite"),
        ("measurement_operator", [[1.0, -0.1], [0.0, 1.0]], "non-negative"),
        ("fixed_measurement_offset", [0.0, np.inf], "finite and non-negative"),
        ("fixed_measurement_offset", [0.0, -1.0], "finite and non-negative"),
        ("observations", [0.0, np.nan], "finite and non-negative"),
        ("observations", [0.0, -1.0], "finite and non-negative"),
        ("observation_weights", [1.0, 0.0], "strictly positive"),
        ("observation_weights", [1.0, np.inf], "strictly positive"),
        ("prior_demand", [1.0, -1.0], "finite and non-negative"),
        ("prior_demand", [1.0, np.nan], "finite and non-negative"),
        ("lower_bounds", [0.0, np.inf], "not NaN or \\+inf"),
        ("upper_bounds", [1.0, -np.inf], "not NaN or -inf"),
        ("variable_scales", [1.0, 0.0], "strictly positive"),
        ("variable_scales", [1.0, np.inf], "strictly positive"),
    ],
)
def test_problem_rejects_invalid_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        valid_problem(**{field: value})


def test_problem_rejects_inconsistent_bounds_and_prior():
    with pytest.raises(ValueError, match="must not exceed"):
        valid_problem(lower_bounds=[0.0, 5.0], upper_bounds=[4.0, 3.0])
    with pytest.raises(ValueError, match="prior_demand must satisfy"):
        valid_problem(lower_bounds=[4.0, 0.0])
    with pytest.raises(ValueError, match="prior_demand must satisfy"):
        valid_problem(upper_bounds=[2.0, 10.0])


def test_problem_validates_explicit_free_od_ordering():
    instance = valid_problem(free_od_indices=[5, 2])
    np.testing.assert_array_equal(instance.free_od_indices, [5, 2])
    assert not instance.free_od_indices.flags.writeable
    with pytest.raises(ValueError, match="must not contain duplicates"):
        valid_problem(free_od_indices=[2, 2])
    with pytest.raises(ValueError, match="must be non-negative"):
        valid_problem(free_od_indices=[2, -1])
    with pytest.raises(TypeError, match="must contain integers"):
        valid_problem(free_od_indices=[1.0, 2.0])


def test_regularization_selection_is_explicit():
    block = LinearRegularizationBlock(
        name="prior",
        operator=np.eye(2),
        target=np.array([3.0, 4.0]),
        strength=2.0,
    )
    configured = valid_problem(
        regularization_selection="configured",
        regularization_blocks=(block,),
    )
    assert configured.regularization_blocks == (block,)

    with pytest.raises(ValueError, match="requires at least one block"):
        valid_problem(regularization_selection="configured")
    with pytest.raises(ValueError, match="blocks require"):
        valid_problem(regularization_blocks=(block,))
    with pytest.raises(ValueError, match="must be 'unspecified'"):
        valid_problem(regularization_selection="automatic")


def test_problem_validates_regularization_dimensions_and_names():
    block = LinearRegularizationBlock("prior", np.eye(2), np.ones(2), 1.0)
    duplicate = LinearRegularizationBlock("prior", np.eye(2), np.ones(2), 2.0)
    wrong_width = LinearRegularizationBlock("other", np.ones((1, 3)), [0.0], 1.0)

    with pytest.raises(ValueError, match="names must be unique"):
        valid_problem(
            regularization_selection="configured",
            regularization_blocks=(block, duplicate),
        )
    with pytest.raises(ValueError, match="must have 2 columns"):
        valid_problem(
            regularization_selection="configured",
            regularization_blocks=(wrong_width,),
        )
    with pytest.raises(TypeError, match="must contain"):
        valid_problem(
            regularization_selection="configured",
            regularization_blocks=(object(),),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "operator": np.eye(2), "target": np.ones(2), "strength": 1.0},
        {"name": "x", "operator": np.ones(2), "target": np.ones(2), "strength": 1.0},
        {"name": "x", "operator": np.eye(2), "target": np.ones(3), "strength": 1.0},
        {"name": "x", "operator": [[np.nan]], "target": [0.0], "strength": 1.0},
        {"name": "x", "operator": [[1.0]], "target": [np.nan], "strength": 1.0},
        {"name": "x", "operator": [[1.0]], "target": [0.0], "strength": -1.0},
    ],
)
def test_regularization_block_rejects_invalid_input(kwargs):
    with pytest.raises((TypeError, ValueError)):
        LinearRegularizationBlock(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"od_layout_fingerprint": ""},
        {"assignment_fingerprint": " "},
        {"mapping_fingerprint": ""},
        {"routing_parameter": 0.0},
        {"routing_parameter": np.inf},
    ],
)
def test_provenance_rejects_invalid_input(kwargs):
    values = {
        "od_layout_fingerprint": "od",
        "assignment_fingerprint": "assignment",
        "mapping_fingerprint": "mapping",
        "routing_parameter": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        FixedRoutingLinearProvenance(**values)
