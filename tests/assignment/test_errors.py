# tests/assignment/test_errors.py
from __future__ import annotations

import pytest

from public_transportation.assignment.errors import (
    AssignmentConfigError,
    AssignmentError,
    CostComputationError,
    DialDPError,
    GraphBuildError,
    GraphConsistencyError,
    NumericalStabilityError,
    ODDemandShapeError,
    ODGroupingError,
    ScenarioCompatibilityError,
)


ALL_ERROR_CLASSES = [
    AssignmentError,
    AssignmentConfigError,
    ScenarioCompatibilityError,
    GraphBuildError,
    GraphConsistencyError,
    ODGroupingError,
    ODDemandShapeError,
    CostComputationError,
    DialDPError,
    NumericalStabilityError,
]


@pytest.mark.parametrize("error_cls", ALL_ERROR_CLASSES)
def test_all_assignment_errors_are_exceptions(error_cls):
    assert issubclass(error_cls, Exception)


@pytest.mark.parametrize(
    ("error_cls", "parent_cls"),
    [
        (AssignmentConfigError, AssignmentError),
        (ScenarioCompatibilityError, AssignmentError),
        (GraphBuildError, AssignmentError),
        (GraphConsistencyError, GraphBuildError),
        (ODGroupingError, AssignmentError),
        (ODDemandShapeError, ODGroupingError),
        (CostComputationError, AssignmentError),
        (DialDPError, AssignmentError),
        (NumericalStabilityError, DialDPError),
    ],
)
def test_error_hierarchy(error_cls, parent_cls):
    assert issubclass(error_cls, parent_cls)


@pytest.mark.parametrize(
    ("error_cls", "broad_cls"),
    [
        (AssignmentConfigError, AssignmentError),
        (ScenarioCompatibilityError, AssignmentError),
        (GraphConsistencyError, AssignmentError),
        (ODDemandShapeError, AssignmentError),
        (CostComputationError, AssignmentError),
        (NumericalStabilityError, AssignmentError),
    ],
)
def test_all_specialized_errors_are_caught_by_assignment_error(error_cls, broad_cls):
    with pytest.raises(broad_cls):
        raise error_cls("test message")


@pytest.mark.parametrize("error_cls", ALL_ERROR_CLASSES)
def test_error_message_is_preserved(error_cls):
    err = error_cls("specific failure")
    assert str(err) == "specific failure"


@pytest.mark.parametrize("error_cls", ALL_ERROR_CLASSES)
def test_error_can_be_raised_and_caught_as_exact_class(error_cls):
    with pytest.raises(error_cls, match="specific failure"):
        raise error_cls("specific failure")


def test_graph_consistency_error_is_caught_as_graph_build_error():
    with pytest.raises(GraphBuildError):
        raise GraphConsistencyError("acyclicity violation")


def test_od_demand_shape_error_is_caught_as_od_grouping_error():
    with pytest.raises(ODGroupingError):
        raise ODDemandShapeError("invalid OD vector shape")


def test_numerical_stability_error_is_caught_as_dial_dp_error():
    with pytest.raises(DialDPError):
        raise NumericalStabilityError("nan encountered")


def test_assignment_error_can_wrap_original_exception():
    original = ValueError("low-level failure")

    try:
        raise AssignmentError("high-level failure") from original
    except AssignmentError as err:
        assert str(err) == "high-level failure"
        assert err.__cause__ is original


@pytest.mark.parametrize("error_cls", ALL_ERROR_CLASSES)
def test_error_classes_have_docstrings(error_cls):
    assert error_cls.__doc__ is not None
    assert error_cls.__doc__.strip()


def test_hierarchy_does_not_make_unrelated_errors_subclasses():
    assert not issubclass(AssignmentConfigError, GraphBuildError)
    assert not issubclass(CostComputationError, DialDPError)
    assert not issubclass(ODGroupingError, CostComputationError)