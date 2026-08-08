from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.inference.assignment_contract import (
    AssignmentCompatibilityError,
    AssignmentOperator,
    CanonicalAssignmentIndex,
    CanonicalMeasurement,
    CanonicalODTimeCell,
    CanonicalTimeInterval,
    FixedRoutingAssignmentOperatorAdapter,
    assert_assignment_artifact_compatible,
    build_canonical_assignment_index,
    build_fixed_routing_artifact_identity,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    FixedRoutingMeasurementOperator,
    MeasurementOperatorMetrics,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout


def _layout(*, fixed_value: float = 4.0, free_baseline: float = 10.0):
    return ODParameterLayout(
        num_od_total=4,
        od_keys=(
            ("o0", "d0", "morning-1"),
            ("o0", "d1", "morning-1"),
            ("o1", "d0", "morning-2"),
            ("o1", "d1", "morning-2"),
        ),
        free_od_indices=(0, 2),
        fixed_od_indices=(1, 3),
        fixed_od_values=(0.0, fixed_value),
        free_baseline_values=(free_baseline, 20.0),
        fixed_zero_indices=(1,),
        fixed_positive_indices=(3,),
    )


def _intervals():
    return (
        CanonicalTimeInterval("morning-1", 0, 900),
        CanonicalTimeInterval("morning-2", 900, 1800),
    )


def _measurements():
    return (
        CanonicalMeasurement(0, "board-stop-a-1", "boarding", "stop-a", "morning-1"),
        CanonicalMeasurement(1, "alight-stop-b-2", "alighting", "stop-b", "morning-2"),
    )


def _index(**layout_options):
    return build_canonical_assignment_index(
        parameter_layout=_layout(**layout_options),
        time_intervals=_intervals(),
        measurements=_measurements(),
    )


def _operator():
    layout = _layout()
    return FixedRoutingMeasurementOperator(
        matrix=jnp.asarray([[1.0, 2.0], [3.0, -1.0]], dtype=jnp.float32),
        fixed_measurement_offset=jnp.asarray([4.0, 5.0], dtype=jnp.float32),
        representation="dense",
        num_active_od=3,
        num_free_od=2,
        num_measurements=2,
        od_layout_fingerprint=layout.fingerprint,
        compact_layout_fingerprint=build_compact_od_assignment_layout(
            parameter_layout=layout
        ).fingerprint,
        assignment_fingerprint="assignment-and-timetable",
        graph_fingerprint="physical-network",
        mapping_fingerprint="measurement-map",
        theta=1.25,
        dtype="float32",
        metrics=MeasurementOperatorMetrics(
            construction_seconds=0.0,
            dense_bytes=16,
            stored_bytes=16,
            peak_construction_bytes=16,
            nonzero_entries=4,
            total_entries=4,
            density=1.0,
            chunk_size=2,
        ),
    )


def _identity(operator, index):
    return build_fixed_routing_artifact_identity(
        operator=operator,
        canonical_index=index,
        temporal_discretization_fingerprint="intervals-v1",
        departure_choice_fingerprint="departure-choice-v1",
        feasibility_fingerprint="feasibility-v1",
        coefficient_policy_fingerprint="exact-float32",
    )


def test_canonical_tables_preserve_full_order_and_reduced_columns():
    index = _index()

    assert index.number_of_physical_demand_cells == 4
    assert index.number_of_demand_cells == 2
    assert index.number_of_measurements == 2
    assert tuple(cell.role for cell in index.demand_cells) == (
        "free",
        "fixed_zero",
        "free",
        "fixed_positive",
    )
    assert tuple(cell.operator_column for cell in index.demand_cells) == (
        0,
        None,
        1,
        None,
    )


def test_artifact_identity_excludes_demand_baselines_and_fixed_binding():
    first = _index(free_baseline=10.0, fixed_value=4.0)
    changed_demand_binding = _index(free_baseline=99.0, fixed_value=8.0)

    assert first.artifact_fingerprint == changed_demand_binding.artifact_fingerprint
    assert first.binding_fingerprint != changed_demand_binding.binding_fingerprint


def test_physical_or_measurement_index_change_invalidates_artifact():
    index = _index()
    changed_measurements = CanonicalAssignmentIndex(
        time_intervals=index.time_intervals,
        demand_cells=index.demand_cells,
        measurements=(
            replace(index.measurements[0], location_id="different-stop"),
            index.measurements[1],
        ),
    )
    changed_intervals = CanonicalAssignmentIndex(
        time_intervals=(
            CanonicalTimeInterval("morning-1", 0, 600),
            CanonicalTimeInterval("morning-2", 600, 1800),
        ),
        demand_cells=index.demand_cells,
        measurements=index.measurements,
    )

    assert changed_measurements.artifact_fingerprint != index.artifact_fingerprint
    assert changed_intervals.artifact_fingerprint != index.artifact_fingerprint


def test_fixed_routing_adapter_preserves_forward_and_adjoint_products():
    operator = _operator()
    index = _index()
    adapter = FixedRoutingAssignmentOperatorAdapter(
        operator=operator,
        canonical_index=index,
        identity=_identity(operator, index),
    )
    demand = jnp.asarray([2.0, 7.0], dtype=jnp.float32)
    residual = jnp.asarray([-0.5, 3.0], dtype=jnp.float32)

    np.testing.assert_array_equal(adapter.matvec(demand), operator.jax_matvec(demand))
    np.testing.assert_array_equal(
        adapter.rmatvec(residual), operator.jax_rmatvec(residual)
    )
    np.testing.assert_array_equal(
        adapter.fixed_measurement_offset, operator.fixed_measurement_offset
    )
    assert isinstance(adapter, AssignmentOperator)
    assert adapter.number_of_demand_cells == 2
    assert adapter.number_of_measurements == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("network_fingerprint", "other-network", "graph fingerprint"),
        ("timetable_fingerprint", "other-timetable", "assignment fingerprint"),
        ("route_choice_fingerprint", "other-theta", "route-choice fingerprint"),
        ("measurement_mapping_fingerprint", "other-map", "mapping fingerprint"),
        ("numeric_dtype", "float64", "dtype is incompatible"),
    ),
)
def test_adapter_rejects_incompatible_fixed_routing_identity(field, value, message):
    operator = _operator()
    index = _index()
    identity = replace(_identity(operator, index), **{field: value})

    with pytest.raises(AssignmentCompatibilityError, match=message):
        FixedRoutingAssignmentOperatorAdapter(operator, index, identity)


def test_adapter_rejects_canonical_dimension_mismatch():
    operator = replace(_operator(), num_free_od=3)
    index = _index()
    with pytest.raises(AssignmentCompatibilityError, match="demand dimension"):
        FixedRoutingAssignmentOperatorAdapter(
            operator, index, _identity(operator, index)
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("od_layout_fingerprint", "OD-layout fingerprint"),
        ("compact_layout_fingerprint", "compact-layout fingerprint"),
    ),
)
def test_adapter_rejects_incompatible_reduced_binding(field, message):
    operator = replace(_operator(), **{field: "different-layout"})
    index = _index()
    with pytest.raises(AssignmentCompatibilityError, match=message):
        FixedRoutingAssignmentOperatorAdapter(
            operator, index, _identity(operator, index)
        )


@pytest.mark.parametrize(
    "field",
    (
        "canonical_index_fingerprint",
        "network_fingerprint",
        "timetable_fingerprint",
        "temporal_discretization_fingerprint",
        "route_choice_fingerprint",
        "departure_choice_fingerprint",
        "feasibility_fingerprint",
        "measurement_mapping_fingerprint",
        "coefficient_policy_fingerprint",
        "numeric_dtype",
    ),
)
def test_identity_compatibility_rejects_every_artifact_dependency(field):
    operator = _operator()
    identity = _identity(operator, _index())
    changed = replace(identity, **{field: f"changed-{field}"})

    with pytest.raises(AssignmentCompatibilityError, match=field):
        assert_assignment_artifact_compatible(expected=identity, actual=changed)


def test_identity_is_immutable_and_stable():
    operator = _operator()
    index = _index()
    first = _identity(operator, index)
    second = _identity(operator, index)

    assert first.fingerprint == second.fingerprint
    with pytest.raises(FrozenInstanceError):
        first.network_fingerprint = "changed"


def test_index_rejects_noncontiguous_operator_columns():
    with pytest.raises(ValueError, match="contiguous operator columns"):
        CanonicalAssignmentIndex(
            time_intervals=_intervals(),
            demand_cells=(
                CanonicalODTimeCell(0, "o", "d", "morning-1", "free", 0),
                CanonicalODTimeCell(1, "o", "d2", "morning-1", "free", 2),
            ),
            measurements=_measurements(),
        )


def test_unsupported_identity_schema_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unsupported assignment-artifact"):
        replace(_identity(_operator(), _index()), schema_version=2)
