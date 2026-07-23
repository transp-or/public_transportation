from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from public_transportation.domain.fixed_demand import FixedODDemand, FixedODRecord
from public_transportation.inference.od_parameter_layout import (
    ODParameterLayout,
    build_od_parameter_layout,
)


def _record(origin: str, destination: str, time_bin: str, flow: float):
    return SimpleNamespace(
        origin_stop_id=origin,
        dest_stop_id=destination,
        time_bin_id=time_bin,
        flow=flow,
    )


def _scenario(records=None):
    if records is None:
        records = [
            _record("A", "B", "t0", 10.0),
            _record("A", "B", "t1", 20.0),
            _record("A", "C", "t0", 30.0),
            _record("B", "C", "t0", 40.0),
            _record("C", "A", "t0", 50.0),
        ]
    return SimpleNamespace(demand=SimpleNamespace(records=records))


def _fixed(*records: FixedODRecord) -> FixedODDemand:
    return FixedODDemand(records=tuple(records))


def test_builds_exact_free_fixed_partition_in_scenario_order():
    layout = build_od_parameter_layout(
        scenario=_scenario(),
        fixed_demand=_fixed(
            FixedODRecord("B", "C", "t0", 12.0),
            FixedODRecord("A", "B", "t1", 0.0),
        ),
    )

    assert layout.num_od_total == 5
    assert layout.od_keys == (
        ("A", "B", "t0"),
        ("A", "B", "t1"),
        ("A", "C", "t0"),
        ("B", "C", "t0"),
        ("C", "A", "t0"),
    )
    assert layout.free_od_indices == (0, 2, 4)
    assert layout.free_baseline_values == (10.0, 30.0, 50.0)
    assert layout.fixed_od_indices == (1, 3)
    assert layout.fixed_od_values == (0.0, 12.0)
    assert layout.fixed_zero_indices == (1,)
    assert layout.fixed_positive_indices == (3,)
    assert layout.num_free == 3
    assert layout.num_fixed == 2
    assert layout.num_fixed_zero == 1
    assert layout.num_fixed_positive == 1


def test_fixed_file_order_does_not_change_layout():
    first = build_od_parameter_layout(
        scenario=_scenario(),
        fixed_demand=_fixed(
            FixedODRecord("A", "B", "t1", 0.0),
            FixedODRecord("B", "C", "t0", 12.0),
        ),
    )
    second = build_od_parameter_layout(
        scenario=_scenario(),
        fixed_demand=_fixed(
            FixedODRecord("B", "C", "t0", 12.0),
            FixedODRecord("A", "B", "t1", 0.0),
        ),
    )
    assert first == second


@pytest.mark.parametrize("fixed_demand", [None, FixedODDemand(records=())])
def test_no_fixed_entries_preserves_all_positive_cells_as_free(fixed_demand):
    layout = build_od_parameter_layout(scenario=_scenario(), fixed_demand=fixed_demand)
    assert layout.free_od_indices == (0, 1, 2, 3, 4)
    assert layout.fixed_od_indices == ()
    assert layout.fixed_od_values == ()
    assert layout.num_free == 5
    assert layout.num_fixed == 0


def test_all_cells_can_be_fixed():
    scenario = _scenario()
    fixed = _fixed(
        *(
            FixedODRecord(r.origin_stop_id, r.dest_stop_id, r.time_bin_id, float(i))
            for i, r in enumerate(scenario.demand.records)
        )
    )
    layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed)
    assert layout.num_free == 0
    assert layout.free_od_indices == ()
    assert layout.free_baseline_values == ()
    assert layout.fixed_od_indices == (0, 1, 2, 3, 4)
    assert layout.fixed_od_values == (0.0, 1.0, 2.0, 3.0, 4.0)


def test_parameter_dimension_depends_only_on_free_cells_and_theta():
    layout = build_od_parameter_layout(
        scenario=_scenario(),
        fixed_demand=_fixed(
            FixedODRecord("A", "B", "t0", 0.0),
            FixedODRecord("A", "B", "t1", 0.0),
            FixedODRecord("A", "C", "t0", 0.0),
        ),
    )
    assert layout.num_free == 2
    assert layout.parameter_dim(estimate_theta=False) == 2
    assert layout.parameter_dim(estimate_theta=True) == 3


def test_milestone_1000_total_990_fixed_has_dimension_10_or_11():
    records = [_record(f"O{i}", f"D{i}", "t0", float(i + 1)) for i in range(1000)]
    frozen = _fixed(
        *(FixedODRecord(f"O{i}", f"D{i}", "t0", 0.0) for i in range(990))
    )
    layout = build_od_parameter_layout(scenario=_scenario(records), fixed_demand=frozen)

    assert layout.num_od_total == 1000
    assert layout.num_fixed == 990
    assert layout.num_free == 10
    assert layout.parameter_dim(estimate_theta=False) == 10
    assert layout.parameter_dim(estimate_theta=True) == 11


def test_free_zero_baseline_is_rejected():
    scenario = _scenario([_record("A", "B", "t0", 0.0)])
    with pytest.raises(ValueError, match="zero baseline.*positive baseline seed or freeze"):
        build_od_parameter_layout(scenario=scenario)


def test_zero_baseline_is_valid_when_frozen_at_zero():
    scenario = _scenario([_record("A", "B", "t0", 0.0)])
    layout = build_od_parameter_layout(
        scenario=scenario,
        fixed_demand=_fixed(FixedODRecord("A", "B", "t0", 0.0)),
    )
    assert layout.num_free == 0
    assert layout.fixed_zero_indices == (0,)


def test_zero_baseline_is_valid_when_frozen_at_positive_value():
    scenario = _scenario([_record("A", "B", "t0", 0.0)])
    layout = build_od_parameter_layout(
        scenario=scenario,
        fixed_demand=_fixed(FixedODRecord("A", "B", "t0", 17.0)),
    )
    assert layout.fixed_od_values == (17.0,)
    assert layout.fixed_positive_indices == (0,)


@pytest.mark.parametrize("baseline", [-1.0, float("nan"), float("inf")])
def test_invalid_scenario_baseline_is_rejected_even_when_cell_is_fixed(baseline):
    scenario = _scenario([_record("A", "B", "t0", baseline)])
    fixed = _fixed(FixedODRecord("A", "B", "t0", 0.0))
    with pytest.raises(ValueError, match="baseline demand.*finite and non-negative"):
        build_od_parameter_layout(scenario=scenario, fixed_demand=fixed)


@pytest.mark.parametrize("fixed_value", [-1.0, float("nan"), float("inf")])
def test_invalid_programmatically_created_fixed_value_is_rejected(fixed_value):
    fixed = _fixed(FixedODRecord("A", "B", "t0", fixed_value))
    with pytest.raises(ValueError, match="fixed demand.*finite and non-negative"):
        build_od_parameter_layout(scenario=_scenario(), fixed_demand=fixed)


def test_fixed_key_absent_from_scenario_is_rejected():
    fixed = _fixed(FixedODRecord("X", "Y", "t0", 0.0))
    with pytest.raises(ValueError, match="not present in scenario.demand.records"):
        build_od_parameter_layout(scenario=_scenario(), fixed_demand=fixed)


def test_duplicate_fixed_key_is_rejected_even_for_programmatic_input():
    fixed = _fixed(
        FixedODRecord("A", "B", "t0", 0.0),
        FixedODRecord("A", "B", "t0", 1.0),
    )
    with pytest.raises(ValueError, match="fixed_demand contains duplicate"):
        build_od_parameter_layout(scenario=_scenario(), fixed_demand=fixed)


def test_duplicate_scenario_key_is_rejected():
    scenario = _scenario(
        [_record("A", "B", "t0", 1.0), _record("A", "B", "t0", 2.0)]
    )
    with pytest.raises(ValueError, match="scenario.demand.records contains duplicate"):
        build_od_parameter_layout(scenario=scenario)


def test_empty_scenario_produces_empty_layout():
    layout = build_od_parameter_layout(scenario=_scenario([]))
    assert layout == ODParameterLayout(
        num_od_total=0,
        od_keys=(),
        free_od_indices=(),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=(),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )


def test_layout_is_frozen_and_uses_immutable_tuples():
    layout = build_od_parameter_layout(scenario=_scenario())
    assert isinstance(layout.free_od_indices, tuple)
    assert isinstance(layout.free_baseline_values, tuple)
    with pytest.raises(FrozenInstanceError):
        layout.num_od_total = 99


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "num_od_total": 1,
            "od_keys": (),
            "free_od_indices": (0,),
            "fixed_od_indices": (),
            "fixed_od_values": (),
            "free_baseline_values": (1.0,),
            "fixed_zero_indices": (),
            "fixed_positive_indices": (),
        },
        {
            "num_od_total": 2,
            "od_keys": (("A", "B", "t0"), ("A", "C", "t0")),
            "free_od_indices": (0,),
            "fixed_od_indices": (),
            "fixed_od_values": (),
            "free_baseline_values": (1.0,),
            "fixed_zero_indices": (),
            "fixed_positive_indices": (),
        },
        {
            "num_od_total": 1,
            "od_keys": (("A", "B", "t0"),),
            "free_od_indices": (0,),
            "fixed_od_indices": (0,),
            "fixed_od_values": (0.0,),
            "free_baseline_values": (1.0,),
            "fixed_zero_indices": (0,),
            "fixed_positive_indices": (),
        },
    ],
)
def test_layout_rejects_invalid_manual_partitions(kwargs):
    with pytest.raises(ValueError):
        ODParameterLayout(**kwargs)
