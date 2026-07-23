from __future__ import annotations

from types import SimpleNamespace

import pytest

from public_transportation.domain.fixed_demand import FixedODDemand, FixedODRecord
from public_transportation.preprocessing import (
    ODPathMetrics,
    ODTimeKey,
    StructuralZeroAnalysisResult,
    StructuralZeroConflictError,
    StructuralZeroReason,
    StructuralZeroRecord,
    load_and_reconcile_fixed_demand,
    reconcile_fixed_demand,
)


def _zero(origin: str, destination: str) -> StructuralZeroRecord:
    return StructuralZeroRecord(
        key=ODTimeKey(origin, destination, "t0"),
        is_structural_zero=True,
        primary_reason=StructuralZeroReason.NO_FEASIBLE_PATH,
        triggered_rules=(StructuralZeroReason.NO_FEASIBLE_PATH,),
        metrics=ODPathMetrics.unreachable(),
    )


def _retained(origin: str, destination: str) -> StructuralZeroRecord:
    return StructuralZeroRecord(
        key=ODTimeKey(origin, destination, "t0"),
        is_structural_zero=False,
        primary_reason=None,
        triggered_rules=(),
        metrics=ODPathMetrics(
            feasible=True,
            minimum_transfers=0,
            minimum_initial_wait_minutes=2.0,
            minimum_journey_time_minutes=10.0,
            feasible_departure_count=2,
            earliest_arrival_seconds=30_000,
        ),
    )


def _analysis(*records: StructuralZeroRecord) -> StructuralZeroAnalysisResult:
    return StructuralZeroAnalysisResult(
        records=tuple(sorted(records, key=lambda record: record.key)),
        scenario_fingerprint="scenario",
        graph_fingerprint="graph",
        configuration_fingerprint="configuration",
    )


def test_new_structural_zeros_are_added_as_sorted_zero_fixed_values() -> None:
    analysis = _analysis(_zero("B", "A"), _retained("A", "B"), _zero("A", "A"))

    result = reconcile_fixed_demand(analysis)

    assert result.fixed_demand.records == (
        FixedODRecord("A", "A", "t0", 0.0),
        FixedODRecord("B", "A", "t0", 0.0),
    )
    assert result.num_existing == 0
    assert result.num_structural_zero == 2
    assert result.num_existing_structural_zero == 0
    assert result.num_added_structural_zero == 2
    assert result.num_merged == 2


def test_compatible_existing_values_are_preserved_exactly() -> None:
    analysis = _analysis(_zero("A", "A"), _retained("A", "B"))
    existing = FixedODDemand(
        records=(
            FixedODRecord("A", "B", "t0", 12.5),
            FixedODRecord("A", "A", "t0", 0.0),
        )
    )

    result = reconcile_fixed_demand(analysis, existing)

    assert result.fixed_demand.records == (
        FixedODRecord("A", "A", "t0", 0.0),
        FixedODRecord("A", "B", "t0", 12.5),
    )
    assert result.num_existing == 2
    assert result.num_existing_structural_zero == 1
    assert result.num_added_structural_zero == 0


def test_nonzero_structural_zero_conflicts_are_all_reported_in_key_order() -> None:
    analysis = _analysis(_zero("A", "A"), _zero("B", "B"))
    existing = FixedODDemand(
        records=(
            FixedODRecord("B", "B", "t0", 3.0),
            FixedODRecord("A", "A", "t0", 2.0),
        )
    )

    with pytest.raises(StructuralZeroConflictError) as captured:
        reconcile_fixed_demand(analysis, existing)

    assert captured.value.conflicts == (
        ("A", "A", "t0", 2.0),
        ("B", "B", "t0", 3.0),
    )
    assert "2 structural-zero cell(s)" in str(captured.value)


def test_existing_key_outside_analysis_is_rejected() -> None:
    analysis = _analysis(_retained("A", "B"))
    existing = FixedODDemand(records=(FixedODRecord("X", "Y", "t0", 0.0),))

    with pytest.raises(ValueError, match="outside the analyzed OD/time universe"):
        reconcile_fixed_demand(analysis, existing)


def test_duplicate_existing_key_is_rejected_even_if_values_match() -> None:
    analysis = _analysis(_retained("A", "B"))
    existing = FixedODDemand(
        records=(
            FixedODRecord("A", "B", "t0", 0.0),
            FixedODRecord("A", "B", "t0", 0.0),
        )
    )

    with pytest.raises(ValueError, match="duplicate key"):
        reconcile_fixed_demand(analysis, existing)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_programmatic_existing_values_are_rejected(value: float) -> None:
    analysis = _analysis(_retained("A", "B"))
    existing = FixedODDemand(records=(FixedODRecord("A", "B", "t0", value),))

    with pytest.raises(ValueError, match="finite and non-negative"):
        reconcile_fixed_demand(analysis, existing)


def test_configured_existing_file_is_loaded_before_reconciliation(tmp_path) -> None:
    path = tmp_path / "fixed.csv"
    path.write_text(
        "origin_stop_id,dest_stop_id,time_bin_id,fixed_flow\nA,B,t0,7.5\n",
        encoding="utf-8",
    )
    scenario = SimpleNamespace(
        stops=[SimpleNamespace(stop_id="A"), SimpleNamespace(stop_id="B")],
        time_bins=[SimpleNamespace(bin_id="t0")],
        demand=SimpleNamespace(
            records=[
                SimpleNamespace(origin_stop_id="A", dest_stop_id="B", time_bin_id="t0")
            ]
        ),
    )
    config = SimpleNamespace(existing_fixed_demand=SimpleNamespace(file=path))

    result = load_and_reconcile_fixed_demand(
        _analysis(_retained("A", "B")), config, scenario=scenario
    )

    assert result.fixed_demand.records == (FixedODRecord("A", "B", "t0", 7.5),)
