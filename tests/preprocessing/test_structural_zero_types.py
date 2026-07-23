from __future__ import annotations

import pytest

from public_transportation.preprocessing import (
    ODPathMetrics,
    ODTimeKey,
    StructuralZeroAnalysisResult,
    StructuralZeroReason,
    StructuralZeroRecord,
)


def _feasible_metrics() -> ODPathMetrics:
    return ODPathMetrics(
        feasible=True,
        minimum_transfers=1,
        minimum_initial_wait_minutes=4.0,
        minimum_journey_time_minutes=27.5,
        feasible_departure_count=3,
        earliest_arrival_seconds=28_500,
    )


def test_path_metric_invariants() -> None:
    assert ODPathMetrics.unreachable().feasible is False

    with pytest.raises(ValueError, match="no path values"):
        ODPathMetrics(
            feasible=False,
            minimum_transfers=0,
            minimum_initial_wait_minutes=None,
            minimum_journey_time_minutes=None,
            feasible_departure_count=0,
            earliest_arrival_seconds=None,
        )

    with pytest.raises(ValueError, match="at least one feasible departure"):
        ODPathMetrics(
            feasible=True,
            minimum_transfers=0,
            minimum_initial_wait_minutes=0.0,
            minimum_journey_time_minutes=10.0,
            feasible_departure_count=0,
            earliest_arrival_seconds=10,
        )


def test_record_reason_invariants() -> None:
    key = ODTimeKey("A", "B", "morning")
    with pytest.raises(ValueError, match="must have triggered rules"):
        StructuralZeroRecord(
            key=key,
            is_structural_zero=True,
            primary_reason=None,
            triggered_rules=(),
            metrics=ODPathMetrics.unreachable(),
        )

    with pytest.raises(ValueError, match="cannot have structural-zero reasons"):
        StructuralZeroRecord(
            key=key,
            is_structural_zero=False,
            primary_reason=StructuralZeroReason.NO_FEASIBLE_PATH,
            triggered_rules=(StructuralZeroReason.NO_FEASIBLE_PATH,),
            metrics=_feasible_metrics(),
        )


def test_analysis_result_counts_and_primary_reasons() -> None:
    zero = StructuralZeroRecord(
        key=ODTimeKey("A", "B", "morning"),
        is_structural_zero=True,
        primary_reason=StructuralZeroReason.NO_FEASIBLE_PATH,
        triggered_rules=(StructuralZeroReason.NO_FEASIBLE_PATH,),
        metrics=ODPathMetrics.unreachable(),
    )
    retained = StructuralZeroRecord(
        key=ODTimeKey("A", "C", "morning"),
        is_structural_zero=False,
        primary_reason=None,
        triggered_rules=(),
        metrics=_feasible_metrics(),
    )
    result = StructuralZeroAnalysisResult(
        records=(zero, retained),
        scenario_fingerprint="scenario",
        graph_fingerprint="graph",
        configuration_fingerprint="configuration",
    )

    assert result.num_cells == 2
    assert result.num_structural_zero == 1
    assert result.num_retained == 1
    assert result.reason_counts[StructuralZeroReason.NO_FEASIBLE_PATH.value] == 1


def test_analysis_records_must_be_sorted_and_unique() -> None:
    def retained(destination: str) -> StructuralZeroRecord:
        return StructuralZeroRecord(
            key=ODTimeKey("A", destination, "morning"),
            is_structural_zero=False,
            primary_reason=None,
            triggered_rules=(),
            metrics=_feasible_metrics(),
        )

    with pytest.raises(ValueError, match="sorted"):
        StructuralZeroAnalysisResult(
            records=(retained("C"), retained("B")),
            scenario_fingerprint="scenario",
            graph_fingerprint="graph",
            configuration_fingerprint="configuration",
        )

    duplicate = retained("B")
    with pytest.raises(ValueError, match="unique"):
        StructuralZeroAnalysisResult(
            records=(duplicate, duplicate),
            scenario_fingerprint="scenario",
            graph_fingerprint="graph",
            configuration_fingerprint="configuration",
        )
