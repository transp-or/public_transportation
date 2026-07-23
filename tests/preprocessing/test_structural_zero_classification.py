from __future__ import annotations

from dataclasses import replace

import pytest

from public_transportation.preprocessing import (
    ODPathMetricRecord,
    ODPathMetrics,
    ODTimeKey,
    PRIMARY_REASON_PRECEDENCE,
    StructuralZeroReason,
    classify_structural_zeros,
)
from public_transportation.preprocessing.structural_zeros.config import (
    EnabledRulesConfig,
    MaximumInitialWaitRuleConfig,
    MaximumJourneyTimeRuleConfig,
    MaximumTransfersRuleConfig,
    MinimumFeasibleDeparturesRuleConfig,
    NoFeasiblePathRuleConfig,
    RulesConfig,
    SameStopRuleConfig,
)


def _rules(*, enabled: EnabledRulesConfig | None = None) -> RulesConfig:
    if enabled is None:
        enabled = EnabledRulesConfig(
            same_stop=True,
            no_feasible_path=True,
            maximum_transfers=True,
            maximum_initial_wait=True,
            maximum_journey_time=True,
            minimum_feasible_departures=True,
        )
    return RulesConfig(
        enabled=enabled,
        same_stop=SameStopRuleConfig(),
        no_feasible_path=NoFeasiblePathRuleConfig(),
        maximum_transfers=MaximumTransfersRuleConfig(max_transfers=2),
        maximum_initial_wait=MaximumInitialWaitRuleConfig(
            max_initial_wait_minutes=30.0
        ),
        maximum_journey_time=MaximumJourneyTimeRuleConfig(
            max_journey_time_minutes=120.0
        ),
        minimum_feasible_departures=MinimumFeasibleDeparturesRuleConfig(
            min_feasible_departures=2
        ),
    )


def _feasible(
    key: ODTimeKey,
    *,
    transfers: int,
    wait: float,
    journey: float,
    departures: int,
) -> ODPathMetricRecord:
    return ODPathMetricRecord(
        key=key,
        metrics=ODPathMetrics(
            feasible=True,
            minimum_transfers=transfers,
            minimum_initial_wait_minutes=wait,
            minimum_journey_time_minutes=journey,
            feasible_departure_count=departures,
            earliest_arrival_seconds=36_000,
        ),
    )


def _classify(records: tuple[ODPathMetricRecord, ...], rules: RulesConfig):
    return classify_structural_zeros(
        records,
        rules=rules,
        scenario_fingerprint="scenario",
        graph_fingerprint="graph",
        configuration_fingerprint="configuration",
    )


def test_all_applicable_reasons_are_preserved_in_precedence_order() -> None:
    excessive = _feasible(
        ODTimeKey("A", "B", "t0"),
        transfers=3,
        wait=31.0,
        journey=121.0,
        departures=1,
    )
    result = _classify((excessive,), _rules())
    record = result.records[0]

    assert record.is_structural_zero
    assert record.primary_reason is StructuralZeroReason.MAXIMUM_TRANSFERS_EXCEEDED
    assert record.triggered_rules == (
        StructuralZeroReason.MAXIMUM_TRANSFERS_EXCEEDED,
        StructuralZeroReason.MAXIMUM_INITIAL_WAIT_EXCEEDED,
        StructuralZeroReason.MAXIMUM_JOURNEY_TIME_EXCEEDED,
        StructuralZeroReason.INSUFFICIENT_FEASIBLE_DEPARTURES,
    )
    assert record.triggered_rules == tuple(
        reason
        for reason in PRIMARY_REASON_PRECEDENCE
        if reason in record.triggered_rules
    )


def test_same_stop_unreachable_cell_has_deterministic_primary_reason() -> None:
    metric_record = ODPathMetricRecord(
        key=ODTimeKey("A", "A", "t0"), metrics=ODPathMetrics.unreachable()
    )
    record = _classify((metric_record,), _rules()).records[0]

    assert record.primary_reason is StructuralZeroReason.SAME_STOP
    assert record.triggered_rules == (
        StructuralZeroReason.SAME_STOP,
        StructuralZeroReason.NO_FEASIBLE_PATH,
        StructuralZeroReason.INSUFFICIENT_FEASIBLE_DEPARTURES,
    )


def test_threshold_equality_is_retained() -> None:
    boundary = _feasible(
        ODTimeKey("A", "B", "t0"),
        transfers=2,
        wait=30.0,
        journey=120.0,
        departures=2,
    )
    record = _classify((boundary,), _rules()).records[0]

    assert not record.is_structural_zero
    assert record.primary_reason is None
    assert record.triggered_rules == ()


def test_disabled_rules_are_ignored_even_when_sections_are_present() -> None:
    enabled = EnabledRulesConfig(
        same_stop=False,
        no_feasible_path=False,
        maximum_transfers=False,
        maximum_initial_wait=False,
        maximum_journey_time=False,
        minimum_feasible_departures=False,
    )
    excessive = _feasible(
        ODTimeKey("A", "B", "t0"),
        transfers=99,
        wait=999.0,
        journey=999.0,
        departures=1,
    )
    record = _classify((excessive,), _rules(enabled=enabled)).records[0]

    assert not record.is_structural_zero


def test_enabled_parameterized_rule_requires_its_section() -> None:
    rules = replace(_rules(), maximum_transfers=None)
    metric_record = _feasible(
        ODTimeKey("A", "B", "t0"),
        transfers=3,
        wait=0.0,
        journey=1.0,
        departures=2,
    )

    with pytest.raises(ValueError, match="rules.maximum_transfers"):
        _classify((metric_record,), rules)


def test_result_is_sorted_and_primary_reason_counts_are_reported() -> None:
    later = ODPathMetricRecord(
        key=ODTimeKey("B", "A", "t0"), metrics=ODPathMetrics.unreachable()
    )
    earlier = _feasible(
        ODTimeKey("A", "B", "t0"),
        transfers=0,
        wait=0.0,
        journey=10.0,
        departures=2,
    )
    result = _classify((later, earlier), _rules())

    assert tuple(record.key for record in result.records) == (earlier.key, later.key)
    assert result.num_cells == 2
    assert result.num_structural_zero == 1
    assert result.reason_counts[StructuralZeroReason.NO_FEASIBLE_PATH.value] == 1
