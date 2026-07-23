"""Classify structural zeros from scheduled OD/time path metrics."""

from __future__ import annotations

from typing import TypeVar

from .config import RulesConfig, StructuralZeroConfig
from .path_metrics import compute_od_path_metrics
from .topology import StructuralZeroTopology
from .types import (
    ODPathMetricRecord,
    ODTimeKey,
    StructuralZeroAnalysisResult,
    StructuralZeroReason,
    StructuralZeroRecord,
)


PRIMARY_REASON_PRECEDENCE = (
    StructuralZeroReason.SAME_STOP,
    StructuralZeroReason.NO_FEASIBLE_PATH,
    StructuralZeroReason.MAXIMUM_TRANSFERS_EXCEEDED,
    StructuralZeroReason.MAXIMUM_INITIAL_WAIT_EXCEEDED,
    StructuralZeroReason.MAXIMUM_JOURNEY_TIME_EXCEEDED,
    StructuralZeroReason.INSUFFICIENT_FEASIBLE_DEPARTURES,
)

_RuleConfigT = TypeVar("_RuleConfigT")


def classify_structural_zeros(
    metric_records: tuple[ODPathMetricRecord, ...],
    *,
    rules: RulesConfig,
    scenario_fingerprint: str,
    graph_fingerprint: str,
    configuration_fingerprint: str,
) -> StructuralZeroAnalysisResult:
    """Apply enabled rules and return a deterministic immutable result.

    Maximum rules reject only values strictly above their threshold. The
    minimum-departures rule rejects only counts strictly below its threshold.
    Consequently, equality is retained.
    """
    records: list[StructuralZeroRecord] = []
    for metric_record in metric_records:
        key = metric_record.key
        metrics = metric_record.metrics
        triggered: set[StructuralZeroReason] = set()

        if rules.enabled.same_stop and key.origin_stop_id == key.dest_stop_id:
            triggered.add(StructuralZeroReason.SAME_STOP)
        if rules.enabled.no_feasible_path and not metrics.feasible:
            triggered.add(StructuralZeroReason.NO_FEASIBLE_PATH)

        if rules.enabled.maximum_transfers and metrics.feasible:
            section = _required_rule_section(
                rules.maximum_transfers, "maximum_transfers"
            )
            assert metrics.minimum_transfers is not None
            if metrics.minimum_transfers > section.max_transfers:
                triggered.add(StructuralZeroReason.MAXIMUM_TRANSFERS_EXCEEDED)

        if rules.enabled.maximum_initial_wait and metrics.feasible:
            section = _required_rule_section(
                rules.maximum_initial_wait, "maximum_initial_wait"
            )
            assert metrics.minimum_initial_wait_minutes is not None
            if metrics.minimum_initial_wait_minutes > section.max_initial_wait_minutes:
                triggered.add(StructuralZeroReason.MAXIMUM_INITIAL_WAIT_EXCEEDED)

        if rules.enabled.maximum_journey_time and metrics.feasible:
            section = _required_rule_section(
                rules.maximum_journey_time, "maximum_journey_time"
            )
            assert metrics.minimum_journey_time_minutes is not None
            if metrics.minimum_journey_time_minutes > section.max_journey_time_minutes:
                triggered.add(StructuralZeroReason.MAXIMUM_JOURNEY_TIME_EXCEEDED)

        if rules.enabled.minimum_feasible_departures:
            section = _required_rule_section(
                rules.minimum_feasible_departures, "minimum_feasible_departures"
            )
            if metrics.feasible_departure_count < section.min_feasible_departures:
                triggered.add(StructuralZeroReason.INSUFFICIENT_FEASIBLE_DEPARTURES)

        ordered_reasons = tuple(
            reason for reason in PRIMARY_REASON_PRECEDENCE if reason in triggered
        )
        records.append(
            StructuralZeroRecord(
                key=key,
                is_structural_zero=bool(ordered_reasons),
                primary_reason=ordered_reasons[0] if ordered_reasons else None,
                triggered_rules=ordered_reasons,
                metrics=metrics,
            )
        )

    return StructuralZeroAnalysisResult(
        records=tuple(sorted(records, key=lambda record: record.key)),
        scenario_fingerprint=scenario_fingerprint,
        graph_fingerprint=graph_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
    )


def analyze_structural_zeros(
    topology: StructuralZeroTopology,
    config: StructuralZeroConfig,
    *,
    scenario_fingerprint: str,
    keys: tuple[ODTimeKey, ...] | None = None,
) -> StructuralZeroAnalysisResult:
    """Compute path metrics and classify all OD/time cells."""
    if topology.assignment_config != config.assignment:
        raise ValueError(
            "Topology feasibility settings do not match config.assignment; "
            "rebuild the topology from this configuration."
        )
    return classify_structural_zeros(
        compute_od_path_metrics(topology, keys=keys),
        rules=config.rules,
        scenario_fingerprint=scenario_fingerprint,
        graph_fingerprint=topology.fingerprint,
        configuration_fingerprint=config.fingerprint,
    )


def _required_rule_section(section: _RuleConfigT | None, name: str) -> _RuleConfigT:
    if section is None:
        raise ValueError(f"rules.{name} must be configured when enabled.")
    return section
