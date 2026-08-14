"""Early validation of reduced-OD time periods and sampling resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .departure_sampling import DepartureTimeSamplingConfig
from .journey_choices import JourneyTimePeriod


@dataclass(frozen=True, slots=True)
class TimePeriodPreflightIssue:
    """One blocking error or advisory warning from period preflight."""

    code: str
    severity: str
    message: str
    period_id: str | None = None
    event_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class ReducedODTimePeriodPreflight:
    """JSON-friendly early report for period coverage and sampling settings."""

    period_ids: tuple[str, ...]
    durations_seconds: Mapping[str, int]
    gaps_seconds: tuple[tuple[int, int], ...]
    relevant_event_count: int
    covered_event_count: int
    uncovered_event_seconds: tuple[int, ...]
    sampling_resolution_seconds: Mapping[str, float]
    issues: tuple[TimePeriodPreflightIssue, ...]

    @property
    def valid(self) -> bool:
        """Whether no blocking issue was found."""
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable representation."""
        return {
            "valid": self.valid,
            "period_ids": list(self.period_ids),
            "durations_seconds": dict(self.durations_seconds),
            "gaps_seconds": [list(gap) for gap in self.gaps_seconds],
            "relevant_event_count": self.relevant_event_count,
            "covered_event_count": self.covered_event_count,
            "uncovered_event_seconds": list(self.uncovered_event_seconds),
            "sampling_resolution_seconds": dict(self.sampling_resolution_seconds),
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "message": issue.message,
                    "period_id": issue.period_id,
                    "event_seconds": issue.event_seconds,
                }
                for issue in self.issues
            ],
        }


def preflight_reduced_od_time_periods(
    time_periods: Sequence[JourneyTimePeriod],
    *,
    relevant_event_seconds: Sequence[int] = (),
    sampling_config: DepartureTimeSamplingConfig | None = None,
    require_contiguous: bool = False,
) -> ReducedODTimePeriodPreflight:
    """Validate period geometry, event coverage, and sampling resolution.

    ``relevant_event_seconds`` should contain event times that must receive a
    period label (normally journey boarding/alighting events, not every event
    in an unrelated service day).  A gap without relevant events is reported
    as an advisory unless ``require_contiguous`` is true.  This function does
    not decide whether a production value is a rate or an interval total; that
    remains an explicit case-study data contract.
    """
    periods = tuple(time_periods)
    issues: list[TimePeriodPreflightIssue] = []
    if sampling_config is not None and not periods:
        issues.append(
            TimePeriodPreflightIssue(
                "sampling_requires_periods",
                "error",
                "departure sampling is configured but no time periods were provided.",
            )
        )
    ids = tuple(period.period_id for period in periods)
    if len(set(ids)) != len(ids):
        issues.append(
            TimePeriodPreflightIssue(
                "duplicate_period_id", "error", "time-period identifiers must be unique."
            )
        )

    ordered = tuple(sorted(periods, key=lambda item: (item.start_seconds, item.period_id)))
    if periods != ordered:
        issues.append(
            TimePeriodPreflightIssue(
                "periods_not_sorted", "error", "time periods must be sorted by start time."
            )
        )
    durations = {period.period_id: period.end_seconds - period.start_seconds for period in periods}
    for period in periods:
        if period.end_seconds <= period.start_seconds:
            issues.append(
                TimePeriodPreflightIssue(
                    "nonpositive_duration",
                    "error",
                    "time period must have positive duration.",
                    period_id=period.period_id,
                )
            )

    gaps: list[tuple[int, int]] = []
    for left, right in zip(ordered, ordered[1:]):
        if left.end_seconds > right.start_seconds:
            issues.append(
                TimePeriodPreflightIssue(
                    "overlapping_periods",
                    "error",
                    f"periods overlap: {left.period_id!r} and {right.period_id!r}.",
                    period_id=right.period_id,
                )
            )
        elif left.end_seconds < right.start_seconds:
            gap = (left.end_seconds, right.start_seconds)
            gaps.append(gap)
            issues.append(
                TimePeriodPreflightIssue(
                    "period_gap",
                    "error" if require_contiguous else "warning",
                    f"there is a gap from {gap[0]} to {gap[1]} seconds.",
                    period_id=right.period_id,
                )
            )

    events = tuple(sorted(set(int(value) for value in relevant_event_seconds)))
    uncovered = tuple(
        seconds
        for seconds in events
        if not any(
            period.start_seconds <= seconds < period.end_seconds for period in periods
        )
    )
    for seconds in uncovered:
        issues.append(
            TimePeriodPreflightIssue(
                "event_outside_periods",
                "error",
                f"event at {seconds} seconds does not map to exactly one time period.",
                event_seconds=seconds,
            )
        )

    resolution: dict[str, float] = {}
    if sampling_config is not None:
        for period in periods:
            try:
                if sampling_config.strategy in {"uniform_midpoint", "fixed_count"}:
                    resolution[period.period_id] = (
                        (period.end_seconds - period.start_seconds)
                        / sampling_config.count_for_period(period.period_id)
                    )
                elif sampling_config.strategy == "fixed_time_step":
                    resolution[period.period_id] = float(
                        sampling_config.step_for_period(period.period_id)
                    )
                else:
                    resolution[period.period_id] = float(
                        sampling_config.initial_interval_seconds
                    )
            except (KeyError, ValueError) as error:
                issues.append(
                    TimePeriodPreflightIssue(
                        "sampling_configuration_missing",
                        "error",
                        str(error),
                        period_id=period.period_id,
                    )
                )

    return ReducedODTimePeriodPreflight(
        period_ids=ids,
        durations_seconds=durations,
        gaps_seconds=tuple(gaps),
        relevant_event_count=len(events),
        covered_event_count=len(events) - len(uncovered),
        uncovered_event_seconds=uncovered,
        sampling_resolution_seconds=resolution,
        issues=tuple(issues),
    )
