"""Suggest demand time bins from timestamped passenger counts.

This module is deliberately a preparation-stage diagnostic.  It does not
modify a scenario or construct an assignment operator.  It turns the finest
available timestamped boarding/alighting observations into a small set of
candidate half-open intervals, scores them, and returns a JSON-serializable
recommendation.

The input must retain timestamps at the resolution at which peaks are to be
detected.  If observations have already been aggregated into wide intervals,
no algorithm can recover the missing within-interval information.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from public_transportation.domain.time_of_day import TimeOfDay
from public_transportation.measurement.io import read_measurements_csv
from public_transportation.measurement.schema import MeasurementRecord, MeasurementType


SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TimeDiscretizationConfig:
    """Controls temporal profiling and candidate generation."""

    base_resolution_minutes: int = 5
    min_bin_minutes: int = 10
    max_bin_minutes: int = 60
    max_bins: int = 24
    smoothing_window_bins: int = 3
    peak_quantile: float = 0.75
    peak_min_fraction_of_max: float = 0.35
    peak_gap_tolerance_bins: int = 1
    complexity_penalty: float = 0.02
    min_events_per_bin: float = 0.0
    num_od_pairs: int | None = None
    max_od_cells: int | None = None
    horizon_start_s: int | None = None
    horizon_end_s: int | None = None
    measurement_types: tuple[str, ...] = ("boarding", "alighting")
    allow_infeasible_budget: bool = False

    def __post_init__(self) -> None:
        if self.base_resolution_minutes <= 0:
            raise ValueError("base_resolution_minutes must be positive")
        if self.min_bin_minutes <= 0:
            raise ValueError("min_bin_minutes must be positive")
        if self.max_bin_minutes < self.min_bin_minutes:
            raise ValueError("max_bin_minutes must be >= min_bin_minutes")
        if self.max_bins <= 0:
            raise ValueError("max_bins must be positive")
        if self.smoothing_window_bins <= 0:
            raise ValueError("smoothing_window_bins must be positive")
        if not 0.0 <= self.peak_quantile <= 1.0:
            raise ValueError("peak_quantile must be in [0, 1]")
        if not 0.0 <= self.peak_min_fraction_of_max <= 1.0:
            raise ValueError("peak_min_fraction_of_max must be in [0, 1]")
        if self.peak_gap_tolerance_bins < 0:
            raise ValueError("peak_gap_tolerance_bins must be non-negative")
        if self.complexity_penalty < 0.0 or not math.isfinite(self.complexity_penalty):
            raise ValueError("complexity_penalty must be finite and non-negative")
        if self.min_events_per_bin < 0.0 or not math.isfinite(self.min_events_per_bin):
            raise ValueError("min_events_per_bin must be finite and non-negative")
        if self.num_od_pairs is not None and self.num_od_pairs <= 0:
            raise ValueError("num_od_pairs must be positive when provided")
        if self.max_od_cells is not None and self.max_od_cells <= 0:
            raise ValueError("max_od_cells must be positive when provided")
        if (
            not self.allow_infeasible_budget
            and
            self.num_od_pairs is not None
            and self.max_od_cells is not None
            and self.max_od_cells < self.num_od_pairs
        ):
            raise ValueError("max_od_cells must allow at least one time bin")
        if self.horizon_start_s is not None and self.horizon_start_s < 0:
            raise ValueError("horizon_start_s must be non-negative")
        if self.horizon_end_s is not None and self.horizon_end_s <= 0:
            raise ValueError("horizon_end_s must be positive")
        if (
            self.horizon_start_s is not None
            and self.horizon_end_s is not None
            and self.horizon_end_s <= self.horizon_start_s
        ):
            raise ValueError("horizon_end_s must be after horizon_start_s")
        allowed = {item.value for item in MeasurementType}
        unknown = set(self.measurement_types) - allowed
        if unknown:
            raise ValueError(f"Unknown measurement types: {sorted(unknown)!r}")


@dataclass(frozen=True, slots=True)
class _Profile:
    start_s: int
    end_s: int
    step_s: int
    raw_counts: tuple[float, ...]
    smoothed_counts: tuple[float, ...]
    peak_mask: tuple[bool, ...]
    total_events: float


@dataclass(frozen=True, slots=True)
class _Candidate:
    edges_s: tuple[int, ...]
    score: float
    within_bin_deviance: float
    num_bins: int
    min_bin_minutes: float
    max_bin_minutes: float
    events_per_bin: tuple[float, ...]
    valid: bool
    invalid_reasons: tuple[str, ...]


def _round_down(value: int, step: int) -> int:
    return (value // step) * step


def _round_up(value: int, step: int) -> int:
    return ((value + step - 1) // step) * step


def _format_time(seconds: int) -> str:
    return TimeOfDay(seconds_from_midnight=int(seconds)).to_string(include_seconds=True)


def _merge_short_gaps(mask: np.ndarray, tolerance: int) -> np.ndarray:
    """Fill short false gaps between two true regions."""
    if tolerance <= 0 or not np.any(mask):
        return mask
    result = mask.copy()
    true_positions = np.flatnonzero(mask)
    for left, right in zip(true_positions[:-1], true_positions[1:], strict=True):
        if right - left - 1 <= tolerance:
            result[left : right + 1] = True
    return result


def _profile_counts(
    records: Iterable[MeasurementRecord], config: TimeDiscretizationConfig
) -> _Profile:
    selected = set(config.measurement_types)
    usable: list[MeasurementRecord] = []
    for record in records:
        if record.measurement_type.value not in selected:
            continue
        value = float(record.value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "Selected measurements must have finite non-negative counts; "
                f"got {record.value!r} at {record.time.to_string()}."
            )
        usable.append(record)
    if not usable:
        raise ValueError("No selected measurement records were provided")

    step_s = config.base_resolution_minutes * 60
    observed_start = min(r.time.seconds_from_midnight for r in usable)
    observed_end = max(r.time.seconds_from_midnight for r in usable) + 1
    start_s = (
        config.horizon_start_s
        if config.horizon_start_s is not None
        else _round_down(observed_start, step_s)
    )
    end_s = (
        config.horizon_end_s
        if config.horizon_end_s is not None
        else _round_up(observed_end, step_s)
    )
    start_s = _round_down(start_s, step_s)
    end_s = _round_up(end_s, step_s)
    if end_s <= start_s:
        raise ValueError("The profiling horizon must contain at least one base bin")

    counts = np.zeros((end_s - start_s) // step_s, dtype=float)
    for record in usable:
        time_s = record.time.seconds_from_midnight
        if time_s < start_s or time_s >= end_s:
            continue
        counts[(time_s - start_s) // step_s] += float(record.value)

    window = config.smoothing_window_bins
    if window == 1 or counts.size == 1:
        smoothed = counts.copy()
    else:
        kernel = np.ones(window, dtype=float) / window
        padded = np.pad(counts, (window // 2,), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")[: counts.size]
    positive = smoothed[smoothed > 0.0]
    if positive.size == 0:
        peak_mask = np.zeros_like(smoothed, dtype=bool)
    else:
        threshold = max(
            float(np.quantile(positive, config.peak_quantile)),
            float(positive.max()) * config.peak_min_fraction_of_max,
        )
        peak_mask = smoothed >= threshold
        peak_mask = _merge_short_gaps(peak_mask, config.peak_gap_tolerance_bins)

    return _Profile(
        start_s=start_s,
        end_s=end_s,
        step_s=step_s,
        raw_counts=tuple(float(value) for value in counts),
        smoothed_counts=tuple(float(value) for value in smoothed),
        peak_mask=tuple(bool(value) for value in peak_mask),
        total_events=float(counts.sum()),
    )


def _uniform_edges(start_s: int, end_s: int, width_s: int) -> tuple[int, ...]:
    edges = [start_s]
    while edges[-1] < end_s:
        edges.append(min(edges[-1] + width_s, end_s))
    return tuple(edges)


def _adaptive_edges(profile: _Profile, config: TimeDiscretizationConfig) -> tuple[int, ...]:
    """Use fine bins in peak regions and coarse bins elsewhere."""
    min_s = config.min_bin_minutes * 60
    max_s = config.max_bin_minutes * 60
    edges = [profile.start_s]
    while edges[-1] < profile.end_s:
        index = min((edges[-1] - profile.start_s) // profile.step_s, len(profile.peak_mask) - 1)
        width = min_s if profile.peak_mask[index] else max_s
        edges.append(min(edges[-1] + width, profile.end_s))
    return tuple(edges)


def _events_for_edges(profile: _Profile, edges_s: Sequence[int]) -> tuple[float, ...]:
    values = np.asarray(profile.raw_counts, dtype=float)
    result = []
    for left, right in zip(edges_s[:-1], edges_s[1:], strict=True):
        first = max(0, (left - profile.start_s) // profile.step_s)
        last = min(values.size, (right - profile.start_s) // profile.step_s)
        result.append(float(values[first:last].sum()))
    return tuple(result)


def _deviance_for_edges(profile: _Profile, edges_s: Sequence[int]) -> float:
    values = np.asarray(profile.raw_counts, dtype=float)
    deviance = 0.0
    for left, right in zip(edges_s[:-1], edges_s[1:], strict=True):
        first = max(0, (left - profile.start_s) // profile.step_s)
        last = min(values.size, (right - profile.start_s) // profile.step_s)
        segment = values[first:last]
        if segment.size == 0:
            continue
        mean = float(segment.mean())
        deviance += float(np.square(segment - mean).sum() / (mean + 1.0))
    return deviance


def _evaluate_candidate(
    profile: _Profile,
    edges_s: Sequence[int],
    config: TimeDiscretizationConfig,
) -> _Candidate:
    durations = np.diff(np.asarray(edges_s, dtype=float)) / 60.0
    events = _events_for_edges(profile, edges_s)
    reasons: list[str] = []
    if len(edges_s) - 1 > config.max_bins:
        reasons.append("exceeds_max_bins")
    if durations.size and float(durations.min()) < config.min_bin_minutes:
        reasons.append("bin_below_minimum_width")
    if durations.size and float(durations.max()) > config.max_bin_minutes:
        reasons.append("bin_above_maximum_width")
    if any(value < config.min_events_per_bin for value in events):
        reasons.append("bin_below_minimum_events")
    if (
        config.num_od_pairs is not None
        and config.max_od_cells is not None
        and (len(edges_s) - 1) * config.num_od_pairs > config.max_od_cells
    ):
        reasons.append("exceeds_max_od_cells")
    deviance = _deviance_for_edges(profile, edges_s)
    score = deviance + config.complexity_penalty * (len(edges_s) - 1)
    return _Candidate(
        edges_s=tuple(int(value) for value in edges_s),
        score=float(score),
        within_bin_deviance=float(deviance),
        num_bins=len(edges_s) - 1,
        min_bin_minutes=float(durations.min()) if durations.size else 0.0,
        max_bin_minutes=float(durations.max()) if durations.size else 0.0,
        events_per_bin=events,
        valid=not reasons,
        invalid_reasons=tuple(reasons),
    )


def _candidate_edges(profile: _Profile, config: TimeDiscretizationConfig) -> list[tuple[str, tuple[int, ...]]]:
    widths = {config.min_bin_minutes, config.max_bin_minutes}
    midpoint = (config.min_bin_minutes + config.max_bin_minutes) // 2
    widths.add(max(config.min_bin_minutes, midpoint))
    candidates = [(f"uniform_{width}m", _uniform_edges(profile.start_s, profile.end_s, width * 60)) for width in sorted(widths)]
    candidates.append(("peak_adaptive", _adaptive_edges(profile, config)))
    return candidates


def _time_bins(edges_s: Sequence[int]) -> list[dict[str, object]]:
    return [
        {
            "bin_id": f"t{index}",
            "start_s": int(left),
            "end_s": int(right),
            "start": _format_time(int(left)),
            "end": _format_time(int(right)),
        }
        for index, (left, right) in enumerate(zip(edges_s[:-1], edges_s[1:], strict=True))
    ]


def recommend_time_discretization(
    records: Iterable[MeasurementRecord], config: TimeDiscretizationConfig | None = None
) -> dict[str, object]:
    """Return a deterministic JSON-compatible time-bin recommendation."""
    resolved = config or TimeDiscretizationConfig()
    profile = _profile_counts(records, resolved)
    candidates = []
    for name, edges in _candidate_edges(profile, resolved):
        candidate = _evaluate_candidate(profile, edges, resolved)
        estimated_od_cells = (
            None if resolved.num_od_pairs is None else candidate.num_bins * resolved.num_od_pairs
        )
        candidates.append(
            {
                "name": name,
                "valid": candidate.valid,
                "invalid_reasons": list(candidate.invalid_reasons),
                "score": candidate.score,
                "within_bin_deviance": candidate.within_bin_deviance,
                "num_bins": candidate.num_bins,
                "min_bin_minutes": candidate.min_bin_minutes,
                "max_bin_minutes": candidate.max_bin_minutes,
                "events_per_bin": list(candidate.events_per_bin),
                "estimated_od_cells": estimated_od_cells,
                "max_od_cells": resolved.max_od_cells,
                "time_bins": _time_bins(candidate.edges_s),
                "edges": [
                    {"time_s": edge, "time": _format_time(edge)}
                    for edge in candidate.edges_s
                ],
            }
        )
    valid = [item for item in candidates if item["valid"]]
    selected = (
        min(valid, key=lambda item: (float(item["score"]), int(item["num_bins"]), str(item["name"])))
        if valid
        else None
    )
    peak_intervals = []
    mask = np.asarray(profile.peak_mask, dtype=bool)
    start: int | None = None
    for index, is_peak in enumerate(np.r_[mask, False]):
        if is_peak and start is None:
            start = index
        elif not is_peak and start is not None:
            peak_intervals.append(
                {
                    "start": _format_time(profile.start_s + start * profile.step_s),
                    "end": _format_time(profile.start_s + index * profile.step_s),
                }
            )
            start = None
    configuration = asdict(resolved)
    configuration.pop("allow_infeasible_budget", None)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if selected is not None else "blocked",
        "retained_pair_count": resolved.num_od_pairs,
        "max_od_cells": resolved.max_od_cells,
        "configuration": configuration,
        "profile": {
            "start": _format_time(profile.start_s),
            "end": _format_time(profile.end_s),
            "base_resolution_minutes": resolved.base_resolution_minutes,
            "raw_counts": list(profile.raw_counts),
            "smoothed_counts": list(profile.smoothed_counts),
            "total_events": profile.total_events,
        },
        "peak_intervals": peak_intervals,
        "candidates": candidates,
        "recommendation": selected,
        "warnings": [
            "Candidate selection uses the timestamp resolution present in the input; "
            "it cannot recover variation hidden by prior aggregation."
        ],
    }
    if selected is None:
        report.update(
            {
                "reason": "no_candidate_within_max_od_cells",
                "required_decision": "approve a larger budget or revise the time-bin policy",
            }
        )
    return report


def recommend_time_discretization_from_csv(
    path: str | Path, config: TimeDiscretizationConfig | None = None
) -> dict[str, object]:
    """Load canonical measurement CSV data and recommend time bins."""
    table = read_measurements_csv(path)
    return recommend_time_discretization(table.records, config=config)


def _parse_time(value: str | None) -> int | None:
    return None if value is None else TimeOfDay.parse(value).seconds_from_midnight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--base-resolution-minutes", type=int, default=5)
    parser.add_argument("--min-bin-minutes", type=int, default=10)
    parser.add_argument("--max-bin-minutes", type=int, default=60)
    parser.add_argument("--max-bins", type=int, default=24)
    parser.add_argument("--smoothing-window-bins", type=int, default=3)
    parser.add_argument("--peak-quantile", type=float, default=0.75)
    parser.add_argument("--peak-min-fraction-of-max", type=float, default=0.35)
    parser.add_argument("--min-events-per-bin", type=float, default=0.0)
    parser.add_argument("--num-od-pairs", type=int)
    parser.add_argument("--max-od-cells", type=int)
    parser.add_argument("--horizon-start")
    parser.add_argument("--horizon-end")
    args = parser.parse_args()
    config = TimeDiscretizationConfig(
        base_resolution_minutes=args.base_resolution_minutes,
        min_bin_minutes=args.min_bin_minutes,
        max_bin_minutes=args.max_bin_minutes,
        max_bins=args.max_bins,
        smoothing_window_bins=args.smoothing_window_bins,
        peak_quantile=args.peak_quantile,
        peak_min_fraction_of_max=args.peak_min_fraction_of_max,
        min_events_per_bin=args.min_events_per_bin,
        num_od_pairs=args.num_od_pairs,
        max_od_cells=args.max_od_cells,
        horizon_start_s=_parse_time(args.horizon_start),
        horizon_end_s=_parse_time(args.horizon_end),
    )
    report = recommend_time_discretization_from_csv(args.measurements, config=config)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        if args.output_json.suffix.lower() != ".json":
            raise ValueError("--output-json must name a .json file")
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
