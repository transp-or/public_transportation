"""Durable, non-invasive profiling for fixed-routing support discovery.

The profiler is intentionally an observer around the support-discovery API.  It
does not change the support evaluator, its scheduling, or any computational
identity.  Callers provide the scientific and input fingerprints in ``metadata``
and choose an isolated output path for the resulting diagnostic JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Mapping

from .fixed_routing_origin_support import (
    GroupSupportTiming,
    GroupSupportTimingCallback,
)

SUPPORT_DISCOVERY_PROFILE_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert common provenance values without changing their meaning."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True, slots=True)
class SupportDiscoveryProfileRecorder:
    """Thread-safe collector used as a ``GroupSupportTiming`` callback."""

    _records: list[GroupSupportTiming]
    _lock: Lock

    @classmethod
    def create(cls) -> "SupportDiscoveryProfileRecorder":
        return cls(_records=[], _lock=Lock())

    def __call__(self, timing: GroupSupportTiming) -> None:
        with self._lock:
            self._records.append(timing)

    def records(self) -> tuple[GroupSupportTiming, ...]:
        with self._lock:
            return tuple(self._records)


def representative_support_groups(
    records: tuple[GroupSupportTiming, ...] | list[GroupSupportTiming],
    *,
    count: int = 3,
) -> tuple[int, ...]:
    """Return deterministic small/typical/large group IDs for a pilot.

    This helper only selects diagnostic groups.  A caller must construct an
    explicitly isolated pilot input if it wants to run only these groups; the
    normal support API always evaluates the complete support universe.
    """

    if count <= 0:
        raise ValueError("count must be positive.")
    by_size = sorted(
        records,
        key=lambda record: (record.selected_od_cells, record.group),
    )
    if not by_size:
        return ()
    if count == 1:
        return (by_size[len(by_size) // 2].group,)
    positions = [0, len(by_size) // 2, len(by_size) - 1]
    selected: list[int] = []
    for position in positions:
        group = by_size[position].group
        if group not in selected:
            selected.append(group)
    if count > len(selected):
        for record in by_size:
            if record.group not in selected:
                selected.append(record.group)
            if len(selected) == count:
                break
    return tuple(selected[:count])


def build_support_discovery_profile(
    records: tuple[GroupSupportTiming, ...] | list[GroupSupportTiming],
    *,
    metadata: Mapping[str, object] | None = None,
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, object]:
    """Build a deterministic JSON-compatible profile from timing records."""

    records = tuple(records)
    supplied_total = (metadata or {}).get("total_groups")
    total_groups = (
        len(records)
        if supplied_total is None
        else max(len(records), int(supplied_total))
    )
    durations = [max(0.0, float(record.total_seconds)) for record in records]
    selected_cells = sum(int(record.selected_od_cells) for record in records)
    origin_chunks = sum(int(record.origin_chunks) for record in records)
    elapsed = (
        max(0.0, float(elapsed_seconds))
        if elapsed_seconds is not None
        else float(sum(durations))
    )
    completed = len(records)
    cpu_allocation_value = (metadata or {}).get("cpu_allocation")
    cpu_allocation = (
        None
        if cpu_allocation_value is None
        else max(1.0, float(cpu_allocation_value))
    )
    cpu_seconds = sum(float(record.cpu_seconds or 0.0) for record in records)
    aggregate = {
        "completed_groups": completed,
        "total_groups": total_groups,
        "cached_groups": sum(record.cached for record in records),
        "rebuilt_groups": sum(not record.cached for record in records),
        "selected_od_cells": selected_cells,
        "free_od_cells": sum(int(record.free_od_cells) for record in records),
        "measurement_count": sum(int(record.measurement_count) for record in records),
        "origin_chunks": origin_chunks,
        "reachability_seconds": sum(float(record.reachability_seconds) for record in records),
        "projection_seconds": sum(float(record.projection_seconds) for record in records),
        "checkpoint_seconds": sum(float(record.checkpoint_seconds) for record in records),
        "group_seconds": sum(durations),
        "cpu_seconds": cpu_seconds,
        "cpu_utilization_fraction": (
            cpu_seconds / (elapsed * cpu_allocation)
            if cpu_allocation is not None and elapsed > 0.0
            else None
        ),
        "effective_parallelism": (
            cpu_seconds / elapsed
            if elapsed > 0.0
            else None
        ),
        "mean_group_seconds": mean(durations) if durations else None,
        "throughput_groups_per_second": completed / elapsed if elapsed > 0 else None,
        "throughput_selected_od_cells_per_second": (
            selected_cells / elapsed if elapsed > 0 else None
        ),
        "peak_rss_bytes": max(
            (record.peak_rss_bytes for record in records if record.peak_rss_bytes is not None),
            default=None,
        ),
        "worker_ids": sorted(
            {record.worker_id for record in records if record.worker_id is not None}
        ),
    }
    return {
        "schema_version": SUPPORT_DISCOVERY_PROFILE_SCHEMA_VERSION,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "elapsed_seconds": elapsed,
        "metadata": _json_safe(dict(metadata or {})),
        "aggregate": aggregate,
        "representative_groups": list(representative_support_groups(records)),
        "groups": [_json_safe(asdict(record)) for record in records],
    }


def write_support_discovery_profile(
    path: str | Path,
    profile: Mapping[str, object],
) -> Path:
    """Atomically write a profile below the caller-selected artifact root."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(profile, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def profile_support_discovery(
    operation: Callable[[GroupSupportTimingCallback], object],
    *,
    metadata: Mapping[str, object] | None = None,
    output_path: str | Path | None = None,
) -> tuple[object, dict[str, object]]:
    """Run an existing support operation while recording a durable profile."""

    recorder = SupportDiscoveryProfileRecorder.create()
    started_at = _utc_now()
    started = perf_counter()
    result = operation(recorder)
    profile = build_support_discovery_profile(
        recorder.records(),
        metadata=metadata,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        elapsed_seconds=perf_counter() - started,
    )
    if output_path is not None:
        write_support_discovery_profile(output_path, profile)
    return result, profile


def run_support_discovery_pilot(
    operation: Callable[[tuple[int, ...], GroupSupportTimingCallback], object],
    representative_groups: tuple[int, ...] | list[int],
    *,
    metadata: Mapping[str, object] | None = None,
    output_path: str | Path | None = None,
) -> tuple[object, dict[str, object]]:
    """Run an explicitly subset-aware, isolated support pilot.

    ``operation`` must construct a pilot input containing only the requested
    destination groups.  The normal production support evaluator is not
    subset-aware and must not be passed directly here.  The selected IDs and
    the explicit ``pilot`` marker are persisted so a partial result cannot be
    mistaken for a complete support artifact.
    """

    groups = tuple(dict.fromkeys(int(group) for group in representative_groups))
    if not groups:
        raise ValueError("representative_groups must not be empty.")
    pilot_metadata = dict(metadata or {})
    pilot_metadata.update(
        {
            "pilot": True,
            "selected_groups": list(groups),
            "support_artifact_complete": False,
        }
    )
    recorder = SupportDiscoveryProfileRecorder.create()
    started_at = _utc_now()
    started = perf_counter()
    result = operation(groups, recorder)
    profile = build_support_discovery_profile(
        recorder.records(),
        metadata=pilot_metadata,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        elapsed_seconds=perf_counter() - started,
    )
    if output_path is not None:
        write_support_discovery_profile(output_path, profile)
    return result, profile


def compare_support_profiles(
    reference: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, float | None]:
    """Return timing-only speedup diagnostics for two completed profiles."""

    reference_elapsed = float(reference.get("elapsed_seconds", 0.0) or 0.0)
    candidate_elapsed = float(candidate.get("elapsed_seconds", 0.0) or 0.0)
    speedup = (
        reference_elapsed / candidate_elapsed
        if candidate_elapsed > 0.0
        else None
    )
    return {
        "reference_elapsed_seconds": reference_elapsed,
        "candidate_elapsed_seconds": candidate_elapsed,
        "speedup": speedup,
    }
