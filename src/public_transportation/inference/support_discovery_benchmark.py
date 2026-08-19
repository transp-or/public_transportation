"""Controlled, isolated benchmarks for the fixed-routing support workload.

The benchmark accepts a case-owned operation factory instead of making any
assumptions about private scenario files.  Each worker count receives its own
artifact and checkpoint roots, so a benchmark cannot contaminate a production
run or accidentally compare a reused cache with a cold run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping

from .fixed_routing_origin_support import GroupSupportTimingCallback
from .support_discovery_profile import (
    build_support_discovery_profile,
    write_support_discovery_profile,
)


@dataclass(frozen=True, slots=True)
class SupportBenchmarkRecord:
    workers: int
    elapsed_seconds: float
    total_groups: int
    groups_per_second: float | None
    selected_od_cells_per_second: float | None
    cpu_seconds: float
    cpu_utilization_fraction: float | None
    peak_rss_bytes: int | None
    effective_parallelism: float | None
    speedup_vs_one_worker: float | None
    profile_path: str
    artifact_root: str
    checkpoint_root: str


@dataclass(frozen=True, slots=True)
class SupportBenchmarkResult:
    records: tuple[SupportBenchmarkRecord, ...]
    summary_path: Path


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def benchmark_support_discovery(
    operation_factory: Callable[
        [int, Path, Path, GroupSupportTimingCallback], object
    ],
    *,
    worker_counts: tuple[int, ...] = (1, 2, 4, 8),
    output_root: str | Path,
    metadata: Mapping[str, object] | None = None,
) -> SupportBenchmarkResult:
    """Run a support benchmark matrix with isolated per-run roots.

    The factory receives ``(workers, artifact_root, checkpoint_root,
    timing_callback)`` and must run the same scientific operation for every
    worker count.  It owns the case-specific construction of inputs and may
    select a pilot subset, but it must never pass a production root.
    """

    normalized_workers = tuple(dict.fromkeys(int(value) for value in worker_counts))
    if not normalized_workers or any(value <= 0 for value in normalized_workers):
        raise ValueError("worker_counts must contain positive integers.")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    records: list[SupportBenchmarkRecord] = []
    profiles: dict[int, dict[str, object]] = {}
    base_metadata = dict(metadata or {})
    for workers in normalized_workers:
        run_root = root / f"workers-{workers:02d}"
        artifact_root = run_root / "artifacts"
        checkpoint_root = run_root / "checkpoints"
        artifact_root.mkdir(parents=True, exist_ok=True)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        profile_path = run_root / "support_profile.json"
        started = perf_counter()
        result, profile = _run_one(
            operation_factory,
            workers=workers,
            artifact_root=artifact_root,
            checkpoint_root=checkpoint_root,
            metadata={
                **base_metadata,
                "workers": workers,
                "benchmark_root": str(run_root),
                "artifact_root": str(artifact_root),
                "checkpoint_root": str(checkpoint_root),
            },
            profile_path=profile_path,
        )
        del result
        elapsed = max(0.0, perf_counter() - started)
        profiles[workers] = profile
        aggregate = profile["aggregate"]
        total_groups = int(aggregate["total_groups"])
        records.append(
            SupportBenchmarkRecord(
                workers=workers,
                elapsed_seconds=elapsed,
                total_groups=total_groups,
                groups_per_second=(
                    total_groups / elapsed if elapsed > 0.0 else None
                ),
                selected_od_cells_per_second=(
                    int(aggregate["selected_od_cells"]) / elapsed
                    if elapsed > 0.0
                    else None
                ),
                cpu_seconds=float(aggregate["cpu_seconds"]),
                cpu_utilization_fraction=aggregate["cpu_utilization_fraction"],
                peak_rss_bytes=aggregate["peak_rss_bytes"],
                effective_parallelism=aggregate["effective_parallelism"],
                speedup_vs_one_worker=None,
                profile_path=str(profile_path),
                artifact_root=str(artifact_root),
                checkpoint_root=str(checkpoint_root),
            )
        )
    baseline = next(
        (record.elapsed_seconds for record in records if record.workers == 1), None
    )
    if baseline is not None and baseline > 0.0:
        records = [
            SupportBenchmarkRecord(
                **{
                    **asdict(record),
                    "speedup_vs_one_worker": baseline / record.elapsed_seconds
                    if record.elapsed_seconds > 0.0
                    else None,
                }
            )
            for record in records
        ]
    summary_path = root / "support_benchmark_summary.json"
    _atomic_json(
        summary_path,
        {
            "schema_version": 1,
            "metadata": dict(base_metadata),
            "isolated_roots": True,
            "records": [asdict(record) for record in records],
            "profiles": profiles,
        },
    )
    return SupportBenchmarkResult(tuple(records), summary_path)


def _run_one(
    operation_factory,
    *,
    workers: int,
    artifact_root: Path,
    checkpoint_root: Path,
    metadata: Mapping[str, object],
    profile_path: Path,
) -> tuple[object, dict[str, object]]:
    from .support_discovery_profile import SupportDiscoveryProfileRecorder

    recorder = SupportDiscoveryProfileRecorder.create()
    started = perf_counter()
    result = operation_factory(workers, artifact_root, checkpoint_root, recorder)
    profile = build_support_discovery_profile(
        recorder.records(),
        metadata=metadata,
        elapsed_seconds=perf_counter() - started,
    )
    write_support_discovery_profile(profile_path, profile)
    return result, profile


def project_support_duration(
    profile: Mapping[str, object], *, total_groups: int
) -> dict[str, object]:
    """Project a complete run from a pilot, with explicit uncertainty."""

    if total_groups <= 0:
        raise ValueError("total_groups must be positive.")
    groups = list(profile.get("groups", []))
    durations = sorted(
        float(item["total_seconds"])
        for item in groups
        if float(item.get("total_seconds", 0.0)) > 0.0
    )
    if not durations:
        return {
            "estimated_seconds": None,
            "lower_seconds": None,
            "upper_seconds": None,
            "confidence": "unavailable",
            "reason": "pilot contains no completed timed groups",
        }
    median = durations[len(durations) // 2]
    lower = durations[max(0, len(durations) // 10)]
    upper = durations[min(len(durations) - 1, (9 * len(durations)) // 10)]
    return {
        "estimated_seconds": float(total_groups * median),
        "lower_seconds": float(total_groups * lower),
        "upper_seconds": float(total_groups * upper),
        "confidence": "low" if len(durations) < 8 else "medium",
        "reason": "heterogeneous destination-group costs; pilot projection only",
    }
