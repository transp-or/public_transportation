from __future__ import annotations

import json

import pytest

from public_transportation.inference.fixed_routing_origin_support import (
    GroupSupportTiming,
)
from public_transportation.inference.fixed_routing_sharded_builder import (
    ShardedConstructionConfig,
)
from public_transportation.inference.support_discovery_profile import (
    SUPPORT_DISCOVERY_PROFILE_SCHEMA_VERSION,
    build_support_discovery_profile,
    compare_support_profiles,
    profile_support_discovery,
    representative_support_groups,
    run_support_discovery_pilot,
)
from public_transportation.inference.support_discovery_benchmark import (
    benchmark_support_discovery,
    project_support_duration,
)


def _timing(group: int, cells: int, seconds: float, *, cached: bool = False):
    return GroupSupportTiming(
        group=group,
        selected_od_cells=cells,
        free_od_cells=cells,
        measurement_count=cells // 2,
        origin_chunks=max(1, cells // 4),
        reachability_seconds=seconds * 0.6,
        projection_seconds=seconds * 0.3,
        checkpoint_seconds=seconds * 0.1,
        total_seconds=seconds,
        cached=cached,
        worker_id=f"worker-{group % 2}",
        peak_rss_bytes=1000 + group,
    )


def test_support_profile_is_json_compatible_and_durable(tmp_path):
    records = (_timing(8, 80, 2.0), _timing(2, 20, 1.0, cached=True))
    profile = build_support_discovery_profile(
        records,
        metadata={"input_fingerprint": "abc", "root": tmp_path},
        elapsed_seconds=2.5,
    )
    assert profile["schema_version"] == SUPPORT_DISCOVERY_PROFILE_SCHEMA_VERSION
    assert profile["aggregate"]["completed_groups"] == 2
    assert profile["aggregate"]["cached_groups"] == 1
    assert profile["aggregate"]["peak_rss_bytes"] == 1008
    json.dumps(profile)

    result, persisted = profile_support_discovery(
        lambda callback: (callback(_timing(4, 40, 0.5)), "support-result")[1],
        metadata={"case": "pilot"},
        output_path=tmp_path / "support-profile.json",
    )
    assert result == "support-result"
    assert persisted["aggregate"]["completed_groups"] == 1
    loaded = json.loads((tmp_path / "support-profile.json").read_text())
    assert loaded["metadata"]["case"] == "pilot"


def test_representative_groups_are_deterministic_and_speedup_is_timing_only():
    records = (_timing(10, 100, 4.0), _timing(3, 10, 1.0), _timing(7, 50, 2.0))
    assert representative_support_groups(records) == (3, 7, 10)
    assert representative_support_groups(records, count=1) == (7,)
    reference = build_support_discovery_profile(records, elapsed_seconds=10.0)
    candidate = build_support_discovery_profile(records, elapsed_seconds=2.0)
    comparison = compare_support_profiles(reference, candidate)
    assert comparison["speedup"] == 5.0


def test_support_worker_setting_is_independent_and_defaults_for_compatibility():
    legacy = ShardedConstructionConfig(workers=4)
    explicit = ShardedConstructionConfig(workers=4, support_workers=2)
    assert legacy.support_workers == 4
    assert explicit.support_workers == 2
    assert legacy.execution_configuration()["shard_construction_workers_requested"] == 4
    assert explicit.execution_configuration()["support_workers_requested"] == 2
    with pytest.raises(ValueError, match="support_workers"):
        ShardedConstructionConfig(support_workers=1.5)


def test_pilot_is_explicitly_marked_incomplete_and_uses_selected_groups(tmp_path):
    seen: list[tuple[int, ...]] = []

    def operation(groups, callback):
        seen.append(groups)
        callback(_timing(groups[0], 12, 0.25))
        return "pilot-result"

    result, profile = run_support_discovery_pilot(
        operation,
        (8, 3, 8),
        metadata={"input_fingerprint": "pilot-input"},
        output_path=tmp_path / "pilot.json",
    )
    assert result == "pilot-result"
    assert seen == [(8, 3)]
    assert profile["metadata"]["pilot"] is True
    assert profile["metadata"]["support_artifact_complete"] is False
    assert json.loads((tmp_path / "pilot.json").read_text())["metadata"]["selected_groups"] == [8, 3]


def test_benchmark_matrix_uses_isolated_roots_and_projects_pilot(tmp_path):
    def operation(workers, artifact_root, checkpoint_root, callback):
        artifact_root.mkdir(parents=True, exist_ok=True)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        callback(_timing(workers, 10 + workers, 0.1))
        return workers

    result = benchmark_support_discovery(
        operation,
        worker_counts=(1, 2),
        output_root=tmp_path / "benchmark",
        metadata={"scenario_fingerprint": "scenario"},
    )
    assert [record.workers for record in result.records] == [1, 2]
    assert all(record.artifact_root != record.checkpoint_root for record in result.records)
    assert result.summary_path.is_file()
    projection = project_support_duration(
        json.loads(result.records[0].profile_path and (tmp_path / "benchmark/workers-01/support_profile.json").read_text()),
        total_groups=100,
    )
    assert projection["estimated_seconds"] is not None
