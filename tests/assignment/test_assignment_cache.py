from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.assignment.assign import assign, prepare_assignment
from public_transportation.assignment.cache import (
    _COST_ARRAYS,
    _GRAPH_ARRAYS,
    _OD_ARRAYS,
    assignment_cache_path,
    assignment_cache_provenance,
)
from public_transportation.assignment.config import AssignmentConfig
from public_transportation.domain import Scenario
from public_transportation.inference.assignment_adapter import (
    assign_link_flow_fixed_routing,
    build_assignment_inputs,
    prepare_fixed_routing,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "docs/source/examples/simple_example_02"
NETWORK_FILES = (
    "metadata.json", "stops.csv", "lines.csv", "trips.csv", "stop_times.csv",
    "time_bins.csv",
)


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    directory = tmp_path_factory.mktemp("assignment-cache-scenario")
    for name in NETWORK_FILES:
        shutil.copy2(EXAMPLE / "data" / name, directory / name)
    shutil.copy2(EXAMPLE / "pre_processing/results/demand.csv", directory / "demand.csv")
    return Scenario.from_folder(directory, strict=True)


def _assert_artifacts_equal(left, right):
    assert left.graph.num_nodes == right.graph.num_nodes
    assert left.graph.num_links == right.graph.num_links
    assert left.od_groups.num_od == right.od_groups.num_od
    for name in _GRAPH_ARRAYS:
        np.testing.assert_array_equal(np.asarray(getattr(left.graph, name)), np.asarray(getattr(right.graph, name)))
    for name in _OD_ARRAYS:
        np.testing.assert_array_equal(np.asarray(getattr(left.od_groups, name)), np.asarray(getattr(right.od_groups, name)))
    for name in _COST_ARRAYS:
        np.testing.assert_array_equal(np.asarray(getattr(left.cost_parts, name)), np.asarray(getattr(right.cost_parts, name)))
    assert left.graph.node_stop_id == right.graph.node_stop_id
    assert left.graph.trip_id == right.graph.trip_id


def _rewrite(path: Path, mutate):
    with np.load(path, allow_pickle=False) as archive:
        content = {name: np.array(archive[name], copy=True) for name in archive.files}
    mutate(content)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **content)


def _cache_path(scenario, config, directory):
    _, key = assignment_cache_provenance(scenario=scenario, config=config)
    return assignment_cache_path(cache_directory=directory, cache_key=key)


def test_cache_round_trip_and_assignment_equivalence(scenario, tmp_path):
    config = AssignmentConfig()
    uncached = prepare_assignment(scenario, config, cache_policy="off")
    populated = prepare_assignment(scenario, config, cache_directory=tmp_path, cache_policy="auto")
    cached = prepare_assignment(scenario, config, cache_directory=tmp_path, cache_policy="auto")
    assert populated.cache_metrics.status == "miss"
    assert cached.cache_metrics.status == "hit"
    _assert_artifacts_equal(uncached, cached)

    demand = jnp.linspace(0.1, 2.0, cached.od_groups.num_od)
    dynamic_uncached = assign(demand, uncached, theta=1.0).link_flow
    dynamic_cached = assign(demand, cached, theta=1.0).link_flow
    np.testing.assert_allclose(dynamic_cached, dynamic_uncached, rtol=2e-6, atol=2e-6)
    uncached_inputs = build_assignment_inputs(artifacts=uncached)
    cached_inputs = build_assignment_inputs(artifacts=cached)
    uncached_routing = prepare_fixed_routing(inputs=uncached_inputs, theta=1.0)
    cached_routing = prepare_fixed_routing(inputs=cached_inputs, theta=1.0)
    fixed_uncached = assign_link_flow_fixed_routing(inputs=uncached_inputs, routing=uncached_routing, f=demand)
    fixed_cached = assign_link_flow_fixed_routing(inputs=cached_inputs, routing=cached_routing, f=demand)
    jax.block_until_ready((fixed_uncached, fixed_cached))
    np.testing.assert_allclose(fixed_cached, fixed_uncached, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("kind", ["truncated", "metadata", "array", "dtype"])
def test_corrupt_cache_rebuilds_safely(scenario, tmp_path, kind):
    config = AssignmentConfig()
    prepare_assignment(scenario, config, cache_directory=tmp_path, cache_policy="refresh")
    path = _cache_path(scenario, config, tmp_path)
    if kind == "truncated":
        path.write_bytes(b"truncated")
    elif kind == "metadata":
        def mutate(content):
            metadata = json.loads(str(content["metadata"].item()))
            metadata["schema_version"] = -1
            content["metadata"] = np.asarray(json.dumps(metadata))
        _rewrite(path, mutate)
    elif kind == "array":
        _rewrite(path, lambda content: content.__setitem__("graph__tail", content["graph__tail"][:-1]))
    else:
        _rewrite(path, lambda content: content.__setitem__("graph__tail", content["graph__tail"].astype(np.float32)))
    rebuilt = prepare_assignment(scenario, config, cache_directory=tmp_path, cache_policy="auto")
    assert rebuilt.cache_metrics.status == "invalid_rebuilt"
    loaded = prepare_assignment(scenario, config, cache_directory=tmp_path, cache_policy="readonly")
    assert loaded.cache_metrics.cache_hit


def test_package_version_mismatch_rebuilds(scenario, tmp_path):
    config = AssignmentConfig()
    prepare_assignment(scenario, config, cache_directory=tmp_path, cache_policy="auto")
    path = _cache_path(scenario, config, tmp_path)
    def mutate(content):
        metadata = json.loads(str(content["metadata"].item()))
        metadata["package_version"] = "incompatible"
        content["metadata"] = np.asarray(json.dumps(metadata))
    _rewrite(path, mutate)
    rebuilt = prepare_assignment(scenario, config, cache_directory=tmp_path, cache_policy="auto")
    assert rebuilt.cache_metrics.status == "invalid_rebuilt"


def test_scenario_and_configuration_change_cache_keys(scenario):
    _, original = assignment_cache_provenance(scenario=scenario, config=AssignmentConfig())
    _, changed_config = assignment_cache_provenance(scenario=scenario, config=AssignmentConfig(max_transfer_wait_min=7.0))
    assert changed_config != original
    previous = scenario.stops[0].name
    scenario.stops[0].name = previous + " changed"
    try:
        _, changed_scenario = assignment_cache_provenance(scenario=scenario, config=AssignmentConfig())
    finally:
        scenario.stops[0].name = previous
    assert changed_scenario != original


def test_readonly_miss_and_invalid_entry_fail_without_writing(scenario, tmp_path):
    config = AssignmentConfig()
    with pytest.raises(FileNotFoundError):
        prepare_assignment(scenario, config, cache_directory=tmp_path, cache_policy="readonly")
    prepare_assignment(scenario, config, cache_directory=tmp_path, cache_policy="auto")
    path = _cache_path(scenario, config, tmp_path)
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="Invalid read-only"):
        prepare_assignment(scenario, config, cache_directory=tmp_path, cache_policy="readonly")
    assert path.read_bytes() == b"invalid"


def test_disabled_cache_does_not_create_directory(scenario, tmp_path):
    directory = tmp_path / "unused"
    artifacts = prepare_assignment(scenario, AssignmentConfig(), cache_directory=directory, cache_policy="off")
    assert artifacts.cache_metrics.status == "bypass"
    assert not directory.exists()


def test_concurrent_writers_leave_valid_entry(scenario, tmp_path):
    config = AssignmentConfig()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: prepare_assignment(scenario, config, cache_directory=tmp_path, cache_policy="refresh"), range(2)))
    _assert_artifacts_equal(results[0], results[1])
    loaded = prepare_assignment(scenario, config, cache_directory=tmp_path, cache_policy="readonly")
    assert loaded.cache_metrics.cache_hit
