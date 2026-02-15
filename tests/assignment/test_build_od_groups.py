# tests/assignment/test_build_od_groups.py
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from public_transportation.assignment.build_od_groups import build_od_groups


# ---------------------------------------------------------------------------
# Minimal stubs (keep tests focused on build_od_groups behavior)
# ---------------------------------------------------------------------------

@dataclass
class _Stop:
    stop_id: str


@dataclass
class _TimeBin:
    # build_od_groups expects start_time/end_time in seconds
    start_time: int
    end_time: int


@dataclass
class _ODRecord:
    origin_stop_id: str
    dest_stop_id: str
    time_bin_id: int | None = None
    time_bin_index: int | None = None


@dataclass
class _Demand:
    records: list[_ODRecord]


def _mk_scenario(
    *,
    stops: list[str] | dict[str, object],
    time_bins: list[tuple[int, int]],
    records: list[_ODRecord] | None,
):
    if isinstance(stops, dict):
        stops_obj = stops
    else:
        stops_obj = [_Stop(s) for s in stops]

    time_bins_obj = [_TimeBin(a, b) for a, b in time_bins]
    demand_obj = None if records is None else _Demand(records=records)

    return SimpleNamespace(stops=stops_obj, time_bins=time_bins_obj, demand=demand_obj)


def _as_np(x) -> np.ndarray:
    # jax arrays -> numpy for assertions
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_build_od_groups_shapes_and_bounds_minutes():
    # Stops are intentionally unsorted to test the "sorted stop_ids" convention
    scenario = _mk_scenario(
        stops=["B", "A", "C"],  # sorted => A:0, B:1, C:2
        time_bins=[(8 * 3600, 8 * 3600 + 900), (9 * 3600, 9 * 3600 + 900)],  # 15min bins
        records=[
            _ODRecord("B", "C", time_bin_id=0),
            _ODRecord("A", "C", time_bin_id=0),
            _ODRecord("A", "B", time_bin_id=1),
            _ODRecord("B", "C", time_bin_id=0),
        ],
    )

    g = build_od_groups(scenario)

    assert g.num_od == 4

    od_origin = _as_np(g.od_origin_node)
    od_dest = _as_np(g.od_dest_node)
    od_a = _as_np(g.od_a_min)
    od_b = _as_np(g.od_b_min)

    # centroid indices from sorted stop_ids: A=0, B=1, C=2
    assert np.array_equal(od_origin, np.array([1, 0, 0, 1], dtype=int))
    assert np.array_equal(od_dest, np.array([2, 2, 1, 2], dtype=int))

    # bin0: [480, 495], bin1: [540, 555] in minutes
    assert np.allclose(od_a, np.array([480.0, 480.0, 540.0, 480.0]))
    assert np.allclose(od_b, np.array([495.0, 495.0, 555.0, 495.0]))

    # group arrays are well-formed CSR-like pointers
    group_start = _as_np(g.group_start)
    group_od_index = _as_np(g.group_od_index)
    assert group_start.ndim == 1
    assert group_start[0] == 0
    assert group_start[-1] == g.num_od
    assert np.all(group_start[1:] >= group_start[:-1])
    assert group_od_index.shape == (g.num_od,)

    # group_od_index must be a permutation of [0..num_od-1]
    assert np.array_equal(np.sort(group_od_index), np.arange(g.num_od))


def test_build_od_groups_grouping_is_by_destination_then_time_bin_and_is_deterministic():
    scenario = _mk_scenario(
        stops=["B", "A", "C"],  # sorted => A:0, B:1, C:2
        time_bins=[(0, 600), (600, 1200)],  # minutes: [0,10], [10,20]
        records=[
            _ODRecord("A", "C", time_bin_id=0),  # dest 2, bin 0
            _ODRecord("B", "C", time_bin_id=0),  # dest 2, bin 0
            _ODRecord("A", "B", time_bin_id=1),  # dest 1, bin 1
            _ODRecord("B", "C", time_bin_id=1),  # dest 2, bin 1
            _ODRecord("A", "B", time_bin_id=1),  # dest 1, bin 1
        ],
    )

    g1 = build_od_groups(scenario)
    g2 = build_od_groups(scenario)

    # determinism: exact same arrays
    assert np.array_equal(_as_np(g1.group_start), _as_np(g2.group_start))
    assert np.array_equal(_as_np(g1.group_dest_node), _as_np(g2.group_dest_node))
    assert np.array_equal(_as_np(g1.group_time_bin), _as_np(g2.group_time_bin))
    assert np.array_equal(_as_np(g1.group_od_index), _as_np(g2.group_od_index))

    # Expected grouping keys (dest_node, time_bin):
    # records keys: (2,0), (2,0), (1,1), (2,1), (1,1)
    # lexsort by (dest, time) -> (1,1) group first, then (2,0), then (2,1)
    group_dest = _as_np(g1.group_dest_node)
    group_tb = _as_np(g1.group_time_bin)
    assert list(zip(group_dest.tolist(), group_tb.tolist())) == [(1, 1), (2, 0), (2, 1)]

    # Check that each group slice actually contains only matching keys
    od_dest = _as_np(g1.od_dest_node)
    # we stored tb per-od only internally in builder; reconstruct from desired bounds:
    # easier: infer from od_a_min/od_b_min corresponding to bin0 or bin1
    od_a = _as_np(g1.od_a_min)
    # bin0 -> a=0, bin1 -> a=10
    od_tb = np.where(np.isclose(od_a, 0.0), 0, 1)

    starts = _as_np(g1.group_start)
    perm = _as_np(g1.group_od_index)

    for gi in range(len(starts) - 1):
        sl = perm[starts[gi]:starts[gi + 1]]
        assert sl.size > 0
        assert np.all(od_dest[sl] == group_dest[gi])
        assert np.all(od_tb[sl] == group_tb[gi])


def test_build_od_groups_accepts_stops_as_dict_keys():
    # build_od_groups allows scenario.stops to be a dict; it uses sorted(keys)
    scenario = _mk_scenario(
        stops={"S2": object(), "S1": object(), "S3": object()},  # sorted => S1:0, S2:1, S3:2
        time_bins=[(0, 60)],
        records=[
            _ODRecord("S2", "S3", time_bin_id=0),
            _ODRecord("S1", "S2", time_bin_id=0),
        ],
    )

    g = build_od_groups(scenario)
    assert np.array_equal(_as_np(g.od_origin_node), np.array([1, 0], dtype=int))
    assert np.array_equal(_as_np(g.od_dest_node), np.array([2, 1], dtype=int))


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_build_od_groups_requires_demand():
    scenario = _mk_scenario(stops=["A"], time_bins=[(0, 60)], records=None)
    with pytest.raises(ValueError, match="Scenario has no demand"):
        build_od_groups(scenario)


def test_build_od_groups_requires_time_bins():
    scenario = _mk_scenario(stops=["A"], time_bins=[], records=[_ODRecord("A", "A", time_bin_id=0)])
    with pytest.raises(ValueError, match="Scenario has no time bins"):
        build_od_groups(scenario)


def test_build_od_groups_rejects_zero_records():
    scenario = _mk_scenario(stops=["A"], time_bins=[(0, 60)], records=[])
    with pytest.raises(ValueError, match="Demand has zero records"):
        build_od_groups(scenario)


def test_build_od_groups_rejects_missing_time_bin_fields():
    scenario = _mk_scenario(
        stops=["A", "B"],
        time_bins=[(0, 60)],
        records=[_ODRecord("A", "B", time_bin_id=None, time_bin_index=None)],
    )
    with pytest.raises(ValueError, match="missing time_bin_index/time_bin_id"):
        build_od_groups(scenario)


def test_build_od_groups_rejects_unknown_origin_stop():
    scenario = _mk_scenario(
        stops=["A", "B"],
        time_bins=[(0, 60)],
        records=[_ODRecord("X", "B", time_bin_id=0)],
    )
    with pytest.raises(ValueError, match=r"Unknown origin_stop_id"):
        build_od_groups(scenario)


def test_build_od_groups_rejects_unknown_dest_stop():
    scenario = _mk_scenario(
        stops=["A", "B"],
        time_bins=[(0, 60)],
        records=[_ODRecord("A", "X", time_bin_id=0)],
    )
    with pytest.raises(ValueError, match=r"Unknown dest_stop_id"):
        build_od_groups(scenario)


def test_build_od_groups_rejects_unknown_time_bin_index():
    scenario = _mk_scenario(
        stops=["A", "B"],
        time_bins=[(0, 60)],  # only bin 0 exists
        records=[_ODRecord("A", "B", time_bin_id=999)],
    )
    with pytest.raises(ValueError, match=r"Unknown time bin index"):
        build_od_groups(scenario)