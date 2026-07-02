from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.assignment.id_manager import (
    AssignmentIDManager,
    ODKey,
    _time_bin_index_from_record,
)
from public_transportation.assignment.jax_graph_types import JaxGraph


@dataclass(frozen=True)
class _TimeBin:
    bin_id: str


@dataclass(frozen=True)
class _DemandRecord:
    origin_stop_id: str | None = None
    dest_stop_id: str | None = None
    time_bin_index: int | None = None
    time_bin_id: str | None = None


def _mk_scenario(records, time_bins=None):
    if time_bins is None:
        time_bins = [_TimeBin("am"), _TimeBin("pm")]
    return SimpleNamespace(
        demand=SimpleNamespace(records=records),
        time_bins=time_bins,
    )


def _mk_graph(
    *,
    capacity=None,
    node_time_bin_index=None,
    node_stop_id=("A", "B", "C"),
    node_stop_name=("Stop A", "Stop B", "Stop C"),
    trip_id=("T1", "T2"),
    trip_line_ref=("L1", "L1"),
) -> JaxGraph:
    num_nodes = 4
    num_links = 3

    return JaxGraph(
        num_nodes=num_nodes,
        num_links=num_links,
        tail=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        head=jnp.asarray([1, 2, 3], dtype=jnp.int32),
        topo_order=jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        topo_order_rev=jnp.asarray([3, 2, 1, 0], dtype=jnp.int32),
        node_time=jnp.asarray([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32),
        node_stop_index=jnp.asarray([0, 1, 2, 2], dtype=jnp.int32),
        node_time_s=jnp.asarray([0, 60, 120, 180], dtype=jnp.int32),
        node_kind=jnp.asarray([0, 1, 1, 3], dtype=jnp.int32),
        node_trip_index=jnp.asarray([-1, 0, 1, -1], dtype=jnp.int32),
        out_start=jnp.asarray([0, 1, 2, 3, 3], dtype=jnp.int32),
        out_links_csr=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        out_links=jnp.asarray([[0], [1], [2], [-1]], dtype=jnp.int32),
        out_mask=jnp.asarray([[True], [True], [True], [False]], dtype=jnp.bool_),
        link_type=jnp.asarray([10, 20, 30], dtype=jnp.int32),
        travel_time=jnp.asarray([5.0, 6.0, 7.0], dtype=jnp.float32),
        capacity=capacity,
        link_trip_index=jnp.asarray([0, 1, -1], dtype=jnp.int32),
        node_time_bin_index=node_time_bin_index,
        node_stop_id=node_stop_id,
        node_stop_name=node_stop_name,
        trip_id=trip_id,
        trip_line_ref=trip_line_ref,
    )


def test_odkey_as_tuple():
    key = ODKey("A", "B", 2)

    assert key.as_tuple() == ("A", "B", 2)


def test_time_bin_index_from_record_prefers_explicit_index():
    record = _DemandRecord(time_bin_index=3, time_bin_id="ignored")

    assert _time_bin_index_from_record(record, bin_index_by_id={"ignored": 99}) == 3


def test_time_bin_index_from_record_resolves_id():
    record = _DemandRecord(time_bin_id="pm")

    assert _time_bin_index_from_record(record, bin_index_by_id={"am": 0, "pm": 1}) == 1


def test_time_bin_index_from_record_rejects_missing_index_and_id():
    record = _DemandRecord()

    with pytest.raises(ValueError, match="time_bin_index or time_bin_id"):
        _time_bin_index_from_record(record, bin_index_by_id={})


def test_time_bin_index_from_record_rejects_unknown_id():
    record = _DemandRecord(time_bin_id="night")

    with pytest.raises(ValueError, match="Unknown time_bin_id"):
        _time_bin_index_from_record(record, bin_index_by_id={"am": 0})


def test_build_rejects_missing_demand():
    scenario = SimpleNamespace(demand=None, time_bins=[_TimeBin("am")])

    with pytest.raises(ValueError, match="no demand"):
        AssignmentIDManager.build(scenario=scenario, graph=_mk_graph())


def test_build_rejects_missing_time_bins():
    scenario = SimpleNamespace(
        demand=SimpleNamespace(records=[_DemandRecord("A", "B", 0)]),
        time_bins=[],
    )

    with pytest.raises(ValueError, match="no time bins"):
        AssignmentIDManager.build(scenario=scenario, graph=_mk_graph())


def test_build_rejects_empty_demand_records():
    scenario = _mk_scenario([])

    with pytest.raises(ValueError, match="zero records"):
        AssignmentIDManager.build(scenario=scenario, graph=_mk_graph())


def test_build_rejects_duplicate_time_bin_ids():
    scenario = _mk_scenario(
        [_DemandRecord("A", "B", time_bin_id="am")],
        time_bins=[_TimeBin("am"), _TimeBin("am")],
    )

    with pytest.raises(ValueError, match="Duplicate time bin id"):
        AssignmentIDManager.build(scenario=scenario, graph=_mk_graph())


def test_build_rejects_missing_origin_or_destination():
    scenario = _mk_scenario([_DemandRecord(origin_stop_id="A", dest_stop_id=None, time_bin_index=0)])

    with pytest.raises(ValueError, match="origin_stop_id and dest_stop_id"):
        AssignmentIDManager.build(scenario=scenario, graph=_mk_graph())


def test_build_rejects_duplicate_od_keys():
    scenario = _mk_scenario(
        [
            _DemandRecord("A", "B", time_bin_index=0),
            _DemandRecord("A", "B", time_bin_index=0),
        ]
    )

    with pytest.raises(ValueError, match="Duplicate OD key"):
        AssignmentIDManager.build(scenario=scenario, graph=_mk_graph())


def test_build_constructs_scenario_and_canonical_od_conventions():
    scenario = _mk_scenario(
        [
            _DemandRecord("B", "C", time_bin_index=1),
            _DemandRecord("A", "C", time_bin_index=0),
            _DemandRecord("A", "B", time_bin_id="pm"),
        ]
    )

    manager = AssignmentIDManager.build(scenario=scenario, graph=_mk_graph())

    assert manager.num_nodes == 4
    assert manager.num_links == 3
    assert manager.num_od == 3

    assert [k.as_tuple() for k in manager.od_keys_scenario] == [
        ("B", "C", 1),
        ("A", "C", 0),
        ("A", "B", 1),
    ]

    assert [k.as_tuple() for k in manager.od_keys_canonical] == [
        ("A", "B", 1),
        ("A", "C", 0),
        ("B", "C", 1),
    ]

    assert np.array_equal(manager.perm_scenario_to_canonical, np.asarray([2, 1, 0], dtype=np.int32))
    assert np.array_equal(manager.perm_canonical_to_scenario, np.asarray([2, 1, 0], dtype=np.int32))

    assert manager.od_index_by_key_scenario[("B", "C", 1)] == 0
    assert manager.od_index_by_key_canonical[("A", "B", 1)] == 0


def test_od_value_permutations_round_trip():
    scenario = _mk_scenario(
        [
            _DemandRecord("B", "C", time_bin_index=1),
            _DemandRecord("A", "C", time_bin_index=0),
            _DemandRecord("A", "B", time_bin_index=1),
        ]
    )
    manager = AssignmentIDManager.build(scenario=scenario, graph=_mk_graph())

    scenario_values = np.asarray([10.0, 20.0, 30.0])
    canonical_values = manager.od_values_scenario_to_canonical(scenario_values)

    assert np.array_equal(canonical_values, np.asarray([30.0, 20.0, 10.0]))
    assert np.array_equal(manager.od_values_canonical_to_scenario(canonical_values), scenario_values)


def test_od_value_permutations_reject_wrong_shape():
    manager = AssignmentIDManager.build(
        scenario=_mk_scenario([_DemandRecord("A", "B", time_bin_index=0)]),
        graph=_mk_graph(),
    )

    with pytest.raises(ValueError, match="Expected od_values shape"):
        manager.od_values_scenario_to_canonical(np.asarray([1.0, 2.0]))

    with pytest.raises(ValueError, match="Expected od_values shape"):
        manager.od_values_canonical_to_scenario(np.asarray([1.0, 2.0]))


def test_find_od_helpers():
    scenario = _mk_scenario(
        [
            _DemandRecord("B", "C", time_bin_index=1),
            _DemandRecord("A", "B", time_bin_index=0),
        ]
    )
    manager = AssignmentIDManager.build(scenario=scenario, graph=_mk_graph())

    assert manager.find_od_scenario("B", "C", 1) == 0
    assert manager.find_od_scenario("A", "B", 0) == 1
    assert manager.find_od_canonical("A", "B", 0) == 0
    assert manager.find_od_canonical("B", "C", 1) == 1


def test_graph_arrays_are_copied_to_numpy_views():
    manager = AssignmentIDManager.build(
        scenario=_mk_scenario([_DemandRecord("A", "B", time_bin_index=0)]),
        graph=_mk_graph(capacity=jnp.asarray([100.0, 200.0, 300.0], dtype=jnp.float32)),
    )

    assert np.array_equal(manager.link_tail, np.asarray([0, 1, 2]))
    assert np.array_equal(manager.link_head, np.asarray([1, 2, 3]))
    assert np.array_equal(manager.link_type, np.asarray([10, 20, 30]))
    assert np.array_equal(manager.link_trip_index, np.asarray([0, 1, -1]))
    assert np.allclose(manager.link_travel_time, np.asarray([5.0, 6.0, 7.0]))
    assert np.allclose(manager.link_capacity, np.asarray([100.0, 200.0, 300.0]))

    assert np.array_equal(manager.node_kind, np.asarray([0, 1, 1, 3]))
    assert np.array_equal(manager.node_stop_index, np.asarray([0, 1, 2, 2]))
    assert np.array_equal(manager.node_time_s, np.asarray([0, 60, 120, 180]))
    assert np.array_equal(manager.node_trip_index, np.asarray([-1, 0, 1, -1]))


def test_capacity_defaults_to_infinity_when_absent():
    manager = AssignmentIDManager.build(
        scenario=_mk_scenario([_DemandRecord("A", "B", time_bin_index=0)]),
        graph=_mk_graph(capacity=None),
    )

    assert manager.link_capacity.shape == (3,)
    assert np.all(np.isinf(manager.link_capacity))


def test_optional_node_time_bin_index_is_preserved_when_present():
    node_time_bin_index = jnp.asarray([0, 0, 1, -1], dtype=jnp.int32)
    manager = AssignmentIDManager.build(
        scenario=_mk_scenario([_DemandRecord("A", "B", time_bin_index=0)]),
        graph=_mk_graph(node_time_bin_index=node_time_bin_index),
    )

    assert np.array_equal(manager.node_time_bin_index, np.asarray([0, 0, 1, -1]))


def test_optional_node_time_bin_index_is_none_when_absent():
    manager = AssignmentIDManager.build(
        scenario=_mk_scenario([_DemandRecord("A", "B", time_bin_index=0)]),
        graph=_mk_graph(node_time_bin_index=None),
    )

    assert manager.node_time_bin_index is None


def test_labels_and_lookup_maps_are_built():
    manager = AssignmentIDManager.build(
        scenario=_mk_scenario([_DemandRecord("A", "B", time_bin_index=0)]),
        graph=_mk_graph(),
    )

    assert manager.stop_id == ("A", "B", "C")
    assert manager.stop_name == ("Stop A", "Stop B", "Stop C")
    assert manager.trip_id == ("T1", "T2")
    assert manager.trip_line_ref == ("L1", "L1")

    assert manager.stop_index_by_id == {"A": 0, "B": 1, "C": 2}
    assert manager.trip_index_by_id == {"T1": 0, "T2": 1}
    assert manager.trip_indices_by_line_ref == {"L1": (0, 1)}


def test_duplicate_stop_ids_are_rejected():
    scenario = _mk_scenario([_DemandRecord("A", "B", time_bin_index=0)])
    graph = _mk_graph(node_stop_id=("A", "A", "C"))

    with pytest.raises(ValueError, match="Duplicate stop_id"):
        AssignmentIDManager.build(scenario=scenario, graph=graph)


def test_duplicate_trip_ids_are_rejected():
    scenario = _mk_scenario([_DemandRecord("A", "B", time_bin_index=0)])
    graph = _mk_graph(trip_id=("T1", "T1"))

    with pytest.raises(ValueError, match="Duplicate trip_id"):
        AssignmentIDManager.build(scenario=scenario, graph=graph)


def test_empty_trip_line_ref_is_ignored_in_line_lookup():
    manager = AssignmentIDManager.build(
        scenario=_mk_scenario([_DemandRecord("A", "B", time_bin_index=0)]),
        graph=_mk_graph(trip_line_ref=("L1", "")),
    )

    assert manager.trip_indices_by_line_ref == {"L1": (0,)}


def test_fingerprint_payload_and_fingerprint_are_stable_for_same_inputs():
    scenario = _mk_scenario([_DemandRecord("A", "B", time_bin_index=0)])
    graph = _mk_graph()

    m1 = AssignmentIDManager.build(scenario=scenario, graph=graph)
    m2 = AssignmentIDManager.build(scenario=scenario, graph=graph)

    assert m1.fingerprint == m2.fingerprint
    assert m1.fingerprint_payload_json == m2.fingerprint_payload_json
    assert m1.fingerprint_payload() == m2.fingerprint_payload()


def test_fingerprint_changes_when_link_convention_changes():
    scenario = _mk_scenario([_DemandRecord("A", "B", time_bin_index=0)])
    m1 = AssignmentIDManager.build(scenario=scenario, graph=_mk_graph())

    changed_graph = _mk_graph()
    changed_graph = changed_graph.__class__(
        **{
            **changed_graph.__dict__,
            "tail": jnp.asarray([0, 0, 2], dtype=jnp.int32),
        }
    ) if hasattr(changed_graph, "__dict__") else changed_graph

    # JaxGraph is a dataclass in the production code.
    from dataclasses import replace
    changed_graph = replace(_mk_graph(), tail=jnp.asarray([0, 0, 2], dtype=jnp.int32))

    m2 = AssignmentIDManager.build(scenario=scenario, graph=changed_graph)

    assert m1.fingerprint != m2.fingerprint


def test_fingerprint_payload_decodes_to_expected_keys():
    manager = AssignmentIDManager.build(
        scenario=_mk_scenario([_DemandRecord("A", "B", time_bin_index=0)]),
        graph=_mk_graph(),
    )

    payload = manager.fingerprint_payload()

    expected_keys = {
        "num_nodes",
        "num_links",
        "num_od",
        "od_keys_canonical",
        "link_tail_hash",
        "link_head_hash",
        "link_type_hash",
        "link_trip_index_hash",
        "node_kind_hash",
        "node_stop_index_hash",
        "node_time_s_hash",
        "node_trip_index_hash",
    }
    assert set(payload) == expected_keys


def test_diff_fingerprint_payloads_reports_missing_extra_and_scalar_differences():
    expected = {"a": 1, "b": 2}
    got = {"a": 1, "c": 3}

    diffs = AssignmentIDManager.diff_fingerprint_payloads(expected, got)

    assert any("Missing keys" in d and "b" in d for d in diffs)
    assert any("Extra keys" in d and "c" in d for d in diffs)


def test_diff_fingerprint_payloads_reports_list_length_difference():
    diffs = AssignmentIDManager.diff_fingerprint_payloads(
        {"od_keys_canonical": [["A", "B", 0]]},
        {"od_keys_canonical": [["A", "B", 0], ["B", "C", 1]]},
    )

    assert any("list lengths" in d for d in diffs)


def test_diff_fingerprint_payloads_reports_list_element_differences_with_limit():
    diffs = AssignmentIDManager.diff_fingerprint_payloads(
        {"od_keys_canonical": [["A", "B", 0], ["C", "D", 1], ["E", "F", 2]]},
        {"od_keys_canonical": [["A", "B", 9], ["C", "X", 1], ["E", "Y", 2]]},
        max_list_diffs=2,
    )

    assert any("index 0" in d for d in diffs)
    assert any("index 1" in d for d in diffs)
    assert any("showing first 2 diffs" in d for d in diffs)


def test_format_fingerprint_mismatch_without_payloads():
    msg = AssignmentIDManager.format_fingerprint_mismatch(
        expected_fingerprint="abc",
        got_fingerprint="def",
    )

    assert "Fingerprint mismatch" in msg
    assert "expected: abc" in msg
    assert "got:      def" in msg
    assert "No fingerprint payloads" in msg


def test_format_fingerprint_mismatch_with_payload_diff():
    msg = AssignmentIDManager.format_fingerprint_mismatch(
        expected_fingerprint="abc",
        got_fingerprint="def",
        expected_payload_json='{"a": 1, "items": [1, 2]}',
        got_payload_json='{"a": 2, "items": [1, 3]}',
    )

    assert "Fingerprint mismatch" in msg
    assert "Details:" in msg
    assert "Key 'a' differs" in msg
    assert "items" in msg


def test_format_fingerprint_mismatch_handles_invalid_json():
    msg = AssignmentIDManager.format_fingerprint_mismatch(
        expected_fingerprint="abc",
        got_fingerprint="def",
        expected_payload_json="{bad json",
        got_payload_json="{}",
    )

    assert "Failed to decode" in msg


def test_format_fingerprint_mismatch_handles_non_dict_payloads():
    msg = AssignmentIDManager.format_fingerprint_mismatch(
        expected_fingerprint="abc",
        got_fingerprint="def",
        expected_payload_json="[]",
        got_payload_json="[]",
    )

    assert "did not decode to dicts" in msg