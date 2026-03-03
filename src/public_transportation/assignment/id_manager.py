from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, TYPE_CHECKING, Mapping

import hashlib
import json

import numpy as np

from .jax_graph_types import JaxGraph

if TYPE_CHECKING:  # pragma: no cover
    from public_transportation.domain.scenario import Scenario


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def _time_bin_index_from_record(record: Any, *, bin_index_by_id: dict[str, int]) -> int:
    """Resolve a demand record's time bin to an integer index."""
    tb_index = getattr(record, "time_bin_index", None)
    if tb_index is not None:
        return int(tb_index)

    tb_id = getattr(record, "time_bin_id", None)
    if tb_id is None:
        raise ValueError("Demand record must have time_bin_index or time_bin_id.")

    tb_id = _as_str(tb_id)
    try:
        return int(bin_index_by_id[tb_id])
    except KeyError as e:
        raise ValueError(f"Unknown time_bin_id in demand record: {tb_id}") from e


@dataclass(frozen=True, slots=True)
class ODKey:
    """Canonical identifier for one OD cell (one demand record)."""
    origin_stop_id: str
    dest_stop_id: str
    time_bin_index: int

    def as_tuple(self) -> tuple[str, str, int]:
        return (self.origin_stop_id, self.dest_stop_id, int(self.time_bin_index))


@dataclass(frozen=True, slots=True)
class AssignmentIDManager:
    """Freeze OD and link index conventions for one built assignment graph.

    This object is built ONCE (Python/NumPy) and then used to:
      - interpret input OD vectors (scenario order vs canonical order),
      - interpret output link flow vectors (graph link order),
      - provide rich metadata for later observation mappers.

    Conventions
    ----------
    Link index:
      - link i corresponds to the i-th entry of graph.tail/head/link_type/...

    OD index:
      - assignment consumes OD vectors in *scenario order* (the iteration order of scenario.demand.records).
      - a separate *canonical order* (lexicographic by (origin_stop_id, dest_stop_id, time_bin_index)) is provided
        for stable external referencing (e.g., observation mappers, debugging, likelihood assembly).
      - permutations are provided to move between the two.
    """

    # -------------------------
    # Core sizes
    # -------------------------
    num_nodes: int
    num_links: int
    num_od: int

    # -------------------------
    # OD conventions + mappings
    # -------------------------
    od_keys_scenario: tuple[ODKey, ...]
    od_keys_canonical: tuple[ODKey, ...]
    # Permutations mapping OD vectors between conventions.
    # Scenario -> canonical: od_values_canonical = od_values_scenario[perm_scenario_to_canonical]
    perm_scenario_to_canonical: np.ndarray  # (num_od,)
    # Canonical -> scenario: od_values_scenario = od_values_canonical[perm_canonical_to_scenario]
    perm_canonical_to_scenario: np.ndarray  # (num_od,)

    # Fast lookup (Python-side) for later tools/mappers
    od_index_by_key_canonical: dict[tuple[str, str, int], int]
    od_index_by_key_scenario: dict[tuple[str, str, int], int]

    # -------------------------
    # Link metadata views (NumPy)
    # -------------------------
    link_tail: np.ndarray          # (num_links,)
    link_head: np.ndarray          # (num_links,)
    link_type: np.ndarray          # (num_links,)
    link_trip_index: np.ndarray    # (num_links,)
    link_travel_time: np.ndarray   # (num_links,)
    link_capacity: np.ndarray      # (num_links,) if present else +inf

    # -------------------------
    # Node metadata views (NumPy)
    # -------------------------
    node_kind: np.ndarray          # (num_nodes,)
    node_stop_index: np.ndarray    # (num_nodes,)
    node_time_s: np.ndarray        # (num_nodes,)
    node_trip_index: np.ndarray    # (num_nodes,)

    # Optional node time-bin info if present in graph (useful for aggregation later)
    node_time_bin_index: np.ndarray | None

    # -------------------------
    # Python-side labels (optional)
    # -------------------------
    stop_id: tuple[str, ...]       # aligned with stop_index
    stop_name: tuple[str, ...]     # aligned with stop_id (may be empty)
    trip_id: tuple[str, ...]       # aligned with trip_index (may be empty)
    trip_line_ref: tuple[str, ...] # aligned with trip_index (may be empty)

    # -------------------------
    # Convenience lookup maps (Python-side)
    # -------------------------
    stop_index_by_id: dict[str, int]
    trip_index_by_id: dict[str, int]
    trip_indices_by_line_ref: dict[str, tuple[int, ...]]

    # -------------------------
    # Fingerprint (detect mismatches)
    # -------------------------
    # Exact payload (JSON) used to compute `fingerprint`.
    # This is stored to support detailed mismatch diagnostics.
    fingerprint_payload_json: str
    fingerprint: str

    @staticmethod
    def build(*, scenario: Scenario, graph: JaxGraph) -> "AssignmentIDManager":
        if scenario.demand is None:
            raise ValueError("Scenario has no demand.")
        if scenario.time_bins is None or len(scenario.time_bins) == 0:
            raise ValueError("Scenario has no time bins.")

        records = list(getattr(scenario.demand, "records"))
        if len(records) == 0:
            raise ValueError("Demand has zero records.")

        # --- Time-bin id -> index lookup (for records using time_bin_id)
        bin_index_by_id: dict[str, int] = {}
        for idx, tb in enumerate(scenario.time_bins):
            tb_id = getattr(tb, "bin_id", None)
            if tb_id is None:
                continue
            key = _as_str(tb_id)
            if key in bin_index_by_id:
                raise ValueError(f"Duplicate time bin id in scenario.time_bins: {key}")
            bin_index_by_id[key] = int(idx)

        # --- OD keys in scenario order
        od_keys_scen: list[ODKey] = []
        for r in records:
            o = _as_str(getattr(r, "origin_stop_id", None))
            d = _as_str(getattr(r, "dest_stop_id", None))
            if not o or not d:
                raise ValueError("Demand record must define origin_stop_id and dest_stop_id.")
            tb_idx = _time_bin_index_from_record(r, bin_index_by_id=bin_index_by_id)
            od_keys_scen.append(ODKey(o, d, tb_idx))

        num_od = len(od_keys_scen)

        # --- Canonical order: stable lexicographic sort
        # We build the permutation explicitly using Python sorting for clarity and determinism.
        sorted_pairs = sorted(enumerate(od_keys_scen), key=lambda p: p[1].as_tuple())
        perm_scen_to_canon = np.array([idx for idx, _ in sorted_pairs], dtype=np.int32)

        # Inverse permutation (canonical -> scenario)
        perm_canon_to_scen = np.empty((num_od,), dtype=np.int32)
        perm_canon_to_scen[perm_scen_to_canon] = np.arange(num_od, dtype=np.int32)

        od_keys_canon = tuple(od_keys_scen[i] for i in perm_scen_to_canon.tolist())

        # --- Build lookup dicts (canonical and scenario)
        od_idx_by_key_scen: dict[tuple[str, str, int], int] = {}
        for i, k in enumerate(od_keys_scen):
            kk = k.as_tuple()
            # allow duplicates? usually not desirable; raise to keep convention unambiguous
            if kk in od_idx_by_key_scen:
                raise ValueError(f"Duplicate OD key in scenario order: {kk}")
            od_idx_by_key_scen[kk] = i

        od_idx_by_key_canon: dict[tuple[str, str, int], int] = {}
        for i, k in enumerate(od_keys_canon):
            kk = k.as_tuple()
            if kk in od_idx_by_key_canon:
                raise ValueError(f"Duplicate OD key in canonical order: {kk}")
            od_idx_by_key_canon[kk] = i

        # --- Extract graph arrays (NumPy views)
        num_nodes = int(graph.num_nodes)
        num_links = int(graph.num_links)

        link_tail = np.asarray(graph.tail)
        link_head = np.asarray(graph.head)
        link_type = np.asarray(graph.link_type)
        link_trip_index = np.asarray(graph.link_trip_index)
        link_travel_time = np.asarray(graph.travel_time)

        # capacity optional-ish; if absent, set +inf
        cap = getattr(graph, "capacity", None)
        if cap is None:
            link_capacity = np.full((num_links,), np.inf, dtype=float)
        else:
            link_capacity = np.asarray(cap)

        node_kind = np.asarray(graph.node_kind)
        node_stop_index = np.asarray(graph.node_stop_index)
        node_time_s = np.asarray(graph.node_time_s)
        node_trip_index = np.asarray(graph.node_trip_index)

        tbi = getattr(graph, "node_time_bin_index", None)
        node_time_bin_index = None if tbi is None else np.asarray(tbi)

        stop_id = tuple(getattr(graph, "node_stop_id", ()))
        stop_name = tuple(getattr(graph, "node_stop_name", ()))
        trip_id = tuple(getattr(graph, "trip_id", ()))
        trip_line_ref = tuple(getattr(graph, "trip_line_ref", ()))

        # --- Convenience lookup dicts
        stop_index_by_id: dict[str, int] = {}
        for i, sid in enumerate(stop_id):
            k = _as_str(sid)
            if k in stop_index_by_id:
                raise ValueError(f"Duplicate stop_id in graph.node_stop_id: {k}")
            stop_index_by_id[k] = int(i)

        trip_index_by_id: dict[str, int] = {}
        for i, tid in enumerate(trip_id):
            k = _as_str(tid)
            if k in trip_index_by_id:
                raise ValueError(f"Duplicate trip_id in graph.trip_id: {k}")
            trip_index_by_id[k] = int(i)

        trip_indices_by_line_ref: dict[str, list[int]] = {}
        for i, lr in enumerate(trip_line_ref):
            k = _as_str(lr)
            if not k:
                continue
            trip_indices_by_line_ref.setdefault(k, []).append(int(i))
        trip_indices_by_line_ref_final: dict[str, tuple[int, ...]] = {
            k: tuple(v) for k, v in trip_indices_by_line_ref.items()
        }

        # --- Fingerprint: detect mismatch between ID manager and later runs
        fp_payload = {
            "num_nodes": num_nodes,
            "num_links": num_links,
            "num_od": num_od,
            # OD canonical keys define the OD convention
            "od_keys_canonical": [k.as_tuple() for k in od_keys_canon],
            # link convention: enough to detect rebuild mismatch
            "link_tail_hash": hashlib.sha256(link_tail.tobytes()).hexdigest(),
            "link_head_hash": hashlib.sha256(link_head.tobytes()).hexdigest(),
            "link_type_hash": hashlib.sha256(link_type.tobytes()).hexdigest(),
            "link_trip_index_hash": hashlib.sha256(link_trip_index.tobytes()).hexdigest(),
            "node_kind_hash": hashlib.sha256(node_kind.tobytes()).hexdigest(),
            "node_stop_index_hash": hashlib.sha256(node_stop_index.tobytes()).hexdigest(),
            "node_time_s_hash": hashlib.sha256(node_time_s.tobytes()).hexdigest(),
            "node_trip_index_hash": hashlib.sha256(node_trip_index.tobytes()).hexdigest(),
        }
        fp_payload_json = json.dumps(fp_payload, sort_keys=True)
        fingerprint = hashlib.sha256(fp_payload_json.encode("utf-8")).hexdigest()

        return AssignmentIDManager(
            num_nodes=num_nodes,
            num_links=num_links,
            num_od=num_od,
            od_keys_scenario=tuple(od_keys_scen),
            od_keys_canonical=od_keys_canon,
            perm_scenario_to_canonical=perm_scen_to_canon,
            perm_canonical_to_scenario=perm_canon_to_scen,
            od_index_by_key_canonical=od_idx_by_key_canon,
            od_index_by_key_scenario=od_idx_by_key_scen,
            link_tail=link_tail,
            link_head=link_head,
            link_type=link_type,
            link_trip_index=link_trip_index,
            link_travel_time=link_travel_time,
            link_capacity=link_capacity,
            node_kind=node_kind,
            node_stop_index=node_stop_index,
            node_time_s=node_time_s,
            node_trip_index=node_trip_index,
            node_time_bin_index=node_time_bin_index,
            stop_id=stop_id,
            stop_name=stop_name,
            trip_id=trip_id,
            trip_line_ref=trip_line_ref,
            stop_index_by_id=stop_index_by_id,
            trip_index_by_id=trip_index_by_id,
            trip_indices_by_line_ref=trip_indices_by_line_ref_final,
            fingerprint_payload_json=fp_payload_json,
            fingerprint=fingerprint,
        )

    # -------------------------
    # Fingerprint diagnostics
    # -------------------------

    def fingerprint_payload(self) -> dict[str, Any]:
        """Return the exact payload that was hashed to obtain `fingerprint`.

        This is intentionally the *exact* payload used in `build()` (no extra keys),
        so that two payloads can be diffed field-by-field when a mismatch occurs.
        """
        try:
            obj = json.loads(self.fingerprint_payload_json)
        except Exception as e:  # pragma: no cover
            raise ValueError("Invalid fingerprint_payload_json stored in AssignmentIDManager") from e
        if not isinstance(obj, dict):
            raise ValueError("fingerprint_payload_json must decode to a dict")
        return obj

    @staticmethod
    def diff_fingerprint_payloads(
        expected: Mapping[str, Any],
        got: Mapping[str, Any],
        *,
        max_list_diffs: int = 5,
    ) -> list[str]:
        """Compute a human-readable diff between two fingerprint payload dicts.

        Returns a list of short strings describing differences.
        """
        diffs: list[str] = []

        exp_keys = set(expected.keys())
        got_keys = set(got.keys())

        missing = sorted(exp_keys - got_keys)
        extra = sorted(got_keys - exp_keys)
        if missing:
            diffs.append(f"Missing keys in got payload: {missing}")
        if extra:
            diffs.append(f"Extra keys in got payload: {extra}")

        common = sorted(exp_keys & got_keys)
        for k in common:
            a = expected.get(k)
            b = got.get(k)
            if a == b:
                continue

            # Special handling for long lists (like od_keys_canonical)
            if isinstance(a, list) and isinstance(b, list):
                if len(a) != len(b):
                    diffs.append(f"Key '{k}' differs: list lengths {len(a)} vs {len(b)}")
                    continue

                # Find first few differing indices
                idxs: list[int] = []
                for i, (ai, bi) in enumerate(zip(a, b)):
                    if ai != bi:
                        idxs.append(i)
                        if len(idxs) >= int(max_list_diffs):
                            break

                if not idxs:
                    # lists compare as different but no element differs? unlikely, but keep safe
                    diffs.append(f"Key '{k}' differs (lists)")
                    continue

                for i in idxs:
                    diffs.append(f"Key '{k}' differs at index {i}: expected={a[i]!r}, got={b[i]!r}")
                if len(idxs) >= int(max_list_diffs):
                    diffs.append(f"Key '{k}': showing first {max_list_diffs} diffs")
                continue

            # Default scalar / object comparison
            diffs.append(f"Key '{k}' differs: expected={a!r}, got={b!r}")

        return diffs

    @staticmethod
    def format_fingerprint_mismatch(
        *,
        expected_fingerprint: str,
        got_fingerprint: str,
        expected_payload_json: str | None = None,
        got_payload_json: str | None = None,
        max_list_diffs: int = 5,
    ) -> str:
        """Build an informative multi-line message for fingerprint mismatches."""
        lines: list[str] = []
        lines.append("Fingerprint mismatch between results and current assignment indexing.")
        lines.append(f"expected: {expected_fingerprint}")
        lines.append(f"got:      {got_fingerprint}")

        if expected_payload_json is None or got_payload_json is None:
            lines.append("(No fingerprint payloads available for detailed diff.)")
            return "\n".join(lines)

        try:
            exp = json.loads(expected_payload_json)
            got = json.loads(got_payload_json)
        except Exception:
            lines.append("(Failed to decode fingerprint payload JSON for diff.)")
            return "\n".join(lines)

        if not isinstance(exp, dict) or not isinstance(got, dict):
            lines.append("(Fingerprint payload JSON did not decode to dicts.)")
            return "\n".join(lines)

        diffs = AssignmentIDManager.diff_fingerprint_payloads(exp, got, max_list_diffs=max_list_diffs)
        if diffs:
            lines.append("Details:")
            for d in diffs:
                lines.append(f"- {d}")
        else:
            lines.append("Details: payloads appear identical, but fingerprints differ (unexpected).")

        return "\n".join(lines)

    # -------------------------
    # Convenience helpers
    # -------------------------

    def od_values_scenario_to_canonical(self, od_values_scenario: np.ndarray) -> np.ndarray:
        od_values_scenario = np.asarray(od_values_scenario)
        if od_values_scenario.shape != (self.num_od,):
            raise ValueError(f"Expected od_values shape {(self.num_od,)}, got {od_values_scenario.shape}.")
        return od_values_scenario[self.perm_scenario_to_canonical]

    def od_values_canonical_to_scenario(self, od_values_canonical: np.ndarray) -> np.ndarray:
        od_values_canonical = np.asarray(od_values_canonical)
        if od_values_canonical.shape != (self.num_od,):
            raise ValueError(f"Expected od_values shape {(self.num_od,)}, got {od_values_canonical.shape}.")
        return od_values_canonical[self.perm_canonical_to_scenario]

    def find_od_canonical(self, origin_stop_id: str, dest_stop_id: str, time_bin_index: int) -> int:
        key = (_as_str(origin_stop_id), _as_str(dest_stop_id), int(time_bin_index))
        return self.od_index_by_key_canonical[key]

    def find_od_scenario(self, origin_stop_id: str, dest_stop_id: str, time_bin_index: int) -> int:
        key = (_as_str(origin_stop_id), _as_str(dest_stop_id), int(time_bin_index))
        return self.od_index_by_key_scenario[key]