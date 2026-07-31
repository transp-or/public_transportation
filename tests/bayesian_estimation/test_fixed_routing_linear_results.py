from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.fixed_routing_linear_quality import (
    analyze_linear_estimate_quality,
)
from public_transportation.inference.fixed_routing_linear_regularization import (
    ridge_to_prior,
)
from public_transportation.inference.fixed_routing_linear_results import (
    FIXED_ROUTING_LINEAR_RESULT_SCHEMA_VERSION,
    build_fixed_routing_linear_result,
    load_fixed_routing_linear_result,
    save_fixed_routing_linear_result,
)
from public_transportation.inference.fixed_routing_linear_trf_solver import (
    solve_trf_lsmr,
)


def _result():
    prior = np.array([2.0, 3.0])
    problem = FixedRoutingLinearProblem(
        measurement_operator=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        fixed_measurement_offset=np.array([0.5, 0.0, 1.0]),
        observations=np.array([2.5, 4.0, 6.0]),
        observation_weights=np.array([1.0, 2.0, 0.5]),
        prior_demand=prior,
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 2.0),
        regularization_selection="configured",
        regularization_blocks=(ridge_to_prior(prior, strength=0.25),),
        variable_scales=np.array([2.0, 4.0]),
        free_od_indices=np.array([7, 3]),
    )
    solved = solve_trf_lsmr(problem)
    quality = analyze_linear_estimate_quality(problem, solved.demand)
    return build_fixed_routing_linear_result(problem, solved, quality)


def _rewrite(path, **updates):
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload.update(updates)
    np.savez_compressed(path, **payload)


def test_result_round_trip_preserves_ordering_and_is_immutable(tmp_path):
    original = _result()
    path = save_fixed_routing_linear_result(
        original, tmp_path / "nested" / "result.npz"
    )
    loaded = load_fixed_routing_linear_result(
        path,
        expected_od_layout_fingerprint="od",
        expected_assignment_fingerprint="assignment",
        expected_mapping_fingerprint="mapping",
    )

    assert loaded.schema_version == FIXED_ROUTING_LINEAR_RESULT_SCHEMA_VERSION
    assert loaded.regularization_names == ("ridge_to_prior",)
    assert loaded.classifications == original.classifications
    np.testing.assert_array_equal(loaded.free_od_indices, [7, 3])
    np.testing.assert_allclose(loaded.estimated_demand, original.estimated_demand)
    with pytest.raises(ValueError, match="read-only"):
        loaded.estimated_demand[0] = 0.0


def test_load_rejects_missing_field(tmp_path):
    path = tmp_path / "result.npz"
    save_fixed_routing_linear_result(_result(), path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files if name != "gradient"}
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="missing fields.*gradient"):
        load_fixed_routing_linear_result(path)


def test_load_rejects_schema_and_provenance_mismatches(tmp_path):
    path = save_fixed_routing_linear_result(_result(), tmp_path / "result.npz")
    with pytest.raises(ValueError, match="mapping_fingerprint mismatch"):
        load_fixed_routing_linear_result(path, expected_mapping_fingerprint="other")

    _rewrite(path, schema_version=np.asarray(999))
    with pytest.raises(ValueError, match="unsupported result schema version"):
        load_fixed_routing_linear_result(path)


def test_load_rejects_inconsistent_objective_and_shape(tmp_path):
    path = save_fixed_routing_linear_result(_result(), tmp_path / "result.npz")
    _rewrite(path, objective=np.asarray(12345.0))
    with pytest.raises(ValueError, match="objective does not equal"):
        load_fixed_routing_linear_result(path)

    save_fixed_routing_linear_result(_result(), path)
    _rewrite(path, gradient=np.zeros(3))
    with pytest.raises(ValueError, match="gradient must have shape"):
        load_fixed_routing_linear_result(path)


def test_constructor_rejects_duplicate_ordering():
    result = _result()
    with pytest.raises(ValueError, match="free_od_indices must be unique"):
        replace(result, free_od_indices=np.array([3, 3]))
