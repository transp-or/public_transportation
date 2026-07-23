from __future__ import annotations

import hashlib
import numpy as np
import pytest

from public_transportation.inference.results_io import (
    ODThetaVIResults,
    load_od_theta_vi_results,
    save_od_theta_vi_results,
)
from public_transportation.inference.runtime_profile import ODAssignmentRuntimeProfile


def _results(*, with_layout: bool = True) -> ODThetaVIResults:
    kwargs = {}
    if with_layout:
        layout_payload = '{"version":1}'
        compact_payload = '{"active":[1,2],"version":1}'
        layout_fingerprint = hashlib.sha256(layout_payload.encode()).hexdigest()
        compact_fingerprint = hashlib.sha256(compact_payload.encode()).hexdigest()
        kwargs = dict(
            free_od_indices=np.asarray([1], dtype=np.int64),
            fixed_od_indices=np.asarray([0, 2], dtype=np.int64),
            fixed_od_values=np.asarray([0.0, 7.5]),
            od_layout_fingerprint=layout_fingerprint,
            od_layout_payload_json=layout_payload,
            compact_layout_fingerprint=compact_fingerprint,
            compact_layout_payload_json=compact_payload,
            runtime_profile=ODAssignmentRuntimeProfile(
                num_od_total=3,
                num_free_od=1,
                num_fixed_od=2,
                num_fixed_zero_od=1,
                num_fixed_positive_od=1,
                assignment_active_od=2,
                original_destination_groups=3,
                active_destination_groups=2,
                removed_destination_groups=1,
                od_layout_fingerprint=layout_fingerprint,
                compact_layout_fingerprint=compact_fingerprint,
            ),
        )
    return ODThetaVIResults(
        fingerprint="fp",
        f0=np.asarray([11.0, 20.0, 33.0]),
        theta_samples=np.asarray([2.0, 3.0]),
        f_samples=np.asarray([[0.0, 18.0, 7.5], [0.0, 22.0, 7.5]]),
        theta_mean=2.5,
        theta_sd=0.5,
        f_mean=np.asarray([0.0, 20.0, 7.5]),
        vi_losses=np.asarray([4.0, 3.0]),
        **kwargs,
    )


def test_round_trip_preserves_reduced_layout(tmp_path):
    path = save_od_theta_vi_results(path=tmp_path / "result.npz", results=_results())
    loaded = load_od_theta_vi_results(path)

    assert loaded.num_od == 3
    assert loaded.num_free_od == 1
    assert loaded.num_fixed_od == 2
    assert np.array_equal(loaded.free_od_indices, [1])
    assert np.array_equal(loaded.fixed_od_indices, [0, 2])
    assert np.array_equal(loaded.fixed_od_values, [0.0, 7.5])
    assert loaded.od_layout_fingerprint == hashlib.sha256(b'{"version":1}').hexdigest()
    assert loaded.od_layout_payload_json == '{"version":1}'
    assert loaded.compact_layout_fingerprint is not None
    assert loaded.runtime_profile is not None
    assert loaded.runtime_profile.assignment_active_od == 2
    assert loaded.runtime_profile.removed_destination_groups == 1


def test_legacy_result_without_layout_loads_as_all_free(tmp_path):
    path = save_od_theta_vi_results(
        path=tmp_path / "legacy.npz",
        results=_results(with_layout=False),
    )
    loaded = load_od_theta_vi_results(path)
    assert loaded.num_free_od == 3
    assert loaded.num_fixed_od == 0
    assert loaded.runtime_profile is None


def test_loader_rejects_samples_that_change_a_fixed_value(tmp_path):
    result = _results()
    path = save_od_theta_vi_results(path=tmp_path / "invalid.npz", results=result)
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    payload["f_samples"] = np.asarray(payload["f_samples"])
    payload["f_samples"][0, 2] = 8.0
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="do not preserve"):
        load_od_theta_vi_results(path)


def test_loader_rejects_incomplete_or_nonpartitioning_layout(tmp_path):
    result = _results()
    path = save_od_theta_vi_results(path=tmp_path / "invalid.npz", results=result)
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}

    payload.pop("fixed_od_values")
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="all reduced-layout arrays"):
        load_od_theta_vi_results(path)

    payload["fixed_od_values"] = np.asarray([0.0, 7.5])
    payload["free_od_indices"] = np.asarray([0])
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="disjoint full partition"):
        load_od_theta_vi_results(path)


def test_loader_rejects_tampered_layout_payload(tmp_path):
    path = save_od_theta_vi_results(path=tmp_path / "tampered.npz", results=_results())
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    payload["od_layout_payload_json"] = '{"version":2}'
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="does not match"):
        load_od_theta_vi_results(path)


def test_loader_rejects_tampered_runtime_counts(tmp_path):
    path = save_od_theta_vi_results(path=tmp_path / "tampered.npz", results=_results())
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    payload["runtime_assignment_active_od"] = 3
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="runtime profile is inconsistent"):
        load_od_theta_vi_results(path)


def test_loader_rejects_tampered_compact_payload(tmp_path):
    path = save_od_theta_vi_results(path=tmp_path / "tampered.npz", results=_results())
    with np.load(path, allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    payload["compact_layout_payload_json"] = '{"active":[],"version":1}'
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="Compact layout fingerprint"):
        load_od_theta_vi_results(path)
