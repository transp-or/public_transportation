from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from public_transportation.preprocessing.reduced_od import (
    NamedImmutableArray,
    ReducedODArrayArtifact,
    ReducedODArtifactKind,
    ReducedODArtifactManifest,
    ReducedODArtifactStoreError,
    canonical_json,
    load_reduced_od_phase_artifact,
    save_reduced_od_phase_artifact,
)


def _manifest() -> ReducedODArtifactManifest:
    return ReducedODArtifactManifest(
        artifact_kind=ReducedODArtifactKind.TIMETABLE,
        configuration_fingerprint="configuration",
        source_fingerprints=(("scenario", "scenario-fingerprint"),),
        dimensions=(("num_stops", 3), ("num_trips", 2)),
    )


def test_manifest_has_canonical_identity() -> None:
    first = _manifest()
    second = _manifest()

    assert first.fingerprint == second.fingerprint
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert len(first.fingerprint) == 64
    with pytest.raises(FrozenInstanceError):
        first.schema_version = 2


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"source_fingerprints": (("z", "1"), ("a", "2"))},
            "sorted",
        ),
        (
            {"source_fingerprints": (("a", "1"), ("a", "2"))},
            "unique",
        ),
        ({"dimensions": (("num", -1),)}, "non-negative"),
    ],
)
def test_manifest_rejects_noncanonical_metadata(kwargs, message) -> None:
    values = {
        "artifact_kind": ReducedODArtifactKind.TIMETABLE,
        "configuration_fingerprint": "configuration",
        "source_fingerprints": (("scenario", "fingerprint"),),
        "dimensions": (),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ReducedODArtifactManifest(**values)


def test_arrays_are_owned_immutable_and_content_sensitive() -> None:
    source = np.arange(6, dtype=np.int32)
    named = NamedImmutableArray("indices", source)
    artifact = ReducedODArrayArtifact(_manifest(), (named,))

    source[0] = 99
    assert named.values[0] == 0
    assert named.values.flags.c_contiguous
    assert not named.values.flags.writeable
    assert artifact.retained_bytes == 24

    changed = ReducedODArrayArtifact(
        _manifest(),
        (NamedImmutableArray("indices", np.arange(6, dtype=np.int32) + 1),),
    )
    assert artifact.fingerprint != changed.fingerprint


def test_array_contract_rejects_object_duplicate_and_unsorted_arrays() -> None:
    with pytest.raises(TypeError, match="object dtype"):
        NamedImmutableArray("bad", np.asarray([object()], dtype=object))

    a = NamedImmutableArray("a", np.asarray([1]))
    b = NamedImmutableArray("b", np.asarray([2]))
    with pytest.raises(ValueError, match="sorted"):
        ReducedODArrayArtifact(_manifest(), (b, a))
    with pytest.raises(ValueError, match="unique"):
        ReducedODArrayArtifact(_manifest(), (a, a))


def test_legacy_dataclass_schema_is_reported_as_cache_incompatibility(
    tmp_path: Path,
) -> None:
    target = tmp_path / "phase"
    save_reduced_od_phase_artifact(
        target,
        phase="test",
        payload=_manifest(),
        configuration_fingerprint="configuration",
    )
    manifest_path = target / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["payload"]["fields"].pop("artifact_kind")
    document.pop("content_fingerprint")
    document["content_fingerprint"] = hashlib.sha256(
        canonical_json(document).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(canonical_json(document), encoding="utf-8")

    with pytest.raises(ReducedODArtifactStoreError, match="incompatible"):
        load_reduced_od_phase_artifact(
            target,
            expected_phase="test",
            expected_configuration_fingerprint="configuration",
            expected_upstream_fingerprints={},
        )


def test_phase_artifact_persistence_reports_completion_after_publication(
    tmp_path: Path,
) -> None:
    events: list[dict[str, object]] = []
    target = tmp_path / "phase-progress"
    save_reduced_od_phase_artifact(
        target,
        phase="test",
        payload=ReducedODArrayArtifact(
            _manifest(),
            (NamedImmutableArray("indices", np.arange(3, dtype=np.int32)),),
        ),
        configuration_fingerprint="configuration",
        progress=events.append,
    )

    assert (target / "manifest.json").is_file()
    assert events[0]["status"] == "started"
    assert events[-1]["status"] == "completed"
    assert events[-1]["estimated_remaining_seconds"] == pytest.approx(0.0)
