from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from public_transportation.inference.block_coordinate._canonical import fingerprint
from public_transportation.inference.gravity import load_gravity_aggregate


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "channel_name": "gps_trip_attributes",
        "kind": "trip_attribute_distribution",
        "histograms": [
            {
                "attribute": "travel_time",
                "unit": "seconds",
                "support": [0, 7200],
                "bins": [
                    {"label": "short", "lower": 0, "upper": 600},
                    {"label": "long", "lower": 600, "upper": 7200},
                ],
                "strata": [
                    {"name": "all", "counts": [10, 20], "total": 30},
                ],
            },
            {
                "attribute": "transfers",
                "unit": "count",
                "support": [0, 3],
                "bins": [
                    {"label": "none", "lower": 0, "upper": 1},
                    {"label": "one_or_more", "lower": 1, "upper": 3},
                ],
                "strata": [
                    {
                        "name": "all",
                        "counts": {"none": 21, "one_or_more": 9},
                        "total": 30,
                    },
                ],
            },
        ],
        "metadata": {
            "collection_period": "2022-10-03/2022-12-23",
            "valid_journeys": 30,
            "excluded_journeys": 2,
            "cleaning_reasons": {"invalid_overnight": 2},
            "apc_overlap_policy": "overlap_recorded_not_independent",
            "strata_exhaustive": True,
        },
        "uncertainty": {
            "likelihood": "dirichlet_multinomial",
            "concentration": 50,
        },
    }


def _write(path: Path, payload: dict[str, object], *, checksum: bool = True) -> None:
    document = dict(payload)
    if checksum:
        document["content_sha256"] = fingerprint(document)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")


def test_load_validates_histograms_metadata_uncertainty_and_writes_audit(tmp_path):
    source = tmp_path / "gps_aggregate.json"
    audit = tmp_path / "audit" / "gps.json"
    _write(source, _payload())

    loaded = load_gravity_aggregate(source, audit_path=audit)

    assert loaded is not None
    assert loaded.channel_name == "gps_trip_attributes"
    assert loaded.valid_journeys == 30
    assert loaded.excluded_journeys == 2
    assert tuple(item.attribute for item in loaded.histograms) == (
        "travel_time",
        "transfers",
    )
    assert loaded.histograms[0].sample_size == 30
    assert loaded.uncertainty.concentration == 50
    assert loaded.source_path == source.resolve()
    assert loaded.file_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(loaded.content_sha256) == 64
    assert len(loaded.fingerprint) == 64

    report = json.loads(audit.read_text(encoding="utf-8"))
    assert report["status"] == "validated"
    assert report["file_sha256"] == loaded.file_sha256
    assert report["content_sha256"] == loaded.content_sha256
    assert report["valid_journeys"] == 30
    assert report["excluded_journeys"] == 2
    assert report["histograms"][0]["bins"] == [
        {"label": "short", "lower": 0.0, "upper": 600.0},
        {"label": "long", "lower": 600.0, "upper": 7200.0},
    ]
    assert report["histograms"][1]["strata"] == [{"name": "all", "total": 30}]


def test_disabled_optional_data_is_absent_without_reading_and_is_audited(tmp_path):
    source = tmp_path / "not-json.json"
    source.write_text("not JSON", encoding="utf-8")
    audit = tmp_path / "audit.json"

    assert load_gravity_aggregate(source, enabled=False, audit_path=audit) is None
    report = json.loads(audit.read_text(encoding="utf-8"))
    assert report == {
        "schema_version": 1,
        "status": "absent",
        "reason": "optional aggregate observation is disabled",
    }


def test_omitted_optional_path_is_absent_by_default(tmp_path):
    audit = tmp_path / "audit.json"

    assert load_gravity_aggregate(None, audit_path=audit) is None
    assert json.loads(audit.read_text(encoding="utf-8"))["status"] == "absent"


def test_explicitly_enabled_without_path_fails(tmp_path):
    audit = tmp_path / "audit.json"

    with pytest.raises(ValueError, match="no file was provided"):
        load_gravity_aggregate(None, enabled=True, audit_path=audit)
    assert json.loads(audit.read_text(encoding="utf-8"))["status"] == "rejected"


def test_enabled_missing_data_fails_before_estimation_and_is_audited(tmp_path):
    source = tmp_path / "missing.json"
    audit = tmp_path / "audit.json"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_gravity_aggregate(source, audit_path=audit)
    report = json.loads(audit.read_text(encoding="utf-8"))
    assert report["status"] == "rejected"
    assert report["file_sha256"] is None


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    (
        (
            "duplicate bins",
            lambda payload: payload["histograms"][0]["bins"][1].update(label="short"),
            "duplicate bin label",
        ),
        (
            "negative count",
            lambda payload: payload["histograms"][0]["strata"][0]["counts"].__setitem__(
                0, -1
            ),
            "non-negative integer",
        ),
        (
            "non-integral count",
            lambda payload: payload["histograms"][0]["strata"][0]["counts"].__setitem__(
                0, 1.5
            ),
            "non-negative integer",
        ),
        (
            "inconsistent total",
            lambda payload: payload["histograms"][0]["strata"][0].update(total=99),
            "total must equal",
        ),
        (
            "overlapping bins",
            lambda payload: payload["histograms"][0]["bins"][1].update(lower=500),
            "overlap",
        ),
        (
            "incomplete bins",
            lambda payload: payload["histograms"][0]["bins"][1].update(lower=700),
            "gap and are incomplete",
        ),
        (
            "invalid unit",
            lambda payload: payload["histograms"][0].update(unit="hours"),
            "unsupported aggregate unit",
        ),
        (
            "unsupported likelihood",
            lambda payload: payload["uncertainty"].update(likelihood="poisson"),
            "unsupported aggregate likelihood",
        ),
    ),
)
def test_invalid_aggregate_documents_are_rejected_and_reported(
    tmp_path, name, mutate, message
):
    del name
    source = tmp_path / "invalid.json"
    audit = tmp_path / "audit.json"
    payload = _payload()
    mutate(payload)
    _write(source, payload)

    with pytest.raises(ValueError, match=message):
        load_gravity_aggregate(source, audit_path=audit)
    report = json.loads(audit.read_text(encoding="utf-8"))
    assert report["status"] == "rejected"
    assert report["file_sha256"]
    assert message in report["error"]


def test_missing_or_tampered_content_checksum_is_rejected(tmp_path):
    missing = tmp_path / "missing-checksum.json"
    _write(missing, _payload(), checksum=False)
    with pytest.raises(ValueError, match="content_sha256"):
        load_gravity_aggregate(missing)

    tampered = tmp_path / "tampered.json"
    payload = _payload()
    payload["content_sha256"] = "0" * 64
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content_sha256"):
        load_gravity_aggregate(tampered)


def test_strata_exhaustive_requires_metadata_valid_journeys(tmp_path):
    source = tmp_path / "inconsistent-strata.json"
    payload = _payload()
    payload["metadata"]["valid_journeys"] = 31
    _write(source, payload)

    with pytest.raises(ValueError, match="strata total"):
        load_gravity_aggregate(source)
