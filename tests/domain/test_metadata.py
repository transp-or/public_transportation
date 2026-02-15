from __future__ import annotations

from public_transportation.domain.metadata import Metadata


def test_metadata_defaults():
    m = Metadata(title="Test")

    assert m.title == "Test"
    assert m.description is None
    assert m.timezone == "Europe/Zurich"
    assert m.cost_unit == "minutes"

    # created_at auto-populated
    assert isinstance(m.created_at, str)
    assert len(m.created_at) > 0

    # mutable defaults exist
    assert m.sources == []
    assert m.extra == {}


def test_metadata_mutable_defaults_are_independent():
    m1 = Metadata(title="A")
    m2 = Metadata(title="B")

    m1.sources.append("file1")
    m1.extra["x"] = 1

    assert m2.sources == []
    assert m2.extra == {}


def test_metadata_full_construction():
    m = Metadata(
        title="Scenario",
        description="Desc",
        timezone="UTC",
        cost_unit="generalized_minutes",
        sources=["gtfs"],
        extra={"version": 1},
    )

    assert m.title == "Scenario"
    assert m.description == "Desc"
    assert m.timezone == "UTC"
    assert m.cost_unit == "generalized_minutes"
    assert m.sources == ["gtfs"]
    assert m.extra["version"] == 1