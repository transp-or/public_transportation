from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from public_transportation.preprocessing.reduced_od import (
    ReducedODConfigError,
    load_reduced_od_config,
)


def _toml(*, mode: str = "provided", extra: str = "") -> str:
    production = (
        'mode = "provided"\nsemantics = "external_journey_productions"\ninput_path = "productions.csv"'
        if mode == "provided"
        else 'mode = "estimated_basis"\nsemantics = "estimated_production_basis"\nbasis = "origin_period"'
    )
    return f"""
schema_version = 2

[observations]
service_day = "2026-01-15"
analysis_start_seconds = 21600
analysis_end_seconds = 108000
after_midnight_convention = "service_day_extended"
apc_policy_identifier = "synthetic-clean-v1"
sensor_coverage_policy = "declared-complete"
sensor_outage_policy = "exclude-declared-outages"
unit = "timetable_event"
accepted_types = ["alighting", "boarding"]
missing_policy = "exclude"
duplicate_policy = "error"
ambiguous_event_policy = "error"
cleaning_stage = "external"

[journeys]
origin_semantics = "first_boarding"
destination_semantics = "final_alighting"
time_bin_membership = "half_open"
maximum_transfers = 2
maximum_waiting_seconds = 3600
maximum_journey_seconds = 10800
maximum_alternatives_per_cell = 4
transfer_footpath_policy = "declared-directed-v1"
route_shares = "fixed_within_fit"

[productions]
{production}

[stops]
mapping_policy = "authoritative"
physical_stop_mapping_path = "physical_stops.csv"
footpaths_path = "footpaths.csv"

[outputs]
spatial_level = "scenario_stop"
reconstruct_full_od = false

[model]
likelihood = "negative_binomial"

[validation]
detailed_assignment = "explicit_only"
{extra}
"""


def _write(tmp_path, text: str, name: str = "reduced_od.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("mode", ["provided", "estimated_basis"])
def test_loads_both_production_modes_and_resolves_paths(tmp_path, mode) -> None:
    config = load_reduced_od_config(_write(tmp_path, _toml(mode=mode)))

    assert config.schema_version == 2
    assert config.productions.mode == mode
    assert config.observations.accepted_types == ("alighting", "boarding")
    assert config.stops.physical_stop_mapping_path == (
        tmp_path / "physical_stops.csv"
    ).resolve()
    assert len(config.fingerprint) == 64
    assert "source_file" not in config.fingerprint_payload_json


def test_config_is_frozen_and_fingerprint_is_deterministic(tmp_path) -> None:
    path = _write(tmp_path, _toml())
    first = load_reduced_od_config(path)
    second = load_reduced_od_config(path)

    assert first.fingerprint_payload_json == second.fingerprint_payload_json
    assert first.fingerprint == second.fingerprint
    with pytest.raises(FrozenInstanceError):
        first.schema_version = 2


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("schema_version = 2", "schema_version = 3", "Unsupported"),
        (
            'unit = "timetable_event"',
            'unit = "stop_period"',
            "observations.unit",
        ),
        (
            'accepted_types = ["alighting", "boarding"]',
            'accepted_types = ["boarding", "load"]',
            "unsupported",
        ),
        ("maximum_transfers = 2", "maximum_transfers = -1", "at least 0"),
        (
            'detailed_assignment = "explicit_only"',
            'detailed_assignment = "inside_objective"',
            "validation.detailed_assignment",
        ),
    ],
)
def test_rejects_invalid_values(tmp_path, old, new, message) -> None:
    with pytest.raises(ReducedODConfigError, match=message):
        load_reduced_od_config(_write(tmp_path, _toml().replace(old, new)))


def test_rejects_unknown_and_missing_keys(tmp_path) -> None:
    with pytest.raises(ReducedODConfigError, match="unknown parameters"):
        load_reduced_od_config(
            _write(
                tmp_path,
                _toml().replace(
                    'unit = "timetable_event"',
                    'unit = "timetable_event"\nsurprise = true',
                ),
            )
        )
    with pytest.raises(ReducedODConfigError, match="missing required"):
        load_reduced_od_config(
            _write(
                tmp_path,
                _toml().replace('missing_policy = "exclude"\n', ""),
            )
        )


def test_production_mode_fields_are_strict(tmp_path) -> None:
    with pytest.raises(ReducedODConfigError, match="input_path is required"):
        load_reduced_od_config(
            _write(
                tmp_path,
                _toml().replace('input_path = "productions.csv"\n', ""),
            )
        )
    with pytest.raises(ReducedODConfigError, match="input_path is not allowed"):
        load_reduced_od_config(
            _write(
                tmp_path,
                _toml(mode="estimated_basis").replace(
                    'basis = "origin_period"',
                    'basis = "origin_period"\ninput_path = "bad.csv"',
                ),
            )
        )
