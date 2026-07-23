from __future__ import annotations

from pathlib import Path

import pytest

from public_transportation.preprocessing import (
    StructuralZeroConfigError,
    load_structural_zero_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTED_EXAMPLE = (
    REPOSITORY_ROOT
    / "docs"
    / "source"
    / "examples"
    / "geneva_gtfs"
    / "structural_zeros.toml"
)


def _valid_toml(*, extra: str = "") -> str:
    return f"""\
version = 1

[scenario]
folder = "scenario"

[output]
folder = "generated"

[rules.enabled]
same_stop = true
no_feasible_path = true
maximum_transfers = true
maximum_initial_wait = false
maximum_journey_time = false
minimum_feasible_departures = false

[rules.same_stop]

[rules.no_feasible_path]

[rules.maximum_transfers]
max_transfers = 2

[assignment]
{extra}
"""


def _write_config(tmp_path: Path, content: str) -> Path:
    (tmp_path / "scenario").mkdir()
    path = tmp_path / "structural_zeros.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_documented_example_loads_and_resolves_paths() -> None:
    config = load_structural_zero_config(DOCUMENTED_EXAMPLE)

    example_root = DOCUMENTED_EXAMPLE.parent.resolve()
    assert config.version == 1
    assert config.source_file == DOCUMENTED_EXAMPLE.resolve()
    assert config.scenario.folder == example_root / "data"
    assert config.scenario.demand_file == example_root / "data/prior_demand.csv"
    assert (
        config.output.folder == example_root / "pre_processing/results/structural_zeros"
    )
    assert config.rules.enabled.maximum_transfers
    assert not config.rules.enabled.maximum_initial_wait
    assert config.rules.maximum_transfers is not None
    assert config.rules.maximum_transfers.max_transfers == 2
    assert config.existing_fixed_demand is not None
    assert config.existing_fixed_demand.file == example_root / "data/fixed_demand.csv"


def test_defaults_and_deterministic_resolved_round_trip(tmp_path: Path) -> None:
    config = load_structural_zero_config(_write_config(tmp_path, _valid_toml()))

    assert config.output.include_retained_cells_in_report
    assert config.assignment.max_access_deviation_minutes == 15.0
    assert config.assignment.max_transfer_wait_minutes == 30.0
    assert config.assignment.minimum_dwell_seconds == 1
    assert len(config.fingerprint) == 64

    resolved = tmp_path / "resolved.toml"
    resolved.write_text(config.to_resolved_toml(), encoding="utf-8")
    reloaded = load_structural_zero_config(resolved)
    assert reloaded.fingerprint_payload_json == config.fingerprint_payload_json
    assert reloaded.fingerprint == config.fingerprint
    assert reloaded.to_resolved_toml() == config.to_resolved_toml()


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("version = 1", "version = 2", "Unsupported configuration version"),
        ("version = 1", 'version = 1\nunknown = "value"', "unknown parameters"),
        ("same_stop = true", 'same_stop = "yes"', "must be true or false"),
        ("max_transfers = 2", "max_transfers = -1", "must be at least 0"),
        (
            "[assignment]\n",
            "[assignment]\nminimum_dwell_seconds = 0\n",
            "must be at least 1",
        ),
    ],
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    content = _valid_toml().replace(old, new)
    path = _write_config(tmp_path, content)

    with pytest.raises(StructuralZeroConfigError, match=message):
        load_structural_zero_config(path)


def test_enabled_rule_requires_parameter_table(tmp_path: Path) -> None:
    content = _valid_toml().replace(
        "[rules.maximum_transfers]\nmax_transfers = 2\n\n", ""
    )
    path = _write_config(tmp_path, content)

    with pytest.raises(StructuralZeroConfigError, match="must be present"):
        load_structural_zero_config(path)


def test_disabled_rule_table_may_be_omitted(tmp_path: Path) -> None:
    config = load_structural_zero_config(_write_config(tmp_path, _valid_toml()))

    assert config.rules.maximum_initial_wait is None
    assert config.rules.maximum_journey_time is None
    assert config.rules.minimum_feasible_departures is None


def test_existing_fixed_demand_has_only_strict_file_contract(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.csv"
    fixed.write_text(
        "origin_stop_id,dest_stop_id,time_bin_id,fixed_flow\n", encoding="utf-8"
    )
    content = _valid_toml(
        extra='\n[existing_fixed_demand]\nfile = "fixed.csv"\nconflict_policy = "overwrite"\n'
    )
    path = _write_config(tmp_path, content)

    with pytest.raises(StructuralZeroConfigError, match="unknown parameters"):
        load_structural_zero_config(path)


def test_missing_input_paths_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "structural_zeros.toml"
    path.write_text(_valid_toml(), encoding="utf-8")

    with pytest.raises(StructuralZeroConfigError, match="existing directory"):
        load_structural_zero_config(path)


def test_output_must_not_be_scenario_folder(tmp_path: Path) -> None:
    content = _valid_toml().replace('folder = "generated"', 'folder = "scenario"')
    path = _write_config(tmp_path, content)

    with pytest.raises(StructuralZeroConfigError, match="must differ"):
        load_structural_zero_config(path)
