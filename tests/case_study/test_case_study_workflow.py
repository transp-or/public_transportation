from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from public_transportation.case_study.adapter import GenericCaseAdapter, load_canonical_measurements
from public_transportation.case_study.config import CaseStudyConfigError, load_case_study_config
from public_transportation.case_study.runner import GenericCaseRunner, main, run_case_stage
from public_transportation.domain import Scenario
from public_transportation.preprocessing.reduced_od import prepare_reduced_od_timetable, resolve_measurements


ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "docs/source/examples/case_study_template"
CASE_CONFIG = TEMPLATE / "config/case.toml"


def _copy_case(tmp_path: Path) -> Path:
    destination = tmp_path / "case"
    shutil.copytree(TEMPLATE, destination, ignore=shutil.ignore_patterns("results"))
    return destination


def _config_text(case: Path) -> str:
    return (case / "config/case.toml").read_text(encoding="utf-8")


def _write_config(case: Path, text: str) -> None:
    (case / "config/case.toml").write_text(text, encoding="utf-8")


def test_template_has_deterministic_fingerprint() -> None:
    first = load_case_study_config(CASE_CONFIG)
    second = load_case_study_config(CASE_CONFIG)
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint_payload_json == second.fingerprint_payload_json


def test_template_declares_data_only_setuptools_project() -> None:
    pyproject = (TEMPLATE / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.setuptools]" in pyproject
    assert "packages = []" in pyproject


def test_missing_configuration_field_is_rejected(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    _write_config(case, _config_text(case).replace('name = "template_small_multiline"\n', ""))
    with pytest.raises(CaseStudyConfigError, match="case is missing required fields"):
        load_case_study_config(case / "config/case.toml", case_root=case)


def test_unknown_configuration_field_is_rejected(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    _write_config(case, _config_text(case).replace("[case]\n", "[case]\nundocumented = true\n", 1))
    with pytest.raises(CaseStudyConfigError, match="contains unknown fields"):
        load_case_study_config(case / "config/case.toml", case_root=case)


def test_invalid_configured_path_fails_before_processing(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    _write_config(case, _config_text(case).replace('scenario_directory = "inputs/scenario"', 'scenario_directory = "inputs/missing"'))
    config = load_case_study_config(case / "config/case.toml", case_root=case)
    with pytest.raises(FileNotFoundError, match="scenario_directory"):
        GenericCaseAdapter(config).data


def test_missing_od_budget_is_rejected(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    _write_config(case, _config_text(case).replace("max_od_cells = 48", "max_od_cells = 0"))
    with pytest.raises(CaseStudyConfigError, match="max_od_cells"):
        load_case_study_config(case / "config/case.toml", case_root=case)


def test_invalid_horizon_is_rejected(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    _write_config(case, _config_text(case).replace('horizon_end = "09:00:00"', 'horizon_end = "08:00:00"'))
    with pytest.raises(CaseStudyConfigError, match="horizon_end"):
        load_case_study_config(case / "config/case.toml", case_root=case)


def test_unknown_measurement_column_is_rejected(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    config = load_case_study_config(case / "config/case.toml", case_root=case)
    path = case / "inputs/measurements_unknown.csv"
    path.write_text(
        "method_id,measurement_type,stop_id,wrong_time,value,trip_id,line_id\n"
        "m,boarding,A,08:00:01,1,L1_AB_0800,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing configured columns"):
        load_canonical_measurements(path, config=config)


def test_duplicate_observation_is_rejected(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    config = load_case_study_config(case / "config/case.toml", case_root=case)
    source = (case / "inputs/measurements_boarding_alighting.csv").read_text(encoding="utf-8")
    path = case / "inputs/measurements_duplicate.csv"
    path.write_text(source + source.splitlines()[1] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate measurement"):
        load_canonical_measurements(path, config=config)


def test_missing_timestamp_is_rejected(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    config = load_case_study_config(case / "config/case.toml", case_root=case)
    path = case / "inputs/measurements_missing_time.csv"
    path.write_text(
        "method_id,measurement_type,stop_id,time,value,trip_id,line_id\n"
        "m,boarding,A,,1,L1_AB_0800,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="timestamp"):
        load_canonical_measurements(path, config=config)


def test_ambiguous_event_match_is_rejected(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    scenario_dir = case / "inputs/scenario"
    with (scenario_dir / "trips.csv").open("a", encoding="utf-8", newline="") as stream:
        stream.write("L1_AB_DUP,L1,100\n")
    with (scenario_dir / "stop_times.csv").open("a", encoding="utf-8", newline="") as stream:
        stream.write("L1_AB_DUP,A,1,08:00:00,08:00:01\nL1_AB_DUP,B,2,08:10:00,08:10:01\n")
    scenario = Scenario.from_folder(scenario_dir, strict=True, demand_file=scenario_dir / "prior_demand.csv")
    timetable = prepare_reduced_od_timetable(scenario, configuration_fingerprint="test", mapping_policy="identity")
    config = load_case_study_config(case / "config/case.toml", case_root=case)
    path = case / "inputs/ambiguous.csv"
    path.write_text(
        "method_id,measurement_type,stop_id,time,value,trip_id,line_id\n"
        "m,boarding,A,08:00:01,1,,L1\n",
        encoding="utf-8",
    )
    table = load_canonical_measurements(path, config=config)
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_measurements(timetable, table)


def test_materialization_requires_candidate_and_reviewer(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    config = load_case_study_config(case / "config/case.toml", case_root=case)
    runner = GenericCaseRunner(config)
    with pytest.raises(ValueError, match="candidate.*reviewer"):
        runner.materialize_bins(candidate=None, reviewer=None)


def test_editable_package_detection_is_recorded(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    (case / "uv.lock").write_text(
        'name = "public-transportation"\nsource = { editable = "." }\n',
        encoding="utf-8",
    )
    config = load_case_study_config(case / "config/case.toml", case_root=case)
    info = GenericCaseRunner(config).package_info()
    assert info["editable_local_package_detected"] is True


def test_cli_returns_nonzero_after_stage_failure(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    _write_config(case, _config_text(case).replace('scenario_directory = "inputs/scenario"', 'scenario_directory = "inputs/missing"'))
    assert main(["check", "--config", str(case / "config/case.toml"), "--case-root", str(case)]) == 1


def test_clean_template_executes_admission_path(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    config_path = case / "config/case.toml"
    kwargs = {"case_root": case}
    assert run_case_stage(config_path, "check", **kwargs)["audit"]["resolved_measurement_count"] == 8
    run_case_stage(config_path, "od-universe", **kwargs)
    run_case_stage(config_path, "time-discretization", **kwargs)
    run_case_stage(config_path, "materialize-bins", candidate="recommendation", reviewer="pytest", **kwargs)
    run_case_stage(config_path, "expand-od", **kwargs)
    run_case_stage(config_path, "structural-zeros", **kwargs)
    prepared = run_case_stage(config_path, "prepare", **kwargs)
    assert prepared["dimensions"]["measurements"] == 8
    preflight = run_case_stage(config_path, "preflight", **kwargs)
    assert preflight["compatible"] is True
    benchmark = run_case_stage(config_path, "benchmark", **kwargs)
    assert benchmark["benchmark"]["finite"] is True
    import jax

    jax.config.update("jax_enable_x64", True)
    fit = run_case_stage(config_path, "fit", method="map", likelihood="poisson", **kwargs)
    fit_path = case / "results/fits/map_poisson.json"
    assert len(fit["raw_parameters"]) > 0
    reconstructed = run_case_stage(config_path, "reconstruct", fit_path=fit_path, **kwargs)
    assert len(reconstructed["keys"]) == len(reconstructed["demand"])


def test_independent_od_workflow_has_pair_and_expansion_audits(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    # The canonical template uses the pair-only input.  Remove the legacy
    # fixture to prove that the new path does not need a time-dependent demand.
    (case / "inputs/scenario/prior_demand.csv").unlink()
    source_snapshots = {
        path: path.read_bytes()
        for path in (
            case / "inputs/od_pairs.csv",
            case / "inputs/measurements_boarding_alighting.csv",
            case / "inputs/scenario/stops.csv",
        )
    }
    config_path = case / "config/case.toml"
    kwargs = {"case_root": case}
    universe = run_case_stage(config_path, "od-universe", **kwargs)
    assert universe["source"] == "file"
    assert universe["retained_pair_count"] == 2
    run_case_stage(config_path, "time-discretization", **kwargs)
    run_case_stage(config_path, "materialize-bins", candidate="recommendation", reviewer="pytest", **kwargs)
    expansion = run_case_stage(config_path, "expand-od", **kwargs)
    assert expansion["retained_cell_count"] >= 2
    assert (case / "results/audit/od_pairs.csv").is_file()
    assert (case / "results/generated_inputs/prior_demand.csv").is_file()
    run_case_stage(config_path, "structural-zeros", **kwargs)
    manifest = run_case_stage(config_path, "check", **kwargs)["manifest"]
    assert manifest["input_semantics"] == "independent_od_universe"
    assert manifest["prior_demand"]["source"] == "all_ones"
    data = GenericCaseAdapter(load_case_study_config(config_path, case_root=case)).load_persisted_data()
    assert data.prior_demand is not None
    assert set(data.prior_demand.values()) == {1.0}
    assert set(data.production_inputs.values()) == {1.0}
    assert set(data.destination_attractiveness.values()) == {1.0}
    assert data.input_semantics == "independent_od_universe"
    assert all(path.read_bytes() == content for path, content in source_snapshots.items())


def test_external_pair_prior_can_use_paths_fallback(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    prior = case / "inputs/prior_pair_values.csv"
    prior.write_text(
        "origin_stop_id,destination_stop_id,prior_value\nA,B,2\nC,D,3\n",
        encoding="utf-8",
    )
    text = _config_text(case).replace(
        'results_directory = "results"',
        'prior_demand = "inputs/prior_pair_values.csv"\nresults_directory = "results"',
    ).replace(
        'source = "all_ones"\nvalue = 1.0\nsemantics = "neutral_seed"',
        'source = "external_file"\nsemantics = "external_prior"',
    )
    _write_config(case, text)
    config_path = case / "config/case.toml"
    config = load_case_study_config(config_path, case_root=case)
    run_case_stage(config_path, "od-universe", case_root=case)
    run_case_stage(config_path, "time-discretization", case_root=case)
    run_case_stage(config_path, "materialize-bins", candidate="recommendation", reviewer="pytest", case_root=case)
    run_case_stage(config_path, "expand-od", case_root=case)
    run_case_stage(config_path, "structural-zeros", case_root=case)
    data = GenericCaseAdapter(config).load_persisted_data()
    assert set(data.prior_demand.values()) == {2.0, 3.0}


def _run_through_expansion(case: Path) -> Path:
    config_path = case / "config/case.toml"
    kwargs = {"case_root": case}
    run_case_stage(config_path, "od-universe", **kwargs)
    run_case_stage(config_path, "time-discretization", **kwargs)
    run_case_stage(
        config_path,
        "materialize-bins",
        candidate="recommendation",
        reviewer="pytest",
        **kwargs,
    )
    run_case_stage(config_path, "expand-od", **kwargs)
    return config_path


def test_lightweight_stages_do_not_construct_timetable_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _copy_case(tmp_path)
    (case / "inputs/scenario/prior_demand.csv").unlink()

    def fail_expansion(*args: object, **kwargs: object) -> object:
        raise AssertionError("lightweight stage invoked OD-time expansion")

    monkeypatch.setattr(
        "public_transportation.case_study.adapter.expand_candidate_od_time_cells",
        fail_expansion,
    )
    config_path = case / "config/case.toml"
    kwargs = {"case_root": case}
    check = run_case_stage(config_path, "check", **kwargs)
    assert check["audit"]["od_universe_status"] == "not_run"
    assert check["audit"]["od_time_expansion_status"] == "not_run"
    assert check["audit"]["timetable_feasibility_status"] == "not_run"
    run_case_stage(config_path, "od-universe", **kwargs)
    run_case_stage(config_path, "time-discretization", **kwargs)
    run_case_stage(
        config_path,
        "materialize-bins",
        candidate="recommendation",
        reviewer="pytest",
        **kwargs,
    )


def test_expand_constructs_one_reusable_timetable_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _copy_case(tmp_path)
    (case / "inputs/scenario/prior_demand.csv").unlink()
    calls: list[int] = []
    from public_transportation.preprocessing import od_universe as module

    original = module.TimetableFeasibilityIndex.from_scenario

    def counted(cls: object, *args: object, **kwargs: object) -> object:
        calls.append(1)
        return original.__func__(cls, *args, **kwargs)  # type: ignore[attr-defined]

    monkeypatch.setattr(
        module.TimetableFeasibilityIndex,
        "from_scenario",
        classmethod(counted),
    )
    _run_through_expansion(case)
    assert calls == [1]


def test_structural_zeros_reuses_persisted_expansion_and_prepare_requires_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _copy_case(tmp_path)
    (case / "inputs/scenario/prior_demand.csv").unlink()
    config_path = case / "config/case.toml"
    with pytest.raises(FileNotFoundError, match="run expand-od before structural-zeros"):
        run_case_stage(config_path, "structural-zeros", case_root=case)
    with pytest.raises(FileNotFoundError, match="run expand-od before structural-zeros"):
        run_case_stage(config_path, "prepare", case_root=case)
    _run_through_expansion(case)
    monkeypatch.setattr(
        "public_transportation.case_study.adapter.expand_candidate_od_time_cells",
        lambda *args, **kwargs: pytest.fail("structural-zeros recomputed expansion"),
    )
    run_case_stage(config_path, "structural-zeros", case_root=case)


def test_expansion_fingerprint_mismatch_is_rejected(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    (case / "inputs/scenario/prior_demand.csv").unlink()
    config_path = _run_through_expansion(case)
    bins = case / "results/generated_inputs/time_bins.csv"
    # Change an approved edge while preserving a valid, sorted contract.
    lines = bins.read_text(encoding="utf-8").splitlines()
    first = lines[1].split(",")
    first[1] = str(int(first[1]) + 1)
    lines[1] = ",".join(first)
    bins.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="time-bin fingerprint"):
        run_case_stage(config_path, "structural-zeros", case_root=case)
