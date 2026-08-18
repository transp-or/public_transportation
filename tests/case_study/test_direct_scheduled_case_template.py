from __future__ import annotations

import hashlib
import json
import runpy
import shutil
import sys
import time
from pathlib import Path

import pytest

from public_transportation.preprocessing import ODTimeExpansionInterrupted


def _fixed_demand_provenance_case(
    tmp_path: Path, *, generated_values: tuple[int, int] | None
) -> Path:
    repository = Path(__file__).parents[2]
    template = repository / "docs/source/examples/direct_scheduled_case_template"
    example = repository / "docs/source/examples/simple_example_02"
    case = tmp_path / "case"
    shutil.copytree(template, case)
    shutil.copytree(example / "data", case / "inputs/scenario")
    shutil.copy(example / "data/fixed_demand.csv", case / "inputs/fixed_demand.csv")
    shutil.copy(
        example / "pre_processing/results/measurements_boarding_alighting.csv",
        case / "inputs/measurements.csv",
    )
    case_config = case / "config/case.toml"
    case_config.write_text(
        case_config.read_text(encoding="utf-8").replace(
            "REPLACE_WITH_PUBLIC_TRANSPORTATION_COMMIT",
            "fixed-demand-provenance-test-revision",
        ),
        encoding="utf-8",
    )
    if generated_values is not None:
        generated = case / "results/structural_zeros/fixed_demand.csv"
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(
            "origin_stop_id,dest_stop_id,time_bin_id,fixed_flow\n"
            f"A,H,t0,{generated_values[0]}\n"
            f"A,H,t1,{generated_values[1]}\n",
            encoding="utf-8",
        )
    return case


def _load_context_and_check_manifest(
    case: Path,
) -> tuple[object, dict[str, object], dict[str, Path]]:
    """Load the template driver while capturing the exact fixed path read."""
    sys.path.insert(0, str(case))
    try:
        namespace = runpy.run_path(str(case / "run_case.py"), run_name="provenance_test")
        adapter_globals = namespace["load_context"].__globals__
        original_reader = adapter_globals["read_fixed_demand_csv"]
        observed: dict[str, Path] = {}

        def capture_reader(path: Path, *, scenario: object) -> object:
            observed["path"] = Path(path)
            return original_reader(path, scenario=scenario)

        adapter_globals["read_fixed_demand_csv"] = capture_reader
        context = namespace["load_context"](case)
        namespace["check"](case)
    finally:
        sys.path.remove(str(case))
    manifest = json.loads(
        (case / "results/manifests/check.json").read_text(encoding="utf-8")
    )
    return context, manifest, observed


def test_direct_scheduled_template_complete_public_fixture(tmp_path: Path) -> None:
    """Exercise every template stage against the committed small fixture."""
    repository = Path(__file__).parents[2]
    template = repository / "docs/source/examples/direct_scheduled_case_template"
    example = repository / "docs/source/examples/simple_example_02"
    case = tmp_path / "case"
    shutil.copytree(template, case)
    (case / "inputs").mkdir()
    shutil.copytree(example / "data", case / "inputs/scenario")
    # The current workflow must generate the prior rather than treating the
    # fixture's prior file as a required pre-existing input.
    (case / "inputs/scenario/prior_demand.csv").unlink()
    shutil.copy(example / "data/fixed_demand.csv", case / "inputs/fixed_demand.csv")
    shutil.copy(
        example / "pre_processing/results/measurements_boarding_alighting.csv",
        case / "inputs/measurements.csv",
    )
    model = case / "config/model.toml"
    model.write_text(
        model.read_text(encoding="utf-8").replace(
            "maximum_iterations = 100", "maximum_iterations = 1"
        ),
        encoding="utf-8",
    )
    case_config = case / "config/case.toml"
    case_config.write_text(
        case_config.read_text(encoding="utf-8").replace(
            "REPLACE_WITH_PUBLIC_TRANSPORTATION_COMMIT",
            "public-fixture-test-revision",
        ),
        encoding="utf-8",
    )

    # Import the case-owned script as a local module, while the public package
    # remains the normally installed package used by this pytest environment.
    sys.path.insert(0, str(case))
    try:
        namespace = runpy.run_path(str(case / "run_case.py"), run_name="template_test")
        for stage in (
            "bootstrap-prior",
            "check",
            "structural-zeros",
            "prepare",
            "preflight",
            "benchmark",
            "fit",
            "validate",
        ):
            namespace["STAGES"][stage](case)
    finally:
        sys.path.remove(str(case))

    manifests = case / "results/manifests"
    for stage in (
        "bootstrap-prior",
        "check",
        "structural-zeros",
        "prepare",
        "preflight",
        "benchmark",
        "fit",
        "validate",
    ):
        payload = json.loads((manifests / f"{stage}.json").read_text(encoding="utf-8"))
        assert payload["status"] == "completed"
        if stage != "bootstrap-prior":
            assert payload["artifact_identity_fingerprint"]
        progress = case / "results/logs" / f"{stage}.jsonl"
        assert progress.is_file()
        assert progress.read_text(encoding="utf-8").strip()
        assert progress.resolve().is_relative_to((case / "results").resolve())
        events = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
        assert events[0]["event"]["phase"] == "initialization"
        assert events[0]["event"]["status"] == "started"
        assert all(record["schema_version"] == 1 for record in events)

    bootstrap_events = [
        json.loads(line)["event"]
        for line in (case / "results/logs/bootstrap-prior.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert bootstrap_events[0]["status"] == "started"
    assert any(event["status"] == "running" for event in bootstrap_events)
    assert bootstrap_events[-1]["status"] == "completed"
    assert all(event.get("stage") == "bootstrap-prior" for event in bootstrap_events)
    checkpoint = json.loads(
        (case / "results/audit/prior_demand_generation.json").read_text(
            encoding="utf-8"
        )
    )["checkpoint_directory"]
    checkpoint_manifest = json.loads(
        (Path(checkpoint) / "manifest.json").read_text(encoding="utf-8")
    )
    assert checkpoint_manifest["status"] == "completed"

    fit = json.loads((manifests / "fit.json").read_text(encoding="utf-8"))
    assert fit["result"]["status"] == "iteration_limit"
    assert json.loads((manifests / "validate.json").read_text(encoding="utf-8"))["acceptance"] == "diagnostic_only"
    prior_header = (case / "inputs/scenario/prior_demand.csv").read_text(encoding="utf-8").splitlines()[0]
    assert prior_header == "origin_stop_id,dest_stop_id,time_bin_id,flow"
    prior_audit = json.loads((case / "results/audit/prior_demand_generation.json").read_text(encoding="utf-8"))
    assert prior_audit["prior_source"] == "all_ones"
    assert prior_audit["expansion"]["retained_cell_count"] > 0
    assert prior_audit["output_sha256"]
    support_audit = json.loads(
        (case / "results/audit/feasibility_support.json").read_text(encoding="utf-8")
    )
    assert support_audit["status"] == "completed"
    assert support_audit["unsupported_retained_cells"] == 0
    assert support_audit["cells_present_only_in_bootstrap_support"] == 0
    assert support_audit["cells_present_only_in_feature_construction_support"] == 0
    assert support_audit["contract"]["fingerprint"] == prior_audit["expansion"]["configuration"]["feasibility_contract_fingerprint"]
    support_cells = case / "results/audit/feasibility_support_cells.jsonl"
    assert support_audit["unsupported_cells_path"] == str(support_cells)
    assert support_cells.is_file()


def test_check_writes_initial_progress_before_slow_context_loading(tmp_path: Path) -> None:
    repository = Path(__file__).parents[2]
    template = repository / "docs/source/examples/direct_scheduled_case_template"
    example = repository / "docs/source/examples/simple_example_02"
    case = tmp_path / "case"
    shutil.copytree(template, case)
    shutil.copytree(example / "data", case / "inputs/scenario")
    shutil.copy(example / "data/fixed_demand.csv", case / "inputs/fixed_demand.csv")
    shutil.copy(
        example / "pre_processing/results/measurements_boarding_alighting.csv",
        case / "inputs/measurements.csv",
    )
    case_config = case / "config/case.toml"
    case_config.write_text(
        case_config.read_text(encoding="utf-8").replace(
            "REPLACE_WITH_PUBLIC_TRANSPORTATION_COMMIT",
            "public-fixture-progress-test-revision",
        ),
        encoding="utf-8",
    )

    sys.path.insert(0, str(case))
    try:
        namespace = runpy.run_path(str(case / "run_case.py"), run_name="progress_test")
        check_globals = namespace["check"].__globals__
        original_load_context = check_globals["load_context"]
        observed: dict[str, object] = {}

        def delayed_load_context(root: Path, **kwargs: object) -> object:
            progress_path = case / "results/logs/check.jsonl"
            assert progress_path.is_file()
            first = json.loads(progress_path.read_text(encoding="utf-8").splitlines()[0])
            observed["first"] = first["event"]
            time.sleep(0.01)
            return original_load_context(root, **kwargs)

        check_globals["load_context"] = delayed_load_context
        namespace["check"](case)
    finally:
        sys.path.remove(str(case))

    assert observed["first"] == {
        "current_unit": "load_context",
        "elapsed_seconds": 0.0,
        "estimated_remaining_seconds": None,
        "eta_confidence": "unavailable",
        "phase": "initialization",
        "schema_version": 1,
        "stage": "check",
        "status": "started",
    }


def test_stage_progress_heartbeats_do_not_fabricate_work_or_eta(tmp_path: Path) -> None:
    repository = Path(__file__).parents[2]
    template = repository / "docs/source/examples/direct_scheduled_case_template"
    example = repository / "docs/source/examples/simple_example_02"
    case = tmp_path / "case"
    shutil.copytree(template, case)
    shutil.copytree(example / "data", case / "inputs/scenario")
    shutil.copy(example / "data/fixed_demand.csv", case / "inputs/fixed_demand.csv")
    shutil.copy(
        example / "pre_processing/results/measurements_boarding_alighting.csv",
        case / "inputs/measurements.csv",
    )
    case_config = case / "config/case.toml"
    case_config.write_text(
        case_config.read_text(encoding="utf-8")
        .replace("REPLACE_WITH_PUBLIC_TRANSPORTATION_COMMIT", "heartbeat-test-revision")
        .replace("progress_interval_seconds = 5.0", "progress_interval_seconds = 0.005"),
        encoding="utf-8",
    )

    sys.path.insert(0, str(case))
    try:
        namespace = runpy.run_path(str(case / "run_case.py"), run_name="heartbeat_test")
        settings = namespace["CaseSettings"].load(case)
        progress = namespace["_StageProgress"](settings, "heartbeat")
        progress.start()
        progress.phase_started("mapping_construction", "mapping_construction")
        time.sleep(0.03)
        progress.finish("completed")
    finally:
        sys.path.remove(str(case))

    events = [
        json.loads(line)["event"]
        for line in (case / "results/logs/heartbeat.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    heartbeats = [event for event in events if event.get("heartbeat") is True]
    assert heartbeats
    assert all("completed_units" not in event for event in heartbeats)
    assert all("total_units" not in event for event in heartbeats)
    assert all(event["estimated_remaining_seconds"] is None for event in heartbeats)
    assert all(event["eta_confidence"] == "unavailable" for event in heartbeats)
    assert events[-1]["status"] == "completed"


def test_check_reports_generated_fixed_demand_provenance(tmp_path: Path) -> None:
    case = _fixed_demand_provenance_case(tmp_path, generated_values=(19, 43))
    context, manifest, observed = _load_context_and_check_manifest(case)
    generated = (case / "results/structural_zeros/fixed_demand.csv").resolve()
    expected_sha256 = hashlib.sha256(generated.read_bytes()).hexdigest()

    assert observed["path"] == generated
    assert context.fixed_demand_path == generated
    assert context.fixed_demand_source == "generated_structural_zeros"
    assert context.fixed_demand_sha256 == expected_sha256
    assert sorted(context.parameter_layout.fixed_od_values) == [19.0, 43.0]
    assert manifest["fixed_demand"] == str(generated)
    assert manifest["fixed_demand_source"] == "generated_structural_zeros"
    assert manifest["fixed_demand_sha256"] == expected_sha256


def test_check_reports_configured_fixed_demand_fallback_provenance(
    tmp_path: Path,
) -> None:
    case = _fixed_demand_provenance_case(tmp_path, generated_values=None)
    context, manifest, observed = _load_context_and_check_manifest(case)
    fallback = (case / "inputs/fixed_demand.csv").resolve()
    expected_sha256 = hashlib.sha256(fallback.read_bytes()).hexdigest()

    assert observed["path"] == fallback
    assert context.fixed_demand_path == fallback
    assert context.fixed_demand_source == "case_config_fallback"
    assert context.fixed_demand_sha256 == expected_sha256
    assert sorted(context.parameter_layout.fixed_od_values) == [18.0, 42.0]
    assert manifest["fixed_demand"] == str(fallback)
    assert manifest["fixed_demand_source"] == "case_config_fallback"
    assert manifest["fixed_demand_sha256"] == expected_sha256


def test_check_failure_writes_terminal_progress_and_manifest(tmp_path: Path) -> None:
    repository = Path(__file__).parents[2]
    template = repository / "docs/source/examples/direct_scheduled_case_template"
    case = tmp_path / "case"
    shutil.copytree(template, case)
    case_config = case / "config/case.toml"
    case_config.write_text(
        case_config.read_text(encoding="utf-8").replace(
            "REPLACE_WITH_PUBLIC_TRANSPORTATION_COMMIT", "failure-progress-test-revision"
        ),
        encoding="utf-8",
    )

    sys.path.insert(0, str(case))
    try:
        namespace = runpy.run_path(str(case / "run_case.py"), run_name="failure_test")

        def fail_load_context(root: Path, **kwargs: object) -> object:
            raise RuntimeError("deliberate context failure")

        namespace["check"].__globals__["load_context"] = fail_load_context
        with pytest.raises(RuntimeError, match="deliberate context failure"):
            namespace["check"](case)
    finally:
        sys.path.remove(str(case))

    manifest = json.loads((case / "results/manifests/check.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "RuntimeError"
    events = [
        json.loads(line)["event"]
        for line in (case / "results/logs/check.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["phase"] == "initialization"
    assert events[-1]["phase"] == "stage_completion"
    assert events[-1]["status"] == "failed"
    assert events[-1]["error_type"] == "RuntimeError"


def test_template_bootstrap_interrupts_resumes_and_publishes_only_on_completion(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[2]
    template = repository / "docs/source/examples/direct_scheduled_case_template"
    example = repository / "docs/source/examples/simple_example_02"
    case = tmp_path / "case"
    shutil.copytree(template, case)
    shutil.copytree(example / "data", case / "inputs/scenario")
    (case / "inputs/scenario/prior_demand.csv").unlink()
    shutil.copy(example / "data/fixed_demand.csv", case / "inputs/fixed_demand.csv")
    shutil.copy(
        example / "pre_processing/results/measurements_boarding_alighting.csv",
        case / "inputs/measurements.csv",
    )
    case_config = case / "config/case.toml"
    case_config.write_text(
        case_config.read_text(encoding="utf-8").replace(
            "REPLACE_WITH_PUBLIC_TRANSPORTATION_COMMIT",
            "public-fixture-resume-test-revision",
        ),
        encoding="utf-8",
    )
    sys.path.insert(0, str(case))
    try:
        namespace = runpy.run_path(str(case / "run_case.py"), run_name="template_resume_test")
        events: list[dict[str, object]] = []

        def interrupt(event: dict[str, object]) -> None:
            events.append(dict(event))
            if event.get("status") == "running" and "completed_chunks" in event:
                raise KeyboardInterrupt

        with pytest.raises(ODTimeExpansionInterrupted):
            namespace["bootstrap_prior_demand"](case, progress=interrupt)
        assert not (case / "inputs/scenario/prior_demand.csv").exists()
        interrupted_manifest = json.loads(
            (
                case / "results/checkpoints/prior_demand"
            ).glob("*/manifest.json").__next__().read_text(encoding="utf-8")
        )
        assert interrupted_manifest["status"] == "interrupted"
        resumed_events: list[dict[str, object]] = []
        audit = namespace["bootstrap_prior_demand"](
            case, resume=True, progress=lambda event: resumed_events.append(dict(event))
        )
    finally:
        sys.path.remove(str(case))
    assert (case / "inputs/scenario/prior_demand.csv").is_file()
    assert audit["expansion"]["checkpoint_reused"] is True
    assert any(event.get("status") == "resuming" for event in resumed_events)
