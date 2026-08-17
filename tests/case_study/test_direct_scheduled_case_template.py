from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path

import pytest

from public_transportation.preprocessing import ODTimeExpansionInterrupted


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
