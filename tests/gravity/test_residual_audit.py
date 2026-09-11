from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from public_transportation.inference.residual_audit import (
    ResidualAuditConfig,
    ResidualAuditRun,
    audit_run,
    compare_runs,
    compute_residual_diagnostics,
    load_run,
    poisson_deviance_residual,
    summarize_by,
    write_residual_audit,
)
from public_transportation.inference.residual_audit.cli import main


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_id": ["m1", "m2", "m3", "m4"],
            "observed_value": [10.0, 0.0, 3.0, 2.0],
            "predicted_value": [9.0, 0.0, 2.0, 1.0e-12],
            "measurement_type": ["boarding", "alighting", "boarding", "alighting"],
            "line": ["L1", "L1", "L2", "L2"],
            "journey_position": ["first", "last", "first", "last"],
        }
    )


def _contributions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_id": ["m1", "m2", "m3", "m4"],
            "od_contribution": [9.0, 0.0, 2.0, 0.0],
            "initial_onboard_contribution": [0.0, 0.0, 0.0, 1.0],
            "terminal_outflow_contribution": [0.0, 0.0, 0.0, 0.0],
            "fixed_offset": [0.0, 0.0, 0.0, 0.0],
            "scaled_mean": [9.0, 0.0, 2.0, 1.0],
        }
    )


def test_exact_predictions_have_zero_residuals() -> None:
    frame = pd.DataFrame(
        {
            "observation_id": [1, 2],
            "observed_value": [2.0, 0.0],
            "predicted_value": [2.0, 0.0],
        }
    )
    result = compute_residual_diagnostics(frame)
    assert np.array_equal(result["raw_residual"], [0.0, 0.0])
    assert np.array_equal(result["pearson_residual"], [0.0, 0.0])
    assert np.array_equal(result["poisson_deviance_residual"], [0.0, 0.0])


def test_positive_near_zero_predictions_are_support_failures() -> None:
    result = compute_residual_diagnostics(_observations())
    row = result.loc[result["observation_id"] == "m4"].iloc[0]
    assert bool(row["positive_observation"])
    assert bool(row["near_zero_prediction"])
    assert bool(row["support_failure"])


def test_zero_observation_and_prediction_are_safe() -> None:
    result = poisson_deviance_residual(np.array([0.0]), np.array([0.0]))
    assert result.tolist() == [0.0]


def test_relative_residual_is_unavailable_for_small_denominator() -> None:
    frame = pd.DataFrame(
        {"observation_id": [1], "observed_value": [1.0e-12], "predicted_value": [0.0]}
    )
    result = compute_residual_diagnostics(
        frame, config=ResidualAuditConfig(relative_observation_threshold=1.0e-8)
    )
    assert pd.isna(result.loc[0, "relative_residual"])


def test_grouped_totals_equal_ungrouped_totals() -> None:
    diagnostics = compute_residual_diagnostics(_observations())
    grouped = summarize_by(diagnostics, ["measurement_type", "line"])
    assert len(summarize_by(_observations(), ["measurement_type"])) == 2
    overall = summarize_by(diagnostics)
    assert grouped["observation_count"].sum() == overall.loc[0, "observation_count"]
    assert np.isclose(grouped["observed_total"].sum(), overall.loc[0, "observed_total"])
    assert np.isclose(grouped["predicted_total"].sum(), overall.loc[0, "predicted_total"])
    assert np.isclose(grouped["raw_residual_total"].sum(), overall.loc[0, "raw_residual_total"])


def test_metadata_can_be_supplied_separately_to_the_public_api() -> None:
    observations = _observations().drop(columns=["measurement_type", "line", "journey_position"])
    metadata = _observations()[["observation_id", "measurement_type", "line", "journey_position"]]
    result = audit_run(
        ResidualAuditRun(observations, metadata=metadata),
        group_by=["measurement_type"],
    )
    assert set(result.grouped_metrics["group_key"]) == {"boarding", "alighting"}


def test_missing_grouping_column_is_clear() -> None:
    diagnostics = compute_residual_diagnostics(_observations())
    with pytest.raises(ValueError, match="grouping column"):
        summarize_by(diagnostics, ["not_present"])


def test_duplicate_observation_ids_are_rejected() -> None:
    frame = _observations()
    frame.loc[1, "observation_id"] = frame.loc[0, "observation_id"]
    with pytest.raises(ValueError, match="duplicate"):
        compute_residual_diagnostics(frame)


def test_missing_columns_and_invalid_numeric_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="required column"):
        compute_residual_diagnostics(pd.DataFrame({"observation_id": [1], "observed_value": [1.0]}))
    invalid = pd.DataFrame(
        {
            "observation_id": [1],
            "observed_value": ["not-a-number"],
            "predicted_value": [1.0],
        }
    )
    with pytest.raises(ValueError, match="numeric"):
        compute_residual_diagnostics(invalid)


def test_contribution_totals_are_checked() -> None:
    result = audit_run(ResidualAuditRun(_observations(), contributions=_contributions()))
    assert result.manifest["contribution_analysis"]["available"] is True
    check = result.manifest["contribution_analysis"]["contribution_check"]
    assert check["rows_not_matching"] == 0
    assert set(result.contribution_summary["component"]) >= {
        "od_contribution",
        "initial_onboard_contribution",
    }
    assert "measurement_type=boarding" in set(result.contribution_summary["scope"])


def test_cross_run_rejects_mismatched_observation_sets() -> None:
    left = _observations()
    right = _observations().iloc[:-1].copy()
    with pytest.raises(ValueError, match="different observation IDs"):
        compare_runs(left, right)


def test_subset_comparison_requires_explicit_opt_in() -> None:
    left = _observations()
    right = _observations().iloc[:-1].copy()
    comparison = compare_runs(left, right, allow_subset_comparison=True)
    assert len(comparison.rows) == 3


def test_cross_run_detects_changed_observed_values() -> None:
    left = _observations()
    right = _observations()
    right.loc[0, "observed_value"] = 11.0
    with pytest.raises(ValueError, match="different observed values"):
        compare_runs(left, right)


def test_comparison_reports_prediction_and_residual_changes() -> None:
    right = _observations().copy()
    right["predicted_value"] += 0.5
    comparison = compare_runs(_observations(), right, group_by=["line"])
    assert len(comparison.rows) == 4
    assert np.allclose(comparison.rows["prediction_difference"], 0.5)
    assert set(comparison.grouped_metrics["group_key"]) == {"L1", "L2"}
    failure = comparison.rows.loc[comparison.rows["observation_id"] == "m4"].iloc[0]
    assert bool(failure["support_failure_run_a"])
    assert not bool(failure["support_failure_run_b"])


def test_fingerprints_are_preserved_in_manifest(tmp_path: Path) -> None:
    predicted = tmp_path / "predicted_measurements.csv"
    _observations().to_csv(predicted, index=False)
    manifest = tmp_path / "report.json"
    manifest.write_text(
        json.dumps(
            {
                "model_fingerprint": "model-1",
                "specification_fingerprint": "spec-1",
            }
        ),
        encoding="utf-8",
    )
    run = load_run(predicted_measurements=predicted, model_manifest=manifest)
    result = write_residual_audit(run, output_directory=tmp_path / "audit")
    payload = json.loads((tmp_path / "audit" / "audit_manifest.json").read_text())
    assert payload["model_fingerprints"] == ["model-1"]
    assert payload["specification_fingerprints"] == ["spec-1"]
    assert payload["input_fingerprints"]["predicted_measurements"]
    assert result.output_files["summary.md"].is_file()


def test_empty_optional_contributions_and_metadata_are_supported(tmp_path: Path) -> None:
    predicted = tmp_path / "predicted.csv"
    _observations().to_csv(predicted, index=False)
    result = write_residual_audit(
        load_run(predicted_measurements=predicted), output_directory=tmp_path / "audit"
    )
    assert result.contribution_summary.empty
    assert (tmp_path / "audit" / "support_failures.csv").is_file()
    assert (tmp_path / "audit" / "run_comparison.csv").is_file()


def test_empty_optional_csv_inputs_are_supported(tmp_path: Path) -> None:
    predicted = tmp_path / "predicted.csv"
    empty_contributions = tmp_path / "contributions.csv"
    empty_metadata = tmp_path / "metadata.csv"
    _observations().to_csv(predicted, index=False)
    pd.DataFrame(columns=["observation_id", "od_contribution"]).to_csv(empty_contributions, index=False)
    pd.DataFrame(columns=["observation_id", "line"]).to_csv(empty_metadata, index=False)
    run = load_run(
        predicted_measurements=predicted,
        contributions=empty_contributions,
        metadata=empty_metadata,
    )
    assert run.contributions is None
    assert run.metadata is None


def test_persisted_report_aliases_preserve_source_row_index(tmp_path: Path) -> None:
    predicted = tmp_path / "predicted.csv"
    _observations().rename(columns={"observation_id": "row_index", "observed_value": "observed", "predicted_value": "modeled"}).to_csv(predicted, index=False)
    run = load_run(predicted_measurements=predicted)
    result = audit_run(run)
    assert "source_row_index" in result.row_diagnostics.columns
    assert result.row_diagnostics["source_row_index"].tolist() == ["m1", "m2", "m3", "m4"]


def test_cli_writes_documented_outputs_and_is_deterministic(tmp_path: Path) -> None:
    predicted = tmp_path / "predicted.csv"
    residuals = tmp_path / "residuals.csv"
    metadata = tmp_path / "metadata.csv"
    _observations().to_csv(predicted, index=False)
    _observations().assign(variance=[9.0, 0.0, 2.0, 1.0]).to_csv(residuals, index=False)
    _observations()[["observation_id", "measurement_type", "line", "journey_position"]].to_csv(metadata, index=False)
    output = tmp_path / "audit"
    args = [
        "--predicted-measurements",
        str(predicted),
        "--residuals",
        str(residuals),
        "--metadata",
        str(metadata),
        "--group-by",
        "measurement_type",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    first = (output / "summary.md").read_bytes()
    assert main([*args, "--force"]) == 0
    assert (output / "summary.md").read_bytes() == first
    assert {
        "audit_manifest.json",
        "summary.md",
        "row_diagnostics.csv",
        "grouped_metrics.csv",
        "support_failures.csv",
        "contribution_summary.csv",
        "run_comparison.csv",
    }.issubset({path.name for path in output.iterdir()})


def test_synthetic_integration_with_boundary_and_two_runs(tmp_path: Path) -> None:
    run_a = _observations()
    run_b = _observations().copy()
    run_b["predicted_value"] = [9.5, 0.0, 2.5, 1.0e-12]
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    run_a.to_csv(first / "predicted_measurements.csv", index=False)
    _contributions().to_csv(first / "measurement_contributions.csv", index=False)
    run_b.to_csv(second / "predicted_measurements.csv", index=False)
    result = write_residual_audit(
        load_run(
            predicted_measurements=first / "predicted_measurements.csv",
            contributions=first / "measurement_contributions.csv",
        ),
        comparison_run=load_run(predicted_measurements=second / "predicted_measurements.csv"),
        group_by=["measurement_type", "journey_position"],
        output_directory=tmp_path / "audit",
    )
    assert len(result.support_failures) == 1
    assert len(result.comparison) == 4
    assert result.manifest["comparison"]["available"] is True
    assert "No observations were automatically removed" in result.summary_markdown
