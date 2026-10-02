from __future__ import annotations

import json

import numpy as np
import pandas as pd

from public_transportation.inference.residual_audit import (
    ResidualBootstrapConfig,
    ResidualPracticalTolerances,
    analyze_measurement_residuals,
    write_measurement_residual_analysis,
)


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observed_value": [12.0, 11.0, 2.0, 1.0, 8.0, 7.0, 2.0, 1.0],
            "predicted_value": [8.0, 7.0, 2.0, 1.0, 8.0, 7.0, 4.0, 3.0],
            "variance": [8.0, 7.0, 2.0, 1.0, 8.0, 7.0, 4.0, 3.0],
            "measurement_type": ["boarding", "boarding", "alighting", "alighting"] * 2,
            "line": ["L1"] * 4 + ["L2"] * 4,
            "direction": ["out"] * 8,
            "stop": ["s1", "s2", "s1", "s2"] * 2,
            "time_period": ["am"] * 4 + ["pm"] * 4,
            "trip_id": ["t1", "t1", "t1", "t1", "t2", "t2", "t2", "t2"],
            "journey_position": list(range(1, 5)) * 2,
        }
    )


def test_residual_conventions_and_supplied_variance() -> None:
    frame = _fixture()
    analysis = analyze_measurement_residuals(
        frame,
        frame["predicted_value"],
        variance=frame["variance"],
        bootstrap_config={"replicates": 3},
        provenance={"likelihood_family": "poisson", "model_fingerprint": "m"},
    )
    assert analysis.row_diagnostics.loc[0, "raw_residual"] == 4.0
    assert analysis.row_diagnostics.loc[0, "absolute_residual"] == 4.0
    assert np.isclose(analysis.row_diagnostics.loc[0, "pearson_residual"], 4.0 / np.sqrt(8.0))
    assert analysis.provenance["variance_convention"] == "caller_supplied_variance"
    reversed_analysis = analyze_measurement_residuals(
        frame,
        frame["predicted_value"],
        residual_convention="modeled_minus_observed",
        variance=frame["variance"],
        bootstrap_config={"replicates": 2},
    )
    assert reversed_analysis.row_diagnostics.loc[0, "raw_residual"] == -4.0


def test_negative_binomial_variance_and_deviance() -> None:
    frame = _fixture()
    analysis = analyze_measurement_residuals(
        frame.drop(columns="variance"),
        frame["predicted_value"],
        provenance={"likelihood_family": "negative_binomial", "dispersion": 4.0},
        bootstrap_config={"replicates": 2},
    )
    expected = frame.loc[0, "predicted_value"] + frame.loc[0, "predicted_value"] ** 2 / 4.0
    assert np.isclose(analysis.row_diagnostics.loc[0, "variance"], expected)
    assert analysis.provenance["deviance_convention"] == "negative_binomial_deviance"


def test_groupings_contain_required_dimensions_and_metrics() -> None:
    analysis = analyze_measurement_residuals(
        _fixture(),
        _fixture()["predicted_value"],
        variance=_fixture()["variance"],
        bootstrap_config={"replicates": 3},
    )
    groupings = set(analysis.grouped_diagnostics["grouping"])
    assert {"overall", "measurement_type", "line_direction", "stop", "time_period", "line_time_period", "stop_measurement_type", "trip_id"} <= groupings
    required = {
        "observation_count",
        "independent_cluster_count",
        "observed_total",
        "modeled_total",
        "signed_residual_total",
        "relative_total_bias",
        "mae",
        "rmse",
        "weighted_rmse",
        "mean_pearson_residual",
        "mean_deviance_residual",
        "fraction_positive_residual",
        "fraction_negative_residual",
        "maximum_absolute_standardized_residual",
        "minimum_fitted_value",
        "maximum_fitted_value",
        "zero_observation_count",
    }
    assert required <= set(analysis.grouped_diagnostics.columns)


def test_cluster_bootstrap_is_reproducible_and_journey_diagnostics_are_present() -> None:
    frame = _fixture()
    config = ResidualBootstrapConfig(replicates=5, seed=19)
    first = analyze_measurement_residuals(frame, frame["predicted_value"], variance=frame["variance"], bootstrap_config=config)
    second = analyze_measurement_residuals(frame, frame["predicted_value"], variance=frame["variance"], bootstrap_config=config)
    pd.testing.assert_frame_equal(first.grouped_diagnostics, second.grouped_diagnostics)
    assert first.manifest["uncertainty_method"] == "cluster_bootstrap"
    assert not first.journey_diagnostics.empty
    assert {"adjacent_residual_correlation", "sign_run_lengths", "cumulative_residual", "cumulative_boarding_alighting_imbalance"} <= set(first.journey_diagnostics.columns)


def test_missing_cluster_requires_explicit_observation_bootstrap() -> None:
    frame = _fixture().drop(columns=["trip_id", "journey_position"])
    unavailable = analyze_measurement_residuals(frame, frame["predicted_value"], variance=frame["variance"], bootstrap_config={"replicates": 2})
    assert unavailable.manifest["uncertainty_method"] == "unavailable"
    assert any("No cluster identifier" in warning for warning in unavailable.manifest["warnings"])
    available = analyze_measurement_residuals(
        frame,
        frame["predicted_value"],
        variance=frame["variance"],
        bootstrap_config={"replicates": 2, "allow_observation_level": True},
    )
    assert available.manifest["uncertainty_method"] == "observation_bootstrap"


def test_candidate_and_suggestion_are_advisory() -> None:
    frame = _fixture()
    analysis = analyze_measurement_residuals(
        frame,
        frame["predicted_value"],
        variance=frame["variance"],
        bootstrap_config={"replicates": 4},
        practical_tolerances=ResidualPracticalTolerances(
            minimum_observations=2,
            minimum_clusters=2,
            minimum_absolute_bias=0.1,
            minimum_relative_bias=0.01,
        ),
    )
    assert analysis.manifest["advisory_only"] is True
    assert "in-sample" in analysis.manifest["interpretation"]
    assert any(item["label"] == "candidate_systematic_bias" for item in analysis.candidate_patterns)
    assert all("suggested_action" in item and "caveat" in item for item in analysis.suggestions)


def test_serialization_round_trip_and_no_mutation(tmp_path) -> None:
    frame = _fixture()
    original = frame.copy(deep=True)
    analysis = analyze_measurement_residuals(
        frame,
        frame["predicted_value"],
        variance=frame["variance"],
        bootstrap_config={"replicates": 2},
        provenance={"model_fingerprint": "model", "artifact_fingerprint": "artifact", "package_revision": "rev"},
    )
    written = write_measurement_residual_analysis(analysis, tmp_path)
    assert set(written.output_files) == {
        "residual_analysis.json",
        "grouped_residual_analysis.csv",
        "journey_residual_analysis.csv",
        "residual_suggestions.json",
        "residual_suggestions.md",
        "residual_analysis_manifest.json",
    }
    manifest = json.loads((tmp_path / "residual_analysis_manifest.json").read_text())
    assert manifest["analysis_fingerprint"] == analysis.analysis_fingerprint
    assert manifest["provenance"]["model_fingerprint"] == "model"
    pd.testing.assert_frame_equal(frame, original)
