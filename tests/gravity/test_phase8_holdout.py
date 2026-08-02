from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import pytest

from public_transportation.inference.gravity import (
    GravityEstimatorConfig,
    GravityExecutionPolicy,
    GravityHoldoutSplit,
    GravityHoldoutSplitConfig,
    GravityValidationMetadata,
    build_gravity_holdout_split,
    estimate_and_validate_gravity_holdout,
    gravity_measurement_identity,
    predict_gravity_measurements,
)
from tests.gravity.test_phase6_recommendations import recommendation_case


def split_metadata() -> GravityValidationMetadata:
    return GravityValidationMetadata(
        8,
        measurement_type=np.asarray(("boarding", "alighting") * 4),
        line=np.asarray(("L1",) * 4 + ("L2",) * 4),
        direction=np.asarray(("out",) * 4 + ("back",) * 4),
        stop=np.asarray(("A", "A", "B", "B", "A", "A", "B", "B")),
        time_period=np.asarray(("am", "am", "pm", "pm") * 2),
        origin_zone=np.asarray(("west",) * 4 + ("east",) * 4),
        destination_zone=np.asarray(("center", "outer") * 4),
        vehicle_journey=np.asarray(("J1",) * 2 + ("J2",) * 2 + ("J3",) * 2 + ("J4",) * 2),
    )


def identity() -> str:
    return gravity_measurement_identity(
        measurement_indices=np.arange(8), label="phase-8 measurement rows"
    )


@pytest.mark.parametrize(
    ("unit", "field"),
    (
        ("vehicle_journey", "vehicle_journey"),
        ("stop_time_series", "stop"),
        ("line", "line"),
        ("direction", "direction"),
        ("time_block", "time_period"),
    ),
)
def test_supported_units_hold_out_complete_groups_deterministically(unit, field):
    metadata = split_metadata()
    config = GravityHoldoutSplitConfig(
        unit=unit,
        holdout_fraction=0.4,
        seed=17,
        stratify_by=("measurement_type",),
    )
    first = build_gravity_holdout_split(
        metadata=metadata, measurement_identity=identity(), config=config
    )
    second = build_gravity_holdout_split(
        metadata=metadata, measurement_identity=identity(), config=config
    )
    np.testing.assert_array_equal(first.holdout_mask, second.holdout_mask)
    assert first.split_fingerprint == second.split_fingerprint
    labels = getattr(metadata, field)
    assert labels is not None
    for label in np.unique(labels):
        local = first.holdout_mask[labels == label]
        assert np.all(local) or not np.any(local)
    assert np.any(first.calibration_mask)
    assert np.any(first.holdout_mask)
    assert not np.any(first.calibration_mask & first.holdout_mask)


def test_explicit_group_split_is_serializable_fingerprinted_and_stratified():
    metadata = split_metadata()
    labels = np.asarray(("g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"))
    split = build_gravity_holdout_split(
        metadata=metadata,
        measurement_identity=identity(),
        config=GravityHoldoutSplitConfig(
            unit="explicit_group",
            holdout_fraction=0.25,
            seed=91,
            stratify_by=("line", "time_period", "origin_zone"),
        ),
        explicit_group_labels=labels,
    )
    restored = GravityHoldoutSplit.from_dict(split.to_dict())
    assert restored.split_fingerprint == split.split_fingerprint
    np.testing.assert_array_equal(restored.calibration_mask, split.calibration_mask)
    assert not restored.calibration_mask.flags.writeable
    corrupted = split.to_dict()
    corrupted["holdout_mask"] = split.calibration_mask.tolist()
    with pytest.raises(ValueError, match="complementary"):
        GravityHoldoutSplit.from_dict(corrupted)


def test_holdout_reestimation_has_no_observation_leakage_and_scores_separately():
    with jax.enable_x64():
        problem, compact, original = recommendation_case(destination_effect=0.4)
        labels = np.asarray(("g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"))
        split = build_gravity_holdout_split(
            metadata=split_metadata(),
            measurement_identity=identity(),
            config=GravityHoldoutSplitConfig(
                unit="explicit_group", holdout_fraction=0.25, seed=5
            ),
            explicit_group_labels=labels,
        )
        settings = {
            "compact_layout": compact,
            "split": split,
            "measurement_identity": identity(),
            "initial_raw_parameters": original.raw_parameters,
            "estimator_config": GravityEstimatorConfig(maximum_iterations=8),
            "execution_policy": GravityExecutionPolicy(gradient_strategy="adjoint"),
        }
        first = estimate_and_validate_gravity_holdout(problem=problem, **settings)
        changed_observations = problem.observations.copy()
        changed_observations[split.holdout_mask] += 1000
        changed_problem = replace(problem, observations=changed_observations)
        second = estimate_and_validate_gravity_holdout(
            problem=changed_problem, **settings
        )
        np.testing.assert_allclose(
            first.estimation_result.raw_parameters,
            second.estimation_result.raw_parameters,
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            first.predicted_measurements,
            second.predicted_measurements,
            rtol=0,
            atol=1e-10,
        )
        assert first.holdout.rmse != second.holdout.rmse
        assert first.calibration == second.calibration
        assert first.calibration.measurements == int(np.count_nonzero(split.calibration_mask))
        assert first.holdout.measurements == int(np.count_nonzero(split.holdout_mask))
        assert first.free_od_demand.shape == (problem.features.num_cells,)
        assert first.full_od_demand.shape == (compact.num_od_total,)
        direct = np.asarray(
            predict_gravity_measurements(
                first.estimation_result.raw_parameters, problem=problem
            )[0]
        )
        np.testing.assert_allclose(first.predicted_measurements, direct)


def test_split_reuse_and_invalid_inputs_are_rejected_explicitly():
    metadata = split_metadata()
    config = GravityHoldoutSplitConfig(unit="vehicle_journey", seed=2)
    split = build_gravity_holdout_split(
        metadata=metadata, measurement_identity=identity(), config=config
    )
    assert split.measurement_identity == identity()
    with pytest.raises(ValueError, match="required"):
        build_gravity_holdout_split(
            metadata=metadata,
            measurement_identity=identity(),
            config=GravityHoldoutSplitConfig(unit="explicit_group"),
        )
    with pytest.raises(ValueError, match="valid only"):
        build_gravity_holdout_split(
            metadata=metadata,
            measurement_identity=identity(),
            config=config,
            explicit_group_labels=np.arange(8),
        )
    with pytest.raises(ValueError, match="stratification"):
        build_gravity_holdout_split(
            metadata=replace(metadata, destination_zone=None),
            measurement_identity=identity(),
            config=GravityHoldoutSplitConfig(
                unit="vehicle_journey", stratify_by=("destination_zone",)
            ),
        )
