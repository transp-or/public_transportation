from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    FixedRoutingMeasurementOperator,
    MeasurementOperatorMetrics,
)
from public_transportation.inference.gravity import (
    GravityAdequacyConfig,
    GravityEstimatorConfig,
    GravityExecutionPolicy,
    GravityFeatures,
    GravityLikelihood,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    GravityValidationMetadata,
    estimate_gravity_model,
    predict_gravity_measurements,
    validate_full_data_gravity_adequacy,
)


def validation_case(observation_shift=None):
    cells = 6
    layout = CompactODAssignmentLayout(
        cells,
        tuple(range(cells)),
        (),
        tuple(range(cells)),
        tuple(range(cells)),
        tuple(range(cells)),
        tuple(1.0 for _ in range(cells)),
        (),
        (),
    )
    features = GravityFeatures(
        canonical_od_index=np.arange(cells),
        origin_index=np.repeat((0, 1), 3),
        destination_index=np.tile(np.arange(3), 2),
        departure_time_index=np.repeat((0, 1), 3),
        origin_time_group_index=np.repeat((0, 1), 3),
        journey_time=np.asarray((2, 8, 15, 12, 3, 7), dtype=float),
        transfer_count=np.asarray((0, 1, 2, 2, 0, 1)),
        structural_feasible=np.ones(cells, dtype=bool),
        origin_time_totals=np.asarray((100.0, 80.0)),
        destination_attractiveness=np.asarray((1, 2, 1, 1, 1, 3), dtype=float),
        num_origins=2,
        num_destinations=3,
        num_departure_times=2,
        od_layout_fingerprint=layout.fingerprint,
        journey_time_scale=10,
    )
    matrix = jnp.eye(cells, dtype=jnp.float64)
    operator = FixedRoutingMeasurementOperator(
        matrix,
        jnp.zeros(cells),
        "dense",
        cells,
        cells,
        cells,
        layout.fingerprint,
        layout.fingerprint,
        "assignment",
        "graph",
        "mapping",
        1.0,
        "float64",
        MeasurementOperatorMetrics(
            0, matrix.nbytes, matrix.nbytes, 0, cells, cells**2, 1 / cells, cells
        ),
    )
    parameters = GravityParameterLayout(GravityModelSpecification())
    placeholder = GravityObjectiveProblem(
        features,
        parameters,
        operator,
        np.ones(cells),
        GravityLikelihood.NEGATIVE_BINOMIAL,
    )
    raw = parameters.raw_from_physical((0.6, 1.1, 5.0))
    mean = np.asarray(predict_gravity_measurements(raw, problem=placeholder)[0])
    observations = mean if observation_shift is None else mean + observation_shift
    problem = replace(placeholder, observations=observations, calibration_mask=None)
    result = estimate_gravity_model(
        problem=problem,
        compact_layout=layout,
        initial_raw_parameters=raw,
        config=GravityEstimatorConfig(maximum_iterations=1),
        execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
    )
    return problem, layout, result


def test_perfect_full_data_report_has_zero_residual_metrics():
    with jax.enable_x64():
        problem, layout, result = validation_case()
        report = validate_full_data_gravity_adequacy(
            result=result, problem=problem, compact_layout=layout
        )
        assert report.measurements == 6
        assert report.observed_total == pytest.approx(report.modeled_total)
        assert report.mae == pytest.approx(0, abs=1e-10)
        assert report.rmse == pytest.approx(0, abs=1e-10)
        assert report.poisson_deviance == pytest.approx(0, abs=1e-10)
        assert report.negative_binomial_deviance == pytest.approx(0, abs=1e-10)
        assert not report.residual.flags.writeable
        assert report.report_fingerprint


def test_grouped_residuals_thresholds_and_journey_correlations_are_reported():
    with jax.enable_x64():
        shift = np.asarray((8.0, 6.0, 4.0, -1.0, -2.0, -3.0))
        problem, layout, result = validation_case(shift)
        metadata = GravityValidationMetadata(
            6,
            measurement_type=np.asarray(("boarding",) * 3 + ("alighting",) * 3),
            line=np.asarray(("L1",) * 6),
            direction=np.asarray(("out",) * 3 + ("back",) * 3),
            stop=np.asarray(("A", "B", "C", "A", "B", "C")),
            time_period=np.asarray(("am",) * 6),
            origin_zone=np.asarray(("west",) * 3 + ("east",) * 3),
            destination_zone=np.asarray(("center", "center", "outer") * 2),
            vehicle_journey=np.asarray(("J1",) * 3 + ("J2",) * 3),
        )
        report = validate_full_data_gravity_adequacy(
            result=result,
            problem=problem,
            compact_layout=layout,
            metadata=metadata,
            config=GravityAdequacyConfig(
                standardized_residual_thresholds=(0.5, 1.0),
                systematic_group_mean_threshold=0.2,
                journey_correlation_threshold=0.0,
            ),
        )
        assert {item.grouping for item in report.grouped_summaries} == {
            "measurement_type",
            "line",
            "direction",
            "stop",
            "time_period",
            "origin_zone",
            "destination_zone",
        }
        boarding = next(
            item
            for item in report.grouped_summaries
            if item.grouping == "measurement_type" and item.label == "boarding"
        )
        indices = np.arange(3)
        assert boarding.observed_total == pytest.approx(
            float(np.sum(problem.observations[indices]))
        )
        assert len(report.threshold_counts) == 2
        assert len(report.journey_correlations) == 2
        assert report.findings.messages


def test_full_data_mode_rejects_holdout_and_fingerprint_mismatch():
    with jax.enable_x64():
        problem, layout, result = validation_case()
        holdout = replace(
            problem, calibration_mask=np.asarray((True, True, True, True, True, False))
        )
        with pytest.raises(ValueError, match="every measurement"):
            validate_full_data_gravity_adequacy(
                result=result, problem=holdout, compact_layout=layout
            )
        changed = replace(problem, observations=problem.observations + 1)
        with pytest.raises(ValueError, match="fingerprints differ"):
            validate_full_data_gravity_adequacy(
                result=result, problem=changed, compact_layout=layout
            )


def test_metadata_dimension_is_validated():
    with pytest.raises(ValueError, match="shape"):
        GravityValidationMetadata(3, line=np.asarray(("L1", "L2")))


def test_float32_recomputation_uses_float32_validation_tolerance():
    with jax.enable_x64():
        problem, layout, _ = validation_case()
        features32 = replace(
            problem.features,
            journey_time=problem.features.journey_time.astype(np.float32),
            origin_time_totals=problem.features.origin_time_totals.astype(np.float32),
            destination_attractiveness=problem.features.destination_attractiveness.astype(
                np.float32
            ),
        )
        matrix32 = jnp.asarray(problem.operator.matrix, dtype=jnp.float32)
        operator32 = replace(
            problem.operator,
            matrix=matrix32,
            fixed_measurement_offset=jnp.asarray(
                problem.operator.fixed_measurement_offset, dtype=jnp.float32
            ),
            dtype="float32",
        )
        problem32 = replace(problem, features=features32, operator=operator32)
        raw = problem32.parameter_layout.raw_from_physical((0.6, 1.1, 5.0))
        result = estimate_gravity_model(
            problem=problem32,
            compact_layout=layout,
            initial_raw_parameters=raw,
            config=GravityEstimatorConfig(maximum_iterations=1),
            execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
        )
        perturbed = replace(
            result,
            predicted_measurements=result.predicted_measurements
            + np.float32(2.0e-6),
        )
        report = validate_full_data_gravity_adequacy(
            result=perturbed, problem=problem32, compact_layout=layout
        )
        assert report.measurements == problem32.observations.size
