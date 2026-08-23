from __future__ import annotations

import json
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
    GravityAggregateBin,
    GravityAggregateHistogram,
    GravityAggregateObservation,
    GravityAggregateObservationChannel,
    GravityAggregateStratum,
    GravityAggregateUncertainty,
    GravityAttributeResponseOperator,
    GravityAttributeResponseProvenance,
    GravityEstimatorConfig,
    GravityExecutionPolicy,
    GravityFeatures,
    GravityGradientStrategy,
    GravityLikelihood,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    GravityObservationBundle,
    build_gravity_aggregate_observation_bundle,
    build_gravity_run_manifest,
    estimate_gravity_model,
    evaluate_gravity_objective,
    gravity_model_fingerprint,
    gravity_value_and_gradient,
)


def setup_problem() -> tuple[GravityObjectiveProblem, CompactODAssignmentLayout]:
    cells = 4
    layout = CompactODAssignmentLayout(
        num_od_total=cells,
        active_full_indices=tuple(range(cells)),
        removed_zero_full_indices=(),
        full_to_compact=tuple(range(cells)),
        free_full_indices=tuple(range(cells)),
        free_compact_indices=tuple(range(cells)),
        free_baseline_values=tuple(1.0 for _ in range(cells)),
        fixed_compact_indices=(),
        fixed_compact_values=(),
    )
    features = GravityFeatures(
        canonical_od_index=np.arange(cells),
        origin_index=np.asarray((0, 0, 1, 1)),
        destination_index=np.asarray((0, 1, 0, 1)),
        departure_time_index=np.asarray((0, 0, 0, 0)),
        origin_time_group_index=np.asarray((0, 0, 1, 1)),
        journey_time=np.asarray((5.0, 15.0, 12.0, 7.0), dtype=np.float64),
        transfer_count=np.asarray((0, 1, 2, 0)),
        structural_feasible=np.ones(cells, dtype=bool),
        origin_time_totals=np.asarray((20.0, 30.0), dtype=np.float64),
        destination_attractiveness=np.asarray((1.0, 3.0, 2.0, 1.0)),
        num_origins=2,
        num_destinations=2,
        num_departure_times=1,
        od_layout_fingerprint=layout.fingerprint,
        journey_time_scale=10.0,
    )
    matrix = jnp.asarray(
        ((1.0, 0.0, 0.5, 0.0), (0.0, 2.0, 1.0, 1.5), (0.2, 0.0, 0.0, 1.0)),
        dtype=jnp.float64,
    )
    operator = FixedRoutingMeasurementOperator(
        matrix=matrix,
        fixed_measurement_offset=jnp.asarray((2.0, 0.0, 1.0), dtype=matrix.dtype),
        representation="dense",
        num_active_od=cells,
        num_free_od=cells,
        num_measurements=3,
        od_layout_fingerprint=layout.fingerprint,
        compact_layout_fingerprint=layout.fingerprint,
        assignment_fingerprint="assignment",
        graph_fingerprint="graph",
        mapping_fingerprint="mapping",
        theta=1.0,
        dtype="float64",
        metrics=MeasurementOperatorMetrics(
            construction_seconds=0.0,
            dense_bytes=int(matrix.nbytes),
            stored_bytes=int(matrix.nbytes),
            peak_construction_bytes=0,
            nonzero_entries=8,
            total_entries=12,
            density=8 / 12,
            chunk_size=4,
        ),
    )
    return (
        GravityObjectiveProblem(
            features=features,
            parameter_layout=GravityParameterLayout(GravityModelSpecification()),
            operator=operator,
            observations=np.asarray((19.0, 31.0, 9.0)),
            likelihood=GravityLikelihood.POISSON,
        ),
        layout,
    )


def aggregate() -> GravityAggregateObservation:
    histogram = GravityAggregateHistogram(
        attribute="travel_time",
        unit="seconds",
        support=(0.0, 7200.0),
        bins=(
            GravityAggregateBin("short", 0.0, 600.0),
            GravityAggregateBin("long", 600.0, 7200.0),
        ),
        strata=(
            GravityAggregateStratum("morning", (3, 1), 4),
            GravityAggregateStratum("evening", (1, 3), 4),
        ),
    )
    return GravityAggregateObservation(
        schema_version=1,
        channel_name="gps_trip_attributes",
        kind="trip_attribute_distribution",
        histograms=(histogram,),
        metadata={
            "collection_period": "fixture",
            "valid_journeys": 8,
            "excluded_journeys": 0,
            "cleaning_reasons": {},
            "apc_overlap_policy": "recorded",
        },
        uncertainty=GravityAggregateUncertainty(
            "dirichlet_multinomial", concentration=12.0
        ),
        source_path="fixture.json",
        file_sha256="file",
        content_sha256="content",
    )


def attribute_operator() -> GravityAttributeResponseOperator:
    categories = (
        ("morning", "short"),
        ("morning", "long"),
        ("evening", "short"),
        ("evening", "long"),
    )
    return GravityAttributeResponseOperator.from_route_shares(
        attribute="travel_time",
        unit="seconds",
        category_labels=categories,
        route_shares=(
            (
                # A cell with two paths contributes to both bins.
                ("morning", "short", 0.7),
                ("morning", "long", 0.3),
            ),
            (
                ("morning", "short", 0.1),
                ("morning", "long", 0.9),
            ),
            (
                ("evening", "short", 0.4),
                ("evening", "long", 0.6),
            ),
            (
                ("evening", "short", 0.8),
                ("evening", "long", 0.2),
            ),
        ),
        fixed_attribute_offset=np.asarray((1.0, 2.0, 0.5, 0.5)),
        provenance=GravityAttributeResponseProvenance(
            od_layout_fingerprint="gravity-layout",
            assignment_fingerprint="assignment",
            graph_fingerprint="graph",
            timetable_fingerprint="timetable",
            feasibility_fingerprint="feasibility",
        ),
    )


def with_channel(problem: GravityObjectiveProblem) -> GravityObjectiveProblem:
    data = aggregate()
    response = attribute_operator()
    bundle = build_gravity_aggregate_observation_bundle(
        data,
        {"travel_time": response},
        num_free_od=problem.operator.num_free_od,
        dtype=np.dtype("float64"),
    )
    return replace(problem, auxiliary_observations=bundle)


def test_auxiliary_log_likelihood_is_added_without_changing_count_component():
    with jax.enable_x64():
        base, _ = setup_problem()
        enriched = with_channel(base)
        raw = np.asarray((0.2, -0.1, 1.0))
        count = evaluate_gravity_objective(raw, problem=base)
        total = evaluate_gravity_objective(raw, problem=enriched)

        assert float(total.auxiliary_log_likelihood) != 0.0
        np.testing.assert_allclose(
            total.data_log_likelihood,
            count.data_log_likelihood + total.auxiliary_log_likelihood,
        )
        np.testing.assert_allclose(
            total.count_log_likelihood, count.data_log_likelihood
        )
        np.testing.assert_allclose(
            total.objective,
            -total.data_log_likelihood + total.regularization,
        )
        assert len(total.auxiliary_channel_log_likelihoods) == 1


def test_auxiliary_adjoint_and_batched_forward_gradients_agree():
    with jax.enable_x64():
        problem, _ = setup_problem()
        enriched = with_channel(problem)
        raw = np.asarray((0.2, -0.1, 1.0))
        forward_evaluation, forward = gravity_value_and_gradient(
            raw, problem=enriched, strategy=GravityGradientStrategy.BATCHED_FORWARD
        )
        adjoint_evaluation, adjoint = gravity_value_and_gradient(
            raw, problem=enriched, strategy=GravityGradientStrategy.ADJOINT
        )
        automatic = jax.grad(
            lambda value: evaluate_gravity_objective(value, problem=enriched).objective
        )(jnp.asarray(raw))
        np.testing.assert_allclose(forward, adjoint, rtol=2e-9, atol=2e-9)
        np.testing.assert_allclose(forward, automatic, rtol=2e-9, atol=2e-9)
        np.testing.assert_allclose(
            forward_evaluation.objective, adjoint_evaluation.objective
        )


def test_channel_reports_fingerprints_and_support_and_bundle_is_protocol_compatible():
    data = aggregate()
    channel = GravityAggregateObservationChannel(
        histogram=data.histograms[0],
        uncertainty=data.uncertainty,
        operator=attribute_operator(),
        aggregate_fingerprint=data.fingerprint,
    )
    assert isinstance(channel, GravityAggregateObservationChannel)
    assert channel.kind == "trip_attribute_distribution"
    report = channel.report()
    assert report["aggregate_fingerprint"] == data.fingerprint
    assert report["support_audit"]["status"] == "supported"
    bundle = GravityObservationBundle((channel,))
    bundle.validate(num_free_od=4, dtype=np.dtype("float64"))
    assert bundle.fingerprint


def test_support_failure_occurs_when_problem_is_built():
    base, _ = setup_problem()
    data = aggregate()
    unsupported_histogram = replace(
        data.histograms[0],
        strata=(GravityAggregateStratum("night", (2, 1), 3),),
    )
    unsupported = replace(data, histograms=(unsupported_histogram,))
    response = attribute_operator()
    bundle = GravityObservationBundle(
        (
            GravityAggregateObservationChannel(
                histogram=unsupported_histogram,
                uncertainty=unsupported.uncertainty,
                operator=response,
                aggregate_fingerprint=unsupported.fingerprint,
            ),
        )
    )
    with pytest.raises(ValueError, match="unsupported positive"):
        replace(base, auxiliary_observations=bundle)


def test_estimator_result_checkpoint_and_manifest_include_auxiliary_provenance(
    tmp_path,
):
    with jax.enable_x64():
        base, layout = setup_problem()
        enriched = with_channel(base)
        checkpoint = tmp_path / "auxiliary.json"
        result = estimate_gravity_model(
            problem=enriched,
            compact_layout=layout,
            initial_raw_parameters=np.zeros(3),
            config=GravityEstimatorConfig(maximum_iterations=1),
            execution=GravityExecutionPolicy(
                gradient_strategy="adjoint", checkpoint_path=checkpoint
            ),
        )
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert "auxiliary_observations" in payload
        assert result.auxiliary_observations is not None
        assert result.auxiliary_log_likelihood == pytest.approx(
            float(result.data_log_likelihood - result.count_log_likelihood)
        )
        manifest = build_gravity_run_manifest(
            problem=enriched,
            compact_layout=layout,
            estimator_config=GravityEstimatorConfig(maximum_iterations=1),
            execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
            repository_revision="fixture",
        )
        assert "auxiliary_observations" in manifest
        assert manifest["auxiliary_observations"]["channels"][0]["name"].startswith(
            "gps_trip_attributes:"
        )


def test_resume_rejects_changed_auxiliary_source(tmp_path):
    with jax.enable_x64():
        base, layout = setup_problem()
        enriched = with_channel(base)
        checkpoint = tmp_path / "auxiliary-resume.json"
        estimate_gravity_model(
            problem=enriched,
            compact_layout=layout,
            initial_raw_parameters=np.zeros(3),
            config=GravityEstimatorConfig(maximum_iterations=1),
            execution=GravityExecutionPolicy(
                gradient_strategy="adjoint", checkpoint_path=checkpoint
            ),
        )
        changed_data = replace(
            aggregate(),
            uncertainty=GravityAggregateUncertainty(
                "dirichlet_multinomial", concentration=4.0
            ),
        )
        changed_bundle = build_gravity_aggregate_observation_bundle(
            changed_data,
            {"travel_time": attribute_operator()},
            num_free_od=4,
        )
        changed = replace(base, auxiliary_observations=changed_bundle)
        assert gravity_model_fingerprint(changed, layout) != gravity_model_fingerprint(
            enriched, layout
        )
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            estimate_gravity_model(
                problem=changed,
                compact_layout=layout,
                initial_raw_parameters=np.zeros(3),
                execution=GravityExecutionPolicy(
                    gradient_strategy="adjoint", checkpoint_path=checkpoint
                ),
                resume=True,
            )
