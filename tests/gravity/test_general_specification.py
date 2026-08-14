from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
    GravityComponentSpecification,
    GravityConstraint,
    GravityEffectScope,
    GravityEstimatorConfig,
    GravityExecutionPolicy,
    GravityFeatures,
    GravityGradientStrategy,
    GravityLikelihood,
    GravityLikelihoodSpecification,
    GravityModelSpecification,
    GravityModelFitSummary,
    GravityObjectiveProblem,
    GravityParameterLayout,
    GravityParameterization,
    GravityRegularization,
    GravityRegularizationType,
    GravityTimeSpecification,
    build_gravity_run_manifest,
    estimate_gravity_model,
    generate_gravity_demand,
    gravity_demand_numpy_reference,
    gravity_model_specification_from_mapping,
    gravity_value_and_gradient,
    evaluate_gravity_objective,
    load_gravity_model_specification,
    rank_gravity_model_summaries,
    validate_gravity_model_specification,
    warm_start_gravity_parameters,
)


def features() -> GravityFeatures:
    origin = np.repeat(np.arange(2), 6)
    departure = np.tile(np.repeat(np.arange(2), 3), 2)
    destination = np.tile(np.arange(3), 4)
    return GravityFeatures(
        canonical_od_index=np.arange(12),
        origin_index=origin,
        destination_index=destination,
        departure_time_index=departure,
        origin_time_group_index=np.repeat(np.arange(4), 3),
        journey_time=np.asarray(
            (4, 9, 15, 8, 5, 11, 20, 10, 7, 12, 18, 6), dtype=np.float64
        ),
        transfer_count=np.asarray((0, 1, 2, 1, 0, 2, 2, 1, 0, 1, 2, 0)),
        structural_feasible=np.asarray(
            (True, True, False, True, True, True, True, True, True, True, True, True)
        ),
        origin_time_totals=np.asarray((20, 30, 40, 50), dtype=np.float64),
        destination_attractiveness=np.asarray(
            (1, 2, 3, 1, 2, 3, 2, 1, 4, 2, 1, 4), dtype=np.float64
        ),
        num_origins=2,
        num_destinations=3,
        num_departure_times=2,
        od_layout_fingerprint="general-layout",
        journey_time_scale=10,
        initial_waiting_time=np.asarray((2, 2, 2, 4, 4, 4) * 2, dtype=np.float64),
        origin_zone_index=origin,
        destination_zone_index=destination,
        time_period_index=departure,
        destination_time_group_index=2 * destination + departure,
        zone_pair_index=3 * origin + destination,
        custom_group_indices={
            "production_group": origin,
            "utility_group": destination,
        },
        smooth_time_basis=np.column_stack(
            (departure.astype(float), departure.astype(float) ** 2)
        ),
    )


def component(
    name: str,
    scope: GravityEffectScope,
    parameterization: GravityParameterization,
    *,
    grouping: str | None = None,
    count: int = 0,
    constraint: GravityConstraint = GravityConstraint.NONE,
    reference: int | None = None,
    ridge: float = 0,
    fixed_value: float | None = None,
    source: str | None = None,
) -> GravityComponentSpecification:
    return GravityComponentSpecification(
        name=name,
        scope=scope,
        parameterization=parameterization,
        grouping=grouping,
        group_count=count,
        constraint=constraint,
        reference_category=reference,
        regularization=(
            GravityRegularization(GravityRegularizationType.RIDGE, ridge)
            if ridge
            else GravityRegularization()
        ),
        fixed_value=fixed_value,
        source=(
            "origin_time_totals"
            if name == "production"
            else "feature_cache"
            if name == "destination_attractiveness"
            else source
        ),
    )


@pytest.mark.parametrize(
    ("scope", "mapping", "count"),
    (
        (GravityEffectScope.ORIGIN, "origin_index", 2),
        (GravityEffectScope.DESTINATION, "destination_index", 3),
        (GravityEffectScope.TIME_PERIOD, "time_period_index", 2),
        (GravityEffectScope.ORIGIN_TIME, "origin_time_group_index", 4),
        (GravityEffectScope.DESTINATION_TIME, "destination_time_group_index", 6),
        (GravityEffectScope.ORIGIN_ZONE, "origin_zone_index", 2),
        (GravityEffectScope.DESTINATION_ZONE, "destination_zone_index", 3),
        (GravityEffectScope.ZONE_PAIR, "zone_pair_index", 6),
        (GravityEffectScope.CUSTOM_GROUP, "utility_group", 3),
    ),
)
def test_all_grouped_scopes_have_explicit_deterministic_blocks(scope, mapping, count):
    specification = GravityModelSpecification(
        components=(
            component(
                "journey_time",
                scope,
                GravityParameterization.POSITIVE,
                grouping=mapping,
                count=count,
                constraint=GravityConstraint.SUM_ZERO,
            ),
        )
    )
    validation = validate_gravity_model_specification(
        specification, features=features()
    )
    block = validation.parameter_layout.block("journey_time")
    assert block is not None
    assert block.size == count
    assert block.mapping == mapping
    assert validation.parameter_layout.names[0] == "beta_time"


def test_none_fixed_global_and_smooth_basis_scopes_are_explicit():
    specification = GravityModelSpecification(
        time=GravityTimeSpecification(smooth_basis_name="smooth_time_basis"),
        components=(
            component(
                "journey_time",
                GravityEffectScope.FIXED,
                GravityParameterization.FIXED,
                fixed_value=0.4,
            ),
            component(
                "waiting_time",
                GravityEffectScope.GLOBAL,
                GravityParameterization.POSITIVE,
            ),
            component(
                "temporal",
                GravityEffectScope.SMOOTH_BASIS,
                GravityParameterization.ADDITIVE,
                grouping="smooth_time_basis",
                count=2,
            ),
        )
    )
    layout = validate_gravity_model_specification(
        specification, features=features()
    ).parameter_layout
    assert layout.block("journey_time") is None
    assert layout.block("waiting_time") is not None
    assert layout.block("temporal").size == 2  # type: ignore[union-attr]


def combined_specification() -> GravityModelSpecification:
    return GravityModelSpecification(
        model_name="combined_test",
        time=GravityTimeSpecification(smooth_basis_name="smooth_time_basis"),
        components=(
            component(
                "journey_time",
                GravityEffectScope.TIME_PERIOD,
                GravityParameterization.POSITIVE,
                grouping="time_period_index",
                count=2,
                constraint=GravityConstraint.SUM_ZERO,
            ),
            component(
                "waiting_time",
                GravityEffectScope.GLOBAL,
                GravityParameterization.POSITIVE,
            ),
            component(
                "production",
                GravityEffectScope.ORIGIN,
                GravityParameterization.LOG_MULTIPLIER,
                grouping="origin_index",
                count=2,
                constraint=GravityConstraint.REFERENCE,
                reference=0,
                ridge=0.5,
            ),
            component(
                "destination_attractiveness",
                GravityEffectScope.DESTINATION,
                GravityParameterization.ADDITIVE,
                grouping="destination_index",
                count=3,
                constraint=GravityConstraint.SUM_ZERO,
                ridge=0.25,
            ),
            component(
                "temporal",
                GravityEffectScope.SMOOTH_BASIS,
                GravityParameterization.ADDITIVE,
                grouping="smooth_time_basis",
                count=2,
                ridge=0.1,
            ),
        ),
    )


def test_general_demand_matches_independent_numpy_and_conserves_corrected_totals():
    item = features()
    layout = GravityParameterLayout(combined_specification())
    raw = np.linspace(-0.3, 0.4, layout.size)
    actual = generate_gravity_demand(raw, features=item, parameter_layout=layout)
    expected = gravity_demand_numpy_reference(
        raw, features=item, parameter_layout=layout
    )
    np.testing.assert_allclose(actual.demand, expected, rtol=2e-6, atol=2e-6)
    production = np.asarray(layout.production_group_log_multiplier(raw, item))
    np.testing.assert_allclose(
        actual.origin_time_sums,
        item.origin_time_totals * np.exp(production),
        rtol=2e-6,
        atol=2e-6,
    )
    assert float(actual.demand[2]) == 0


def test_parameter_round_trip_centering_reference_regularization_and_fingerprint():
    layout = GravityParameterLayout(combined_specification())
    physical = np.linspace(0.2, 1.2, layout.size)
    raw = layout.raw_from_physical(physical)
    np.testing.assert_allclose(layout.physical_vector(raw), physical, rtol=2e-7)
    np.testing.assert_allclose(
        np.sum(layout.constrained_deviations(raw, "destination_attractiveness")),
        0,
        atol=1e-15,
    )
    assert layout.constrained_deviations(raw, "production")[0] == 0
    assert float(layout.regularization(raw)) > 0
    restored = GravityParameterLayout.from_dict(layout.to_dict())
    assert restored.names == layout.names
    assert restored.fingerprint == layout.fingerprint
    corrupted = layout.to_dict()
    corrupted["blocks"][0]["stop"] = 99
    with pytest.raises(ValueError, match="blocks do not match"):
        GravityParameterLayout.from_dict(corrupted)


def test_production_modes_and_destination_modes_preserve_expected_mass():
    item = features()
    cases = (
        (GravityEffectScope.GLOBAL, None, 0, GravityConstraint.NONE, 0),
        (GravityEffectScope.ORIGIN, "origin_index", 2, GravityConstraint.SUM_ZERO, 0),
        (GravityEffectScope.TIME_PERIOD, "time_period_index", 2, GravityConstraint.SUM_ZERO, 0),
        (GravityEffectScope.ORIGIN_TIME, "origin_time_group_index", 4, GravityConstraint.SUM_ZERO, 1),
        (GravityEffectScope.ORIGIN_ZONE, "origin_zone_index", 2, GravityConstraint.SUM_ZERO, 0),
        (GravityEffectScope.CUSTOM_GROUP, "production_group", 2, GravityConstraint.SUM_ZERO, 1),
    )
    for scope, mapping, count, constraint, ridge in cases:
        layout = GravityParameterLayout(
            GravityModelSpecification(
                components=(
                    component(
                        "production",
                        scope,
                        GravityParameterization.LOG_MULTIPLIER,
                        grouping=mapping,
                        count=count,
                        constraint=constraint,
                        ridge=ridge,
                    ),
                )
            )
        )
        raw = np.full(layout.size, 0.1)
        result = generate_gravity_demand(raw, features=item, parameter_layout=layout)
        expected = item.origin_time_totals * np.exp(
            np.asarray(layout.production_group_log_multiplier(raw, item))
        )
        np.testing.assert_allclose(result.origin_time_sums, expected)

    for scope, mapping, count in (
        (GravityEffectScope.GLOBAL, None, 0),
        (GravityEffectScope.DESTINATION, "destination_index", 3),
        (GravityEffectScope.TIME_PERIOD, "time_period_index", 2),
        (GravityEffectScope.DESTINATION_TIME, "destination_time_group_index", 6),
        (GravityEffectScope.DESTINATION_ZONE, "destination_zone_index", 3),
    ):
        constraint = (
            GravityConstraint.NONE
            if scope is GravityEffectScope.GLOBAL
            else GravityConstraint.SUM_ZERO
        )
        layout = GravityParameterLayout(
            GravityModelSpecification(
                components=(
                    component(
                        "destination_attractiveness",
                        scope,
                        GravityParameterization.ADDITIVE,
                        grouping=mapping,
                        count=count,
                        constraint=constraint,
                    ),
                )
            )
        )
        result = generate_gravity_demand(
            np.full(layout.size, 0.1), features=item, parameter_layout=layout
        )
        np.testing.assert_allclose(result.origin_time_sums, item.origin_time_totals)


def test_yaml_template_loading_serialization_and_validation_summary(tmp_path):
    template = (
        Path(__file__).parents[2] / "configs/gravity/gravity_model_spec.yaml"
    )
    loaded = load_gravity_model_specification(
        template,
        features=features(),
        calibration_mask=np.asarray((True,) * 10 + (False, False)),
        unsupported_measurement_mask=np.asarray((False,) * 10 + (True, True)),
        structural_zero_fingerprint="structural-zero-test",
    )
    assert loaded.specification.model_name == "global_production"
    assert loaded.parameter_layout.names[-1] == "production_scale"
    assert "Calibration rows: 10" in loaded.summary
    assert "structural-zero-test" in loaded.summary
    assert (
        GravityModelSpecification.from_dict(loaded.specification.to_dict())
        == loaded.specification
    )

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: 1\nunknown_option: true\n")
    with pytest.raises(ValueError, match="unsupported gravity specification"):
        load_gravity_model_specification(invalid)


def test_missing_noncontiguous_mappings_and_identifiability_failures():
    item = features()
    missing = combined_specification()
    with pytest.raises(ValueError, match="smooth_time_basis"):
        validate_gravity_model_specification(
            missing, features=replace(item, smooth_time_basis=None)
        )
    broken = GravityFeatures.from_dict(item.to_dict())
    object.__setattr__(broken, "destination_zone_index", np.asarray((0, 2, 3) * 4))
    specification = GravityModelSpecification(
        components=(
            component(
                "destination_attractiveness",
                GravityEffectScope.DESTINATION_ZONE,
                GravityParameterization.ADDITIVE,
                grouping="destination_zone_index",
                count=3,
                constraint=GravityConstraint.SUM_ZERO,
            ),
        )
    )
    with pytest.raises(ValueError, match="contiguous"):
        validate_gravity_model_specification(specification, features=broken)
    with pytest.raises(ValueError, match="detection-rate"):
        gravity_model_specification_from_mapping(
            {
                "schema_version": 1,
                "likelihood": {"detection_rate_estimated": True},
                "production": {"correction": {"scope": "global"}},
            }
        )
    with pytest.raises(ValueError, match="requires ridge"):
        GravityModelSpecification(
            components=(
                component(
                    "production",
                    GravityEffectScope.ORIGIN_TIME,
                    GravityParameterization.LOG_MULTIPLIER,
                    grouping="origin_time_group_index",
                    count=4,
                    constraint=GravityConstraint.SUM_ZERO,
                ),
            )
        )


def objective_problem() -> tuple[GravityObjectiveProblem, CompactODAssignmentLayout]:
    item = features()
    compact = CompactODAssignmentLayout(
        num_od_total=item.num_cells,
        active_full_indices=tuple(range(item.num_cells)),
        removed_zero_full_indices=(),
        full_to_compact=tuple(range(item.num_cells)),
        free_full_indices=tuple(range(item.num_cells)),
        free_compact_indices=tuple(range(item.num_cells)),
        free_baseline_values=tuple(1.0 for _ in range(item.num_cells)),
        fixed_compact_indices=(),
        fixed_compact_values=(),
    )
    item = replace(item, od_layout_fingerprint=compact.fingerprint)
    matrix = jnp.eye(item.num_cells, dtype=jnp.float64)
    operator = FixedRoutingMeasurementOperator(
        matrix=matrix,
        fixed_measurement_offset=jnp.arange(item.num_cells, dtype=jnp.float64) / 10,
        representation="dense",
        num_active_od=item.num_cells,
        num_free_od=item.num_cells,
        num_measurements=item.num_cells,
        od_layout_fingerprint=compact.fingerprint,
        compact_layout_fingerprint=compact.fingerprint,
        assignment_fingerprint="assignment-general",
        graph_fingerprint="graph-general",
        mapping_fingerprint="mapping-general",
        theta=1.0,
        dtype="float64",
        metrics=MeasurementOperatorMetrics(
            0,
            matrix.nbytes,
            matrix.nbytes,
            0,
            item.num_cells,
            item.num_cells**2,
            1 / item.num_cells,
            item.num_cells,
        ),
    )
    layout = GravityParameterLayout(combined_specification())
    seed = generate_gravity_demand(
        np.zeros(layout.size), features=item, parameter_layout=layout
    ).demand
    observations = np.asarray(seed + operator.fixed_measurement_offset)
    calibration = np.ones(item.num_cells, dtype=bool)
    calibration[-1] = False
    return (
        GravityObjectiveProblem(
            features=item,
            parameter_layout=layout,
            operator=operator,
            observations=observations,
            calibration_mask=calibration,
        ),
        compact,
    )


def test_combined_objective_adjoint_gradient_manifest_and_nested_warm_start():
    with jax.enable_x64():
        problem, compact = objective_problem()
        raw = np.linspace(-0.2, 0.2, problem.parameter_layout.size)
        evaluation, adjoint = gravity_value_and_gradient(
            raw, problem=problem, strategy=GravityGradientStrategy.ADJOINT
        )
        automatic = jax.grad(
            lambda value: evaluate_gravity_objective(
                value, problem=problem
            ).objective
        )(jnp.asarray(raw))
        np.testing.assert_allclose(adjoint, automatic, rtol=1e-9, atol=1e-9)
        assert float(evaluation.regularization) > 0

        parent = GravityParameterLayout(GravityModelSpecification())
        child = problem.parameter_layout
        warm = warm_start_gravity_parameters(parent, child, np.zeros(parent.size))
        assert warm.shape == (child.size,)
        assert np.count_nonzero(warm) == 0

        unsupported = np.zeros(problem.observations.size, dtype=bool)
        unsupported[-1] = True
        manifest = build_gravity_run_manifest(
            problem=problem,
            compact_layout=compact,
            estimator_config=GravityEstimatorConfig(maximum_iterations=1),
            execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
            repository_revision="general-test",
            unsupported_measurement_mask=unsupported,
            holdout_mask=unsupported,
            structural_zero_fingerprint="zeros-general",
        )
        assert manifest["model_specification"] == combined_specification().to_dict()
        assert manifest["specification_fingerprint"] == combined_specification().fingerprint
        assert len(manifest["parameter_names"]) == child.size
        assert manifest["fingerprints"]["features"] == problem.features.fingerprint
        assert manifest["observation_masks"]["unsupported"]["excluded"] == 1


def test_general_specification_supports_poisson_and_explicit_masks():
    with jax.enable_x64():
        original, _ = objective_problem()
        specification = replace(
            combined_specification(),
            likelihood=GravityLikelihoodSpecification(family="poisson"),
            components=(
                *combined_specification().components,
                component(
                    "dispersion",
                    GravityEffectScope.NONE,
                    GravityParameterization.FIXED,
                ),
            ),
        )
        problem = replace(
            original,
            parameter_layout=GravityParameterLayout(specification),
            likelihood=GravityLikelihood.POISSON,
        )
        evaluation, gradient = gravity_value_and_gradient(
            np.zeros(problem.parameter_layout.size),
            problem=problem,
            strategy=GravityGradientStrategy.ADJOINT,
        )
        assert np.isfinite(float(evaluation.objective))
        assert np.all(np.isfinite(np.asarray(gradient)))
        assert int(evaluation.calibration_measurements) == int(
            np.count_nonzero(problem.calibration_mask)
        )


def test_model_ranking_requires_and_prioritizes_held_out_diagnostics():
    def summary(name: str, heldout: float | None, rmse: float | None, size: int):
        return GravityModelFitSummary(
            model_name=name,
            specification_fingerprint=name,
            parameter_count=size,
            in_sample_objective=1.0,
            held_out_data_log_likelihood=heldout,
            held_out_rmse=rmse,
            convergence_status="converged",
            success=True,
            gradient_inf_norm=1e-7,
            regularization_contribution=0.0,
            calibration_measurements=10,
            excluded_measurements=2,
            observed_count_mass=100.0,
            predicted_count_mass=99.0,
            unsupported_rows=0,
        )

    ranked = rank_gravity_model_summaries(
        (summary("larger", -9.0, 2.0, 8), summary("smaller", -10.0, 1.0, 3))
    )
    assert ranked[0].model_name == "larger"
    with pytest.raises(ValueError, match="held-out"):
        rank_gravity_model_summaries((summary("missing", None, None, 3),))


def test_combined_checkpointed_estimation_records_complete_specification(tmp_path):
    with jax.enable_x64():
        problem, compact = objective_problem()
        result = estimate_gravity_model(
            problem=problem,
            compact_layout=compact,
            initial_raw_parameters=np.zeros(problem.parameter_layout.size),
            config=GravityEstimatorConfig(maximum_iterations=1),
            execution=GravityExecutionPolicy(
                gradient_strategy="adjoint",
                checkpoint_path=tmp_path / "combined-checkpoint.json",
            ),
        )
        assert result.model_specification == combined_specification().to_dict()
        assert result.specification_fingerprint == combined_specification().fingerprint
        assert result.parameter_names == problem.parameter_layout.names
        assert len(result.parameter_blocks) == len(problem.parameter_layout.blocks)
        assert result.feature_cache_fingerprint == problem.features.fingerprint
        assert result.destination_attractiveness_provenance == "feature_cache"
        assert result.time_discretization["smooth_basis_name"] == "smooth_time_basis"
        assert result.calibration_measurements == problem.calibration_measurements
        assert result.excluded_measurements == problem.excluded_measurements
        resumed = estimate_gravity_model(
            problem=problem,
            compact_layout=compact,
            initial_raw_parameters=np.zeros(problem.parameter_layout.size),
            config=GravityEstimatorConfig(maximum_iterations=2),
            execution=GravityExecutionPolicy(
                gradient_strategy="adjoint",
                checkpoint_path=tmp_path / "combined-checkpoint.json",
            ),
            resume=True,
        )
        assert resumed.resumed
        assert resumed.specification_fingerprint == result.specification_fingerprint
