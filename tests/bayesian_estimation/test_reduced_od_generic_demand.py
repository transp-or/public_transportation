from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import jax
import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from public_transportation.inference.reduced_od import (
    AttractionSpecification,
    DemandModelDimensions,
    DemandFitIdentity,
    DemandModelProblem,
    DemandModelSpecification,
    ImpedanceSpecification,
    InteractionSpecification,
    ObservationSpecification,
    ProductionSpecification,
    build_demand_parameter_layout,
    benchmark_demand_model,
    build_evidence_assumption_report,
    build_grouping_hierarchy,
    demand_model_value_and_gradient,
    estimate_demand_model,
    evaluate_demand_model,
    progressive_model_ladder,
    resolve_observation_model,
    recommend_demand_relaxations,
    warm_start_demand_parameters,
    ReducedODFitConfig,
)
from public_transportation.inference.reduced_od.features import (
    ConditionalGravityFeatures,
)
from public_transportation.inference.reduced_od.response_operator import (
    ReducedResponseOperator,
)
from public_transportation.preprocessing.reduced_od import ResponseCellKey


def test_specification_is_stable_composable_and_counts_only_active_blocks() -> None:
    dimensions = DemandModelDimensions(3, 4, 5, 2)
    minimal = DemandModelSpecification()
    assert minimal.parameter_counts(dimensions) == {
        "production": 0,
        "attraction": 0,
        "impedance": 2,
        "interaction": 0,
        "observation": 0,
    }
    rich = DemandModelSpecification(
        production=ProductionSpecification(True, True, True, True),
        attraction=AttractionSpecification(True, True, True, True),
        impedance=ImpedanceSpecification("period", "period"),
        interaction=InteractionSpecification(2),
        observation=ObservationSpecification("zinb", "design"),
    )
    counts = rich.parameter_counts(dimensions)
    assert counts == {
        "production": 12,
        "attraction": 15,
        "impedance": 6,
        "interaction": 18,
        "observation": 3,
    }
    assert rich.fingerprint == rich.fingerprint
    json.dumps(rich.to_dict())
    assert "total=54" in rich.summary(dimensions)
    assert set(progressive_model_ladder()) >= {"M0", "M1", "M2", "M3", "M4", "M5"}


def test_layout_slices_warm_start_and_disabled_blocks() -> None:
    dimensions = DemandModelDimensions(2, 3, 4)
    parent_spec = DemandModelSpecification(
        production=ProductionSpecification(intercept=True)
    )
    child_spec = DemandModelSpecification(
        production=ProductionSpecification(True, True, True),
        interaction=InteractionSpecification(1),
        observation=ObservationSpecification("negative_binomial", "none"),
    )
    parent = build_demand_parameter_layout(parent_spec, dimensions)
    child = build_demand_parameter_layout(child_spec, dimensions)
    assert "zero_inflation" not in child.slices
    assert parent.slices["production_intercept"] == slice(0, 1)
    raw, report = warm_start_demand_parameters(parent, child, np.arange(parent.size))
    assert raw[child.slices["production_intercept"]] == pytest.approx([0.0])
    assert "production_intercept" in report.copied
    assert "observation_dispersion" in report.initialized
    assert np.any(raw[child.slices["interaction_origin"]] != 0.0)


@pytest.mark.parametrize("family", ["poisson", "negative_binomial", "zip", "zinb"])
def test_observation_models_match_references_and_have_finite_gradients(
    family: str,
) -> None:
    y = np.asarray([0.0, 1.0, 8.0])
    mu = np.asarray([0.2, 2.0, 7.0])
    dispersion = 3.0 if family in {"negative_binomial", "zinb"} else None
    logits = np.asarray([-1.2, -1.2, -1.2]) if family in {"zip", "zinb"} else None
    model = resolve_observation_model(family)
    actual = np.asarray(
        model.log_likelihood(y, mu, dispersion=dispersion, inflation_logits=logits)
    )
    if family in {"poisson", "zip"}:
        count = poisson.logpmf(y, mu)
    else:
        assert dispersion is not None
        probability = dispersion / (dispersion + mu)
        count = nbinom.logpmf(y, dispersion, probability)
    if logits is not None:
        pi = 1.0 / (1.0 + np.exp(-logits))
        expected = np.where(
            y == 0,
            np.log(pi + (1.0 - pi) * np.exp(count)),
            np.log1p(-pi) + count,
        )
    else:
        expected = count
    assert actual == pytest.approx(expected, abs=3.0e-6)
    gradient = jax.grad(
        lambda value: (
            -np.float32(1.0)
            * resolve_observation_model(family)
            .log_likelihood(
                y,
                value,
                dispersion=dispersion,
                inflation_logits=logits,
            )
            .sum()
        )
    )(mu)
    assert np.all(np.isfinite(gradient))
    assert np.all(
        np.isfinite(
            model.zero_probability(mu, dispersion=dispersion, inflation_logits=logits)
        )
    )


def test_grouping_hierarchy_is_deterministic_nested_and_reports_orphans() -> None:
    kwargs = dict(
        node_ids=("D", "B", "A", "C"),
        signatures=np.asarray([[3.0, 0.0], [1.0, 0.0], [0.0, 0.0], [2.0, 0.0]]),
        adjacency=(("A", "B"), ("B", "C")),
    )
    first = build_grouping_hierarchy(**kwargs)
    second = build_grouping_hierarchy(**kwargs)
    assert first.fingerprint == second.fingerprint
    assert [len(set(level)) for level in first.levels] == [4, 3, 2, 1]
    assert first.disconnected_nodes == ("D",)
    coarse = first.membership(2)
    fine = first.membership(3)
    for left in range(4):
        for right in range(4):
            if fine[left] == fine[right]:
                assert coarse[left] == coarse[right]


def test_evidence_report_avoids_spurious_percentages_and_recommends_small_steps() -> (
    None
):
    report = build_evidence_assumption_report(
        block_jacobians={"identified": np.eye(2), "silent": np.zeros((2, 1))},
        block_gradients={"identified": np.ones(2), "silent": np.zeros(1)},
        block_penalties={"identified": 0.0, "silent": 1.0},
        observations=np.asarray([0.0, 3.0]),
        expected_zero_probabilities=np.asarray([0.4, 0.1]),
        assumptions=("fixed_routing", "gravity_functional_form"),
    )
    support = {item.block: item.support for item in report.blocks}
    assert support == {"identified": "data_supported", "silent": "unidentified"}
    assert report.excess_zero_count == pytest.approx(0.5)
    recommendations = recommend_demand_relaxations(
        period_residual_ratio=0.2,
        variance_to_mean_ratio=2.0,
        excess_zero_fraction_after_nb=0.0,
    )
    assert [item.code for item in recommendations] == [
        "period_effects",
        "negative_binomial",
    ]


@pytest.mark.parametrize(
    ("family", "zero_mode", "zero_columns"),
    [
        ("poisson", "none", 0),
        ("negative_binomial", "none", 0),
        ("zip", "intercept", 0),
        ("zinb", "design", 2),
    ],
)
def test_unified_demand_objective_has_finite_value_and_gradient(
    family: str, zero_mode: str, zero_columns: int
) -> None:
    features = ConditionalGravityFeatures(
        cell_keys=(
            ResponseCellKey("A", "C", "P0"),
            ResponseCellKey("A", "D", "P0"),
            ResponseCellKey("B", "C", "P1"),
            ResponseCellKey("B", "D", "P1"),
        ),
        origin_time_group_index=np.asarray([0, 0, 1, 1]),
        destination_index=np.asarray([0, 1, 0, 1]),
        journey_time_seconds=np.asarray([600.0, 900.0, 700.0, 1000.0]),
        transfer_count=np.asarray([0.0, 1.0, 0.0, 1.0]),
        destination_attractiveness=np.ones(4),
        baseline_productions=np.asarray([20.0, 30.0]),
        origin_time_group_keys=(("A", "P0"), ("B", "P1")),
        destination_ids=("C", "D"),
    )
    operator = ReducedResponseOperator(
        number_of_measurements=3,
        number_of_free_cells=4,
        measurement_index=np.asarray([0, 0, 1, 1, 2, 2]),
        response_class_index=np.asarray([0, 1, 1, 2, 2, 3]),
        response_values=np.ones(6),
        class_by_free_cell=np.arange(4),
        fixed_offset=np.zeros(3),
        original_nnz=6,
    )
    specification = DemandModelSpecification(
        production=ProductionSpecification(True, True, True),
        attraction=AttractionSpecification(destination_group_effects=True),
        impedance=ImpedanceSpecification("period", "global"),
        interaction=InteractionSpecification(1),
        observation=ObservationSpecification(family, zero_mode),  # type: ignore[arg-type]
    )
    dimensions = DemandModelDimensions(2, 2, 2, zero_columns)
    layout = build_demand_parameter_layout(specification, dimensions)
    zero_design = (
        None
        if zero_mode in {"none", "intercept"}
        else np.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    )
    problem = DemandModelProblem(
        features,
        operator,
        np.asarray([10.0, 15.0, 8.0]),
        specification,
        layout,
        group_period_index=np.asarray([0, 1]),
        origin_group_index=np.asarray([0, 1]),
        cell_destination_group_index=np.asarray([0, 1, 0, 1]),
        zero_inflation_design=zero_design,
    )
    raw = np.zeros(layout.size)
    evaluation, gradient = demand_model_value_and_gradient(raw, problem=problem)
    assert np.isfinite(evaluation.objective)
    assert np.all(np.isfinite(gradient))
    assert np.asarray(evaluation.productions) == pytest.approx([20.0, 30.0])
    step = 2.0e-3
    finite = np.empty(layout.size)
    for index in range(layout.size):
        direction = np.zeros(layout.size)
        direction[index] = step
        finite[index] = (
            float(evaluate_demand_model(raw + direction, problem=problem).objective)
            - float(evaluate_demand_model(raw - direction, problem=problem).objective)
        ) / (2.0 * step)
    assert np.asarray(gradient) == pytest.approx(finite, abs=2.0e-2, rel=2.0e-2)


def test_generic_estimator_progress_checkpoint_and_identity_rejection(
    tmp_path: Path,
) -> None:
    # Reuse the smallest synthetic problem built by the parametrized objective test.
    features = ConditionalGravityFeatures(
        cell_keys=(ResponseCellKey("A", "C", "P"), ResponseCellKey("A", "D", "P")),
        origin_time_group_index=np.asarray([0, 0]),
        destination_index=np.asarray([0, 1]),
        journey_time_seconds=np.asarray([600.0, 900.0]),
        transfer_count=np.asarray([0.0, 1.0]),
        destination_attractiveness=np.ones(2),
        baseline_productions=np.asarray([20.0]),
        origin_time_group_keys=(("A", "P"),),
        destination_ids=("C", "D"),
    )
    operator = ReducedResponseOperator(
        2,
        2,
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        np.ones(2),
        np.arange(2),
        np.zeros(2),
        2,
    )
    specification = DemandModelSpecification(
        production=ProductionSpecification(intercept=True)
    )
    layout = build_demand_parameter_layout(
        specification, DemandModelDimensions(1, 1, 1)
    )
    problem = DemandModelProblem(
        features,
        operator,
        np.asarray([8.0, 12.0]),
        specification,
        layout,
        np.asarray([0]),
        np.asarray([0]),
        np.asarray([0, 0]),
    )
    identity = DemandFitIdentity(
        specification.fingerprint,
        layout.fingerprint,
        features.fingerprint,
        "groups",
        "operator",
        "data",
    )
    events: list[Mapping[str, object]] = []
    checkpoint = tmp_path / "fit.json"
    result = estimate_demand_model(
        problem=problem,
        initial_raw_parameters=np.zeros(layout.size),
        identity=identity,
        config=ReducedODFitConfig(maximum_iterations=2, checkpoint_every_iterations=1),
        checkpoint_path=checkpoint,
        progress=events.append,
    )
    assert np.isfinite(result.objective)
    assert checkpoint.is_file()
    assert events[0]["status"] == "started"
    benchmark = benchmark_demand_model(
        problem=problem, raw_parameters=result.raw_parameters, warm_evaluations=2
    )
    assert benchmark.parameter_count == layout.size
    assert benchmark.warm_value_gradient_seconds > 0.0
    incompatible = DemandFitIdentity(
        specification.fingerprint,
        layout.fingerprint,
        features.fingerprint,
        "changed",
        "operator",
        "data",
    )
    with pytest.raises(ValueError, match="incompatible"):
        estimate_demand_model(
            problem=problem,
            initial_raw_parameters=np.zeros(layout.size),
            identity=incompatible,
            config=ReducedODFitConfig(maximum_iterations=1),
            checkpoint_path=checkpoint,
            resume=True,
        )
