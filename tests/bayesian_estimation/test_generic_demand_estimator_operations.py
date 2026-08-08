from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import public_transportation.inference.reduced_od.demand_estimator as demand_estimator_module

from public_transportation.inference.reduced_od import (
    DemandFitIdentity,
    DemandModelDimensions,
    DemandModelProblem,
    DemandModelSpecification,
    GaussianRawParameterPrior,
    ImpedanceSpecification,
    ProductionSpecification,
    ReducedODFitConfig,
    ReducedODNamedRawParameterBounds,
    build_demand_parameter_layout,
    estimate_demand_model,
    evaluate_demand_model,
)
from public_transportation.inference.reduced_od.features import ConditionalGravityFeatures
from public_transportation.inference.reduced_od.response_operator import ReducedResponseOperator
from public_transportation.preprocessing.reduced_od import ResponseCellKey


def _problem(observations: np.ndarray | None = None) -> tuple[DemandModelProblem, DemandFitIdentity]:
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
        2, 2, np.asarray([0, 1]), np.asarray([0, 1]), np.ones(2),
        np.arange(2), np.zeros(2), 2,
    )
    specification = DemandModelSpecification(
        production=ProductionSpecification(intercept=True),
        impedance=ImpedanceSpecification("none", "none"),
    )
    layout = build_demand_parameter_layout(specification, DemandModelDimensions(1, 1, 1))
    problem = DemandModelProblem(
        features, operator, np.asarray([10.0, 10.0]) if observations is None else observations,
        specification, layout, np.asarray([0]), np.asarray([0]), np.asarray([0, 0]),
    )
    identity = DemandFitIdentity(
        specification.fingerprint, layout.fingerprint, features.fingerprint,
        "groups", "operator", "data",
    )
    return problem, identity


def test_generic_fit_recovers_a_known_production_effect() -> None:
    seed, identity = _problem()
    truth = np.asarray([0.4])
    observations = np.asarray(evaluate_demand_model(truth, problem=seed).measurement_mean)
    problem, _ = _problem(observations)
    result = estimate_demand_model(
        problem=problem, initial_raw_parameters=np.zeros(1), identity=identity,
        config=ReducedODFitConfig(maximum_iterations=50),
    )
    assert result.status == "complete"
    assert result.raw_parameters == pytest.approx(truth, abs=2.0e-4)


def test_generic_map_and_named_bounds_are_applied() -> None:
    problem, identity = _problem(np.asarray([20.0, 20.0]))
    with pytest.raises(ValueError, match="requires a prior"):
        estimate_demand_model(
            problem=problem, initial_raw_parameters=np.zeros(1), identity=identity,
            config=ReducedODFitConfig(method="map", maximum_iterations=2),
        )
    result = estimate_demand_model(
        problem=problem, initial_raw_parameters=np.zeros(1), identity=identity,
        prior=GaussianRawParameterPrior(np.zeros(1), np.asarray([100.0])),
        config=ReducedODFitConfig(
            method="map", maximum_iterations=30,
            named_raw_parameter_bounds=ReducedODNamedRawParameterBounds(
                {"production_intercept": (-0.1, 0.2)}
            ),
        ),
    )
    assert result.raw_parameters[0] == pytest.approx(0.2, abs=1.0e-7)


def test_generic_deadline_checkpoint_can_resume(tmp_path: Path) -> None:
    problem, identity = _problem(np.asarray([20.0, 20.0]))
    checkpoint = tmp_path / "deadline.json"
    result = estimate_demand_model(
        problem=problem, initial_raw_parameters=np.zeros(1), identity=identity,
        config=ReducedODFitConfig(maximum_iterations=30, deadline_seconds=1.0e-12),
        checkpoint_path=checkpoint,
    )
    assert result.status == "deadline"
    assert checkpoint.is_file()
    resumed = estimate_demand_model(
        problem=problem, initial_raw_parameters=np.zeros(1), identity=identity,
        config=ReducedODFitConfig(maximum_iterations=30), checkpoint_path=checkpoint,
        resume=True,
    )
    assert resumed.resumed_from_iteration == result.iterations
    assert resumed.status == "complete"


def test_generic_progress_reports_iteration_timing_and_eta() -> None:
    problem, identity = _problem(np.asarray([20.0, 20.0]))
    events: list[dict[str, object]] = []
    result = estimate_demand_model(
        problem=problem,
        initial_raw_parameters=np.zeros(1),
        identity=identity,
        config=ReducedODFitConfig(maximum_iterations=3),
        progress=events.append,
    )
    updates = [event for event in events if event["status"] == "in_progress"]
    assert updates
    latest = updates[-1]
    assert float(latest["iteration_seconds"]) >= 0.0
    assert float(latest["rolling_average_iteration_seconds"]) >= 0.0
    assert float(latest["estimated_remaining_seconds"]) >= 0.0
    assert isinstance(latest["expected_completion_utc"], str)
    assert result.average_iteration_seconds is not None


def test_keyboard_interrupt_saves_latest_accepted_iterate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, identity = _problem(np.asarray([20.0, 20.0]))
    checkpoint = tmp_path / "interrupt.json"

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(demand_estimator_module, "minimize", interrupt)
    result = estimate_demand_model(
        problem=problem,
        initial_raw_parameters=np.zeros(1),
        identity=identity,
        config=ReducedODFitConfig(maximum_iterations=3),
        checkpoint_path=checkpoint,
    )
    assert result.status == "interrupted"
    assert checkpoint.is_file()
