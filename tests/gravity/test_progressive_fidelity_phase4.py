from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import pytest

from public_transportation.inference.gravity import (
    GravityFidelityEvaluationInterrupted,
    GravityFidelityExecution,
    GravityFidelityRequest,
    build_gravity_fidelity_anchor,
    gravity_value_and_gradient_progressive,
)
from tests.gravity.test_phase2_objective import problem
from tests.gravity.test_progressive_fidelity_phase2 import split_context


def test_anchor_at_requested_parameters_reproduces_stored_prediction():
    with jax.enable_x64():
        item = problem()
        context = split_context(item, count=16)
        raw = np.asarray((0.2, -0.1, 1.0))
        exact = gravity_value_and_gradient_progressive(
            raw,
            problem=item,
            fidelity=GravityFidelityRequest(effort_percent=100),
            context=context,
        )
        anchor = build_gravity_fidelity_anchor(raw, problem=item, result=exact)
        anchored = gravity_value_and_gradient_progressive(
            raw,
            problem=item,
            fidelity=GravityFidelityRequest(
                effort_percent=10, seed=7, anchor=anchor
            ),
            context=context,
        )
        np.testing.assert_array_equal(
            anchored.evaluation.measurement_mean, anchor.measurement_mean
        )
        assert float(anchored.evaluation.objective) == pytest.approx(
            float(exact.evaluation.objective)
        )
        assert anchored.quality.measurement_coverage_fraction == 1.0


def test_nearby_anchor_reduces_median_prediction_error():
    with jax.enable_x64():
        item = problem()
        context = split_context(item, count=24)
        anchor_raw = np.asarray((0.2, -0.1, 1.0))
        requested_raw = anchor_raw + np.asarray((0.02, -0.01, 0.01))
        anchor_result = gravity_value_and_gradient_progressive(
            anchor_raw,
            problem=item,
            fidelity=GravityFidelityRequest(effort_percent=100),
            context=context,
        )
        exact_requested = gravity_value_and_gradient_progressive(
            requested_raw,
            problem=item,
            fidelity=GravityFidelityRequest(effort_percent=100),
            context=context,
        )
        anchor = build_gravity_fidelity_anchor(
            anchor_raw, problem=item, result=anchor_result
        )
        unanchored_errors = []
        anchored_errors = []
        exact_mean = np.asarray(exact_requested.evaluation.measurement_mean)
        for seed in range(30):
            unanchored = gravity_value_and_gradient_progressive(
                requested_raw,
                problem=item,
                fidelity=GravityFidelityRequest(effort_percent=20, seed=seed),
                context=context,
            )
            anchored = gravity_value_and_gradient_progressive(
                requested_raw,
                problem=item,
                fidelity=GravityFidelityRequest(
                    effort_percent=20, seed=seed, anchor=anchor
                ),
                context=context,
            )
            unanchored_errors.append(
                np.linalg.norm(
                    np.asarray(unanchored.evaluation.measurement_mean) - exact_mean
                )
            )
            anchored_errors.append(
                np.linalg.norm(
                    np.asarray(anchored.evaluation.measurement_mean) - exact_mean
                )
            )
        assert np.median(anchored_errors) < np.median(unanchored_errors)


def test_incompatible_anchor_is_rejected():
    with jax.enable_x64():
        item = problem()
        context = split_context(item)
        raw = np.zeros(3)
        exact = gravity_value_and_gradient_progressive(
            raw,
            problem=item,
            fidelity=GravityFidelityRequest(),
            context=context,
        )
        anchor = build_gravity_fidelity_anchor(raw, problem=item, result=exact)
        changed = replace(item, rho=2.0)
        changed_context = split_context(changed)
        with pytest.raises(ValueError, match="identities differ"):
            gravity_value_and_gradient_progressive(
                raw,
                problem=changed,
                fidelity=GravityFidelityRequest(effort_percent=20, anchor=anchor),
                context=changed_context,
            )


def test_interruption_checkpoint_and_resume_match_uninterrupted(tmp_path):
    with jax.enable_x64():
        item = problem()
        context = split_context(item, count=20)
        raw = np.asarray((0.2, -0.1, 1.0))
        request = GravityFidelityRequest(effort_percent=70, seed=8)
        checkpoint = tmp_path / "progressive.npz"
        events = []
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls > 2

        with pytest.raises(GravityFidelityEvaluationInterrupted) as caught:
            gravity_value_and_gradient_progressive(
                raw,
                problem=item,
                fidelity=request,
                context=context,
                execution=GravityFidelityExecution(
                    checkpoint_path=checkpoint,
                    cancellation_requested=cancel,
                    progress=events.append,
                ),
            )
        assert caught.value.reason == "cancelled"
        assert caught.value.completed_shards == 2
        assert checkpoint.exists()
        assert events[-1].partial_result

        resumed_events = []
        resumed = gravity_value_and_gradient_progressive(
            raw,
            problem=item,
            fidelity=request,
            context=context,
            execution=GravityFidelityExecution(
                checkpoint_path=checkpoint,
                resume=True,
                progress=resumed_events.append,
            ),
        )
        uninterrupted = gravity_value_and_gradient_progressive(
            raw, problem=item, fidelity=request, context=context
        )
        np.testing.assert_array_equal(
            resumed.evaluation.measurement_mean,
            uninterrupted.evaluation.measurement_mean,
        )
        np.testing.assert_array_equal(resumed.gradient, uninterrupted.gradient)
        assert resumed.fidelity.selection_identity == (
            uninterrupted.fidelity.selection_identity
        )
        assert resumed_events[0].phase == "resumed"
        assert resumed_events[-1].phase == "completed"
        assert not resumed_events[-1].partial_result


def test_expired_deadline_checkpoints_without_publishing_result(tmp_path):
    with jax.enable_x64():
        item = problem()
        context = split_context(item)
        checkpoint = tmp_path / "deadline.npz"
        with pytest.raises(GravityFidelityEvaluationInterrupted) as caught:
            gravity_value_and_gradient_progressive(
                np.zeros(3),
                problem=item,
                fidelity=GravityFidelityRequest(effort_percent=30, seed=3),
                context=context,
                execution=GravityFidelityExecution(
                    checkpoint_path=checkpoint,
                    absolute_deadline=0.0,
                ),
            )
        assert caught.value.reason == "deadline_reached"
        assert caught.value.completed_shards == 0
        assert checkpoint.exists()


def test_resume_rejects_changed_subset(tmp_path):
    with jax.enable_x64():
        item = problem()
        context = split_context(item)
        checkpoint = tmp_path / "changed.npz"
        with pytest.raises(GravityFidelityEvaluationInterrupted):
            gravity_value_and_gradient_progressive(
                np.zeros(3),
                problem=item,
                fidelity=GravityFidelityRequest(effort_percent=30, seed=3),
                context=context,
                execution=GravityFidelityExecution(
                    checkpoint_path=checkpoint,
                    absolute_deadline=0.0,
                ),
            )
        with pytest.raises(ValueError, match="seed mismatch"):
            gravity_value_and_gradient_progressive(
                np.zeros(3),
                problem=item,
                fidelity=GravityFidelityRequest(effort_percent=30, seed=4),
                context=context,
                execution=GravityFidelityExecution(
                    checkpoint_path=checkpoint, resume=True
                ),
            )
