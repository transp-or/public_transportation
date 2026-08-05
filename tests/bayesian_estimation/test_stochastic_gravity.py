from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import gc
import weakref

import numpy as np
import pytest

from benchmarks.benchmark_sharded_gravity_operator import _problem
from public_transportation.inference.gravity import gravity_value_and_gradient_adjoint
from public_transportation.inference.parallel_gravity_anchor import (
    create_parallel_gravity_anchor,
)
from public_transportation.inference.stochastic_gravity import (
    StochasticGravityConfig,
    select_stochastic_routing_shards,
    stochastic_gravity_value_and_gradient,
)
from public_transportation.inference.sharded_matrix_free_operator import (
    ShardedMatrixFreeFixedRoutingMeasurementOperator,
)
from tests.bayesian_estimation.test_parallel_routing_executor import _operator


def test_seeded_persisted_shard_selection_is_deterministic_nested_and_weighted():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary), groups=16)
        descriptors = tuple(operator.routing.shard_partition)
        low = select_stochastic_routing_shards(
            descriptors, effort_percent=25, seed=19
        )
        repeated = select_stochastic_routing_shards(
            descriptors, effort_percent=25, seed=19
        )
        high = select_stochastic_routing_shards(
            descriptors, effort_percent=50, seed=19
        )
        changed = select_stochastic_routing_shards(
            descriptors, effort_percent=25, seed=20
        )
        assert low == repeated
        assert set(low.selected_shard_ids) < set(high.selected_shard_ids)
        assert low.selected_shard_ids != changed.selected_shard_ids
        assert len(low.selected_shard_ids) == 2
        assert low.sampling_weights == (4.0, 4.0)
        assert low.realized_effort_percent == 25.0


def test_full_effort_delegates_to_exact_numerical_path():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary))
        problem, raw = _problem(operator)
        expected, expected_gradient = gravity_value_and_gradient_adjoint(
            raw, problem=problem
        )
        result = stochastic_gravity_value_and_gradient(
            raw,
            problem=problem,
            config=StochasticGravityConfig(effort_percent=100),
        )
        assert result.status == "complete"
        assert result.exact
        np.testing.assert_allclose(result.evaluation.objective, expected.objective)
        np.testing.assert_allclose(
            result.predicted_measurements, expected.measurement_mean
        )
        np.testing.assert_allclose(result.gradient, expected_gradient)
        assert result.progress == ()


def test_streaming_forward_reverse_share_selection_and_evict_every_shard(monkeypatch):
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary), groups=16)
        problem, raw = _problem(operator)
        calls: list[tuple[str, tuple[int, ...]]] = []
        resident_after_calls: list[int] = []
        operator_type = ShardedMatrixFreeFixedRoutingMeasurementOperator
        original_forward = operator_type.partial_matvec
        original_reverse = operator_type.partial_rmatvec

        def forward(self, *args, **kwargs):
            calls.append(("forward", kwargs["destination_group_indices"]))
            return original_forward(self, *args, **kwargs)

        def reverse(self, *args, **kwargs):
            calls.append(("reverse", kwargs["destination_group_indices"]))
            return original_reverse(self, *args, **kwargs)

        original_evict = operator_type.evict_resident_shards

        def evict(self):
            original_evict(self)
            resident_after_calls.append(self.resident_shards)

        monkeypatch.setattr(operator_type, "partial_matvec", forward)
        monkeypatch.setattr(operator_type, "partial_rmatvec", reverse)
        monkeypatch.setattr(operator_type, "evict_resident_shards", evict)
        result = stochastic_gravity_value_and_gradient(
            raw,
            problem=problem,
            config=StochasticGravityConfig(
                effort_percent=50, seed=7, rss_safety_margin_bytes=0
            ),
        )
        selected_groups = [
            tuple(operator.routing.shard_partition[index].destination_group_indices)
            for index in result.selection.selected_shard_ids
        ]
        forward_groups = [groups for phase, groups in calls if phase == "forward"]
        reverse_groups = [groups for phase, groups in calls if phase == "reverse"]
        assert forward_groups == selected_groups
        assert reverse_groups == selected_groups
        assert resident_after_calls == [0] * (2 * len(selected_groups))
        assert operator.resident_shards == 0
        assert result.completed_forward_shards == len(selected_groups)
        assert result.completed_reverse_shards == len(selected_groups)
        assert result.quality is not None
        assert np.isfinite(
            result.quality.measurement_standard_error_indicator
        )
        assert np.isfinite(result.quality.gradient_standard_error_indicator)


def test_sampled_forward_reverse_obey_adjoint_identity():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary), groups=16)
        selection = select_stochastic_routing_shards(
            tuple(operator.routing.shard_partition), effort_percent=25, seed=4
        )
        x = np.linspace(0.2, 1.2, operator.num_free_od, dtype=operator.dtype)
        y = np.linspace(-1.0, 1.0, operator.num_measurements, dtype=operator.dtype)
        forward = np.zeros(operator.num_measurements, dtype=operator.dtype)
        reverse = np.zeros(operator.num_free_od, dtype=operator.dtype)
        by_id = {
            item.shard_index: item for item in operator.routing.shard_partition
        }
        for shard_id, weight in zip(
            selection.selected_shard_ids, selection.sampling_weights, strict=True
        ):
            descriptor = by_id[shard_id]
            kwargs = {
                "destination_group_indices": descriptor.destination_group_indices,
                "padded_groups": descriptor.num_groups,
                "group_weights": (weight,) * descriptor.num_groups,
            }
            forward += operator.partial_matvec(x, **kwargs)
            operator.evict_resident_shards()
            reverse += operator.partial_rmatvec(y, **kwargs)
            operator.evict_resident_shards()
        np.testing.assert_allclose(
            np.vdot(forward, y), np.vdot(x, reverse), rtol=2.0e-5, atol=2.0e-5
        )


def test_anchor_is_reproduced_without_dispatching_a_shard(monkeypatch):
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary))
        problem, raw = _problem(operator)
        anchor = create_parallel_gravity_anchor(raw, problem=problem)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("anchor evaluation must not dispatch routing")

        monkeypatch.setattr(
            ShardedMatrixFreeFixedRoutingMeasurementOperator,
            "partial_matvec",
            forbidden,
        )
        monkeypatch.setattr(
            ShardedMatrixFreeFixedRoutingMeasurementOperator,
            "partial_rmatvec",
            forbidden,
        )
        result = stochastic_gravity_value_and_gradient(
            raw,
            problem=problem,
            anchor=anchor,
            config=StochasticGravityConfig(effort_percent=25),
        )
        np.testing.assert_array_equal(result.gradient, anchor.gradient)
        np.testing.assert_array_equal(
            result.predicted_measurements, anchor.measurement_mean
        )
        assert not result.exact


def test_anchored_approximation_matches_exact_small_operator_at_full_shard_sample():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary))
        problem, raw = _problem(operator)
        anchor = create_parallel_gravity_anchor(raw, problem=problem)
        target = raw + np.asarray([0.03, -0.02, 0.01], dtype=raw.dtype)
        exact, exact_gradient = gravity_value_and_gradient_adjoint(
            target, problem=problem
        )
        # 99% rounds up to every one of the four persisted shards while retaining
        # the sampled code path; every Horvitz--Thompson weight is therefore one.
        result = stochastic_gravity_value_and_gradient(
            target,
            problem=problem,
            anchor=anchor,
            config=StochasticGravityConfig(effort_percent=99),
        )
        assert not result.exact
        assert result.selection.realized_effort_percent == 100.0
        np.testing.assert_allclose(
            result.predicted_measurements,
            exact.measurement_mean,
            rtol=2.0e-5,
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            result.gradient, exact_gradient, rtol=2.0e-5, atol=2.0e-5
        )


def test_rss_guard_interrupts_before_loading_unsafe_shard(monkeypatch):
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary))
        problem, raw = _problem(operator)
        calls = 0

        def forbidden(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("unsafe shard must not be loaded")

        monkeypatch.setattr(
            ShardedMatrixFreeFixedRoutingMeasurementOperator,
            "partial_matvec",
            forbidden,
        )
        result = stochastic_gravity_value_and_gradient(
            raw,
            problem=problem,
            config=StochasticGravityConfig(
                effort_percent=25,
                rss_ceiling_bytes=1,
                rss_safety_margin_bytes=0,
                conservative_next_shard_bytes=1,
            ),
        )
        assert result.status == "interrupted"
        assert result.interrupted_phase == "forward"
        assert result.completed_forward_shards == 0
        assert result.resumable
        assert calls == 0


def test_invalid_concurrency_is_rejected_until_admission_control_supports_it():
    with pytest.raises(ValueError, match="concurrency=1"):
        StochasticGravityConfig(concurrency=2)


def test_streaming_temporary_allocations_are_bounded_and_collectible(monkeypatch):
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary), groups=16)
        problem, raw = _problem(operator)
        operator_type = ShardedMatrixFreeFixedRoutingMeasurementOperator
        original_forward = operator_type.partial_matvec
        original_reverse = operator_type.partial_rmatvec
        active = 0
        peak_active = 0
        references: list[weakref.ReferenceType[np.ndarray]] = []

        def instrument(original):
            def wrapped(self, *args, **kwargs):
                nonlocal active, peak_active
                allocation = np.ones(2 * 1024 * 1024, dtype=np.float32)
                references.append(weakref.ref(allocation))
                active += 1
                peak_active = max(peak_active, active)
                try:
                    return original(self, *args, **kwargs)
                finally:
                    active -= 1

            return wrapped

        monkeypatch.setattr(
            operator_type, "partial_matvec", instrument(original_forward)
        )
        monkeypatch.setattr(
            operator_type, "partial_rmatvec", instrument(original_reverse)
        )
        result = stochastic_gravity_value_and_gradient(
            raw,
            problem=problem,
            config=StochasticGravityConfig(effort_percent=50),
        )
        gc.collect()
        assert result.status == "complete"
        assert peak_active == 1
        assert all(reference() is None for reference in references)


def test_quality_dispersion_indicators_improve_with_nested_effort():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary), groups=16)
        problem, raw = _problem(operator)
        low = stochastic_gravity_value_and_gradient(
            raw,
            problem=problem,
            config=StochasticGravityConfig(effort_percent=25, seed=11),
        )
        high = stochastic_gravity_value_and_gradient(
            raw,
            problem=problem,
            config=StochasticGravityConfig(effort_percent=75, seed=11),
        )
        assert low.quality is not None and high.quality is not None
        assert (
            high.quality.measurement_standard_error_indicator
            < low.quality.measurement_standard_error_indicator
        )
        assert (
            high.quality.gradient_standard_error_indicator
            < low.quality.gradient_standard_error_indicator
        )
