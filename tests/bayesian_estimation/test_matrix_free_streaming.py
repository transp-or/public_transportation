from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.inference.matrix_free_streaming import (
    StreamedDestinationGroup,
    replayable_streamed_measurement_value_and_grad,
    streamed_measurement_value_and_grad,
)


def test_streamed_two_pass_matches_monolithic_autodiff_with_fixed_offsets():
    parameter = jnp.asarray([0.2, -0.4, 0.7, 0.1], dtype=jnp.float32)
    first_matrix = jnp.asarray(
        [[1.0, 2.0], [0.0, 3.0], [2.0, -1.0]], dtype=jnp.float32
    )
    second_matrix = jnp.asarray(
        [[-1.0, 0.5], [2.0, 0.0], [1.0, 4.0]], dtype=jnp.float32
    )
    fixed_offset = jnp.asarray([3.0, 1.0, 2.0], dtype=jnp.float32)

    def first(local):
        return first_matrix @ jnp.exp(local) + fixed_offset

    def second(local):
        return second_matrix @ jnp.square(local)

    groups = (
        StreamedDestinationGroup(np.asarray([0, 2]), first),
        StreamedDestinationGroup(np.asarray([1, 3]), second),
    )

    def objective(prediction):
        return jnp.sum(jnp.log1p(jnp.square(prediction)))

    streamed = streamed_measurement_value_and_grad(
        parameter=parameter,
        groups=groups,
        num_measurements=3,
        measurement_objective=objective,
    )

    def monolithic(value):
        return objective(first(value[jnp.asarray([0, 2])]) + second(value[jnp.asarray([1, 3])]))

    expected_value, expected_gradient = jax.value_and_grad(monolithic)(parameter)
    np.testing.assert_allclose(streamed.value, expected_value, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        streamed.gradient, expected_gradient, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        streamed.prediction,
        first(parameter[jnp.asarray([0, 2])])
        + second(parameter[jnp.asarray([1, 3])]),
        rtol=0,
        atol=0,
    )
    assert streamed.num_groups == 2


def test_streamed_rejects_overlapping_parameter_groups():
    groups = (
        StreamedDestinationGroup(np.asarray([0, 1]), lambda x: x),
        StreamedDestinationGroup(np.asarray([1, 2]), lambda x: x),
    )
    with pytest.raises(ValueError, match="unique across groups"):
        streamed_measurement_value_and_grad(
            parameter=jnp.ones((3,), dtype=jnp.float32),
            groups=groups,
            num_measurements=2,
            measurement_objective=jnp.sum,
        )


def test_streamed_rejects_wrong_prediction_shape():
    with pytest.raises(ValueError, match="prediction must have shape"):
        streamed_measurement_value_and_grad(
            parameter=jnp.ones((1,), dtype=jnp.float32),
            groups=(
                StreamedDestinationGroup(
                    np.asarray([0]), lambda x: jnp.concatenate((x, x))
                ),
            ),
            num_measurements=1,
            measurement_objective=jnp.sum,
        )


def test_replayable_provider_matches_in_memory_streaming_and_runs_twice():
    parameter = jnp.asarray([0.3, -0.2, 0.8], dtype=jnp.float32)
    calls = 0

    def provider():
        nonlocal calls
        calls += 1
        yield StreamedDestinationGroup(
            np.asarray([0, 2]),
            lambda x: jnp.asarray([x[0] + x[1], x[0] * x[1]]),
            persistent_bytes=128,
            label="first",
        )
        yield StreamedDestinationGroup(
            np.asarray([1]),
            lambda x: jnp.asarray([jnp.exp(x[0]), jnp.square(x[0])]),
            persistent_bytes=64,
            label="second",
        )

    def objective(prediction):
        return jnp.sum(jnp.log1p(jnp.square(prediction)))

    replayed = replayable_streamed_measurement_value_and_grad(
        parameter=parameter,
        group_provider=provider,
        num_measurements=2,
        measurement_objective=objective,
        memory_ceiling_bytes=256,
    )
    expected = streamed_measurement_value_and_grad(
        parameter=parameter,
        groups=tuple(provider()),
        num_measurements=2,
        measurement_objective=objective,
    )

    assert calls == 3
    assert replayed.maximum_group_bytes == 128
    np.testing.assert_allclose(replayed.value, expected.value, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        replayed.gradient, expected.gradient, rtol=2e-6, atol=2e-6
    )


def test_replayable_provider_enforces_memory_ceiling():
    def provider():
        yield StreamedDestinationGroup(
            np.asarray([0]), lambda x: x, persistent_bytes=129, label="large"
        )

    with pytest.raises(MemoryError, match="large.*129.*128-byte ceiling"):
        replayable_streamed_measurement_value_and_grad(
            parameter=jnp.ones((1,), dtype=jnp.float32),
            group_provider=provider,
            num_measurements=1,
            measurement_objective=jnp.sum,
            memory_ceiling_bytes=128,
        )


@pytest.mark.parametrize("mode", ["changed", "missing", "extra"])
def test_replayable_provider_rejects_inconsistent_second_pass(mode):
    calls = 0

    def provider():
        nonlocal calls
        calls += 1
        yield StreamedDestinationGroup(np.asarray([0]), lambda x: x)
        if calls == 1 or mode == "extra":
            yield StreamedDestinationGroup(np.asarray([1]), lambda x: x)
        if calls == 2 and mode == "changed":
            yield StreamedDestinationGroup(np.asarray([2]), lambda x: x)
        if calls == 2 and mode == "extra":
            yield StreamedDestinationGroup(np.asarray([2]), lambda x: x)

    message = {
        "changed": "changed indices",
        "missing": "fewer groups",
        "extra": "extra groups",
    }[mode]
    with pytest.raises(ValueError, match=message):
        replayable_streamed_measurement_value_and_grad(
            parameter=jnp.ones((3,), dtype=jnp.float32),
            group_provider=provider,
            num_measurements=1,
            measurement_objective=jnp.sum,
        )
