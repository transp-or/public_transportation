from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.measurement.event_aligned import (
    build_event_aligned_aggregation_spec,
    predict_measurements_event_aligned,
)
from public_transportation.measurement.likelihood_jax import (
    predict_measurements_from_link_flow,
)
from public_transportation.measurement.mapping.spec import AggregationSpec


def test_event_aligned_prediction_and_gradient_match_generic_scatter():
    generic = AggregationSpec(
        num_measurements=4,
        measurement_index=np.asarray([2, 0, 3, 1, 0, 2], dtype=np.int32),
        link_index=np.asarray([5, 1, 0, 4, 3, 2], dtype=np.int32),
    )
    direct = build_event_aligned_aggregation_spec(generic)
    flow = jnp.linspace(0.2, 2.0, 6, dtype=jnp.float32)

    def generic_objective(value):
        prediction = predict_measurements_from_link_flow(
            value,
            spec_num_measurements=generic.num_measurements,
            spec_measurement_index=jnp.asarray(generic.measurement_index),
            spec_link_index=jnp.asarray(generic.link_index),
        )
        return jnp.square(prediction).sum()

    def direct_objective(value):
        prediction = predict_measurements_event_aligned(
            value,
            jnp.asarray(direct.primary_link_index),
            jnp.asarray(direct.secondary_measurement_index),
            jnp.asarray(direct.secondary_link_index),
        )
        return jnp.square(prediction).sum()

    generic_value, generic_gradient = jax.value_and_grad(generic_objective)(flow)
    direct_value, direct_gradient = jax.value_and_grad(direct_objective)(flow)
    np.testing.assert_allclose(direct_value, generic_value, rtol=0, atol=0)
    np.testing.assert_allclose(direct_gradient, generic_gradient, rtol=0, atol=0)
    np.testing.assert_array_equal(direct.primary_link_index, [1, 4, 5, 0])
    np.testing.assert_array_equal(direct.secondary_measurement_index, [0, 2])
    np.testing.assert_array_equal(direct.secondary_link_index, [3, 2])


def test_event_aligned_rejects_more_than_two_links():
    with pytest.raises(ValueError, match="at most two"):
        build_event_aligned_aggregation_spec(
            AggregationSpec(
                num_measurements=1,
                measurement_index=np.asarray([0, 0, 0], dtype=np.int32),
                link_index=np.asarray([1, 2, 3], dtype=np.int32),
            )
        )
