from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from public_transportation.inference.reduced_od import (
    JourneyODTimeKey,
    ReducedODModelContract,
    ReducedODProblemContract,
)


def _keys() -> tuple[JourneyODTimeKey, ...]:
    return (
        JourneyODTimeKey("A", "B", "T1"),
        JourneyODTimeKey("A", "C", "T1"),
        JourneyODTimeKey("B", "C", "T1"),
    )


def _problem() -> ReducedODProblemContract:
    return ReducedODProblemContract(
        configuration_fingerprint="configuration",
        timetable_artifact_fingerprint="timetable",
        response_artifact_fingerprint="response",
        od_keys=_keys(),
        free_od_indices=np.asarray([0, 2]),
        fixed_od_indices=np.asarray([1]),
        fixed_od_values=np.asarray([0.0]),
    )


def test_problem_partition_is_exact_immutable_and_fingerprinted() -> None:
    problem = _problem()

    assert problem.num_od == 3
    assert problem.num_free_od == 2
    assert len(problem.fingerprint) == 64
    assert not problem.free_od_indices.flags.writeable
    assert not problem.fixed_od_indices.flags.writeable
    assert not problem.fixed_od_values.flags.writeable
    with pytest.raises(FrozenInstanceError):
        problem.schema_version = 2


def test_problem_owns_array_inputs_and_is_deterministic() -> None:
    free = np.asarray([0, 2])
    first = replace(_problem(), free_od_indices=free)
    second = _problem()
    free[0] = 1

    assert first.free_od_indices.tolist() == [0, 2]
    assert first.fingerprint_payload_json == second.fingerprint_payload_json
    assert first.fingerprint == second.fingerprint


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"od_keys": tuple(reversed(_keys()))}, "canonical sorted"),
        ({"free_od_indices": np.asarray([0, 1])}, "disjoint"),
        ({"free_od_indices": np.asarray([2, 0])}, "strictly increasing"),
        (
            {"fixed_od_values": np.asarray([-1.0])},
            "non-negative",
        ),
        (
            {"fixed_od_indices": np.asarray([], dtype=int),
             "fixed_od_values": np.asarray([], dtype=float)},
            "partition",
        ),
    ],
)
def test_problem_rejects_invalid_partitions(overrides, message) -> None:
    values = {
        "configuration_fingerprint": "configuration",
        "timetable_artifact_fingerprint": "timetable",
        "response_artifact_fingerprint": "response",
        "od_keys": _keys(),
        "free_od_indices": np.asarray([0, 2]),
        "fixed_od_indices": np.asarray([1]),
        "fixed_od_values": np.asarray([0.0]),
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        ReducedODProblemContract(**values)


def test_model_contract_is_canonical_and_sensitive() -> None:
    first = ReducedODModelContract(
        problem_fingerprint=_problem().fingerprint,
        model_name="J0",
        production_mode="provided",
        likelihood="poisson",
        estimated_parameters=("beta_time", "beta_transfer"),
    )
    again = replace(first)
    changed = replace(first, likelihood="negative_binomial")

    assert first.fingerprint == again.fingerprint
    assert first.fingerprint != changed.fingerprint
    with pytest.raises(ValueError, match="sorted"):
        replace(first, estimated_parameters=("z", "a"))
