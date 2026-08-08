"""Compact conditional-gravity features aligned with free response cells."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from public_transportation.preprocessing.reduced_od.artifacts import canonical_json
from public_transportation.preprocessing.reduced_od.journey_choices import (
    JourneyChoiceResult,
)
from public_transportation.preprocessing.reduced_od.response_atoms import (
    MeasurementResponseArtifact,
    ResponseCellKey,
)


def _immutable(value: object, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ConditionalGravityFeatures:
    """One feature row per free journey cell and compact origin-time groups."""

    cell_keys: tuple[ResponseCellKey, ...]
    origin_time_group_index: np.ndarray
    destination_index: np.ndarray
    journey_time_seconds: np.ndarray
    transfer_count: np.ndarray
    destination_attractiveness: np.ndarray
    baseline_productions: np.ndarray
    origin_time_group_keys: tuple[tuple[str, str], ...]
    destination_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.cell_keys != tuple(sorted(self.cell_keys)) or len(
            set(self.cell_keys)
        ) != len(self.cell_keys):
            raise ValueError("cell_keys must be unique and sorted.")
        groups = _immutable(
            self.origin_time_group_index,
            np.dtype(np.int64),
            "origin_time_group_index",
        )
        destinations = _immutable(
            self.destination_index, np.dtype(np.int64), "destination_index"
        )
        times = _immutable(
            self.journey_time_seconds,
            np.dtype(np.float64),
            "journey_time_seconds",
        )
        transfers = _immutable(
            self.transfer_count, np.dtype(np.float64), "transfer_count"
        )
        attractiveness = _immutable(
            self.destination_attractiveness,
            np.dtype(np.float64),
            "destination_attractiveness",
        )
        productions = _immutable(
            self.baseline_productions,
            np.dtype(np.float64),
            "baseline_productions",
        )
        cells = len(self.cell_keys)
        if any(
            value.size != cells
            for value in (groups, destinations, times, transfers, attractiveness)
        ):
            raise ValueError("every gravity cell feature must align with cell_keys.")
        if self.origin_time_group_keys != tuple(sorted(self.origin_time_group_keys)):
            raise ValueError("origin_time_group_keys must be sorted.")
        if len(set(self.origin_time_group_keys)) != len(self.origin_time_group_keys):
            raise ValueError("origin_time_group_keys must be unique.")
        if self.destination_ids != tuple(sorted(self.destination_ids)) or len(
            set(self.destination_ids)
        ) != len(self.destination_ids):
            raise ValueError("destination_ids must be unique and sorted.")
        if productions.size != len(self.origin_time_group_keys):
            raise ValueError("baseline productions must align with origin-time groups.")
        if cells and (
            np.any(groups < 0)
            or np.any(groups >= productions.size)
            or np.any(destinations < 0)
            or np.any(destinations >= len(self.destination_ids))
        ):
            raise ValueError("gravity group or destination indices are invalid.")
        if productions.size and not np.array_equal(
            np.unique(groups), np.arange(productions.size)
        ):
            raise ValueError("every origin-time group must contain a free destination.")
        if (
            not np.all(np.isfinite(times))
            or np.any(times < 0.0)
            or not np.all(np.isfinite(transfers))
            or np.any(transfers < 0.0)
            or not np.all(np.isfinite(attractiveness))
            or np.any(attractiveness <= 0.0)
            or not np.all(np.isfinite(productions))
            or np.any(productions < 0.0)
        ):
            raise ValueError("gravity features must be finite with valid signs.")
        object.__setattr__(self, "origin_time_group_index", groups)
        object.__setattr__(self, "destination_index", destinations)
        object.__setattr__(self, "journey_time_seconds", times)
        object.__setattr__(self, "transfer_count", transfers)
        object.__setattr__(self, "destination_attractiveness", attractiveness)
        object.__setattr__(self, "baseline_productions", productions)

    @property
    def number_of_cells(self) -> int:
        return len(self.cell_keys)

    @property
    def number_of_origin_time_groups(self) -> int:
        return len(self.origin_time_group_keys)

    @property
    def fingerprint(self) -> str:
        payload = {
            "cell_keys": [list(key.tuple) for key in self.cell_keys],
            "destination_ids": list(self.destination_ids),
            "origin_time_group_keys": [
                list(key) for key in self.origin_time_group_keys
            ],
            "arrays": [
                [
                    array.dtype.str,
                    list(array.shape),
                    hashlib.sha256(array.tobytes()).hexdigest(),
                ]
                for array in (
                    self.origin_time_group_index,
                    self.destination_index,
                    self.journey_time_seconds,
                    self.transfer_count,
                    self.destination_attractiveness,
                    self.baseline_productions,
                )
            ],
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_conditional_gravity_features(
    *,
    response: MeasurementResponseArtifact,
    journey_choices: JourneyChoiceResult,
    productions: Mapping[tuple[str, str], float],
    destination_attractiveness: Mapping[tuple[str, str], float],
) -> ConditionalGravityFeatures:
    """Aggregate fixed alternative features and align them with free columns."""
    choices = {
        ResponseCellKey(
            item.origin_physical_stop_id,
            item.destination_physical_stop_id,
            item.origin_time_period_id,
        ): item
        for item in journey_choices.choice_sets
    }
    missing = sorted(set(response.free_cell_keys) - set(choices))
    if missing:
        raise ValueError(
            f"free response cells lack journey choices: {[key.tuple for key in missing]}."
        )
    group_keys = tuple(
        sorted(
            {
                (key.origin_physical_stop_id, key.origin_time_period_id)
                for key in response.free_cell_keys
            }
        )
    )
    if set(productions) != set(group_keys):
        raise ValueError(
            "productions must cover exactly the free origin-time groups; "
            f"provided={sorted(productions)}, required={list(group_keys)}."
        )
    group_index = {key: index for index, key in enumerate(group_keys)}
    destination_ids = tuple(
        sorted({key.destination_physical_stop_id for key in response.free_cell_keys})
    )
    destination_index = {
        destination: index for index, destination in enumerate(destination_ids)
    }
    expected_time: list[float] = []
    expected_transfers: list[float] = []
    attractions: list[float] = []
    groups: list[int] = []
    destinations: list[int] = []
    for key in response.free_cell_keys:
        choice = choices[key]
        expected_time.append(
            sum(
                share * alternative.travel_seconds
                for alternative, share in zip(
                    choice.alternatives, choice.initial_shares, strict=True
                )
            )
        )
        expected_transfers.append(
            sum(
                share * alternative.transfers
                for alternative, share in zip(
                    choice.alternatives, choice.initial_shares, strict=True
                )
            )
        )
        attraction_key = (
            key.destination_physical_stop_id,
            key.origin_time_period_id,
        )
        if attraction_key not in destination_attractiveness:
            raise ValueError(f"missing destination attractiveness {attraction_key!r}.")
        attractions.append(float(destination_attractiveness[attraction_key]))
        groups.append(
            group_index[(key.origin_physical_stop_id, key.origin_time_period_id)]
        )
        destinations.append(destination_index[key.destination_physical_stop_id])
    return ConditionalGravityFeatures(
        cell_keys=response.free_cell_keys,
        origin_time_group_index=np.asarray(groups, dtype=np.int64),
        destination_index=np.asarray(destinations, dtype=np.int64),
        journey_time_seconds=np.asarray(expected_time, dtype=np.float64),
        transfer_count=np.asarray(expected_transfers, dtype=np.float64),
        destination_attractiveness=np.asarray(attractions, dtype=np.float64),
        baseline_productions=np.asarray(
            [float(productions[key]) for key in group_keys], dtype=np.float64
        ),
        origin_time_group_keys=group_keys,
        destination_ids=destination_ids,
    )
