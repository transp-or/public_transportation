"""Auditable advisory parent/child lineage and zero-effect warm starts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from public_transportation.preprocessing.reduced_od.artifacts import fingerprint_json

from .diagnostics import ReducedODModelStage


@dataclass(frozen=True, slots=True)
class ReducedODLineageNode:
    identifier: str
    stage: ReducedODModelStage
    model_fingerprint: str
    parent_identifier: str | None
    parameter_names: tuple[str, ...]
    raw_parameters: np.ndarray
    fitted: bool

    def __post_init__(self) -> None:
        raw = np.array(self.raw_parameters, dtype=np.float64, copy=True)
        if raw.shape != (len(self.parameter_names),) or not np.all(np.isfinite(raw)):
            raise ValueError("lineage parameters must align and be finite.")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("lineage parameter names must be unique.")
        raw.setflags(write=False)
        object.__setattr__(self, "raw_parameters", raw)


def create_reduced_od_root_node(
    *,
    model_fingerprint: str,
    parameter_names: tuple[str, ...],
    raw_parameters: object,
) -> ReducedODLineageNode:
    raw = np.asarray(raw_parameters, dtype=np.float64)
    payload = {
        "stage": "J0",
        "model_fingerprint": model_fingerprint,
        "parameter_names": list(parameter_names),
        "raw_parameters": raw.tolist(),
    }
    return ReducedODLineageNode(
        identifier=fingerprint_json(payload),
        stage="J0",
        model_fingerprint=model_fingerprint,
        parent_identifier=None,
        parameter_names=parameter_names,
        raw_parameters=raw,
        fitted=True,
    )


@dataclass(frozen=True, slots=True)
class ReducedODChildWarmStart:
    parent_identifier: str
    stage: ReducedODModelStage
    parameter_names: tuple[str, ...]
    raw_parameters: np.ndarray
    maximum_parent_prediction_difference: float
    verified: bool

    def __post_init__(self) -> None:
        raw = np.array(self.raw_parameters, dtype=np.float64, copy=True)
        raw.setflags(write=False)
        object.__setattr__(self, "raw_parameters", raw)
        if not self.verified:
            raise ValueError("an unverified child warm start cannot be used.")


def construct_reduced_od_child_warm_start(
    *,
    parent: ReducedODLineageNode,
    stage: ReducedODModelStage,
    added_parameter_names: tuple[str, ...],
    parent_prediction: object,
    child_prediction_at_warm_start: object,
    tolerance: float = 1.0e-10,
) -> ReducedODChildWarmStart:
    """Append zero-effect parameters and verify exact parent prediction."""
    if stage == "J0":
        raise ValueError("J0 is a root, not a child relaxation.")
    if not added_parameter_names or set(added_parameter_names) & set(
        parent.parameter_names
    ):
        raise ValueError("child parameter names must be new and non-empty.")
    first = np.asarray(parent_prediction, dtype=np.float64)
    second = np.asarray(child_prediction_at_warm_start, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("parent and child predictions must be aligned vectors.")
    difference = float(np.max(np.abs(first - second), initial=0.0))
    if difference > tolerance:
        raise ValueError("child warm start does not reproduce the parent prediction.")
    names = parent.parameter_names + added_parameter_names
    raw = np.concatenate((parent.raw_parameters, np.zeros(len(added_parameter_names))))
    return ReducedODChildWarmStart(
        parent_identifier=parent.identifier,
        stage=stage,
        parameter_names=names,
        raw_parameters=raw,
        maximum_parent_prediction_difference=difference,
        verified=True,
    )


def create_reduced_od_advisory_child(
    *,
    parent: ReducedODLineageNode,
    warm_start: ReducedODChildWarmStart,
    model_fingerprint: str,
) -> ReducedODLineageNode:
    if warm_start.parent_identifier != parent.identifier:
        raise ValueError("warm-start parent does not match the lineage parent.")
    payload = {
        "stage": warm_start.stage,
        "model_fingerprint": model_fingerprint,
        "parent_identifier": parent.identifier,
        "parameter_names": list(warm_start.parameter_names),
        "raw_parameters": warm_start.raw_parameters.tolist(),
    }
    return ReducedODLineageNode(
        identifier=fingerprint_json(payload),
        stage=warm_start.stage,
        model_fingerprint=model_fingerprint,
        parent_identifier=parent.identifier,
        parameter_names=warm_start.parameter_names,
        raw_parameters=warm_start.raw_parameters,
        fitted=False,
    )


@dataclass(frozen=True, slots=True)
class ReducedODModelLineage:
    nodes: tuple[ReducedODLineageNode, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for node in self.nodes:
            if node.identifier in seen:
                raise ValueError("lineage contains a duplicate node.")
            if (
                node.parent_identifier is not None
                and node.parent_identifier not in seen
            ):
                raise ValueError("every child parent must precede it in the lineage.")
            seen.add(node.identifier)

    def append(self, node: ReducedODLineageNode) -> ReducedODModelLineage:
        return ReducedODModelLineage(self.nodes + (node,))
