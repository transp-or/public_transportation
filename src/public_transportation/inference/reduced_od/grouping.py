"""Deterministic network-constrained hierarchy for model-resolution regions."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from public_transportation.preprocessing.reduced_od.artifacts import canonical_json

GroupingProgress = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class GroupingHierarchy:
    node_ids: tuple[str, ...]
    normalized_signatures: np.ndarray
    levels: tuple[tuple[int, ...], ...]
    parent_labels: tuple[tuple[int, ...], ...]
    normalization_mean: tuple[float, ...]
    normalization_scale: tuple[float, ...]
    disconnected_nodes: tuple[str, ...]
    schema_version: int = 1

    @property
    def fingerprint(self) -> str:
        payload = {
            **asdict(self),
            "normalized_signatures": np.asarray(self.normalized_signatures).tolist(),
        }
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()

    def membership(self, target_groups: int) -> tuple[int, ...]:
        matches = [level for level in self.levels if len(set(level)) == target_groups]
        if not matches:
            raise ValueError(
                "requested group count is not represented in the hierarchy."
            )
        return matches[0]


def build_grouping_hierarchy(
    *,
    node_ids: Sequence[str],
    signatures: object,
    adjacency: Sequence[tuple[str, str]] = (),
    progress: GroupingProgress | None = None,
) -> GroupingHierarchy:
    """Agglomerate adjacent groups by normalized signature distance."""
    ids = tuple(sorted(node_ids))
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("node identifiers must be unique and nonempty.")
    source_order = {value: index for index, value in enumerate(node_ids)}
    x = np.asarray(signatures, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != len(ids) or not np.all(np.isfinite(x)):
        raise ValueError("signatures must be a finite node-by-feature matrix.")
    x = x[[source_order[item] for item in ids]]
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    z = (x - mean) / scale
    index = {item: position for position, item in enumerate(ids)}
    edges = {
        tuple(sorted((index[left], index[right])))
        for left, right in adjacency
        if left in index and right in index and left != right
    }
    neighbors = {value for edge in edges for value in edge}
    disconnected = tuple(ids[item] for item in range(len(ids)) if item not in neighbors)
    clusters: dict[int, tuple[int, ...]] = {item: (item,) for item in range(len(ids))}
    levels: list[tuple[int, ...]] = []
    parents: list[tuple[int, ...]] = []
    started = __import__("time").perf_counter()
    while clusters:
        ordered = sorted(
            clusters.values(), key=lambda members: tuple(ids[item] for item in members)
        )
        labels = tuple(
            next(label for label, members in enumerate(ordered) if node in members)
            for node in range(len(ids))
        )
        levels.append(labels)
        if len(clusters) == 1:
            break
        candidates: list[tuple[float, tuple[str, ...], int, int]] = []
        keys = sorted(clusters)
        for position, left in enumerate(keys):
            for right in keys[position + 1 :]:
                adjacent = not edges or any(
                    tuple(sorted((a, b))) in edges
                    for a in clusters[left]
                    for b in clusters[right]
                )
                if not adjacent:
                    continue
                left_mean, right_mean = (
                    z[list(clusters[left])].mean(axis=0),
                    z[list(clusters[right])].mean(axis=0),
                )
                names = tuple(
                    ids[item] for item in sorted(clusters[left] + clusters[right])
                )
                candidates.append(
                    (float(np.sum((left_mean - right_mean) ** 2)), names, left, right)
                )
        if not candidates:
            # Connect components deterministically at the top of the hierarchy.
            left, right = keys[:2]
        else:
            _, _, left, right = min(candidates)
        merged = tuple(sorted(clusters.pop(left) + clusters.pop(right)))
        clusters[min(left, right)] = merged
        parent = list(range(len(set(labels))))
        left_label, right_label = labels[merged[0]], labels[merged[-1]]
        parent[right_label] = left_label
        parents.append(tuple(parent))
        if progress is not None:
            elapsed = __import__("time").perf_counter() - started
            progress(
                {
                    "phase": "grouping_hierarchy",
                    "status": "completed" if len(clusters) == 1 else "in_progress",
                    "completed_merges": len(ids) - len(clusters),
                    "total_merges": max(0, len(ids) - 1),
                    "elapsed_seconds": elapsed,
                    "current_groups": len(clusters),
                }
            )
    array = np.array(z, copy=True)
    array.setflags(write=False)
    return GroupingHierarchy(
        ids,
        array,
        tuple(levels),
        tuple(parents),
        tuple(mean),
        tuple(scale),
        disconnected,
    )
