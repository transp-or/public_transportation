"""Conflict-aware deterministic scheduling and parallel block batches."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from threadpoolctl import threadpool_limits

from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
)

from ._canonical import fingerprint
from .blocks import ODBlock
from .incremental import (
    BlockUpdateProposal,
    IncrementalLinearState,
    apply_incremental_update,
)
from .objective import SeparableQuadraticPrior, build_conditional_block_objective
from .operator import BlockLinearOperatorProtocol
from .results import BlockObjectiveComponents
from .solver import (
    BlockSolverConfig,
    BlockSolverResult,
    BlockUpdateDecision,
    BlockUpdatePolicy,
    decide_block_update,
    solve_conditional_block,
)

BlockOperatorFactory = Callable[[ODBlock], BlockLinearOperatorProtocol]


@dataclass(frozen=True, slots=True)
class BlockConflictGraph:
    """Undirected conflicts in deterministic partition order."""

    blocks: tuple[ODBlock, ...]
    adjacency: tuple[tuple[int, ...], ...]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        adjacency = tuple(tuple(neighbors) for neighbors in self.adjacency)
        if len(adjacency) != len(blocks):
            raise ValueError("conflict adjacency must have one row per block.")
        for index, neighbors in enumerate(adjacency):
            if neighbors != tuple(sorted(set(neighbors))):
                raise ValueError("conflict neighbors must be unique and ascending.")
            if any(item < 0 or item >= len(blocks) or item == index for item in neighbors):
                raise ValueError("conflict adjacency contains an invalid neighbor.")
            if any(index not in adjacency[item] for item in neighbors):
                raise ValueError("conflict adjacency must be symmetric.")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "adjacency", adjacency)
        object.__setattr__(
            self,
            "fingerprint",
            fingerprint(
                {
                    "version": 1,
                    "block_fingerprints": tuple(block.fingerprint for block in blocks),
                    "adjacency": adjacency,
                }
            ),
        )

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (left, right)
            for left, neighbors in enumerate(self.adjacency)
            for right in neighbors
            if left < right
        )


def build_block_conflict_graph(
    blocks: tuple[ODBlock, ...],
    *,
    additional_couplings: tuple[tuple[str, str], ...] = (),
) -> BlockConflictGraph:
    """Connect blocks sharing measurement support or declared coupling."""
    blocks = tuple(blocks)
    if any(block.measurement_support_indices is None for block in blocks):
        raise ValueError(
            "conflict scheduling requires exact measurement_support_indices "
            "for every block"
        )
    identifiers = [block.block_id for block in blocks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("block identifiers must be unique.")
    by_identifier = {identifier: index for index, identifier in enumerate(identifiers)}
    adjacency = [set() for _ in blocks]
    owners: dict[int, list[int]] = {}
    for index, block in enumerate(blocks):
        assert block.measurement_support_indices is not None
        for measurement in block.measurement_support_indices:
            for other in owners.setdefault(measurement, []):
                adjacency[index].add(other)
                adjacency[other].add(index)
            owners[measurement].append(index)
    for left_id, right_id in additional_couplings:
        try:
            left = by_identifier[left_id]
            right = by_identifier[right_id]
        except KeyError as error:
            raise ValueError("additional coupling contains an unknown block.") from error
        if left == right:
            raise ValueError("additional coupling may not be a self-edge.")
        adjacency[left].add(right)
        adjacency[right].add(left)
    return BlockConflictGraph(
        blocks=blocks,
        adjacency=tuple(tuple(sorted(neighbors)) for neighbors in adjacency),
    )


@dataclass(frozen=True, slots=True)
class ConflictFreeBlockSchedule:
    graph_fingerprint: str
    batches: tuple[tuple[ODBlock, ...], ...]
    color_by_block: tuple[int, ...]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.graph_fingerprint.strip():
            raise ValueError("graph_fingerprint must be nonempty.")
        batches = tuple(tuple(batch) for batch in self.batches)
        if not batches or any(not batch for batch in batches):
            raise ValueError("conflict-free schedule batches must be nonempty.")
        object.__setattr__(self, "batches", batches)
        object.__setattr__(
            self,
            "fingerprint",
            fingerprint(
                {
                    "version": 1,
                    "graph_fingerprint": self.graph_fingerprint,
                    "batches": tuple(
                        tuple(block.fingerprint for block in batch) for batch in batches
                    ),
                    "color_by_block": self.color_by_block,
                }
            ),
        )


def color_block_conflict_graph(
    graph: BlockConflictGraph,
) -> ConflictFreeBlockSchedule:
    """Apply deterministic first-fit coloring in partition order."""
    colors: list[int] = []
    for index, neighbors in enumerate(graph.adjacency):
        unavailable = {colors[item] for item in neighbors if item < index}
        color = 0
        while color in unavailable:
            color += 1
        colors.append(color)
    batches = tuple(
        tuple(graph.blocks[index] for index, color in enumerate(colors) if color == value)
        for value in range(max(colors, default=-1) + 1)
    )
    return ConflictFreeBlockSchedule(
        graph_fingerprint=graph.fingerprint,
        batches=batches,
        color_by_block=tuple(colors),
    )


@dataclass(frozen=True, slots=True)
class ParallelBlockExecutionConfig:
    construction_workers: int = 1
    solver_workers: int = 1
    threads_per_worker: int = 1
    available_cpus: int | None = None

    def __post_init__(self) -> None:
        for name in ("construction_workers", "solver_workers", "threads_per_worker"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        available = os.cpu_count() if self.available_cpus is None else self.available_cpus
        if available is not None and available <= 0:
            raise ValueError("available_cpus must be positive when provided.")
        maximum_workers = max(self.construction_workers, self.solver_workers)
        if available is not None and maximum_workers * self.threads_per_worker > available:
            raise ValueError(
                "worker count multiplied by threads_per_worker exceeds available CPUs."
            )


def construct_block_operators(
    blocks: tuple[ODBlock, ...],
    factory: BlockOperatorFactory,
    *,
    workers: int,
) -> tuple[BlockLinearOperatorProtocol, ...]:
    """Construct independent operators concurrently and preserve input order."""
    if workers <= 0:
        raise ValueError("construction workers must be positive.")
    if workers == 1 or len(blocks) <= 1:
        return tuple(factory(block) for block in blocks)
    with ThreadPoolExecutor(max_workers=min(workers, len(blocks))) as executor:
        return tuple(executor.map(factory, blocks))


@dataclass(frozen=True, slots=True)
class ConflictFreeBatchDecision:
    batch_id: str
    blocks: tuple[ODBlock, ...]
    state: IncrementalLinearState
    block_decisions: tuple[BlockUpdateDecision, ...]
    components: BlockObjectiveComponents
    merged_proposal: BlockUpdateProposal | None
    accepted_blocks: int
    rejected_blocks: int
    objective_improvement: float
    maximum_flow_change: float
    maximum_projected_gradient: float

    @property
    def accepted(self) -> bool:
        return self.accepted_blocks > 0


def conflict_free_batch_id(blocks: tuple[ODBlock, ...]) -> str:
    return "batch-" + fingerprint(tuple(block.fingerprint for block in blocks))[:16]


def solve_conflict_free_batch(
    *,
    problem: FixedRoutingLinearProblem,
    prior: SeparableQuadraticPrior,
    state: IncrementalLinearState,
    blocks: tuple[ODBlock, ...],
    operator_factory: BlockOperatorFactory,
    solver_config: BlockSolverConfig,
    update_policy: BlockUpdatePolicy,
    parallel_config: ParallelBlockExecutionConfig,
) -> ConflictFreeBatchDecision:
    """Solve independent blocks concurrently and merge accepted deltas atomically."""
    if not blocks:
        raise ValueError("a parallel batch must contain at least one block.")
    supports: set[int] = set()
    for block in blocks:
        if block.measurement_support_indices is None:
            raise ValueError("parallel blocks require exact measurement support.")
        overlap = supports.intersection(block.measurement_support_indices)
        if overlap:
            raise ValueError("parallel batch contains overlapping measurement support.")
        supports.update(block.measurement_support_indices)
    with threadpool_limits(limits=parallel_config.threads_per_worker):
        operators = construct_block_operators(
            blocks,
            operator_factory,
            workers=parallel_config.construction_workers,
        )
    objectives = tuple(
        build_conditional_block_objective(problem, prior, state, block, operator)
        for block, operator in zip(blocks, operators, strict=True)
    )
    initial_values = tuple(
        state.free_flow[np.asarray(block.free_column_indices, dtype=np.intp)]
        for block in blocks
    )

    def solve(arguments) -> BlockSolverResult:
        objective, initial = arguments
        return solve_conditional_block(objective, initial, config=solver_config)

    arguments = tuple(zip(objectives, initial_values, strict=True))
    with threadpool_limits(limits=parallel_config.threads_per_worker):
        if parallel_config.solver_workers == 1 or len(blocks) == 1:
            solved = tuple(solve(argument) for argument in arguments)
        else:
            with ThreadPoolExecutor(
                max_workers=min(parallel_config.solver_workers, len(blocks))
            ) as executor:
                solved = tuple(executor.map(solve, arguments))
    decisions = tuple(
        decide_block_update(
            state,
            block,
            operator,
            objective,
            result,
            policy=update_policy,
        )
        for block, operator, objective, result in zip(
            blocks, operators, objectives, solved, strict=True
        )
    )
    accepted = tuple(decision for decision in decisions if decision.accepted)
    if not accepted:
        residual = state.prediction - problem.observations
        components = BlockObjectiveComponents(
            data=float(0.5 * np.dot(problem.observation_weights, residual * residual)),
            prior=prior.objective(state.free_flow),
        )
        return ConflictFreeBatchDecision(
            batch_id=conflict_free_batch_id(blocks),
            blocks=blocks,
            state=state,
            block_decisions=decisions,
            components=components,
            merged_proposal=None,
            accepted_blocks=0,
            rejected_blocks=len(decisions),
            objective_improvement=0.0,
            maximum_flow_change=0.0,
            maximum_projected_gradient=max(
                decision.accepted_evaluation.projected_gradient_norm
                for decision in decisions
            ),
        )
    columns = tuple(
        sorted(
            column
            for decision in accepted
            for column in decision.proposal.free_column_indices  # type: ignore[union-attr]
        )
    )
    before = state.free_flow[np.asarray(columns, dtype=np.intp)]
    after_by_column = dict(zip(columns, before, strict=True))
    prediction_delta = np.zeros_like(state.prediction)
    for decision in accepted:
        assert decision.proposal is not None
        after_by_column.update(
            zip(
                decision.proposal.free_column_indices,
                decision.proposal.flow_after,
                strict=True,
            )
        )
        prediction_delta += decision.proposal.prediction_delta
    after = np.asarray([after_by_column[column] for column in columns])
    merged = BlockUpdateProposal(
        block_id=conflict_free_batch_id(blocks),
        block_fingerprint=fingerprint(
            tuple(block.fingerprint for block in blocks)
        ),
        free_column_indices=columns,
        flow_before=before,
        flow_after=after,
        flow_delta=after - before,
        prediction_delta=prediction_delta,
        trial_prediction=state.prediction + prediction_delta,
    )
    merged_state = apply_incremental_update(state, merged)
    residual = merged_state.prediction - problem.observations
    components = BlockObjectiveComponents(
        data=float(0.5 * np.dot(problem.observation_weights, residual * residual)),
        prior=prior.objective(merged_state.free_flow),
    )
    initial_residual = state.prediction - problem.observations
    initial_objective = float(
        0.5 * np.dot(problem.observation_weights, initial_residual * initial_residual)
        + prior.objective(state.free_flow)
    )
    tolerance = len(accepted) * (
        update_policy.absolute_objective_tolerance
        + update_policy.relative_objective_tolerance * abs(initial_objective)
    )
    if components.total > initial_objective + tolerance:
        raise RuntimeError("conflict-free batch merge increased the global objective.")
    return ConflictFreeBatchDecision(
        batch_id=merged.block_id,
        blocks=blocks,
        state=merged_state,
        block_decisions=decisions,
        components=components,
        merged_proposal=merged,
        accepted_blocks=len(accepted),
        rejected_blocks=len(decisions) - len(accepted),
        objective_improvement=initial_objective - components.total,
        maximum_flow_change=max(
            decision.maximum_flow_change for decision in accepted
        ),
        maximum_projected_gradient=max(
            decision.accepted_evaluation.projected_gradient_norm
            for decision in decisions
        ),
    )
