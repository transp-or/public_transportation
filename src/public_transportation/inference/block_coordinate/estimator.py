"""Sequential anytime block-coordinate MAP estimation."""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import cast

import numpy as np

from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
)
from public_transportation.inference.linear_operator import LinearOperatorProtocol

from .blocks import ODBlock
from .checkpoint import BlockCoordinateFingerprints
from .checkpoint_store import BlockCheckpointStore
from .config import BlockCoordinateMAPConfig
from .incremental import IncrementalLinearState
from .fixed_routing_selected_block_builder import (
    SelectedBlockConstructionDeadlineError,
)
from .objective import (
    SeparableQuadraticPrior,
    build_conditional_block_objective,
    prepare_separable_quadratic_prior,
    projected_gradient,
)
from .operator import BlockLinearOperatorProtocol, ColumnSelectedLinearOperator
from .partition import ODBlockPartition
from .progress import BlockProgressEvent, DiagnosticValue
from .results import (
    BlockConvergenceDiagnostics,
    BlockCoordinateMAPResult,
    BlockCoordinateState,
    BlockCoordinateStatus,
    BlockCoordinateWorkDiagnostics,
    BlockObjectiveComponents,
)
from .solver import (
    BlockSolverConfig,
    BlockUpdatePolicy,
    solve_and_decide_block_update,
)
from .scheduling import (
    ParallelBlockExecutionConfig,
    build_block_conflict_graph,
    color_block_conflict_graph,
    conflict_free_batch_id,
    solve_conflict_free_batch,
)

Array = np.ndarray
BlockOperatorFactory = Callable[[ODBlock], BlockLinearOperatorProtocol]
ProgressCallback = Callable[[BlockProgressEvent], None]
StopCallback = Callable[[BlockProgressEvent], bool]


def _global_components(
    problem: FixedRoutingLinearProblem,
    prior: SeparableQuadraticPrior,
    state: IncrementalLinearState,
) -> BlockObjectiveComponents:
    residual = state.prediction - problem.observations
    data = float(0.5 * np.dot(problem.observation_weights, residual * residual))
    return BlockObjectiveComponents(data=data, prior=prior.objective(state.free_flow))


def _random_state_json(generator: np.random.Generator) -> str:
    return json.dumps(
        generator.bit_generator.state,
        sort_keys=True,
        separators=(",", ":"),
    )


def _schedule(
    blocks: tuple[ODBlock, ...],
    *,
    order: str,
    generator: np.random.Generator,
) -> tuple[tuple[ODBlock, ...], tuple[int, ...]]:
    if order == "cyclic":
        return blocks, tuple(1 for _ in blocks)
    if order == "shuffled":
        permutation = generator.permutation(len(blocks))
        shuffled = tuple(blocks[int(index)] for index in permutation)
        return shuffled, tuple(1 for _ in shuffled)
    if order == "interleaved":
        batches = color_block_conflict_graph(
            build_block_conflict_graph(blocks)
        ).batches
        return (
            tuple(block for batch in batches for block in batch),
            tuple(len(batch) for batch in batches),
        )
    raise ValueError("unknown block order")


def _batch_size_by_position(batch_sizes: tuple[int, ...]) -> dict[int, int]:
    result: dict[int, int] = {}
    position = 0
    for size in batch_sizes:
        result[position] = size
        position += size
    return result


@dataclass(slots=True)
class BlockCoordinateMAPEstimator:
    """Interruptible block-coordinate estimator for a fixed-routing MAP problem."""

    problem: FixedRoutingLinearProblem
    partition: ODBlockPartition
    config: BlockCoordinateMAPConfig
    fingerprints: BlockCoordinateFingerprints
    block_operator_factory: BlockOperatorFactory | None = None
    progress_callback: ProgressCallback | None = None
    stop_callback: StopCallback | None = None
    logger: logging.Logger | None = None
    clock: Callable[[], float] = perf_counter
    absolute_deadline: float | None = None
    _selected_cache_hits: int = field(default=0, init=False)
    _selected_cache_misses: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.partition.num_free_variables != self.problem.num_free_od:
            raise ValueError("partition dimension does not match the linear problem.")
        if np.any(self.problem.lower_bounds < 0.0):
            raise ValueError(
                "block-coordinate OD estimation requires non-negative lower bounds."
            )
        if not self.partition.blocks:
            raise ValueError("the estimator requires at least one OD block.")
        if self.fingerprints.partition != self.partition.fingerprint:
            raise ValueError("partition fingerprint does not match the partition.")
        if self.fingerprints.solver_semantics != self.config.fingerprint:
            raise ValueError("solver-semantics fingerprint does not match the config.")
        if not callable(self.clock):
            raise TypeError("clock must be callable.")
        if self.absolute_deadline is not None and not math.isfinite(
            self.absolute_deadline
        ):
            raise ValueError("absolute_deadline must be finite when provided.")
        ParallelBlockExecutionConfig(
            construction_workers=self.config.construction_workers,
            solver_workers=self.config.solver_workers,
            threads_per_worker=self.config.threads_per_worker,
        )

    def _operator(self, block: ODBlock) -> BlockLinearOperatorProtocol:
        operator: BlockLinearOperatorProtocol
        if self.block_operator_factory is None:
            operator = cast(
                BlockLinearOperatorProtocol,
                ColumnSelectedLinearOperator(
                    cast(LinearOperatorProtocol, self.problem.measurement_operator),
                    block.free_column_indices,
                    measurement_support_indices=block.measurement_support_indices,
                ),
            )
        else:
            build_result = getattr(self.block_operator_factory, "build_result", None)
            if callable(build_result):
                if (
                    getattr(
                        self.block_operator_factory,
                        "supports_absolute_deadline",
                        False,
                    )
                    is True
                ):
                    constructed = build_result(
                        block, absolute_deadline=self.absolute_deadline
                    )
                else:
                    constructed = build_result(block)
                operator = constructed.operator
                if constructed.cache_hit:
                    self._selected_cache_hits += 1
                else:
                    self._selected_cache_misses += 1
            else:
                operator = self.block_operator_factory(block)
        if not isinstance(operator, BlockLinearOperatorProtocol):
            raise TypeError("block_operator_factory returned an invalid operator.")
        if operator.shape != (
            self.problem.num_measurements,
            block.num_free_variables,
        ):
            raise ValueError(
                f"operator for block {block.block_id!r} has shape {operator.shape}, "
                f"expected {(self.problem.num_measurements, block.num_free_variables)}."
            )
        return operator

    def _eligible_blocks(self) -> tuple[ODBlock, ...]:
        if self.config.pilot_block_schedule is None:
            return self.partition.blocks
        by_id = {block.block_id: block for block in self.partition.blocks}
        try:
            return tuple(by_id[block_id] for block_id in self.config.pilot_block_schedule)
        except KeyError as error:
            raise ValueError("pilot schedule contains an unknown block.") from error

    def run(
        self,
        initial_free_flow: object | None = None,
        *,
        initial_prediction: object | None = None,
        fixed_measurement_offset: object | None = None,
        initial_prediction_fingerprint: str | None = None,
        resume: bool = False,
    ) -> BlockCoordinateMAPResult:
        """Run until convergence, a configured budget, or graceful interruption."""
        prior = prepare_separable_quadratic_prior(self.problem)
        store = BlockCheckpointStore(self.config.checkpoint_directory, self.fingerprints)
        started = self.clock()
        forward_count = 0
        forward_seconds = 0.0
        transpose_count = 0
        transpose_seconds = 0.0
        construction_seconds = 0.0
        solve_seconds = 0.0
        checkpoint_seconds = 0.0
        deadline_overrun = False
        construction_attempts = 0
        constructions_completed = 0
        constructions_deadline_stopped = 0
        selected_deadline_phase: str | None = None
        selected_deadline_overshoot = 0.0
        solver_started = False
        checkpoint_preserved = False
        scheduled_not_solved: str | None = None
        policy = self.config.global_product_policy
        global_operator = cast(LinearOperatorProtocol, self.problem.measurement_operator)
        initial_prediction_source = "checkpoint" if resume else policy.initial_prediction_mode
        resume_validation_status = "not_applicable"
        final_validation_status = "deferred"
        self._selected_cache_hits = 0
        self._selected_cache_misses = 0

        def invocation_deadline_reached() -> bool:
            return (
                self.absolute_deadline is not None
                and self.clock() >= self.absolute_deadline
            ) or (
                self.config.maximum_elapsed_seconds is not None
                and max(0.0, self.clock() - started) >= self.config.maximum_elapsed_seconds
            )

        def global_forward(flow: Array) -> Array:
            nonlocal forward_count, forward_seconds
            phase_started = self.clock()
            value = np.asarray(global_operator.matvec(flow))
            forward_seconds += max(0.0, self.clock() - phase_started)
            forward_count += 1
            return value

        def exact_gradient_norm(state: IncrementalLinearState) -> float:
            nonlocal transpose_count, transpose_seconds
            residual = state.prediction - self.problem.observations
            phase_started = self.clock()
            gradient = global_operator.rmatvec(
                self.problem.observation_weights * residual
            ) + prior.gradient(state.free_flow)
            transpose_seconds += max(0.0, self.clock() - phase_started)
            transpose_count += 1
            projected = projected_gradient(
                state.free_flow,
                gradient,
                self.problem.lower_bounds,
                self.problem.upper_bounds,
            )
            return float(np.linalg.norm(projected))
        if resume:
            if (
                initial_free_flow is not None
                or initial_prediction is not None
                or fixed_measurement_offset is not None
            ):
                raise ValueError("initial state cannot be provided when resuming.")
            restored = store.load()
            incremental = IncrementalLinearState(
                restored.current_free_flow,
                restored.current_prediction,
                restored.fixed_measurement_offset,
            )
            if policy.resume_prediction_validation == "exact":
                recomputed = global_forward(incremental.free_flow) + self.problem.fixed_measurement_offset
                if not np.allclose(
                    recomputed, incremental.prediction, rtol=1.0e-10, atol=1.0e-10
                ):
                    raise ValueError("resumed prediction does not match the linear problem.")
                resume_validation_status = "exact"
            elif policy.resume_prediction_validation == "sampled":
                raise ValueError(
                    "sampled resume validation requires a restricted-product callback; "
                    "use exact or deferred validation."
                )
            else:
                resume_validation_status = "deferred"
            components = restored.current_components
            current_objective = restored.current_objective
            best_objective = restored.best_objective
            best_flow = np.array(restored.best_free_flow, copy=True)
            best_components = restored.best_components
            accepted_updates = restored.accepted_updates
            rejected_updates = restored.rejected_updates
            completed_sweeps = restored.sweep
            schedule_position = restored.schedule_position
            by_id = {block.block_id: block for block in self.partition.blocks}
            try:
                schedule = tuple(by_id[block_id] for block_id in restored.block_schedule)
            except KeyError as error:
                raise ValueError("checkpoint schedule contains an unknown block.") from error
            eligible = self._eligible_blocks()
            eligible_ids = {block.block_id for block in eligible}
            if (
                len(schedule) != len(eligible)
                or set(restored.block_schedule) != eligible_ids
            ):
                raise ValueError("checkpoint schedule does not cover the authorized blocks exactly.")
            generator = np.random.default_rng()
            try:
                generator.bit_generator.state = json.loads(restored.random_state_json)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError("checkpoint random state is invalid.") from error
            if self.config.block_order == "interleaved":
                expected_schedule, batch_sizes = _schedule(
                    eligible,
                    order=self.config.block_order,
                    generator=generator,
                )
                if tuple(block.block_id for block in expected_schedule) != tuple(
                    restored.block_schedule
                ):
                    raise ValueError("checkpoint conflict-free schedule is inconsistent.")
            else:
                batch_sizes = tuple(1 for _ in schedule)
            latest_block_gradient = restored.diagnostics.latest_block_projected_gradient
            sampled_gradient = restored.diagnostics.estimated_global_projected_gradient
            exact_gradient = restored.diagnostics.exact_global_projected_gradient
            last_exact_sweep = exact_gradient.computed_at_sweep
            maximum_flow_change = restored.diagnostics.maximum_block_flow_change
            initial_objective = (
                current_objective
                + restored.diagnostics.initialization_objective_improvement
            )
            current_sweep_improvement = (
                restored.diagnostics.current_sweep_objective_improvement
            )
            current_sweep_start_objective = (
                current_objective + current_sweep_improvement
            )
            previous_sweep_improvement = (
                restored.diagnostics.previous_sweep_objective_improvement
            )
            base_elapsed = restored.elapsed_seconds
            variables_visited = sum(
                block.num_free_variables for block in schedule[:schedule_position]
            )
            if schedule_position == len(schedule):
                previous_sweep_improvement = current_sweep_improvement
                current_sweep_start_objective = current_objective
                current_sweep_improvement = 0.0
                maximum_flow_change = 0.0
                schedule, batch_sizes = _schedule(
                    self._eligible_blocks(),
                    order=self.config.block_order,
                    generator=generator,
                )
                schedule_position = 0
                variables_visited = 0
        else:
            initial = (
                np.asarray(self.problem.prior_demand, dtype=float)
                if initial_free_flow is None
                else np.asarray(initial_free_flow, dtype=float)
            )
            if initial.shape != (self.problem.num_free_od,) or not np.all(
                np.isfinite(initial)
            ):
                raise ValueError(
                    "initial_free_flow must be finite with shape "
                    f"({self.problem.num_free_od},)."
                )
            if np.any(initial < self.problem.lower_bounds) or np.any(
                initial > self.problem.upper_bounds
            ):
                raise ValueError("initial_free_flow violates the problem bounds.")
            if policy.initial_prediction_mode == "provided":
                if initial_prediction is None:
                    raise ValueError("provided initial prediction mode requires initial_prediction.")
                if initial_prediction_fingerprint != self.fingerprints.fingerprint:
                    raise ValueError("initial prediction fingerprint is incompatible.")
                prediction = np.array(initial_prediction, dtype=float, copy=True)
                if prediction.shape != (self.problem.num_measurements,) or not np.all(
                    np.isfinite(prediction)
                ):
                    raise ValueError(
                        "initial_prediction must be finite with shape "
                        f"({self.problem.num_measurements},)."
                    )
                offset = np.array(
                    self.problem.fixed_measurement_offset
                    if fixed_measurement_offset is None
                    else fixed_measurement_offset,
                    dtype=float,
                    copy=True,
                )
                if (
                    offset.shape != (self.problem.num_measurements,)
                    or not np.all(np.isfinite(offset))
                    or not np.array_equal(offset, self.problem.fixed_measurement_offset)
                ):
                    raise ValueError("fixed_measurement_offset is incompatible with the problem.")
                if policy.initial_prediction_validation == "exact":
                    recomputed = global_forward(initial) + offset
                    if not np.allclose(recomputed, prediction, rtol=1.0e-10, atol=1.0e-10):
                        raise ValueError("supplied initial prediction is numerically inconsistent.")
                elif policy.initial_prediction_validation == "sampled":
                    raise ValueError(
                        "sampled initial validation requires a restricted-product callback."
                    )
                prediction.setflags(write=False)
                offset.setflags(write=False)
                incremental = IncrementalLinearState(
                    initial, prediction, offset
                )
            else:
                if (
                    initial_prediction is not None
                    or initial_prediction_fingerprint is not None
                    or fixed_measurement_offset is not None
                ):
                    raise ValueError(
                        "initial prediction may only be supplied in provided mode."
                    )
                incremental = IncrementalLinearState(
                    initial,
                    global_forward(initial) + self.problem.fixed_measurement_offset,
                    self.problem.fixed_measurement_offset,
                )
            components = _global_components(self.problem, prior, incremental)
            initial_objective = components.total
            current_objective = initial_objective
            best_objective = initial_objective
            best_flow = np.array(initial, copy=True)
            best_components = components
            generator = np.random.default_rng(self.config.random_seed)
            schedule, batch_sizes = _schedule(
                self._eligible_blocks(),
                order=self.config.block_order,
                generator=generator,
            )
            accepted_updates = 0
            rejected_updates = 0
            completed_sweeps = 0
            schedule_position = 0
            latest_block_gradient = DiagnosticValue(None, "unavailable")
            sampled_gradient = DiagnosticValue(None, "unavailable")
            if policy.initial_exact_gradient and not invocation_deadline_reached():
                exact_gradient = DiagnosticValue(
                    exact_gradient_norm(incremental), "exact", computed_at_sweep=0
                )
                last_exact_sweep = 0
            else:
                exact_gradient = DiagnosticValue(None, "deferred")
                last_exact_sweep = None
            maximum_flow_change = 0.0
            current_sweep_start_objective = current_objective
            previous_sweep_improvement = None
            current_sweep_improvement = 0.0
            variables_visited = 0
            base_elapsed = 0.0
        latest_improvement = 0.0
        sampled_block_norms: deque[float] = deque(
            maxlen=max(1, self.config.sampled_gradient_blocks)
        )
        # ETA uses only a bounded history of completed batch durations.  A
        # batch may contain several conflict-free blocks, so store its
        # duration per completed schedule position.
        unit_durations: deque[float] = deque(maxlen=32)
        if sampled_gradient.value is not None:
            sampled_block_norms.append(sampled_gradient.value)
        status: BlockCoordinateStatus | None = None
        message = ""
        last_progress_event: BlockProgressEvent | None = None
        last_compact_elapsed = base_elapsed
        batch_size_at_position = _batch_size_by_position(batch_sizes)

        def emit_progress(event: BlockProgressEvent) -> None:
            if self.progress_callback is None:
                return
            try:
                self.progress_callback(event)
            except Exception:
                # Progress is observability only.  Preserve control-flow
                # exceptions such as KeyboardInterrupt and never turn a
                # successful scientific update into a reporting failure.
                return

        def elapsed() -> float:
            return base_elapsed + max(0.0, float(self.clock() - started))

        def make_state() -> BlockCoordinateState:
            diagnostics = BlockConvergenceDiagnostics(
                latest_block_projected_gradient=latest_block_gradient,
                estimated_global_projected_gradient=sampled_gradient,
                exact_global_projected_gradient=exact_gradient,
                maximum_block_flow_change=maximum_flow_change,
                initialization_objective_improvement=(
                    initial_objective - current_objective
                ),
                current_sweep_objective_improvement=current_sweep_improvement,
                previous_sweep_objective_improvement=previous_sweep_improvement,
            )
            return BlockCoordinateState(
                current_free_flow=incremental.free_flow,
                best_free_flow=best_flow,
                current_prediction=incremental.prediction,
                fixed_measurement_offset=incremental.fixed_measurement_offset,
                current_objective=current_objective,
                best_objective=best_objective,
                current_components=components,
                best_components=best_components,
                sweep=completed_sweeps,
                schedule_position=schedule_position,
                accepted_updates=accepted_updates,
                rejected_updates=rejected_updates,
                elapsed_seconds=elapsed(),
                block_schedule=tuple(block.block_id for block in schedule),
                random_state_json=_random_state_json(generator),
                diagnostics=diagnostics,
                fingerprints=self.fingerprints,
            )

        if not resume:
            store.initialize(make_state())

        try:
            while status is None:
                if (
                    self.config.maximum_sweeps is not None
                    and completed_sweeps >= self.config.maximum_sweeps
                ):
                    status = "stopped_by_sweep_budget"
                    message = "Maximum sweep budget reached."
                    break
                if invocation_deadline_reached():
                    status = "stopped_by_time_budget"
                    message = "Maximum elapsed-time budget reached."
                    break
                attempts = accepted_updates + rejected_updates
                if self.config.maximum_block_updates is not None and attempts >= (
                    self.config.maximum_block_updates
                ):
                    status = "stopped_by_update_budget"
                    message = "Maximum block-update budget reached."
                    break

                planned_batch_size = batch_size_at_position[schedule_position]
                if self.config.maximum_block_updates is not None:
                    remaining_updates = self.config.maximum_block_updates - attempts
                    planned_batch_size = min(planned_batch_size, remaining_updates)
                batch = schedule[
                    schedule_position : schedule_position + planned_batch_size
                ]
                batch_started = self.clock()
                objective_before_update = current_objective
                solver_config = BlockSolverConfig(
                    maximum_iterations=self.config.block_solver_max_iterations,
                    tolerance=self.config.block_solver_tolerance,
                )
                update_policy = BlockUpdatePolicy(
                    update_damping=self.config.update_damping
                )
                if len(batch) == 1:
                    block = batch[0]
                    phase_started = self.clock()
                    construction_attempts += 1
                    try:
                        operator = self._operator(block)
                    except SelectedBlockConstructionDeadlineError as error:
                        construction_seconds += max(
                            0.0, self.clock() - phase_started
                        )
                        constructions_deadline_stopped += 1
                        selected_deadline_phase = error.diagnostics.phase
                        selected_deadline_overshoot = (
                            error.diagnostics.deadline_overshoot_seconds
                        )
                        deadline_overrun = (
                            error.diagnostics.indivisible_operation_overshoot
                        )
                        scheduled_not_solved = block.block_id
                        checkpoint_preserved = True
                        status = "stopped_by_time_budget"
                        message = (
                            "Selected-block construction reached the absolute deadline "
                            f"during {error.diagnostics.phase}; the pending block was "
                            "not sent to the solver."
                        )
                        break
                    construction_seconds += max(0.0, self.clock() - phase_started)
                    constructions_completed += 1
                    if invocation_deadline_reached():
                        deadline_overrun = True
                        scheduled_not_solved = block.block_id
                        checkpoint_preserved = True
                        status = "stopped_by_time_budget"
                        message = (
                            "Elapsed-time budget was reached during selected-block "
                            "construction; the solve was not started."
                        )
                        break
                    conditional = build_conditional_block_objective(
                        self.problem, prior, incremental, block, operator
                    )
                    phase_started = self.clock()
                    solver_started = True
                    decision = solve_and_decide_block_update(
                        incremental,
                        block,
                        operator,
                        conditional,
                        solver_config=solver_config,
                        update_policy=update_policy,
                    )
                    solve_seconds += max(0.0, self.clock() - phase_started)
                    accepted_in_batch = int(decision.accepted)
                    rejected_in_batch = int(not decision.accepted)
                    if decision.accepted:
                        incremental = decision.state
                        components = decision.accepted_evaluation.components
                    proposal = decision.proposal
                    latest_improvement = decision.objective_improvement
                    latest_flow_change = decision.maximum_flow_change
                    latest_norm = decision.accepted_evaluation.projected_gradient_norm
                    batch_label = block.block_id
                    fatal_reasons: tuple[str, ...] = (
                        (decision.reason,)
                        if decision.reason
                        in {"solver_failure", "nonfinite_candidate", "bound_violation"}
                        else ()
                    )
                else:
                    batch_decision = solve_conflict_free_batch(
                        problem=self.problem,
                        prior=prior,
                        state=incremental,
                        blocks=batch,
                        operator_factory=self._operator,
                        solver_config=solver_config,
                        update_policy=update_policy,
                        parallel_config=ParallelBlockExecutionConfig(
                            construction_workers=self.config.construction_workers,
                            solver_workers=self.config.solver_workers,
                            threads_per_worker=self.config.threads_per_worker,
                        ),
                    )
                    incremental = batch_decision.state
                    components = batch_decision.components
                    accepted_in_batch = batch_decision.accepted_blocks
                    rejected_in_batch = batch_decision.rejected_blocks
                    proposal = batch_decision.merged_proposal
                    latest_improvement = batch_decision.objective_improvement
                    latest_flow_change = batch_decision.maximum_flow_change
                    latest_norm = batch_decision.maximum_projected_gradient
                    batch_label = conflict_free_batch_id(batch)
                    fatal_reasons = tuple(
                        item.reason
                        for item in batch_decision.block_decisions
                        if item.reason
                        in {"solver_failure", "nonfinite_candidate", "bound_violation"}
                    )
                accepted_updates += accepted_in_batch
                rejected_updates += rejected_in_batch
                if accepted_in_batch:
                    current_objective = components.total
                    maximum_flow_change = max(
                        maximum_flow_change, latest_flow_change
                    )
                    best_solution_updated = current_objective < best_objective
                    if best_solution_updated:
                        best_objective = current_objective
                        best_flow = np.array(incremental.free_flow, copy=True)
                        best_components = components
                else:
                    best_solution_updated = False
                    latest_improvement = 0.0
                latest_block_gradient = DiagnosticValue(
                    latest_norm, "exact", computed_at_sweep=completed_sweeps + 1
                )
                if self.config.sampled_gradient_blocks > 0:
                    sampled_block_norms.append(latest_norm)
                    sampled_gradient = DiagnosticValue(
                        max(sampled_block_norms),
                        "sampled",
                        computed_at_sweep=completed_sweeps + 1,
                    )
                schedule_position += len(batch)
                variables_visited += sum(
                    block.num_free_variables for block in batch
                )
                current_sweep_improvement = (
                    current_sweep_start_objective - current_objective
                )
                completed_this_sweep = schedule_position == len(schedule)
                if completed_this_sweep:
                    completed_sweeps += 1
                    if (
                        self.config.exact_global_diagnostic_every_sweeps is not None
                        and completed_sweeps
                        % self.config.exact_global_diagnostic_every_sweeps
                        == 0
                    ):
                        exact_gradient = DiagnosticValue(
                            exact_gradient_norm(incremental),
                            "exact",
                            computed_at_sweep=completed_sweeps,
                        )
                        last_exact_sweep = completed_sweeps
                    elif exact_gradient.value is not None:
                        exact_gradient = DiagnosticValue(
                            exact_gradient.value,
                            "stale",
                            computed_at_sweep=exact_gradient.computed_at_sweep,
                        )

                checkpoint_committed = False
                state_after_update = make_state()
                if (
                    accepted_in_batch > 0
                    and self.config.save_after_every_block
                    and proposal is not None
                ):
                    phase_started = self.clock()
                    store.append_accepted_update(
                        proposal=proposal,
                        objective_before=objective_before_update,
                        state_after=state_after_update,
                        best_solution_updated=best_solution_updated,
                    )
                    checkpoint_seconds += max(0.0, self.clock() - phase_started)
                    checkpoint_committed = True
                should_compact = (
                    completed_this_sweep
                    or (
                        accepted_in_batch > 0
                        and accepted_updates
                        // self.config.compact_checkpoint_every_blocks
                        > (accepted_updates - accepted_in_batch)
                        // self.config.compact_checkpoint_every_blocks
                    )
                    or elapsed() - last_compact_elapsed
                    >= self.config.compact_checkpoint_every_seconds
                )
                if should_compact:
                    phase_started = self.clock()
                    store.compact(make_state())
                    checkpoint_seconds += max(0.0, self.clock() - phase_started)
                    last_compact_elapsed = elapsed()
                    checkpoint_committed = True

                elapsed_now = elapsed()
                batch_seconds = max(0.0, self.clock() - batch_started)
                if batch_seconds > 0.0 and batch:
                    unit_durations.extend(
                        [batch_seconds / len(batch)] * len(batch)
                    )
                from public_transportation.inference.construction_control import (
                    estimate_completed_unit_eta,
                )

                eta = estimate_completed_unit_eta(
                    unit_durations,
                    completed_units=schedule_position,
                    total_units=len(schedule),
                    parallelism=1,
                    elapsed_seconds=max(0.0, elapsed_now),
                )
                remaining = eta.predicted_remaining_seconds
                event = BlockProgressEvent(
                    sweep=completed_sweeps if completed_this_sweep else completed_sweeps + 1,
                    block_or_batch=batch_label,
                    blocks_completed_in_sweep=schedule_position,
                    total_blocks=len(schedule),
                    variables_visited=variables_visited,
                    total_variables=self.problem.num_free_od,
                    elapsed_seconds=elapsed_now,
                    current_objective=current_objective,
                    best_objective=best_objective,
                    data_objective=components.data,
                    prior_objective=components.prior,
                    latest_objective_improvement=latest_improvement,
                    latest_block_flow_change=latest_flow_change,
                    latest_block_projected_gradient=latest_norm,
                    estimated_global_projected_gradient=sampled_gradient,
                    exact_global_projected_gradient=exact_gradient,
                    last_exact_global_diagnostic_sweep=last_exact_sweep,
                    checkpoint_committed=checkpoint_committed,
                    estimated_remaining_sweep_seconds=remaining,
                    initial_prediction_source=initial_prediction_source,
                    global_forward_count=forward_count,
                    global_forward_seconds=forward_seconds,
                    global_transpose_count=transpose_count,
                    global_transpose_seconds=transpose_seconds,
                    selected_block_construction_seconds=construction_seconds,
                    selected_block_cache_hits=self._selected_cache_hits,
                    selected_block_cache_misses=self._selected_cache_misses,
                    block_solve_seconds=solve_seconds,
                    checkpoint_seconds=checkpoint_seconds,
                    phase_elapsed_seconds=elapsed_now,
                    completed_units=schedule_position,
                    total_units=len(schedule),
                    predicted_remaining_seconds=remaining,
                    eta_confidence=eta.eta_confidence,
                    eta_reason=eta.eta_reason,
                    estimated_completion_at_utc=eta.estimated_completion_at_utc,
                    work_stack=(
                        {
                            "name": "block_coordinate_schedule",
                            "completed_units": schedule_position,
                            "total_units": len(schedule),
                            "current_unit": batch_label,
                            "status": "running",
                        },
                    ),
                    active_units=(batch_label,),
                    queued_units=max(0, len(schedule) - schedule_position),
                    requested_workers=1,
                    completed_weight=float(schedule_position),
                    total_weight=float(len(schedule)),
                    weighted_fraction=(
                        schedule_position / max(len(schedule), 1)
                    ),
                    checkpoint_reusable=checkpoint_committed,
                    next_resumable_position=(
                        None
                        if schedule_position >= len(schedule)
                        else f"schedule-{schedule_position:06d}"
                    ),
                    job_elapsed_seconds=elapsed_now,
                    eta_lower_seconds=eta.eta_lower_seconds,
                    eta_upper_seconds=eta.eta_upper_seconds,
                    predicted_job_remaining_seconds=remaining,
                    job_eta_confidence=eta.eta_confidence,
                    job_eta_reason=eta.eta_reason,
                    estimated_job_completion_at_utc=eta.estimated_completion_at_utc,
                    deadline_remaining_seconds=(
                        None
                        if self.absolute_deadline is None
                        else max(0.0, self.absolute_deadline - self.clock())
                    ),
                    deadline_margin_seconds=(
                        None
                        if self.absolute_deadline is None or remaining is None
                        else max(0.0, self.absolute_deadline - self.clock()) - remaining
                    ),
                    will_finish_before_deadline=(
                        None
                        if self.absolute_deadline is None or remaining is None
                        else remaining <= max(0.0, self.absolute_deadline - self.clock())
                    ),
                )
                last_progress_event = event
                if self.logger is not None:
                    self.logger.info(event.to_json_line().rstrip())
                emit_progress(event)
                if self.stop_callback is not None and self.stop_callback(event):
                    status = "interrupted_with_approximate_solution"
                    message = "Stopped by the user callback after an atomic block or batch update."
                    break
                if fatal_reasons:
                    status = "numerical_failure"
                    message = (
                        f"Block batch {batch_label!r} failed safely: "
                        f"{', '.join(fatal_reasons)}."
                    )
                    break

                if completed_this_sweep:
                    convergence_reasons: list[str] = []
                    if (
                        self.config.global_projected_gradient_tolerance is not None
                        and exact_gradient.kind == "exact"
                        and exact_gradient.value is not None
                        and exact_gradient.value
                        <= self.config.global_projected_gradient_tolerance
                    ):
                        convergence_reasons.append("global projected gradient")
                    relative_improvement = current_sweep_improvement / max(
                        abs(current_sweep_start_objective), np.finfo(float).tiny
                    )
                    if (
                        self.config.relative_sweep_objective_tolerance is not None
                        and relative_improvement
                        <= self.config.relative_sweep_objective_tolerance
                    ):
                        convergence_reasons.append("relative sweep objective")
                    if (
                        self.config.maximum_flow_change_tolerance is not None
                        and maximum_flow_change
                        <= self.config.maximum_flow_change_tolerance
                    ):
                        convergence_reasons.append("maximum flow change")
                    if convergence_reasons:
                        status = "converged"
                        message = "Converged by " + ", ".join(convergence_reasons) + "."
                        break
                    if (
                        self.config.maximum_sweeps is not None
                        and completed_sweeps >= self.config.maximum_sweeps
                    ):
                        status = "stopped_by_sweep_budget"
                        message = "Maximum sweep budget reached."
                        break
                    previous_sweep_improvement = current_sweep_improvement
                    current_sweep_start_objective = current_objective
                    current_sweep_improvement = 0.0
                    maximum_flow_change = 0.0
                    schedule, batch_sizes = _schedule(
                        self._eligible_blocks(),
                        order=self.config.block_order,
                        generator=generator,
                    )
                    schedule_position = 0
                    batch_size_at_position = _batch_size_by_position(batch_sizes)
                    variables_visited = 0
                    sampled_block_norms.clear()
        except SelectedBlockConstructionDeadlineError as error:
            constructions_deadline_stopped += 1
            selected_deadline_phase = error.diagnostics.phase
            selected_deadline_overshoot = error.diagnostics.deadline_overshoot_seconds
            deadline_overrun = error.diagnostics.indivisible_operation_overshoot
            scheduled_not_solved = error.diagnostics.block_id
            checkpoint_preserved = True
            status = "stopped_by_time_budget"
            message = (
                "Selected-block construction reached the absolute deadline during "
                f"{error.diagnostics.phase}; the pending batch was not committed."
            )
        except KeyboardInterrupt:
            status = "interrupted_with_approximate_solution"
            message = "Interrupted after preserving the latest atomic block update."
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            status = "numerical_failure"
            message = f"Numerical failure: {error}"

        if status is None:  # pragma: no cover - defensive exhaustiveness
            raise RuntimeError("estimator stopped without a result status")
        if last_progress_event is not None:
            terminal_remaining = last_progress_event.predicted_remaining_seconds
            terminal_current = last_progress_event.block_or_batch
            if status == "converged":
                terminal_remaining = 0.0
                terminal_current = ""
            terminal_stack = tuple(
                {
                    **stack,
                    "status": str(status),
                    "current_unit": (
                        None if status == "converged" else stack.get("current_unit")
                    ),
                }
                for stack in last_progress_event.work_stack
            )
            terminal_event = replace(
                last_progress_event,
                status=str(status),
                block_or_batch=terminal_current or last_progress_event.block_or_batch,
                estimated_remaining_sweep_seconds=terminal_remaining,
                predicted_remaining_seconds=terminal_remaining,
                eta_confidence=(
                    "high" if status == "converged" else last_progress_event.eta_confidence
                ),
                eta_reason=(
                    "all units completed"
                    if status == "converged"
                    else last_progress_event.eta_reason
                ),
                eta_lower_seconds=(
                    0.0 if status == "converged" else last_progress_event.eta_lower_seconds
                ),
                eta_upper_seconds=(
                    0.0 if status == "converged" else last_progress_event.eta_upper_seconds
                ),
                predicted_job_remaining_seconds=terminal_remaining,
                job_eta_confidence=(
                    "high" if status == "converged" else last_progress_event.job_eta_confidence
                ),
                job_eta_reason=(
                    "all units completed"
                    if status == "converged"
                    else last_progress_event.job_eta_reason
                ),
                work_stack=terminal_stack,
                active_units=() if status == "converged" else last_progress_event.active_units,
            )
            if self.logger is not None:
                self.logger.info(terminal_event.to_json_line().rstrip())
            emit_progress(terminal_event)
        if policy.final_exact_gradient and not invocation_deadline_reached():
            exact_gradient = DiagnosticValue(
                exact_gradient_norm(incremental),
                "exact",
                computed_at_sweep=completed_sweeps,
            )
        if policy.final_prediction_validation == "exact" and not invocation_deadline_reached():
            recomputed = global_forward(incremental.free_flow) + incremental.fixed_measurement_offset
            if not np.allclose(recomputed, incremental.prediction, rtol=1.0e-10, atol=1.0e-10):
                raise ValueError("final prediction does not match the linear problem.")
            final_validation_status = "exact"
        elif policy.final_prediction_validation == "sampled":
            final_validation_status = "unavailable"
        elif policy.final_prediction_validation == "exact":
            final_validation_status = "deferred"
        final_state = make_state()
        if not checkpoint_preserved:
            phase_started = self.clock()
            store.compact(final_state)
            checkpoint_seconds += max(0.0, self.clock() - phase_started)
        return BlockCoordinateMAPResult(
            status=status,
            message=message,
            state=final_state,
            checkpoint_directory=self.config.checkpoint_directory,
            resume_configuration_fingerprint=self.config.fingerprint,
            work=BlockCoordinateWorkDiagnostics(
                initial_prediction_source=initial_prediction_source,
                global_forward_count=forward_count,
                global_forward_seconds=forward_seconds,
                global_transpose_count=transpose_count,
                global_transpose_seconds=transpose_seconds,
                selected_block_construction_seconds=construction_seconds,
                selected_block_cache_hits=self._selected_cache_hits,
                selected_block_cache_misses=self._selected_cache_misses,
                selected_block_construction_attempts=construction_attempts,
                selected_block_constructions_completed=constructions_completed,
                selected_block_constructions_deadline_stopped=(
                    constructions_deadline_stopped
                ),
                block_solve_seconds=solve_seconds,
                checkpoint_seconds=checkpoint_seconds,
                resume_prediction_validation=resume_validation_status,
                final_prediction_validation=final_validation_status,
                deadline_exceeded_by_indivisible_operation=deadline_overrun,
                selected_block_deadline_phase=selected_deadline_phase,
                selected_block_deadline_overshoot_seconds=(
                    selected_deadline_overshoot
                ),
                solver_started=solver_started,
                checkpoint_preserved=checkpoint_preserved,
                scheduled_block_not_attempted_by_solver=scheduled_not_solved,
            ),
        )


def run_block_coordinate_map(
    *,
    problem: FixedRoutingLinearProblem,
    partition: ODBlockPartition,
    config: BlockCoordinateMAPConfig,
    fingerprints: BlockCoordinateFingerprints,
    initial_free_flow: object | None = None,
    initial_prediction: object | None = None,
    fixed_measurement_offset: object | None = None,
    initial_prediction_fingerprint: str | None = None,
    block_operator_factory: BlockOperatorFactory | None = None,
    progress_callback: ProgressCallback | None = None,
    stop_callback: StopCallback | None = None,
    logger: logging.Logger | None = None,
    resume: bool = False,
    absolute_deadline: float | None = None,
) -> BlockCoordinateMAPResult:
    """Convenience entry point for the anytime block-coordinate estimator."""
    estimator = BlockCoordinateMAPEstimator(
        problem=problem,
        partition=partition,
        config=config,
        fingerprints=fingerprints,
        block_operator_factory=block_operator_factory,
        progress_callback=progress_callback,
        stop_callback=stop_callback,
        logger=logger,
        absolute_deadline=absolute_deadline,
    )
    return estimator.run(
        initial_free_flow,
        initial_prediction=initial_prediction,
        fixed_measurement_offset=fixed_measurement_offset,
        initial_prediction_fingerprint=initial_prediction_fingerprint,
        resume=resume,
    )


def resume_block_coordinate_map(
    *,
    problem: FixedRoutingLinearProblem,
    partition: ODBlockPartition,
    config: BlockCoordinateMAPConfig,
    fingerprints: BlockCoordinateFingerprints,
    block_operator_factory: BlockOperatorFactory | None = None,
    progress_callback: ProgressCallback | None = None,
    stop_callback: StopCallback | None = None,
    logger: logging.Logger | None = None,
    absolute_deadline: float | None = None,
) -> BlockCoordinateMAPResult:
    """Resume a fingerprint-compatible run from its durable checkpoint."""
    return run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=config,
        fingerprints=fingerprints,
        block_operator_factory=block_operator_factory,
        progress_callback=progress_callback,
        stop_callback=stop_callback,
        logger=logger,
        absolute_deadline=absolute_deadline,
        resume=True,
    )
