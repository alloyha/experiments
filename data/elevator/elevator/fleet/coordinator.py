"""
FleetCoordinator — multi-elevator dispatch and coordination.

Responsibilities
----------------
1. Assign each incoming CALL to the best elevator (not just the nearest).
2. Maintain shared ``building_stop_counts`` so ShaftsPerFloorConstraint
   can enforce per-floor concurrency limits.
3. Prevent two elevators from being routed to the same floor at the same
   time (soft collision avoidance).
4. Optimise global IQR across all passengers of all elevators jointly by
   choosing the assignment that minimises the projected composite score.

Usage
-----
    coordinator = FleetCoordinator(
        elevators  = [state_e1, state_e2],
        pipelines  = {state_e1.id: pipeline1, state_e2.id: pipeline2},
    )
    assignments = coordinator.assign_call(passenger, t_now)
    # assignments: dict[elevator_id → OptimizationResult]
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

from elevator.domain.models import ElevatorState, Passenger, Route
from elevator.pipeline.pipeline import OptimizationPipeline, OptimizationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Assignment result
# ---------------------------------------------------------------------------

@dataclass
class AssignmentResult:
    """Result of a fleet-level call assignment decision.

    Attributes
    ----------
    chosen_elevator_id : id of the elevator assigned to serve the call
    optimization_results : per-elevator OptimizationResult after re-routing
    global_score       : combined objective score across all elevators
    """
    chosen_elevator_id   : str
    optimization_results : dict[str, OptimizationResult]
    global_score         : float


# ---------------------------------------------------------------------------
# FleetCoordinator
# ---------------------------------------------------------------------------

class FleetCoordinator:
    """Coordinates route planning across a fleet of elevators.

    Parameters
    ----------
    elevators  : list of ElevatorState objects, one per elevator
    pipelines  : mapping elevator_id → OptimizationPipeline
    """

    def __init__(
        self,
        elevators : list[ElevatorState],
        pipelines : dict[str, OptimizationPipeline],
    ) -> None:
        self._elevators = {s.id: s for s in elevators}
        self._pipelines = pipelines

    # -- public API ---------------------------------------------------------

    def assign_call(self, passenger: Passenger, t_now: float) -> AssignmentResult:
        """Assign a newly arrived passenger to the best elevator.

        For each candidate elevator the coordinator:
        1. Temporarily adds the passenger to that elevator's waiting list.
        2. Runs the pipeline to get a candidate route and score.
        3. Reverts the temporary state.
        4. Picks the elevator with the lowest projected score.

        After the best elevator is chosen the passenger is formally added
        to its waiting list and the route is committed.  All elevators'
        ``building_stop_counts`` are refreshed.
        """
        if not self._elevators:
            raise RuntimeError("FleetCoordinator has no elevators.")

        best_id    : Optional[str]               = None
        best_score : float                       = float("inf")
        results    : dict[str, OptimizationResult] = {}

        # Respect passenger's preferred elevator if set
        candidate_ids = (
            [passenger.elevator_id]
            if passenger.elevator_id and passenger.elevator_id in self._elevators
            else list(self._elevators.keys())
        )

        for eid in candidate_ids:
            state    = self._elevators[eid]
            pipeline = self._pipelines.get(eid)
            if pipeline is None:
                logger.warning("No pipeline for elevator %s — skipping", eid)
                continue

            # Temporarily add passenger
            state.waiting.append(passenger)
            state.t_now = t_now
            self._refresh_stop_counts()

            result = pipeline.optimize(state)
            results[eid] = result

            # Score = this elevator's best_score (lower = better assignment)
            if result.best_score < best_score:
                best_score = result.best_score
                best_id    = eid

            # Revert temporary addition
            state.waiting = [p for p in state.waiting if p.id != passenger.id]

        if best_id is None:
            best_id = candidate_ids[0]
            logger.warning("No feasible assignment found — defaulting to %s", best_id)

        # Formally assign
        chosen_state = self._elevators[best_id]
        chosen_state.waiting.append(passenger)
        self._refresh_stop_counts()
        pipeline = self._pipelines.get(best_id)
        if pipeline is not None:
            final_result = pipeline.optimize(chosen_state)
            results[best_id] = final_result
            chosen_state.current_route = final_result.best_route

        logger.info(
            "[t=%.2f] Assigned passenger %s to elevator %s (score=%.3f)",
            t_now, passenger.id, best_id, best_score,
        )
        return AssignmentResult(
            chosen_elevator_id   = best_id,
            optimization_results = results,
            global_score         = best_score,
        )

    def reoptimize_all(self, t_now: float) -> dict[str, OptimizationResult]:
        """Reoptimize all elevators and refresh shared stop counts.

        Call this after any EMBARK or DISEMBARK event to keep all routes
        consistent with the current global state.
        """
        self._refresh_stop_counts()
        results: dict[str, OptimizationResult] = {}
        for eid, state in self._elevators.items():
            state.t_now = t_now
            pipeline    = self._pipelines.get(eid)
            if pipeline is None:
                continue
            result = pipeline.optimize(state)
            state.current_route = result.best_route
            results[eid]        = result
        self._refresh_stop_counts()
        return results

    # -- internal helpers ---------------------------------------------------

    def _refresh_stop_counts(self) -> None:
        """Recompute ``building_stop_counts`` for every elevator.

        Each elevator's count map reflects the number of stops the *other*
        elevators have at each floor (not counting its own route).
        """
        # Build aggregate: floor → total stops across all elevators
        all_counts: dict[int, int] = {}
        for state in self._elevators.values():
            for stop in state.current_route.stops:
                all_counts[stop.floor] = all_counts.get(stop.floor, 0) + 1

        # Each elevator sees counts from all others
        for state in self._elevators.values():
            own_counts: dict[int, int] = {}
            for stop in state.current_route.stops:
                own_counts[stop.floor] = own_counts.get(stop.floor, 0) + 1
            state.building_stop_counts = {
                floor: total - own_counts.get(floor, 0)
                for floor, total in all_counts.items()
            }
