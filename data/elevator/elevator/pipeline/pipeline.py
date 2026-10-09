"""
OptimizationPipeline — the central orchestrator.

Follows Dependency Inversion: depends only on abstractions
(Heuristic, Constraint, ObjectiveFunction), never on concretions.

Usage
-----
pipeline = (
    OptimizationPipeline()
    .add_heuristic(BeamSearchHeuristic(beam_width=15))
    .add_heuristic(NearestNeighbourHeuristic())
    .set_objective(IQRObjective())
    .set_constraints(registry)
)

result = pipeline.optimize(state)
print(result.best_route, result.best_score)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time

from elevator.domain.models import Route, ElevatorState, Stop, StopType
from elevator.constraints.base import Constraint, ConstraintRegistry
from elevator.objective.objective import ObjectiveFunction, IQRObjective
from elevator.heuristics.heuristics import (
    Heuristic,
    NearestNeighbourHeuristic,
    BeamSearchHeuristic,
)


# ---------------------------------------------------------------------------
# Safety: ensure every active passenger is in the route
# ---------------------------------------------------------------------------

def _ensure_all_passengers_covered(route: Route, state: ElevatorState) -> Route:
    """Guarantee that no active passenger is silently omitted from a route.

    Waiting passengers must have an EMBARK stop followed by a DISEMBARK.
    Onboard passengers (already embarked) must have a DISEMBARK stop.

    Any missing stops are appended at the end using cheapest-insertion so
    the route remains executable even when a heuristic returns an
    incomplete result.
    """
    stops = list(route.stops)
    embarked_in_route = {
        s.passenger_id for s in stops if s.stop_type == StopType.EMBARK
    }
    disembark_in_route = {
        s.passenger_id for s in stops if s.stop_type == StopType.DISEMBARK
    }

    for p in state.waiting:
        if p.id not in embarked_in_route:
            stops.append(Stop(floor=p.origin, stop_type=StopType.EMBARK, passenger_id=p.id))
            embarked_in_route.add(p.id)
        if p.id not in disembark_in_route:
            stops.append(Stop(floor=p.destination, stop_type=StopType.DISEMBARK, passenger_id=p.id))
            disembark_in_route.add(p.id)

    for p in state.onboard:
        if p.id not in disembark_in_route:
            stops.append(Stop(floor=p.destination, stop_type=StopType.DISEMBARK, passenger_id=p.id))
            disembark_in_route.add(p.id)

    if stops == list(route.stops):
        return route
    return Route(stops)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    best_route       : Route
    best_score       : float
    evaluated_routes : int
    feasible_routes  : int
    elapsed_ms       : float
    heuristic_used   : str
    violations       : list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        floors = [s.floor for s in self.best_route.stops]
        return (
            f"OptimizationResult("
            f"score={self.best_score:.3f}, "
            f"route={floors}, "
            f"evaluated={self.evaluated_routes}, "
            f"feasible={self.feasible_routes}, "
            f"elapsed={self.elapsed_ms:.1f}ms, "
            f"heuristic={self.heuristic_used})"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class OptimizationPipeline:
    """Assembles heuristics, constraints, and an objective function into
    a single optimise() call.

    Design notes
    ------------
    - Open/Closed: add heuristics or constraints without changing this class.
    - Single Responsibility: this class only orchestrates; it does not
      implement any optimisation logic itself.
    - Dependency Inversion: all collaborators are injected as abstractions.
    """

    def __init__(self) -> None:
        self._heuristics  : list[Heuristic]      = []
        self._constraints : ConstraintRegistry   = ConstraintRegistry()
        self._objective   : ObjectiveFunction    = IQRObjective()
        self._fallback    : Optional[Heuristic]  = NearestNeighbourHeuristic()

    # -- builder methods (fluent API) ---------------------------------------

    def add_heuristic(self, heuristic: Heuristic) -> "OptimizationPipeline":
        self._heuristics.append(heuristic)
        return self

    def set_objective(self, objective: ObjectiveFunction) -> "OptimizationPipeline":
        self._objective = objective
        return self

    def set_constraints(self, registry: ConstraintRegistry) -> "OptimizationPipeline":
        self._constraints = registry
        return self

    def add_constraint(self, constraint: Constraint) -> "OptimizationPipeline":
        self._constraints.add(constraint)
        return self

    def set_fallback(self, heuristic: Optional[Heuristic]) -> "OptimizationPipeline":
        """Fallback heuristic used when no feasible route is found."""
        self._fallback = heuristic
        return self

    # -- main entry point ---------------------------------------------------

    def optimize(self, state: ElevatorState) -> OptimizationResult:
        """Find the best feasible route for the given elevator state."""
        t0 = time.perf_counter()

        if not state.all_active:
            return OptimizationResult(
                best_route       = Route([]),
                best_score       = 0.0,
                evaluated_routes = 0,
                feasible_routes  = 0,
                elapsed_ms       = 0.0,
                heuristic_used   = "none",
            )

        heuristics = self._heuristics or [NearestNeighbourHeuristic()]

        best_route  : Optional[Route] = None
        best_score  = float("inf")
        best_hname  = "none"
        total_eval  = 0
        total_feas  = 0

        for heuristic in heuristics:
            candidates = heuristic.generate(state)
            total_eval += len(candidates)

            for route in candidates:
                if not self._constraints.check_all(route, state):
                    continue
                total_feas += 1
                score = self._objective.evaluate(route, state)
                if score < best_score:
                    best_score  = score
                    best_route  = route
                    best_hname  = heuristic.name

        # Fallback: if no feasible route found, use fallback heuristic
        # ignoring hard constraints (better than returning nothing)
        if best_route is None and self._fallback is not None:
            fb_candidates = self._fallback.generate(state)
            total_eval   += len(fb_candidates)
            for route in fb_candidates:
                score = self._objective.evaluate(route, state)
                if score < best_score:
                    best_score = score
                    best_route = route
                    best_hname = f"{self._fallback.name}[fallback]"

        if best_route is None:
            best_route = Route([])

        best_route = _ensure_all_passengers_covered(best_route, state)
        violations = self._constraints.violations(best_route, state)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return OptimizationResult(
            best_route       = best_route,
            best_score       = best_score,
            evaluated_routes = total_eval,
            feasible_routes  = total_feas,
            elapsed_ms       = elapsed_ms,
            heuristic_used   = best_hname,
            violations       = violations,
        )
