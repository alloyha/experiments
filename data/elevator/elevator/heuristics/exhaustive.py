"""
Branch-and-bound exhaustive heuristic.
"""
from __future__ import annotations
from typing import Callable, Optional

from elevator.domain.models import Route, Stop, StopType, ElevatorState
from elevator.heuristics.base import (
    Heuristic, build_stop_pool, is_valid_route, route_distance,
)


def _partial_travel_cost(
    stops: list[Stop],
    pos  : float,
    state: ElevatorState,
) -> float:
    total = 0.0
    for s in stops:
        total += abs(s.floor - pos) * state.tau
        pos    = s.floor
    return total


class ExhaustiveHeuristic(Heuristic):
    """Branch-and-bound exhaustive search over all valid stop sequences.

    Prunes branches whose partial travel cost already exceeds the
    best complete route found so far (lower-bound pruning).  Falls
    back to NearestNeighbour when the stop count exceeds ``max_stops``.
    """

    def __init__(
        self,
        max_stops: int = 10,
        cost_fn  : Optional[Callable[[list[Stop], float, ElevatorState], float]] = None,
    ) -> None:
        self.max_stops = max_stops
        self._cost_fn  = cost_fn or _partial_travel_cost

    @property
    def name(self) -> str:
        return "exhaustive_bnb"

    def generate(self, state: ElevatorState) -> list[Route]:
        from elevator.heuristics.nearest_neighbour import NearestNeighbourHeuristic

        embark, disembark = build_stop_pool(state)
        onboard_ids   = {p.id for p in state.onboard}
        disembark_map = {s.passenger_id: s for s in disembark}

        initial: list[Stop] = list(embark)
        for pid in onboard_ids:
            if pid in disembark_map:
                initial.append(disembark_map[pid])

        if len(initial) + len(disembark) > self.max_stops:
            return NearestNeighbourHeuristic().generate(state)

        best_cost: list[float]       = [float("inf")]
        best_routes: list[list[Stop]] = []

        def _bnb(
            partial    : list[Stop],
            available  : list[Stop],
            embarked   : set[str],
            pos        : float,
            cost_so_far: float,
        ) -> None:
            if not available:
                route = Route(list(partial))
                if is_valid_route(route):
                    if cost_so_far < best_cost[0]:
                        best_cost[0] = cost_so_far
                        best_routes.clear()
                        best_routes.append(partial[:])
                    elif cost_so_far == best_cost[0]:  # pragma: no cover
                        best_routes.append(partial[:])  # pragma: no cover
                return
            for stop in available:
                step_cost = abs(stop.floor - pos) * state.tau
                new_cost  = cost_so_far + step_cost
                if new_cost >= best_cost[0]:
                    continue
                new_available = [s for s in available if s is not stop]
                new_embarked  = set(embarked)
                if stop.stop_type == StopType.EMBARK:
                    new_embarked.add(stop.passenger_id)
                    if stop.passenger_id in disembark_map:
                        new_available.append(disembark_map[stop.passenger_id])
                else:
                    if stop.passenger_id not in embarked:
                        continue  # pragma: no cover
                partial.append(stop)
                _bnb(partial, new_available, new_embarked, stop.floor, new_cost)
                partial.pop()

        _bnb([], initial, set(onboard_ids), state.current_floor, 0.0)
        return [Route(r) for r in best_routes] or NearestNeighbourHeuristic().generate(state)
