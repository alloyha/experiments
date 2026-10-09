"""
Simulated annealing heuristic.
"""
from __future__ import annotations
import math
import random
from typing import Callable, Optional

from elevator.domain.models import Route, ElevatorState
from elevator.heuristics.base import Heuristic, is_valid_route, route_distance


class SimulatedAnnealingHeuristic(Heuristic):
    """Simulated Annealing over the stop-sequencing space.

    Starts from a NearestNeighbour solution and applies random
    swap/insert moves, accepting worse solutions with probability
    exp(-Δ/T).

    Parameters
    ----------
    initial_temp  : starting temperature
    cooling_rate  : multiplicative cooling factor per iteration
    max_iter      : total iterations per call to generate()
    seed          : optional RNG seed for reproducibility
    """

    def __init__(
        self,
        initial_temp : float = 100.0,
        cooling_rate : float = 0.995,
        max_iter     : int   = 2000,
        cost_fn      : Optional[Callable] = None,
        seed         : Optional[int]      = None,
    ) -> None:
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter     = max_iter
        self._cost_fn     = cost_fn or (lambda r, sf, tau: route_distance(r, sf, tau))
        self._rng         = random.Random(seed)

    @property
    def name(self) -> str:
        return f"simulated_annealing_i{self.max_iter}"

    def generate(self, state: ElevatorState) -> list[Route]:
        from elevator.heuristics.nearest_neighbour import NearestNeighbourHeuristic

        start_routes = NearestNeighbourHeuristic().generate(state)
        if not start_routes or not start_routes[0].stops:
            return start_routes

        current_stops = list(start_routes[0].stops)
        current_cost  = self._cost_fn(Route(current_stops), state.current_floor, state.tau)
        best_stops    = list(current_stops)
        best_cost     = current_cost
        T             = self.initial_temp

        for _ in range(self.max_iter):
            if len(current_stops) < 2:
                break  # pragma: no cover

            move      = self._rng.choice(("swap", "relocate"))
            candidate = list(current_stops)

            if move == "swap":
                i, j = self._rng.sample(range(len(candidate)), 2)
                candidate[i], candidate[j] = candidate[j], candidate[i]
            else:
                i    = self._rng.randrange(len(candidate))
                j    = self._rng.randrange(len(candidate))
                stop = candidate.pop(i)
                candidate.insert(j, stop)

            if not is_valid_route(Route(candidate)):
                T *= self.cooling_rate
                continue

            cand_cost = self._cost_fn(Route(candidate), state.current_floor, state.tau)
            delta     = cand_cost - current_cost

            if delta < 0 or (T > 1e-10 and self._rng.random() < math.exp(-delta / T)):
                current_stops = candidate
                current_cost  = cand_cost
                if current_cost < best_cost:  # pragma: no cover
                    best_stops = list(current_stops)  # pragma: no cover
                    best_cost  = current_cost  # pragma: no cover

            T *= self.cooling_rate

        return [Route(best_stops)]
