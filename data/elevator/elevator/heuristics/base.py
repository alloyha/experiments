"""
Base heuristic abstraction and shared utilities.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

from elevator.domain.models import Route, Stop, StopType, ElevatorState


class Heuristic(ABC):
    """Abstract heuristic strategy.

    Each implementation generates one or more candidate routes.
    The pipeline selects the best using the objective function.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate(self, state: ElevatorState) -> list[Route]:
        """Return a non-empty list of candidate routes.

        All routes must satisfy: for each passenger, their EMBARK
        stop precedes their DISEMBARK stop.
        """


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def build_stop_pool(state: ElevatorState) -> tuple[list[Stop], list[Stop]]:
    """Return (embark_stops, disembark_stops) from the active state."""
    embark_stops = [
        Stop(floor=p.origin, stop_type=StopType.EMBARK, passenger_id=p.id)
        for p in state.waiting
    ]
    disembark_stops = [
        Stop(floor=p.destination, stop_type=StopType.DISEMBARK, passenger_id=p.id)
        for p in state.all_active
    ]
    return embark_stops, disembark_stops


def is_valid_route(route: Route) -> bool:
    """Check EMBARK-before-DISEMBARK invariant for all passengers."""
    seen: set[str] = set()
    for stop in route:
        if stop.stop_type == StopType.EMBARK:
            seen.add(stop.passenger_id)
        else:
            if stop.passenger_id not in seen:
                return False
    return True


def route_distance(route: Route, start_floor: float, tau: float) -> float:
    """Total travel time for a route (ignoring stop durations)."""
    pos   = start_floor
    total = 0.0
    for stop in route:
        total += abs(stop.floor - pos) * tau
        pos    = stop.floor
    return total
