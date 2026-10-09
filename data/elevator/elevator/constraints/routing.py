"""
Routing constraints: floor range, no-reverse, minimum stop gap.
"""
from __future__ import annotations

from elevator.domain.models import Route, ElevatorState, StopType
from elevator.constraints.base import Constraint


class FloorRangeConstraint(Constraint):
    """Restricts the elevator to a contiguous range of floors."""

    def __init__(self, min_floor: int, max_floor: int) -> None:
        self.min_floor = min_floor
        self.max_floor = max_floor

    @property
    def name(self) -> str:
        return f"floor_range_{self.min_floor}-{self.max_floor}"

    def check(self, route: Route, state: ElevatorState) -> bool:
        return all(
            self.min_floor <= stop.floor <= self.max_floor
            for stop in route
        )


class NoReverseConstraint(Constraint):
    """Soft constraint that penalises each direction reversal in the route."""

    def __init__(self, reversal_cost: float = 1.0) -> None:
        self.reversal_cost = reversal_cost

    @property
    def name(self) -> str:
        return "no_reverse"

    def check(self, route: Route, state: ElevatorState) -> bool:
        return True  # soft constraint — always feasible

    def penalty(self, route: Route, state: ElevatorState) -> float:
        floors    = [int(state.current_floor)] + [s.floor for s in route]
        reversals = 0
        for i in range(1, len(floors) - 1):
            d_prev = floors[i] - floors[i - 1]
            d_next = floors[i + 1] - floors[i]
            if d_prev != 0 and d_next != 0 and (d_prev > 0) != (d_next > 0):
                reversals += 1
        return reversals * self.reversal_cost


class MinStopGapConstraint(Constraint):
    """Consecutive stops must be at least ``min_gap`` floors apart."""

    def __init__(self, min_gap: int = 1) -> None:
        self.min_gap = min_gap

    @property
    def name(self) -> str:
        return f"min_stop_gap_{self.min_gap}"

    def check(self, route: Route, state: ElevatorState) -> bool:
        floors = [s.floor for s in route]
        return all(
            abs(floors[i] - floors[i - 1]) >= self.min_gap
            for i in range(1, len(floors))
        )
