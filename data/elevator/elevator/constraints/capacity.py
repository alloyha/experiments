"""
Capacity constraints: weight, passenger count, and shafts-per-floor.
"""
from __future__ import annotations

from elevator.domain.models import Route, ElevatorState, StopType
from elevator.constraints.base import Constraint


class WeightCapacityConstraint(Constraint):
    """Ensures total passenger weight never exceeds ``max_kg``."""

    def __init__(self, max_kg: float) -> None:
        self.max_kg = max_kg

    @property
    def name(self) -> str:
        return f"weight_capacity_{self.max_kg}kg"

    def check(self, route: Route, state: ElevatorState) -> bool:
        weight_map     = {p.id: p.weight_kg for p in state.all_active}
        current_weight = state.total_weight
        for stop in route:
            if stop.stop_type == StopType.EMBARK:
                current_weight += weight_map.get(stop.passenger_id, 0.0)
            else:
                current_weight -= weight_map.get(stop.passenger_id, 0.0)
            if current_weight > self.max_kg:
                return False
        return True

    def penalty(self, route: Route, state: ElevatorState) -> float:
        weight_map     = {p.id: p.weight_kg for p in state.all_active}
        current_weight = state.total_weight
        max_excess     = 0.0
        for stop in route:
            if stop.stop_type == StopType.EMBARK:
                current_weight += weight_map.get(stop.passenger_id, 0.0)
            else:
                current_weight -= weight_map.get(stop.passenger_id, 0.0)
            excess = current_weight - self.max_kg
            if excess > max_excess:
                max_excess = excess
        return max_excess


class PassengerCapacityConstraint(Constraint):
    """Ensures the number of simultaneous passengers never exceeds ``max_pax``."""

    def __init__(self, max_pax: int) -> None:
        self.max_pax = max_pax

    @property
    def name(self) -> str:
        return f"passenger_capacity_{self.max_pax}pax"

    def check(self, route: Route, state: ElevatorState) -> bool:
        count = state.passenger_count
        for stop in route:
            count += 1 if stop.stop_type == StopType.EMBARK else -1
            if count > self.max_pax:
                return False
        return True

    def penalty(self, route: Route, state: ElevatorState) -> float:
        count     = state.passenger_count
        max_excess = 0
        for stop in route:
            count += 1 if stop.stop_type == StopType.EMBARK else -1
            excess = count - self.max_pax
            if excess > max_excess:
                max_excess = excess
        return float(max_excess)


class ShaftsPerFloorConstraint(Constraint):
    """Limits how many elevators can stop at the same floor simultaneously.

    Requires ``state.building_stop_counts: dict[int, int]``.
    """

    def __init__(self, max_shafts: int) -> None:
        self.max_shafts = max_shafts

    @property
    def name(self) -> str:
        return f"shafts_per_floor_{self.max_shafts}"

    def check(self, route: Route, state: ElevatorState) -> bool:
        existing: dict[int, int] = getattr(state, "building_stop_counts", {})
        planned: dict[int, int]  = {}
        for stop in route:
            planned[stop.floor] = planned.get(stop.floor, 0) + 1
        for floor, count in planned.items():
            if existing.get(floor, 0) + count > self.max_shafts:
                return False
        return True
