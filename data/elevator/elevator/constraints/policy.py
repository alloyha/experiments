"""
Temporal and policy constraints: time windows, VIP priority,
maintenance floor, and peak-hour policy.
"""
from __future__ import annotations

from elevator.domain.models import Route, ElevatorState, StopType
from elevator.constraints.base import Constraint


class TimeWindowConstraint(Constraint):
    """A waiting passenger must be picked up before ``t_call + max_wait``.

    Per-passenger ``max_wait_time`` takes precedence over ``default_max_wait``.
    """

    def __init__(self, default_max_wait: float = float("inf")) -> None:
        self.default_max_wait = default_max_wait

    @property
    def name(self) -> str:
        return f"time_window_dw{self.default_max_wait:.0f}s"

    def _project_embark_times(
        self, route: Route, state: ElevatorState
    ) -> dict[str, float]:
        tau  = state.tau
        t    = state.t_now
        pos  = state.current_floor
        times: dict[str, float] = {}
        for stop in route:
            t   += abs(stop.floor - pos) * tau
            pos  = stop.floor
            if stop.stop_type == StopType.EMBARK:
                times[stop.passenger_id] = t
        return times

    def check(self, route: Route, state: ElevatorState) -> bool:
        embark_time = self._project_embark_times(route, state)
        for p in state.waiting:
            max_wait = p.max_wait_time if p.max_wait_time is not None else self.default_max_wait
            projected = embark_time.get(p.id)
            if projected is None or projected > p.t_call + max_wait:
                return False
        return True

    def penalty(self, route: Route, state: ElevatorState) -> float:
        embark_time  = self._project_embark_times(route, state)
        total_excess = 0.0
        for p in state.waiting:
            max_wait  = p.max_wait_time if p.max_wait_time is not None else self.default_max_wait
            projected = embark_time.get(p.id, float("inf"))
            total_excess += max(0.0, projected - (p.t_call + max_wait))
        return total_excess


class VIPConstraint(Constraint):
    """VIP passengers must be embarked before any non-VIP waiting passenger.

    Soft by default (``hard=False``); set ``hard=True`` to make it a
    hard feasibility constraint.
    """

    def __init__(self, vip_penalty: float = 1000.0, hard: bool = False) -> None:
        self.vip_penalty = vip_penalty
        self.hard        = hard

    @property
    def name(self) -> str:
        return "vip_priority"

    def _count_violations(self, route: Route, state: ElevatorState) -> int:
        vip_ids      = {p.id for p in state.waiting if p.is_vip}
        non_vip_ids  = {p.id for p in state.waiting if not p.is_vip}
        seen_non_vip = False
        violations   = 0
        for stop in route:
            if stop.stop_type != StopType.EMBARK:
                continue
            if stop.passenger_id in non_vip_ids:
                seen_non_vip = True
            elif stop.passenger_id in vip_ids and seen_non_vip:
                violations += 1
        return violations

    def check(self, route: Route, state: ElevatorState) -> bool:
        if not self.hard:
            return True
        return self._count_violations(route, state) == 0

    def penalty(self, route: Route, state: ElevatorState) -> float:
        return self._count_violations(route, state) * self.vip_penalty


class MaintenanceFloorConstraint(Constraint):
    """Soft constraint: penalises routes that skip the maintenance floor
    when the cumulative travel exceeds ``interval_floors``.
    """

    def __init__(
        self,
        maintenance_floor : int   = 1,
        interval_floors   : float = 100.0,
        penalty_per_floor : float = 1.0,
    ) -> None:
        self.maintenance_floor = maintenance_floor
        self.interval_floors   = interval_floors
        self.penalty_per_floor = penalty_per_floor

    @property
    def name(self) -> str:
        return f"maintenance_floor_{self.maintenance_floor}"

    def check(self, route: Route, state: ElevatorState) -> bool:
        return True  # soft constraint

    def penalty(self, route: Route, state: ElevatorState) -> float:
        cumulative: float  = getattr(state, "cumulative_travel", 0.0)
        floors_since_maint = cumulative % self.interval_floors
        pos     = state.current_floor
        visited = False
        for stop in route:
            floors_since_maint += abs(stop.floor - pos)
            pos = stop.floor
            if stop.floor == self.maintenance_floor:
                visited = True
                floors_since_maint = 0.0
        if floors_since_maint > self.interval_floors and not visited:
            return (floors_since_maint - self.interval_floors) * self.penalty_per_floor
        return 0.0


class PeakHourConstraint(Constraint):
    """Capacity and direction policy that varies by time of day.

    During ``peak_windows`` the constraint enforces ``peak_max_pax`` and
    applies ``direction_penalty`` per reversal.  Outside peak hours it
    enforces ``off_peak_max_pax`` with no direction penalty.
    """

    def __init__(
        self,
        peak_windows     : list[tuple[float, float]],
        peak_max_pax     : int   = 4,
        off_peak_max_pax : int   = 8,
        direction_penalty: float = 5.0,
    ) -> None:
        self.peak_windows      = peak_windows
        self.peak_max_pax      = peak_max_pax
        self.off_peak_max_pax  = off_peak_max_pax
        self.direction_penalty = direction_penalty

    @property
    def name(self) -> str:
        return "peak_hour_policy"

    def _is_peak(self, t: float) -> bool:
        return any(start <= t <= end for start, end in self.peak_windows)

    def check(self, route: Route, state: ElevatorState) -> bool:
        max_pax = self.peak_max_pax if self._is_peak(state.t_now) else self.off_peak_max_pax
        count   = state.passenger_count
        for stop in route:
            count += 1 if stop.stop_type == StopType.EMBARK else -1
            if count > max_pax:
                return False
        return True

    def penalty(self, route: Route, state: ElevatorState) -> float:
        if not self._is_peak(state.t_now):
            return 0.0
        floors    = [int(state.current_floor)] + [s.floor for s in route]
        reversals = 0
        for i in range(1, len(floors) - 1):
            d_prev = floors[i] - floors[i - 1]
            d_next = floors[i + 1] - floors[i]
            if d_prev != 0 and d_next != 0 and (d_prev > 0) != (d_next > 0):
                reversals += 1
        return reversals * self.direction_penalty
