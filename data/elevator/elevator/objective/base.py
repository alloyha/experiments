"""
Objective function base class and RouteProjector.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

from elevator.domain.models import Route, ElevatorState, StopType


class ObjectiveFunction(ABC):
    """Interface for all objective functions.  Lower score = better route."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def evaluate(self, route: Route, state: ElevatorState) -> float:
        """Compute the objective value.  Lower = better."""


class RouteProjector:
    """Computes projected total times for every active passenger given a route.

    Total time = waiting time + flight time
      waiting time = t_embark - t_call
      flight time  = (floors travelled while aboard) * tau
    """

    def project(self, route: Route, state: ElevatorState) -> dict[str, float]:
        """Return passenger_id → projected total time under ``route``."""
        tau = state.tau
        t   = state.t_now
        pos = state.current_floor

        embark_time: dict[str, float] = {p.id: p.t_embark for p in state.onboard}  # type: ignore
        disembark_time: dict[str, float] = {}
        call_time: dict[str, float] = {p.id: p.t_call for p in state.all_active}

        for stop in route:
            t   += abs(stop.floor - pos) * tau
            pos  = stop.floor
            if stop.stop_type == StopType.EMBARK:
                embark_time[stop.passenger_id] = t
            else:
                disembark_time[stop.passenger_id] = t

        result: dict[str, float] = {}
        for p in state.all_active:
            t_emb = embark_time.get(p.id)
            t_dis = disembark_time.get(p.id)
            if t_emb is None or t_dis is None:
                result[p.id] = float("inf")
            else:
                result[p.id] = t_dis - call_time[p.id]
        return result
