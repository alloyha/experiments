"""
Mean-time and composite objective functions.
"""
from __future__ import annotations
import statistics

from elevator.domain.models import Route, ElevatorState
from elevator.objective.base import ObjectiveFunction, RouteProjector


class MeanTimeObjective(ObjectiveFunction):
    """Minimize the mean total time across active passengers."""

    def __init__(self, projector: RouteProjector | None = None) -> None:
        self._projector = projector or RouteProjector()

    @property
    def name(self) -> str:
        return "mean_total_time"

    def evaluate(self, route: Route, state: ElevatorState) -> float:
        times = list(self._projector.project(route, state).values())
        if not times or any(t == float("inf") for t in times):
            return float("inf")
        return statistics.mean(times)


class CompositeObjective(ObjectiveFunction):
    """Weighted sum of multiple ObjectiveFunction instances.

    Usage
    -----
    obj = CompositeObjective()
        .add(IQRObjective(),      weight=1.0)
        .add(MeanTimeObjective(), weight=0.5)
    """

    def __init__(self) -> None:
        self._objectives: list[tuple[ObjectiveFunction, float]] = []

    @property
    def name(self) -> str:
        parts = "+".join(f"{w}*{o.name}" for o, w in self._objectives)
        return f"composite({parts})"

    def add(self, objective: ObjectiveFunction, weight: float = 1.0) -> "CompositeObjective":
        self._objectives.append((objective, weight))
        return self

    def evaluate(self, route: Route, state: ElevatorState) -> float:
        return sum(
            weight * obj.evaluate(route, state)
            for obj, weight in self._objectives
        )
