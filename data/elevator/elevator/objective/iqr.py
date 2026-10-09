"""
IQR-based objective function with explicit fallback policies.
"""
from __future__ import annotations
import logging
import statistics
from enum import Enum

from elevator.domain.models import Route, ElevatorState
from elevator.objective.base import ObjectiveFunction, RouteProjector

logger = logging.getLogger(__name__)


class IQRFallbackPolicy(Enum):
    """Defines behaviour of IQRObjective when fewer than 4 passengers
    are active (IQR is statistically ill-defined below that threshold).

    Attributes
    ----------
    RANGE  : use max - min of total times
    STDEV  : use sample standard deviation
    MEAN   : use arithmetic mean
    ZERO   : return 0.0 — no penalty applied in the small-n case
    """
    RANGE = "range"
    STDEV = "stdev"
    MEAN  = "mean"
    ZERO  = "zero"


class IQRObjective(ObjectiveFunction):
    """Minimize the interquartile range of total passenger times.

    Parameters
    ----------
    fallback  : IQRFallbackPolicy (or string for backward compat)
    projector : optional injected RouteProjector
    """

    def __init__(
        self,
        fallback  : IQRFallbackPolicy | str = IQRFallbackPolicy.RANGE,
        projector : RouteProjector | None   = None,
    ) -> None:
        if isinstance(fallback, str):
            fallback = IQRFallbackPolicy(fallback)
        self.fallback   = fallback
        self._projector = projector or RouteProjector()

    @property
    def name(self) -> str:
        return f"iqr_total_time[fb={self.fallback.value}]"

    def evaluate(self, route: Route, state: ElevatorState) -> float:
        times = list(self._projector.project(route, state).values())
        if any(t == float("inf") for t in times):
            return float("inf")
        if len(times) < 2:
            return 0.0
        if len(times) < 4:
            logger.debug(
                "IQRObjective: only %d passengers — applying fallback policy %s",
                len(times), self.fallback.value,
            )
            return self._apply_fallback(times)
        return self._iqr(times)

    @staticmethod
    def _iqr(values: list[float]) -> float:
        s  = sorted(values)
        n  = len(s)
        q1 = statistics.median(s[: n // 2])
        q3 = statistics.median(s[n // 2 + (n % 2):])
        return q3 - q1

    def _apply_fallback(self, values: list[float]) -> float:
        if self.fallback == IQRFallbackPolicy.RANGE:
            return max(values) - min(values)
        if self.fallback == IQRFallbackPolicy.STDEV:
            return statistics.stdev(values) if len(values) > 1 else 0.0
        if self.fallback == IQRFallbackPolicy.MEAN:
            return statistics.mean(values)
        return 0.0  # ZERO
