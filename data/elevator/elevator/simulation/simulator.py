"""
Event-driven simulator for the elevator system.

Drives the OptimizationPipeline by generating CALL, EMBARK, and
DISEMBARK events, updating state accordingly, and recording metrics.

Correctness features
--------------------
- Route deduplication: each call to _apply_route invalidates all
  previously scheduled EMBARK/DISEMBARK events and replaces them
  with fresh ones derived from the new route.  A monotonically
  increasing ``_route_gen`` counter tags every queued event; stale
  events are silently dropped when dequeued.
- Continuous floor interpolation: ``current_floor`` is updated to the
  true fractional position of the elevator at the moment each event
  fires, based on the last known position and the elapsed time since
  the previous route was applied.
- Instant embark: if a new CALL arrives when the elevator is already
  on the passenger's origin floor, the EMBARK event is scheduled at
  ``t_now + epsilon`` rather than after a full route recomputation
  cycle.
"""

from __future__ import annotations
import copy
import heapq
import logging
from dataclasses import dataclass, field
from typing import Optional
import statistics

from elevator.domain.models import (
    ElevatorState, Passenger, Event, EventType, Route, Stop, StopType
)
from elevator.pipeline.pipeline import OptimizationPipeline, OptimizationResult

logger = logging.getLogger(__name__)

_INSTANT_EMBARK_EPSILON = 1e-6   # seconds after call before instant embark fires


# ---------------------------------------------------------------------------
# Simulation tick record  (includes IQR snapshot for convergence tracking)
# ---------------------------------------------------------------------------

@dataclass
class SimulationTick:
    t            : float
    event        : Event
    result       : OptimizationResult
    state_after  : ElevatorState
    iqr_snapshot : float = 0.0   # IQR of delivered passengers at this point


# ---------------------------------------------------------------------------
# Metrics snapshot
# ---------------------------------------------------------------------------

@dataclass
class SimulationMetrics:
    delivered_wait_times  : list[float]  = field(default_factory=list)
    delivered_flight_times: list[float]  = field(default_factory=list)
    delivered_total_times : list[float]  = field(default_factory=list)
    # convergence: (t, iqr) recorded each time a passenger is delivered
    iqr_over_time         : list[tuple[float, float]] = field(default_factory=list)

    @property
    def iqr_total(self) -> float:
        return _iqr(self.delivered_total_times)

    @property
    def mean_wait(self) -> float:
        return statistics.mean(self.delivered_wait_times) if self.delivered_wait_times else 0.0

    @property
    def mean_flight(self) -> float:
        return statistics.mean(self.delivered_flight_times) if self.delivered_flight_times else 0.0

    @property
    def mean_total(self) -> float:
        return statistics.mean(self.delivered_total_times) if self.delivered_total_times else 0.0

    def summary(self) -> str:
        n = len(self.delivered_total_times)
        if n == 0:
            return "No passengers delivered yet."
        return (
            f"Delivered: {n} passengers\n"
            f"  IQR total time : {self.iqr_total:.2f}s\n"
            f"  Mean wait      : {self.mean_wait:.2f}s\n"
            f"  Mean flight    : {self.mean_flight:.2f}s\n"
            f"  Mean total     : {self.mean_total:.2f}s\n"
        )


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    s  = sorted(values)
    n  = len(s)
    q1 = statistics.median(s[: n // 2])
    q3 = statistics.median(s[n // 2 + (n % 2) :])
    return q3 - q1


# ---------------------------------------------------------------------------
# Internal queued event (wraps Event with a generation tag)
# ---------------------------------------------------------------------------

@dataclass(order=False)
class _QueuedEvent:
    """Heap entry that carries a route-generation tag for deduplication."""
    t         : float
    seq       : int
    generation: int
    event     : Event

    def __lt__(self, other: "_QueuedEvent") -> bool:
        return (self.t, self.seq) < (other.t, other.seq)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class ElevatorSimulator:
    """Discrete-event simulator.

    Parameters
    ----------
    pipeline       : configured OptimizationPipeline
    initial_state  : starting ElevatorState
    verbose        : print events as they happen
    seed           : optional integer seed for reproducible random scenarios
    """

    def __init__(
        self,
        pipeline      : OptimizationPipeline,
        initial_state : ElevatorState,
        verbose       : bool = False,
        seed          : Optional[int] = None,
    ) -> None:
        self._pipeline   = pipeline
        self._state      = initial_state
        self._verbose    = verbose
        self._metrics    = SimulationMetrics()
        self._ticks      : list[SimulationTick] = []
        self._event_q    : list[_QueuedEvent]   = []
        self._seq        = 0        # tiebreaker for heap ordering
        self._route_gen  = 0        # incremented every time a new route is applied
        self._current_gen = 0       # generation of the currently active route
        # For continuous interpolation: track where and when the last route began
        self._route_start_floor : float = initial_state.current_floor
        self._route_start_t     : float = initial_state.t_now
        self._route_direction   : float = 0.0   # +1 up, -1 down, 0 idle
        if seed is not None:
            import random
            random.seed(seed)
            logger.debug("Simulator seeded with %d", seed)

    # -- public API ---------------------------------------------------------

    def schedule_call(self, t: float, passenger: Passenger) -> None:
        """Schedule a CALL event at time ``t``."""
        self._push_external(Event(t=t, event_type=EventType.CALL, passenger=passenger))

    def run(self) -> SimulationMetrics:
        """Process all scheduled events in chronological order."""
        while self._event_q:
            entry = heapq.heappop(self._event_q)
            # Drop stale route-derived events (deduplication)
            if entry.generation != 0 and entry.generation != self._current_gen:
                logger.debug("Dropping stale event gen=%d (current=%d): %s",
                             entry.generation, self._current_gen, entry.event.event_type)
                continue
            self._process(entry.event)
        return self._metrics

    @property
    def metrics(self) -> SimulationMetrics:
        return self._metrics

    @property
    def ticks(self) -> list[SimulationTick]:
        return self._ticks

    # -- internal push helpers ----------------------------------------------

    def _push_external(self, event: Event) -> None:
        """Push a CALL event that is always processed (generation=0)."""
        entry = _QueuedEvent(t=event.t, seq=self._seq, generation=0, event=event)
        heapq.heappush(self._event_q, entry)
        self._seq += 1

    def _push_route_event(self, event: Event, gen: int) -> None:
        """Push an EMBARK/DISEMBARK event tagged with a route generation."""
        entry = _QueuedEvent(t=event.t, seq=self._seq, generation=gen, event=event)
        heapq.heappush(self._event_q, entry)
        self._seq += 1

    # -- floor interpolation ------------------------------------------------

    def _interpolate_floor(self, t_now: float) -> float:
        """Return the elevator's true fractional floor position at ``t_now``.

        Uses the direction and speed recorded when the current route was
        applied, clamped to the next scheduled stop so the elevator never
        overshoots.
        """
        elapsed   = t_now - self._route_start_t
        travelled = elapsed / self._state.tau   # floors covered since route start
        pos       = self._route_start_floor + self._route_direction * travelled

        # Clamp: find the nearest stop in the current route
        if self._state.current_route.stops:
            next_floor = self._state.current_route.stops[0].floor
            if self._route_direction > 0:
                pos = min(pos, float(next_floor))
            elif self._route_direction < 0:
                pos = max(pos, float(next_floor))

        return pos

    # -- main event handler -------------------------------------------------

    def _process(self, event: Event) -> None:
        t = event.t
        p = event.passenger
        s = self._state

        # Update continuous floor position before mutating state
        s.current_floor = self._interpolate_floor(t)
        s.t_now         = t

        logger.debug("[t=%.2f] %s passenger=%s floor=%.2f",
                     t, event.event_type.value.upper(), p.id, s.current_floor)

        if self._verbose:
            loc = p.origin if event.event_type == EventType.CALL else int(s.current_floor)
            print(f"[t={t:.2f}] {event.event_type.value.upper()} "
                  f"passenger={p.id} floor={loc}")

        if event.event_type == EventType.CALL:
            s.waiting.append(p)

            # Instant embark: elevator already at passenger's origin
            if abs(s.current_floor - p.origin) < 0.5:
                logger.debug("Instant embark for %s at floor %d", p.id, p.origin)
                instant_event = Event(
                    t          = t + _INSTANT_EMBARK_EPSILON,
                    event_type = EventType.EMBARK,
                    passenger  = p,
                )
                self._push_external(instant_event)
                # Still optimise so the rest of the route is correct
                result = self._pipeline.optimize(s)
                self._apply_route(result.best_route, t)
                self._record(t, event, result)
                return

            result = self._pipeline.optimize(s)
            self._apply_route(result.best_route, t)
            self._record(t, event, result)

        elif event.event_type == EventType.EMBARK:
            # Guard: passenger must still be waiting (not already served)
            if not any(x.id == p.id for x in s.waiting):
                logger.debug("Dropping duplicate EMBARK for %s", p.id)
                return
            s.waiting       = [x for x in s.waiting if x.id != p.id]
            boarded         = p.with_embark(t)
            s.onboard.append(boarded)
            s.current_floor = float(p.origin)
            result = self._pipeline.optimize(s)
            self._apply_route(result.best_route, t)
            self._record(t, event, result)

        elif event.event_type == EventType.DISEMBARK:
            # Guard: passenger must still be onboard
            match = next((x for x in s.onboard if x.id == p.id), None)
            if match is None:
                logger.debug("Dropping duplicate DISEMBARK for %s", p.id)  # pragma: no cover
                return  # pragma: no cover
            s.onboard       = [x for x in s.onboard if x.id != p.id]
            s.current_floor = float(p.destination)
            t_emb       = match.t_embark if match.t_embark is not None else t
            wait_time   = t_emb - match.t_call
            flight_time = t - t_emb
            self._metrics.delivered_wait_times.append(wait_time)
            self._metrics.delivered_flight_times.append(flight_time)
            self._metrics.delivered_total_times.append(wait_time + flight_time)
            # convergence snapshot
            self._metrics.iqr_over_time.append((t, self._metrics.iqr_total))
            if self._verbose:
                print(f"         → delivered: wait={wait_time:.2f}s "
                      f"flight={flight_time:.2f}s "
                      f"total={wait_time+flight_time:.2f}s")
            result = self._pipeline.optimize(s)
            self._apply_route(result.best_route, t)
            self._record(t, event, result)

    # -- route application --------------------------------------------------

    def _apply_route(self, route: Route, t_now: float) -> None:
        """Replace the active route.

        Bumps the route generation counter so any previously queued
        EMBARK/DISEMBARK events are invalidated.  Schedules fresh events
        for every stop in the new route.
        """
        # Bump generation — all previously queued route events are now stale
        self._route_gen  += 1
        self._current_gen = self._route_gen

        s = self._state
        s.current_route = route

        if not route.stops:
            self._route_direction = 0.0
            return

        # Record interpolation anchor
        self._route_start_floor = s.current_floor
        self._route_start_t     = t_now
        first_floor             = route.stops[0].floor
        diff                    = first_floor - s.current_floor
        self._route_direction   = (1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0))

        pos      = s.current_floor
        t_cursor = t_now
        gen      = self._current_gen

        for stop in route.stops:
            t_cursor += abs(stop.floor - pos) * s.tau
            pos       = stop.floor
            new_event = Event(
                t          = t_cursor,
                event_type = (EventType.EMBARK if stop.stop_type == StopType.EMBARK
                              else EventType.DISEMBARK),
                passenger  = self._find_passenger(stop.passenger_id),
            )
            self._push_route_event(new_event, gen)

    def _find_passenger(self, pid: str) -> Passenger:
        for p in self._state.all_active:
            if p.id == pid:
                return p
        raise ValueError(f"Passenger {pid!r} not found in active state.")  # pragma: no cover

    def _record(self, t: float, event: Event, result: OptimizationResult) -> None:
        self._ticks.append(SimulationTick(
            t            = t,
            event        = event,
            result       = result,
            state_after  = copy.deepcopy(self._state),
            iqr_snapshot = self._metrics.iqr_total,
        ))
