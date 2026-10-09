"""
Core domain models for the elevator optimization system.
All time units are seconds. Distance units are floors.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class StopType(Enum):
    EMBARK    = "embark"
    DISEMBARK = "disembark"


class EventType(Enum):
    CALL      = "call"       # new passenger requests pickup
    EMBARK    = "embark"     # passenger boards elevator
    DISEMBARK = "disembark"  # passenger exits elevator


# ---------------------------------------------------------------------------
# Passenger
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Passenger:
    """Immutable passenger record.

    Attributes
    ----------
    id              : unique identifier
    origin          : floor where they are waiting
    destination     : floor where they want to go (known at call time)
    t_call          : absolute time of the call event
    weight_kg       : passenger weight (used by weight capacity constraint)
    t_embark        : filled when they board (None if still waiting)
    max_wait_time   : max seconds willing to wait after t_call (None = no limit)
    is_vip          : if True, this passenger has priority
    elevator_id     : which elevator to prefer (None = any)
    """
    id              : str
    origin          : int
    destination     : int
    t_call          : float
    weight_kg       : float           = 70.0
    t_embark        : Optional[float] = None
    max_wait_time   : Optional[float] = None
    is_vip          : bool            = False
    elevator_id     : Optional[str]   = None

    @property
    def is_waiting(self) -> bool:
        return self.t_embark is None

    @property
    def is_onboard(self) -> bool:
        return self.t_embark is not None

    def with_embark(self, t: float) -> Passenger:
        """Return a new Passenger with t_embark set."""
        return Passenger(
            id=self.id,
            origin=self.origin,
            destination=self.destination,
            t_call=self.t_call,
            weight_kg=self.weight_kg,
            t_embark=t,
            max_wait_time=self.max_wait_time,
            is_vip=self.is_vip,
            elevator_id=self.elevator_id,
        )


# ---------------------------------------------------------------------------
# Stop  (a single planned visit in a route)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stop:
    """One planned action in a route.

    Attributes
    ----------
    floor        : andar a ser visitado
    stop_type    : EMBARK or DISEMBARK
    passenger_id : which passenger this stop serves
    """
    floor       : int
    stop_type   : StopType
    passenger_id: str


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@dataclass
class Route:
    """Ordered sequence of stops the elevator will execute.

    Invariant: for each passenger, their EMBARK stop precedes their
    DISEMBARK stop in the sequence.
    
    Attributes
    ----------
    stops       : list of Stop objects
    route_id    : unique identifier for deduplication
    computed_at : absolute time when this route was computed
    """
    stops       : list[Stop] = field(default_factory=list)
    route_id    : str        = field(default_factory=lambda: __import__('uuid').uuid4().hex[:8])
    computed_at : float      = 0.0

    def __len__(self) -> int:
        return len(self.stops)

    def __iter__(self):
        return iter(self.stops)

    def floors(self) -> list[int]:
        return [s.floor for s in self.stops]


# ---------------------------------------------------------------------------
# ElevatorState
# ---------------------------------------------------------------------------

@dataclass
class ElevatorState:
    """Mutable snapshot of a single elevator cabin.

    Attributes
    ----------
    id                   : elevator identifier
    current_floor        : fractional floor position (e.g. 3.5 = mid-transit 3→4)
    onboard              : passengers currently inside the cabin
    waiting              : passengers waiting to be picked up
    t_now                : current simulation time
    tau                  : travel time per floor (seconds/floor)
    current_route        : active route being executed (may be empty)
    building_stop_counts : floor → number of stops other elevators already have
                           scheduled at that floor (populated by FleetCoordinator)
    cumulative_travel    : total floors the elevator has ever travelled
                           (used by MaintenanceFloorConstraint)
    """
    id                   : str
    current_floor        : float
    onboard              : list[Passenger]
    waiting              : list[Passenger]
    t_now                : float
    tau                  : float            = 1.0
    current_route        : Route            = field(default_factory=Route)
    building_stop_counts : dict[int, int]   = field(default_factory=dict)
    cumulative_travel    : float            = 0.0

    # -- derived helpers --------------------------------------------------

    @property
    def all_active(self) -> list[Passenger]:
        return self.waiting + self.onboard

    @property
    def total_weight(self) -> float:
        return sum(p.weight_kg for p in self.onboard)

    @property
    def passenger_count(self) -> int:
        return len(self.onboard)

    def travel_time_to(self, floor: int) -> float:
        """Seconds needed to reach `floor` from current position."""
        return abs(floor - self.current_floor) * self.tau


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """Discrete simulation event that triggers a route recomputation.

    Attributes
    ----------
    t           : absolute time of the event
    event_type  : CALL | EMBARK | DISEMBARK
    passenger   : the passenger involved
    """
    t          : float
    event_type : EventType
    passenger  : Passenger
