"""
Backward-compatible re-export shim.

All concrete constraint implementations have been moved to:
  - elevator.constraints.capacity  (weight, pax count, shafts-per-floor)
  - elevator.constraints.routing   (floor range, no-reverse, stop gap)
  - elevator.constraints.policy    (time window, VIP, maintenance, peak)
"""
from elevator.constraints.capacity import (  # noqa: F401
    WeightCapacityConstraint,
    PassengerCapacityConstraint,
    ShaftsPerFloorConstraint,
)
from elevator.constraints.routing import (   # noqa: F401
    FloorRangeConstraint,
    NoReverseConstraint,
    MinStopGapConstraint,
)
from elevator.constraints.policy import (    # noqa: F401
    TimeWindowConstraint,
    VIPConstraint,
    MaintenanceFloorConstraint,
    PeakHourConstraint,
)
