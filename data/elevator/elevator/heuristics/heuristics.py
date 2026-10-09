"""
Backward-compatible re-export shim.

All concrete heuristics have been moved to individual modules.
Import from ``elevator.heuristics`` (the package) or directly from
the sub-modules.  This file exists for backward compatibility only.
"""
from elevator.heuristics.base import (  # noqa: F401
    Heuristic,
    build_stop_pool,
    is_valid_route,
    route_distance,
)

from elevator.heuristics.nearest_neighbour import NearestNeighbourHeuristic  # noqa: F401
from elevator.heuristics.beam_search import BeamSearchHeuristic               # noqa: F401
from elevator.heuristics.scan import SCANHeuristic                            # noqa: F401
from elevator.heuristics.exhaustive import ExhaustiveHeuristic                # noqa: F401
from elevator.heuristics.simulated_annealing import SimulatedAnnealingHeuristic  # noqa: F401
from elevator.heuristics.incremental import IncrementalHeuristic              # noqa: F401

# Legacy private alias
_build_stop_pool = build_stop_pool
