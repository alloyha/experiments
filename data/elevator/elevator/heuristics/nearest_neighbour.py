"""
Nearest-neighbour (greedy) heuristic.
"""
from __future__ import annotations

from elevator.domain.models import Route, StopType, ElevatorState
from elevator.heuristics.base import Heuristic, build_stop_pool


class NearestNeighbourHeuristic(Heuristic):
    """At each step, visit the nearest stop that is currently reachable.

    A disembark stop is reachable only after its embark stop has been
    visited.  An embark stop is always reachable.
    """

    @property
    def name(self) -> str:
        return "nearest_neighbour"

    def generate(self, state: ElevatorState) -> list[Route]:
        embark, disembark = build_stop_pool(state)
        disembark_map = {s.passenger_id: s for s in disembark}

        available   = list(embark)
        onboard_ids = {p.id for p in state.onboard}
        for pid in onboard_ids:
            if pid in disembark_map:
                available.append(disembark_map[pid])

        pos      = state.current_floor
        visited  = []
        embarked: set[str] = set(onboard_ids)

        while available:
            nxt = min(available, key=lambda s: abs(s.floor - pos))
            visited.append(nxt)
            available.remove(nxt)
            pos = nxt.floor

            if nxt.stop_type == StopType.EMBARK:
                embarked.add(nxt.passenger_id)
                if nxt.passenger_id in disembark_map:
                    available.append(disembark_map[nxt.passenger_id])

        return [Route(visited)]
