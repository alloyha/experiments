"""
SCAN (elevator algorithm) heuristic.
"""
from __future__ import annotations

from elevator.domain.models import Route, Stop, StopType, ElevatorState
from elevator.heuristics.base import Heuristic, build_stop_pool, is_valid_route


class SCANHeuristic(Heuristic):
    """Classic elevator SCAN algorithm — sweep in one direction,
    service all stops, then reverse.

    Two candidate routes are generated (up-first, down-first) and the
    pipeline selects the one with the better objective score.
    """

    @property
    def name(self) -> str:
        return "scan"

    def generate(self, state: ElevatorState) -> list[Route]:
        from elevator.heuristics.nearest_neighbour import NearestNeighbourHeuristic

        embark, disembark = build_stop_pool(state)
        disembark_map = {s.passenger_id: s for s in disembark}
        onboard_ids   = {p.id for p in state.onboard}

        all_stops = list(embark) + [
            s for s in disembark if s.passenger_id in onboard_ids
        ]

        pos    = state.current_floor
        above  = sorted((s for s in all_stops if s.floor >= pos), key=lambda s: s.floor)
        below  = sorted((s for s in all_stops if s.floor < pos),
                        key=lambda s: s.floor, reverse=True)

        routes = []
        for primary, secondary in [(above, below), (below, above)]:
            ordered = list(primary) + list(secondary)
            result: list[Stop] = []
            embarked: set[str]           = set(onboard_ids)
            pending_dis: dict[str, Stop] = {
                pid: disembark_map[pid]
                for pid in onboard_ids if pid in disembark_map
            }

            for stop in ordered:
                result.append(stop)
                if stop.stop_type == StopType.EMBARK:
                    embarked.add(stop.passenger_id)
                    if stop.passenger_id in disembark_map:
                        pending_dis[stop.passenger_id] = disembark_map[stop.passenger_id]

            in_result = {(s.passenger_id, s.stop_type) for s in result}
            for pid, ds in pending_dis.items():
                if (pid, StopType.DISEMBARK) not in in_result:
                    result.append(ds)

            route = Route(result)
            if is_valid_route(route):
                routes.append(route)

        return routes or NearestNeighbourHeuristic().generate(state)
