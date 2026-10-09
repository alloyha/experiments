"""
Incremental reoptimization heuristic.
"""
from __future__ import annotations

from elevator.domain.models import Route, StopType, ElevatorState
from elevator.heuristics.base import Heuristic, build_stop_pool, is_valid_route, route_distance


class IncrementalHeuristic(Heuristic):
    """Warm-start from the previous route and mutate only the affected segment.

    1. Takes the existing route from ``state.current_route``.
    2. Removes any stops that are no longer valid.
    3. Inserts new stops for passengers not yet in the route using
       cheapest-insertion.

    This is O(n²) vs O(n!) for exhaustive and produces good incremental
    improvements without a full recomputation.
    """

    @property
    def name(self) -> str:
        return "incremental"

    def generate(self, state: ElevatorState) -> list[Route]:
        from elevator.heuristics.nearest_neighbour import NearestNeighbourHeuristic

        active_ids  = {p.id for p in state.all_active}
        onboard_ids = {p.id for p in state.onboard}
        embark, disembark = build_stop_pool(state)
        disembark_map = {s.passenger_id: s for s in disembark}

        existing = [
            s for s in state.current_route.stops
            if s.passenger_id in active_ids
        ]
        in_route_pids = {s.passenger_id for s in existing}

        missing_embarks    = [s for s in embark if s.passenger_id not in in_route_pids]
        missing_disembarks = [
            s for s in disembark
            if s.passenger_id in onboard_ids and s.passenger_id not in in_route_pids
        ]
        to_insert = missing_embarks + missing_disembarks

        route_stops = list(existing)
        for stop in to_insert:
            best_pos  = 0
            best_cost = float("inf")
            for i in range(len(route_stops) + 1):
                candidate = route_stops[:i] + [stop] + route_stops[i:]
                if not is_valid_route(Route(candidate)):
                    continue
                cost = route_distance(Route(candidate), state.current_floor, state.tau)
                if cost < best_cost:
                    best_cost = cost
                    best_pos  = i
            route_stops.insert(best_pos, stop)
            if stop.stop_type == StopType.EMBARK and stop.passenger_id in disembark_map:
                dis = disembark_map[stop.passenger_id]
                for j in range(best_pos + 1, len(route_stops) + 1):
                    candidate = route_stops[:j] + [dis] + route_stops[j:]
                    if is_valid_route(Route(candidate)):
                        route_stops.insert(j, dis)
                        break

        result = Route(route_stops)
        if is_valid_route(result):
            return [result]
        return NearestNeighbourHeuristic().generate(state)
