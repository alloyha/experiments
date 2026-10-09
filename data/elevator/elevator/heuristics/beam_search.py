"""
Beam-search heuristic.
"""
from __future__ import annotations
from dataclasses import dataclass

from elevator.domain.models import Route, Stop, StopType, ElevatorState
from elevator.heuristics.base import Heuristic, build_stop_pool


@dataclass
class _BeamNode:
    route      : list[Stop]
    pos        : float
    embarked   : frozenset[str]
    available  : tuple[Stop, ...]
    cost       : float


class BeamSearchHeuristic(Heuristic):
    """Beam search over the stop-sequencing space.

    Parameters
    ----------
    beam_width : number of partial routes kept at each expansion step
    """

    def __init__(self, beam_width: int = 10) -> None:
        self.beam_width = beam_width

    @property
    def name(self) -> str:
        return f"beam_search_w{self.beam_width}"

    def generate(self, state: ElevatorState) -> list[Route]:
        embark, disembark = build_stop_pool(state)
        disembark_map = {s.passenger_id: s for s in disembark}
        onboard_ids   = frozenset(p.id for p in state.onboard)

        initial_available: list[Stop] = list(embark)
        for pid in onboard_ids:
            if pid in disembark_map:
                initial_available.append(disembark_map[pid])

        beam: list[_BeamNode] = [_BeamNode(
            route     = [],
            pos       = state.current_floor,
            embarked  = onboard_ids,
            available = tuple(initial_available),
            cost      = 0.0,
        )]

        while beam and any(node.available for node in beam):
            candidates: list[_BeamNode] = []
            for node in beam:
                if not node.available:  # pragma: no cover
                    candidates.append(node)  # pragma: no cover
                    continue  # pragma: no cover
                for stop in node.available:
                    new_route    = node.route + [stop]
                    new_pos      = stop.floor
                    new_embarked = (
                        frozenset(node.embarked | {stop.passenger_id})
                        if stop.stop_type == StopType.EMBARK
                        else node.embarked
                    )
                    new_available = list(node.available)
                    new_available.remove(stop)
                    if stop.stop_type == StopType.EMBARK and stop.passenger_id in disembark_map:
                        new_available.append(disembark_map[stop.passenger_id])
                    new_cost = node.cost + abs(stop.floor - node.pos) * state.tau
                    candidates.append(_BeamNode(
                        route     = new_route,
                        pos       = new_pos,
                        embarked  = new_embarked,
                        available = tuple(new_available),
                        cost      = new_cost,
                    ))

            candidates.sort(key=lambda n: n.cost)
            beam = candidates[: self.beam_width]

        return [Route(node.route) for node in beam] or [Route([])]
