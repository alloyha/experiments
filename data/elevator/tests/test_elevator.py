"""Test suite for the elevator optimizer.

Covers:
- RouteProjector correctness
- Constraint checks (all builtin constraints)
- IQRObjective and fallback policies
- All heuristics (generate valid routes)
- Simulator: deduplication, instant embark, convergence tracking
- FleetCoordinator: assignment and stop-count refresh
"""
from __future__ import annotations

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from elevator.domain.models import (
    Passenger, ElevatorState, Route, Stop, StopType, Event, EventType
)
from elevator.constraints.base import ConstraintRegistry
from elevator.constraints.builtin import (
    WeightCapacityConstraint,
    PassengerCapacityConstraint,
    FloorRangeConstraint,
    NoReverseConstraint,
    MinStopGapConstraint,
    TimeWindowConstraint,
    VIPConstraint,
    MaintenanceFloorConstraint,
    PeakHourConstraint,
    ShaftsPerFloorConstraint,
)
from elevator.objective.objective import (
    RouteProjector, IQRObjective, IQRFallbackPolicy,
    MeanTimeObjective, CompositeObjective,
)
from elevator.heuristics.heuristics import (
    NearestNeighbourHeuristic, BeamSearchHeuristic, SCANHeuristic,
    ExhaustiveHeuristic, SimulatedAnnealingHeuristic, IncrementalHeuristic,
    is_valid_route,
)
from elevator.pipeline.pipeline import OptimizationPipeline
from elevator.simulation.simulator import ElevatorSimulator
from elevator.fleet.coordinator import FleetCoordinator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _p(pid: str, origin: int, dest: int, t_call: float = 0.0,
       weight: float = 70.0, max_wait: float | None = None,
       is_vip: bool = False) -> Passenger:
    return Passenger(id=pid, origin=origin, destination=dest,
                     t_call=t_call, weight_kg=weight,
                     max_wait_time=max_wait, is_vip=is_vip)


def _state(passengers: list[Passenger] | None = None, floor: float = 1.0,
           tau: float = 1.0) -> ElevatorState:
    return ElevatorState(
        id="E1", current_floor=floor,
        onboard=[], waiting=passengers or [],
        t_now=0.0, tau=tau,
    )


def _route(*stops: tuple[int, StopType, str]) -> Route:
    return Route([Stop(floor=f, stop_type=st, passenger_id=pid)
                  for f, st, pid in stops])


def _simple_pipeline() -> OptimizationPipeline:
    return (
        OptimizationPipeline()
        .add_heuristic(BeamSearchHeuristic(beam_width=5))
        .set_objective(IQRObjective())
    )


# ---------------------------------------------------------------------------
# RouteProjector
# ---------------------------------------------------------------------------

class TestRouteProjector:
    def test_single_passenger_total_time(self):
        """Total time = wait + flight = (embark - call) + (disembark - embark)."""
        p     = _p("A", origin=1, dest=5, t_call=0.0)
        state = _state([p], floor=1.0)
        route = _route((1, StopType.EMBARK, "A"), (5, StopType.DISEMBARK, "A"))
        proj  = RouteProjector().project(route, state)
        # elevator at floor 1, tau=1 → embark at t=0, disembark at t=4
        assert proj["A"] == pytest.approx(4.0)

    def test_elevator_must_travel_to_origin(self):
        """If elevator starts at floor 3 and passenger is at floor 1."""
        p     = _p("A", origin=1, dest=5, t_call=0.0)
        state = _state([p], floor=3.0)
        route = _route((1, StopType.EMBARK, "A"), (5, StopType.DISEMBARK, "A"))
        proj  = RouteProjector().project(route, state)
        # travel to 1 = 2s, then 4 floors to 5 = 4s → total=6s, wait=2s, flight=4s
        assert proj["A"] == pytest.approx(6.0)

    def test_missing_passenger_returns_inf(self):
        p     = _p("A", origin=1, dest=5, t_call=0.0)
        state = _state([p], floor=1.0)
        route = Route([])   # empty route — passenger never embarks
        proj  = RouteProjector().project(route, state)
        assert proj["A"] == float("inf")

    def test_two_passengers_ordered(self):
        p1 = _p("A", origin=2, dest=6, t_call=0.0)
        p2 = _p("B", origin=4, dest=8, t_call=0.0)
        state = _state([p1, p2], floor=1.0)
        route = _route(
            (2, StopType.EMBARK, "A"),
            (4, StopType.EMBARK, "B"),
            (6, StopType.DISEMBARK, "A"),
            (8, StopType.DISEMBARK, "B"),
        )
        proj = RouteProjector().project(route, state)
        # A: embark at t=1, disembark at t=5 → total=5
        # B: embark at t=3, disembark at t=7 → total=7
        assert proj["A"] == pytest.approx(5.0)
        assert proj["B"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class TestWeightCapacity:
    def test_within_capacity(self):
        p     = _p("A", 1, 5, weight=100.0)
        state = _state([p], floor=1.0)
        route = _route((1, StopType.EMBARK, "A"), (5, StopType.DISEMBARK, "A"))
        assert WeightCapacityConstraint(max_kg=400.0).check(route, state)

    def test_exceeds_capacity(self):
        p     = _p("A", 1, 5, weight=500.0)
        state = _state([p], floor=1.0)
        route = _route((1, StopType.EMBARK, "A"), (5, StopType.DISEMBARK, "A"))
        assert not WeightCapacityConstraint(max_kg=400.0).check(route, state)

    def test_penalty_proportional_to_excess(self):
        p     = _p("A", 1, 5, weight=450.0)
        state = _state([p], floor=1.0)
        route = _route((1, StopType.EMBARK, "A"), (5, StopType.DISEMBARK, "A"))
        penalty = WeightCapacityConstraint(max_kg=400.0).penalty(route, state)
        assert penalty == pytest.approx(50.0)


class TestPassengerCapacity:
    def test_within_capacity(self):
        p1, p2 = _p("A", 1, 5), _p("B", 2, 6)
        state  = _state([p1, p2])
        route  = _route(
            (1, StopType.EMBARK, "A"), (2, StopType.EMBARK, "B"),
            (5, StopType.DISEMBARK, "A"), (6, StopType.DISEMBARK, "B"),
        )
        assert PassengerCapacityConstraint(max_pax=3).check(route, state)

    def test_exceeds_capacity(self):
        p1, p2 = _p("A", 1, 5), _p("B", 2, 6)
        state  = _state([p1, p2])
        route  = _route(
            (1, StopType.EMBARK, "A"), (2, StopType.EMBARK, "B"),
            (5, StopType.DISEMBARK, "A"), (6, StopType.DISEMBARK, "B"),
        )
        assert not PassengerCapacityConstraint(max_pax=1).check(route, state)


class TestFloorRange:
    def test_all_in_range(self):
        p     = _p("A", 2, 8)
        state = _state([p])
        route = _route((2, StopType.EMBARK, "A"), (8, StopType.DISEMBARK, "A"))
        assert FloorRangeConstraint(1, 10).check(route, state)

    def test_out_of_range(self):
        p     = _p("A", 2, 12)
        state = _state([p])
        route = _route((2, StopType.EMBARK, "A"), (12, StopType.DISEMBARK, "A"))
        assert not FloorRangeConstraint(1, 10).check(route, state)


class TestNoReverse:
    def test_no_reversal_zero_penalty(self):
        p1, p2 = _p("A", 1, 5), _p("B", 3, 7)
        state  = _state([p1, p2], floor=1.0)
        route  = _route(
            (1, StopType.EMBARK, "A"), (3, StopType.EMBARK, "B"),
            (5, StopType.DISEMBARK, "A"), (7, StopType.DISEMBARK, "B"),
        )
        assert NoReverseConstraint().penalty(route, state) == 0.0

    def test_reversal_incurs_penalty(self):
        p1, p2 = _p("A", 5, 1), _p("B", 3, 7)
        state  = _state([p1, p2], floor=1.0)
        route  = _route(
            (5, StopType.EMBARK, "A"), (1, StopType.DISEMBARK, "A"),
            (3, StopType.EMBARK, "B"), (7, StopType.DISEMBARK, "B"),
        )
        penalty = NoReverseConstraint(reversal_cost=2.0).penalty(route, state)
        assert penalty > 0.0


class TestTimeWindow:
    def test_within_deadline(self):
        p     = _p("A", 1, 5, t_call=0.0, max_wait=10.0)
        state = _state([p], floor=1.0)
        route = _route((1, StopType.EMBARK, "A"), (5, StopType.DISEMBARK, "A"))
        assert TimeWindowConstraint().check(route, state)

    def test_exceeds_deadline(self):
        p     = _p("A", 10, 1, t_call=0.0, max_wait=3.0)
        state = _state([p], floor=1.0)
        # Must travel 9 floors to embark → t_embark = 9 > deadline=3
        route = _route((10, StopType.EMBARK, "A"), (1, StopType.DISEMBARK, "A"))
        assert not TimeWindowConstraint().check(route, state)

    def test_global_default_applies(self):
        p     = _p("A", 10, 1, t_call=0.0)   # no per-passenger max_wait
        state = _state([p], floor=1.0)
        route = _route((10, StopType.EMBARK, "A"), (1, StopType.DISEMBARK, "A"))
        # default_max_wait=5 → 9 floors to embark → violates
        assert not TimeWindowConstraint(default_max_wait=5.0).check(route, state)


class TestVIPConstraint:
    def test_vip_before_non_vip_no_penalty(self):
        vip  = _p("V", 1, 5, is_vip=True)
        norm = _p("N", 3, 7, is_vip=False)
        state = _state([vip, norm])
        route = _route(
            (1, StopType.EMBARK, "V"), (3, StopType.EMBARK, "N"),
            (5, StopType.DISEMBARK, "V"), (7, StopType.DISEMBARK, "N"),
        )
        assert VIPConstraint().penalty(route, state) == 0.0

    def test_non_vip_before_vip_incurs_penalty(self):
        vip  = _p("V", 3, 5, is_vip=True)
        norm = _p("N", 1, 7, is_vip=False)
        state = _state([vip, norm])
        route = _route(
            (1, StopType.EMBARK, "N"), (3, StopType.EMBARK, "V"),
            (5, StopType.DISEMBARK, "V"), (7, StopType.DISEMBARK, "N"),
        )
        assert VIPConstraint(vip_penalty=500.0).penalty(route, state) == pytest.approx(500.0)


class TestPeakHourConstraint:
    def test_within_pax_limit_during_peak(self):
        p1, p2 = _p("A", 1, 5), _p("B", 2, 6)
        state  = ElevatorState(
            id="E1", current_floor=1.0, onboard=[], waiting=[p1, p2],
            t_now=9 * 3600.0, tau=1.0,
        )
        route = _route(
            (1, StopType.EMBARK, "A"), (2, StopType.EMBARK, "B"),
            (5, StopType.DISEMBARK, "A"), (6, StopType.DISEMBARK, "B"),
        )
        # peak window 8h–10h, max 3 pax → 2 pax OK
        c = PeakHourConstraint(
            peak_windows=[(8 * 3600, 10 * 3600)], peak_max_pax=3
        )
        assert c.check(route, state)

    def test_exceeds_pax_limit_during_peak(self):
        passengers = [_p(f"P{i}", i, i + 5) for i in range(1, 6)]
        state = ElevatorState(
            id="E1", current_floor=1.0, onboard=[], waiting=passengers,
            t_now=9 * 3600.0, tau=1.0,
        )
        stops = []
        for p in passengers:
            stops.append(Stop(p.origin, StopType.EMBARK, p.id))
        for p in passengers:
            stops.append(Stop(p.destination, StopType.DISEMBARK, p.id))
        route = Route(stops)
        c = PeakHourConstraint(
            peak_windows=[(8 * 3600, 10 * 3600)], peak_max_pax=3
        )
        assert not c.check(route, state)


class TestShaftsPerFloor:
    def test_no_conflict(self):
        p     = _p("A", 3, 7)
        state = ElevatorState(
            id="E1", current_floor=1.0, onboard=[], waiting=[p],
            t_now=0.0, tau=1.0,
            building_stop_counts={3: 0},
        )
        route = _route((3, StopType.EMBARK, "A"), (7, StopType.DISEMBARK, "A"))
        assert ShaftsPerFloorConstraint(max_shafts=2).check(route, state)

    def test_conflict_exceeds_shafts(self):
        p     = _p("A", 3, 7)
        state = ElevatorState(
            id="E1", current_floor=1.0, onboard=[], waiting=[p],
            t_now=0.0, tau=1.0,
            building_stop_counts={3: 2},   # other elevators already have 2 stops at floor 3
        )
        route = _route((3, StopType.EMBARK, "A"), (7, StopType.DISEMBARK, "A"))
        assert not ShaftsPerFloorConstraint(max_shafts=2).check(route, state)


# ---------------------------------------------------------------------------
# IQRObjective
# ---------------------------------------------------------------------------

class TestIQRObjective:
    def test_iqr_four_passengers(self):
        passengers = [_p(f"P{i}", i, i + 4, t_call=0.0) for i in range(1, 5)]
        state = _state(passengers, floor=1.0)
        stops = []
        for p in passengers:
            stops += [
                Stop(p.origin, StopType.EMBARK, p.id),
                Stop(p.destination, StopType.DISEMBARK, p.id),
            ]
        route = Route(stops)
        score = IQRObjective().evaluate(route, state)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_fallback_range(self):
        p     = _p("A", 1, 5, t_call=0.0)
        state = _state([p], floor=1.0)
        route = _route((1, StopType.EMBARK, "A"), (5, StopType.DISEMBARK, "A"))
        score = IQRObjective(fallback=IQRFallbackPolicy.RANGE).evaluate(route, state)
        assert score == pytest.approx(0.0)  # only 1 passenger → max-min = 0

    def test_fallback_accepts_string(self):
        obj = IQRObjective(fallback="stdev")
        assert obj.fallback == IQRFallbackPolicy.STDEV

    def test_name_includes_fallback(self):
        assert "range" in IQRObjective(fallback=IQRFallbackPolicy.RANGE).name


# ---------------------------------------------------------------------------
# Heuristics — all must produce valid routes
# ---------------------------------------------------------------------------

class TestHeuristics:
    @pytest.fixture
    def state_2p(self):
        p1 = _p("A", 2, 7, t_call=0.0)
        p2 = _p("B", 5, 1, t_call=0.0)
        return _state([p1, p2], floor=1.0)

    def _check_routes(self, routes: list[Route], state: ElevatorState):
        assert len(routes) >= 1
        for r in routes:
            assert is_valid_route(r), f"Invalid route: {[s.floor for s in r.stops]}"

    def test_nearest_neighbour(self, state_2p):
        routes = NearestNeighbourHeuristic().generate(state_2p)
        self._check_routes(routes, state_2p)

    def test_beam_search(self, state_2p):
        routes = BeamSearchHeuristic(beam_width=5).generate(state_2p)
        self._check_routes(routes, state_2p)

    def test_scan(self, state_2p):
        routes = SCANHeuristic().generate(state_2p)
        self._check_routes(routes, state_2p)

    def test_exhaustive_bnb(self, state_2p):
        routes = ExhaustiveHeuristic(max_stops=8).generate(state_2p)
        self._check_routes(routes, state_2p)

    def test_simulated_annealing(self, state_2p):
        routes = SimulatedAnnealingHeuristic(max_iter=200, seed=42).generate(state_2p)
        self._check_routes(routes, state_2p)

    def test_incremental(self, state_2p):
        routes = IncrementalHeuristic().generate(state_2p)
        self._check_routes(routes, state_2p)

    def test_exhaustive_falls_back_when_too_many_stops(self):
        """With 12 passengers (24 stops) exhaustive should fall back gracefully."""
        passengers = [_p(f"P{i}", i, i + 13, t_call=0.0) for i in range(1, 13)]
        state  = _state(passengers, floor=1.0)
        routes = ExhaustiveHeuristic(max_stops=8).generate(state)
        assert len(routes) >= 1


# ---------------------------------------------------------------------------
# Simulator correctness
# ---------------------------------------------------------------------------

class TestSimulator:
    def _make_sim(self, passengers: list[Passenger], verbose: bool = False,
                  seed: int | None = None) -> ElevatorSimulator:
        state    = _state(floor=1.0)
        pipeline = _simple_pipeline()
        sim      = ElevatorSimulator(pipeline, state, verbose=verbose, seed=seed)
        for p in passengers:
            sim.schedule_call(t=p.t_call, passenger=p)
        return sim

    def test_all_passengers_delivered(self):
        passengers = [
            _p("P1", 3, 8,  t_call=0.0),
            _p("P2", 7, 2,  t_call=1.5),
            _p("P3", 1, 10, t_call=2.0),
        ]
        sim     = self._make_sim(passengers)
        metrics = sim.run()
        assert len(metrics.delivered_total_times) == 3

    def test_no_duplicate_deliveries(self):
        """Each passenger should be delivered exactly once."""
        passengers = [
            _p("P1", 3, 8, t_call=0.0),
            _p("P2", 7, 2, t_call=1.0),
        ]
        sim     = self._make_sim(passengers)
        metrics = sim.run()
        assert len(metrics.delivered_total_times) == len(passengers)

    def test_instant_embark_when_at_origin(self):
        """If elevator starts at the passenger's origin, embark is near-instant.
        
        The wait time (t_embark - t_call) should be tiny (≈ epsilon),
        while total time = wait + flight will be non-zero.
        """
        p   = _p("A", 1, 5, t_call=0.0)   # elevator starts at floor 1 = origin
        sim = self._make_sim([p])
        metrics = sim.run()
        assert len(metrics.delivered_total_times) == 1
        # wait time should be tiny (≈ epsilon), flight = 4s
        assert metrics.delivered_wait_times[0] < 0.01
        assert metrics.delivered_flight_times[0] == pytest.approx(4.0, abs=0.1)

    def test_convergence_tracking(self):
        passengers = [
            _p("P1", 3, 8, t_call=0.0),
            _p("P2", 7, 2, t_call=1.0),
            _p("P3", 1, 5, t_call=2.0),
        ]
        sim     = self._make_sim(passengers)
        metrics = sim.run()
        assert len(metrics.iqr_over_time) == 3   # one snapshot per delivery
        # All timestamps should be non-decreasing
        times = [t for t, _ in metrics.iqr_over_time]
        assert times == sorted(times)

    def test_deterministic_with_seed(self):
        passengers = [_p(f"P{i}", i, i + 3, t_call=float(i)) for i in range(1, 5)]
        m1 = self._make_sim(passengers, seed=99).run()
        m2 = self._make_sim(passengers, seed=99).run()
        assert m1.delivered_total_times == m2.delivered_total_times

    def test_total_times_are_positive(self):
        passengers = [_p("P1", 2, 9, t_call=0.0)]
        sim     = self._make_sim(passengers)
        metrics = sim.run()
        assert all(t > 0 for t in metrics.delivered_total_times)


# ---------------------------------------------------------------------------
# FleetCoordinator
# ---------------------------------------------------------------------------

class TestFleetCoordinator:
    def _make_fleet(self):
        s1 = ElevatorState(id="E1", current_floor=1.0, onboard=[], waiting=[],
                           t_now=0.0, tau=1.0)
        s2 = ElevatorState(id="E2", current_floor=10.0, onboard=[], waiting=[],
                           t_now=0.0, tau=1.0)
        p1 = _simple_pipeline()
        p2 = _simple_pipeline()
        coord = FleetCoordinator(
            elevators=[s1, s2],
            pipelines={"E1": p1, "E2": p2},
        )
        return coord, s1, s2

    def test_assigns_to_one_elevator(self):
        coord, s1, s2 = self._make_fleet()
        p   = _p("A", 2, 8, t_call=0.0)
        res = coord.assign_call(p, t_now=0.0)
        assert res.chosen_elevator_id in ("E1", "E2")

    def test_pinned_elevator_respected(self):
        """When passenger specifies elevator_id, that elevator must be chosen."""
        coord, s1, s2 = self._make_fleet()
        import dataclasses
        p = dataclasses.replace(_p("A", 9, 3, t_call=0.0), elevator_id="E2")
        res = coord.assign_call(p, t_now=0.0)
        assert res.chosen_elevator_id == "E2"

    def test_stop_counts_refreshed(self):
        coord, s1, s2 = self._make_fleet()
        p = _p("A", 5, 8, t_call=0.0)
        coord.assign_call(p, t_now=0.0)
        # After assignment, building_stop_counts are dicts[int, int]
        assert isinstance(s1.building_stop_counts, dict)
        assert isinstance(s2.building_stop_counts, dict)

    def test_reoptimize_all_returns_results(self):
        coord, s1, s2 = self._make_fleet()
        results = coord.reoptimize_all(t_now=0.0)
        assert set(results.keys()) == {"E1", "E2"}


# ---------------------------------------------------------------------------
# Pipeline integration smoke test
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_pipeline_returns_valid_route(self):
        p1 = _p("A", 2, 7)
        p2 = _p("B", 5, 1)
        state    = _state([p1, p2], floor=1.0)
        pipeline = (
            OptimizationPipeline()
            .add_heuristic(BeamSearchHeuristic(beam_width=10))
            .add_heuristic(NearestNeighbourHeuristic())
            .set_objective(
                CompositeObjective()
                .add(IQRObjective(), weight=1.0)
                .add(MeanTimeObjective(), weight=0.1)
            )
            .add_constraint(WeightCapacityConstraint(max_kg=400.0))
            .add_constraint(FloorRangeConstraint(1, 10))
        )
        result = pipeline.optimize(state)
        assert result.best_score < float("inf")
        assert is_valid_route(result.best_route)

    def test_pipeline_with_no_passengers_returns_empty_route(self):
        state    = _state([], floor=1.0)
        pipeline = _simple_pipeline()
        result   = pipeline.optimize(state)
        assert result.best_route.stops == []
        assert result.best_score == 0.0
