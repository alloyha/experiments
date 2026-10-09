"""
Coverage-boosting tests.

Targets every uncovered line reported by pytest-cov across all modules
except async_interface/loop.py (see test_async.py) and
config/loader.py (see test_config.py).
"""
from __future__ import annotations

import sys, os, dataclasses, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from elevator.domain.models import (
    Passenger, ElevatorState, Route, Stop, StopType, Event, EventType,
)
from elevator.constraints.base import Constraint, ConstraintRegistry
from elevator.constraints.capacity import (
    WeightCapacityConstraint, PassengerCapacityConstraint, ShaftsPerFloorConstraint,
)
from elevator.constraints.routing import (
    FloorRangeConstraint, NoReverseConstraint, MinStopGapConstraint,
)
from elevator.constraints.policy import (
    TimeWindowConstraint, VIPConstraint, MaintenanceFloorConstraint, PeakHourConstraint,
)
from elevator.objective.base import RouteProjector
from elevator.objective.iqr import IQRObjective, IQRFallbackPolicy
from elevator.objective.composite import MeanTimeObjective, CompositeObjective
from elevator.heuristics.nearest_neighbour import NearestNeighbourHeuristic
from elevator.heuristics.beam_search import BeamSearchHeuristic
from elevator.heuristics.scan import SCANHeuristic
from elevator.heuristics.exhaustive import ExhaustiveHeuristic, _partial_travel_cost
from elevator.heuristics.simulated_annealing import SimulatedAnnealingHeuristic
from elevator.heuristics.incremental import IncrementalHeuristic
from elevator.pipeline.pipeline import OptimizationPipeline, OptimizationResult
from elevator.simulation.simulator import ElevatorSimulator, SimulationMetrics
from elevator.fleet.coordinator import FleetCoordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p(pid, origin, dest, t_call=0.0, weight=70.0, max_wait=None, is_vip=False):
    return Passenger(id=pid, origin=origin, destination=dest, t_call=t_call,
                     weight_kg=weight, max_wait_time=max_wait, is_vip=is_vip)


def _state(waiting=None, onboard=None, floor=1.0, tau=1.0, t_now=0.0):
    return ElevatorState(
        id="E1", current_floor=floor,
        onboard=onboard or [], waiting=waiting or [],
        t_now=t_now, tau=tau,
    )


def _route(*stops):
    return Route([Stop(floor=f, stop_type=st, passenger_id=pid) for f, st, pid in stops])


# ---------------------------------------------------------------------------
# domain/models.py  (lines 60, 64, 122, 128, 179)
# ---------------------------------------------------------------------------

class TestModels:
    def test_passenger_is_waiting(self):
        p = _p("A", 1, 5)
        assert p.is_waiting is True
        assert p.is_onboard is False

    def test_passenger_is_onboard(self):
        p = _p("A", 1, 5)
        boarded = p.with_embark(1.0)
        assert boarded.is_onboard is True
        assert boarded.is_waiting is False

    def test_route_len_and_floors(self):
        r = _route((2, StopType.EMBARK, "A"), (5, StopType.DISEMBARK, "A"))
        assert len(r) == 2
        assert r.floors() == [2, 5]

    def test_state_travel_time_to(self):
        s = _state(floor=3.0, tau=2.0)
        assert s.travel_time_to(7) == pytest.approx(8.0)   # 4 floors * 2s


# ---------------------------------------------------------------------------
# constraints/base.py  (lines 43, 46, 69, 74-75, 78, 81, 84, 99)
# ---------------------------------------------------------------------------

class TestConstraintBase:
    class _AlwaysFeasible(Constraint):
        """Concrete test double — always check=True, no override of penalty."""
        @property
        def name(self): return "always_ok"
        def check(self, route, state): return True

    class _AlwaysInfeasible(Constraint):
        """Concrete test double — check=False, default penalty = inf."""
        @property
        def name(self): return "always_fail"
        def check(self, route, state): return False

    def test_default_penalty_feasible(self):
        c = self._AlwaysFeasible()
        r, s = Route([]), _state()
        assert c.penalty(r, s) == 0.0

    def test_default_penalty_infeasible(self):
        c = self._AlwaysInfeasible()
        r, s = Route([]), _state()
        assert c.penalty(r, s) == float("inf")

    def test_repr(self):
        c = self._AlwaysFeasible()
        assert "always_ok" in repr(c)

    def test_registry_remove_and_get(self):
        c = self._AlwaysFeasible()
        reg = ConstraintRegistry()
        reg.add(c)
        assert reg.get("always_ok") is c
        assert len(reg) == 1
        reg.remove("always_ok")
        assert reg.get("always_ok") is None
        assert len(reg) == 0

    def test_registry_duplicate_raises(self):
        c = self._AlwaysFeasible()
        reg = ConstraintRegistry()
        reg.add(c)
        with pytest.raises(ValueError):
            reg.add(self._AlwaysFeasible())

    def test_registry_check_all_and_violations(self):
        reg = ConstraintRegistry()
        reg.add(self._AlwaysFeasible())
        reg.add(self._AlwaysInfeasible())
        r, s = Route([]), _state()
        assert not reg.check_all(r, s)
        assert "always_fail" in reg.violations(r, s)

    def test_registry_total_penalty(self):
        reg = ConstraintRegistry()
        reg.add(self._AlwaysFeasible())
        reg.add(self._AlwaysInfeasible())
        r, s = Route([]), _state()
        assert reg.total_penalty(r, s) == float("inf")


# ---------------------------------------------------------------------------
# constraints/capacity.py  (lines 55, 66-73, 87)
# ---------------------------------------------------------------------------

class TestCapacityExtra:
    def test_passenger_capacity_penalty(self):
        p1, p2 = _p("A", 1, 5), _p("B", 2, 6)
        state  = _state([p1, p2])
        route  = _route(
            (1, StopType.EMBARK, "A"), (2, StopType.EMBARK, "B"),
            (5, StopType.DISEMBARK, "A"), (6, StopType.DISEMBARK, "B"),
        )
        # max_pax=1 → 1 excess at peak
        pen = PassengerCapacityConstraint(max_pax=1).penalty(route, state)
        assert pen == pytest.approx(1.0)

    def test_shafts_per_floor_check_false(self):
        p     = _p("A", 3, 7)
        state = ElevatorState(
            id="E1", current_floor=1.0, onboard=[], waiting=[p],
            t_now=0.0, tau=1.0,
            building_stop_counts={3: 2},
        )
        route = _route((3, StopType.EMBARK, "A"), (7, StopType.DISEMBARK, "A"))
        assert not ShaftsPerFloorConstraint(max_shafts=2).check(route, state)

    def test_shafts_per_floor_penalty(self):
        p     = _p("A", 3, 7)
        state = ElevatorState(
            id="E1", current_floor=1.0, onboard=[], waiting=[p],
            t_now=0.0, tau=1.0,
            building_stop_counts={3: 2},
        )
        route = _route((3, StopType.EMBARK, "A"), (7, StopType.DISEMBARK, "A"))
        # penalty=inf because check=False and default penalty used
        pen = ShaftsPerFloorConstraint(max_shafts=2).penalty(route, state)
        assert pen == float("inf")


# ---------------------------------------------------------------------------
# constraints/routing.py  (lines 36, 39, 56, 60, 63-64)
# ---------------------------------------------------------------------------

class TestRoutingExtra:
    def test_no_reverse_check_always_true(self):
        r, s = Route([]), _state()
        assert NoReverseConstraint().check(r, s) is True

    def test_no_reverse_penalty_empty_route(self):
        r, s = Route([]), _state()
        assert NoReverseConstraint().penalty(r, s) == 0.0

    def test_min_stop_gap_satisfied(self):
        r = _route((1, StopType.EMBARK, "A"), (3, StopType.DISEMBARK, "A"))
        s = _state([_p("A", 1, 3)])
        assert MinStopGapConstraint(min_gap=2).check(r, s) is True

    def test_min_stop_gap_violated(self):
        r = _route((1, StopType.EMBARK, "A"), (2, StopType.DISEMBARK, "A"))
        s = _state([_p("A", 1, 2)])
        assert MinStopGapConstraint(min_gap=2).check(r, s) is False

    def test_min_stop_gap_single_stop(self):
        # Single stop — no pair, should be trivially satisfied
        r = _route((5, StopType.EMBARK, "A"))
        s = _state([_p("A", 5, 9)])
        assert MinStopGapConstraint(min_gap=3).check(r, s) is True


# ---------------------------------------------------------------------------
# constraints/policy.py  (many lines)
# ---------------------------------------------------------------------------

class TestTimeWindowExtra:
    def test_penalty_within_deadline_is_zero(self):
        p     = _p("A", 1, 5, t_call=0.0, max_wait=10.0)
        state = _state([p], floor=1.0)
        route = _route((1, StopType.EMBARK, "A"), (5, StopType.DISEMBARK, "A"))
        assert TimeWindowConstraint().penalty(route, state) == pytest.approx(0.0)

    def test_penalty_proportional_to_excess(self):
        p     = _p("A", 10, 1, t_call=0.0, max_wait=5.0)
        state = _state([p], floor=1.0)
        route = _route((10, StopType.EMBARK, "A"), (1, StopType.DISEMBARK, "A"))
        # 9 floors travel → embark at t=9; deadline = 0+5=5; excess = 4
        pen = TimeWindowConstraint().penalty(route, state)
        assert pen == pytest.approx(4.0)

    def test_check_uses_default_when_no_per_passenger_limit(self):
        p     = _p("A", 10, 1, t_call=0.0)  # max_wait_time = None
        state = _state([p], floor=1.0)
        route = _route((10, StopType.EMBARK, "A"), (1, StopType.DISEMBARK, "A"))
        assert not TimeWindowConstraint(default_max_wait=3.0).check(route, state)

    def test_check_passenger_not_in_route_fails(self):
        p     = _p("A", 10, 1, t_call=0.0, max_wait=100.0)
        state = _state([p], floor=1.0)
        # Route doesn't embark A → projected time is None → check fails
        route = Route([])
        assert not TimeWindowConstraint(default_max_wait=100.0).check(route, state)


class TestVIPExtra:
    def test_hard_vip_check_fails_when_non_vip_first(self):
        vip  = _p("V", 3, 5, is_vip=True)
        norm = _p("N", 1, 7, is_vip=False)
        state = _state([vip, norm])
        route = _route(
            (1, StopType.EMBARK, "N"), (3, StopType.EMBARK, "V"),
            (5, StopType.DISEMBARK, "V"), (7, StopType.DISEMBARK, "N"),
        )
        assert not VIPConstraint(hard=True).check(route, state)

    def test_soft_vip_check_always_true(self):
        state = _state([_p("V", 1, 5, is_vip=True)])
        route = _route((1, StopType.EMBARK, "V"), (5, StopType.DISEMBARK, "V"))
        assert VIPConstraint(hard=False).check(route, state) is True


class TestMaintenanceFloorConstraint:
    def test_no_penalty_when_visited(self):
        p     = _p("A", 1, 3)
        state = ElevatorState(
            id="E1", current_floor=1.0, onboard=[], waiting=[p],
            t_now=0.0, tau=1.0, cumulative_travel=90.0,
        )
        # Route passes through floor 1 (maintenance floor) — resets counter
        route = _route((1, StopType.EMBARK, "A"), (3, StopType.DISEMBARK, "A"))
        pen = MaintenanceFloorConstraint(
            maintenance_floor=1, interval_floors=100.0
        ).penalty(route, state)
        assert pen == pytest.approx(0.0)

    def test_penalty_when_not_visited_and_overdue(self):
        p     = _p("A", 5, 9)
        state = ElevatorState(
            id="E1", current_floor=5.0, onboard=[], waiting=[p],
            t_now=0.0, tau=1.0, cumulative_travel=95.0,
        )
        # Route visits floors 5 and 9 — skips maintenance floor 1
        # cumulative = 95 + 4 = 99 at disembark; 99 % 100 = 99
        # after full route: floors_since_maint = 95%100 + 4 = 3 + 4 = 7 < 100 → NO penalty
        # Need cumulative_travel such that route pushes past interval
        state2 = ElevatorState(
            id="E1", current_floor=5.0, onboard=[], waiting=[p],
            t_now=0.0, tau=1.0, cumulative_travel=96.0,
        )
        route = _route((5, StopType.EMBARK, "A"), (9, StopType.DISEMBARK, "A"))
        # floors_since_maint at start = 96 % 100 = 96; route adds 4 → 100; > interval=100? No (==100)
        # Let's force it: cumulative_travel=97 → 97%100=97 + 4=101 > 100
        state3 = ElevatorState(
            id="E1", current_floor=5.0, onboard=[], waiting=[p],
            t_now=0.0, tau=1.0, cumulative_travel=97.0,
        )
        pen = MaintenanceFloorConstraint(
            maintenance_floor=1, interval_floors=100.0, penalty_per_floor=2.0
        ).penalty(route, state3)
        assert pen > 0.0

    def test_check_always_true(self):
        r, s = Route([]), _state()
        assert MaintenanceFloorConstraint().check(r, s) is True


class TestPeakHourExtra:
    PEAK = [(8 * 3600, 10 * 3600)]

    def _peak_state(self, waiting, t_now):
        return ElevatorState(
            id="E1", current_floor=1.0, onboard=[], waiting=waiting,
            t_now=t_now, tau=1.0,
        )

    def test_off_peak_no_penalty(self):
        p     = _p("A", 1, 5)
        state = self._peak_state([p], t_now=6 * 3600.0)   # 06:00 — off peak
        route = _route((1, StopType.EMBARK, "A"), (5, StopType.DISEMBARK, "A"))
        c = PeakHourConstraint(peak_windows=self.PEAK, direction_penalty=10.0)
        assert c.penalty(route, state) == pytest.approx(0.0)

    def test_peak_direction_penalty(self):
        p1, p2 = _p("A", 5, 1), _p("B", 3, 7)
        state  = self._peak_state([p1, p2], t_now=9 * 3600.0)   # 09:00 — peak
        route  = _route(
            (5, StopType.EMBARK, "A"), (1, StopType.DISEMBARK, "A"),
            (3, StopType.EMBARK, "B"), (7, StopType.DISEMBARK, "B"),
        )
        c = PeakHourConstraint(
            peak_windows=self.PEAK, peak_max_pax=4, direction_penalty=5.0
        )
        pen = c.penalty(route, state)
        assert pen > 0.0   # reversal penalty applies

    def test_off_peak_check_uses_off_peak_limit(self):
        passengers = [_p(f"P{i}", i, i + 5) for i in range(1, 10)]
        state = self._peak_state(passengers, t_now=6 * 3600.0)
        stops = []
        for p in passengers:
            stops.append(Stop(p.origin, StopType.EMBARK, p.id))
        for p in passengers:
            stops.append(Stop(p.destination, StopType.DISEMBARK, p.id))
        route = Route(stops)
        # off-peak limit = 8, trying to embark 9 → check fails
        c = PeakHourConstraint(
            peak_windows=self.PEAK, peak_max_pax=4, off_peak_max_pax=8
        )
        assert not c.check(route, state)

    def test_is_peak(self):
        c = PeakHourConstraint(peak_windows=self.PEAK)
        assert c._is_peak(9 * 3600.0) is True
        assert c._is_peak(11 * 3600.0) is False


# ---------------------------------------------------------------------------
# objective/iqr.py  (lines 58, 80-84)
# ---------------------------------------------------------------------------

class TestIQRExtra:
    def test_single_passenger_returns_zero(self):
        p     = _p("A", 1, 5)
        state = _state([p], floor=1.0)
        route = _route((1, StopType.EMBARK, "A"), (5, StopType.DISEMBARK, "A"))
        # len(times) == 1 < 2 → return 0.0
        assert IQRObjective().evaluate(route, state) == pytest.approx(0.0)

    def test_fallback_stdev(self):
        p1 = _p("A", 1, 3)
        p2 = _p("B", 1, 7)
        p3 = _p("C", 1, 5)
        state = _state([p1, p2, p3], floor=1.0)
        route = _route(
            (1, StopType.EMBARK, "A"), (3, StopType.DISEMBARK, "A"),
            (1, StopType.EMBARK, "B"), (7, StopType.DISEMBARK, "B"),
            (1, StopType.EMBARK, "C"), (5, StopType.DISEMBARK, "C"),
        )
        score = IQRObjective(fallback=IQRFallbackPolicy.STDEV).evaluate(route, state)
        assert isinstance(score, float) and score >= 0.0

    def test_fallback_mean(self):
        p1 = _p("A", 1, 3)
        p2 = _p("B", 1, 7)
        p3 = _p("C", 1, 5)
        state = _state([p1, p2, p3], floor=1.0)
        route = _route(
            (1, StopType.EMBARK, "A"), (3, StopType.DISEMBARK, "A"),
            (1, StopType.EMBARK, "B"), (7, StopType.DISEMBARK, "B"),
            (1, StopType.EMBARK, "C"), (5, StopType.DISEMBARK, "C"),
        )
        score = IQRObjective(fallback=IQRFallbackPolicy.MEAN).evaluate(route, state)
        assert score > 0.0

    def test_fallback_zero(self):
        p1 = _p("A", 1, 3)
        p2 = _p("B", 1, 7)
        p3 = _p("C", 1, 5)
        state = _state([p1, p2, p3], floor=1.0)
        route = _route(
            (1, StopType.EMBARK, "A"), (3, StopType.DISEMBARK, "A"),
            (1, StopType.EMBARK, "B"), (7, StopType.DISEMBARK, "B"),
            (1, StopType.EMBARK, "C"), (5, StopType.DISEMBARK, "C"),
        )
        assert IQRObjective(fallback=IQRFallbackPolicy.ZERO).evaluate(route, state) == 0.0


# ---------------------------------------------------------------------------
# objective/composite.py  (lines 19, 24, 43-44)
# ---------------------------------------------------------------------------

class TestCompositeExtra:
    def test_mean_time_objective_empty_returns_inf(self):
        p     = _p("A", 1, 5)
        state = _state([p])
        route = Route([])   # passenger never embarked → inf
        assert MeanTimeObjective().evaluate(route, state) == float("inf")

    def test_mean_time_objective_normal(self):
        p     = _p("A", 1, 5)
        state = _state([p], floor=1.0)
        route = _route((1, StopType.EMBARK, "A"), (5, StopType.DISEMBARK, "A"))
        score = MeanTimeObjective().evaluate(route, state)
        assert score == pytest.approx(4.0)

    def test_composite_name_and_evaluate(self):
        obj = (
            CompositeObjective()
            .add(IQRObjective(), weight=1.0)
            .add(MeanTimeObjective(), weight=0.5)
        )
        p     = _p("A", 1, 5)
        state = _state([p], floor=1.0)
        route = _route((1, StopType.EMBARK, "A"), (5, StopType.DISEMBARK, "A"))
        name  = obj.name
        assert "composite" in name
        score = obj.evaluate(route, state)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# heuristics (uncovered branches)
# ---------------------------------------------------------------------------

class TestHeuristicsExtra:
    def test_nearest_neighbour_name(self):
        assert NearestNeighbourHeuristic().name == "nearest_neighbour"

    def test_nearest_neighbour_onboard_passenger(self):
        """Onboard passenger only has DISEMBARK, not EMBARK in pool."""
        onboard_p = dataclasses.replace(_p("A", 1, 5), t_embark=0.0)
        state = ElevatorState(
            id="E1", current_floor=1.0,
            onboard=[onboard_p], waiting=[],
            t_now=0.0, tau=1.0,
        )
        routes = NearestNeighbourHeuristic().generate(state)
        assert len(routes) == 1
        floors = [s.floor for s in routes[0].stops]
        assert 5 in floors   # disembark at destination

    def test_beam_search_with_beam_node_no_available(self):
        """Trigger the branch where a node has no available stops mid-beam."""
        # Two passengers but beam completes one while other is pending
        p1 = _p("A", 1, 3)
        p2 = _p("B", 2, 5)
        state = _state([p1, p2], floor=1.0)
        routes = BeamSearchHeuristic(beam_width=1).generate(state)
        assert len(routes) >= 1

    def test_scan_fallback_no_stops(self):
        """SCAN with no passengers — result should be valid (empty or fallback)."""
        state = _state([], floor=1.0)
        routes = SCANHeuristic().generate(state)
        # Either a valid route or empty is acceptable
        assert isinstance(routes, list)

    def test_exhaustive_partial_cost_helper(self):
        """Directly test the standalone _partial_travel_cost function."""
        p     = _p("A", 1, 5)
        state = _state([p], floor=1.0)
        stops = [
            Stop(floor=3, stop_type=StopType.EMBARK, passenger_id="A"),
            Stop(floor=7, stop_type=StopType.DISEMBARK, passenger_id="A"),
        ]
        cost = _partial_travel_cost(stops, pos=1.0, state=state)
        assert cost == pytest.approx(2.0 + 4.0)  # 1→3 + 3→7

    def test_exhaustive_equal_cost_routes(self):
        """Trigger the equal-cost branch in BnB (cost == best_cost)."""
        # Single passenger, only one possible route — hits cost==best_cost path
        p     = _p("A", 1, 5)
        state = _state([p], floor=1.0)
        routes = ExhaustiveHeuristic(max_stops=10).generate(state)
        assert len(routes) >= 1

    def test_simulated_annealing_name(self):
        sa = SimulatedAnnealingHeuristic(max_iter=500)
        assert "simulated_annealing" in sa.name

    def test_simulated_annealing_short_stops(self):
        """Single passenger — only 2 stops, covers the `< 2` break guard."""
        p     = _p("A", 1, 5)
        state = _state([p], floor=1.0)
        routes = SimulatedAnnealingHeuristic(max_iter=50, seed=1).generate(state)
        assert len(routes) == 1

    def test_simulated_annealing_improvement_branch(self):
        """Run long enough to hit the best-cost improvement branch."""
        p1 = _p("A", 1, 8)
        p2 = _p("B", 3, 6)
        state = _state([p1, p2], floor=1.0)
        routes = SimulatedAnnealingHeuristic(
            max_iter=2000, initial_temp=200.0, seed=7
        ).generate(state)
        assert len(routes) == 1

    def test_simulated_annealing_empty_start(self):
        """No passengers → NearestNeighbour returns empty → SA returns it."""
        state = _state([], floor=1.0)
        routes = SimulatedAnnealingHeuristic(max_iter=10).generate(state)
        assert isinstance(routes, list)

    def test_incremental_name(self):
        assert IncrementalHeuristic().name == "incremental"

    def test_incremental_fallback_when_invalid(self):
        """Force incremental to produce invalid intermediate, triggering fallback."""
        # Start with an onboard passenger whose disembark is already in route
        onboard_p = dataclasses.replace(_p("A", 1, 5), t_embark=0.0)
        waiting_p = _p("B", 3, 7)
        existing_route = _route(
            (5, StopType.DISEMBARK, "A"),
        )
        state = ElevatorState(
            id="E1", current_floor=1.0,
            onboard=[onboard_p], waiting=[waiting_p],
            t_now=0.0, tau=1.0,
            current_route=existing_route,
        )
        routes = IncrementalHeuristic().generate(state)
        assert len(routes) >= 1


# ---------------------------------------------------------------------------
# pipeline/pipeline.py  (uncovered lines)
# ---------------------------------------------------------------------------

class TestPipelineExtra:
    def test_result_repr(self):
        p     = _p("A", 1, 5)
        state = _state([p], floor=1.0)
        pipeline = (
            OptimizationPipeline()
            .add_heuristic(NearestNeighbourHeuristic())
            .set_objective(IQRObjective())
        )
        result = pipeline.optimize(state)
        r = repr(result)
        assert "OptimizationResult" in r

    def test_set_fallback_none(self):
        """set_fallback(None) means no fallback — no exception raised."""
        pipeline = (
            OptimizationPipeline()
            .add_heuristic(NearestNeighbourHeuristic())
            .set_fallback(None)
        )
        p     = _p("A", 1, 5)
        state = _state([p], floor=1.0)
        result = pipeline.optimize(state)
        assert result.best_route is not None

    def test_fallback_used_when_all_routes_infeasible(self):
        """Hard constraint that rejects everything forces fallback heuristic."""
        from elevator.constraints.base import Constraint as _C

        class _RejectAll(_C):
            @property
            def name(self): return "reject_all"
            def check(self, route, state): return False

        pipeline = (
            OptimizationPipeline()
            .add_heuristic(NearestNeighbourHeuristic())
            .add_constraint(_RejectAll())
            .set_fallback(NearestNeighbourHeuristic())
        )
        p     = _p("A", 1, 5)
        state = _state([p], floor=1.0)
        result = pipeline.optimize(state)
        # Fallback should have been used
        assert "fallback" in result.heuristic_used

    def test_no_heuristics_uses_default(self):
        """Pipeline with no heuristics falls back to internal NearestNeighbour."""
        pipeline = OptimizationPipeline()
        p     = _p("A", 1, 5)
        state = _state([p], floor=1.0)
        result = pipeline.optimize(state)
        assert len(result.best_route.stops) > 0

    def test_ensure_passengers_covered_adds_missing_disembark(self):
        """Onboard passenger whose disembark is not in route gets it appended."""
        onboard_p = dataclasses.replace(_p("A", 1, 5), t_embark=0.0)
        state = ElevatorState(
            id="E1", current_floor=1.0,
            onboard=[onboard_p], waiting=[],
            t_now=0.0, tau=1.0,
        )
        # Heuristic that returns an empty route
        from elevator.constraints.base import Constraint as _C

        class _EmptyHeuristic:
            @property
            def name(self): return "empty"
            def generate(self, state): return [Route([])]

        pipeline = (
            OptimizationPipeline()
            .add_heuristic(_EmptyHeuristic())
            .set_fallback(None)
        )
        result = pipeline.optimize(state)
        pids = {s.passenger_id for s in result.best_route.stops}
        assert "A" in pids   # safety net appended the DISEMBARK


# ---------------------------------------------------------------------------
# simulation/simulator.py  (uncovered properties and branches)
# ---------------------------------------------------------------------------

class TestSimulatorExtra:
    def test_metrics_properties(self):
        m = SimulationMetrics()
        m.delivered_wait_times   = [2.0, 3.0]
        m.delivered_flight_times = [4.0, 5.0]
        m.delivered_total_times  = [6.0, 8.0]
        assert m.mean_flight == pytest.approx(4.5)
        assert m.mean_total  == pytest.approx(7.0)

    def test_summary_non_empty(self):
        m = SimulationMetrics()
        m.delivered_wait_times   = [1.0]
        m.delivered_flight_times = [3.0]
        m.delivered_total_times  = [4.0]
        text = m.summary()
        assert "Delivered" in text
        assert "Mean" in text

    def test_summary_empty(self):
        m = SimulationMetrics()
        assert "No passengers" in m.summary()

    def test_metrics_and_ticks_properties(self):
        p       = _p("A", 1, 5)
        state   = _state(floor=1.0)
        pipeline = (
            OptimizationPipeline()
            .add_heuristic(NearestNeighbourHeuristic())
        )
        sim = ElevatorSimulator(pipeline, state)
        sim.schedule_call(t=0.0, passenger=p)
        sim.run()
        # Access properties
        assert sim.metrics is not None
        assert isinstance(sim.ticks, list)

    def test_verbose_output(self, capsys):
        p       = _p("A", 1, 5)
        state   = _state(floor=1.0)
        pipeline = (
            OptimizationPipeline()
            .add_heuristic(NearestNeighbourHeuristic())
        )
        sim = ElevatorSimulator(pipeline, state, verbose=True)
        sim.schedule_call(t=0.0, passenger=p)
        sim.run()
        captured = capsys.readouterr()
        assert "CALL" in captured.out or "DISEMBARK" in captured.out

    def test_stale_event_dropped(self):
        """Schedule two CALL events so the second re-routes, making the first
        route's EMBARK/DISEMBARK events stale (wrong generation)."""
        p1 = _p("P1", 1, 10, t_call=0.0)
        p2 = _p("P2", 3, 8,  t_call=0.5)
        state    = _state(floor=1.0)
        pipeline = (
            OptimizationPipeline()
            .add_heuristic(BeamSearchHeuristic(beam_width=5))
        )
        sim = ElevatorSimulator(pipeline, state)
        sim.schedule_call(t=0.0,  passenger=p1)
        sim.schedule_call(t=0.5, passenger=p2)
        metrics = sim.run()
        assert len(metrics.delivered_total_times) == 2


# ---------------------------------------------------------------------------
# fleet/coordinator.py  (uncovered branches)
# ---------------------------------------------------------------------------

class TestFleetExtra:
    def test_no_elevators_raises(self):
        coord = FleetCoordinator(elevators=[], pipelines={})
        p = _p("A", 1, 5)
        with pytest.raises(RuntimeError):
            coord.assign_call(p, t_now=0.0)

    def test_missing_pipeline_warning_path(self):
        """Elevator has no pipeline → warning logged, defaulted to first."""
        s1 = ElevatorState(id="E1", current_floor=1.0, onboard=[], waiting=[],
                           t_now=0.0, tau=1.0)
        # No pipeline for E1
        coord = FleetCoordinator(elevators=[s1], pipelines={})
        p = _p("A", 1, 5)
        res = coord.assign_call(p, t_now=0.0)
        # E1 is the only elevator — must be chosen despite missing pipeline
        assert res.chosen_elevator_id == "E1"

    def test_reoptimize_skips_missing_pipeline(self):
        s1 = ElevatorState(id="E1", current_floor=1.0, onboard=[], waiting=[],
                           t_now=0.0, tau=1.0)
        coord = FleetCoordinator(elevators=[s1], pipelines={})
        results = coord.reoptimize_all(t_now=0.0)
        assert results == {}


# Additional targeted tests for remaining uncovered lines
class TestConstraintRegistryIter:
    """Test ConstraintRegistry.__iter__ (line 81)."""
    def test_iter_allows_for_loop_over_constraints(self):
        from elevator.constraints.base import ConstraintRegistry, Constraint
        
        class DummyConstraint(Constraint):
            def __init__(self, name_str: str):
                self._name = name_str
            
            @property
            def name(self):
                return self._name
            
            def check(self, route, state):
                return True
            
            def penalty(self, route, state):
                return 0.0
        
        reg = ConstraintRegistry()
        c1 = DummyConstraint("c1")
        c2 = DummyConstraint("c2")
        reg.add(c1)
        reg.add(c2)
        
        constraints_from_iter = list(reg)
        assert len(constraints_from_iter) == 2
        assert all(isinstance(c, DummyConstraint) for c in constraints_from_iter)


class TestFleetCoordinatorBuildingStopCounts:
    """Test FleetCoordinator._refresh_stop_counts (lines 182, 188)."""
    def test_building_stop_counts_multifloor_routes(self):
        p1 = Passenger("P1", origin=1, destination=5, t_call=0.0)
        p2 = Passenger("P2", origin=3, destination=7, t_call=0.0)
        
        s1 = ElevatorState(id="E1", current_floor=1.0,
                           onboard=[p1], waiting=[],
                           t_now=0.0, tau=1.0,
                           current_route=Route([
                               Stop(floor=5.0, stop_type=StopType.DISEMBARK, passenger_id="P1"),
                               Stop(floor=3.0, stop_type=StopType.EMBARK, passenger_id="P2"),
                           ]))
        s2 = ElevatorState(id="E2", current_floor=2.0,
                           onboard=[], waiting=[p2],
                           t_now=0.0, tau=1.0,
                           current_route=Route([
                               Stop(floor=3.0, stop_type=StopType.EMBARK, passenger_id="P2"),
                               Stop(floor=7.0, stop_type=StopType.DISEMBARK, passenger_id="P2"),
                           ]))
        
        coord = FleetCoordinator(elevators=[s1, s2], pipelines={})
        coord._refresh_stop_counts()
        
        assert s1.building_stop_counts.get(3, 0) >= 1
        assert s1.building_stop_counts.get(7, 0) == 1


class TestPipelineSafetyNetCompleteness:
    """Test pipeline safety net for incomplete routes (lines 60-61, 63-64)."""
    def test_ensure_all_passengers_adds_missing_embark(self):
        from elevator.pipeline.pipeline import _ensure_all_passengers_covered
        
        p1 = Passenger("P1", origin=1, destination=5, t_call=0.0)
        p2 = Passenger("P2", origin=2, destination=6, t_call=0.0)
        
        route = Route([Stop(floor=1.0, stop_type=StopType.EMBARK, passenger_id="P1")])
        
        state = ElevatorState(
            id="E1", current_floor=1.0,
            onboard=[], waiting=[p1, p2],
            t_now=0.0, tau=1.0
        )
        
        new_route = _ensure_all_passengers_covered(route, state)
        embark_ids = {s.passenger_id for s in new_route.stops
                      if s.stop_type == StopType.EMBARK}
        assert "P2" in embark_ids
    
    def test_ensure_all_passengers_adds_missing_disembark(self):
        from elevator.pipeline.pipeline import _ensure_all_passengers_covered
        
        p1 = Passenger("P1", origin=1, destination=5, t_call=0.0)
        
        route = Route([
            Stop(floor=1.0, stop_type=StopType.EMBARK, passenger_id="P1")
        ])
        
        state = ElevatorState(
            id="E1", current_floor=1.0,
            onboard=[p1], waiting=[],
            t_now=0.0, tau=1.0
        )
        
        new_route = _ensure_all_passengers_covered(route, state)
        disembark_ids = {s.passenger_id for s in new_route.stops
                         if s.stop_type == StopType.DISEMBARK}
        assert "P1" in disembark_ids


class TestHeuristicsNameProperties:
    """Test heuristic name property getters (exhaustive.py:43, scan.py:20)."""
    def test_exhaustive_name_property(self):
        from elevator.heuristics.exhaustive import ExhaustiveHeuristic
        h = ExhaustiveHeuristic()
        assert h.name == "exhaustive_bnb"
    
    def test_scan_name_property(self):
        from elevator.heuristics.scan import SCANHeuristic
        h = SCANHeuristic()
        assert h.name == "scan"


class TestExhaustiveOnboardDisembark:
    """Test exhaustive BnB with onboard passenger having disembark (lines 54-55)."""
    def test_bnb_onboard_passenger_with_disembark_mapping(self):
        from elevator.heuristics.exhaustive import ExhaustiveHeuristic
        
        p_onboard = Passenger("O1", origin=1, destination=5, t_call=0.0)
        p_waiting = Passenger("W1", origin=2, destination=6, t_call=0.0)
        
        state = ElevatorState(
            id="E1", current_floor=1.0,
            onboard=[p_onboard], waiting=[p_waiting],
            t_now=0.0, tau=1.0
        )
        
        h = ExhaustiveHeuristic(max_stops=10)
        routes = h.generate(state)
        
        # Should generate routes with both passengers
        assert len(routes) > 0
        assert any(any(s.passenger_id == "O1" for s in r.stops) for r in routes)


class TestExhaustiveEqualCostRoutes:
    """Test exhaustive BnB finding multiple equal-cost routes (lines 77-78)."""
    def test_bnb_equal_cost_branches_collected(self):
        from elevator.heuristics.exhaustive import ExhaustiveHeuristic
        
        # Create scenario where multiple permutations have same cost
        waiting = [Passenger(f"W{i}", origin=float(i), destination=float(i+5), t_call=0.0)
                   for i in range(2)]
        state = ElevatorState(
            id="E1", current_floor=1.0,
            onboard=[], waiting=waiting,
            t_now=0.0, tau=1.0
        )
        
        h = ExhaustiveHeuristic(max_stops=10)
        routes = h.generate(state)
        
        # Multiple routes can have equal cost
        assert len(routes) >= 1
        if len(routes) > 1:
            assert all(r.stops for r in routes)


class TestSimulatedAnnealingConvergence:
    """Test SA early convergence (simulated_annealing.py:61)."""
    def test_sa_convergence_with_low_iterations(self):
        from elevator.heuristics.simulated_annealing import SimulatedAnnealingHeuristic
        
        waiting = [Passenger("W1", origin=1, destination=5, t_call=0.0)]
        state = ElevatorState(
            id="E1", current_floor=1.0,
            onboard=[], waiting=waiting,
            t_now=0.0, tau=1.0
        )
        
        # Use very low iteration to test early exit patterns
        h = SimulatedAnnealingHeuristic(
            max_iter=2, 
            initial_temp=10.0,
            cooling_rate=0.99
        )
        routes = h.generate(state)
        assert len(routes) > 0


class TestSimulatedAnnealingBestUpdate:
    """Test SA updating best solution (simulated_annealing.py:86-87)."""
    def test_sa_tracks_best_solution(self):
        from elevator.heuristics.simulated_annealing import SimulatedAnnealingHeuristic
        
        waiting = [Passenger(f"W{i}", origin=float(i), destination=float(i+5), t_call=0.0)
                   for i in range(3)]
        state = ElevatorState(
            id="E1", current_floor=1.0,
            onboard=[], waiting=waiting,
            t_now=0.0, tau=1.0
        )
        
        h = SimulatedAnnealingHeuristic(
            max_iter=50,
            initial_temp=50.0,
            cooling_rate=0.95
        )
        routes = h.generate(state)
        assert len(routes) > 0
        assert all(isinstance(r, Route) for r in routes)




# Note: Remaining uncovered lines are edge cases in heuristics and simulator:
# - async_interface/loop.py:145 - stale event deduplication (generation mismatch)
# - exhaustive.py:93 - BnB cost pruning edge case (pragma: no cover)
# - simulator.py:349 - missing passenger error (pragma: no cover)
# These represent correctness guarantees and error paths that are difficult to reach
# in normal operation without artificially constructing test scenarios.
