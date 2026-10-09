"""
Example: demonstrates the elevator optimization pipeline end-to-end.

Scenario
--------
10-floor building, elevator starts at floor 1.
5 passengers call at different times with different origins/destinations.
Three heuristics compete: BeamSearch, NearestNeighbour, SCAN.
Constraints: weight capacity 400kg, no-reverse soft penalty.
Objective: IQR of total passenger time.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from elevator.domain.models import Passenger, ElevatorState, Route
from elevator.constraints.base import ConstraintRegistry
from elevator.constraints.builtin import (
    WeightCapacityConstraint,
    PassengerCapacityConstraint,
    NoReverseConstraint,
    FloorRangeConstraint,
)
from elevator.objective.objective import IQRObjective, CompositeObjective, MeanTimeObjective
from elevator.heuristics.heuristics import (
    BeamSearchHeuristic,
    NearestNeighbourHeuristic,
    SCANHeuristic,
    ExhaustiveHeuristic,
)
from elevator.pipeline.pipeline import OptimizationPipeline
from elevator.simulation.simulator import ElevatorSimulator


def build_pipeline(heuristics=None) -> OptimizationPipeline:
    # Constraints
    registry = (
        ConstraintRegistry()
        .add(WeightCapacityConstraint(max_kg=400.0))
        .add(PassengerCapacityConstraint(max_pax=8))
        .add(FloorRangeConstraint(min_floor=1, max_floor=10))
        .add(NoReverseConstraint(reversal_cost=0.5))  # soft
    )

    # Objective: IQR + small mean penalty
    objective = (
        CompositeObjective()
        .add(IQRObjective(fallback="range"), weight=1.0)
        .add(MeanTimeObjective(),            weight=0.1)
    )

    # Heuristics
    if heuristics is None:
        heuristics = [
            BeamSearchHeuristic(beam_width=20),
            NearestNeighbourHeuristic(),
            SCANHeuristic(),
        ]

    pipeline = OptimizationPipeline()
    for h in heuristics:
        pipeline.add_heuristic(h)
    pipeline.set_objective(objective)
    pipeline.set_constraints(registry)

    return pipeline


def run_scenario():
    print("=" * 60)
    print("ELEVATOR OPTIMIZER — Demo Scenario")
    print("=" * 60)

    # Initial elevator state: at floor 1, nobody aboard, t=0
    state = ElevatorState(
        id            = "E1",
        current_floor = 1.0,
        onboard       = [],
        waiting       = [],
        t_now         = 0.0,
        tau           = 1.0,  # 1 second per floor
    )

    pipeline  = build_pipeline()
    simulator = ElevatorSimulator(pipeline, state, verbose=True)

    # Schedule passenger calls
    passengers = [
        Passenger(id="P1", origin=3, destination=8,  t_call=0.0,  weight_kg=70),
        Passenger(id="P2", origin=7, destination=2,  t_call=1.5,  weight_kg=80),
        Passenger(id="P3", origin=1, destination=10, t_call=2.0,  weight_kg=65),
        Passenger(id="P4", origin=5, destination=3,  t_call=5.0,  weight_kg=90),
        Passenger(id="P5", origin=9, destination=4,  t_call=8.0,  weight_kg=75),
    ]

    for p in passengers:
        simulator.schedule_call(t=p.t_call, passenger=p)

    metrics = simulator.run()

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(metrics.summary())

    print("Per-tick optimization results:")
    for tick in simulator.ticks:
        print(f"  [t={tick.t:.2f}] {tick.event.event_type.value:<10} "
              f"score={tick.result.best_score:.3f}  "
              f"heuristic={tick.result.heuristic_used:<30}  "
              f"elapsed={tick.result.elapsed_ms:.1f}ms")

    return metrics


def compare_heuristics():
    """Run the same scenario with each heuristic in isolation and compare."""
    print()
    print("=" * 60)
    print("HEURISTIC COMPARISON")
    print("=" * 60)

    heuristic_sets = {
        "ExhaustiveSearch" : [ExhaustiveHeuristic(max_stops=12)],
        "BeamSearch(w=20)" : [BeamSearchHeuristic(beam_width=20)],
        "NearestNeighbour" : [NearestNeighbourHeuristic()],
        "SCAN"             : [SCANHeuristic()],
    }

    for label, heuristics in heuristic_sets.items():
        state = ElevatorState(
            id="E1", current_floor=1.0,
            onboard=[], waiting=[], t_now=0.0, tau=1.0,
        )
        pipeline  = build_pipeline(heuristics)
        simulator = ElevatorSimulator(pipeline, state, verbose=False)

        passengers = [
            Passenger(id="P1", origin=3, destination=8,  t_call=0.0,  weight_kg=70),
            Passenger(id="P2", origin=7, destination=2,  t_call=1.5,  weight_kg=80),
            Passenger(id="P3", origin=1, destination=10, t_call=2.0,  weight_kg=65),
            Passenger(id="P4", origin=5, destination=3,  t_call=5.0,  weight_kg=90),
            Passenger(id="P5", origin=9, destination=4,  t_call=8.0,  weight_kg=75),
        ]
        for p in passengers:
            simulator.schedule_call(t=p.t_call, passenger=p)

        metrics = simulator.run()
        print(f"\n{label}:")
        print(f"  IQR   = {metrics.iqr_total:.3f}s")
        print(f"  Mean  = {metrics.mean_total:.3f}s")
        print(f"  Delivered = {len(metrics.delivered_total_times)}")


if __name__ == "__main__":
    run_scenario()
    compare_heuristics()
