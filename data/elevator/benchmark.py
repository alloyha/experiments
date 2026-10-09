"""
Benchmark harness for the elevator optimization pipeline.

Sweeps over passenger counts, building heights, and call rates, runs every
heuristic configuration and emits a CSV + summary table to stdout.

Usage
-----
    python3 benchmark.py               # default sweep (may take ~30-60s)
    python3 benchmark.py --quick       # small sweep for CI / smoke test
    python3 benchmark.py --csv results.csv

Output columns
--------------
n_passengers  n_floors  call_rate  heuristic
iqr_mean  iqr_std  mean_total  mean_wait  elapsed_ms
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import statistics
import sys
import time
from dataclasses import dataclass, field

from elevator.domain.models import ElevatorState, Passenger
from elevator.heuristics.heuristics import (
    NearestNeighbourHeuristic,
    BeamSearchHeuristic,
    SCANHeuristic,
    ExhaustiveHeuristic,
    SimulatedAnnealingHeuristic,
    IncrementalHeuristic,
)
from elevator.objective.objective import IQRObjective, MeanTimeObjective, CompositeObjective
from elevator.pipeline.pipeline import OptimizationPipeline
from elevator.simulation.simulator import ElevatorSimulator, SimulationMetrics


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def generate_passengers(
    n: int,
    n_floors: int,
    call_rate: float,
    seed: int,
) -> list[Passenger]:
    """Generate ``n`` random passengers arriving at ``call_rate`` calls/second."""
    rng = random.Random(seed)
    passengers = []
    t = 0.0
    for i in range(n):
        # inter-arrival time ~ Exponential(call_rate)
        t += rng.expovariate(call_rate)
        origin = rng.randint(1, n_floors)
        dest   = rng.randint(1, n_floors)
        while dest == origin:
            dest = rng.randint(1, n_floors)
        passengers.append(Passenger(
            id          = f"P{i}",
            origin      = origin,
            destination = dest,
            t_call      = round(t, 3),
        ))
    return passengers


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

HEURISTIC_CONFIGS: list[tuple[str, object]] = [
    ("nearest_neighbour", NearestNeighbourHeuristic()),
    ("scan",              SCANHeuristic()),
    ("beam_10",           BeamSearchHeuristic(beam_width=10)),
    ("beam_20",           BeamSearchHeuristic(beam_width=20)),
    ("exhaustive_bnb",    ExhaustiveHeuristic(max_stops=10)),
    ("sim_annealing",     SimulatedAnnealingHeuristic(max_iter=500, seed=0)),
    ("incremental",       IncrementalHeuristic()),
]


def _make_pipeline(heuristic) -> OptimizationPipeline:
    return (
        OptimizationPipeline()
        .add_heuristic(heuristic)
        .set_objective(
            CompositeObjective()
            .add(IQRObjective(), weight=1.0)
            .add(MeanTimeObjective(), weight=0.1)
        )
    )


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    n_passengers : int
    n_floors     : int
    call_rate    : float
    heuristic    : str
    iqr_mean     : float
    iqr_std      : float
    mean_total   : float
    mean_wait    : float
    elapsed_ms   : float
    iqr_over_time: list[tuple[float, float]] = field(default_factory=list)


def run_scenario(
    passengers : list[Passenger],
    n_floors   : int,
    heuristic_name: str,
    heuristic,
    seed       : int,
    call_rate  : float,
) -> BenchResult:
    state    = ElevatorState(id="E1", current_floor=1.0, onboard=[],
                              waiting=[], t_now=0.0, tau=1.0)
    pipeline = _make_pipeline(heuristic)
    sim      = ElevatorSimulator(pipeline, state, verbose=False, seed=seed)

    for p in passengers:
        sim.schedule_call(t=p.t_call, passenger=p)

    t0      = time.perf_counter()
    metrics = sim.run()
    elapsed = (time.perf_counter() - t0) * 1000.0

    totals = metrics.delivered_total_times
    return BenchResult(
        n_passengers  = len(passengers),
        n_floors      = n_floors,
        call_rate     = call_rate,
        heuristic     = heuristic_name,
        iqr_mean      = statistics.mean(totals) if totals else 0.0,
        iqr_std       = statistics.stdev(totals) if len(totals) >= 2 else 0.0,
        mean_total    = metrics.mean_total,
        mean_wait     = metrics.mean_wait,
        elapsed_ms    = elapsed,
        iqr_over_time = metrics.iqr_over_time,
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

FULL_SWEEP = {
    "n_passengers": [5, 10, 20, 50],
    "n_floors"    : [5, 10, 20],
    "call_rates"  : [0.5, 1.0, 2.0],
    "seed"        : 42,
}

QUICK_SWEEP = {
    "n_passengers": [5, 10],
    "n_floors"    : [5, 10],
    "call_rates"  : [1.0],
    "seed"        : 42,
}


def run_sweep(cfg: dict, verbose: bool = True) -> list[BenchResult]:
    results: list[BenchResult] = []
    seed = cfg["seed"]

    combos = [
        (n, f, r)
        for n in cfg["n_passengers"]
        for f in cfg["n_floors"]
        for r in cfg["call_rates"]
    ]
    total = len(combos) * len(HEURISTIC_CONFIGS)
    done  = 0

    for n, n_floors, call_rate in combos:
        passengers = generate_passengers(n, n_floors, call_rate, seed=seed)
        for name, heuristic in HEURISTIC_CONFIGS:
            if verbose:
                print(f"  [{done+1:3}/{total}] n={n} floors={n_floors} "
                      f"rate={call_rate} heuristic={name} ...", end=" ", flush=True)
            result = run_scenario(
                passengers     = passengers,
                n_floors       = n_floors,
                heuristic_name = name,
                heuristic      = heuristic,
                seed           = seed,
                call_rate      = call_rate,
            )
            if verbose:
                print(f"mean_total={result.mean_total:.1f}s "
                      f"elapsed={result.elapsed_ms:.0f}ms")
            results.append(result)
            done += 1

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "n_passengers", "n_floors", "call_rate", "heuristic",
    "iqr_mean", "iqr_std", "mean_total", "mean_wait", "elapsed_ms",
]


def write_csv(results: list[BenchResult], dest: io.TextIOBase | None = None) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    for r in results:
        writer.writerow({
            "n_passengers": r.n_passengers,
            "n_floors"    : r.n_floors,
            "call_rate"   : r.call_rate,
            "heuristic"   : r.heuristic,
            "iqr_mean"    : f"{r.iqr_mean:.2f}",
            "iqr_std"     : f"{r.iqr_std:.2f}",
            "mean_total"  : f"{r.mean_total:.2f}",
            "mean_wait"   : f"{r.mean_wait:.2f}",
            "elapsed_ms"  : f"{r.elapsed_ms:.1f}",
        })
    csv_text = buf.getvalue()
    if dest:
        dest.write(csv_text)
    return csv_text


def print_summary(results: list[BenchResult]) -> None:
    """Print a condensed comparison table grouped by heuristic."""
    print("\n=== Heuristic Summary (averaged across all scenarios) ===\n")
    by_heuristic: dict[str, list[BenchResult]] = {}
    for r in results:
        by_heuristic.setdefault(r.heuristic, []).append(r)

    header = f"{'Heuristic':<20} {'mean_total':>10} {'mean_wait':>10} {'elapsed_ms':>12}"
    print(header)
    print("-" * len(header))
    for name, rs in sorted(by_heuristic.items()):
        mt  = statistics.mean(r.mean_total  for r in rs)
        mw  = statistics.mean(r.mean_wait   for r in rs)
        el  = statistics.mean(r.elapsed_ms  for r in rs)
        print(f"{name:<20} {mt:>10.1f} {mw:>10.1f} {el:>12.1f}")
    print()


def print_convergence(results: list[BenchResult], heuristic: str = "beam_20") -> None:
    """Print IQR convergence over simulation time for one heuristic in one scenario."""
    sample = next(
        (r for r in results if r.heuristic == heuristic and r.n_passengers >= 10),
        None,
    )
    if sample is None:
        return
    print(f"\n=== IQR over time ({heuristic}, "
          f"n={sample.n_passengers}, floors={sample.n_floors}) ===\n")
    print(f"  {'t (s)':>8}  {'iqr (s)':>8}")
    for t, iqr in sample.iqr_over_time:
        print(f"  {t:>8.1f}  {iqr:>8.2f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Elevator optimizer benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Run a smaller sweep (fast, for CI)")
    parser.add_argument("--csv", metavar="FILE",
                        help="Write full results to a CSV file")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-run output")
    args = parser.parse_args()

    cfg     = QUICK_SWEEP if args.quick else FULL_SWEEP
    verbose = not args.quiet

    print(f"Running {'quick' if args.quick else 'full'} benchmark sweep…")
    print(f"  {len(cfg['n_passengers'])} passenger counts × "
          f"{len(cfg['n_floors'])} floor counts × "
          f"{len(cfg['call_rates'])} call rates × "
          f"{len(HEURISTIC_CONFIGS)} heuristics\n")

    results = run_sweep(cfg, verbose=verbose)

    print_summary(results)
    print_convergence(results)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            write_csv(results, f)
        print(f"CSV written to {args.csv}")
    else:
        print("--- CSV (stdout) ---")
        print(write_csv(results))


if __name__ == "__main__":
    main()
