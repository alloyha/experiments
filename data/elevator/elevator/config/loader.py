"""
TOML-driven pipeline assembly.

Reads a TOML config file and constructs a fully wired OptimizationPipeline
without touching Python code.  This allows operators to swap heuristics,
constraints, and objective weights via configuration alone.

Example config (elevator.toml)
-------------------------------
[pipeline]
fallback = "nearest_neighbour"

[[pipeline.heuristics]]
type = "beam_search"
beam_width = 20

[[pipeline.heuristics]]
type = "simulated_annealing"
max_iter = 1500
initial_temp = 80.0

[pipeline.objective]
type = "composite"

[[pipeline.objective.terms]]
type = "iqr"
weight = 1.0
fallback = "range"

[[pipeline.objective.terms]]
type = "mean"
weight = 0.1

[[pipeline.constraints]]
type = "weight_capacity"
max_kg = 400.0

[[pipeline.constraints]]
type = "passenger_capacity"
max_pax = 8

[[pipeline.constraints]]
type = "floor_range"
min_floor = 1
max_floor = 10

[[pipeline.constraints]]
type = "no_reverse"
reversal_cost = 0.5

[[pipeline.constraints]]
type = "time_window"
default_max_wait = 120.0

[elevator]
id = "E1"
start_floor = 1
tau = 1.0

[logging]
level = "INFO"
format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Any

# tomllib is stdlib in Python 3.11+; fall back to tomli for older versions
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    try:  # pragma: no cover
        import tomllib  # type: ignore[no-redef]  # pragma: no cover
    except ImportError:  # pragma: no cover
        try:  # pragma: no cover
            import tomli as tomllib  # type: ignore[no-redef]  # pragma: no cover
        except ImportError:  # pragma: no cover
            tomllib = None  # type: ignore[assignment]  # pragma: no cover

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
from elevator.domain.models import ElevatorState, Route
from elevator.heuristics.heuristics import (
    BeamSearchHeuristic,
    NearestNeighbourHeuristic,
    SCANHeuristic,
    ExhaustiveHeuristic,
    SimulatedAnnealingHeuristic,
    IncrementalHeuristic,
)
from elevator.objective.objective import (
    IQRObjective,
    IQRFallbackPolicy,
    MeanTimeObjective,
    CompositeObjective,
)
from elevator.pipeline.pipeline import OptimizationPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pipeline(path: str | Path) -> OptimizationPipeline:
    """Load and assemble a pipeline from a TOML config file."""
    cfg = _read_toml(path)
    return _build_pipeline(cfg.get("pipeline", {}))


def load_elevator_state(path: str | Path) -> ElevatorState:
    """Load initial ElevatorState from a TOML config file."""
    cfg  = _read_toml(path)
    ecfg = cfg.get("elevator", {})
    return ElevatorState(
        id            = ecfg.get("id", "E1"),
        current_floor = float(ecfg.get("start_floor", 1)),
        onboard       = [],
        waiting       = [],
        t_now         = 0.0,
        tau           = float(ecfg.get("tau", 1.0)),
    )


def configure_logging(path: str | Path) -> None:
    """Apply logging config from the TOML file."""
    cfg  = _read_toml(path)
    lcfg = cfg.get("logging", {})
    logging.basicConfig(
        level  = getattr(logging, lcfg.get("level", "WARNING").upper(), logging.WARNING),
        format = lcfg.get("format", "%(levelname)s %(name)s: %(message)s"),
    )


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _read_toml(path: str | Path) -> dict[str, Any]:
    if tomllib is None:  # pragma: no cover
        raise ImportError(
            "TOML support requires Python 3.11+ or 'tomli' package "
            "(pip install tomli)."
        )
    with open(path, "rb") as f:
        return tomllib.load(f)


def _build_pipeline(cfg: dict[str, Any]) -> OptimizationPipeline:
    pipeline = OptimizationPipeline()

    for hcfg in cfg.get("heuristics", []):
        pipeline.add_heuristic(_build_heuristic(hcfg))

    objective = _build_objective(cfg.get("objective", {}))
    pipeline.set_objective(objective)

    registry = ConstraintRegistry()
    for ccfg in cfg.get("constraints", []):
        registry.add(_build_constraint(ccfg))
    pipeline.set_constraints(registry)

    fallback_type = cfg.get("fallback", "nearest_neighbour")
    pipeline.set_fallback(_build_heuristic({"type": fallback_type}))

    return pipeline


def _build_heuristic(cfg: dict[str, Any]):
    t = cfg.get("type", "nearest_neighbour")
    if t == "beam_search":
        return BeamSearchHeuristic(beam_width=cfg.get("beam_width", 10))
    if t == "nearest_neighbour":
        return NearestNeighbourHeuristic()
    if t == "scan":
        return SCANHeuristic()
    if t == "exhaustive" or t == "exhaustive_bnb":
        return ExhaustiveHeuristic(max_stops=cfg.get("max_stops", 10))
    if t == "simulated_annealing":
        return SimulatedAnnealingHeuristic(
            initial_temp = cfg.get("initial_temp", 100.0),
            cooling_rate = cfg.get("cooling_rate", 0.995),
            max_iter     = cfg.get("max_iter", 2000),
            seed         = cfg.get("seed"),
        )
    if t == "incremental":
        return IncrementalHeuristic()
    raise ValueError(f"Unknown heuristic type: {t!r}")


def _build_objective(cfg: dict[str, Any]):
    t = cfg.get("type", "iqr")
    if t == "iqr":
        return IQRObjective(
            fallback=IQRFallbackPolicy(cfg.get("fallback", "range")),
        )
    if t == "mean":
        return MeanTimeObjective()
    if t == "composite":
        obj = CompositeObjective()
        for term in cfg.get("terms", []):
            weight = term.get("weight", 1.0)
            obj.add(_build_objective(term), weight=weight)
        return obj
    raise ValueError(f"Unknown objective type: {t!r}")


def _build_constraint(cfg: dict[str, Any]) -> any:
    t = cfg.get("type")
    if t == "weight_capacity":
        return WeightCapacityConstraint(max_kg=cfg["max_kg"])
    if t == "passenger_capacity":
        return PassengerCapacityConstraint(max_pax=cfg["max_pax"])
    if t == "floor_range":
        return FloorRangeConstraint(min_floor=cfg["min_floor"], max_floor=cfg["max_floor"])
    if t == "no_reverse":
        return NoReverseConstraint(reversal_cost=cfg.get("reversal_cost", 1.0))
    if t == "min_stop_gap":
        return MinStopGapConstraint(min_gap=cfg.get("min_gap", 1))
    if t == "time_window":
        return TimeWindowConstraint(default_max_wait=cfg.get("default_max_wait", float("inf")))
    if t == "vip":
        return VIPConstraint(
            vip_penalty=cfg.get("vip_penalty", 1000.0),
            hard=cfg.get("hard", False),
        )
    if t == "maintenance_floor":
        return MaintenanceFloorConstraint(
            maintenance_floor=cfg.get("maintenance_floor", 1),
            interval_floors=cfg.get("interval_floors", 100.0),
            penalty_per_floor=cfg.get("penalty_per_floor", 1.0),
        )
    if t == "peak_hour":
        return PeakHourConstraint(
            peak_windows=[(w[0], w[1]) for w in cfg.get("peak_windows", [])],
            peak_max_pax=cfg.get("peak_max_pax", 4),
            off_peak_max_pax=cfg.get("off_peak_max_pax", 8),
            direction_penalty=cfg.get("direction_penalty", 5.0),
        )
    if t == "shafts_per_floor":
        return ShaftsPerFloorConstraint(max_shafts=cfg["max_shafts"])
    raise ValueError(f"Unknown constraint type: {t!r}")
