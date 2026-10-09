"""
Tests for elevator.config.loader.

Uses tmp_path (pytest built-in) to write real TOML files then loads them.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pathlib import Path

from elevator.config.loader import (
    load_pipeline,
    load_elevator_state,
    configure_logging,
    _build_heuristic,
    _build_objective,
    _build_constraint,
)
from elevator.pipeline.pipeline import OptimizationPipeline
from elevator.domain.models import ElevatorState


# ---------------------------------------------------------------------------
# TOML fixtures
# ---------------------------------------------------------------------------

MINIMAL_TOML = """\
[pipeline]
fallback = "nearest_neighbour"

[[pipeline.heuristics]]
type = "nearest_neighbour"

[pipeline.objective]
type = "iqr"
fallback = "range"

[elevator]
id = "E1"
start_floor = 2
tau = 1.5

[logging]
level = "WARNING"
"""

FULL_TOML = """\
[pipeline]
fallback = "beam_search"

[[pipeline.heuristics]]
type = "beam_search"
beam_width = 5

[[pipeline.heuristics]]
type = "scan"

[[pipeline.heuristics]]
type = "exhaustive"
max_stops = 6

[[pipeline.heuristics]]
type = "simulated_annealing"
max_iter = 100
initial_temp = 50.0
cooling_rate = 0.99
seed = 42

[[pipeline.heuristics]]
type = "incremental"

[pipeline.objective]
type = "composite"

[[pipeline.objective.terms]]
type = "iqr"
weight = 1.0
fallback = "stdev"

[[pipeline.objective.terms]]
type = "mean"
weight = 0.5

[[pipeline.constraints]]
type = "weight_capacity"
max_kg = 400.0

[[pipeline.constraints]]
type = "passenger_capacity"
max_pax = 6

[[pipeline.constraints]]
type = "floor_range"
min_floor = 1
max_floor = 20

[[pipeline.constraints]]
type = "no_reverse"
reversal_cost = 0.5

[[pipeline.constraints]]
type = "min_stop_gap"
min_gap = 1

[[pipeline.constraints]]
type = "time_window"
default_max_wait = 120.0

[[pipeline.constraints]]
type = "vip"
vip_penalty = 1000.0
hard = false

[[pipeline.constraints]]
type = "maintenance_floor"
maintenance_floor = 1
interval_floors = 100.0
penalty_per_floor = 1.0

[[pipeline.constraints]]
type = "peak_hour"
peak_max_pax = 4
off_peak_max_pax = 8
direction_penalty = 5.0
peak_windows = [[28800, 36000]]

[[pipeline.constraints]]
type = "shafts_per_floor"
max_shafts = 2

[elevator]
id = "E2"
start_floor = 1
tau = 1.0

[logging]
level = "DEBUG"
format = "%(levelname)s: %(message)s"
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoader:
    def test_load_pipeline_minimal(self, tmp_path: Path):
        f = tmp_path / "elevator.toml"
        f.write_text(MINIMAL_TOML)
        pipeline = load_pipeline(f)
        assert isinstance(pipeline, OptimizationPipeline)

    def test_load_pipeline_full(self, tmp_path: Path):
        f = tmp_path / "elevator.toml"
        f.write_text(FULL_TOML)
        pipeline = load_pipeline(f)
        assert isinstance(pipeline, OptimizationPipeline)

    def test_load_elevator_state(self, tmp_path: Path):
        f = tmp_path / "elevator.toml"
        f.write_text(MINIMAL_TOML)
        state = load_elevator_state(f)
        assert isinstance(state, ElevatorState)
        assert state.id == "E1"
        assert state.current_floor == pytest.approx(2.0)
        assert state.tau == pytest.approx(1.5)

    def test_configure_logging(self, tmp_path: Path):
        f = tmp_path / "elevator.toml"
        f.write_text(MINIMAL_TOML)
        # Should not raise
        configure_logging(f)

    def test_load_pipeline_full_elevator_state(self, tmp_path: Path):
        f = tmp_path / "elevator.toml"
        f.write_text(FULL_TOML)
        state = load_elevator_state(f)
        assert state.id == "E2"

    # -- _build_heuristic ---------------------------------------------------

    def test_build_heuristic_all_types(self):
        types = [
            {"type": "nearest_neighbour"},
            {"type": "beam_search", "beam_width": 10},
            {"type": "scan"},
            {"type": "exhaustive", "max_stops": 8},
            {"type": "exhaustive_bnb"},
            {"type": "simulated_annealing"},
            {"type": "incremental"},
        ]
        for cfg in types:
            h = _build_heuristic(cfg)
            assert h is not None

    def test_build_heuristic_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown heuristic"):
            _build_heuristic({"type": "unicorn"})

    # -- _build_objective ---------------------------------------------------

    def test_build_objective_iqr(self):
        obj = _build_objective({"type": "iqr", "fallback": "mean"})
        assert obj is not None

    def test_build_objective_mean(self):
        obj = _build_objective({"type": "mean"})
        assert obj is not None

    def test_build_objective_composite(self):
        cfg = {
            "type": "composite",
            "terms": [
                {"type": "iqr", "weight": 1.0},
                {"type": "mean", "weight": 0.5},
            ],
        }
        obj = _build_objective(cfg)
        assert obj is not None

    def test_build_objective_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown objective"):
            _build_objective({"type": "banana"})

    # -- _build_constraint --------------------------------------------------

    def test_build_all_constraint_types(self):
        cfgs = [
            {"type": "weight_capacity", "max_kg": 400.0},
            {"type": "passenger_capacity", "max_pax": 6},
            {"type": "floor_range", "min_floor": 1, "max_floor": 20},
            {"type": "no_reverse"},
            {"type": "min_stop_gap"},
            {"type": "time_window"},
            {"type": "vip"},
            {"type": "maintenance_floor"},
            {"type": "peak_hour", "peak_windows": [[28800, 36000]]},
            {"type": "shafts_per_floor", "max_shafts": 2},
        ]
        for cfg in cfgs:
            c = _build_constraint(cfg)
            assert c is not None

    def test_build_constraint_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown constraint"):
            _build_constraint({"type": "flying_car"})
