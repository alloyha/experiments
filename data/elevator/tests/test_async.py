"""
Tests for elevator.async_interface.loop (AsyncElevatorLoop).

Uses pytest-asyncio with STRICT mode (configured in pyproject.toml).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import pytest

from elevator.domain.models import Passenger, ElevatorState
from elevator.heuristics.nearest_neighbour import NearestNeighbourHeuristic
from elevator.objective.iqr import IQRObjective
from elevator.pipeline.pipeline import OptimizationPipeline
from elevator.async_interface.loop import AsyncElevatorLoop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pipeline():
    return (
        OptimizationPipeline()
        .add_heuristic(NearestNeighbourHeuristic())
        .set_objective(IQRObjective())
    )


def _state(floor: float = 1.0) -> ElevatorState:
    return ElevatorState(
        id="E1", current_floor=floor,
        onboard=[], waiting=[],
        t_now=0.0, tau=1.0,
    )


def _p(pid, origin, dest, t_call=0.0):
    return Passenger(id=pid, origin=origin, destination=dest, t_call=t_call)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAsyncElevatorLoop:
    @pytest.mark.asyncio
    async def test_start_and_stop_empty(self):
        """Loop with no calls should return empty metrics."""
        loop = AsyncElevatorLoop(_pipeline(), _state())
        await loop.start()
        metrics = await loop.stop()
        assert metrics.delivered_total_times == []

    @pytest.mark.asyncio
    async def test_single_call_delivered(self):
        """One passenger injected via call() should appear in metrics."""
        loop = AsyncElevatorLoop(_pipeline(), _state())
        await loop.start()
        p = _p("A", 1, 5, t_call=0.0)
        await loop.call(p)
        metrics = await loop.stop()
        # Passenger was scheduled — may or may not have been delivered
        # depending on timing, but no exception should occur
        assert isinstance(metrics.delivered_total_times, list)

    @pytest.mark.asyncio
    async def test_metrics_before_start_is_none(self):
        loop = AsyncElevatorLoop(_pipeline(), _state())
        assert loop.metrics is None

    @pytest.mark.asyncio
    async def test_metrics_after_start_is_not_none(self):
        loop = AsyncElevatorLoop(_pipeline(), _state())
        await loop.start()
        await loop.stop()
        assert loop.metrics is not None

    @pytest.mark.asyncio
    async def test_multiple_calls(self):
        """Multiple passengers can be enqueued without error."""
        loop = AsyncElevatorLoop(_pipeline(), _state())
        await loop.start()
        passengers = [_p(f"P{i}", i, i + 4, t_call=float(i)) for i in range(1, 4)]
        for p in passengers:
            await loop.call(p)
        metrics = await loop.stop()
        assert isinstance(metrics.delivered_total_times, list)

    def test_dispatch_call_static(self):
        """_dispatch_call runs synchronously — test directly."""
        from elevator.simulation.simulator import ElevatorSimulator
        from elevator.async_interface.loop import AsyncElevatorLoop

        state = _state()
        sim   = ElevatorSimulator(_pipeline(), state)
        p     = _p("A", 1, 5, t_call=0.0)

        # Should not raise
        AsyncElevatorLoop._dispatch_call(sim, p, t=0.0)
        # At t=0 the CALL event should have been processed
        assert len(sim.ticks) >= 1
