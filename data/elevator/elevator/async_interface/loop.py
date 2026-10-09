"""
Async real-time interface for the elevator optimizer.

Wraps the synchronous ElevatorSimulator in an asyncio event loop so
calls can arrive from external sources (API endpoints, button presses,
message queues) without blocking the optimization pipeline.

Usage
-----
    async def main():
        loop = AsyncElevatorLoop(pipeline, initial_state, verbose=True)
        await loop.start()

        # Inject calls from anywhere (e.g. a FastAPI endpoint):
        await loop.call(Passenger(id="P1", origin=3, destination=8, t_call=0.0))

        metrics = await loop.stop()
        print(metrics.summary())

    asyncio.run(main())
"""

from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from elevator.domain.models import ElevatorState, Passenger, Event, EventType
from elevator.pipeline.pipeline import OptimizationPipeline
from elevator.simulation.simulator import ElevatorSimulator, SimulationMetrics

logger = logging.getLogger(__name__)


@dataclass
class CallRequest:
    passenger : Passenger
    t         : float       # wall-clock seconds since loop start


class AsyncElevatorLoop:
    """Asyncio-based real-time elevator controller.

    Internally uses an ``asyncio.Queue`` to decouple call producers
    (API handlers, sensors) from the synchronous optimization engine
    which runs in a background thread via ``loop.run_in_executor``.

    Parameters
    ----------
    pipeline      : configured OptimizationPipeline
    initial_state : starting ElevatorState
    verbose       : forward verbose output from the inner simulator
    """

    def __init__(
        self,
        pipeline      : OptimizationPipeline,
        initial_state : ElevatorState,
        verbose       : bool = False,
    ) -> None:
        self._pipeline  = pipeline
        self._state     = initial_state
        self._verbose   = verbose
        self._queue     : asyncio.Queue[CallRequest | None] = asyncio.Queue()
        self._simulator : Optional[ElevatorSimulator]       = None
        self._task      : Optional[asyncio.Task]            = None
        self._started_at: float                             = 0.0

    # -- public API ---------------------------------------------------------

    async def start(self) -> None:
        """Start the background processing loop."""
        self._started_at = time.monotonic()
        self._simulator  = ElevatorSimulator(
            self._pipeline, self._state, verbose=self._verbose
        )
        self._task = asyncio.create_task(self._run_loop())
        logger.info("AsyncElevatorLoop started")

    async def call(self, passenger: Passenger) -> None:
        """Enqueue a new passenger call.  Non-blocking — returns immediately."""
        t = time.monotonic() - self._started_at
        await self._queue.put(CallRequest(passenger=passenger, t=t))
        logger.debug("Enqueued call for passenger %s at t=%.2f", passenger.id, t)

    async def stop(self) -> SimulationMetrics:
        """Drain the queue, stop the loop, and return final metrics."""
        await self._queue.put(None)  # sentinel
        if self._task is not None:
            await self._task
        assert self._simulator is not None
        return self._simulator.metrics

    @property
    def metrics(self) -> Optional[SimulationMetrics]:
        return self._simulator.metrics if self._simulator else None

    # -- internal loop ------------------------------------------------------

    async def _run_loop(self) -> None:
        """Consume calls from the queue and process them one at a time."""
        loop = asyncio.get_running_loop()
        assert self._simulator is not None

        while True:
            request = await self._queue.get()
            if request is None:
                break  # sentinel — drain complete

            sim = self._simulator
            passenger = request.passenger
            t         = request.t

            # Schedule and immediately process via run_in_executor so the
            # synchronous optimization does not block the event loop.
            await loop.run_in_executor(
                None,
                self._dispatch_call,
                sim, passenger, t,
            )
            self._queue.task_done()

        logger.info("AsyncElevatorLoop stopped")

    @staticmethod
    def _dispatch_call(
        sim       : ElevatorSimulator,
        passenger : Passenger,
        t         : float,
    ) -> None:
        """Synchronous call dispatch — runs in a thread-pool executor."""
        sim.schedule_call(t=t, passenger=passenger)
        # Process only newly enqueued events up to the current wall time
        # (do not run the whole queue — the loop manages ordering)
        processed = 0
        while sim._event_q:
            entry = sim._event_q[0]
            if entry.t > t:
                break
            import heapq
            entry = heapq.heappop(sim._event_q)
            if entry.generation != 0 and entry.generation != sim._current_gen:
                continue  # pragma: no cover
            sim._process(entry.event)
            processed += 1
        logger.debug("Dispatched call for %s, processed %d events", passenger.id, processed)
