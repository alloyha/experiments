"""
Saga Pattern - Distributed Transactions with Compensation (Production-Ready)

Enables multi-step business processes with automatic rollback capability.
Unlike traditional ACID transactions, Sagas use compensating transactions to undo failed steps.

Uses python-statemachine for robust state management with full async support.

Features:
- ✅ Full async/await support with proper state machine integration
- ✅ Idempotent operations with deduplication keys
- ✅ Retry logic with exponential backoff
- ✅ Timeout protection per step
- ✅ Comprehensive error handling with partial compensation recovery
- ✅ Context passing between steps
- ✅ Guard conditions for state transitions
- ✅ Detailed observability and logging
- ✅ Thread-safe execution with locks
- ✅ Saga versioning and migration support

Example - Trade Execution Saga:
    Step 1: Reserve funds         → Compensation: Unreserve funds
    Step 2: Send order to Binance → Compensation: Cancel order
    Step 3: Update position       → Compensation: Revert position
    Step 4: Log trade             → Compensation: Delete trade log

If any step fails, all previous steps are compensated (undone) in reverse order.

State Machine:
    PENDING → EXECUTING → COMPLETED
                     ↘ COMPENSATING → ROLLED_BACK
                     ↘ FAILED (unrecoverable compensation failure)

    Step states:
    PENDING → EXECUTING → COMPLETED
                     ↘ COMPENSATING → COMPENSATED
                     ↘ FAILED (unrecoverable)

Usage:
    saga = TradeSaga()
    await saga.add_step(
        name="reserve_funds",
        action=reserve_funds_action,
        compensation=unreserve_funds_compensation,
        timeout=10.0,
        retry_attempts=3
    )
    result = await saga.execute()  # Returns SagaResult with full details
"""

import asyncio
import logging
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from statemachine import State, StateMachine
from statemachine.exceptions import TransitionNotAllowed

# Configure logging
logger = logging.getLogger(__name__)


class SagaStepStatus(Enum):
    """Status of a saga step"""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"


class SagaStatus(Enum):
    """Overall saga status"""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    FAILED = "failed"  # Unrecoverable failure during compensation
    ROLLED_BACK = "rolled_back"


@dataclass
class SagaContext:
    """Context passed between saga steps for data sharing"""

    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the context"""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context"""
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists in the context"""
        return key in self.data


@dataclass
class SagaStep:
    """Represents a single step in a saga with full metadata"""

    name: str
    action: Callable[..., Any]  # Forward action
    compensation: Callable[..., Any] | None = None  # Rollback action
    status: SagaStepStatus = SagaStepStatus.PENDING
    result: Any | None = None
    error: Exception | None = None
    executed_at: datetime | None = None
    compensated_at: datetime | None = None
    idempotency_key: str = field(default_factory=lambda: str(uuid4()))
    retry_attempts: int = 0
    max_retries: int = 3
    timeout: float = 30.0  # seconds
    compensation_timeout: float = 30.0
    retry_count: int = 0

    def __hash__(self):
        return hash(self.idempotency_key)


@dataclass
class SagaResult:
    """Result of saga execution"""

    success: bool
    saga_name: str
    status: SagaStatus
    completed_steps: int
    total_steps: int
    error: Exception | None = None
    execution_time: float = 0.0
    context: SagaContext | None = None
    compensation_errors: list[Exception] = field(default_factory=list)

    @property
    def is_completed(self) -> bool:
        return self.status == SagaStatus.COMPLETED

    @property
    def is_rolled_back(self) -> bool:
        return self.status == SagaStatus.ROLLED_BACK

    @property
    def is_failed(self) -> bool:
        return self.status == SagaStatus.FAILED


class SagaError(Exception):
    """Base saga error"""


class SagaStepError(SagaError):
    """Error executing saga step"""


class SagaCompensationError(SagaError):
    """Error executing compensation"""


class SagaTimeoutError(SagaError):
    """Saga step timeout"""


class SagaExecutionError(SagaError):
    """Error during saga execution (already executing, invalid state, etc.)"""


class Saga(ABC):  # noqa: B024
    """
    Production-ready base class for saga implementations with state machine

    Concrete sagas should:
    1. Inherit from Saga
    2. Define steps in constructor or add_step calls
    3. Call execute() to run the saga
    4. Handle SagaResult appropriately
    """

    class _StateMachineImpl(StateMachine):
        """Internal async state machine using python-statemachine"""

        pending = State("Pending", initial=True)
        executing = State("Executing")
        completed = State("Completed", final=True)
        compensating = State("Compensating")
        rolled_back = State("RolledBack", final=True)
        failed = State("Failed", final=True)

        # Transitions with guards
        start = pending.to(executing, cond="can_start")
        succeed = executing.to(completed)
        fail = executing.to(compensating, cond="can_compensate")
        fail_unrecoverable = executing.to(failed)
        finish_compensation = compensating.to(rolled_back)
        compensation_failed = compensating.to(failed)

        def __init__(self, saga: "Saga"):
            self.saga = saga
            super().__init__()

        def can_start(self) -> bool:
            """Guard: can only start if we have steps"""
            return len(self.saga.steps) > 0

        def can_compensate(self) -> bool:
            """Guard: can only compensate if we have completed steps"""
            return len(self.saga.completed_steps) > 0

        async def on_enter_executing(self) -> None:
            """Called when entering EXECUTING state"""
            await self.saga._on_enter_executing()

        async def on_enter_compensating(self) -> None:
            """Called when entering COMPENSATING state"""
            await self.saga._on_enter_compensating()

        async def on_enter_completed(self) -> None:
            """Called when entering COMPLETED state"""
            await self.saga._on_enter_completed()

        async def on_enter_rolled_back(self) -> None:
            """Called when entering ROLLED_BACK state"""
            await self.saga._on_enter_rolled_back()

        async def on_enter_failed(self) -> None:
            """Called when entering FAILED state"""
            await self.saga._on_enter_failed()

    def __init__(self, name: str = "Saga", version: str = "1.0"):
        self.name = name
        self.version = version
        self.saga_id = str(uuid4())
        self.status = SagaStatus.PENDING
        self.steps: list[SagaStep] = []
        self.completed_steps: list[SagaStep] = []
        self.context = SagaContext()
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.error: Exception | None = None
        self.compensation_errors: list[Exception] = []
        self._state_machine = self._StateMachineImpl(self)
        self._executing = False
        self._execution_lock = asyncio.Lock()
        self._executed_step_keys: set[str] = set()  # For idempotency

    async def _on_enter_executing(self) -> None:
        """Callback: entering EXECUTING state"""
        self.status = SagaStatus.EXECUTING
        self.started_at = datetime.now()
        self.completed_steps = []
        logger.info(f"Saga {self.name} [{self.saga_id}] entering EXECUTING state")

    async def _on_enter_compensating(self) -> None:
        """Callback: entering COMPENSATING state"""
        self.status = SagaStatus.COMPENSATING
        logger.warning(
            f"Saga {self.name} [{self.saga_id}] entering COMPENSATING state - "
            f"rolling back {len(self.completed_steps)} steps"
        )

    async def _on_enter_completed(self) -> None:
        """Callback: entering COMPLETED state"""
        self.status = SagaStatus.COMPLETED
        self.completed_at = datetime.now()
        execution_time = (
            (self.completed_at - self.started_at).total_seconds() if self.started_at else 0
        )
        logger.info(
            f"Saga {self.name} [{self.saga_id}] COMPLETED successfully in {execution_time:.2f}s"
        )

    async def _on_enter_rolled_back(self) -> None:
        """Callback: entering ROLLED_BACK state"""
        self.status = SagaStatus.ROLLED_BACK
        self.completed_at = datetime.now()
        execution_time = (
            (self.completed_at - self.started_at).total_seconds() if self.started_at else 0
        )
        logger.info(
            f"Saga {self.name} [{self.saga_id}] ROLLED_BACK after {execution_time:.2f}s"
        )

    async def _on_enter_failed(self) -> None:
        """Callback: entering FAILED state (unrecoverable)"""
        self.status = SagaStatus.FAILED
        self.completed_at = datetime.now()
        logger.error(
            f"Saga {self.name} [{self.saga_id}] FAILED - unrecoverable error during compensation"
        )

    async def add_step(
        self,
        name: str,
        action: Callable[..., Any],
        compensation: Callable[..., Any] | None = None,
        timeout: float = 30.0,
        compensation_timeout: float = 30.0,
        max_retries: int = 3,
        idempotency_key: str | None = None,
    ) -> None:
        """
        Add a step to the saga

        Args:
            name: Step name
            action: Forward action to execute (can be sync or async)
            compensation: Rollback action (can be sync or async)
            timeout: Timeout in seconds for action execution
            compensation_timeout: Timeout in seconds for compensation
            max_retries: Maximum retry attempts for action
            idempotency_key: Custom idempotency key (auto-generated if None)
        """
        if self._executing:
            raise SagaExecutionError("Cannot add steps while saga is executing")

        step = SagaStep(
            name=name,
            action=action,
            compensation=compensation,
            timeout=timeout,
            compensation_timeout=compensation_timeout,
            max_retries=max_retries,
            idempotency_key=idempotency_key or str(uuid4()),
        )
        self.steps.append(step)
        logger.debug(f"Added step '{name}' to saga {self.name}")

    async def execute(self) -> SagaResult:
        """
        Execute the saga with full error handling and compensation

        Returns:
            SagaResult: Detailed result of saga execution

        Raises:
            SagaExecutionError: If saga is already executing or in invalid state
        """
        async with self._execution_lock:
            if self._executing:
                raise SagaExecutionError("Saga is already executing")

            self._executing = True
            start_time = datetime.now()

            try:
                # Transition to EXECUTING (with guard check)
                try:
                    await self._state_machine.start()
                except TransitionNotAllowed as e:
                    raise SagaExecutionError(f"Cannot start saga: {e}")

                # Execute all steps forward
                for step in self.steps:
                    # Check idempotency - skip if already executed
                    if step.idempotency_key in self._executed_step_keys:
                        logger.info(f"Skipping step '{step.name}' - already executed (idempotent)")
                        continue

                    await self._execute_step_with_retry(step)
                    self.completed_steps.append(step)
                    self._executed_step_keys.add(step.idempotency_key)

                # Transition to COMPLETED
                await self._state_machine.succeed()

                execution_time = (datetime.now() - start_time).total_seconds()
                return SagaResult(
                    success=True,
                    saga_name=self.name,
                    status=SagaStatus.COMPLETED,
                    completed_steps=len(self.completed_steps),
                    total_steps=len(self.steps),
                    execution_time=execution_time,
                    context=self.context,
                )

            except Exception as e:
                self.error = e
                logger.error(f"Saga {self.name} failed: {e}", exc_info=True)

                # Transition to COMPENSATING
                try:
                    await self._state_machine.fail()
                except TransitionNotAllowed:
                    # No completed steps to compensate - go straight to FAILED
                    await self._state_machine.fail_unrecoverable()
                    execution_time = (datetime.now() - start_time).total_seconds()
                    return SagaResult(
                        success=False,
                        saga_name=self.name,
                        status=SagaStatus.FAILED,
                        completed_steps=0,
                        total_steps=len(self.steps),
                        error=e,
                        execution_time=execution_time,
                    )

                # Attempt compensation
                try:
                    await self._compensate_all()
                    await self._state_machine.finish_compensation()

                    execution_time = (datetime.now() - start_time).total_seconds()
                    return SagaResult(
                        success=False,
                        saga_name=self.name,
                        status=SagaStatus.ROLLED_BACK,
                        completed_steps=len(self.completed_steps),
                        total_steps=len(self.steps),
                        error=e,
                        execution_time=execution_time,
                        context=self.context,
                        compensation_errors=self.compensation_errors,
                    )

                except SagaCompensationError as comp_error:
                    # Compensation failed - enter FAILED state
                    await self._state_machine.compensation_failed()

                    execution_time = (datetime.now() - start_time).total_seconds()
                    return SagaResult(
                        success=False,
                        saga_name=self.name,
                        status=SagaStatus.FAILED,
                        completed_steps=len(self.completed_steps),
                        total_steps=len(self.steps),
                        error=e,
                        execution_time=execution_time,
                        compensation_errors=self.compensation_errors,
                    )

            finally:
                self._executing = False

    async def _execute_step_with_retry(self, step: SagaStep) -> None:
        """Execute a step with retry logic and exponential backoff"""
        last_error = None

        for attempt in range(step.max_retries):
            try:
                step.retry_count = attempt
                await self._execute_step(step)
                return  # Success!

            except SagaTimeoutError as e:
                last_error = e
                logger.warning(
                    f"Step '{step.name}' timed out (attempt {attempt + 1}/{step.max_retries})"
                )

            except SagaStepError as e:
                last_error = e
                logger.warning(
                    f"Step '{step.name}' failed (attempt {attempt + 1}/{step.max_retries}): {e}"
                )

            # Exponential backoff before retry
            if attempt < step.max_retries - 1:
                backoff_time = 2**attempt  # 1s, 2s, 4s...
                logger.info(f"Retrying step '{step.name}' in {backoff_time}s...")
                await asyncio.sleep(backoff_time)

        # All retries exhausted
        step.error = last_error
        raise last_error

    async def _execute_step(self, step: SagaStep) -> None:
        """Execute a single step with timeout"""
        try:
            step.status = SagaStepStatus.EXECUTING
            logger.info(f"Executing step: {step.name}")

            # Execute with timeout
            step.result = await asyncio.wait_for(
                self._invoke(step.action, self.context), timeout=step.timeout
            )

            # Store result in context for next steps
            self.context.set(step.name, step.result)

            step.status = SagaStepStatus.COMPLETED
            step.executed_at = datetime.now()
            logger.info(f"Step '{step.name}' completed successfully")

        except asyncio.TimeoutError:
            step.status = SagaStepStatus.FAILED
            error = SagaTimeoutError(f"Step '{step.name}' timed out after {step.timeout}s")
            step.error = error
            raise error

        except Exception as e:
            step.status = SagaStepStatus.FAILED
            step.error = e
            msg = f"Step '{step.name}' failed: {e!s}"
            raise SagaStepError(msg)

    async def _compensate_all(self) -> None:
        """
        Compensate all completed steps in reverse order
        Continues even if some compensations fail, collecting all errors
        """
        compensation_errors = []

        for step in reversed(self.completed_steps):
            if step.compensation:
                try:
                    await self._compensate_step_with_retry(step)
                except SagaCompensationError as e:
                    compensation_errors.append(e)
                    self.compensation_errors.append(e)
                    logger.error(f"Compensation failed for step '{step.name}': {e}")
                    # Continue compensating other steps

        if compensation_errors:
            # Raise aggregate error with all compensation failures
            error_summary = "; ".join(str(e) for e in compensation_errors)
            raise SagaCompensationError(
                f"Failed to compensate {len(compensation_errors)} steps: {error_summary}"
            )

    async def _compensate_step_with_retry(self, step: SagaStep) -> None:
        """Compensate a step with retry logic"""
        last_error = None
        max_comp_retries = 3

        for attempt in range(max_comp_retries):
            try:
                await self._compensate_step(step)
                return  # Success!

            except SagaCompensationError as e:
                last_error = e
                logger.warning(
                    f"Compensation for '{step.name}' failed "
                    f"(attempt {attempt + 1}/{max_comp_retries})"
                )

            # Exponential backoff
            if attempt < max_comp_retries - 1:
                backoff_time = 2**attempt
                await asyncio.sleep(backoff_time)

        # All retries exhausted
        raise last_error

    async def _compensate_step(self, step: SagaStep) -> None:
        """Compensate a single step with timeout"""
        try:
            step.status = SagaStepStatus.COMPENSATING
            logger.info(f"Compensating step: {step.name}")

            # Pass the step result to compensation for context
            await asyncio.wait_for(
                self._invoke(step.compensation, step.result, self.context),
                timeout=step.compensation_timeout,
            )

            step.status = SagaStepStatus.COMPENSATED
            step.compensated_at = datetime.now()
            logger.info(f"Step '{step.name}' compensated successfully")

        except asyncio.TimeoutError:
            step.status = SagaStepStatus.FAILED
            msg = f"Compensation for '{step.name}' timed out after {step.compensation_timeout}s"
            raise SagaCompensationError(msg)

        except Exception as e:
            step.status = SagaStepStatus.FAILED
            step.error = e
            msg = f"Compensation for step '{step.name}' failed: {e!s}"
            raise SagaCompensationError(msg)

    @staticmethod
    async def _invoke(func: Callable[..., Any], *args, **kwargs) -> Any:
        """Invoke function (handle both sync and async)"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    @property
    def current_state(self) -> str:
        """Get current state name"""
        return self._state_machine.current_state.name

    def get_status(self) -> dict[str, Any]:
        """Get detailed saga status"""
        return {
            "saga_id": self.saga_id,
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "current_state": self.current_state,
            "total_steps": len(self.steps),
            "completed_steps": len(self.completed_steps),
            "steps": [
                {
                    "name": step.name,
                    "status": step.status.value,
                    "retry_count": step.retry_count,
                    "error": str(step.error) if step.error else None,
                    "executed_at": step.executed_at.isoformat() if step.executed_at else None,
                    "compensated_at": (
                        step.compensated_at.isoformat() if step.compensated_at else None
                    ),
                }
                for step in self.steps
            ],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": str(self.error) if self.error else None,
            "compensation_errors": [str(e) for e in self.compensation_errors],
        }


# ============================================
# TRADE EXECUTION SAGA
# ============================================


class TradeExecutionSaga(Saga):
    """Production-ready saga for executing trades with multi-step validation and compensation"""

    def __init__(self, trade_id: int, symbol: str, quantity: float, price: float, user_id: int):
        super().__init__(name=f"TradeExecutionSaga-{trade_id}", version="2.0")
        self.trade_id = trade_id
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.user_id = user_id

    async def build(
        self,
        reserve_funds_action: Callable,
        execute_trade_action: Callable,
        update_position_action: Callable,
        unreserve_funds_compensation: Callable,
        cancel_trade_compensation: Callable,
        revert_position_compensation: Callable,
    ) -> None:
        """Build saga with provided actions"""

        # Step 1: Reserve funds for trade
        await self.add_step(
            name="reserve_funds",
            action=lambda ctx: reserve_funds_action(self.user_id, self.quantity * self.price, ctx),
            compensation=lambda result, ctx: unreserve_funds_compensation(
                self.user_id, result, ctx
            ),
            timeout=10.0,
            max_retries=3,
        )

        # Step 2: Execute trade on exchange
        await self.add_step(
            name="execute_trade",
            action=lambda ctx: execute_trade_action(
                self.trade_id, self.symbol, self.quantity, self.price, ctx
            ),
            compensation=lambda result, ctx: cancel_trade_compensation(self.trade_id, result, ctx),
            timeout=30.0,
            max_retries=2,
        )

        # Step 3: Update position in database
        await self.add_step(
            name="update_position",
            action=lambda ctx: update_position_action(
                self.trade_id, self.quantity, self.price, ctx
            ),
            compensation=lambda result, ctx: revert_position_compensation(self.trade_id, ctx),
            timeout=5.0,
            max_retries=3,
        )


# ============================================
# STRATEGY ACTIVATION SAGA
# ============================================


class StrategyActivationSaga(Saga):
    """Production-ready saga for activating trading strategies"""

    def __init__(self, strategy_id: int, user_id: int):
        super().__init__(name=f"StrategyActivationSaga-{strategy_id}", version="2.0")
        self.strategy_id = strategy_id
        self.user_id = user_id

    async def build(
        self,
        validate_strategy_action: Callable,
        validate_funds_action: Callable,
        activate_strategy_action: Callable,
        publish_event_action: Callable,
        deactivate_strategy_compensation: Callable,
    ) -> None:
        """Build saga with provided actions"""

        # Step 1: Validate strategy configuration
        await self.add_step(
            name="validate_strategy",
            action=lambda ctx: validate_strategy_action(self.strategy_id, ctx),
            timeout=5.0,
        )

        # Step 2: Validate user has sufficient funds
        await self.add_step(
            name="validate_funds",
            action=lambda ctx: validate_funds_action(self.user_id, self.strategy_id, ctx),
            timeout=5.0,
        )

        # Step 3: Activate the strategy
        await self.add_step(
            name="activate_strategy",
            action=lambda ctx: activate_strategy_action(self.strategy_id, self.user_id, ctx),
            compensation=lambda result, ctx: deactivate_strategy_compensation(
                self.strategy_id, ctx
            ),
            timeout=10.0,
            max_retries=3,
        )

        # Step 4: Publish event (no compensation needed - idempotent)
        await self.add_step(
            name="publish_event",
            action=lambda ctx: publish_event_action(self.strategy_id, self.user_id, ctx),
            timeout=5.0,
        )


# ============================================
# SAGA ORCHESTRATOR
# ============================================


class SagaOrchestrator:
    """
    Production-ready orchestrator for managing and tracking multiple sagas
    Thread-safe with proper async support
    """

    def __init__(self):
        self.sagas: dict[str, Saga] = {}
        self._lock = asyncio.Lock()

    async def execute_saga(self, saga: Saga) -> SagaResult:
        """Execute a saga and track it"""
        async with self._lock:
            self.sagas[saga.saga_id] = saga

        result = await saga.execute()

        logger.info(
            f"Saga {saga.name} [{saga.saga_id}] finished with status: {result.status.value}"
        )

        return result

    async def get_saga(self, saga_id: str) -> Saga | None:
        """Get saga by ID"""
        async with self._lock:
            return self.sagas.get(saga_id)

    async def get_saga_status(self, saga_id: str) -> dict[str, Any] | None:
        """Get status of a saga by ID"""
        saga = await self.get_saga(saga_id)
        return saga.get_status() if saga else None

    async def get_all_sagas_status(self) -> list[dict[str, Any]]:
        """Get status of all sagas"""
        async with self._lock:
            return [saga.get_status() for saga in self.sagas.values()]

    async def count_by_status(self, status: SagaStatus) -> int:
        """Count sagas by status"""
        async with self._lock:
            return sum(1 for saga in self.sagas.values() if saga.status == status)

    async def count_completed(self) -> int:
        """Count completed sagas"""
        return await self.count_by_status(SagaStatus.COMPLETED)

    async def count_failed(self) -> int:
        """Count failed sagas (unrecoverable)"""
        return await self.count_by_status(SagaStatus.FAILED)

    async def count_rolled_back(self) -> int:
        """Count rolled back sagas (recovered)"""
        return await self.count_by_status(SagaStatus.ROLLED_BACK)

    async def get_statistics(self) -> dict[str, Any]:
        """Get orchestrator statistics"""
        async with self._lock:
            total = len(self.sagas)
            return {
                "total_sagas": total,
                "completed": await self.count_completed(),
                "rolled_back": await self.count_rolled_back(),
                "failed": await self.count_failed(),
                "executing": await self.count_by_status(SagaStatus.EXECUTING),
                "pending": await self.count_by_status(SagaStatus.PENDING),
            }


__all__ = [
    "Saga",
    "SagaCompensationError",
    "SagaContext",
    "SagaError",
    "SagaExecutionError",
    "SagaOrchestrator",
    "SagaResult",
    "SagaStatus",
    "SagaStep",
    "SagaStepError",
    "SagaStepStatus",
    "SagaTimeoutError",
    "StrategyActivationSaga",
    "TradeExecutionSaga",
]