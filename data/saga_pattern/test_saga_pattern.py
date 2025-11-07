"""
Comprehensive test suite for production-ready Saga Pattern implementation

Tests cover:
- ✅ Successful saga execution
- ✅ Failure with compensation (rollback)
- ✅ Partial compensation failure
- ✅ Retry logic with exponential backoff
- ✅ Timeout handling
- ✅ Idempotency
- ✅ Context passing between steps
- ✅ Concurrent execution protection
- ✅ State machine transitions
- ✅ Orchestrator management
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime

# Import from saga_pattern module (assuming it's in the same package)
from saga_pattern import (
    Saga,
    SagaCompensationError,
    SagaContext,
    SagaError,
    SagaExecutionError,
    SagaOrchestrator,
    SagaResult,
    SagaStatus,
    SagaStepError,
    SagaTimeoutError,
    TradeExecutionSaga,
    StrategyActivationSaga,
)


# ============================================
# TEST HELPERS
# ============================================


class TestSaga(Saga):
    """Simple saga for testing"""

    def __init__(self, name: str = "TestSaga"):
        super().__init__(name=name)


async def mock_action(ctx: SagaContext, value: str = "success") -> str:
    """Mock action that returns a value"""
    await asyncio.sleep(0.01)  # Simulate work
    return value


async def mock_failing_action(ctx: SagaContext) -> str:
    """Mock action that always fails"""
    await asyncio.sleep(0.01)
    raise ValueError("Action failed")


async def mock_compensation(result: str, ctx: SagaContext) -> None:
    """Mock compensation"""
    await asyncio.sleep(0.01)


async def mock_failing_compensation(result: str, ctx: SagaContext) -> None:
    """Mock compensation that fails"""
    await asyncio.sleep(0.01)
    raise ValueError("Compensation failed")


# ============================================
# BASIC TESTS
# ============================================


@pytest.mark.asyncio
async def test_successful_saga_execution():
    """Test successful saga execution with all steps completing"""
    saga = TestSaga("SuccessTest")

    await saga.add_step("step1", mock_action, mock_compensation)
    await saga.add_step("step2", mock_action, mock_compensation)
    await saga.add_step("step3", mock_action)

    result = await saga.execute()

    assert result.success is True
    assert result.status == SagaStatus.COMPLETED
    assert result.completed_steps == 3
    assert result.total_steps == 3
    assert result.error is None
    assert saga.status == SagaStatus.COMPLETED
    assert len(saga.completed_steps) == 3


@pytest.mark.asyncio
async def test_saga_failure_with_compensation():
    """Test saga failure with successful compensation (rollback)"""
    saga = TestSaga("FailureTest")

    # Add successful steps
    await saga.add_step("step1", mock_action, mock_compensation)
    await saga.add_step("step2", mock_action, mock_compensation)

    # Add failing step
    await saga.add_step("step3", mock_failing_action, mock_compensation)

    result = await saga.execute()

    assert result.success is False
    assert result.status == SagaStatus.ROLLED_BACK
    assert result.completed_steps == 2
    assert result.total_steps == 3
    assert result.error is not None
    assert isinstance(result.error, SagaStepError)
    assert saga.status == SagaStatus.ROLLED_BACK
    assert len(saga.compensation_errors) == 0  # No compensation errors


@pytest.mark.asyncio
async def test_saga_compensation_failure():
    """Test saga with compensation failure (unrecoverable)"""
    saga = TestSaga("CompensationFailureTest")

    # Add step with failing compensation
    await saga.add_step("step1", mock_action, mock_failing_compensation)

    # Add failing step to trigger compensation
    await saga.add_step("step2", mock_failing_action)

    result = await saga.execute()

    assert result.success is False
    assert result.status == SagaStatus.FAILED  # Unrecoverable
    assert result.completed_steps == 1
    assert len(result.compensation_errors) > 0
    assert saga.status == SagaStatus.FAILED


# ============================================
# CONTEXT TESTS
# ============================================


@pytest.mark.asyncio
async def test_context_passing_between_steps():
    """Test that context is passed between steps and steps can share data"""
    saga = TestSaga("ContextTest")

    async def step1(ctx: SagaContext) -> dict:
        ctx.set("user_id", 123)
        return {"amount": 100.0}

    async def step2(ctx: SagaContext) -> dict:
        # Access data from previous step's context
        user_id = ctx.get("user_id")
        amount = ctx.get("step1")["amount"]
        return {"user_id": user_id, "total": amount * 2}

    async def step3(ctx: SagaContext) -> dict:
        # Access all previous data
        total = ctx.get("step2")["total"]
        return {"final_total": total + 50}

    await saga.add_step("step1", step1)
    await saga.add_step("step2", step2)
    await saga.add_step("step3", step3)

    result = await saga.execute()

    assert result.success is True
    assert saga.context.get("user_id") == 123
    assert saga.context.get("step1")["amount"] == 100.0
    assert saga.context.get("step2")["total"] == 200.0
    assert saga.context.get("step3")["final_total"] == 250.0


# ============================================
# TIMEOUT TESTS
# ============================================


@pytest.mark.asyncio
async def test_step_timeout():
    """Test that steps timeout correctly"""
    saga = TestSaga("TimeoutTest")

    async def slow_action(ctx: SagaContext) -> str:
        await asyncio.sleep(5.0)  # Will timeout
        return "success"

    await saga.add_step("slow_step", slow_action, timeout=0.5, max_retries=1)

    result = await saga.execute()

    assert result.success is False
    assert saga.steps[0].error is not None
    # Should have attempted once and failed with timeout


@pytest.mark.asyncio
async def test_compensation_timeout():
    """Test that compensation timeouts are handled"""
    saga = TestSaga("CompensationTimeoutTest")

    async def slow_compensation(result, ctx: SagaContext) -> None:
        await asyncio.sleep(5.0)  # Will timeout
        return None

    await saga.add_step(
        "step1", mock_action, slow_compensation, compensation_timeout=0.5, max_retries=1
    )
    await saga.add_step("step2", mock_failing_action, max_retries=1)

    result = await saga.execute()

    assert result.success is False
    assert result.status == SagaStatus.FAILED  # Compensation failed
    assert len(result.compensation_errors) > 0


# ============================================
# RETRY TESTS
# ============================================


@pytest.mark.asyncio
async def test_step_retry_with_eventual_success():
    """Test that failing steps retry and can eventually succeed"""
    attempt_count = 0

    async def flaky_action(ctx: SagaContext) -> str:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError("Not yet!")
        return "success"

    saga = TestSaga("RetryTest")
    await saga.add_step("flaky_step", flaky_action, max_retries=3)

    result = await saga.execute()

    assert result.success is True
    assert attempt_count == 3
    assert saga.steps[0].retry_count == 2  # 0-indexed


@pytest.mark.asyncio
async def test_step_retry_exhaustion():
    """Test that steps fail after exhausting retries"""
    saga = TestSaga("RetryExhaustionTest")

    await saga.add_step("failing_step", mock_failing_action, max_retries=2)

    result = await saga.execute()

    assert result.success is False
    assert saga.steps[0].retry_count == 1  # Tried twice (0-indexed)


# ============================================
# IDEMPOTENCY TESTS
# ============================================


@pytest.mark.asyncio
async def test_idempotency_key_prevents_duplicate_execution():
    """Test that steps with same idempotency key are not executed twice"""
    execution_count = 0

    async def counted_action(ctx: SagaContext) -> str:
        nonlocal execution_count
        execution_count += 1
        return f"execution_{execution_count}"

    saga = TestSaga("IdempotencyTest")

    # Add same step twice with same idempotency key
    idempotency_key = "unique-key-123"
    await saga.add_step("step1", counted_action, idempotency_key=idempotency_key)
    await saga.add_step("step2", counted_action, idempotency_key=idempotency_key)

    result = await saga.execute()

    assert result.success is True
    # Should only execute once due to idempotency
    assert execution_count == 1


# ============================================
# CONCURRENT EXECUTION TESTS
# ============================================


@pytest.mark.asyncio
async def test_concurrent_execution_protection():
    """Test that saga cannot be executed concurrently"""
    saga = TestSaga("ConcurrentTest")

    async def slow_step(ctx: SagaContext) -> str:
        await asyncio.sleep(0.5)
        return "done"

    await saga.add_step("slow_step", slow_step)

    # Start first execution
    task1 = asyncio.create_task(saga.execute())

    # Try to start second execution immediately
    await asyncio.sleep(0.1)  # Give first execution time to start

    with pytest.raises(SagaExecutionError, match="already executing"):
        await saga.execute()

    # Wait for first execution to finish
    result = await task1
    assert result.success is True


@pytest.mark.asyncio
async def test_cannot_add_steps_during_execution():
    """Test that steps cannot be added while saga is executing"""
    saga = TestSaga("AddStepTest")

    async def slow_step(ctx: SagaContext) -> str:
        await asyncio.sleep(0.5)
        return "done"

    await saga.add_step("step1", slow_step)

    # Start execution
    task = asyncio.create_task(saga.execute())

    # Try to add step during execution
    await asyncio.sleep(0.1)

    with pytest.raises(SagaExecutionError, match="Cannot add steps while saga is executing"):
        await saga.add_step("step2", mock_action)

    await task


# ============================================
# STATE MACHINE TESTS
# ============================================


@pytest.mark.asyncio
async def test_state_transitions():
    """Test that state machine transitions correctly"""
    saga = TestSaga("StateTest")

    # Initial state
    assert saga.status == SagaStatus.PENDING
    assert saga.current_state == "Pending"

    await saga.add_step("step1", mock_action)

    result = await saga.execute()

    # Final state
    assert saga.status == SagaStatus.COMPLETED
    assert saga.current_state == "Completed"


@pytest.mark.asyncio
async def test_cannot_start_saga_without_steps():
    """Test that saga with no steps cannot start (guard condition)"""
    saga = TestSaga("NoStepsTest")

    # Don't add any steps

    with pytest.raises(SagaExecutionError, match="Cannot start saga"):
        await saga.execute()


# ============================================
# ORCHESTRATOR TESTS
# ============================================


@pytest.mark.asyncio
async def test_orchestrator_execute_saga():
    """Test orchestrator can execute and track sagas"""
    orchestrator = SagaOrchestrator()

    saga1 = TestSaga("Saga1")
    await saga1.add_step("step1", mock_action)

    saga2 = TestSaga("Saga2")
    await saga2.add_step("step1", mock_action)

    result1 = await orchestrator.execute_saga(saga1)
    result2 = await orchestrator.execute_saga(saga2)

    assert result1.success is True
    assert result2.success is True
    assert len(orchestrator.sagas) == 2


@pytest.mark.asyncio
async def test_orchestrator_statistics():
    """Test orchestrator statistics tracking"""
    orchestrator = SagaOrchestrator()

    # Create successful saga
    saga1 = TestSaga("Success")
    await saga1.add_step("step1", mock_action)
    await orchestrator.execute_saga(saga1)

    # Create failed saga
    saga2 = TestSaga("Failure")
    await saga2.add_step("step1", mock_action, mock_compensation)
    await saga2.add_step("step2", mock_failing_action)
    await orchestrator.execute_saga(saga2)

    stats = await orchestrator.get_statistics()

    assert stats["total_sagas"] == 2
    assert stats["completed"] == 1
    assert stats["rolled_back"] == 1


@pytest.mark.asyncio
async def test_orchestrator_get_saga_status():
    """Test retrieving saga status from orchestrator"""
    orchestrator = SagaOrchestrator()

    saga = TestSaga("StatusTest")
    await saga.add_step("step1", mock_action)

    await orchestrator.execute_saga(saga)

    status = await orchestrator.get_saga_status(saga.saga_id)

    assert status is not None
    assert status["name"] == "StatusTest"
    assert status["status"] == "completed"


# ============================================
# TRADE EXECUTION SAGA TESTS
# ============================================


@pytest.mark.asyncio
async def test_trade_execution_saga():
    """Test TradeExecutionSaga with mocked dependencies"""

    # Mock actions
    reserve_funds = AsyncMock(return_value={"reserved_amount": 1000.0})
    execute_trade = AsyncMock(return_value={"order_id": "ORDER123"})
    update_position = AsyncMock(return_value={"position_id": "POS456"})

    # Mock compensations
    unreserve_funds = AsyncMock()
    cancel_trade = AsyncMock()
    revert_position = AsyncMock()

    saga = TradeExecutionSaga(
        trade_id=1, symbol="BTCUSDT", quantity=0.5, price=50000.0, user_id=123
    )

    await saga.build(
        reserve_funds_action=reserve_funds,
        execute_trade_action=execute_trade,
        update_position_action=update_position,
        unreserve_funds_compensation=unreserve_funds,
        cancel_trade_compensation=cancel_trade,
        revert_position_compensation=revert_position,
    )

    result = await saga.execute()

    assert result.success is True
    assert result.completed_steps == 3
    reserve_funds.assert_called_once()
    execute_trade.assert_called_once()
    update_position.assert_called_once()


@pytest.mark.asyncio
async def test_trade_execution_saga_with_rollback():
    """Test TradeExecutionSaga rollback when trade execution fails"""

    reserve_funds = AsyncMock(return_value={"reserved_amount": 1000.0})
    execute_trade = AsyncMock(side_effect=ValueError("Exchange error"))
    update_position = AsyncMock()

    unreserve_funds = AsyncMock()
    cancel_trade = AsyncMock()
    revert_position = AsyncMock()

    saga = TradeExecutionSaga(
        trade_id=2, symbol="ETHUSDT", quantity=1.0, price=3000.0, user_id=456
    )

    await saga.build(
        reserve_funds_action=reserve_funds,
        execute_trade_action=execute_trade,
        update_position_action=update_position,
        unreserve_funds_compensation=unreserve_funds,
        cancel_trade_compensation=cancel_trade,
        revert_position_compensation=revert_position,
    )

    result = await saga.execute()

    assert result.success is False
    assert result.status == SagaStatus.ROLLED_BACK
    assert result.completed_steps == 1  # Only reserve_funds completed
    reserve_funds.assert_called_once()
    execute_trade.assert_called()
    update_position.assert_not_called()
    unreserve_funds.assert_called_once()  # Should compensate


# ============================================
# STRATEGY ACTIVATION SAGA TESTS
# ============================================


@pytest.mark.asyncio
async def test_strategy_activation_saga():
    """Test StrategyActivationSaga successful activation"""

    validate_strategy = AsyncMock(return_value={"valid": True})
    validate_funds = AsyncMock(return_value={"sufficient": True})
    activate_strategy = AsyncMock(return_value={"activated": True})
    publish_event = AsyncMock()

    deactivate_strategy = AsyncMock()

    saga = StrategyActivationSaga(strategy_id=1, user_id=123)

    await saga.build(
        validate_strategy_action=validate_strategy,
        validate_funds_action=validate_funds,
        activate_strategy_action=activate_strategy,
        publish_event_action=publish_event,
        deactivate_strategy_compensation=deactivate_strategy,
    )

    result = await saga.execute()

    assert result.success is True
    assert result.completed_steps == 4
    validate_strategy.assert_called_once()
    validate_funds.assert_called_once()
    activate_strategy.assert_called_once()
    publish_event.assert_called_once()


# ============================================
# RUN TESTS
# ============================================

if __name__ == "__main__":
    # Run tests with pytest
    # pytest saga_tests.py -v
    print("Run with: pytest saga_tests.py -v")