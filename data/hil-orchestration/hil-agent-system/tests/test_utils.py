"""
Test utilities and helper functions.
"""

import asyncio
import uuid
from datetime import UTC, datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock

import pytest

from app.models.agent import (
    Agent,
    AgentExecution,
    AgentType,
    ExecutionStatus,
    ModelProfile,
)
from app.models.tool import Tool, ToolExecution
from app.models.workflow import Workflow, WorkflowExecution


class TestDataFactory:
    """Factory for creating test data."""

    @staticmethod
    def create_agent(**kwargs) -> Agent:
        """Create a test agent with default values."""
        defaults = {
            "name": f"test-agent-{uuid.uuid4().hex[:8]}",
            "agent_type": AgentType.SIMPLE,
            "description": "Test agent for unit testing",
            "default_model_profile": ModelProfile.BALANCED,
            "default_timeout": 300,
            "system_prompt": "You are a test agent.",
            "agent_metadata": {},
        }
        defaults.update(kwargs)
        return Agent(**defaults)

    @staticmethod
    def create_agent_execution(agent_id: uuid.UUID = None, **kwargs) -> AgentExecution:
        """Create a test agent execution with default values."""
        if agent_id is None:
            agent_id = uuid.uuid4()

        defaults = {
            "agent_id": agent_id,
            "input_data": {"message": "Test input"},
            "model_profile": ModelProfile.BALANCED,
            "timeout": 300,
            "status": ExecutionStatus.PENDING,
        }
        defaults.update(kwargs)
        return AgentExecution(**defaults)

    @staticmethod
    def create_workflow(**kwargs) -> Workflow:
        """Create a test workflow with default values."""
        defaults = {
            "name": f"test-workflow-{uuid.uuid4().hex[:8]}",
            "description": "Test workflow for unit testing",
            "version": "1.0.0",
            "nodes": {
                "start": {"type": "agent", "agent_type": "simple"},
                "end": {"type": "output"},
            },
            "edges": {"start": {"target": "end", "condition": "always"}},
        }
        defaults.update(kwargs)
        return Workflow(**defaults)

    @staticmethod
    def create_tool(**kwargs) -> Tool:
        """Create a test tool with default values."""
        defaults = {
            "name": f"test-tool-{uuid.uuid4().hex[:8]}",
            "tool_type": "builtin",
            "description": "Test tool for unit testing",
            "available_actions": ["execute"],
            "configuration": {},
        }
        defaults.update(kwargs)
        return Tool(**defaults)


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[str] = None):
        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.last_request = None

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Mock completion method."""
        self.last_request = {"messages": messages, "kwargs": kwargs}

        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        # Simulate API delay
        await asyncio.sleep(0.01)

        return response

    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.last_request = None


class MockToolProvider:
    """Mock tool provider for testing."""

    def __init__(self, responses: dict[str, Any] = None):
        self.responses = responses or {"default": {"status": "success"}}
        self.call_count = 0
        self.last_request = None

    async def execute_action(
        self, action: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Mock tool execution method."""
        self.last_request = {"action": action, "parameters": parameters}

        response = self.responses.get(action, self.responses.get("default", {}))
        self.call_count += 1

        # Simulate tool execution delay
        await asyncio.sleep(0.02)

        return response

    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.last_request = None


@pytest.fixture
def test_data_factory():
    """Provide test data factory."""
    return TestDataFactory()


@pytest.fixture
def mock_llm_provider():
    """Provide mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_tool_provider():
    """Provide mock tool provider."""
    return MockToolProvider()


class DatabaseHelpers:
    """Helper functions for database testing."""

    @staticmethod
    async def create_test_agent(session, **kwargs) -> Agent:
        """Create and persist a test agent."""
        agent = TestDataFactory.create_agent(**kwargs)
        session.add(agent)
        await session.commit()
        await session.refresh(agent)
        return agent

    @staticmethod
    async def create_test_workflow(session, **kwargs) -> Workflow:
        """Create and persist a test workflow."""
        workflow = TestDataFactory.create_workflow(**kwargs)
        session.add(workflow)
        await session.commit()
        await session.refresh(workflow)
        return workflow

    @staticmethod
    async def create_test_tool(session, **kwargs) -> Tool:
        """Create and persist a test tool."""
        tool = TestDataFactory.create_tool(**kwargs)
        session.add(tool)
        await session.commit()
        await session.refresh(tool)
        return tool

    @staticmethod
    async def cleanup_test_data(session, *objects):
        """Clean up test data from database."""
        for obj in objects:
            await session.delete(obj)
        await session.commit()


@pytest.fixture
def db_helpers():
    """Provide database helper functions."""
    return DatabaseHelpers()


class AsyncTestHelpers:
    """Helper functions for async testing."""

    @staticmethod
    async def run_with_timeout(coro, timeout: float = 1.0):
        """Run coroutine with timeout."""
        return await asyncio.wait_for(coro, timeout=timeout)

    @staticmethod
    async def simulate_concurrent_requests(requests: list, max_concurrent: int = 5):
        """Simulate concurrent requests with concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_request(request):
            async with semaphore:
                return await request

        tasks = [bounded_request(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def create_async_mock(return_value=None):
        """Create async mock with return value."""
        mock = AsyncMock()
        if return_value is not None:
            mock.return_value = return_value
        return mock


@pytest.fixture
def async_helpers():
    """Provide async test helper functions."""
    return AsyncTestHelpers()


class ValidationHelpers:
    """Helper functions for validation testing."""

    @staticmethod
    def assert_uuid_format(value: str):
        """Assert that value is a valid UUID string."""
        try:
            uuid.UUID(value)
        except ValueError:
            pytest.fail(f"'{value}' is not a valid UUID")

    @staticmethod
    def assert_datetime_format(value: str):
        """Assert that value is a valid datetime string."""
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"'{value}' is not a valid datetime")

    @staticmethod
    def assert_enum_value(value: str, enum_class):
        """Assert that value is a valid enum member."""
        try:
            enum_class(value)
        except ValueError:
            valid_values = [e.value for e in enum_class]
            pytest.fail(
                f"'{value}' is not a valid {enum_class.__name__}. Valid values: {valid_values}"
            )

    @staticmethod
    def assert_response_structure(
        response_data: dict[str, Any], required_fields: list[str]
    ):
        """Assert that response has required structure."""
        for field in required_fields:
            assert field in response_data, (
                f"Required field '{field}' missing from response"
            )

    @staticmethod
    def assert_error_response(response_data: dict[str, Any]):
        """Assert that response follows error format."""
        required_error_fields = ["detail", "status_code"]
        ValidationHelpers.assert_response_structure(
            response_data, required_error_fields
        )


@pytest.fixture
def validation_helpers():
    """Provide validation helper functions."""
    return ValidationHelpers()


class PerformanceHelpers:
    """Helper functions for performance testing."""

    @staticmethod
    async def measure_execution_time(coro):
        """Measure coroutine execution time."""
        start_time = datetime.now(UTC)
        result = await coro
        end_time = datetime.now(UTC)
        execution_time = (end_time - start_time).total_seconds()

        return {
            "result": result,
            "execution_time": execution_time,
            "start_time": start_time,
            "end_time": end_time,
        }

    @staticmethod
    async def stress_test(operation, num_iterations: int = 100):
        """Run stress test with multiple iterations."""
        results = []
        errors = []

        for i in range(num_iterations):
            try:
                result = await operation()
                results.append(result)
            except Exception as e:
                errors.append((i, e))

        return {
            "successful_iterations": len(results),
            "failed_iterations": len(errors),
            "success_rate": len(results) / num_iterations,
            "results": results,
            "errors": errors,
        }


@pytest.fixture
def performance_helpers():
    """Provide performance testing helper functions."""
    return PerformanceHelpers()
