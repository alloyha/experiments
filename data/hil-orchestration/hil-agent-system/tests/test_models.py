"""
Test models and database functionality.
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import (
    Agent,
    AgentExecution,
    AgentType,
    ExecutionStatus,
    ModelProfile,
)
from app.models.tool import Tool, ToolExecution, ToolExecutionStatus, ToolType
from app.models.workflow import (
    Workflow,
    WorkflowExecution,
    WorkflowExecutionStatus,
    WorkflowStatus,
)


class TestAgentModels:
    """Test agent models."""

    def test_agent_creation(self):
        """Test agent model creation."""
        agent = Agent(
            name="test-agent",
            agent_type=AgentType.SIMPLE,
            description="Test agent for unit testing",
            default_model_profile=ModelProfile.FAST,
            default_timeout=120,
            system_prompt="You are a test agent.",
            agent_metadata={"test": True},
        )

        assert agent.name == "test-agent"
        assert agent.agent_type == AgentType.SIMPLE
        assert agent.default_model_profile == ModelProfile.FAST
        assert agent.default_timeout == 120
        assert agent.is_active is True
        assert agent.agent_metadata["test"] is True

    def test_agent_execution_creation(self):
        """Test agent execution model creation."""
        agent_id = uuid.uuid4()
        execution = AgentExecution(
            agent_id=agent_id,
            input_data={"message": "Hello, world!"},
            model_profile=ModelProfile.BALANCED,
            timeout=300,
            status=ExecutionStatus.PENDING,
        )

        assert execution.agent_id == agent_id
        assert execution.input_data["message"] == "Hello, world!"
        assert execution.model_profile == ModelProfile.BALANCED
        assert execution.status == ExecutionStatus.PENDING
        assert execution.cost == 0.0
        assert execution.iterations == 1

    @pytest.mark.asyncio
    async def test_agent_database_operations(self, test_engine):
        """Test agent database operations."""
        # Create our own session for this test
        from sqlalchemy.ext.asyncio import async_sessionmaker

        async_session_maker = async_sessionmaker(
            test_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        async with async_session_maker() as session:
            # Create agent
            agent = Agent(
                name="db-test-agent",
                agent_type=AgentType.REASONING,
                description="Database test agent",
            )

            session.add(agent)
            await session.commit()
            await session.refresh(agent)

            assert agent.id is not None
            assert agent.created_at is not None

            # Create execution
            execution = AgentExecution(
                agent_id=agent.id,
                input_data={"test": "data"},
                model_profile=ModelProfile.POWERFUL,
                timeout=600,
            )

            session.add(execution)
            await session.commit()
            await session.refresh(execution)

            assert execution.id is not None
            assert execution.agent_id == agent.id


class TestWorkflowModels:
    """Test workflow models."""

    def test_workflow_creation(self):
        """Test workflow model creation."""
        workflow = Workflow(
            name="test-workflow",
            description="Test workflow for unit testing",
            version="1.0.0",
            nodes={
                "start": {"type": "agent", "agent_type": "simple"},
                "end": {"type": "output"},
            },
            edges={"start": {"target": "end", "condition": "always"}},
            max_execution_time=1800,
            status=WorkflowStatus.ACTIVE,
            tags=["test", "unit-test"],
        )

        assert workflow.name == "test-workflow"
        assert workflow.version == "1.0.0"
        assert workflow.status == WorkflowStatus.ACTIVE
        assert len(workflow.tags) == 2
        assert "test" in workflow.tags
        assert workflow.max_execution_time == 1800

    def test_workflow_execution_creation(self):
        """Test workflow execution model creation."""
        workflow_id = uuid.uuid4()
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            input_data={"user_input": "Start workflow"},
            configuration={"debug": True},
            status=WorkflowExecutionStatus.RUNNING,
        )

        assert execution.workflow_id == workflow_id
        assert execution.input_data["user_input"] == "Start workflow"
        assert execution.configuration["debug"] is True
        assert execution.status == WorkflowExecutionStatus.RUNNING
        assert execution.total_cost == 0.0


class TestToolModels:
    """Test tool models."""

    def test_tool_creation(self):
        """Test tool model creation."""
        tool = Tool(
            name="http-client",
            tool_type=ToolType.BUILTIN,
            description="HTTP client tool for API calls",
            configuration={"timeout": 30},
            available_actions=["get", "post", "put", "delete"],
            auth_required=False,
            rate_limit_requests=100,
            rate_limit_period=60,
        )

        assert tool.name == "http-client"
        assert tool.tool_type == ToolType.BUILTIN
        assert tool.auth_required is False
        assert len(tool.available_actions) == 4
        assert "get" in tool.available_actions
        assert tool.rate_limit_requests == 100

    def test_tool_execution_creation(self):
        """Test tool execution model creation."""
        tool_id = uuid.uuid4()
        execution = ToolExecution(
            tool_id=tool_id,
            action="get",
            input_parameters={"url": "https://api.example.com/users"},
            timeout=30,
            status=ToolExecutionStatus.COMPLETED,
            http_method="GET",
            http_status_code=200,
        )

        assert execution.tool_id == tool_id
        assert execution.action == "get"
        assert execution.input_parameters["url"] == "https://api.example.com/users"
        assert execution.status == ToolExecutionStatus.COMPLETED
        assert execution.http_method == "GET"
        assert execution.http_status_code == 200


class TestModelValidation:
    """Test model validation and constraints."""

    def test_default_values(self):
        """Test default values are set correctly."""
        agent = Agent(name="default-test", agent_type=AgentType.SIMPLE)

        assert agent.default_model_profile == ModelProfile.BALANCED
        assert agent.default_timeout == 300
        assert agent.max_iterations == 10
        assert agent.is_active is True
        assert agent.agent_metadata == {}


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_agent_json_serialization(self):
        """Test agent JSON serialization."""
        agent = Agent(
            name="json-test",
            agent_type=AgentType.SIMPLE,
            description="JSON test agent",
            agent_metadata={"version": "1.0", "tags": ["test"]},
        )

        # Test model dump
        agent_dict = agent.model_dump()
        assert agent_dict["name"] == "json-test"
        assert agent_dict["agent_type"] == "simple"
        assert agent_dict["agent_metadata"]["version"] == "1.0"

    def test_agent_json_deserialization(self):
        """Test agent JSON deserialization."""
        agent_data = {
            "name": "deserialized-agent",
            "agent_type": "reasoning",
            "description": "Deserialized test agent",
            "default_model_profile": "powerful",
            "agent_metadata": {"source": "json"},
        }

        agent = Agent(**agent_data)
        assert agent.name == "deserialized-agent"
        assert agent.agent_type == AgentType.REASONING
        assert agent.default_model_profile == ModelProfile.POWERFUL
        assert agent.agent_metadata["source"] == "json"
