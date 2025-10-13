"""
Tests for workflow integration service.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.workflow_integration import (
    WorkflowIntegrationService,
    execute_workflow,
    get_available_workflows,
    initialize_workflow_system,
    workflow_integration,
)


class TestWorkflowIntegrationService:
    """Test WorkflowIntegrationService."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return WorkflowIntegrationService()

    @pytest.fixture
    def mock_yaml_workflow(self):
        """Mock YAML workflow data."""
        return {
            "name": "simple_intent_classification",
            "version": "v1",
            "description": "Simple intent classification workflow",
            "category": "classification",
            "tags": ["nlp", "intent"],
            "inputs": {
                "text": {"type": "string", "description": "Input text to classify"}
            },
            "outputs": {
                "intent": {"type": "string", "description": "Classified intent"},
                "confidence": {
                    "type": "number",
                    "description": "Classification confidence",
                },
            },
            "nodes": {
                "classify_intent": {
                    "type": "simple_agent",
                    "agent_config": {
                        "model": "gpt-3.5-turbo",
                        "temperature": 0.1,
                        "max_tokens": 100,
                        "prompt_template": "Classify the intent of: {user_input}",
                    },
                }
            },
            "edges": [],
            "metadata": {"estimated_cost": 0.002, "estimated_time": 2.0},
        }

    @pytest.fixture
    def mock_workflow_model(self):
        """Mock workflow database model."""
        workflow = MagicMock()
        workflow.id = "test-workflow-id"
        workflow.name = "simple_intent_classification"
        workflow.version = "v1"
        workflow.description = "Simple intent classification workflow"
        workflow.nodes = {"classify_intent": {"type": "simple_agent"}}
        workflow.edges = []
        workflow.workflow_metadata = {"category": "classification", "tags": ["nlp"]}
        workflow.created_at = "2025-01-01T00:00:00Z"
        workflow.updated_at = "2025-01-01T00:00:00Z"
        return workflow

    @pytest.mark.asyncio
    async def test_initialize_workflows(self, service, mock_yaml_workflow):
        """Test workflow initialization."""
        # Mock the YAML loader and registry
        with (
            patch.object(service, "loader") as mock_loader,
            patch.object(service, "registry") as mock_registry,
            patch("app.services.workflow_integration.get_session") as mock_get_session,
        ):
            # Mock database session
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None

            mock_registry.sync_workflows_from_yaml.return_value = {
                "total_yaml_workflows": 1,
                "created": ["test_workflow"],
                "updated": [],
                "errors": [],
            }

            result = await service.initialize_workflows()

            assert result["total_yaml_workflows"] == 1
            assert len(result["created"]) == 1
            assert len(result["errors"]) == 0
            mock_registry.sync_workflows_from_yaml.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_workflow(self, service, mock_workflow_model):
        """Test getting a workflow."""
        with patch.object(service, "registry") as mock_registry:
            mock_registry.get_workflow.return_value = mock_workflow_model

            workflow = await service.get_workflow("simple_intent_classification", "v1")

            assert workflow is not None
            assert workflow.name == "simple_intent_classification"
            assert workflow.version == "v1"
            mock_registry.get_workflow.assert_called_once_with(
                "simple_intent_classification", "v1"
            )

    @pytest.mark.asyncio
    async def test_execute_workflow_simple_agent(self, service, mock_yaml_workflow):
        """Test executing a simple agent workflow."""
        # Mock dependencies
        with (
            patch.object(service, "registry") as mock_registry,
            patch("app.services.workflow_integration.SimpleAgent") as mock_agent_class,
        ):
            # Setup mocks
            mock_workflow = MagicMock()
            mock_workflow.nodes = mock_yaml_workflow["nodes"]
            mock_workflow.workflow_metadata = mock_yaml_workflow["metadata"]
            mock_registry.get_workflow.return_value = mock_workflow

            mock_agent = AsyncMock()
            mock_agent_result = {
                "content": "booking",
                "confidence": 0.95,
                "cost": 0.001,
                "latency": 1.2,
                "tokens_used": 50,
            }
            mock_agent.execute.return_value = mock_agent_result
            mock_agent_class.return_value = mock_agent

            # Execute workflow
            input_data = {"text": "I want to book a flight"}
            result = await service.execute_workflow(
                "simple_intent_classification", input_data, "v1"
            )

            # Verify results
            assert result["success"] is True
            assert "output" in result
            assert "execution_metadata" in result
            assert result["output"]["content"] == "booking"
            assert result["output"]["confidence"] == 0.95

            # Verify agent was called correctly
            mock_agent.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_workflow_not_found(self, service):
        """Test executing a non-existent workflow."""
        with patch.object(service, "registry") as mock_registry:
            mock_registry.get_workflow.return_value = None

            result = await service.execute_workflow("nonexistent", {})

            assert result["success"] is False
            assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_list_workflows(self, service):
        """Test listing workflows."""
        mock_workflows = [
            {
                "id": "workflow-1",
                "name": "simple_intent_classification",
                "version": "v1",
                "description": "Intent classification",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
                "tags": ["nlp"],
                "category": "classification",
            }
        ]

        with patch.object(service, "registry") as mock_registry:
            mock_registry.list_workflows.return_value = mock_workflows

            workflows = await service.list_workflows()

            assert len(workflows) == 1
            assert workflows[0]["name"] == "simple_intent_classification"
            assert workflows[0]["category"] == "classification"

    @pytest.mark.asyncio
    async def test_get_workflow_versions(self, service):
        """Test getting workflow versions."""
        with patch("app.services.workflow_integration.get_session") as mock_get_session:
            # Mock database session
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_get_session.return_value.__aexit__.return_value = None

            # Mock query result
            mock_result = MagicMock()
            mock_result.all.return_value = ["v1", "v2"]
            mock_session.exec.return_value = mock_result

            versions = await service.get_workflow_versions(
                "simple_intent_classification"
            )

            assert versions == ["v1", "v2"]


class TestWorkflowIntegrationHelpers:
    """Test workflow integration helper functions."""

    @pytest.mark.asyncio
    async def test_execute_workflow_helper(self):
        """Test execute_workflow helper function."""
        with patch.object(workflow_integration, "execute_workflow") as mock_execute:
            mock_execute.return_value = {"success": True, "output": {"result": "test"}}

            result = await execute_workflow("test_workflow", {"input": "test"})

            assert result["success"] is True
            mock_execute.assert_called_once_with(
                "test_workflow", {"input": "test"}, None
            )

    @pytest.mark.asyncio
    async def test_get_available_workflows_helper(self):
        """Test get_available_workflows helper function."""
        with patch.object(workflow_integration, "list_workflows") as mock_list:
            mock_workflows = [{"name": "test", "version": "v1"}]
            mock_list.return_value = mock_workflows

            workflows = await get_available_workflows()

            assert workflows == mock_workflows
            mock_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_workflow_system_helper(self):
        """Test initialize_workflow_system helper function."""
        with patch.object(workflow_integration, "initialize_workflows") as mock_init:
            mock_init.return_value = {
                "total_yaml_workflows": 2,
                "created": [],
                "updated": [],
                "errors": [],
            }

            result = await initialize_workflow_system()

            assert result["total_yaml_workflows"] == 2
            mock_init.assert_called_once()


class TestWorkflowIntegrationErrors:
    """Test error handling in workflow integration."""

    @pytest.mark.asyncio
    async def test_execute_workflow_with_exception(self):
        """Test workflow execution with exception."""
        service = WorkflowIntegrationService()

        with patch.object(service, "registry") as mock_registry:
            mock_registry.get_workflow.side_effect = Exception("Database error")

            result = await service.execute_workflow("test", {})

            assert result["success"] is False
            assert "Database error" in result["error"]

    @pytest.mark.asyncio
    async def test_get_workflow_versions_with_exception(self):
        """Test getting workflow versions with exception."""
        service = WorkflowIntegrationService()

        with patch("app.services.workflow_integration.get_session") as mock_get_session:
            mock_get_session.side_effect = Exception("Connection error")

            versions = await service.get_workflow_versions("test")

            assert versions == []


@pytest.mark.asyncio
async def test_workflow_integration_real_files():
    """Test workflow integration with real YAML files."""
    # Skip this test since it requires database initialization
    pytest.skip("Database initialization required for real file testing")


@pytest.mark.asyncio
async def test_workflow_integration_basic_functionality():
    """Test basic workflow integration functionality with mocked dependencies."""
    service = WorkflowIntegrationService()

    # Mock the registry dependency to avoid database calls
    with patch.object(service, "registry") as mock_registry:
        mock_registry.get_workflow.return_value = None

        # Test that the service handles missing workflows gracefully
        workflow = await service.get_workflow("nonexistent", "v1")
        assert workflow is None

        # Test execution with missing workflow
        result = await service.execute_workflow("nonexistent", {})
        assert result["success"] is False
        assert "not found" in result["error"].lower()
