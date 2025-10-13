"""
Simple unit tests for workflow registry service to improve coverage.
These tests focus on specific methods without complex async mocking.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.services.workflow_registry import WorkflowRegistryService


class TestWorkflowRegistrySimple:
    """Simple unit tests for workflow registry methods."""

    def test_init(self):
        """Test WorkflowRegistryService initialization."""
        registry = WorkflowRegistryService(workflows_dir="test_dir")
        assert registry.loader is not None
        assert str(registry.loader.workflows_dir) == "test_dir"

    def test_yaml_to_db_format_simple(self):
        """Test YAML to DB format conversion."""
        registry = WorkflowRegistryService(workflows_dir="test_dir")

        yaml_data = {
            "name": "test_workflow",
            "version": "1.0.0",
            "description": "Test workflow",
            "workflow": {
                "nodes": [{"id": "start", "type": "action"}],
                "edges": [{"from": "start", "to": "end"}],
            },
            "metadata": {"author": "test"},
            "config": {"max_execution_time": 300},
        }

        result = registry._yaml_to_db_format(yaml_data)

        assert result["name"] == "test_workflow"
        assert result["version"] == "1.0.0"
        assert result["description"] == "Test workflow"
        assert result["nodes"] == [{"id": "start", "type": "action"}]
        assert result["edges"] == [{"from": "start", "to": "end"}]
        assert result["workflow_metadata"]["yaml_metadata"] == {"author": "test"}
        assert result["workflow_metadata"]["yaml_config"] == {"max_execution_time": 300}
        assert result["max_execution_time"] == 300

    def test_yaml_to_db_format_minimal(self):
        """Test YAML to DB format conversion with minimal data."""
        registry = WorkflowRegistryService(workflows_dir="test_dir")

        yaml_data = {
            "name": "minimal_workflow",
            "version": "2.0.0",
            "workflow": {"nodes": [], "edges": []},
        }

        result = registry._yaml_to_db_format(yaml_data)

        assert result["name"] == "minimal_workflow"
        assert result["version"] == "2.0.0"
        assert result["description"] is None
        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["workflow_metadata"]["yaml_metadata"] == {}
        assert result["workflow_metadata"]["yaml_config"] == {}
        assert result["max_execution_time"] == 1800  # default

    def test_workflow_needs_update_true(self):
        """Test workflow needs update detection - true case."""
        registry = WorkflowRegistryService(workflows_dir="test_dir")

        existing_workflow = Mock()
        existing_workflow.description = "Old description"
        existing_workflow.workflow_metadata = {}
        existing_workflow.nodes = []
        existing_workflow.edges = []
        existing_workflow.max_execution_time = 3600

        yaml_data = {
            "name": "test_workflow",
            "version": "1.0.0",
            "description": "New description",  # Changed
            "workflow": {"nodes": [], "edges": []},
        }

        result = registry._workflow_needs_update(existing_workflow, yaml_data)
        assert result is True

    def test_workflow_needs_update_false(self):
        """Test workflow needs update detection - false case."""
        registry = WorkflowRegistryService(workflows_dir="test_dir")

        existing_workflow = Mock()
        existing_workflow.description = "Same description"
        existing_workflow.nodes = [{"id": "start"}]
        existing_workflow.edges = [{"from": "start", "to": "end"}]
        existing_workflow.max_execution_time = 1800
        existing_workflow.max_parallel_nodes = 10
        existing_workflow.retry_policy = {}
        existing_workflow.tags = []

        # Use the DB format that _workflow_needs_update expects
        db_format_data = {
            "description": "Same description",
            "nodes": [{"id": "start"}],
            "edges": [{"from": "start", "to": "end"}],
            "max_execution_time": 1800,
            "max_parallel_nodes": 10,
            "retry_policy": {},
            "tags": [],
        }

        result = registry._workflow_needs_update(existing_workflow, db_format_data)
        assert result is False

    def test_workflow_needs_update_nodes_changed(self):
        """Test workflow needs update when nodes change."""
        registry = WorkflowRegistryService(workflows_dir="test_dir")

        existing_workflow = Mock()
        existing_workflow.description = "Test"
        existing_workflow.workflow_metadata = {}
        existing_workflow.nodes = [{"id": "start"}]
        existing_workflow.edges = []
        existing_workflow.max_execution_time = 3600

        yaml_data = {
            "name": "test_workflow",
            "version": "1.0.0",
            "description": "Test",
            "workflow": {
                "nodes": [{"id": "start"}, {"id": "end"}],  # Added node
                "edges": [],
            },
        }

        result = registry._workflow_needs_update(existing_workflow, yaml_data)
        assert result is True

    def test_workflow_needs_update_edges_changed(self):
        """Test workflow needs update when edges change."""
        registry = WorkflowRegistryService(workflows_dir="test_dir")

        existing_workflow = Mock()
        existing_workflow.description = "Test"
        existing_workflow.workflow_metadata = {}
        existing_workflow.nodes = []
        existing_workflow.edges = []
        existing_workflow.max_execution_time = 3600

        yaml_data = {
            "name": "test_workflow",
            "version": "1.0.0",
            "description": "Test",
            "workflow": {
                "nodes": [],
                "edges": [{"from": "start", "to": "end"}],  # Added edge
            },
        }

        result = registry._workflow_needs_update(existing_workflow, yaml_data)
        assert result is True

    def test_workflow_needs_update_metadata_changed(self):
        """Test workflow needs update when metadata changes."""
        registry = WorkflowRegistryService(workflows_dir="test_dir")

        existing_workflow = Mock()
        existing_workflow.description = "Test"
        existing_workflow.workflow_metadata = {
            "yaml_metadata": {"old": "value"},
            "yaml_config": {},
        }
        existing_workflow.nodes = []
        existing_workflow.edges = []
        existing_workflow.max_execution_time = 1800

        yaml_data = {
            "name": "test_workflow",
            "version": "1.0.0",
            "description": "Test",
            "workflow": {"nodes": [], "edges": []},
            "metadata": {"new": "value"},  # Changed metadata
        }

        result = registry._workflow_needs_update(existing_workflow, yaml_data)
        assert result is True

    def test_workflow_needs_update_execution_time_changed(self):
        """Test workflow needs update when max execution time changes."""
        registry = WorkflowRegistryService(workflows_dir="test_dir")

        existing_workflow = Mock()
        existing_workflow.description = "Test"
        existing_workflow.workflow_metadata = {"yaml_metadata": {}, "yaml_config": {}}
        existing_workflow.nodes = []
        existing_workflow.edges = []
        existing_workflow.max_execution_time = 1800

        yaml_data = {
            "name": "test_workflow",
            "version": "1.0.0",
            "description": "Test",
            "workflow": {"nodes": [], "edges": []},
            "config": {"max_execution_time": 900},  # Changed time
        }

        result = registry._workflow_needs_update(existing_workflow, yaml_data)
        assert result is True

    @pytest.mark.asyncio
    async def test_create_workflow_success(self):
        """Test successful workflow creation."""
        registry = WorkflowRegistryService(workflows_dir="test_dir")

        mock_session = AsyncMock()
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        workflow_data = {
            "name": "test_workflow",
            "version": "1.0.0",
            "description": "Test",
            "nodes": [],
            "edges": [],
            "workflow_metadata": {},
            "max_execution_time": 3600,
        }

        with patch("app.services.workflow_registry.Workflow") as mock_workflow_class:
            mock_workflow = Mock()
            mock_workflow.id = "test-uuid"
            mock_workflow_class.return_value = mock_workflow

            result = await registry.create_workflow(mock_session, workflow_data)

            assert result == mock_workflow
            mock_session.add.assert_called_once_with(mock_workflow)
            mock_session.commit.assert_called_once()
            mock_session.refresh.assert_called_once_with(mock_workflow)

    @pytest.mark.asyncio
    async def test_create_workflow_error_handling(self):
        """Test workflow creation error handling."""
        registry = WorkflowRegistryService(workflows_dir="test_dir")

        mock_session = AsyncMock()
        mock_session.add = Mock(side_effect=Exception("Database error"))

        workflow_data = {
            "name": "test_workflow",
            "version": "1.0.0",
            "description": "Test",
            "nodes": [],
            "edges": [],
            "workflow_metadata": {},
            "max_execution_time": 3600,
        }

        with patch("app.services.workflow_registry.Workflow"):
            with pytest.raises(Exception, match="Database error"):
                await registry.create_workflow(mock_session, workflow_data)
