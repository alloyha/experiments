"""
Test suite for YAML Workflow System.

Tests the complete YAML workflow loading, validation, and registry system
following TDD approach for the HIL Agent System.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from app.services.workflow_loader import (
    WorkflowLoaderError,
    WorkflowValidationError,
    YAMLWorkflowLoader,
)
from app.services.workflow_registry import (
    WorkflowRegistryError,
    WorkflowRegistryService,
)


class TestYAMLWorkflowLoader:
    """Test YAML workflow loader functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workflows_dir = Path(self.temp_dir) / "workflows"
        self.workflows_dir.mkdir()

        # Create a simple test workflow
        self.test_workflow = {
            "name": "test_workflow",
            "version": "1.0.0",
            "description": "Test workflow for unit testing",
            "workflow": {
                "nodes": [
                    {
                        "id": "test_node",
                        "type": "agent",
                        "agent": "simple_test_agent",
                        "config": {"model_profile": "fast"},
                    }
                ],
                "edges": [{"from": "start", "to": "test_node"}],
            },
        }

    def test_loader_initialization(self):
        """Test workflow loader initialization."""
        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))

        assert loader.workflows_dir == self.workflows_dir
        assert loader.schema is not None  # Should load schema

    def test_discover_workflows_empty_directory(self):
        """Test workflow discovery in empty directory."""
        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))
        workflows = loader.discover_workflows()

        assert workflows == []

    def test_discover_workflows_with_files(self):
        """Test workflow discovery with YAML files."""
        # Create test workflow files
        (self.workflows_dir / "workflow1.yaml").write_text(
            yaml.dump(self.test_workflow)
        )
        (self.workflows_dir / "workflow2.yml").write_text(yaml.dump(self.test_workflow))
        (self.workflows_dir / "not_workflow.txt").write_text("ignore me")

        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))
        workflows = loader.discover_workflows()

        assert len(workflows) == 2
        assert any("workflow1.yaml" in w for w in workflows)
        assert any("workflow2.yml" in w for w in workflows)
        assert not any("not_workflow.txt" in w for w in workflows)

    def test_load_workflow_success(self):
        """Test successful workflow loading."""
        workflow_file = self.workflows_dir / "test.yaml"
        workflow_file.write_text(yaml.dump(self.test_workflow))

        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))
        loaded_workflow = loader.load_workflow(str(workflow_file))

        assert loaded_workflow["name"] == "test_workflow"
        assert loaded_workflow["version"] == "1.0.0"
        assert "_metadata" in loaded_workflow
        assert loaded_workflow["_metadata"]["file_name"] == "test.yaml"

    def test_load_workflow_file_not_found(self):
        """Test workflow loading with non-existent file."""
        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))

        with pytest.raises(WorkflowLoaderError, match="Workflow file not found"):
            loader.load_workflow("nonexistent.yaml")

    def test_load_workflow_invalid_yaml(self):
        """Test workflow loading with invalid YAML."""
        workflow_file = self.workflows_dir / "invalid.yaml"
        workflow_file.write_text("invalid: yaml: content: [")

        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))

        with pytest.raises(WorkflowLoaderError, match="YAML parsing error"):
            loader.load_workflow(str(workflow_file))

    def test_load_workflow_empty_file(self):
        """Test workflow loading with empty file."""
        workflow_file = self.workflows_dir / "empty.yaml"
        workflow_file.write_text("")

        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))

        with pytest.raises(WorkflowLoaderError, match="Empty or invalid YAML"):
            loader.load_workflow(str(workflow_file))

    def test_load_all_workflows(self):
        """Test loading all workflows from directory."""
        # Create multiple workflow files
        workflow1 = self.test_workflow.copy()
        workflow1["name"] = "workflow_one"

        workflow2 = self.test_workflow.copy()
        workflow2["name"] = "workflow_two"
        workflow2["version"] = "2.0.0"

        (self.workflows_dir / "workflow1.yaml").write_text(yaml.dump(workflow1))
        (self.workflows_dir / "workflow2.yaml").write_text(yaml.dump(workflow2))

        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))
        all_workflows = loader.load_all_workflows()

        assert len(all_workflows) == 2
        assert "workflow_one.v1.0.0" in all_workflows
        assert "workflow_two.v2.0.0" in all_workflows

    def test_get_workflow_versions(self):
        """Test getting all versions of a specific workflow."""
        # Create multiple versions of the same workflow
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            workflow = self.test_workflow.copy()
            workflow["version"] = version

            workflow_file = self.workflows_dir / f"test_workflow.v{version}.yaml"
            workflow_file.write_text(yaml.dump(workflow))

        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))
        versions = loader.get_workflow_versions("test_workflow")

        assert len(versions) == 3
        assert versions[0][0] == "1.0.0"  # First version
        assert versions[-1][0] == "2.0.0"  # Last version

    def test_get_latest_workflow(self):
        """Test getting the latest version of a workflow."""
        # Create multiple versions
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            workflow = self.test_workflow.copy()
            workflow["version"] = version

            workflow_file = self.workflows_dir / f"test_workflow.v{version}.yaml"
            workflow_file.write_text(yaml.dump(workflow))

        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))
        latest = loader.get_latest_workflow("test_workflow")

        assert latest is not None
        assert latest["version"] == "2.0.0"

    def test_get_latest_workflow_not_found(self):
        """Test getting latest workflow when none exists."""
        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))
        latest = loader.get_latest_workflow("nonexistent_workflow")

        assert latest is None

    def test_validate_workflow_file_success(self):
        """Test workflow file validation success."""
        workflow_file = self.workflows_dir / "valid.yaml"
        workflow_file.write_text(yaml.dump(self.test_workflow))

        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))
        is_valid, error = loader.validate_workflow_file(str(workflow_file))

        assert is_valid is True
        assert error is None

    def test_validate_workflow_file_failure(self):
        """Test workflow file validation failure."""
        invalid_workflow = {"name": "test"}  # Missing required fields

        workflow_file = self.workflows_dir / "invalid.yaml"
        workflow_file.write_text(yaml.dump(invalid_workflow))

        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))
        is_valid, error = loader.validate_workflow_file(str(workflow_file))

        assert is_valid is False
        assert error is not None
        assert "validation" in error.lower()


class TestWorkflowRegistryService:
    """Test workflow registry service functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workflows_dir = Path(self.temp_dir) / "workflows"
        self.workflows_dir.mkdir()

        # Create test workflow
        self.test_workflow_yaml = {
            "name": "test_registry_workflow",
            "version": "1.0.0",
            "description": "Test workflow for registry testing",
            "metadata": {"author": "test_author", "tags": ["test", "registry"]},
            "config": {"max_execution_time": 300, "retry_policy": {"max_attempts": 2}},
            "workflow": {
                "nodes": [
                    {"id": "test_node", "type": "agent", "agent": "simple_test_agent"}
                ],
                "edges": [{"from": "start", "to": "test_node"}],
            },
        }

    def test_registry_initialization(self):
        """Test workflow registry initialization."""
        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))

        assert registry.loader is not None
        assert registry.loader.workflows_dir == self.workflows_dir

    def test_yaml_to_db_format_conversion(self):
        """Test conversion from YAML format to database format."""
        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))
        db_data = registry._yaml_to_db_format(self.test_workflow_yaml)

        assert db_data["name"] == "test_registry_workflow"
        assert db_data["version"] == "1.0.0"
        assert db_data["description"] == "Test workflow for registry testing"
        assert db_data["max_execution_time"] == 300
        assert db_data["tags"] == ["test", "registry"]
        assert db_data["created_by"] == "test_author"
        assert "nodes" in db_data
        assert "edges" in db_data

    def test_workflow_needs_update_true(self):
        """Test workflow update detection when update is needed."""
        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))

        # Mock existing workflow
        existing = Mock()
        existing.name = "test_workflow"
        existing.description = "Old description"
        existing.nodes = [{"id": "old_node"}]
        existing.edges = []
        existing.max_execution_time = 1800
        existing.max_parallel_nodes = 10
        existing.retry_policy = {}
        existing.tags = []

        # New data with changes
        new_data = {
            "description": "New description",  # Changed
            "nodes": [{"id": "old_node"}],
            "edges": [],
            "max_execution_time": 1800,
            "max_parallel_nodes": 10,
            "retry_policy": {},
            "tags": [],
        }

        needs_update = registry._workflow_needs_update(existing, new_data)
        assert needs_update is True

    def test_workflow_needs_update_false(self):
        """Test workflow update detection when no update is needed."""
        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))

        # Mock existing workflow
        existing = Mock()
        existing.name = "test_workflow"
        existing.description = "Same description"
        existing.nodes = [{"id": "same_node"}]
        existing.edges = []
        existing.max_execution_time = 1800
        existing.max_parallel_nodes = 10
        existing.retry_policy = {}
        existing.tags = []

        # New data - identical
        new_data = {
            "description": "Same description",
            "nodes": [{"id": "same_node"}],
            "edges": [],
            "max_execution_time": 1800,
            "max_parallel_nodes": 10,
            "retry_policy": {},
            "tags": [],
        }

        needs_update = registry._workflow_needs_update(existing, new_data)
        assert needs_update is False

    @pytest.mark.asyncio
    async def test_create_workflow(self):
        """Test creating a workflow directly in database."""
        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))

        # Mock database session
        mock_session = AsyncMock()

        # Make session.add() a regular Mock (not async)
        mock_session.add = Mock()

        mock_workflow = Mock()
        mock_workflow.id = "test-uuid"
        mock_workflow.name = "test_workflow"
        mock_workflow.version = "1.0.0"

        # Mock session.refresh to set the workflow object
        async def mock_refresh(obj):
            obj.id = "test-uuid"

        mock_session.refresh = mock_refresh

        with patch("app.services.workflow_registry.Workflow") as mock_workflow_class:
            mock_workflow_class.return_value = mock_workflow

            workflow_data = {
                "name": "test_workflow",
                "version": "1.0.0",
                "nodes": [],
                "edges": [],
            }

            result = await registry.create_workflow(mock_session, workflow_data)

            assert result == mock_workflow
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    def test_validate_yaml_workflows(self):
        """Test validation of all YAML workflow files."""
        # Create valid and invalid workflow files
        valid_workflow = self.test_workflow_yaml
        invalid_workflow = {"name": "invalid"}  # Missing required fields

        (self.workflows_dir / "valid.yaml").write_text(yaml.dump(valid_workflow))
        (self.workflows_dir / "invalid.yaml").write_text(yaml.dump(invalid_workflow))

        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))
        report = registry.validate_yaml_workflows()

        assert report["total_files"] == 2
        assert len(report["valid"]) == 1
        assert len(report["invalid"]) == 1
        assert "valid.yaml" in report["valid"][0]
        assert "invalid.yaml" in report["invalid"][0]["file"]


class TestYAMLWorkflowIntegration:
    """Integration tests for the complete YAML workflow system."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workflows_dir = Path(self.temp_dir) / "workflows"
        self.workflows_dir.mkdir()

        # Create a complete workflow example
        self.complete_workflow = {
            "name": "integration_test_workflow",
            "version": "1.0.0",
            "description": "Complete workflow for integration testing",
            "metadata": {
                "author": "integration_test",
                "created_at": "2025-10-12",
                "tags": ["integration", "test", "complete"],
                "category": "testing",
            },
            "config": {
                "max_execution_time": 600,
                "max_parallel_nodes": 3,
                "retry_policy": {
                    "max_attempts": 3,
                    "backoff_factor": 2.0,
                    "exceptions": ["LLMProviderError"],
                },
            },
            "workflow": {
                "input_schema": {
                    "type": "object",
                    "required": ["message"],
                    "properties": {"message": {"type": "string"}},
                },
                "nodes": [
                    {
                        "id": "classify",
                        "type": "agent",
                        "agent": "simple_classifier",
                        "config": {"model_profile": "fast"},
                        "prompt": "Classify: {{input.message}}",
                    },
                    {
                        "id": "process",
                        "type": "agent",
                        "agent": "reasoning_processor",
                        "config": {"model_profile": "balanced"},
                        "condition": {
                            "type": "jmespath",
                            "expression": "classify.output.confidence > 0.8",
                        },
                    },
                ],
                "edges": [
                    {"from": "start", "to": "classify"},
                    {
                        "from": "classify",
                        "to": "process",
                        "condition": {
                            "type": "jmespath",
                            "expression": "output.confidence > 0.8",
                        },
                    },
                ],
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
            },
        }

    def test_end_to_end_workflow_loading_and_registry(self):
        """Test complete end-to-end workflow loading and registry."""
        # 1. Create workflow file
        workflow_file = self.workflows_dir / "complete_workflow.yaml"
        workflow_file.write_text(yaml.dump(self.complete_workflow))

        # 2. Initialize loader and registry
        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))
        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))

        # 3. Discover and load workflows
        discovered = loader.discover_workflows()
        assert len(discovered) == 1

        # 4. Load specific workflow
        loaded_workflow = loader.load_workflow(str(workflow_file))
        assert loaded_workflow["name"] == "integration_test_workflow"
        assert len(loaded_workflow["workflow"]["nodes"]) == 2

        # 5. Convert to database format
        db_format = registry._yaml_to_db_format(loaded_workflow)
        assert db_format["name"] == "integration_test_workflow"
        assert db_format["max_execution_time"] == 600
        assert len(db_format["nodes"]) == 2

        # 6. Validate workflow
        is_valid, error = loader.validate_workflow_file(str(workflow_file))
        assert is_valid is True
        assert error is None

    def test_multiple_workflow_versions_handling(self):
        """Test handling of multiple workflow versions."""
        base_workflow = self.complete_workflow.copy()

        # Create multiple versions
        versions = ["1.0.0", "1.1.0", "2.0.0"]
        for version in versions:
            workflow = base_workflow.copy()
            workflow["version"] = version
            workflow["description"] = f"Version {version} of test workflow"

            workflow_file = self.workflows_dir / f"test_workflow.v{version}.yaml"
            workflow_file.write_text(yaml.dump(workflow))

        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))

        # Test version discovery
        all_workflows = loader.load_all_workflows()
        assert len(all_workflows) == 3

        # Test version-specific loading
        versions_found = loader.get_workflow_versions("integration_test_workflow")
        assert len(versions_found) == 3

        # Test latest version
        latest = loader.get_latest_workflow("integration_test_workflow")
        assert latest["version"] == "2.0.0"

    def test_workflow_validation_comprehensive(self):
        """Test comprehensive workflow validation."""
        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))

        # Test valid workflow
        valid_file = self.workflows_dir / "valid_complete.yaml"
        valid_file.write_text(yaml.dump(self.complete_workflow))

        is_valid, error = loader.validate_workflow_file(str(valid_file))
        assert is_valid is True
        assert error is None

        # Test invalid workflows
        invalid_cases = [
            {"name": "missing_version"},  # Missing version
            {"name": "test", "version": "1.0.0"},  # Missing workflow
            {
                "name": "test",
                "version": "invalid_version",
                "workflow": {},
            },  # Invalid version format
        ]

        for i, invalid_workflow in enumerate(invalid_cases):
            invalid_file = self.workflows_dir / f"invalid_{i}.yaml"
            invalid_file.write_text(yaml.dump(invalid_workflow))

            is_valid, error = loader.validate_workflow_file(str(invalid_file))
            assert is_valid is False
            assert error is not None


class TestWorkflowRegistryServiceAdvanced:
    """Advanced tests for WorkflowRegistryService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.workflows_dir = Path("test_workflows_advanced")
        self.workflows_dir.mkdir(exist_ok=True)

        # Create test YAML workflow files
        self.create_test_workflow_files()

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.workflows_dir.exists():
            shutil.rmtree(self.workflows_dir)

    def create_test_workflow_files(self):
        """Create test workflow YAML files."""
        # Simple workflow
        simple_workflow = {
            "name": "simple_workflow",
            "version": "1.0.0",
            "description": "Simple test workflow",
            "workflow": {
                "nodes": [
                    {"id": "start", "type": "start"},
                    {"id": "end", "type": "end"},
                ],
                "edges": [{"from": "start", "to": "end"}],
            },
        }

        with open(self.workflows_dir / "simple_workflow.v1.yaml", "w") as f:
            yaml.dump(simple_workflow, f)

        # Complex workflow
        complex_workflow = {
            "name": "complex_workflow",
            "version": "2.1.0",
            "description": "Complex test workflow",
            "workflow": {
                "nodes": [
                    {"id": "start", "type": "start"},
                    {"id": "agent1", "type": "agent", "agent_type": "simple"},
                    {"id": "decision", "type": "decision"},
                    {"id": "end", "type": "end"},
                ],
                "edges": [
                    {"from": "start", "to": "agent1"},
                    {"from": "agent1", "to": "decision"},
                    {"from": "decision", "to": "end", "condition": "success"},
                ],
            },
        }

        with open(self.workflows_dir / "complex_workflow.v2.yaml", "w") as f:
            yaml.dump(complex_workflow, f)

    @pytest.mark.asyncio
    async def test_sync_workflows_from_yaml_creates_new_workflows(self):
        """Test sync_workflows_from_yaml creates new workflows from YAML files."""
        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))

        # Mock the loader to return test workflows
        mock_yaml_workflows = {
            "test_workflow.v1.0.0": {
                "name": "test_workflow",
                "version": "1.0.0",
                "description": "Test workflow",
                "workflow": {
                    "nodes": [{"id": "start", "type": "action", "action": "test"}],
                    "edges": [],
                },
            }
        }

        # Mock database session
        mock_session = AsyncMock()
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()

        # Mock workflow queries to return None (no existing workflows)
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.exec.return_value = mock_result

        # Mock session operations to avoid actual database interaction
        added_workflows = []
        def mock_add(workflow):
            added_workflows.append(workflow)
        mock_session.add.side_effect = mock_add
        mock_session.flush = AsyncMock()

        with patch.object(
            registry.loader, "load_all_workflows", return_value=mock_yaml_workflows
        ):
            result = await registry.sync_workflows_from_yaml(
                mock_session
            )  # Verify results
            assert result["total_yaml_workflows"] == 1
            assert len(result["created"]) == 1
            assert len(result["updated"]) == 0
            assert len(result["skipped"]) == 0
            assert len(result["errors"]) == 0

            # Verify session calls
            assert mock_session.add.call_count == 1  # One workflow added
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_workflows_from_yaml_updates_existing_workflows(self):
        """Test sync_workflows_from_yaml updates existing workflows when needed."""
        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))

        # Mock the loader to return test workflows
        mock_yaml_workflows = {
            "simple_workflow.v1.0.0": {
                "name": "simple_workflow",
                "version": "1.0.0",
                "description": "Updated description",
                "workflow": {
                    "nodes": [{"id": "start", "type": "action", "action": "test"}],
                    "edges": [],
                },
            }
        }

        # Mock database session
        mock_session = AsyncMock()
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()

        # Mock existing workflow that needs update
        existing_workflow = Mock()
        existing_workflow.id = "existing-uuid"
        existing_workflow.name = "simple_workflow"
        existing_workflow.version = "1.0.0"
        existing_workflow.description = "Old description"  # Different from YAML
        existing_workflow.workflow_metadata = {}
        existing_workflow.nodes = []
        existing_workflow.edges = []

        mock_result = Mock()
        mock_result.first.return_value = existing_workflow
        mock_session.exec.return_value = mock_result

        with patch.object(
            registry.loader, "load_all_workflows", return_value=mock_yaml_workflows
        ):
            result = await registry.sync_workflows_from_yaml(
                mock_session
            )  # Verify results
        assert result["total_yaml_workflows"] == 1
        assert len(result["updated"]) == 1  # One workflow should be updated
        assert len(result["created"]) == 0
        assert len(result["skipped"]) == 0
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_sync_workflows_from_yaml_handles_errors(self):
        """Test sync_workflows_from_yaml handles individual workflow errors gracefully."""
        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))

        # Mock the loader to return test workflows
        mock_yaml_workflows = {
            "workflow1.v1.0.0": {
                "name": "workflow1",
                "version": "1.0.0",
                "description": "First workflow",
                "workflow": {"nodes": [], "edges": []},
            },
            "workflow2.v1.0.0": {
                "name": "workflow2",
                "version": "1.0.0",
                "description": "Second workflow",
                "workflow": {"nodes": [], "edges": []},
            },
        }

        # Mock database session
        mock_session = AsyncMock()
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        # Mock session.exec to raise an error for first workflow
        call_count = [0]

        async def mock_exec(stmt):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Database error")
            # Return None for second workflow (create new)
            mock_result = Mock()
            mock_result.first.return_value = None
            return mock_result

        mock_session.exec.side_effect = mock_exec

        # Mock session operations to avoid actual database interaction
        mock_session.flush = AsyncMock()

        with patch.object(
            registry.loader, "load_all_workflows", return_value=mock_yaml_workflows
        ):
            result = await registry.sync_workflows_from_yaml(
                mock_session
            )  # Verify results - one error, one success
            assert result["total_yaml_workflows"] == 2
            assert len(result["errors"]) == 1  # First workflow should error due to mock_exec
            assert len(result["created"]) == 1  # Second workflow should succeed
            assert len(result["updated"]) == 0
            assert len(result["skipped"]) == 0

    @pytest.mark.asyncio
    async def test_create_workflow_error_handling(self):
        """Test create_workflow handles errors properly."""
        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))

        # Mock database session that raises error
        mock_session = AsyncMock()
        mock_session.add = Mock(side_effect=Exception("Database error"))

        workflow_data = {
            "name": "test_workflow",
            "version": "1.0.0",
            "nodes": [],
            "edges": [],
        }

        with pytest.raises(WorkflowRegistryError) as exc_info:
            await registry.create_workflow(mock_session, workflow_data)

        assert "Failed to create workflow" in str(exc_info.value)
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_workflow_by_name_and_version(self):
        """Test retrieving workflow by name and version."""
        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))

        # Mock database session
        mock_session = AsyncMock()
        mock_workflow = Mock()
        mock_workflow.id = "test-uuid"
        mock_workflow.name = "test_workflow"
        mock_workflow.version = "1.0.0"

        mock_result = Mock()
        mock_result.first.return_value = mock_workflow
        mock_session.exec.return_value = mock_result

        result = await registry.get_workflow_by_name_version(
            mock_session, "test_workflow", "1.0.0"
        )

        assert result == mock_workflow
        mock_session.exec.assert_called_once()
        mock_session.exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_workflow_not_found(self):
        """Test retrieving non-existent workflow."""
        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))

        # Mock database session
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.first.return_value = None
        mock_session.exec.return_value = mock_result

        result = await registry.get_workflow_by_name_version(
            mock_session, "nonexistent", "1.0.0"
        )

        assert result is None
        mock_session.exec.assert_called_once()

    def test_yaml_to_db_format_conversion(self):
        """Test YAML data conversion to database format."""
        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))

        yaml_data = {
            "name": "test_workflow",
            "version": "1.0.0",
            "description": "Test workflow",
            "workflow": {
                "nodes": [{"id": "start", "type": "start"}],
                "edges": [{"from": "start", "to": "end"}],
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
            "metadata": {"author": "test_user", "tags": ["test"]},
            "config": {"max_execution_time": 300},
        }

        result = registry._yaml_to_db_format(yaml_data)

        assert result["name"] == "test_workflow"
        assert result["version"] == "1.0.0"
        assert result["description"] == "Test workflow"
        assert result["nodes"] == [{"id": "start", "type": "start"}]
        assert result["edges"] == [{"from": "start", "to": "end"}]
        assert result["max_execution_time"] == 300
        assert result["created_by"] == "test_user"
        assert result["tags"] == ["test"]
        assert "input_schema" in result["workflow_metadata"]

    def test_workflow_needs_update_detection(self):
        """Test workflow update detection logic."""
        registry = WorkflowRegistryService(workflows_dir=str(self.workflows_dir))

        # Mock existing workflow
        existing_workflow = Mock()
        existing_workflow.name = "test_workflow"
        existing_workflow.description = "Old description"
        existing_workflow.nodes = [{"id": "old"}]
        existing_workflow.edges = []
        existing_workflow.max_execution_time = 1800
        existing_workflow.max_parallel_nodes = 10
        existing_workflow.retry_policy = {}
        existing_workflow.tags = []

        # New data with changes
        new_data = {
            "description": "New description",  # Different
            "nodes": [{"id": "old"}],  # Same
            "edges": [],  # Same
            "max_execution_time": 1800,  # Same
            "max_parallel_nodes": 10,  # Same
            "retry_policy": {},  # Same
            "tags": [],  # Same
        }

        result = registry._workflow_needs_update(existing_workflow, new_data)
        assert result is True

        # Test no changes needed
        no_change_data = {
            "description": "Old description",  # Same
            "nodes": [{"id": "old"}],  # Same
            "edges": [],  # Same
            "max_execution_time": 1800,  # Same
            "max_parallel_nodes": 10,  # Same
            "retry_policy": {},  # Same
            "tags": [],  # Same
        }

        result = registry._workflow_needs_update(existing_workflow, no_change_data)
        assert result is False
