"""
Comprehensive tests for YAML Workflow Loader to improve coverage.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from app.services.workflow_loader import (
    WorkflowLoaderError,
    WorkflowValidationError,
    YAMLWorkflowLoader,
)


class TestYAMLWorkflowLoader:
    """Test cases for YAML Workflow Loader."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workflows_dir = Path(self.temp_dir) / "workflows"
        self.workflows_dir.mkdir(parents=True, exist_ok=True)

        # Create test schema
        self.schema_file = Path(self.temp_dir) / "schema.json"
        self.test_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "workflow": {"type": "object"},
            },
            "required": ["name", "version", "workflow"],
        }
        with open(self.schema_file, "w") as f:
            json.dump(self.test_schema, f)

    def test_init_default_paths(self):
        """Test initialization with default paths."""
        loader = YAMLWorkflowLoader()
        assert loader.workflows_dir is not None
        assert loader.schema_file is not None
        assert "workflows" in str(loader.workflows_dir)
        assert "workflow_schema.json" in str(loader.schema_file)

    def test_init_custom_paths(self):
        """Test initialization with custom paths."""
        loader = YAMLWorkflowLoader(
            workflows_dir=str(self.workflows_dir), schema_file=str(self.schema_file)
        )
        assert loader.workflows_dir == self.workflows_dir
        assert loader.schema_file == self.schema_file
        assert loader.schema == self.test_schema

    def test_load_schema_success(self):
        """Test successful schema loading."""
        loader = YAMLWorkflowLoader(schema_file=str(self.schema_file))
        assert loader.schema == self.test_schema

    def test_load_schema_file_not_found(self):
        """Test schema loading when file doesn't exist."""
        non_existent = Path(self.temp_dir) / "nonexistent.json"
        loader = YAMLWorkflowLoader(schema_file=str(non_existent))
        assert loader.schema is None

    def test_load_schema_invalid_json(self):
        """Test schema loading with invalid JSON."""
        invalid_schema = Path(self.temp_dir) / "invalid.json"
        with open(invalid_schema, "w") as f:
            f.write("invalid json content")

        loader = YAMLWorkflowLoader(schema_file=str(invalid_schema))
        assert loader.schema is None

    def test_discover_workflows_empty_directory(self):
        """Test workflow discovery in empty directory."""
        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))
        workflows = loader.discover_workflows()
        assert workflows == []

    def test_discover_workflows_nonexistent_directory(self):
        """Test workflow discovery in nonexistent directory."""
        nonexistent = Path(self.temp_dir) / "nonexistent"
        loader = YAMLWorkflowLoader(workflows_dir=str(nonexistent))
        workflows = loader.discover_workflows()
        assert workflows == []

    def test_discover_workflows_with_files(self):
        """Test workflow discovery with YAML files."""
        # Create test YAML files
        (self.workflows_dir / "workflow1.yaml").touch()
        (self.workflows_dir / "workflow2.yml").touch()
        (self.workflows_dir / "not_yaml.txt").touch()

        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))
        workflows = loader.discover_workflows()

        assert len(workflows) == 2
        assert any("workflow1.yaml" in w for w in workflows)
        assert any("workflow2.yml" in w for w in workflows)
        assert not any("not_yaml.txt" in w for w in workflows)

    def test_load_workflow_success(self):
        """Test successful workflow loading."""
        # Create valid workflow file
        workflow_content = {
            "name": "test_workflow",
            "version": "1.0.0",
            "description": "Test workflow",
            "workflow": {"nodes": [{"id": "start", "type": "action"}], "edges": []},
        }

        workflow_file = self.workflows_dir / "test.yaml"
        with open(workflow_file, "w") as f:
            import yaml

            yaml.dump(workflow_content, f)

        loader = YAMLWorkflowLoader(
            workflows_dir=str(self.workflows_dir), schema_file=str(self.schema_file)
        )

        result = loader.load_workflow(str(workflow_file))

        assert result["name"] == "test_workflow"
        assert result["version"] == "1.0.0"
        assert "_metadata" in result
        assert result["_metadata"]["file_name"] == "test.yaml"

    def test_load_workflow_file_not_found(self):
        """Test loading nonexistent workflow file."""
        loader = YAMLWorkflowLoader()

        with pytest.raises(WorkflowLoaderError, match="Workflow file not found"):
            loader.load_workflow("/nonexistent/file.yaml")

    def test_load_workflow_empty_file(self):
        """Test loading empty YAML file."""
        empty_file = self.workflows_dir / "empty.yaml"
        empty_file.touch()

        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))

        with pytest.raises(WorkflowLoaderError, match="Empty or invalid YAML file"):
            loader.load_workflow(str(empty_file))

    def test_load_workflow_invalid_yaml(self):
        """Test loading invalid YAML file."""
        invalid_file = self.workflows_dir / "invalid.yaml"
        with open(invalid_file, "w") as f:
            f.write("invalid: yaml: content: [")

        loader = YAMLWorkflowLoader(workflows_dir=str(self.workflows_dir))

        with pytest.raises(WorkflowLoaderError):
            loader.load_workflow(str(invalid_file))

    def test_load_workflow_validation_error(self):
        """Test workflow loading with validation error."""
        # Create invalid workflow (missing required fields)
        invalid_workflow = {
            "name": "test_workflow"
            # Missing required "version" and "workflow" fields
        }

        workflow_file = self.workflows_dir / "invalid.yaml"
        with open(workflow_file, "w") as f:
            import yaml

            yaml.dump(invalid_workflow, f)

        loader = YAMLWorkflowLoader(
            workflows_dir=str(self.workflows_dir), schema_file=str(self.schema_file)
        )

        with pytest.raises(WorkflowLoaderError, match="Failed to load workflow"):
            loader.load_workflow(str(workflow_file))

    def test_load_workflow_no_schema(self):
        """Test workflow loading without schema validation."""
        workflow_content = {"name": "test_workflow", "version": "1.0.0"}

        workflow_file = self.workflows_dir / "test.yaml"
        with open(workflow_file, "w") as f:
            import yaml

            yaml.dump(workflow_content, f)

        # Create loader without schema
        loader = YAMLWorkflowLoader(
            workflows_dir=str(self.workflows_dir),
            schema_file="/nonexistent/schema.json",
        )

        result = loader.load_workflow(str(workflow_file))
        assert result["name"] == "test_workflow"

    def test_validate_workflow_success(self):
        """Test successful workflow validation."""
        valid_workflow = {
            "name": "test_workflow",
            "version": "1.0.0",
            "description": "Test",
            "workflow": {"nodes": [], "edges": []},
        }

        loader = YAMLWorkflowLoader(schema_file=str(self.schema_file))
        # Should not raise exception
        loader._validate_workflow(valid_workflow, "test.yaml")

    def test_validate_workflow_failure(self):
        """Test workflow validation failure."""
        invalid_workflow = {
            "name": "test_workflow"
            # Missing required fields
        }

        loader = YAMLWorkflowLoader(schema_file=str(self.schema_file))

        with pytest.raises(WorkflowValidationError):
            loader._validate_workflow(invalid_workflow, "test.yaml")

    def test_get_workflow_versions(self):
        """Test getting workflow versions."""
        # Create workflow files with versions
        workflow1_v1 = {
            "name": "test_workflow",
            "version": "1.0.0",
            "description": "Version 1.0.0",
            "workflow": {"nodes": [], "edges": []},
        }

        workflow1_v2 = {
            "name": "test_workflow",
            "version": "2.0.0",
            "description": "Version 2.0.0",
            "workflow": {"nodes": [], "edges": []},
        }

        import yaml

        file1 = self.workflows_dir / "test_workflow.v1.0.0.yaml"
        with open(file1, "w") as f:
            yaml.dump(workflow1_v1, f)

        file2 = self.workflows_dir / "test_workflow.v2.0.0.yaml"
        with open(file2, "w") as f:
            yaml.dump(workflow1_v2, f)

        loader = YAMLWorkflowLoader(
            workflows_dir=str(self.workflows_dir), schema_file=str(self.schema_file)
        )

        versions = loader.get_workflow_versions("test_workflow")
        assert len(versions) == 2

        # Sort by version for consistent testing
        versions.sort(key=lambda x: x[0])
        assert versions[0][0] == "1.0.0"
        assert versions[1][0] == "2.0.0"

    def test_load_all_workflows_success(self):
        """Test loading all workflows in directory."""
        # Create multiple workflow files
        workflows = [
            {
                "name": "workflow1",
                "version": "1.0.0",
                "description": "First workflow",
                "workflow": {"nodes": [], "edges": []},
            },
            {
                "name": "workflow2",
                "version": "2.0.0",
                "description": "Second workflow",
                "workflow": {"nodes": [], "edges": []},
            },
        ]

        import yaml

        for i, workflow in enumerate(workflows, 1):
            file_path = self.workflows_dir / f"workflow{i}.v{workflow['version']}.yaml"
            with open(file_path, "w") as f:
                yaml.dump(workflow, f)

        loader = YAMLWorkflowLoader(
            workflows_dir=str(self.workflows_dir), schema_file=str(self.schema_file)
        )

        result = loader.load_all_workflows()

        assert len(result) == 2
        assert "workflow1.v1.0.0" in result
        assert "workflow2.v2.0.0" in result
        assert result["workflow1.v1.0.0"]["name"] == "workflow1"
        assert result["workflow2.v2.0.0"]["name"] == "workflow2"

    def test_load_all_workflows_with_errors(self):
        """Test loading workflows with some errors."""
        # Create one valid and one invalid workflow
        valid_workflow = {
            "name": "valid_workflow",
            "version": "1.0.0",
            "description": "Valid",
            "workflow": {"nodes": [], "edges": []},
        }

        invalid_workflow = {
            "name": "invalid_workflow"
            # Missing required fields
        }

        import yaml

        valid_file = self.workflows_dir / "valid.yaml"
        with open(valid_file, "w") as f:
            yaml.dump(valid_workflow, f)

        invalid_file = self.workflows_dir / "invalid.yaml"
        with open(invalid_file, "w") as f:
            yaml.dump(invalid_workflow, f)

        loader = YAMLWorkflowLoader(
            workflows_dir=str(self.workflows_dir), schema_file=str(self.schema_file)
        )

        result = loader.load_all_workflows()

        # Should only contain the valid workflow
        assert len(result) == 1
        assert any("valid_workflow" in key for key in result.keys())

    def test_cleanup_temp_files(self):
        """Clean up temporary test files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
