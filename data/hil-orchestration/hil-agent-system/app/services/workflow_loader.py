"""
YAML Workflow Loader for HIL Agent System.

Handles loading, parsing, and validating YAML workflow definitions
according to the HIL Agent System workflow schema.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from jsonschema import (
    ValidationError as JSONSchemaValidationError,
    validate,
)

from app.core.exceptions import HILAgentError
from app.core.logging import get_logger

logger = get_logger(__name__)


class WorkflowValidationError(HILAgentError):
    """Exception raised when workflow validation fails."""


class WorkflowLoaderError(HILAgentError):
    """Exception raised when workflow loading fails."""


class YAMLWorkflowLoader:
    """
    Loads and validates YAML workflow definitions.

    Features:
    - YAML parsing with error handling
    - JSON Schema validation
    - Version management
    - File discovery and loading
    """

    def __init__(
        self, workflows_dir: str | None = None, schema_file: str | None = None
    ):
        """
        Initialize workflow loader.

        Args:
            workflows_dir: Directory containing workflow YAML files
            schema_file: Path to JSON schema for validation
        """
        # Default paths
        base_dir = Path(__file__).parent.parent.parent
        self.workflows_dir = (
            Path(workflows_dir) if workflows_dir else base_dir / "config" / "workflows"
        )
        self.schema_file = (
            Path(schema_file)
            if schema_file
            else base_dir / "config" / "workflow_schema.json"
        )

        # Load schema for validation
        self.schema = self._load_schema()

        logger.info(
            "YAML Workflow Loader initialized",
            extra={
                "workflows_dir": str(self.workflows_dir),
                "schema_file": str(self.schema_file),
                "schema_loaded": self.schema is not None,
            },
        )

    def _load_schema(self) -> dict[str, Any] | None:
        """Load JSON schema for validation."""
        try:
            if self.schema_file.exists():
                with open(self.schema_file, encoding="utf-8") as f:
                    schema = json.load(f)
                logger.debug(f"Loaded workflow schema from {self.schema_file}")
                return schema
            logger.warning(f"Schema file not found: {self.schema_file}")
            return None
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            return None

    def discover_workflows(self) -> list[str]:
        """
        Discover all workflow YAML files in the workflows directory.

        Returns:
            List of workflow file paths
        """
        if not self.workflows_dir.exists():
            logger.warning(f"Workflows directory does not exist: {self.workflows_dir}")
            return []

        workflow_files = []

        # Find all .yaml and .yml files
        for pattern in ["*.yaml", "*.yml"]:
            workflow_files.extend(self.workflows_dir.glob(pattern))

        # Convert to string paths and sort
        file_paths = [str(f) for f in workflow_files]
        file_paths.sort()

        logger.info(f"Discovered {len(file_paths)} workflow files")
        return file_paths

    def load_workflow(self, file_path: str) -> dict[str, Any]:
        """
        Load and validate a single workflow YAML file.

        Args:
            file_path: Path to workflow YAML file

        Returns:
            Parsed and validated workflow definition

        Raises:
            WorkflowLoaderError: If loading fails
            WorkflowValidationError: If validation fails
        """
        try:
            # Load YAML file
            workflow_path = Path(file_path)
            if not workflow_path.exists():
                raise WorkflowLoaderError(f"Workflow file not found: {file_path}")

            with open(workflow_path, encoding="utf-8") as f:
                workflow_data = yaml.safe_load(f)

            if not workflow_data:
                raise WorkflowLoaderError(f"Empty or invalid YAML file: {file_path}")

            logger.debug(f"Loaded YAML workflow from {file_path}")

            # Validate against schema
            if self.schema:
                self._validate_workflow(workflow_data, file_path)
            else:
                logger.warning("No schema available for validation")

            # Add metadata
            workflow_data["_metadata"] = {
                "file_path": str(workflow_path),
                "file_name": workflow_path.name,
                "last_modified": workflow_path.stat().st_mtime,
            }

            logger.info(
                f"Successfully loaded workflow: {workflow_data.get('name', 'unknown')}",
                extra={
                    "workflow_name": workflow_data.get("name"),
                    "version": workflow_data.get("version"),
                    "file_path": file_path,
                },
            )

            return workflow_data

        except yaml.YAMLError as e:
            error_msg = f"YAML parsing error in {file_path}: {e}"
            logger.error(error_msg)
            raise WorkflowLoaderError(error_msg)
        except Exception as e:
            error_msg = f"Failed to load workflow {file_path}: {e}"
            logger.error(error_msg)
            raise WorkflowLoaderError(error_msg)

    def _validate_workflow(self, workflow_data: dict[str, Any], file_path: str) -> None:
        """
        Validate workflow data against JSON schema.

        Args:
            workflow_data: Parsed workflow data
            file_path: Path to workflow file (for error messages)

        Raises:
            WorkflowValidationError: If validation fails
        """
        try:
            validate(instance=workflow_data, schema=self.schema)
            logger.debug(f"Workflow validation passed for {file_path}")

        except JSONSchemaValidationError as e:
            error_msg = f"Workflow validation failed for {file_path}: {e.message}"
            if e.absolute_path:
                error_msg += (
                    f" (at path: {' -> '.join(str(p) for p in e.absolute_path)})"
                )

            logger.error(error_msg)
            raise WorkflowValidationError(error_msg)

    def load_all_workflows(self) -> dict[str, dict[str, Any]]:
        """
        Load all workflows from the workflows directory.

        Returns:
            Dictionary mapping workflow names to workflow definitions

        Raises:
            WorkflowLoaderError: If loading fails
        """
        workflows = {}
        workflow_files = self.discover_workflows()

        if not workflow_files:
            logger.warning("No workflow files found")
            return workflows

        failed_loads = []

        for file_path in workflow_files:
            try:
                workflow_data = self.load_workflow(file_path)
                workflow_name = workflow_data.get("name")

                if not workflow_name:
                    logger.warning(f"Workflow in {file_path} has no name, skipping")
                    continue

                # Handle versioned workflows
                version = workflow_data.get("version", "1.0.0")
                workflow_key = f"{workflow_name}.v{version}"

                workflows[workflow_key] = workflow_data

            except (WorkflowLoaderError, WorkflowValidationError) as e:
                logger.error(f"Failed to load {file_path}: {e}")
                failed_loads.append(file_path)
                continue

        logger.info(
            f"Loaded {len(workflows)} workflows from {len(workflow_files)} files",
            extra={
                "total_files": len(workflow_files),
                "successful_loads": len(workflows),
                "failed_loads": len(failed_loads),
            },
        )

        if failed_loads and len(failed_loads) == len(workflow_files):
            raise WorkflowLoaderError("Failed to load any workflows")

        return workflows

    def get_workflow_versions(
        self, workflow_name: str
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Get all versions of a specific workflow.

        Args:
            workflow_name: Name of the workflow

        Returns:
            List of (version, workflow_data) tuples, sorted by version
        """
        all_workflows = self.load_all_workflows()
        versions = []

        for key, workflow_data in all_workflows.items():
            if key.startswith(f"{workflow_name}.v"):
                version = workflow_data.get("version", "1.0.0")
                versions.append((version, workflow_data))

        # Sort by version (simple string sort should work for semantic versions)
        versions.sort(key=lambda x: x[0])

        logger.debug(f"Found {len(versions)} versions for workflow {workflow_name}")
        return versions

    def get_latest_workflow(self, workflow_name: str) -> dict[str, Any] | None:
        """
        Get the latest version of a specific workflow.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Latest workflow data or None if not found
        """
        versions = self.get_workflow_versions(workflow_name)

        if not versions:
            logger.warning(f"No versions found for workflow: {workflow_name}")
            return None

        latest_version, latest_workflow = versions[-1]  # Last in sorted list

        logger.debug(f"Latest version of {workflow_name} is {latest_version}")
        return latest_workflow

    def validate_workflow_file(self, file_path: str) -> tuple[bool, str | None]:
        """
        Validate a workflow file without loading it into memory.

        Args:
            file_path: Path to workflow file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self.load_workflow(file_path)
            return True, None
        except (WorkflowLoaderError, WorkflowValidationError) as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {e}"
