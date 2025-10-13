"""
Workflow Registry Service for HIL Agent System.

Manages workflow definitions, versions, and lifecycle operations.
Provides bridge between YAML workflow files and database storage.
"""

import uuid
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.core.exceptions import HILAgentError
from app.core.logging import get_logger
from app.models.workflow import Workflow, WorkflowBase, WorkflowStatus
from app.services.workflow_loader import WorkflowLoaderError, YAMLWorkflowLoader

logger = get_logger(__name__)


class WorkflowRegistryError(HILAgentError):
    """Exception raised by workflow registry operations."""


class WorkflowRegistryService:
    """
    Workflow Registry Service for managing workflow definitions.

    Features:
    - YAML workflow loading and database synchronization
    - Version management and lifecycle operations
    - Workflow discovery and retrieval
    - Execution history tracking
    - A/B testing support
    """

    def __init__(self, workflows_dir: str | None = None):
        """
        Initialize workflow registry service.

        Args:
            workflows_dir: Directory containing workflow YAML files
        """
        self.loader = YAMLWorkflowLoader(workflows_dir=workflows_dir)

        logger.info(
            "Workflow Registry Service initialized",
            extra={"workflows_dir": str(self.loader.workflows_dir)},
        )

    async def sync_workflows_from_yaml(self, session: AsyncSession) -> dict[str, Any]:
        """
        Synchronize workflows from YAML files to database.

        Args:
            session: Database session

        Returns:
            Synchronization report

        Raises:
            WorkflowRegistryError: If synchronization fails
        """
        try:
            # Load all workflows from YAML files
            yaml_workflows = self.loader.load_all_workflows()

            sync_report = {
                "total_yaml_workflows": len(yaml_workflows),
                "created": [],
                "updated": [],
                "skipped": [],
                "errors": [],
            }

            for workflow_key, yaml_data in yaml_workflows.items():
                try:
                    result = await self._sync_single_workflow(session, yaml_data)
                    sync_report[result["action"]].append(
                        {
                            "name": yaml_data["name"],
                            "version": yaml_data["version"],
                            "details": result.get("details"),
                        }
                    )

                except Exception as e:
                    logger.error(f"Failed to sync workflow {workflow_key}: {e}")
                    sync_report["errors"].append(
                        {"workflow": workflow_key, "error": str(e)}
                    )

            await session.commit()

            logger.info(
                "Workflow synchronization completed",
                extra={
                    "total": sync_report["total_yaml_workflows"],
                    "created": len(sync_report["created"]),
                    "updated": len(sync_report["updated"]),
                    "skipped": len(sync_report["skipped"]),
                    "errors": len(sync_report["errors"]),
                },
            )

            return sync_report

        except Exception as e:
            await session.rollback()
            error_msg = f"Workflow synchronization failed: {e}"
            logger.error(error_msg)
            raise WorkflowRegistryError(error_msg)

    async def _sync_single_workflow(
        self, session: AsyncSession, yaml_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Synchronize a single workflow to database.

        Args:
            session: Database session
            yaml_data: YAML workflow data

        Returns:
            Sync operation result
        """
        workflow_name = yaml_data["name"]
        workflow_version = yaml_data["version"]

        # Check if workflow already exists
        stmt = select(Workflow).where(
            Workflow.name == workflow_name, Workflow.version == workflow_version
        )
        existing_workflow = await session.exec(stmt)
        existing_workflow = existing_workflow.first()

        # Convert YAML data to database model
        db_workflow_data = self._yaml_to_db_format(yaml_data)

        if existing_workflow:
            # Check if update is needed
            if self._workflow_needs_update(existing_workflow, db_workflow_data):
                # Update existing workflow
                for field, value in db_workflow_data.items():
                    setattr(existing_workflow, field, value)

                existing_workflow.updated_at = datetime.now(UTC)
                session.add(existing_workflow)

                logger.debug(f"Updated workflow {workflow_name} v{workflow_version}")
                return {"action": "updated", "workflow_id": existing_workflow.id}
            logger.debug(f"Workflow {workflow_name} v{workflow_version} unchanged")
            return {"action": "skipped", "workflow_id": existing_workflow.id}
        # Create new workflow
        new_workflow = Workflow(**db_workflow_data)
        session.add(new_workflow)
        await session.flush()  # Get ID

        logger.debug(f"Created workflow {workflow_name} v{workflow_version}")
        return {"action": "created", "workflow_id": new_workflow.id}

    def _yaml_to_db_format(self, yaml_data: dict[str, Any]) -> dict[str, Any]:
        """
        Convert YAML workflow data to database format.

        Args:
            yaml_data: YAML workflow definition

        Returns:
            Database-compatible workflow data
        """
        workflow_def = yaml_data.get("workflow", {})
        metadata = yaml_data.get("metadata", {})
        config = yaml_data.get("config", {})

        # Extract core fields
        db_data = {
            "name": yaml_data["name"],
            "description": yaml_data.get("description"),
            "version": yaml_data["version"],
            "status": WorkflowStatus.ACTIVE,  # New workflows are active by default
            # Workflow definition
            "nodes": workflow_def.get("nodes", []),
            "edges": workflow_def.get("edges", []),
            # Configuration
            "max_execution_time": config.get("max_execution_time", 1800),
            "max_parallel_nodes": config.get("max_parallel_nodes", 10),
            "retry_policy": config.get("retry_policy", {}),
            # Metadata
            "tags": metadata.get("tags", []),
            "workflow_metadata": {
                "input_schema": workflow_def.get("input_schema"),
                "output_schema": workflow_def.get("output_schema"),
                "yaml_metadata": metadata,
                "yaml_config": config,
            },
            # Audit
            "created_by": metadata.get("author", "yaml_import"),
        }

        return db_data

    def _workflow_needs_update(
        self, existing: Workflow, new_data: dict[str, Any]
    ) -> bool:
        """
        Check if workflow needs to be updated.

        Args:
            existing: Existing workflow from database
            new_data: New workflow data from YAML

        Returns:
            True if update is needed
        """
        # Check key fields that indicate changes
        fields_to_check = [
            "description",
            "nodes",
            "edges",
            "max_execution_time",
            "max_parallel_nodes",
            "retry_policy",
            "tags",
        ]

        for field in fields_to_check:
            existing_value = getattr(existing, field)
            new_value = new_data.get(field)

            if existing_value != new_value:
                logger.debug(f"Field {field} changed for workflow {existing.name}")
                return True

        return False

    async def create_workflow(
        self, session: AsyncSession, workflow_data: dict[str, Any]
    ) -> Workflow:
        """
        Create a new workflow directly in database.

        Args:
            session: Database session
            workflow_data: Workflow definition data

        Returns:
            Created workflow
        """
        try:
            workflow = Workflow(**workflow_data)
            session.add(workflow)
            await session.commit()
            await session.refresh(workflow)

            logger.info(
                f"Created workflow {workflow.name} v{workflow.version}",
                extra={"workflow_id": workflow.id},
            )

            return workflow

        except Exception as e:
            await session.rollback()
            error_msg = f"Failed to create workflow: {e}"
            logger.error(error_msg)
            raise WorkflowRegistryError(error_msg)

    async def get_workflow(
        self, session: AsyncSession, workflow_id: uuid.UUID
    ) -> Workflow | None:
        """
        Get workflow by ID.

        Args:
            session: Database session
            workflow_id: Workflow UUID

        Returns:
            Workflow or None if not found
        """
        stmt = select(Workflow).where(Workflow.id == workflow_id)
        result = await session.exec(stmt)
        return result.first()

    async def get_workflow_by_name_version(
        self, session: AsyncSession, name: str, version: str | None = None
    ) -> Workflow | None:
        """
        Get workflow by name and version.

        Args:
            session: Database session
            name: Workflow name
            version: Workflow version (latest if None)

        Returns:
            Workflow or None if not found
        """
        stmt = select(Workflow).where(Workflow.name == name)

        if version:
            stmt = stmt.where(Workflow.version == version)
        else:
            # Get latest version
            stmt = stmt.order_by(Workflow.version.desc())

        result = await session.exec(stmt)
        return result.first()

    async def list_workflows(
        self, session: AsyncSession, filters: dict[str, Any] | None = None
    ) -> list[Workflow]:
        """
        List workflows with optional filtering.

        Args:
            session: Database session
            filters: Optional filters (status, tags, etc.)

        Returns:
            List of workflows
        """
        stmt = select(Workflow)

        if filters:
            if "status" in filters:
                stmt = stmt.where(Workflow.status == filters["status"])

            if "tag" in filters:
                # Filter by tag (PostgreSQL JSON contains operator)
                stmt = stmt.where(Workflow.tags.contains([filters["tag"]]))

        # Order by name and version
        stmt = stmt.order_by(Workflow.name, Workflow.version.desc())

        result = await session.exec(stmt)
        return result.all()

    async def update_workflow(
        self, session: AsyncSession, workflow_id: uuid.UUID, updates: dict[str, Any]
    ) -> Workflow | None:
        """
        Update workflow.

        Args:
            session: Database session
            workflow_id: Workflow UUID
            updates: Fields to update

        Returns:
            Updated workflow or None if not found
        """
        workflow = await self.get_workflow(session, workflow_id)

        if not workflow:
            return None

        try:
            # Apply updates
            for field, value in updates.items():
                if hasattr(workflow, field):
                    setattr(workflow, field, value)

            workflow.updated_at = datetime.now(UTC)
            session.add(workflow)
            await session.commit()
            await session.refresh(workflow)

            logger.info(
                f"Updated workflow {workflow.name} v{workflow.version}",
                extra={"workflow_id": workflow.id},
            )

            return workflow

        except Exception as e:
            await session.rollback()
            error_msg = f"Failed to update workflow {workflow_id}: {e}"
            logger.error(error_msg)
            raise WorkflowRegistryError(error_msg)

    async def delete_workflow(
        self, session: AsyncSession, workflow_id: uuid.UUID
    ) -> bool:
        """
        Delete workflow (soft delete - set status to deprecated).

        Args:
            session: Database session
            workflow_id: Workflow UUID

        Returns:
            True if deleted, False if not found
        """
        workflow = await self.get_workflow(session, workflow_id)

        if not workflow:
            return False

        try:
            workflow.status = WorkflowStatus.DEPRECATED
            workflow.updated_at = datetime.now(UTC)
            session.add(workflow)
            await session.commit()

            logger.info(
                f"Deprecated workflow {workflow.name} v{workflow.version}",
                extra={"workflow_id": workflow.id},
            )

            return True

        except Exception as e:
            await session.rollback()
            error_msg = f"Failed to delete workflow {workflow_id}: {e}"
            logger.error(error_msg)
            raise WorkflowRegistryError(error_msg)

    def validate_yaml_workflows(self) -> dict[str, Any]:
        """
        Validate all YAML workflow files.

        Returns:
            Validation report
        """
        workflow_files = self.loader.discover_workflows()

        validation_report = {
            "total_files": len(workflow_files),
            "valid": [],
            "invalid": [],
        }

        for file_path in workflow_files:
            is_valid, error_message = self.loader.validate_workflow_file(file_path)

            if is_valid:
                validation_report["valid"].append(file_path)
            else:
                validation_report["invalid"].append(
                    {"file": file_path, "error": error_message}
                )

        logger.info(
            f"Validated {len(workflow_files)} workflow files",
            extra={
                "valid": len(validation_report["valid"]),
                "invalid": len(validation_report["invalid"]),
            },
        )

        return validation_report
