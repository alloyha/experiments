"""
Workflow Integration Service for HIL Agent System.

Provides complete integration between YAML workflows, database persistence,
and agent execution with proper versioning.
"""

import asyncio
from datetime import UTC, datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.agents.types.simple_agent import SimpleAgent
from app.core.database import get_session
from app.core.llm_router import LLMRouter, ModelProfile
from app.core.logging import get_logger
from app.models.workflow import Workflow
from app.services.workflow_loader import YAMLWorkflowLoader
from app.services.workflow_registry import WorkflowRegistryService

logger = get_logger(__name__)


class WorkflowIntegrationService:
    """
    Complete workflow integration service.

    Handles:
    - YAML workflow loading and persistence
    - Workflow versioning and updates
    - Agent integration with workflows
    - Execution context management
    """

    def __init__(self, workflows_dir: str | None = None):
        """Initialize workflow integration service."""
        self.loader = YAMLWorkflowLoader(workflows_dir=workflows_dir)
        self.registry = WorkflowRegistryService(workflows_dir=workflows_dir)

        logger.info(
            "Workflow Integration Service initialized",
            extra={
                "workflows_dir": str(self.loader.workflows_dir),
                "loader_ready": self.loader.schema is not None,
            },
        )

    async def initialize_workflows(self) -> dict[str, Any]:
        """
        Initialize workflows from YAML files into database.

        Returns:
            Initialization report with sync statistics
        """
        try:
            async for session in get_session():
                logger.info("Starting workflow initialization...")

                # Sync all YAML workflows to database
                sync_report = await self.registry.sync_workflows_from_yaml(session)

                logger.info(
                    "Workflow initialization completed",
                    extra={
                        "total_workflows": sync_report["total_yaml_workflows"],
                        "created": len(sync_report["created"]),
                        "updated": len(sync_report["updated"]),
                        "errors": len(sync_report["errors"]),
                    },
                )

                return sync_report

        except Exception as e:
            error_msg = f"Failed to initialize workflows: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def get_workflow(
        self, name: str, version: str | None = None
    ) -> Workflow | None:
        """
        Get workflow by name and version.

        Args:
            name: Workflow name
            version: Workflow version (latest if None)

        Returns:
            Workflow model or None if not found
        """
        try:
            async for session in get_session():
                if version:
                    # Get specific version
                    stmt = select(Workflow).where(
                        Workflow.name == name, Workflow.version == version
                    )
                else:
                    # Get latest version
                    stmt = (
                        select(Workflow)
                        .where(Workflow.name == name)
                        .order_by(Workflow.created_at.desc())
                    )

                result = await session.exec(stmt)
                workflow = result.first()

                if workflow:
                    logger.debug(f"Retrieved workflow {name} v{workflow.version}")
                else:
                    logger.warning(f"Workflow {name} v{version or 'latest'} not found")

                return workflow

        except Exception as e:
            logger.error(f"Failed to get workflow {name}: {e}")
            return None

    async def list_workflows(self) -> list[dict[str, Any]]:
        """
        List all available workflows with versions.

        Returns:
            List of workflow summaries
        """
        try:
            async for session in get_session():
                stmt = select(Workflow).order_by(Workflow.name, Workflow.version)
                result = await session.exec(stmt)
                workflows = result.all()

                workflow_summaries = []
                for workflow in workflows:
                    workflow_summaries.append(
                        {
                            "id": workflow.id,
                            "name": workflow.name,
                            "version": workflow.version,
                            "description": workflow.description,
                            "created_at": workflow.created_at,
                            "updated_at": workflow.updated_at,
                            "tags": workflow.workflow_metadata.get("tags", []),
                            "category": workflow.workflow_metadata.get(
                                "category", "general"
                            ),
                        }
                    )

                logger.info(f"Listed {len(workflow_summaries)} workflows")
                return workflow_summaries

        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")
            return []

    async def execute_workflow(
        self,
        workflow_name: str,
        input_data: dict[str, Any],
        version: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute a workflow with the given input data.

        Args:
            workflow_name: Name of workflow to execute
            input_data: Input data for workflow
            version: Specific version (latest if None)

        Returns:
            Workflow execution result
        """
        start_time = datetime.now(UTC)

        try:
            # Get workflow from database
            workflow = await self.get_workflow(workflow_name, version)
            if not workflow:
                raise ValueError(
                    f"Workflow {workflow_name} v{version or 'latest'} not found"
                )

            logger.info(
                f"Executing workflow {workflow_name} v{workflow.version}",
                extra={"input_data": input_data},
            )

            # Execute based on workflow type
            if self._is_simple_agent_workflow(workflow):
                result = await self._execute_simple_agent_workflow(workflow, input_data)
            else:
                # For complex workflows, we'll implement orchestrator later
                raise NotImplementedError(
                    "Complex workflow execution not yet implemented"
                )

            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            # Add execution metadata
            result["execution_metadata"] = {
                "workflow_name": workflow_name,
                "workflow_version": workflow.version,
                "execution_time": execution_time,
                "timestamp": start_time.isoformat(),
            }

            logger.info(
                "Workflow execution completed",
                extra={
                    "workflow": workflow_name,
                    "version": workflow.version,
                    "execution_time": execution_time,
                    "success": True,
                },
            )

            return result

        except Exception as e:
            execution_time = (datetime.now(UTC) - start_time).total_seconds()
            error_msg = f"Workflow execution failed: {e}"
            logger.error(
                error_msg,
                extra={
                    "workflow": workflow_name,
                    "version": version,
                    "execution_time": execution_time,
                    "error": str(e),
                },
            )

            return {
                "success": False,
                "error": str(e),
                "execution_metadata": {
                    "workflow_name": workflow_name,
                    "workflow_version": version,
                    "execution_time": execution_time,
                    "timestamp": start_time.isoformat(),
                },
            }

    def _is_simple_agent_workflow(self, workflow: Workflow) -> bool:
        """Check if workflow is a simple single-agent workflow."""
        nodes = workflow.nodes or []
        return (
            len(nodes) == 1
            and nodes[0].get("type") == "agent"
            and "simple" in nodes[0].get("agent", "").lower()
        )

    async def _execute_simple_agent_workflow(
        self, workflow: Workflow, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute a simple agent workflow.

        Args:
            workflow: Workflow model
            input_data: Input data

        Returns:
            Execution result
        """
        try:
            # Get the agent node
            agent_node = workflow.nodes[0]
            node_config = agent_node.get("config", {})

            # Determine model profile
            model_profile_str = node_config.get("model_profile", "fast")
            
            # Create LLM router for the agent
            llm_router = LLMRouter()

            # Create Simple Agent with correct parameters
            simple_agent = SimpleAgent(
                llm_router=llm_router,
                model_profile=model_profile_str,
                temperature=node_config.get("temperature", 0.3),
            )

            # Execute agent
            prompt = agent_node.get("prompt", "Process the following input: {{ text }}")
            system_prompt = node_config.get("system_prompt")
            agent_result = await simple_agent.run(
                prompt=prompt,
                input_data=input_data,
                system_prompt=system_prompt
            )

            # Format result according to workflow output schema
            formatted_result = {
                "success": True,
                "output": agent_result,  # agent_result is the actual output
                "agent_metadata": {
                    "agent_name": f"{workflow.name}_agent",
                    "model_used": "unknown",  # Would need routing decision from agent
                    "execution_time": getattr(simple_agent, 'last_execution_time', 0),
                    "cost": getattr(simple_agent, 'last_execution_cost', 0),
                    "tokens_used": 0,  # Would need to track tokens from provider
                },
            }

            return formatted_result

        except Exception as e:
            logger.error(f"Simple agent workflow execution failed: {e}")
            raise

    async def reload_workflows(self) -> dict[str, Any]:
        """
        Reload workflows from YAML files.

        Returns:
            Reload report
        """
        logger.info("Reloading workflows from YAML files...")
        return await self.initialize_workflows()

    async def get_workflow_versions(self, workflow_name: str) -> list[str]:
        """
        Get all versions of a workflow.

        Args:
            workflow_name: Name of workflow

        Returns:
            List of versions sorted by creation date
        """
        try:
            async for session in get_session():
                stmt = (
                    select(Workflow.version)
                    .where(Workflow.name == workflow_name)
                    .order_by(Workflow.created_at.desc())
                )
                result = await session.exec(stmt)
                versions = result.all()

                logger.debug(f"Found {len(versions)} versions for {workflow_name}")
                return versions

        except Exception as e:
            logger.error(f"Failed to get workflow versions for {workflow_name}: {e}")
            return []


# Global service instance
workflow_integration = WorkflowIntegrationService()


async def initialize_workflow_system() -> dict[str, Any]:
    """Initialize the complete workflow system."""
    return await workflow_integration.initialize_workflows()


async def execute_workflow(
    workflow_name: str, input_data: dict[str, Any], version: str | None = None
) -> dict[str, Any]:
    """Execute a workflow by name."""
    return await workflow_integration.execute_workflow(
        workflow_name, input_data, version
    )


async def get_available_workflows() -> list[dict[str, Any]]:
    """Get list of all available workflows."""
    return await workflow_integration.list_workflows()
