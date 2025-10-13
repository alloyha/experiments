"""
Workflow API endpoints.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.workflow_integration import (
    execute_workflow,
    get_available_workflows,
    initialize_workflow_system,
    workflow_integration,
)

router = APIRouter()


class WorkflowExecutionRequest(BaseModel):
    """Workflow execution request."""

    workflow_name: str
    input_data: dict[str, Any]
    version: str | None = None


class WorkflowExecutionResponse(BaseModel):
    """Workflow execution response."""

    success: bool
    output: dict[str, Any] | None = None
    error: str | None = None
    execution_metadata: dict[str, Any]
    agent_metadata: dict[str, Any] | None = None


class WorkflowSummary(BaseModel):
    """Workflow summary."""

    id: str
    name: str
    version: str
    description: str
    created_at: str
    updated_at: str
    tags: list[str]
    category: str


@router.post("/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow_endpoint(request: WorkflowExecutionRequest):
    """Execute a workflow."""
    try:
        result = await execute_workflow(
            workflow_name=request.workflow_name,
            input_data=request.input_data,
            version=request.version,
        )

        return WorkflowExecutionResponse(
            success=result.get("success", False),
            output=result.get("output"),
            error=result.get("error"),
            execution_metadata=result["execution_metadata"],
            agent_metadata=result.get("agent_metadata"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=list[WorkflowSummary])
async def list_workflows():
    """List available workflows."""
    try:
        workflows = await get_available_workflows()
        return [
            WorkflowSummary(
                id=w["id"],
                name=w["name"],
                version=w["version"],
                description=w["description"],
                created_at=w["created_at"].isoformat() if w["created_at"] else "",
                updated_at=w["updated_at"].isoformat() if w["updated_at"] else "",
                tags=w["tags"],
                category=w["category"],
            )
            for w in workflows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize")
async def initialize_workflows():
    """Initialize workflows from YAML files."""
    try:
        result = await initialize_workflow_system()
        return {
            "message": "Workflows initialized successfully",
            "total_workflows": result["total_yaml_workflows"],
            "created": len(result["created"]),
            "updated": len(result["updated"]),
            "errors": len(result["errors"]),
            "details": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_name}")
async def get_workflow_details(workflow_name: str, version: str | None = None):
    """Get workflow details."""
    try:
        workflow = await workflow_integration.get_workflow(workflow_name, version)
        if not workflow:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_name} v{version or 'latest'} not found",
            )

        return {
            "id": workflow.id,
            "name": workflow.name,
            "version": workflow.version,
            "description": workflow.description,
            "nodes": workflow.nodes,
            "edges": workflow.edges,
            "metadata": workflow.workflow_metadata,
            "created_at": workflow.created_at,
            "updated_at": workflow.updated_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_name}/versions")
async def get_workflow_versions(workflow_name: str):
    """Get all versions of a workflow."""
    try:
        versions = await workflow_integration.get_workflow_versions(workflow_name)
        return {
            "workflow_name": workflow_name,
            "versions": versions,
            "total_versions": len(versions),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload")
async def reload_workflows():
    """Reload workflows from YAML files."""
    try:
        result = await workflow_integration.reload_workflows()
        return {
            "message": "Workflows reloaded successfully",
            "total_workflows": result["total_yaml_workflows"],
            "created": len(result["created"]),
            "updated": len(result["updated"]),
            "errors": len(result["errors"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
