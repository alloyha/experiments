"""Tests for app/api/v1/endpoints/workflows.py - Workflow API endpoints."""

import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from app.main import app


class TestWorkflowEndpoints:
    """Test suite for workflow API endpoints."""

    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, test_client: AsyncClient):
        """Test successful workflow execution."""
        mock_result = {
            "success": True,
            "output": {"result": "classified"},
            "error": None,
            "execution_metadata": {
                "execution_id": "test-123",
                "duration": 1.5,
                "timestamp": "2024-01-01T00:00:00"
            },
            "agent_metadata": {
                "agent_type": "simple",
                "model_used": "gpt-3.5-turbo"
            }
        }
        
        with patch("app.api.v1.endpoints.workflows.execute_workflow", new_callable=AsyncMock, return_value=mock_result):
            workflow_data = {
                "workflow_name": "test_workflow",
                "input_data": {"text": "test input"},
                "version": "v1"
            }
            
            response = await test_client.post(
                "/api/v1/workflows/execute",
                json=workflow_data
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result["success"] is True
            assert result["output"] == {"result": "classified"}
            assert result["error"] is None
            assert "execution_metadata" in result

    @pytest.mark.asyncio
    async def test_execute_workflow_without_version(self, test_client: AsyncClient):
        """Test workflow execution without version (defaults to None)."""
        mock_result = {
            "success": True,
            "output": {},
            "error": None,
            "execution_metadata": {"execution_id": "test-456"}
        }
        
        with patch("app.api.v1.endpoints.workflows.execute_workflow", new_callable=AsyncMock, return_value=mock_result):
            workflow_data = {
                "workflow_name": "test_workflow",
                "input_data": {"text": "test"}
            }
            
            response = await test_client.post(
                "/api/v1/workflows/execute",
                json=workflow_data
            )
            
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_execute_workflow_error(self, test_client: AsyncClient):
        """Test workflow execution with error."""
        mock_result = {
            "success": False,
            "output": None,
            "error": "Workflow failed",
            "execution_metadata": {"execution_id": "test-789"}
        }
        
        with patch("app.api.v1.endpoints.workflows.execute_workflow", new_callable=AsyncMock, return_value=mock_result):
            workflow_data = {
                "workflow_name": "failing_workflow",
                "input_data": {"text": "test"}
            }
            
            response = await test_client.post(
                "/api/v1/workflows/execute",
                json=workflow_data
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result["success"] is False
            assert result["error"] == "Workflow failed"

    @pytest.mark.asyncio
    async def test_execute_workflow_exception(self, test_client: AsyncClient):
        """Test workflow execution raises exception."""
        with patch("app.api.v1.endpoints.workflows.execute_workflow", new_callable=AsyncMock, side_effect=Exception("Database error")):
            workflow_data = {
                "workflow_name": "test_workflow",
                "input_data": {"text": "test"}
            }
            
            response = await test_client.post(
                "/api/v1/workflows/execute",
                json=workflow_data
            )
            
            assert response.status_code == 500
            assert "Database error" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_list_workflows_success(self, test_client: AsyncClient):
        """Test listing workflows successfully."""
        mock_workflows = [
            {
                "id": "wf-1",
                "name": "workflow_1",
                "version": "v1",
                "description": "Test workflow 1",
                "created_at": datetime(2024, 1, 1),
                "updated_at": datetime(2024, 1, 2),
                "tags": ["tag1", "tag2"],
                "category": "classification"
            },
            {
                "id": "wf-2",
                "name": "workflow_2",
                "version": "v2",
                "description": "Test workflow 2",
                "created_at": datetime(2024, 1, 3),
                "updated_at": datetime(2024, 1, 4),
                "tags": ["tag3"],
                "category": "generation"
            }
        ]
        
        with patch("app.api.v1.endpoints.workflows.get_available_workflows", new_callable=AsyncMock, return_value=mock_workflows):
            response = await test_client.get("/api/v1/workflows/")
            
            assert response.status_code == 200
            result = response.json()
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["id"] == "wf-1"
            assert result[0]["name"] == "workflow_1"
            assert result[0]["tags"] == ["tag1", "tag2"]

    @pytest.mark.asyncio
    async def test_list_workflows_with_none_dates(self, test_client: AsyncClient):
        """Test listing workflows with None created/updated dates."""
        mock_workflows = [
            {
                "id": "wf-1",
                "name": "workflow_1",
                "version": "v1",
                "description": "Test",
                "created_at": None,
                "updated_at": None,
                "tags": [],
                "category": "test"
            }
        ]
        
        with patch("app.api.v1.endpoints.workflows.get_available_workflows", new_callable=AsyncMock, return_value=mock_workflows):
            response = await test_client.get("/api/v1/workflows/")
            
            assert response.status_code == 200
            result = response.json()
            assert result[0]["created_at"] == ""
            assert result[0]["updated_at"] == ""

    @pytest.mark.asyncio
    async def test_list_workflows_exception(self, test_client: AsyncClient):
        """Test listing workflows raises exception."""
        with patch("app.api.v1.endpoints.workflows.get_available_workflows", new_callable=AsyncMock, side_effect=Exception("DB error")):
            response = await test_client.get("/api/v1/workflows/")
            
            assert response.status_code == 500
            assert "DB error" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_initialize_workflows_success(self, test_client: AsyncClient):
        """Test workflow initialization."""
        mock_result = {
            "total_yaml_workflows": 5,
            "created": ["wf1", "wf2"],
            "updated": ["wf3"],
            "errors": []
        }
        
        with patch("app.api.v1.endpoints.workflows.initialize_workflow_system", new_callable=AsyncMock, return_value=mock_result):
            response = await test_client.post("/api/v1/workflows/initialize")
            
            assert response.status_code == 200
            result = response.json()
            assert result["message"] == "Workflows initialized successfully"
            assert result["total_workflows"] == 5
            assert result["created"] == 2
            assert result["updated"] == 1
            assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_initialize_workflows_with_errors(self, test_client: AsyncClient):
        """Test workflow initialization with errors."""
        mock_result = {
            "total_yaml_workflows": 3,
            "created": ["wf1"],
            "updated": [],
            "errors": ["error1", "error2"]
        }
        
        with patch("app.api.v1.endpoints.workflows.initialize_workflow_system", new_callable=AsyncMock, return_value=mock_result):
            response = await test_client.post("/api/v1/workflows/initialize")
            
            assert response.status_code == 200
            result = response.json()
            assert result["errors"] == 2
            assert "details" in result

    @pytest.mark.asyncio
    async def test_initialize_workflows_exception(self, test_client: AsyncClient):
        """Test workflow initialization raises exception."""
        with patch("app.api.v1.endpoints.workflows.initialize_workflow_system", new_callable=AsyncMock, side_effect=Exception("Init error")):
            response = await test_client.post("/api/v1/workflows/initialize")
            
            assert response.status_code == 500
            assert "Init error" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_workflow_details_success(self, test_client: AsyncClient):
        """Test getting workflow details."""
        mock_workflow = MagicMock()
        mock_workflow.id = "wf-123"
        mock_workflow.name = "test_workflow"
        mock_workflow.version = "v1"
        mock_workflow.description = "Test workflow"
        mock_workflow.nodes = [{"id": "node1"}]
        mock_workflow.edges = [{"from": "node1", "to": "node2"}]
        mock_workflow.workflow_metadata = {"key": "value"}
        mock_workflow.created_at = datetime(2024, 1, 1)
        mock_workflow.updated_at = datetime(2024, 1, 2)
        
        with patch("app.api.v1.endpoints.workflows.workflow_integration") as mock_integration:
            mock_integration.get_workflow = AsyncMock(return_value=mock_workflow)
            
            response = await test_client.get("/api/v1/workflows/test_workflow?version=v1")
            
            assert response.status_code == 200
            result = response.json()
            assert result["id"] == "wf-123"
            assert result["name"] == "test_workflow"
            assert result["version"] == "v1"
            assert "nodes" in result
            assert "edges" in result

    @pytest.mark.asyncio
    async def test_get_workflow_details_not_found(self, test_client: AsyncClient):
        """Test getting workflow details when not found."""
        with patch("app.api.v1.endpoints.workflows.workflow_integration") as mock_integration:
            mock_integration.get_workflow = AsyncMock(return_value=None)
            
            response = await test_client.get("/api/v1/workflows/nonexistent")
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_workflow_details_exception(self, test_client: AsyncClient):
        """Test getting workflow details raises exception."""
        with patch("app.api.v1.endpoints.workflows.workflow_integration") as mock_integration:
            mock_integration.get_workflow = AsyncMock(side_effect=Exception("Query error"))
            
            response = await test_client.get("/api/v1/workflows/test_workflow")
            
            assert response.status_code == 500
            assert "Query error" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_workflow_versions_success(self, test_client: AsyncClient):
        """Test getting workflow versions."""
        mock_versions = ["v1", "v2", "v3"]
        
        with patch("app.api.v1.endpoints.workflows.workflow_integration") as mock_integration:
            mock_integration.get_workflow_versions = AsyncMock(return_value=mock_versions)
            
            response = await test_client.get("/api/v1/workflows/test_workflow/versions")
            
            assert response.status_code == 200
            result = response.json()
            assert result["workflow_name"] == "test_workflow"
            assert result["versions"] == ["v1", "v2", "v3"]
            assert result["total_versions"] == 3

    @pytest.mark.asyncio
    async def test_get_workflow_versions_exception(self, test_client: AsyncClient):
        """Test getting workflow versions raises exception."""
        with patch("app.api.v1.endpoints.workflows.workflow_integration") as mock_integration:
            mock_integration.get_workflow_versions = AsyncMock(side_effect=Exception("Version error"))
            
            response = await test_client.get("/api/v1/workflows/test_workflow/versions")
            
            assert response.status_code == 500
            assert "Version error" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_reload_workflows_success(self, test_client: AsyncClient):
        """Test reloading workflows."""
        mock_result = {
            "total_yaml_workflows": 10,
            "created": ["wf1", "wf2"],
            "updated": ["wf3", "wf4", "wf5"],
            "errors": []
        }
        
        with patch("app.api.v1.endpoints.workflows.workflow_integration") as mock_integration:
            mock_integration.reload_workflows = AsyncMock(return_value=mock_result)
            
            response = await test_client.post("/api/v1/workflows/reload")
            
            assert response.status_code == 200
            result = response.json()
            assert result["message"] == "Workflows reloaded successfully"
            assert result["total_workflows"] == 10
            assert result["created"] == 2
            assert result["updated"] == 3
            assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_reload_workflows_exception(self, test_client: AsyncClient):
        """Test reloading workflows raises exception."""
        with patch("app.api.v1.endpoints.workflows.workflow_integration") as mock_integration:
            mock_integration.reload_workflows = AsyncMock(side_effect=Exception("Reload error"))
            
            response = await test_client.post("/api/v1/workflows/reload")
            
            assert response.status_code == 500
            assert "Reload error" in response.json()["detail"]
