"""
Test main application functionality.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestMainAppEndpoints:
    """Test main application endpoints and functionality."""

    def test_health_check_endpoint_function(self):
        """Test the health check endpoint function directly."""
        from app.main import health_check

        # Test the endpoint function directly
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(health_check())
            assert result["status"] == "healthy"
            assert result["service"] == "hil-agent-system"
            assert result["version"] == "0.1.0"
        finally:
            loop.close()

    def test_root_endpoint_function(self):
        """Test the root endpoint function directly."""
        from app.main import root

        # Test the endpoint function directly
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(root())
            assert "HIL Agent System" in result["message"]
            assert result["version"] == "0.1.0"
            assert result["docs"] == "/docs"
        finally:
            loop.close()

    @pytest.mark.asyncio
    async def test_lifespan_manager(self):
        """Test the application lifespan manager."""
        from app.main import lifespan

        # Create a mock FastAPI app
        mock_app = MagicMock()

        with (
            patch("app.main.init_db", new_callable=AsyncMock) as mock_init_db,
            patch("app.main.setup_metrics") as mock_setup_metrics,
        ):
            # Test the lifespan context manager
            async with lifespan(mock_app):
                # Verify initialization calls
                mock_init_db.assert_called_once()
                mock_setup_metrics.assert_called_once_with(mock_app)

    def test_main_execution_block_concept(self):
        """Test the main execution block concept (without actual module reload)."""
        # This test verifies the structure of the main block without actually running it
        import ast
        import inspect

        # Read the main.py file source to verify the if __name__ == "__main__" block exists
        import app.main
        from app.main import settings

        source = inspect.getsource(app.main)

        # Parse the AST to find the main block
        tree = ast.parse(source)

        # Look for if __name__ == "__main__" block
        main_block_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if (
                    isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "__name__"
                    and len(node.test.comparators) == 1
                    and isinstance(node.test.comparators[0], ast.Constant)
                    and node.test.comparators[0].value == "__main__"
                ):
                    main_block_found = True
                    break

        assert main_block_found, "Main execution block not found in app.main"

        # Verify that settings is accessible (used in the main block)
        assert hasattr(settings, "DEBUG")


class TestMainAppIntegration:
    """Integration tests for main app with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_app_creation_structure(self):
        """Test that the FastAPI app is properly structured."""
        from app.main import app

        # Test app properties
        assert app.title == "HIL Agent System"
        assert "Production-Ready AI Workflow Orchestration" in app.description
        assert app.version == "0.1.0"

        # Test that routes are registered
        route_paths = [route.path for route in app.routes]
        assert "/health" in route_paths
        assert "/" in route_paths

        # Check for API v1 routes (they will be prefixed with /api/v1)
        api_routes = [
            route.path for route in app.routes if route.path.startswith("/api/v1")
        ]
        assert len(api_routes) > 0, "No API v1 routes found"

    def test_cors_and_middleware_setup(self):
        """Test that CORS and middleware are properly configured."""
        from app.main import app

        # Verify middleware is configured
        middleware_classes = [
            middleware.cls.__name__ for middleware in app.user_middleware
        ]

        # Should have CORS middleware
        assert any("CORS" in name for name in middleware_classes)
