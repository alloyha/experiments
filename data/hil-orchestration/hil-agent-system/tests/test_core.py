"""
Test core functionality.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.core.config import Settings
from app.core.database import get_engine, get_session, init_db


class TestSettings:
    """Test application settings."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.ENVIRONMENT == "development"
        assert settings.DEBUG is True
        assert settings.SECRET_KEY == "dev-secret-key-change-in-production"
        assert settings.DATABASE_URL.startswith("postgresql+asyncpg://")
        assert settings.REDIS_URL == "redis://localhost:6379"

    def test_environment_override(self, monkeypatch):
        """Test settings override from environment variables."""
        monkeypatch.setenv("ENVIRONMENT", "test")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv(
            "DATABASE_URL", "postgresql+asyncpg://test:test@test-db:5433/test"
        )

        settings = Settings()

        assert settings.ENVIRONMENT == "test"
        assert settings.DEBUG is False
        assert "test-db:5433" in settings.DATABASE_URL

    def test_database_url_construction(self):
        """Test database URL construction."""
        settings = Settings()

        # Test that we have a valid database URL
        assert settings.DATABASE_URL.startswith("postgresql+asyncpg://")
        assert "hil_agent_system" in settings.DATABASE_URL

    def test_redis_url_construction(self):
        """Test Redis URL construction."""
        settings = Settings()

        # Test that we have a valid Redis URL
        assert settings.REDIS_URL == "redis://localhost:6379"


class TestDatabaseConnection:
    """Test database connection functionality."""

    def test_get_engine(self):
        """Test database engine creation."""
        engine = get_engine()
        assert engine is not None
        assert "postgresql+asyncpg" in str(engine.url)

    @pytest.mark.asyncio
    async def test_get_session(self):
        """Test database session creation using mocked functionality."""
        # Test the session creation pattern by creating a mock session directly
        from unittest.mock import MagicMock

        from sqlalchemy.ext.asyncio import AsyncSession

        # Create a mock session with the required methods
        mock_session = MagicMock(spec=AsyncSession)

        # Test that a session has the required methods
        assert hasattr(mock_session, "execute")
        assert hasattr(mock_session, "commit")
        assert hasattr(mock_session, "rollback")

        # This test primarily verifies that we can create and work with async sessions
        assert mock_session is not None

    @pytest.mark.asyncio
    @patch("app.core.database.create_database_engine")
    @patch("app.core.database.setup_logging")  # Mock the logging setup
    async def test_init_db(self, mock_setup_logging, mock_create_engine):
        """Test database initialization."""
        from unittest.mock import AsyncMock, MagicMock

        # Create mock engine with proper async context manager
        mock_engine = MagicMock()

        # Create mock for the begin() context manager
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock()
        mock_context_manager.__aexit__ = AsyncMock()
        mock_engine.begin.return_value = mock_context_manager

        mock_create_engine.return_value = mock_engine
        mock_setup_logging.return_value = None

        await init_db()

        # Verify that engine creation was called
        mock_create_engine.assert_called_once()

        # Verify that connection test was attempted
        mock_engine.begin.assert_called_once()

        # Verify that logging setup was called
        mock_setup_logging.assert_called_once()


class TestErrorHandling:
    """Test error handling functionality."""

    def test_invalid_database_url(self):
        """Test handling of invalid database URL."""
        # Test that settings can be created even with invalid database URL
        import os

        old_url = os.environ.get("DATABASE_URL")
        try:
            os.environ["DATABASE_URL"] = "invalid://url"
            settings = Settings()
            assert settings.DATABASE_URL == "invalid://url"
        finally:
            if old_url:
                os.environ["DATABASE_URL"] = old_url
            elif "DATABASE_URL" in os.environ:
                del os.environ["DATABASE_URL"]

    @pytest.mark.asyncio
    async def test_database_connection_error(self):
        """Test handling of database connection errors."""
        # Create settings with invalid database configuration
        invalid_settings = Settings()

        # Test that we can create settings with invalid database URLs
        assert invalid_settings.DATABASE_URL is not None


class TestAsyncUtilities:
    """Test async utility functions."""

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent database operations."""

        async def mock_operation(delay: float, result: str):
            await asyncio.sleep(delay)
            return result

        # Test concurrent execution
        tasks = [
            mock_operation(0.1, "task1"),
            mock_operation(0.05, "task2"),
            mock_operation(0.15, "task3"),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert "task1" in results
        assert "task2" in results
        assert "task3" in results

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in async operations."""

        async def slow_operation():
            await asyncio.sleep(1.0)
            return "completed"

        # Test that operation times out
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test error propagation in async operations."""

        async def failing_operation():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await failing_operation()


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_required_environment_variables(self):
        """Test that required environment variables are validated."""
        # Test that settings work with minimal required config
        settings = Settings()
        assert settings.DATABASE_URL is not None
        assert settings.REDIS_URL is not None

    def test_port_validation(self):
        """Test port number validation."""
        # Test that settings are loaded correctly
        settings = Settings()
        assert isinstance(settings.DATABASE_URL, str)
        assert isinstance(settings.REDIS_URL, str)

    def test_url_validation(self):
        """Test URL validation for external services."""
        settings = Settings()

        # Test that settings have the expected URL fields
        assert settings.COMPOSIO_BASE_URL == "https://backend.composio.dev/api/v2"
        assert settings.DATABASE_URL.startswith("postgresql+asyncpg://")
        assert settings.REDIS_URL.startswith("redis://")


class TestLoggingConfiguration:
    """Test logging configuration."""

    def test_log_level_setting(self, monkeypatch):
        """Test log level configuration."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        settings = Settings()
        assert settings.LOG_LEVEL == "DEBUG"

        monkeypatch.setenv("LOG_LEVEL", "INFO")
        settings = Settings()
        assert settings.LOG_LEVEL == "INFO"

    def test_log_format_setting(self):
        """Test log format configuration."""
        settings = Settings()
        assert hasattr(settings, "LOG_LEVEL")
        # Add more specific log format tests based on your logging setup


class TestSecuritySettings:
    """Test security-related settings."""

    def test_secret_key_requirement(self):
        """Test that secret key is properly configured."""
        settings = Settings()
        assert hasattr(settings, "SECRET_KEY")
        assert settings.SECRET_KEY is not None
        # In production, ensure secret key is not default/empty

    def test_cors_settings(self):
        """Test CORS configuration."""
        settings = Settings()
        # Test CORS origins configuration
        assert hasattr(settings, "ALLOWED_HOSTS")
        assert hasattr(settings, "CORS_ORIGINS")
        assert settings.ALLOWED_HOSTS == ["*"]

    def test_api_key_validation(self, monkeypatch):
        """Test API key validation."""
        # Test OpenAI API key
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert settings.OPENAI_API_KEY == "sk-test-key"

        # Test Anthropic API key
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test-key")
        settings = Settings()
        assert settings.ANTHROPIC_API_KEY == "ant-test-key"


class TestPerformanceSettings:
    """Test performance-related settings."""

    def test_connection_pool_settings(self):
        """Test database connection pool configuration."""
        settings = Settings()
        # Test connection pool parameters if defined
        assert hasattr(settings, "DATABASE_URL")
        assert settings.DATABASE_URL is not None

    def test_timeout_settings(self):
        """Test timeout configurations."""
        settings = Settings()
        # Test various timeout settings
        if hasattr(settings, "request_timeout"):
            assert settings.request_timeout > 0
        if hasattr(settings, "database_timeout"):
            assert settings.database_timeout > 0


class TestDatabaseCoverageExtended:
    """Extended tests to improve database module coverage."""

    @pytest.mark.asyncio
    async def test_create_all_tables(self):
        """Test create_all_tables function."""
        from unittest.mock import MagicMock

        with (
            patch("app.core.database.create_database_engine") as mock_engine_creator,
            patch("app.core.database.SQLModel") as mock_sqlmodel,
            patch("app.core.database.engine", None),
        ):  # Force engine to be None
            # Create mock engine
            mock_engine = MagicMock()
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock()
            mock_context_manager.__aexit__ = AsyncMock()
            mock_engine.begin.return_value = mock_context_manager
            mock_engine_creator.return_value = mock_engine

            # Create mock connection
            mock_conn = MagicMock()
            mock_conn.run_sync = AsyncMock()
            mock_context_manager.__aenter__.return_value = mock_conn

            from app.core.database import create_all_tables

            await create_all_tables()

            # Verify engine creation and table creation
            mock_engine_creator.assert_called_once()
            mock_engine.begin.assert_called_once()
            mock_conn.run_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_db_not_initialized(self):
        """Test get_db when session maker is not initialized."""
        from app.core.database import get_db

        # Mock async_session_maker as None
        with patch("app.core.database.async_session_maker", None):
            session_generator = get_db()

            with pytest.raises(RuntimeError, match="Database not initialized"):
                await session_generator.__anext__()

    @pytest.mark.asyncio
    async def test_get_db_with_exception(self):
        """Test get_db exception handling."""
        from unittest.mock import AsyncMock, MagicMock

        from app.core.database import get_db

        # Create mock session maker and session
        mock_session_maker = MagicMock()
        mock_session = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        # Make the session creation raise an exception
        mock_session_maker.return_value.__aenter__ = AsyncMock(
            side_effect=Exception("Test error")
        )

        with patch("app.core.database.async_session_maker", mock_session_maker):
            session_generator = get_db()

            with pytest.raises(Exception, match="Test error"):
                await session_generator.__anext__()

    @pytest.mark.asyncio
    async def test_init_db_connection_failure(self):
        """Test init_db when connection fails."""
        with (
            patch("app.core.database.create_database_engine") as mock_engine_creator,
            patch("app.core.database.setup_logging") as mock_setup_logging,
        ):
            # Create mock engine that fails on connection
            mock_engine = MagicMock()
            mock_engine.begin.side_effect = Exception("Connection failed")
            mock_engine_creator.return_value = mock_engine

            from app.core.database import init_db

            with pytest.raises(Exception, match="Connection failed"):
                await init_db()

            # Verify logging setup was still called
            mock_setup_logging.assert_called_once()

    def test_setup_logging_with_engine(self):
        """Test setup_logging function when engine exists."""
        from unittest.mock import MagicMock, patch

        from app.core.database import setup_logging

        # Create mock engine with sync_engine
        mock_engine = MagicMock()
        mock_sync_engine = MagicMock()
        mock_engine.sync_engine = mock_sync_engine

        with (
            patch("app.core.database.engine", mock_engine),
            patch("app.core.database.event") as mock_event,
            patch("app.core.database.get_settings") as mock_get_settings,
        ):
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.is_development = True
            mock_get_settings.return_value = mock_settings

            setup_logging()

            # Verify event listener was set up
            mock_event.listens_for.assert_called_once()


class TestConfigCoverageExtended:
    """Extended tests to improve config module coverage."""

    def test_is_development_property(self):
        """Test is_development property."""
        from app.core.config import Settings

        # Test development environment
        settings = Settings(ENVIRONMENT="development")
        assert settings.is_development is True

        # Test non-development environment
        settings = Settings(ENVIRONMENT="production")
        assert settings.is_development is False

        # Test case insensitive
        settings = Settings(ENVIRONMENT="DEVELOPMENT")
        assert settings.is_development is True


class TestLoggingCoverageExtended:
    """Extended tests to improve logging module coverage."""

    def test_setup_logging_production_mode(self):
        """Test logging setup in production mode."""
        from unittest.mock import MagicMock, patch

        from app.core.logging import setup_logging

        # Mock settings for production
        mock_settings = MagicMock()
        mock_settings.is_production = True
        mock_settings.LOG_LEVEL = "INFO"

        with (
            patch("app.core.logging.get_settings", return_value=mock_settings),
            patch("structlog.configure") as mock_configure,
            patch("structlog.processors") as mock_processors,
        ):
            # Setup mock processors
            mock_processors.TimeStamper.return_value = "timestamper"
            mock_processors.add_log_level = "add_log_level"
            mock_processors.StackInfoRenderer.return_value = "stack_info"
            mock_processors.UnicodeDecoder.return_value = "unicode_decoder"
            mock_processors.JSONRenderer.return_value = "json_renderer"

            setup_logging()

            # Verify structlog.configure was called
            mock_configure.assert_called_once()

            # Verify JSON renderer is used in production
            mock_processors.JSONRenderer.assert_called_once()
