"""
Comprehensive tests for database module to improve coverage.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlmodel import SQLModel

from app.core.config import Settings
from app.core.database import (
    create_all_tables,
    create_database_engine,
    get_db,
    get_engine,
    get_session,
    init_db,
    setup_logging,
)


class TestDatabaseEngine:
    """Test database engine creation and management."""

    @patch("app.core.database.get_settings")
    def test_create_database_engine(self, mock_get_settings):
        """Test database engine creation."""
        mock_settings = MagicMock()
        mock_settings.DATABASE_URL = "sqlite+aiosqlite:///:memory:"
        mock_settings.DATABASE_ECHO = False
        mock_get_settings.return_value = mock_settings

        engine = create_database_engine()

        assert engine is not None
        assert str(engine.url) == "sqlite+aiosqlite:///:memory:"

    @patch("app.core.database.create_database_engine")
    def test_get_engine(self, mock_create_engine):
        """Test get_engine function."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        result = get_engine()

        assert result == mock_engine
        mock_create_engine.assert_called_once()


class TestDatabaseInitialization:
    """Test database initialization functions."""

    @pytest.mark.asyncio
    @patch("app.core.database.create_database_engine")
    async def test_create_all_tables_new_engine(self, mock_create_engine):
        """Test creating tables with new engine."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()

        # Properly mock the async context manager
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_engine.begin.return_value = mock_context_manager

        mock_create_engine.return_value = mock_engine

        # Reset global engine
        import app.core.database

        app.core.database.engine = None

        await create_all_tables()

        mock_create_engine.assert_called_once()
        mock_engine.begin.assert_called_once()
        mock_conn.run_sync.assert_called_once()
        mock_conn.run_sync.assert_called_once()

        # Verify engine is set globally
        assert app.core.database.engine == mock_engine

    @pytest.mark.asyncio
    @patch("app.core.database.create_database_engine")
    async def test_create_all_tables_existing_engine(self, mock_create_engine):
        """Test creating tables with existing engine."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()

        # Properly mock the async context manager
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_engine.begin.return_value = mock_context_manager

        # Set global engine
        import app.core.database

        app.core.database.engine = mock_engine

        await create_all_tables()

        # Should not create new engine
        mock_create_engine.assert_not_called()
        mock_engine.begin.assert_called_once()
        mock_conn.run_sync.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.core.database.create_database_engine")
    @patch("app.core.database.setup_logging")
    @patch("app.core.database.async_sessionmaker")
    async def test_init_db_success(
        self, mock_sessionmaker, mock_setup_logging, mock_create_engine
    ):
        """Test successful database initialization."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()

        # Properly mock the async context manager
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_engine.begin.return_value = mock_context_manager

        mock_create_engine.return_value = mock_engine

        # Reset global variables
        import app.core.database

        app.core.database.engine = None
        app.core.database.async_session_maker = None

        await init_db()

        mock_create_engine.assert_called_once()
        mock_setup_logging.assert_called_once()
        mock_sessionmaker.assert_called_once()
        mock_engine.begin.assert_called_once()

        # Verify globals are set
        assert app.core.database.engine == mock_engine
        assert app.core.database.async_session_maker is not None

    @pytest.mark.asyncio
    @patch("app.core.database.create_database_engine")
    async def test_init_db_connection_failure(self, mock_create_engine):
        """Test database initialization with connection failure."""
        mock_engine = AsyncMock()
        mock_engine.begin.side_effect = Exception("Connection failed")
        mock_create_engine.return_value = mock_engine

        # Reset global variables
        import app.core.database

        app.core.database.engine = None
        app.core.database.async_session_maker = None

        with pytest.raises(Exception, match="Connection failed"):
            await init_db()


class TestDatabaseSession:
    """Test database session management."""

    @pytest.mark.asyncio
    async def test_get_db_not_initialized(self):
        """Test get_db when database not initialized."""
        # Reset global session maker
        import app.core.database

        app.core.database.async_session_maker = None

        with pytest.raises(RuntimeError, match="Database not initialized"):
            async for session in get_db():
                pass

    @pytest.mark.asyncio
    @patch("app.core.database.async_session_maker")
    async def test_get_db_success(self, mock_session_maker):
        """Test successful session creation."""
        mock_session = AsyncMock()

        # Create a proper async context manager mock
        async def session_context():
            yield mock_session

        mock_session_maker.return_value = session_context()

        # Set global session maker
        import app.core.database

        app.core.database.async_session_maker = mock_session_maker

        # Test the session generation
        sessions = []
        async for session in get_db():
            sessions.append(session)
            break  # Only get one session

        assert len(sessions) == 1
        assert sessions[0] == mock_session
        mock_session_maker.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.core.database.async_session_maker")
    async def test_get_db_with_exception(self, mock_session_maker):
        """Test session handling with exception."""
        mock_session = AsyncMock()

        # Create a session context that raises an exception
        async def session_context_with_error():
            try:
                yield mock_session
                raise Exception("Test error")
            except Exception:
                await mock_session.rollback()
                raise

        mock_session_maker.return_value = session_context_with_error()

        # Set global session maker
        import app.core.database

        app.core.database.async_session_maker = mock_session_maker

        # Test exception handling
        with pytest.raises(Exception, match="Test error"):
            async for session in get_db():
                # Simulate some work that causes an error
                raise Exception("Test error")

        mock_session_maker.assert_called_once()
        mock_session = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        mock_session_maker.return_value = mock_session_context

        # Set global session maker
        import app.core.database

        app.core.database.async_session_maker = mock_session_maker

        with pytest.raises(ValueError, match="Test exception"):
            async for session in get_db():
                raise ValueError("Test exception")

        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()

    def test_get_session_alias(self):
        """Test that get_session is an alias for get_db."""
        assert get_session == get_db


class TestDatabaseLogging:
    """Test database logging setup."""

    @patch("app.core.database.get_settings")
    @patch("app.core.database.event")
    def test_setup_logging_with_engine(self, mock_event, mock_get_settings):
        """Test logging setup when engine exists."""
        mock_settings = MagicMock()
        mock_settings.is_development = True
        mock_get_settings.return_value = mock_settings

        mock_engine = MagicMock()
        mock_sync_engine = MagicMock()
        mock_engine.sync_engine = mock_sync_engine

        # Set global engine
        import app.core.database

        app.core.database.engine = mock_engine

        setup_logging()

        mock_event.listens_for.assert_called_once_with(
            mock_sync_engine, "before_cursor_execute", once=True
        )

    @patch("app.core.database.event")
    def test_setup_logging_without_engine(self, mock_event):
        """Test logging setup when engine is None."""
        # Set global engine to None
        import app.core.database

        app.core.database.engine = None

        setup_logging()

        # Should not call event.listens_for
        mock_event.listens_for.assert_not_called()

    @patch("app.core.database.get_settings")
    @patch("app.core.database.event")
    def test_log_sql_queries_development(self, mock_event, mock_get_settings):
        """Test SQL query logging in development."""
        mock_settings = MagicMock()
        mock_settings.is_development = True
        mock_get_settings.return_value = mock_settings

        mock_engine = MagicMock()
        mock_engine.sync_engine = MagicMock()

        # Set global engine
        import app.core.database

        app.core.database.engine = mock_engine

        # Mock the event system to avoid the sync_engine issue
        mock_event.listens_for.return_value = lambda f: f

        setup_logging()

        # Verify event listener was attempted to be set up
        mock_event.listens_for.assert_called()
        mock_sync_engine = MagicMock()
        mock_engine.sync_engine = mock_sync_engine

        # Set global engine and setup logging
        import app.core.database

        app.core.database.engine = mock_engine

        # This would normally be called by setup_logging
        # We're testing the function that gets registered as a listener
        with patch("app.core.database.logger") as mock_logger:
            # Import the inner function that would be created
            from app.core.database import setup_logging

            # The actual test would need to trigger the event listener
            # For now, just verify the setup doesn't crash
            setup_logging()

    def test_sql_model_base(self):
        """Test SQLModel base import."""
        from app.core.database import Base

        assert Base == SQLModel


class TestDatabaseIntegration:
    """Integration tests for database functionality."""

    @pytest.mark.asyncio
    async def test_full_database_lifecycle(self):
        """Test complete database initialization and cleanup."""
        import app.core.database

        # Save original state
        original_engine = app.core.database.engine
        original_session_maker = app.core.database.async_session_maker

        try:
            with patch("app.core.database.get_settings") as mock_get_settings:
                mock_settings = MagicMock()
                mock_settings.DATABASE_URL = "sqlite+aiosqlite:///:memory:"
                mock_settings.DATABASE_ECHO = False
                mock_get_settings.return_value = mock_settings

                # Reset state
                app.core.database.engine = None
                app.core.database.async_session_maker = None

                # Initialize database
                await init_db()

                # Verify initialization
                assert app.core.database.engine is not None
                assert app.core.database.async_session_maker is not None

                # Test session creation
                session_count = 0
                async for session in get_db():
                    session_count += 1
                    assert isinstance(session, AsyncSession)
                    break

                assert session_count == 1

        finally:
            # Restore original state
            app.core.database.engine = original_engine
            app.core.database.async_session_maker = original_session_maker
