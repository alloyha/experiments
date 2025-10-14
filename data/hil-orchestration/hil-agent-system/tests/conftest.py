"""
Test configuration and fixtures.
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.core.config import get_settings
from app.core.database import get_session
from app.main import app


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for the test session."""
    return asyncio.DefaultEventLoopPolicy()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create test database engine."""

    # Use in-memory SQLite for fast tests
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine):
    """Create test database session with transaction rollback."""
    
    # Start a connection
    connection = await test_engine.connect()
    
    # Begin a transaction
    transaction = await connection.begin()
    
    # Create session bound to connection
    async_session_maker = async_sessionmaker(
        bind=connection,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session_maker() as session:
        yield session
        
        # Rollback everything after test
        await transaction.rollback()
    
    await connection.close()


@pytest_asyncio.fixture
async def test_client(db_session):
    """Create test HTTP client with test database."""
    from fastapi.testclient import TestClient

    async def get_test_session():
        yield db_session

    # Override the database dependency
    app.dependency_overrides[get_session] = get_test_session

    # For async tests, we need to use httpx.AsyncClient with ASGITransport
    from httpx import ASGITransport

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    # Clean up override
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def mock_settings():
    """Mock settings for testing."""
    from app.core.config import Settings

    return Settings(
        ENVIRONMENT="test",
        DEBUG=True,
        DATABASE_URL="sqlite+aiosqlite:///:memory:",
        REDIS_URL="redis://localhost:6379/15",  # Test database
        OPENAI_API_KEY="test-key",
        ANTHROPIC_API_KEY="test-key",
        COMPOSIO_API_KEY="test-key",
    )
