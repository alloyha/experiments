"""
Database configuration and management.
"""

from collections.abc import AsyncGenerator

import structlog
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel

from app.core.config import get_settings

logger = structlog.get_logger(__name__)

# SQLModel already provides a base class
Base = SQLModel

# Global variables
engine = None
async_session_maker = None


def create_database_engine():
    """Create database engine."""
    settings = get_settings()

    # Different configuration for SQLite vs PostgreSQL
    if "sqlite" in settings.DATABASE_URL.lower():
        return create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DATABASE_ECHO,
            pool_pre_ping=True,
        )
    return create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DATABASE_ECHO,
        pool_size=20,
        max_overflow=0,
        pool_pre_ping=True,
        pool_recycle=3600,
    )


def get_engine():
    """Get database engine for testing."""
    return create_database_engine()


async def create_all_tables() -> None:
    """Create all database tables."""
    global engine
    if engine is None:
        engine = create_database_engine()

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    logger.info("Database tables created successfully")


async def init_db() -> None:
    """Initialize database."""
    global engine, async_session_maker

    logger.info("Initializing database connection")

    engine = create_database_engine()
    setup_logging()  # Setup logging after engine is created

    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Test connection
    try:
        async with engine.begin() as conn:
            await conn.run_sync(lambda _: None)
        logger.info("Database connection established successfully")
    except Exception as e:
        logger.error("Failed to connect to database", error=str(e))
        raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    if async_session_maker is None:
        raise RuntimeError("Database not initialized")

    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Alias for dependency injection
get_session = get_db


# SQLAlchemy event listeners for logging
def setup_logging():
    """Setup database logging after engine is created."""
    global engine
    if engine is not None:
        # For async engines, we need to listen on the sync_engine
        @event.listens_for(engine.sync_engine, "before_cursor_execute", once=True)
        def log_sql_queries(conn, cursor, statement, parameters, context, executemany):
            """Log SQL queries in development."""
            settings = get_settings()
            if settings.is_development:
                logger.debug(
                    "Executing SQL query",
                    statement=statement,
                    parameters=parameters,
                )
