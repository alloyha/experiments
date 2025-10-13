"""
Main FastAPI application entry point.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.database import init_db
from app.core.logging import setup_logging
from app.core.metrics import setup_metrics

# Setup logging
setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting HIL Agent System...")

    # Initialize database
    await init_db()

    # Setup metrics
    setup_metrics(app)

    logger.info("HIL Agent System started successfully")
    yield

    logger.info("Shutting down HIL Agent System...")


# Create FastAPI app
app = FastAPI(
    title="HIL Agent System",
    description="Production-Ready AI Workflow Orchestration with Code Agents",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Get settings
settings = get_settings()

# Add security middleware
if settings.ALLOWED_HOSTS:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS,
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "hil-agent-system",
        "version": "0.1.0",
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "HIL Agent System - Production-Ready AI Workflow Orchestration",
        "version": "0.1.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_config=None,  # Use our custom logging
    )
