"""
Application configuration management.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Environment
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=True)

    # Security
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production")
    ALLOWED_HOSTS: list[str] = Field(default=["*"])
    CORS_ORIGINS: list[str] = Field(default=["*"])

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/hil_agent_system"
    )
    DATABASE_ECHO: bool = Field(default=False)

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379")

    # Celery
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2")

    # LLM Providers
    OPENAI_API_KEY: str | None = Field(default=None)
    ANTHROPIC_API_KEY: str | None = Field(default=None)

    # Composio
    COMPOSIO_API_KEY: str | None = Field(default=None)
    COMPOSIO_BASE_URL: str = Field(default="https://backend.composio.dev/api/v2")

    # Agent Settings
    DEFAULT_AGENT_TIMEOUT: int = Field(default=300)  # 5 minutes
    MAX_TOOL_RETRIES: int = Field(default=3)
    MAX_REASONING_ITERATIONS: int = Field(default=10)

    # Memory Settings
    MAX_SHORT_TERM_MESSAGES: int = Field(default=50)
    VECTOR_DIMENSION: int = Field(default=1536)  # OpenAI ada-002 dimension

    # Workflow Settings
    MAX_WORKFLOW_EXECUTION_TIME: int = Field(default=1800)  # 30 minutes
    MAX_PARALLEL_NODES: int = Field(default=10)

    # Security Settings
    CODE_EXECUTION_TIMEOUT: int = Field(default=60)
    SANDBOX_MEMORY_LIMIT: str = Field(default="512m")
    SANDBOX_CPU_LIMIT: str = Field(default="0.5")

    # Observability
    METRICS_ENABLED: bool = Field(default=True)
    TRACING_ENABLED: bool = Field(default=True)
    LOG_LEVEL: str = Field(default="INFO")

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.ENVIRONMENT.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
