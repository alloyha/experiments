"""
Tool-related database models using SQLModel.
"""

import uuid
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Column, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlmodel import Field, SQLModel


class ToolType(str, Enum):
    """Tool type enumeration."""

    BUILTIN = "builtin"
    COMPOSIO = "composio"
    CUSTOM = "custom"


class ToolExecutionStatus(str, Enum):
    """Tool execution status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


class ToolBase(SQLModel):
    """Tool base model."""

    name: str = Field(max_length=255, unique=True, index=True)
    tool_type: ToolType
    description: str | None = Field(default=None, sa_column=Column(Text))

    # Tool configuration
    configuration: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    available_actions: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    # Composio-specific fields
    composio_app_name: str | None = Field(default=None, max_length=255)
    composio_entity_id: str | None = Field(default=None, max_length=255)

    # Authentication
    auth_required: bool = False
    auth_type: str | None = Field(default=None, max_length=100)  # oauth, api_key, etc.
    auth_config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Rate limiting
    rate_limit_requests: int | None = None  # requests per period
    rate_limit_period: int | None = None  # period in seconds

    # Statistics
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time: float | None = None

    # Status
    is_active: bool = True
    last_health_check: datetime | None = None
    health_status: str = Field(default="unknown", max_length=50)

    # Metadata
    tool_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    # Audit fields
    created_by: str | None = Field(default=None, max_length=255)


class Tool(ToolBase, table=True):
    """Tool definition table."""

    __tablename__ = "tools"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = Field(default=None)


class ToolExecutionBase(SQLModel):
    """Tool execution base model."""

    tool_id: uuid.UUID = Field(sa_column=Column(UUID(as_uuid=True)))
    action: str = Field(max_length=255)

    # Request data
    input_parameters: dict[str, Any] = Field(sa_column=Column(JSON))
    timeout: int = 60

    # Execution tracking
    status: ToolExecutionStatus = ToolExecutionStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    execution_time: float | None = None  # seconds
    attempt: int = 1

    # Results
    output_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    error_message: str | None = Field(default=None, sa_column=Column(Text))
    error_code: str | None = Field(default=None, max_length=100)
    error_traceback: str | None = Field(default=None, sa_column=Column(Text))

    # HTTP-specific fields (for HTTP tools)
    http_method: str | None = Field(default=None, max_length=10)
    http_url: str | None = Field(default=None, sa_column=Column(Text))
    http_status_code: int | None = None
    http_response_headers: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON)
    )

    # Rate limiting
    rate_limited: bool = False
    rate_limit_reset_at: datetime | None = None

    # Cost (if applicable)
    cost: float = 0.0

    # Context
    correlation_id: str | None = Field(default=None, max_length=255)
    agent_execution_id: uuid.UUID | None = Field(
        default=None, sa_column=Column(UUID(as_uuid=True))
    )
    workflow_execution_id: uuid.UUID | None = Field(
        default=None, sa_column=Column(UUID(as_uuid=True))
    )


class ToolExecution(ToolExecutionBase, table=True):
    """Tool execution tracking table."""

    __tablename__ = "tool_executions"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolActionBase(SQLModel):
    """Tool action base model."""

    tool_id: uuid.UUID = Field(sa_column=Column(UUID(as_uuid=True)))
    name: str = Field(max_length=255)
    description: str | None = Field(default=None, sa_column=Column(Text))

    # Action schema
    input_schema: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )  # JSON Schema for input validation
    output_schema: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON)
    )  # JSON Schema for output validation

    # Configuration
    timeout: int = 60
    retry_count: int = 3
    retry_delay: int = 1  # seconds

    # Metadata
    action_metadata: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )
    examples: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(JSON)
    )  # Usage examples

    # Statistics
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time: float | None = None

    # Status
    is_active: bool = True


class ToolAction(ToolActionBase, table=True):
    """Tool action definition table."""

    __tablename__ = "tool_actions"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = Field(default=None)


# Response models for API
class ToolResponse(ToolBase):
    """Tool response model."""

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None


class ToolExecutionResponse(ToolExecutionBase):
    """Tool execution response model."""

    id: uuid.UUID
    created_at: datetime


class ToolActionResponse(ToolActionBase):
    """Tool action response model."""

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None
