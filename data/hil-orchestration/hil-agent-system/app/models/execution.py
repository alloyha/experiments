"""
Execution tracking and audit models using SQLModel.
"""

import uuid
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Column, Float, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlmodel import Field, SQLModel


class AuditEventType(str, Enum):
    """Audit event type enumeration."""

    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    DATA_CHANGE = "data_change"
    ERROR_EVENT = "error_event"


class SystemMetricBase(SQLModel):
    """System metric base model."""

    metric_name: str = Field(max_length=255, index=True)
    metric_value: float
    metric_unit: str | None = Field(
        default=None, max_length=50
    )  # seconds, bytes, count, etc.

    # Dimensions for grouping
    dimensions: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Temporal
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SystemMetric(SystemMetricBase, table=True):
    """System-wide metrics and performance tracking."""

    __tablename__ = "system_metrics"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )


class AuditLogBase(SQLModel):
    """Audit log base model."""

    event_type: AuditEventType
    user_id: str | None = Field(default=None, max_length=255)
    session_id: str | None = Field(default=None, max_length=255)

    # Event details
    action: str = Field(max_length=255)
    resource_type: str | None = Field(default=None, max_length=100)
    resource_id: str | None = Field(default=None, max_length=255)

    # Change tracking
    old_values: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    new_values: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))

    # Context
    ip_address: str | None = Field(default=None, max_length=45)
    user_agent: str | None = Field(default=None, sa_column=Column(Text))

    # Additional metadata
    audit_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Results
    success: bool = True
    error_message: str | None = Field(default=None, sa_column=Column(Text))

    # Timing
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AuditLog(AuditLogBase, table=True):
    """Audit log for security and compliance tracking."""

    __tablename__ = "audit_logs"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )


class PerformanceMetricBase(SQLModel):
    """Performance metric base model."""

    component: str = Field(max_length=100)  # agent, workflow, tool, api
    operation: str = Field(max_length=100)  # execute, create, update, delete

    # Timing metrics
    duration_ms: float = Field(ge=0.0)
    cpu_time_ms: float | None = Field(default=None, ge=0.0)
    memory_usage_mb: float | None = Field(default=None, ge=0.0)

    # Request/response metrics
    request_size_bytes: int | None = Field(default=None, ge=0)
    response_size_bytes: int | None = Field(default=None, ge=0)

    # LLM-specific metrics
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    llm_provider: str | None = Field(default=None, max_length=100)
    llm_model: str | None = Field(default=None, max_length=100)
    cost: float | None = Field(default=None, ge=0.0)

    # Context
    user_id: str | None = Field(default=None, max_length=255)
    session_id: str | None = Field(default=None, max_length=255)
    correlation_id: str | None = Field(default=None, max_length=255)

    # Status
    success: bool = True
    error_type: str | None = Field(default=None, max_length=100)

    # Additional metadata
    performance_metadata: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PerformanceMetric(PerformanceMetricBase, table=True):
    """Performance metrics for monitoring and optimization."""

    __tablename__ = "performance_metrics"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )


class ErrorLogBase(SQLModel):
    """Error log base model."""

    component: str = Field(max_length=100)
    error_type: str = Field(max_length=100)
    error_message: str = Field(sa_column=Column(Text))
    error_traceback: str | None = Field(default=None, sa_column=Column(Text))

    # Context
    user_id: str | None = Field(default=None, max_length=255)
    session_id: str | None = Field(default=None, max_length=255)
    correlation_id: str | None = Field(default=None, max_length=255)

    # Request context
    request_id: str | None = Field(default=None, max_length=255)
    endpoint: str | None = Field(default=None, max_length=500)
    method: str | None = Field(default=None, max_length=10)

    # Additional context
    context: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Severity
    severity: str = Field(
        default="error", max_length=20
    )  # debug, info, warning, error, critical

    # Resolution
    resolved: bool = False
    resolution_notes: str | None = Field(default=None, sa_column=Column(Text))

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ErrorLog(ErrorLogBase, table=True):
    """Error tracking for debugging and monitoring."""

    __tablename__ = "error_logs"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )


# Response models for API
class SystemMetricResponse(SystemMetricBase):
    """System metric response model."""

    id: uuid.UUID


class AuditLogResponse(AuditLogBase):
    """Audit log response model."""

    id: uuid.UUID


class PerformanceMetricResponse(PerformanceMetricBase):
    """Performance metric response model."""

    id: uuid.UUID


class ErrorLogResponse(ErrorLogBase):
    """Error log response model."""

    id: uuid.UUID
