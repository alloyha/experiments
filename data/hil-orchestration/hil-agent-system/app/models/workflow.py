"""
Workflow-related database models using SQLModel.
"""

import uuid
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Column, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlmodel import Field, SQLModel


class WorkflowStatus(str, Enum):
    """Workflow status enumeration."""

    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"


class WorkflowExecutionStatus(str, Enum):
    """Workflow execution status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class NodeStatus(str, Enum):
    """Node execution status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowBase(SQLModel):
    """Workflow base model."""

    name: str = Field(max_length=255)
    description: str | None = Field(default=None, sa_column=Column(Text))
    version: str = Field(default="1.0.0", max_length=50)

    # Workflow definition (DAG)
    nodes: dict[str, Any] = Field(sa_column=Column(JSON))  # Node definitions
    edges: dict[str, Any] = Field(
        sa_column=Column(JSON)
    )  # Edge definitions with conditions

    # Configuration
    max_execution_time: int = 1800  # 30 minutes
    max_parallel_nodes: int = 10
    retry_policy: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Status and metadata
    status: WorkflowStatus = WorkflowStatus.DRAFT
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    workflow_metadata: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )

    # Statistics
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time: float | None = None
    avg_cost: float | None = None

    # Audit fields
    created_by: str | None = Field(default=None, max_length=255)


class Workflow(WorkflowBase, table=True):
    """Workflow definition table."""

    __tablename__ = "workflows"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = Field(default=None)


class WorkflowExecutionBase(SQLModel):
    """Workflow execution base model."""

    workflow_id: uuid.UUID = Field(sa_column=Column(UUID(as_uuid=True)))

    # Input and configuration
    input_data: dict[str, Any] = Field(sa_column=Column(JSON))
    configuration: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Execution tracking
    status: WorkflowExecutionStatus = WorkflowExecutionStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    execution_time: float | None = None  # seconds

    # Results
    output_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    error_message: str | None = Field(default=None, sa_column=Column(Text))
    error_traceback: str | None = Field(default=None, sa_column=Column(Text))

    # Execution statistics
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    skipped_nodes: int = 0

    # Cost tracking
    total_cost: float = 0.0
    llm_cost: float = 0.0
    tool_cost: float = 0.0

    # Context
    correlation_id: str | None = Field(default=None, max_length=255)
    user_id: str | None = Field(default=None, max_length=255)
    source: str | None = Field(default=None, max_length=100)  # API, CLI, UI, etc.


class WorkflowExecution(WorkflowExecutionBase, table=True):
    """Workflow execution tracking table."""

    __tablename__ = "workflow_executions"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NodeExecutionBase(SQLModel):
    """Node execution base model."""

    workflow_execution_id: uuid.UUID = Field(sa_column=Column(UUID(as_uuid=True)))
    node_id: str = Field(max_length=255)  # Node identifier within workflow

    # Node configuration
    node_type: str = Field(max_length=100)  # agent, condition, parallel, etc.
    node_config: dict[str, Any] = Field(sa_column=Column(JSON))

    # Execution tracking
    status: NodeStatus = NodeStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    execution_time: float | None = None  # seconds
    attempt: int = 1

    # Input/Output
    input_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    output_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    error_message: str | None = Field(default=None, sa_column=Column(Text))
    error_traceback: str | None = Field(default=None, sa_column=Column(Text))

    # Agent execution reference (if applicable)
    agent_execution_id: uuid.UUID | None = Field(
        default=None, sa_column=Column(UUID(as_uuid=True))
    )

    # Cost
    cost: float = 0.0


class NodeExecution(NodeExecutionBase, table=True):
    """Node execution tracking table."""

    __tablename__ = "node_executions"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# Response models for API
class WorkflowResponse(WorkflowBase):
    """Workflow response model."""

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None


class WorkflowExecutionResponse(WorkflowExecutionBase):
    """Workflow execution response model."""

    id: uuid.UUID
    created_at: datetime


class NodeExecutionResponse(NodeExecutionBase):
    """Node execution response model."""

    id: uuid.UUID
    created_at: datetime
