"""
Agent-related database models using SQLModel.
"""

import uuid
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Column, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlmodel import Field, SQLModel


class AgentType(str, Enum):
    """Agent type enumeration."""

    SIMPLE = "simple"
    REASONING = "reasoning"
    CODE = "code"


class ModelProfile(str, Enum):
    """LLM model profile enumeration."""

    FAST = "fast"
    BALANCED = "balanced"
    POWERFUL = "powerful"


class ExecutionStatus(str, Enum):
    """Execution status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class AgentBase(SQLModel):
    """Agent base model."""

    name: str = Field(max_length=255, unique=True, index=True)
    agent_type: AgentType
    description: str | None = Field(default=None, sa_column=Column(Text))

    # Configuration
    default_model_profile: ModelProfile = ModelProfile.BALANCED
    default_timeout: int = 300  # seconds
    max_iterations: int = 10  # for reasoning agents

    # Prompt templates
    system_prompt: str | None = Field(default=None, sa_column=Column(Text))
    user_prompt_template: str | None = Field(default=None, sa_column=Column(Text))

    # Metadata
    agent_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Audit fields
    created_by: str | None = Field(default=None, max_length=255)
    is_active: bool = True


class Agent(AgentBase, table=True):
    """Agent configuration table."""

    __tablename__ = "agents"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = Field(default=None)


class AgentExecutionBase(SQLModel):
    """Agent execution base model."""

    agent_id: uuid.UUID = Field(sa_column=Column(UUID(as_uuid=True)))

    # Request data
    input_data: dict[str, Any] = Field(sa_column=Column(JSON))
    model_profile: ModelProfile
    timeout: int

    # Execution tracking
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    execution_time: float | None = None  # seconds

    # Results
    result: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    error_message: str | None = Field(default=None, sa_column=Column(Text))
    error_traceback: str | None = Field(default=None, sa_column=Column(Text))

    # LLM usage tracking
    llm_provider: str | None = Field(default=None, max_length=100)
    llm_model: str | None = Field(default=None, max_length=100)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    # Iterations (for reasoning agents)
    iterations: int = 1
    reasoning_steps: list[dict[str, Any]] | None = Field(
        default=None, sa_column=Column(JSON)
    )

    # Tools used
    tools_used: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    tool_executions: int = 0

    # Context
    correlation_id: str | None = Field(default=None, max_length=255)
    workflow_execution_id: uuid.UUID | None = Field(
        default=None, sa_column=Column(UUID(as_uuid=True))
    )


class AgentExecution(AgentExecutionBase, table=True):
    """Agent execution tracking table."""

    __tablename__ = "agent_executions"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# Response models for API
class AgentResponse(AgentBase):
    """Agent response model."""

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None


class AgentExecutionResponse(AgentExecutionBase):
    """Agent execution response model."""

    id: uuid.UUID
    created_at: datetime
