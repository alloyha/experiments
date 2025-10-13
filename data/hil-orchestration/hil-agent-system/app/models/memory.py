"""
Memory and conversation management models using SQLModel.
"""

import uuid
from datetime import UTC, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Column, DateTime, Float, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlmodel import Field, SQLModel


class MemoryType(str, Enum):
    """Memory type enumeration."""

    SHORT_TERM = "short_term"  # Working memory for current conversation
    LONG_TERM = "long_term"  # Persistent memories across sessions
    EPISODIC = "episodic"  # Specific events and interactions
    SEMANTIC = "semantic"  # Facts and knowledge
    PROCEDURAL = "procedural"  # How-to knowledge and patterns


class MessageType(str, Enum):
    """Conversation message type enumeration."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    ERROR = "error"


class MemoryBase(SQLModel):
    """Memory base model."""

    agent_id: uuid.UUID | None = Field(
        default=None, sa_column=Column(UUID(as_uuid=True))
    )
    memory_type: MemoryType

    # Core content
    content: str = Field(sa_column=Column(Text))
    embedding: list[float] | None = Field(default=None, sa_column=Column(JSON))

    # Context and metadata
    context: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    # Session and correlation
    session_id: str | None = Field(default=None, max_length=255)
    workflow_execution_id: uuid.UUID | None = Field(
        default=None, sa_column=Column(UUID(as_uuid=True))
    )
    correlation_id: str | None = Field(default=None, max_length=255)

    # Importance and access tracking
    importance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    last_accessed: datetime | None = None

    # Lifecycle
    expires_at: datetime | None = None
    is_active: bool = True


class Memory(MemoryBase, table=True):
    """Memory storage table."""

    __tablename__ = "memories"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = Field(default=None)


class ConversationMessageBase(SQLModel):
    """Conversation message base model."""

    session_id: str = Field(max_length=255, index=True)

    # Message content
    message_type: MessageType
    content: str = Field(sa_column=Column(Text))

    # Tool-specific fields
    tool_name: str | None = Field(default=None, max_length=255)
    tool_action: str | None = Field(default=None, max_length=255)

    # Message metadata
    message_metadata: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )

    # Ordering
    sequence_number: int = Field(ge=0)

    # Relationships
    agent_id: uuid.UUID | None = Field(
        default=None, sa_column=Column(UUID(as_uuid=True))
    )
    workflow_execution_id: uuid.UUID | None = Field(
        default=None, sa_column=Column(UUID(as_uuid=True))
    )


class ConversationMessage(ConversationMessageBase, table=True):
    """Conversation message tracking table."""

    __tablename__ = "conversation_messages"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationSessionBase(SQLModel):
    """Conversation session base model."""

    session_id: str = Field(max_length=255, unique=True, index=True)

    # Session metadata
    title: str | None = Field(default=None, max_length=500)
    summary: str | None = Field(default=None, sa_column=Column(Text))

    # Participants
    user_id: str | None = Field(default=None, max_length=255)
    agent_ids: list[uuid.UUID] = Field(default_factory=list, sa_column=Column(JSON))

    # Session configuration
    session_config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Statistics
    message_count: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    total_cost: float = Field(default=0.0, ge=0.0)

    # Lifecycle
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime | None = None
    ended_at: datetime | None = None
    is_active: bool = True


class ConversationSession(ConversationSessionBase, table=True):
    """Conversation session tracking table."""

    __tablename__ = "conversation_sessions"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = Field(default=None)


class KnowledgeEntryBase(SQLModel):
    """Knowledge entry base model."""

    title: str = Field(max_length=500)
    content: str = Field(sa_column=Column(Text))
    knowledge_type: str = Field(max_length=100)  # fact, procedure, concept, etc.

    # Semantic information
    embedding: list[float] | None = Field(default=None, sa_column=Column(JSON))
    keywords: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    categories: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    # Source and validation
    source: str | None = Field(default=None, max_length=500)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    verified: bool = False

    # Usage tracking
    access_count: int = Field(default=0, ge=0)
    last_accessed: datetime | None = None

    # Relationships
    related_entries: list[uuid.UUID] = Field(
        default_factory=list, sa_column=Column(JSON)
    )

    # Lifecycle
    is_active: bool = True


class KnowledgeEntry(KnowledgeEntryBase, table=True):
    """Knowledge base entry table."""

    __tablename__ = "knowledge_entries"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = Field(default=None)


# Response models for API
class MemoryResponse(MemoryBase):
    """Memory response model."""

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None


class ConversationMessageResponse(ConversationMessageBase):
    """Conversation message response model."""

    id: uuid.UUID
    created_at: datetime


class ConversationSessionResponse(ConversationSessionBase):
    """Conversation session response model."""

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None


class KnowledgeEntryResponse(KnowledgeEntryBase):
    """Knowledge entry response model."""

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime | None
