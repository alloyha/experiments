"""
Conversation turn tracking model with idempotency support.

This model prevents duplicate processing and enables anti-echo functionality.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import ConfigDict
from sqlmodel import Field, SQLModel


class ConversationTurn(SQLModel, table=True):
    """
    Tracks individual conversation turns with idempotency keys.
    
    Prevents:
    - Duplicate message processing
    - Concurrent memory corruption
    - Response repetition
    
    Usage:
        # Create turn with idempotency
        turn = ConversationTurn(
            conversation_id=conv_id,
            turn_number=1,
            idempotency_key=f"{session_id}:{turn_id}",
            user_input_hash=sha256(message.encode()).hexdigest(),
            processing_status="PROCESSING"
        )
    """
    
    __tablename__ = "conversation_turns"
    
    # Primary key
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # Foreign keys (nullable for testing - will be made non-null when conversations table exists)
    conversation_id: UUID = Field(index=True)
    
    # Turn tracking
    turn_number: int = Field(description="Sequential turn number within conversation")
    idempotency_key: str = Field(
        unique=True,
        index=True,
        description="Unique key: session_id:turn_id format"
    )
    
    # Content hashing for deduplication
    user_input_hash: str = Field(
        description="SHA-256 hash of user input for duplicate detection"
    )
    agent_response_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of agent response"
    )
    
    # Processing status
    processing_status: str = Field(
        description="PROCESSING, COMPLETED, FAILED, DUPLICATE"
    )
    
    # Metadata
    user_input: str = Field(description="Original user message")
    agent_response: Optional[str] = Field(default=None, description="Agent's response")
    
    # Execution tracking (nullable - will add FK when executions table exists)
    execution_id: Optional[UUID] = Field(
        default=None,
        description="Associated workflow execution"
    )
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Error handling
    error_message: Optional[str] = Field(default=None)
    retry_count: int = Field(default=0)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                "turn_number": 1,
                "idempotency_key": "session_123:turn_001",
                "user_input_hash": "a1b2c3...",
                "processing_status": "COMPLETED",
                "user_input": "I need help with my order",
                "agent_response": "I can help you with that. What's your order number?"
            }
        }
    )
