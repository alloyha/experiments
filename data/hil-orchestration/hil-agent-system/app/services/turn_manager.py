"""
Turn management service for conversation idempotency and tracking.

Prevents duplicate processing and enables anti-echo functionality.
"""

import hashlib
from datetime import datetime
from typing import Optional
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation_turn import ConversationTurn

logger = structlog.get_logger(__name__)


class TurnManager:
    """
    Manages conversation turns with idempotency guarantees.
    
    Key Features:
    - Idempotent turn creation (session_id:turn_id)
    - Duplicate message detection via hashing
    - Turn status tracking (PROCESSING → COMPLETED)
    - Automatic turn numbering
    
    Usage:
        turn_manager = TurnManager(db_session)
        
        # Create new turn
        turn = await turn_manager.create_turn(
            conversation_id=conv_id,
            session_id="session_123",
            turn_id="turn_001",
            user_input="Hello"
        )
        
        # Check if already processed
        existing = await turn_manager.get_by_idempotency_key(
            "session_123:turn_001"
        )
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_turn(
        self,
        conversation_id: UUID,
        session_id: str,
        turn_id: str,
        user_input: str
    ) -> ConversationTurn:
        """
        Create a new conversation turn with idempotency.
        
        Args:
            conversation_id: UUID of conversation
            session_id: Session identifier
            turn_id: Turn identifier (unique within session)
            user_input: User's message
            
        Returns:
            ConversationTurn instance
            
        Raises:
            ValueError: If turn with same idempotency key already exists
        """
        
        # Create idempotency key
        idempotency_key = f"{session_id}:{turn_id}"
        
        # Check if already exists
        existing = await self.get_by_idempotency_key(idempotency_key)
        if existing:
            logger.warning(
                "duplicate_turn_creation_attempt",
                idempotency_key=idempotency_key,
                existing_status=existing.processing_status
            )
            raise ValueError(
                f"Turn with key {idempotency_key} already exists "
                f"(status: {existing.processing_status})"
            )
        
        # Hash user input for duplicate detection
        user_input_hash = self._hash_content(user_input)
        
        # Get next turn number
        turn_number = await self._get_next_turn_number(conversation_id)
        
        # Create turn
        turn = ConversationTurn(
            conversation_id=conversation_id,
            turn_number=turn_number,
            idempotency_key=idempotency_key,
            user_input_hash=user_input_hash,
            user_input=user_input,
            processing_status="PROCESSING"
        )
        
        self.db.add(turn)
        await self.db.commit()
        await self.db.refresh(turn)
        
        logger.info(
            "turn_created",
            turn_id=str(turn.id),
            conversation_id=str(conversation_id),
            turn_number=turn_number,
            idempotency_key=idempotency_key
        )
        
        return turn
    
    async def complete_turn(
        self,
        turn_id: UUID,
        agent_response: str,
        execution_id: Optional[UUID] = None
    ) -> ConversationTurn:
        """
        Mark turn as completed with agent response.
        
        Args:
            turn_id: Turn UUID
            agent_response: Agent's response text
            execution_id: Optional execution ID
            
        Returns:
            Updated turn
        """
        
        turn = await self.get_by_id(turn_id)
        if not turn:
            raise ValueError(f"Turn {turn_id} not found")
        
        # Hash response for deduplication
        turn.agent_response = agent_response
        turn.agent_response_hash = self._hash_content(agent_response)
        turn.processing_status = "COMPLETED"
        turn.completed_at = datetime.now()
        
        if execution_id:
            turn.execution_id = execution_id
        
        await self.db.commit()
        await self.db.refresh(turn)
        
        logger.info(
            "turn_completed",
            turn_id=str(turn_id),
            processing_time_ms=int(
                (turn.completed_at - turn.created_at).total_seconds() * 1000
            )
        )
        
        return turn
    
    async def fail_turn(
        self,
        turn_id: UUID,
        error_message: str
    ) -> ConversationTurn:
        """
        Mark turn as failed.
        
        Args:
            turn_id: Turn UUID
            error_message: Error description
            
        Returns:
            Updated turn
        """
        
        turn = await self.get_by_id(turn_id)
        if not turn:
            raise ValueError(f"Turn {turn_id} not found")
        
        turn.processing_status = "FAILED"
        turn.error_message = error_message
        turn.completed_at = datetime.now()
        
        await self.db.commit()
        await self.db.refresh(turn)
        
        logger.error(
            "turn_failed",
            turn_id=str(turn_id),
            error=error_message
        )
        
        return turn
    
    async def get_by_id(self, turn_id: UUID) -> Optional[ConversationTurn]:
        """Get turn by ID."""
        result = await self.db.execute(
            select(ConversationTurn).where(ConversationTurn.id == turn_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_idempotency_key(
        self,
        idempotency_key: str
    ) -> Optional[ConversationTurn]:
        """
        Get turn by idempotency key.
        
        Used to check if turn already processed.
        
        Args:
            idempotency_key: Format "session_id:turn_id"
            
        Returns:
            Existing turn or None
        """
        result = await self.db.execute(
            select(ConversationTurn).where(
                ConversationTurn.idempotency_key == idempotency_key
            )
        )
        return result.scalar_one_or_none()
    
    async def get_conversation_turns(
        self,
        conversation_id: UUID,
        limit: int = 50
    ) -> list[ConversationTurn]:
        """
        Get recent turns for conversation.
        
        Args:
            conversation_id: Conversation UUID
            limit: Max turns to return
            
        Returns:
            List of turns ordered by turn_number
        """
        result = await self.db.execute(
            select(ConversationTurn)
            .where(ConversationTurn.conversation_id == conversation_id)
            .order_by(ConversationTurn.turn_number.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def check_duplicate_input(
        self,
        conversation_id: UUID,
        user_input: str,
        window_size: int = 5
    ) -> Optional[ConversationTurn]:
        """
        Check if user input is duplicate of recent turn.
        
        Args:
            conversation_id: Conversation UUID
            user_input: User's message
            window_size: How many recent turns to check
            
        Returns:
            Matching turn or None
        """
        
        input_hash = self._hash_content(user_input)
        
        # Get recent turns
        recent_turns = await self.get_conversation_turns(
            conversation_id,
            limit=window_size
        )
        
        # Check for matching hash
        for turn in recent_turns:
            if turn.user_input_hash == input_hash:
                logger.info(
                    "duplicate_input_detected",
                    conversation_id=str(conversation_id),
                    original_turn=str(turn.id),
                    original_turn_number=turn.turn_number
                )
                return turn
        
        return None
    
    async def _get_next_turn_number(self, conversation_id: UUID) -> int:
        """Get next sequential turn number for conversation."""
        
        result = await self.db.execute(
            select(ConversationTurn)
            .where(ConversationTurn.conversation_id == conversation_id)
            .order_by(ConversationTurn.turn_number.desc())
            .limit(1)
        )
        
        last_turn = result.scalar_one_or_none()
        return (last_turn.turn_number + 1) if last_turn else 1
    
    @staticmethod
    def _hash_content(content: str) -> str:
        """
        Create SHA-256 hash of content.
        
        Used for duplicate detection.
        
        Args:
            content: Text to hash
            
        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(content.encode()).hexdigest()
