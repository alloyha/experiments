"""
Anti-echo service to prevent repetitive agent responses.

Detects and suppresses duplicate or highly similar responses within conversations.
"""

import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation_turn import ConversationTurn

logger = structlog.get_logger(__name__)


class AntiEchoMemory:
    """
    Prevents the agent from repeating itself within conversations.
    
    Key Features:
    - Detects exact duplicate responses (100% match)
    - Detects highly similar responses (>90% similarity)
    - Maintains sliding window of recent responses
    - Configurable similarity thresholds
    - Performance optimized with hash-based lookups
    
    Usage:
        anti_echo = AntiEchoMemory(db_session)
        
        # Check if response would be repetitive
        should_block, reason = await anti_echo.should_suppress_response(
            conversation_id=conv_id,
            proposed_response="Hello! How can I help you?"
        )
        
        if should_block:
            logger.warning("response_suppressed", reason=reason)
            # Generate alternative response
        else:
            # Safe to send response
            await send_response(...)
    """
    
    def __init__(
        self,
        db: AsyncSession,
        window_size: int = 10,
        exact_match_threshold: float = 1.0,
        high_similarity_threshold: float = 0.9,
        min_response_length: int = 10
    ):
        """
        Initialize anti-echo service.
        
        Args:
            db: Database session
            window_size: Number of recent responses to check (default: 10)
            exact_match_threshold: Threshold for exact duplicates (default: 1.0)
            high_similarity_threshold: Threshold for similar responses (default: 0.9)
            min_response_length: Minimum response length to check (default: 10 chars)
        """
        self.db = db
        self.window_size = window_size
        self.exact_match_threshold = exact_match_threshold
        self.high_similarity_threshold = high_similarity_threshold
        self.min_response_length = min_response_length
    
    async def should_suppress_response(
        self,
        conversation_id: str,
        proposed_response: str,
        time_window_minutes: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if proposed response is too similar to recent responses.
        
        Args:
            conversation_id: UUID of conversation
            proposed_response: Response text to check
            time_window_minutes: Optional time window to check (None = window_size)
            
        Returns:
            Tuple of (should_suppress: bool, reason: Optional[str])
            - (True, reason) if response should be suppressed
            - (False, None) if response is acceptable
            
        Example:
            should_block, reason = await anti_echo.should_suppress_response(
                conv_id,
                "Hello! How can I help you today?"
            )
            
            if should_block:
                print(f"Blocked: {reason}")
        """
        # Skip check for very short responses
        if len(proposed_response.strip()) < self.min_response_length:
            return False, None
        
        # Get recent responses from conversation
        recent_responses = await self._get_recent_responses(
            conversation_id,
            time_window_minutes
        )
        
        if not recent_responses:
            # No history to compare against
            return False, None
        
        # Hash the proposed response for comparison
        proposed_hash = self._hash_text(proposed_response)
        
        # Check for exact duplicates
        for turn in recent_responses:
            if turn.agent_response_hash == proposed_hash:
                logger.warning(
                    "exact_duplicate_detected",
                    conversation_id=conversation_id,
                    original_turn=turn.turn_number,
                    proposed_response_preview=proposed_response[:100]
                )
                return True, f"Exact duplicate of turn {turn.turn_number}"
        
        # Check for high similarity
        for turn in recent_responses:
            if turn.agent_response:
                similarity = self._calculate_similarity(
                    proposed_response,
                    turn.agent_response
                )
                
                if similarity >= self.high_similarity_threshold:
                    logger.warning(
                        "high_similarity_detected",
                        conversation_id=conversation_id,
                        original_turn=turn.turn_number,
                        similarity=similarity,
                        proposed_response_preview=proposed_response[:100]
                    )
                    return True, (
                        f"Too similar ({similarity:.1%}) to turn {turn.turn_number}"
                    )
        
        logger.info(
            "response_acceptable",
            conversation_id=conversation_id,
            checked_against=len(recent_responses)
        )
        
        return False, None
    
    async def _get_recent_responses(
        self,
        conversation_id: str,
        time_window_minutes: Optional[int] = None
    ) -> List[ConversationTurn]:
        """
        Retrieve recent responses from conversation history.
        
        Args:
            conversation_id: UUID of conversation (as string)
            time_window_minutes: Optional time window (None = use window_size)
            
        Returns:
            List of recent ConversationTurn objects with responses
        """
        # Convert string to UUID for query
        conv_uuid = UUID(conversation_id) if isinstance(conversation_id, str) else conversation_id
        
        query = (
            select(ConversationTurn)
            .where(ConversationTurn.conversation_id == conv_uuid)
            .where(ConversationTurn.agent_response.isnot(None))
            .where(ConversationTurn.processing_status == "COMPLETED")
            .order_by(ConversationTurn.turn_number.desc())
        )
        
        # Apply time window if specified
        if time_window_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            query = query.where(ConversationTurn.created_at >= cutoff_time)
        else:
            # Use window_size limit
            query = query.limit(self.window_size)
        
        result = await self.db.execute(query)
        turns = result.scalars().all()
        
        return list(turns)
    
    def _hash_text(self, text: str) -> str:
        """
        Create SHA-256 hash of text.
        
        Must match the hashing used in TurnManager for consistency.
        
        Args:
            text: Text to hash
            
        Returns:
            Hex digest of hash
        """
        # Use same hashing as TurnManager (no normalization)
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Uses Jaccard similarity on word sets for efficiency.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Normalize texts
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity: intersection / union
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def get_response_history(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[dict]:
        """
        Get response history for analysis/debugging.
        
        Args:
            conversation_id: UUID of conversation
            limit: Maximum number of responses to return
            
        Returns:
            List of response summaries with metadata
        """
        recent_responses = await self._get_recent_responses(
            conversation_id,
            time_window_minutes=None
        )
        
        history = []
        for turn in recent_responses[:limit]:
            history.append({
                "turn_number": turn.turn_number,
                "response_preview": turn.agent_response[:100] if turn.agent_response else None,
                "response_hash": turn.agent_response_hash,
                "created_at": turn.created_at.isoformat() if turn.created_at else None,
                "status": turn.processing_status
            })
        
        return history
    
    async def analyze_conversation_patterns(
        self,
        conversation_id: str
    ) -> dict:
        """
        Analyze conversation for repetitive patterns.
        
        Args:
            conversation_id: UUID of conversation
            
        Returns:
            Analysis report with statistics
        """
        all_responses = await self._get_recent_responses(
            conversation_id,
            time_window_minutes=None  # Get all
        )
        
        if not all_responses:
            return {
                "total_responses": 0,
                "unique_responses": 0,
                "duplicate_count": 0,
                "average_similarity": 0.0,
                "patterns": []
            }
        
        # Track unique response hashes
        unique_hashes = set()
        duplicate_count = 0
        similarities = []
        
        for i, turn in enumerate(all_responses):
            if turn.agent_response_hash:
                if turn.agent_response_hash in unique_hashes:
                    duplicate_count += 1
                else:
                    unique_hashes.add(turn.agent_response_hash)
            
            # Calculate similarity with previous response
            if i > 0 and turn.agent_response and all_responses[i-1].agent_response:
                sim = self._calculate_similarity(
                    turn.agent_response,
                    all_responses[i-1].agent_response
                )
                similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Find most repeated responses
        hash_counts = {}
        for turn in all_responses:
            if turn.agent_response_hash:
                hash_counts[turn.agent_response_hash] = hash_counts.get(
                    turn.agent_response_hash, 0
                ) + 1
        
        patterns = [
            {
                "hash": hash_val,
                "count": count,
                "example": next(
                    t.agent_response[:100] for t in all_responses 
                    if t.agent_response_hash == hash_val
                )
            }
            for hash_val, count in sorted(
                hash_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5 patterns
        ]
        
        return {
            "total_responses": len(all_responses),
            "unique_responses": len(unique_hashes),
            "duplicate_count": duplicate_count,
            "average_similarity": round(avg_similarity, 3),
            "patterns": patterns
        }
    
    async def clear_old_responses(
        self,
        conversation_id: str,
        keep_recent: int = 20
    ) -> int:
        """
        Clear old responses to manage memory (optional maintenance).
        
        Args:
            conversation_id: UUID of conversation (as string)
            keep_recent: Number of recent turns to keep
            
        Returns:
            Number of responses cleared
        """
        # Convert string to UUID for query
        conv_uuid = UUID(conversation_id) if isinstance(conversation_id, str) else conversation_id
        
        # Get all turns for conversation
        query = (
            select(ConversationTurn)
            .where(ConversationTurn.conversation_id == conv_uuid)
            .order_by(ConversationTurn.turn_number.desc())
        )
        
        result = await self.db.execute(query)
        all_turns = list(result.scalars().all())
        
        if len(all_turns) <= keep_recent:
            return 0
        
        # Clear agent_response and hash from old turns
        turns_to_clear = all_turns[keep_recent:]
        cleared = 0
        
        for turn in turns_to_clear:
            if turn.agent_response:
                turn.agent_response = None
                turn.agent_response_hash = None
                cleared += 1
        
        await self.db.commit()
        
        logger.info(
            "old_responses_cleared",
            conversation_id=conversation_id,
            cleared=cleared,
            kept=keep_recent
        )
        
        return cleared
