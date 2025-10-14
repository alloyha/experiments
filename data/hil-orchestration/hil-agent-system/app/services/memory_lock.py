"""
Distributed locking service using Redis for conversation memory protection.

Prevents concurrent memory corruption and enables idempotent execution.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from uuid import uuid4

import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)


class MemoryLockService:
    """
    Redis-based distributed locking for conversation safety.
    
    Key Features:
    - Prevents concurrent memory corruption
    - Session-level locking (session_id)
    - Auto-expiring locks (timeout protection)
    - Idempotency tracking
    
    Usage:
        lock_service = MemoryLockService(redis_client)
        
        # Acquire lock for session
        async with lock_service.acquire_conversation_lock("session_123"):
            # Safe to modify conversation memory
            await update_memory(...)
            
        # Check idempotency
        if await lock_service.ensure_idempotent_execution("unique_key"):
            # Safe to proceed
            await process_request()
    """
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize lock service.
        
        Args:
            redis_client: Async Redis client instance
        """
        self.redis = redis_client
        self.lock_prefix = "lock:conversation:"
        self.idempotency_prefix = "idempotency:"
        self.default_timeout = 30  # seconds
        self.lock_poll_interval = 0.1  # seconds
    
    @asynccontextmanager
    async def acquire_conversation_lock(
        self,
        session_id: str,
        timeout: int = 30,
        max_wait: int = 60
    ) -> AsyncIterator[str]:
        """
        Acquire distributed lock for a conversation session.
        
        Prevents concurrent operations on the same conversation that could
        corrupt memory or cause race conditions.
        
        Args:
            session_id: Unique session identifier
            timeout: Lock expiration time in seconds (auto-release)
            max_wait: Maximum time to wait for lock acquisition
            
        Yields:
            lock_id: Unique identifier for this lock acquisition
            
        Raises:
            TimeoutError: If lock cannot be acquired within max_wait
            
        Example:
            async with lock_service.acquire_conversation_lock("session_123"):
                # Only one worker can execute this block at a time
                await conversation.add_memory(...)
        """
        lock_key = f"{self.lock_prefix}{session_id}"
        lock_id = str(uuid4())
        
        logger.info(
            "attempting_lock_acquisition",
            session_id=session_id,
            lock_key=lock_key,
            lock_id=lock_id,
            timeout=timeout
        )
        
        # Try to acquire lock
        acquired = False
        elapsed = 0.0
        
        while not acquired and elapsed < max_wait:
            # SET NX (set if not exists) with expiration
            acquired = await self.redis.set(
                lock_key,
                lock_id,
                nx=True,  # Only set if doesn't exist
                ex=timeout  # Expire after timeout seconds
            )
            
            if acquired:
                logger.info(
                    "lock_acquired",
                    session_id=session_id,
                    lock_id=lock_id,
                    elapsed=elapsed
                )
                break
            
            # Wait before retry
            await asyncio.sleep(self.lock_poll_interval)
            elapsed += self.lock_poll_interval
        
        if not acquired:
            logger.error(
                "lock_acquisition_timeout",
                session_id=session_id,
                max_wait=max_wait,
                elapsed=elapsed
            )
            raise TimeoutError(
                f"Could not acquire lock for session {session_id} "
                f"after {max_wait} seconds"
            )
        
        try:
            yield lock_id
        finally:
            # Release lock only if we still own it
            # Use Lua script for atomic check-and-delete
            release_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            
            released = await self.redis.eval(
                release_script,
                1,  # Number of keys
                lock_key,
                lock_id
            )
            
            if released:
                logger.info(
                    "lock_released",
                    session_id=session_id,
                    lock_id=lock_id
                )
            else:
                logger.warning(
                    "lock_already_released",
                    session_id=session_id,
                    lock_id=lock_id,
                    reason="Lock expired or owned by another process"
                )
    
    async def ensure_idempotent_execution(
        self,
        idempotency_key: str,
        ttl: int = 3600
    ) -> bool:
        """
        Check if operation with given key has already been processed.
        
        Uses Redis SET NX to atomically check and mark as processed.
        
        Args:
            idempotency_key: Unique key for operation (e.g., "session:turn")
            ttl: Time to live for idempotency record (seconds)
            
        Returns:
            True if safe to proceed (not processed before)
            False if already processed (duplicate)
            
        Example:
            key = f"{session_id}:{turn_id}"
            if await lock_service.ensure_idempotent_execution(key):
                # First time seeing this request
                await process_turn(...)
            else:
                # Duplicate - skip processing
                logger.warning("duplicate_request", key=key)
        """
        redis_key = f"{self.idempotency_prefix}{idempotency_key}"
        
        # Try to set key - returns True if key didn't exist
        is_first_execution = await self.redis.set(
            redis_key,
            "processed",
            nx=True,  # Only set if not exists
            ex=ttl
        )
        
        if is_first_execution:
            logger.info(
                "idempotency_check_passed",
                idempotency_key=idempotency_key,
                status="first_execution"
            )
        else:
            logger.warning(
                "idempotency_check_failed",
                idempotency_key=idempotency_key,
                status="duplicate_request"
            )
        
        return bool(is_first_execution)
    
    async def check_if_processed(
        self,
        idempotency_key: str
    ) -> bool:
        """
        Check if an operation has been processed without marking it.
        
        Args:
            idempotency_key: Unique key for operation
            
        Returns:
            True if already processed, False otherwise
        """
        redis_key = f"{self.idempotency_prefix}{idempotency_key}"
        exists = await self.redis.exists(redis_key)
        return bool(exists)
    
    async def get_lock_info(
        self,
        session_id: str
    ) -> Optional[dict]:
        """
        Get information about current lock for a session.
        
        Args:
            session_id: Session to check
            
        Returns:
            Dict with lock_id and ttl, or None if no lock
        """
        lock_key = f"{self.lock_prefix}{session_id}"
        
        # Get lock owner
        lock_id = await self.redis.get(lock_key)
        if not lock_id:
            return None
        
        # Get TTL
        ttl = await self.redis.ttl(lock_key)
        
        return {
            "lock_id": lock_id.decode() if isinstance(lock_id, bytes) else lock_id,
            "ttl": ttl,
            "session_id": session_id
        }
    
    async def force_release_lock(
        self,
        session_id: str
    ) -> bool:
        """
        Force release a lock (admin operation).
        
        WARNING: Use with caution - can cause race conditions
        
        Args:
            session_id: Session whose lock to release
            
        Returns:
            True if lock was released, False if no lock existed
        """
        lock_key = f"{self.lock_prefix}{session_id}"
        deleted = await self.redis.delete(lock_key)
        
        if deleted:
            logger.warning(
                "lock_force_released",
                session_id=session_id,
                reason="admin_action"
            )
        
        return bool(deleted)
    
    async def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.
        
        Returns:
            True if Redis is accessible
        """
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.error("redis_health_check_failed", error=str(e))
            return False
