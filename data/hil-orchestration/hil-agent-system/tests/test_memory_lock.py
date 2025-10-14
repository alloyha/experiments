"""
Tests for MemoryLockService - Redis distributed locking.
"""

import asyncio
from uuid import uuid4

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from app.services.memory_lock import MemoryLockService


@pytest_asyncio.fixture
async def redis_client():
    """Create fake Redis client for testing."""
    # Use fakeredis for testing without needing a real Redis server
    client = FakeAsyncRedis(
        decode_responses=True,
        version=7  # Simulate Redis 7.x
    )
    
    # Clear test database before tests
    await client.flushdb()
    
    yield client
    
    # Cleanup after tests
    await client.flushdb()
    await client.aclose()


@pytest_asyncio.fixture
async def lock_service(redis_client):
    """Create MemoryLockService instance."""
    return MemoryLockService(redis_client)


@pytest.mark.asyncio
async def test_acquire_and_release_lock(lock_service):
    """Test basic lock acquisition and release."""
    session_id = "test-session-001"
    
    # Acquire lock
    async with lock_service.acquire_conversation_lock(session_id) as lock_id:
        assert lock_id is not None
        
        # Lock should exist in Redis
        info = await lock_service.get_lock_info(session_id)
        assert info is not None
        assert info["lock_id"] == lock_id
        assert info["ttl"] > 0
    
    # Lock should be released after context exit
    info = await lock_service.get_lock_info(session_id)
    assert info is None


@pytest.mark.asyncio
async def test_concurrent_lock_blocks(lock_service):
    """Test that concurrent lock attempts block correctly."""
    session_id = "test-session-002"
    results = []
    
    async def worker(worker_id: int, delay: float):
        """Worker that acquires lock and records timing."""
        await asyncio.sleep(delay)
        start = asyncio.get_event_loop().time()
        
        async with lock_service.acquire_conversation_lock(session_id, timeout=5):
            end = asyncio.get_event_loop().time()
            results.append({
                "worker_id": worker_id,
                "wait_time": end - start
            })
            # Hold lock briefly
            await asyncio.sleep(0.5)
    
    # Start two workers simultaneously
    await asyncio.gather(
        worker(1, 0.0),  # First worker
        worker(2, 0.0)   # Second worker (should wait)
    )
    
    assert len(results) == 2
    # First worker should acquire immediately
    assert results[0]["wait_time"] < 0.1
    # Second worker should wait ~0.5 seconds
    assert results[1]["wait_time"] > 0.4


@pytest.mark.asyncio
async def test_lock_timeout_prevents_deadlock(lock_service):
    """Test that locks auto-expire to prevent deadlocks."""
    session_id = "test-session-003"
    
    # Acquire lock with 1 second timeout
    async with lock_service.acquire_conversation_lock(session_id, timeout=1):
        # Simulate process crash (don't release)
        pass
    
    # Wait for lock to expire
    await asyncio.sleep(1.5)
    
    # Should be able to acquire again
    async with lock_service.acquire_conversation_lock(session_id, timeout=1):
        info = await lock_service.get_lock_info(session_id)
        assert info is not None


@pytest.mark.asyncio
async def test_lock_acquisition_timeout(lock_service):
    """Test timeout when lock cannot be acquired."""
    session_id = "test-session-004"
    
    # Hold lock for 5 seconds
    async def long_operation():
        async with lock_service.acquire_conversation_lock(session_id, timeout=10):
            await asyncio.sleep(5)
    
    # Start long operation
    task = asyncio.create_task(long_operation())
    await asyncio.sleep(0.1)  # Let it acquire lock
    
    # Try to acquire with short max_wait
    with pytest.raises(TimeoutError, match="Could not acquire lock"):
        async with lock_service.acquire_conversation_lock(
            session_id,
            timeout=10,
            max_wait=1  # Only wait 1 second
        ):
            pass
    
    await task


@pytest.mark.asyncio
async def test_idempotent_execution_first_time(lock_service):
    """Test idempotency check on first execution."""
    key = f"test-operation-{uuid4()}"
    
    # First execution should return True
    result = await lock_service.ensure_idempotent_execution(key)
    assert result is True


@pytest.mark.asyncio
async def test_idempotent_execution_duplicate(lock_service):
    """Test idempotency check detects duplicates."""
    key = f"test-operation-{uuid4()}"
    
    # First execution
    result1 = await lock_service.ensure_idempotent_execution(key)
    assert result1 is True
    
    # Duplicate execution should return False
    result2 = await lock_service.ensure_idempotent_execution(key)
    assert result2 is False


@pytest.mark.asyncio
async def test_idempotency_ttl_expiration(lock_service):
    """Test that idempotency records expire."""
    key = f"test-operation-{uuid4()}"
    
    # Mark as processed with 1 second TTL
    result1 = await lock_service.ensure_idempotent_execution(key, ttl=1)
    assert result1 is True
    
    # Wait for expiration
    await asyncio.sleep(1.5)
    
    # Should be allowed again after TTL
    result2 = await lock_service.ensure_idempotent_execution(key, ttl=1)
    assert result2 is True


@pytest.mark.asyncio
async def test_check_if_processed(lock_service):
    """Test checking processing status without marking."""
    key = f"test-operation-{uuid4()}"
    
    # Initially not processed
    is_processed = await lock_service.check_if_processed(key)
    assert is_processed is False
    
    # Mark as processed
    await lock_service.ensure_idempotent_execution(key)
    
    # Now should show as processed
    is_processed = await lock_service.check_if_processed(key)
    assert is_processed is True


@pytest.mark.asyncio
async def test_force_release_lock(lock_service):
    """Test admin force-release of lock."""
    session_id = "test-session-005"
    
    # Acquire lock
    async with lock_service.acquire_conversation_lock(session_id):
        # Verify lock exists
        info = await lock_service.get_lock_info(session_id)
        assert info is not None
        
        # Force release (simulating admin action)
        released = await lock_service.force_release_lock(session_id)
        assert released is True
        
        # Lock should be gone
        info = await lock_service.get_lock_info(session_id)
        assert info is None


@pytest.mark.asyncio
async def test_get_lock_info_no_lock(lock_service):
    """Test getting info for non-existent lock."""
    info = await lock_service.get_lock_info("nonexistent-session")
    assert info is None


@pytest.mark.asyncio
async def test_health_check(lock_service):
    """Test Redis health check."""
    is_healthy = await lock_service.health_check()
    assert is_healthy is True


@pytest.mark.asyncio
async def test_multiple_sessions_independent(lock_service):
    """Test that locks for different sessions are independent."""
    session1 = "test-session-006"
    session2 = "test-session-007"
    
    # Acquire locks for both sessions simultaneously
    async with lock_service.acquire_conversation_lock(session1):
        async with lock_service.acquire_conversation_lock(session2):
            # Both should succeed
            info1 = await lock_service.get_lock_info(session1)
            info2 = await lock_service.get_lock_info(session2)
            
            assert info1 is not None
            assert info2 is not None
            assert info1["lock_id"] != info2["lock_id"]


@pytest.mark.asyncio
async def test_lock_with_exception_still_releases(lock_service):
    """Test that lock is released even if exception occurs."""
    session_id = "test-session-008"
    
    # Simulate exception within lock context
    with pytest.raises(ValueError):
        async with lock_service.acquire_conversation_lock(session_id):
            raise ValueError("Simulated error")
    
    # Lock should still be released
    info = await lock_service.get_lock_info(session_id)
    assert info is None


@pytest.mark.asyncio
async def test_realistic_conversation_scenario(lock_service):
    """Test realistic conversation processing scenario."""
    session_id = "user-session-abc123"
    turn_id = "turn-001"
    idempotency_key = f"{session_id}:{turn_id}"
    
    # Simulate incoming request
    async def process_turn():
        # Check idempotency first
        if not await lock_service.ensure_idempotent_execution(idempotency_key):
            return "DUPLICATE"
        
        # Acquire session lock for memory safety
        async with lock_service.acquire_conversation_lock(session_id):
            # Simulate processing
            await asyncio.sleep(0.1)
            return "PROCESSED"
    
    # First request should process
    result1 = await process_turn()
    assert result1 == "PROCESSED"
    
    # Duplicate request should be rejected
    result2 = await process_turn()
    assert result2 == "DUPLICATE"
