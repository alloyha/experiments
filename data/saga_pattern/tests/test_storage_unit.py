"""
Unit tests for storage backends without requiring Docker infrastructure.
These tests use mocking to verify the storage logic without actual database connections.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sage.types import SagaStatus
from sage.storage.base import SagaStorageConnectionError, SagaNotFoundError


class TestRedisSagaStorageUnit:
    """Unit tests for RedisSagaStorage without actual Redis"""
    
    @pytest.mark.asyncio
    async def test_redis_initialization(self):
        """Test Redis storage initialization"""
        from sage.storage.redis import RedisSagaStorage
        
        storage = RedisSagaStorage(
            redis_url="redis://localhost:6379",
            key_prefix="test:",
            default_ttl=3600
        )
        
        assert storage.redis_url == "redis://localhost:6379"
        assert storage.key_prefix == "test:"
        assert storage.default_ttl == 3600
    
    @pytest.mark.asyncio
    async def test_redis_key_generation(self):
        """Test Redis key generation methods"""
        from sage.storage.redis import RedisSagaStorage
        
        storage = RedisSagaStorage(key_prefix="saga:")
        
        assert storage._saga_key("test-123") == "saga:test-123"
        assert storage._step_key("test-123", "step1") == "saga:test-123:step:step1"
        assert storage._index_key("status") == "saga:index:status"
    
    @pytest.mark.asyncio
    async def test_redis_connection_error_handling(self):
        """Test Redis connection error handling"""
        from sage.storage.redis import RedisSagaStorage
        
        storage = RedisSagaStorage(redis_url="redis://invalid:9999")
        
        # Mock redis.from_url to raise an exception
        with patch('sage.storage.redis.redis.from_url', side_effect=Exception("Connection refused")):
            with pytest.raises(SagaStorageConnectionError, match="Failed to connect to Redis"):
                await storage._get_redis()


class TestPostgreSQLSagaStorageUnit:
    """Unit tests for PostgreSQLSagaStorage without actual PostgreSQL"""
    
    @pytest.mark.asyncio
    async def test_postgresql_initialization(self):
        """Test PostgreSQL storage initialization"""
        from sage.storage.postgresql import PostgreSQLSagaStorage
        
        storage = PostgreSQLSagaStorage(
            connection_string="postgresql://localhost/test"
        )
        
        assert storage.connection_string == "postgresql://localhost/test"
    
    @pytest.mark.asyncio
    async def test_postgresql_connection_error_handling(self):
        """Test PostgreSQL connection error handling"""
        from sage.storage.postgresql import PostgreSQLSagaStorage
        
        storage = PostgreSQLSagaStorage(
            connection_string="postgresql://invalid:9999/test"
        )
        
        # Mock asyncpg.connect to raise an exception
        with patch('sage.storage.postgresql.asyncpg.create_pool', side_effect=Exception("Connection refused")):
            with pytest.raises(SagaStorageConnectionError, match="Failed to connect to PostgreSQL"):
                await storage._get_pool()
