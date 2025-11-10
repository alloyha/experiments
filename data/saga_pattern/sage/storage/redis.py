"""
Redis storage implementation for saga state

Provides Redis-based persistent storage for saga state with support for
clustering, replication, and automatic expiration of completed sagas.

Requires: pip install redis
"""

import json
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from sage.storage.base import (
    SagaStorage, 
    SagaStepState, 
    SagaStorageError, 
    SagaNotFoundError,
    SagaStorageConnectionError
)
from sage.types import SagaStatus, SagaStepStatus

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


class RedisSagaStorage(SagaStorage):
    """
    Redis implementation of saga storage
    
    Uses Redis for persistent, distributed saga state storage.
    Supports automatic cleanup and clustering.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "saga:",
        default_ttl: Optional[int] = None,  # TTL in seconds for completed sagas
        **redis_kwargs
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.redis_kwargs = redis_kwargs
        self._redis = None
        self._lock = asyncio.Lock()
    
    async def _get_redis(self):
        """Get Redis connection, creating if necessary"""
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.redis_url, **self.redis_kwargs)
                # Test connection
                await self._redis.ping()
            except Exception as e:
                raise SagaStorageConnectionError(f"Failed to connect to Redis: {e}")
        
        return self._redis
    
    def _saga_key(self, saga_id: str) -> str:
        """Generate Redis key for saga"""
        return f"{self.key_prefix}{saga_id}"
    
    def _step_key(self, saga_id: str, step_name: str) -> str:
        """Generate Redis key for step"""
        return f"{self.key_prefix}{saga_id}:step:{step_name}"
    
    def _index_key(self, index_type: str) -> str:
        """Generate Redis key for indexes"""
        return f"{self.key_prefix}index:{index_type}"
    
    async def save_saga_state(
        self,
        saga_id: str,
        saga_name: str,
        status: SagaStatus,
        steps: List[Dict[str, Any]],
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save saga state to Redis"""
        
        redis_client = await self._get_redis()
        
        saga_data = {
            "saga_id": saga_id,
            "saga_name": saga_name,
            "status": status.value,
            "steps": steps,
            "context": context,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        saga_key = self._saga_key(saga_id)
        
        # Use pipeline for atomic operations
        async with redis_client.pipeline() as pipe:
            # Store saga data
            await pipe.hset(saga_key, mapping={
                "data": json.dumps(saga_data, default=str)
            })
            
            # Add to status index
            status_index = self._index_key(f"status:{status.value}")
            await pipe.sadd(status_index, saga_id)
            
            # Add to name index
            name_index = self._index_key(f"name:{saga_name}")
            await pipe.sadd(name_index, saga_id)
            
            # Set TTL for completed sagas
            if self.default_ttl and status in [SagaStatus.COMPLETED, SagaStatus.ROLLED_BACK]:
                await pipe.expire(saga_key, self.default_ttl)
            
            await pipe.execute()
    
    async def load_saga_state(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Load saga state from Redis"""
        
        redis_client = await self._get_redis()
        saga_key = self._saga_key(saga_id)
        
        saga_data_json = await redis_client.hget(saga_key, "data")
        if not saga_data_json:
            return None
        
        try:
            return json.loads(saga_data_json)
        except json.JSONDecodeError as e:
            raise SagaStorageError(f"Failed to decode saga data for {saga_id}: {e}")
    
    async def delete_saga_state(self, saga_id: str) -> bool:
        """Delete saga state from Redis"""
        
        redis_client = await self._get_redis()
        
        # First load the saga to get its status and name for index cleanup
        saga_data = await self.load_saga_state(saga_id)
        if not saga_data:
            return False
        
        saga_key = self._saga_key(saga_id)
        
        async with redis_client.pipeline() as pipe:
            # Delete main saga data
            await pipe.delete(saga_key)
            
            # Remove from indexes
            status_index = self._index_key(f"status:{saga_data['status']}")
            await pipe.srem(status_index, saga_id)
            
            name_index = self._index_key(f"name:{saga_data['saga_name']}")
            await pipe.srem(name_index, saga_id)
            
            result = await pipe.execute()
        
        return result[0] > 0  # First command (delete) returns count
    
    async def list_sagas(
        self,
        status: Optional[SagaStatus] = None,
        saga_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List sagas with filtering"""
        
        redis_client = await self._get_redis()
        saga_ids = set()
        
        # Get saga IDs from indexes
        if status:
            status_index = self._index_key(f"status:{status.value}")
            status_ids = await redis_client.smembers(status_index)
            # Decode bytes to strings
            saga_ids.update(s.decode() if isinstance(s, bytes) else s for s in status_ids)
        
        if saga_name:
            name_index = self._index_key(f"name:{saga_name}")
            name_ids = await redis_client.smembers(name_index)
            # Decode bytes to strings
            decoded_name_ids = {s.decode() if isinstance(s, bytes) else s for s in name_ids}
            if saga_ids:
                saga_ids.intersection_update(decoded_name_ids)
            else:
                saga_ids.update(decoded_name_ids)
        
        # If no filters applied, get all saga keys
        if not saga_ids and not status and not saga_name:
            pattern = f"{self.key_prefix}*"
            keys = await redis_client.keys(pattern)
            saga_ids = {key.decode().replace(self.key_prefix, "") for key in keys 
                       if ":step:" not in key.decode() and ":index:" not in key.decode()}
        
        # Convert to list and apply pagination
        saga_id_list = list(saga_ids)[offset:offset + limit]
        
        # Load saga summaries
        results = []
        for saga_id in saga_id_list:
            saga_data = await self.load_saga_state(saga_id)
            if saga_data:
                summary = {
                    "saga_id": saga_data["saga_id"],
                    "saga_name": saga_data["saga_name"],
                    "status": saga_data["status"],
                    "created_at": saga_data["created_at"],
                    "updated_at": saga_data["updated_at"],
                    "step_count": len(saga_data["steps"]),
                    "completed_steps": sum(
                        1 for step in saga_data["steps"]
                        if step.get("status") == SagaStepStatus.COMPLETED.value
                    ),
                }
                results.append(summary)
        
        # Sort by created_at (newest first)
        results.sort(key=lambda x: x["created_at"], reverse=True)
        return results
    
    async def update_step_state(
        self,
        saga_id: str,
        step_name: str,
        status: SagaStepStatus,
        result: Any = None,
        error: Optional[str] = None,
        executed_at: Optional[datetime] = None
    ) -> None:
        """Update individual step state"""
        
        # Load current saga state
        saga_data = await self.load_saga_state(saga_id)
        if not saga_data:
            raise SagaNotFoundError(f"Saga {saga_id} not found")
        
        # Update the step
        step_updated = False
        for step in saga_data["steps"]:
            if step["name"] == step_name:
                step["status"] = status.value
                step["result"] = result
                step["error"] = error
                if executed_at:
                    step["executed_at"] = executed_at.isoformat()
                step_updated = True
                break
        
        if not step_updated:
            raise SagaStorageError(f"Step {step_name} not found in saga {saga_id}")
        
        # Save updated saga state
        saga_data["updated_at"] = datetime.utcnow().isoformat()
        await self.save_saga_state(
            saga_id=saga_data["saga_id"],
            saga_name=saga_data["saga_name"],
            status=SagaStatus(saga_data["status"]),
            steps=saga_data["steps"],
            context=saga_data["context"],
            metadata=saga_data["metadata"]
        )
    
    async def get_saga_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        redis_client = await self._get_redis()
        
        # Get memory info
        memory_info = await redis_client.info("memory")
        
        # Count sagas by status
        status_counts = {}
        for status in SagaStatus:
            status_index = self._index_key(f"status:{status.value}")
            count = await redis_client.scard(status_index)
            status_counts[status.value] = count
        
        total_sagas = sum(status_counts.values())
        
        return {
            "total_sagas": total_sagas,
            "by_status": status_counts,
            "redis_memory_usage": memory_info.get("used_memory", 0),
            "redis_memory_human": memory_info.get("used_memory_human", "0B"),
        }
    
    async def cleanup_completed_sagas(
        self,
        older_than: datetime,
        statuses: Optional[List[SagaStatus]] = None
    ) -> int:
        """Clean up old completed sagas"""
        
        if statuses is None:
            statuses = [SagaStatus.COMPLETED, SagaStatus.ROLLED_BACK]
        
        redis_client = await self._get_redis()
        deleted_count = 0
        
        # Get saga IDs for each status
        for status in statuses:
            status_index = self._index_key(f"status:{status.value}")
            saga_ids = await redis_client.smembers(status_index)
            
            # Decode bytes to strings
            decoded_saga_ids = [s.decode() if isinstance(s, bytes) else s for s in saga_ids]
            
            for saga_id in decoded_saga_ids:
                saga_data = await self.load_saga_state(saga_id)
                if not saga_data:
                    continue
                
                try:
                    updated_at = datetime.fromisoformat(saga_data["updated_at"])
                    if updated_at < older_than:
                        if await self.delete_saga_state(saga_id):
                            deleted_count += 1
                except (ValueError, KeyError):
                    # Skip sagas with invalid timestamps
                    continue
        
        return deleted_count
    
    async def health_check(self) -> Dict[str, Any]:
        """Check storage health"""
        
        try:
            redis_client = await self._get_redis()
            
            # Test basic operations
            test_key = f"{self.key_prefix}health_check"
            await redis_client.set(test_key, "ok", ex=10)  # Expire in 10 seconds
            result = await redis_client.get(test_key)
            await redis_client.delete(test_key)
            
            if result != b"ok":
                raise Exception("Read/write test failed")
            
            # Get Redis info
            info = await redis_client.info()
            
            return {
                "status": "healthy",
                "storage_type": "redis",
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "timestamp": datetime.utcnow().isoformat(),
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "storage_type": "redis",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._get_redis()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._redis:
            await self._redis.aclose()