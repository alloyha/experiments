"""
In-memory storage implementation for saga state

Provides a simple in-memory storage backend for development and testing.
Not suitable for production use as state is lost on process restart.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from sage.storage.base import SagaStorage, SagaStepState, SagaStorageError
from sage.types import SagaStatus, SagaStepStatus


class InMemorySagaStorage(SagaStorage):
    """
    In-memory implementation of saga storage
    
    Stores all saga state in memory using dictionaries.
    Provides fast access but no persistence across restarts.
    """
    
    def __init__(self):
        self._sagas: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def save_saga_state(
        self,
        saga_id: str,
        saga_name: str,
        status: SagaStatus,
        steps: List[Dict[str, Any]],
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save saga state to memory"""
        
        async with self._lock:
            self._sagas[saga_id] = {
                "saga_id": saga_id,
                "saga_name": saga_name,
                "status": status.value,
                "steps": steps,
                "context": context,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
    
    async def load_saga_state(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Load saga state from memory"""
        
        async with self._lock:
            saga_data = self._sagas.get(saga_id)
            if saga_data:
                # Return a copy to prevent external modification
                return dict(saga_data)
            return None
    
    async def delete_saga_state(self, saga_id: str) -> bool:
        """Delete saga state from memory"""
        
        async with self._lock:
            if saga_id in self._sagas:
                del self._sagas[saga_id]
                return True
            return False
    
    async def list_sagas(
        self,
        status: Optional[SagaStatus] = None,
        saga_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List sagas with filtering"""
        
        async with self._lock:
            results = []
            
            for saga_data in self._sagas.values():
                # Apply status filter
                if status and saga_data["status"] != status.value:
                    continue
                
                # Apply name filter (simple substring match)
                if saga_name and saga_name.lower() not in saga_data["saga_name"].lower():
                    continue
                
                # Create summary (exclude full context and steps for performance)
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
            
            # Apply pagination
            return results[offset:offset + limit]
    
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
        
        async with self._lock:
            saga_data = self._sagas.get(saga_id)
            if not saga_data:
                raise SagaStorageError(f"Saga {saga_id} not found")
            
            # Find and update the step
            for step in saga_data["steps"]:
                if step["name"] == step_name:
                    step["status"] = status.value
                    step["result"] = result
                    step["error"] = error
                    if executed_at:
                        step["executed_at"] = executed_at.isoformat()
                    break
            else:
                # Step not found, this shouldn't happen
                raise SagaStorageError(f"Step {step_name} not found in saga {saga_id}")
            
            # Update saga timestamp
            saga_data["updated_at"] = datetime.utcnow().isoformat()
    
    async def get_saga_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        async with self._lock:
            stats = {
                "total_sagas": len(self._sagas),
                "by_status": {},
                "memory_usage_bytes": self._estimate_memory_usage(),
            }
            
            # Count by status
            for saga_data in self._sagas.values():
                status = saga_data["status"]
                stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            
            return stats
    
    async def cleanup_completed_sagas(
        self,
        older_than: datetime,
        statuses: Optional[List[SagaStatus]] = None
    ) -> int:
        """Clean up old completed sagas"""
        
        if statuses is None:
            statuses = [SagaStatus.COMPLETED, SagaStatus.ROLLED_BACK]
        
        status_values = [s.value for s in statuses]
        deleted_count = 0
        
        async with self._lock:
            to_delete = []
            
            for saga_id, saga_data in self._sagas.items():
                # Check if saga matches cleanup criteria
                if saga_data["status"] not in status_values:
                    continue
                
                # Parse updated_at timestamp
                try:
                    updated_at = datetime.fromisoformat(saga_data["updated_at"])
                    if updated_at < older_than:
                        to_delete.append(saga_id)
                except (ValueError, KeyError):
                    # Skip sagas with invalid timestamps
                    continue
            
            # Delete matching sagas
            for saga_id in to_delete:
                del self._sagas[saga_id]
                deleted_count += 1
        
        return deleted_count
    
    async def health_check(self) -> Dict[str, Any]:
        """Check storage health"""
        
        async with self._lock:
            return {
                "status": "healthy",
                "storage_type": "in_memory",
                "total_sagas": len(self._sagas),
                "memory_usage_bytes": self._estimate_memory_usage(),
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    def _estimate_memory_usage(self) -> int:
        """
        Rough estimate of memory usage
        
        This is a very approximate calculation for monitoring purposes.
        """
        import sys
        
        total_size = 0
        for saga_data in self._sagas.values():
            total_size += sys.getsizeof(saga_data)
            total_size += sum(sys.getsizeof(v) for v in saga_data.values())
        
        return total_size
    
    async def clear_all(self) -> int:
        """
        Clear all saga data (for testing purposes)
        
        Returns:
            Number of sagas deleted
        """
        async with self._lock:
            count = len(self._sagas)
            self._sagas.clear()
            return count
    
    def get_saga_count(self) -> int:
        """Get current saga count (synchronous for testing)"""
        return len(self._sagas)