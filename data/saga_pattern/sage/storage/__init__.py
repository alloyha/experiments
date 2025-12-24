"""
Saga storage abstractions and implementations

Provides pluggable storage backends for saga state persistence.

Quick Start:
    >>> from sage.storage import create_storage
    
    # In-memory (for development/testing)
    >>> storage = create_storage("memory")
    
    # Redis (for distributed systems)
    >>> storage = create_storage("redis", redis_url="redis://localhost:6379")
    
    # PostgreSQL (for ACID compliance)
    >>> storage = create_storage("postgresql", connection_string="postgresql://...")
"""

from .base import (
    SagaStorage,
    SagaStepState,
    SagaStorageError,
    SagaNotFoundError,
    SagaStorageConnectionError,
)
from .memory import InMemorySagaStorage
from .redis import RedisSagaStorage
from .postgresql import PostgreSQLSagaStorage
from .factory import create_storage, get_available_backends, print_available_backends

__all__ = [
    # Factory functions (recommended API)
    "create_storage",
    "get_available_backends",
    "print_available_backends",
    
    # Base classes and exceptions
    "SagaStorage",
    "SagaStepState", 
    "SagaStorageError",
    "SagaNotFoundError",
    "SagaStorageConnectionError",
    
    # Storage implementations (for direct use)
    "InMemorySagaStorage",
    "RedisSagaStorage", 
    "PostgreSQLSagaStorage",
]