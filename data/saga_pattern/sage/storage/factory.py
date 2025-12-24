"""
Storage Factory - Simplified API for creating saga storage backends

This module provides a user-friendly factory function for creating storage
backends without needing to import specific storage classes directly.
"""

from typing import Optional, Any, Dict, Union
from sage.storage.base import SagaStorage
from sage.storage.memory import InMemorySagaStorage
from sage.exceptions import MissingDependencyError


def create_storage(
    backend: str = "memory",
    *,
    # Redis options
    redis_url: str = "redis://localhost:6379",
    key_prefix: str = "saga:",
    default_ttl: Optional[int] = None,
    # PostgreSQL options
    connection_string: Optional[str] = None,
    pool_min_size: int = 5,
    pool_max_size: int = 20,
    # Additional kwargs
    **kwargs
) -> SagaStorage:
    """
    Create a saga storage backend with a simple, unified API.
    
    This factory function makes it easy to switch between storage backends
    without changing your code structure.
    
    Args:
        backend: Storage backend type - "memory", "redis", or "postgresql"
        redis_url: Redis connection URL (for redis backend)
        key_prefix: Key prefix for Redis (for redis backend)
        default_ttl: TTL for completed sagas in seconds (for redis backend)
        connection_string: PostgreSQL connection string (for postgresql backend)
        pool_min_size: Minimum pool size (for postgresql backend)
        pool_max_size: Maximum pool size (for postgresql backend)
        **kwargs: Additional backend-specific options
    
    Returns:
        Configured SagaStorage instance
    
    Raises:
        ValueError: If an unknown backend is specified
        MissingDependencyError: If required packages aren't installed
    
    Examples:
        # In-memory storage (great for development/testing)
        >>> storage = create_storage("memory")
        
        # Redis storage (for distributed systems)
        >>> storage = create_storage(
        ...     "redis",
        ...     redis_url="redis://localhost:6379",
        ...     default_ttl=3600  # Expire completed sagas after 1 hour
        ... )
        
        # PostgreSQL storage (for ACID compliance)
        >>> storage = create_storage(
        ...     "postgresql",
        ...     connection_string="postgresql://user:pass@localhost/mydb"
        ... )
    """
    backend = backend.lower().strip()
    
    if backend == "memory":
        return InMemorySagaStorage()
    
    elif backend == "redis":
        try:
            from sage.storage.redis import RedisSagaStorage
            return RedisSagaStorage(
                redis_url=redis_url,
                key_prefix=key_prefix,
                default_ttl=default_ttl,
                **kwargs
            )
        except MissingDependencyError:  # pragma: no cover
            raise
    
    elif backend in ("postgresql", "postgres", "pg"):
        if not connection_string:
            raise ValueError(
                "PostgreSQL backend requires a connection_string.\n"
                "Example: create_storage('postgresql', connection_string='postgresql://user:pass@localhost/db')"
            )
        try:
            from sage.storage.postgresql import PostgreSQLSagaStorage
            return PostgreSQLSagaStorage(
                connection_string=connection_string,
                pool_min_size=pool_min_size,
                pool_max_size=pool_max_size,
                **kwargs
            )
        except MissingDependencyError:  # pragma: no cover
            raise
    
    else:
        available_backends = ["memory", "redis", "postgresql"]
        raise ValueError(
            f"Unknown storage backend: '{backend}'\n"
            f"Available backends: {', '.join(available_backends)}"
        )


def get_available_backends() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available storage backends.
    
    Returns a dictionary with backend names as keys and their availability
    status and requirements as values.
    
    Returns:
        Dictionary of backend information
    
    Example:
        >>> backends = get_available_backends()
        >>> for name, info in backends.items():
        ...     status = "✓" if info["available"] else "✗"
        ...     print(f"{status} {name}: {info['description']}")
    """
    backends = {
        "memory": {
            "available": True,
            "description": "In-memory storage (no persistence)",
            "install": None,
            "best_for": "Development, testing, single-process applications"
        }
    }
    
    # Check Redis availability
    try:
        import redis.asyncio
        backends["redis"] = {
            "available": True,
            "description": "Redis-based distributed storage",
            "install": None,
            "best_for": "Distributed systems, high throughput, auto-expiration"
        }
    except ImportError:  # pragma: no cover
        backends["redis"] = {
            "available": False,
            "description": "Redis-based distributed storage",
            "install": "pip install redis",
            "best_for": "Distributed systems, high throughput, auto-expiration"
        }
    
    # Check PostgreSQL availability
    try:
        import asyncpg
        backends["postgresql"] = {
            "available": True,
            "description": "PostgreSQL ACID-compliant storage",
            "install": None,
            "best_for": "ACID compliance, complex queries, data integrity"
        }
    except ImportError:  # pragma: no cover
        backends["postgresql"] = {
            "available": False,
            "description": "PostgreSQL ACID-compliant storage",
            "install": "pip install asyncpg",
            "best_for": "ACID compliance, complex queries, data integrity"
        }
    
    return backends


def print_available_backends() -> None:
    """
    Print a formatted summary of available storage backends.
    
    Useful for checking which backends are available in the current environment.
    """
    backends = get_available_backends()
    
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║                    Available Storage Backends                     ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    
    for name, info in backends.items():
        status = "✓" if info["available"] else "✗"
        print(f"║  {status} {name:<12} - {info['description']:<42} ║")
        
        if not info["available"] and info["install"]:  # pragma: no cover
            print(f"║    Install: {info['install']:<52} ║")
    
    print("╚═══════════════════════════════════════════════════════════════════╝\n")
