"""
Outbox Storage Implementations

Provides storage backends for the transactional outbox pattern.

Available backends:
    - InMemoryOutboxStorage: For testing
    - PostgreSQLOutboxStorage: Production (requires asyncpg)
"""

from sage.outbox.storage.base import OutboxStorage, OutboxStorageError
from sage.outbox.storage.memory import InMemoryOutboxStorage


# Lazy import for optional PostgreSQL backend
def PostgreSQLOutboxStorage(*args, **kwargs):
    """PostgreSQL outbox storage (requires asyncpg)."""
    from sage.outbox.storage.postgresql import PostgreSQLOutboxStorage as _Impl
    return _Impl(*args, **kwargs)


__all__ = [
    "OutboxStorage",
    "OutboxStorageError",
    "InMemoryOutboxStorage",
    "PostgreSQLOutboxStorage",
]
