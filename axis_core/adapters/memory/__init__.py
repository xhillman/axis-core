"""Memory adapters for state persistence.

This module provides built-in memory adapter implementations:

- EphemeralMemory: In-memory dictionary storage (not persistent)
- SQLiteMemory: SQLite-based persistent storage (to be implemented)
- RedisMemory: Redis-based storage with TTL support (to be implemented)
"""

from axis_core.adapters.memory.ephemeral import EphemeralMemory
from axis_core.engine.registry import memory_registry

# Register built-in memory adapters
memory_registry.register("ephemeral", EphemeralMemory)

__all__ = [
    "EphemeralMemory",
]
