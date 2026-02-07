"""Memory adapters for state persistence.

This module provides built-in memory adapter implementations with lazy loading
for optional dependencies.

Available adapters:
- EphemeralMemory: In-memory dictionary storage (no dependencies)
- SQLiteMemory: SQLite-based persistent storage (requires: pip install axis-core[sqlite])
- RedisMemory: Redis-based storage with TTL support (requires: pip install axis-core[redis])
"""

from axis_core.engine.registry import make_lazy_factory, memory_registry

__all__: list[str] = []

_MEMORY_MODULE = "axis_core.adapters.memory"

# Register built-in memory adapters
memory_registry.register(
    "ephemeral",
    make_lazy_factory(f"{_MEMORY_MODULE}.ephemeral", "EphemeralMemory"),
)

memory_registry.register(
    "sqlite",
    make_lazy_factory(
        f"{_MEMORY_MODULE}.sqlite",
        "SQLiteMemory",
        missing_dep_message=(
            "Memory adapter 'sqlite' requires the aiosqlite package. "
            "Install with: pip install 'axis-core[sqlite]'"
        ),
    ),
)

memory_registry.register(
    "redis",
    make_lazy_factory(
        f"{_MEMORY_MODULE}.redis",
        "RedisMemory",
        missing_dep_message=(
            "Memory adapter 'redis' requires the redis package. "
            "Install with: pip install 'axis-core[redis]'"
        ),
    ),
)


# ===========================================================================
# Eager export of memory classes (for direct use)
# ===========================================================================

try:
    from axis_core.adapters.memory.ephemeral import EphemeralMemory  # noqa: F401

    __all__.extend(["EphemeralMemory"])
except ImportError:
    pass

try:
    from axis_core.adapters.memory.sqlite import SQLiteMemory  # noqa: F401

    __all__.extend(["SQLiteMemory"])
except ImportError:
    pass

try:
    from axis_core.adapters.memory.redis import RedisMemory  # noqa: F401

    __all__.extend(["RedisMemory"])
except ImportError:
    pass
