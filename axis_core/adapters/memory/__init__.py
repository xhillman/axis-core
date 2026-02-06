"""Memory adapters for state persistence.

This module provides built-in memory adapter implementations with lazy loading
for optional dependencies.

Available adapters:
- EphemeralMemory: In-memory dictionary storage (no dependencies)
- SQLiteMemory: SQLite-based persistent storage (requires: pip install axis-core[sqlite])
- RedisMemory: Redis-based storage with TTL support (requires: pip install axis-core[redis])
"""

from typing import Any

from axis_core.engine.registry import memory_registry
from axis_core.errors import ConfigError

__all__: list[str] = []


# ===========================================================================
# Lazy factory for Ephemeral memory (no optional deps required)
# ===========================================================================


def _make_lazy_ephemeral_factory() -> type[Any]:
    """Create a lazy-loading factory for EphemeralMemory.

    EphemeralMemory has no optional dependencies, but we use lazy loading
    for consistency and to avoid circular imports.
    """

    class LazyEphemeralFactory:
        """Lazy factory for EphemeralMemory."""

        def __init__(self, **kwargs: Any) -> None:
            from axis_core.adapters.memory.ephemeral import EphemeralMemory

            instance = EphemeralMemory(**kwargs)
            self.__dict__.update(instance.__dict__)
            self.__class__ = instance.__class__  # type: ignore[assignment]

    return LazyEphemeralFactory


# Register ephemeral memory adapter (always available)
memory_registry.register("ephemeral", _make_lazy_ephemeral_factory())


# ===========================================================================
# Lazy factory for SQLite memory adapter
# ===========================================================================


def _make_lazy_sqlite_factory() -> type[Any]:
    """Create a lazy-loading factory for SQLiteMemory."""

    class LazySQLiteFactory:
        """Lazy factory for SQLiteMemory."""

        def __init__(self, **kwargs: Any) -> None:
            try:
                from axis_core.adapters.memory.sqlite import SQLiteMemory
            except ImportError as e:
                raise ConfigError(
                    message=(
                        "Memory adapter 'sqlite' requires the aiosqlite package. "
                        "Install with: pip install 'axis-core[sqlite]'"
                    ),
                ) from e
            instance = SQLiteMemory(**kwargs)
            self.__dict__.update(instance.__dict__)
            self.__class__ = instance.__class__  # type: ignore[assignment]

    return LazySQLiteFactory


memory_registry.register("sqlite", _make_lazy_sqlite_factory())


# ===========================================================================
# Lazy factory for Redis memory adapter
# ===========================================================================


def _make_lazy_redis_factory() -> type[Any]:
    """Create a lazy-loading factory for RedisMemory."""

    class LazyRedisFactory:
        """Lazy factory for RedisMemory."""

        def __init__(self, **kwargs: Any) -> None:
            try:
                from axis_core.adapters.memory.redis import RedisMemory
            except ImportError as e:
                raise ConfigError(
                    message=(
                        "Memory adapter 'redis' requires the redis package. "
                        "Install with: pip install 'axis-core[redis]'"
                    ),
                ) from e
            instance = RedisMemory(**kwargs)
            self.__dict__.update(instance.__dict__)
            self.__class__ = instance.__class__  # type: ignore[assignment]

    return LazyRedisFactory


memory_registry.register("redis", _make_lazy_redis_factory())


# ===========================================================================
# Eager export of memory classes (for direct use)
# ===========================================================================

# Try to export the actual class for users who want to import it directly
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
