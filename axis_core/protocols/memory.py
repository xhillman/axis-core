"""Memory adapter protocol and associated dataclasses.

This module defines the MemoryAdapter protocol interface for state persistence backends,
along with enums for capabilities and dataclasses for memory items.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from axis_core.session import Session


class MemoryCapability(Enum):
    """Capabilities that a memory adapter may support.

    Different memory backends provide different features. Agents can query capabilities
    at runtime to adapt their behavior.

    Attributes:
        SEMANTIC_SEARCH: Vector-based semantic similarity search
        KEYWORD_SEARCH: Text-based keyword search
        TTL: Time-to-live expiration for entries
        NAMESPACES: Logical partitioning of memory spaces
    """

    SEMANTIC_SEARCH = "semantic_search"
    KEYWORD_SEARCH = "keyword_search"
    TTL = "ttl"
    NAMESPACES = "namespaces"


@dataclass(frozen=True)
class MemoryItem:
    """A single item retrieved from memory.

    Attributes:
        key: Unique identifier for this item
        value: The stored value (can be any JSON-serializable type)
        metadata: Additional key-value metadata
        score: Relevance score for search results (None for direct retrieval)
        namespace: Namespace this item belongs to (None for default)
        created_at: When this item was created (None if not tracked)
        expires_at: When this item expires (None if no TTL)
    """

    key: str
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    namespace: str | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None


@runtime_checkable
class MemoryAdapter(Protocol):
    """Protocol for memory/state persistence adapters.

    Memory adapters provide a uniform interface to different persistence backends
    (ephemeral, SQLite, Redis, vector databases, etc.). They handle storage, retrieval,
    search, and lifecycle management of agent state.

    Implementations must provide:
    - capabilities property returning supported features
    - store() for persisting values
    - retrieve() for fetching by key
    - search() for querying by content/metadata
    - delete() for removing items
    - clear() for bulk deletion
    """

    @property
    def capabilities(self) -> set[MemoryCapability]:
        """Set of capabilities supported by this adapter."""
        ...

    async def store(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
        namespace: str | None = None,
    ) -> None:
        """Store a value in memory.

        Args:
            key: Unique identifier for this value
            value: Value to store (must be JSON-serializable)
            metadata: Additional metadata for search/filtering
            ttl: Time-to-live in seconds (None = no expiration)
            namespace: Namespace to store in (None = default)

        Raises:
            ValueError: If TTL is requested but not supported
            ValueError: If namespace is requested but not supported
        """
        ...

    async def retrieve(
        self,
        key: str,
        namespace: str | None = None,
    ) -> Any | None:
        """Retrieve a value by key.

        Args:
            key: Key to retrieve
            namespace: Namespace to retrieve from (None = default)

        Returns:
            The stored value, or None if not found or expired
        """
        ...

    async def search(
        self,
        query: str,
        limit: int = 10,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Search for items by content or metadata.

        Args:
            query: Search query (semantic or keyword depending on capabilities)
            limit: Maximum number of results to return
            namespace: Namespace to search in (None = default)
            filters: Metadata filters to apply (e.g., {"type": "observation"})

        Returns:
            List of matching MemoryItems, sorted by relevance (highest score first)

        Raises:
            NotImplementedError: If search is not supported
        """
        ...

    async def delete(
        self,
        key: str,
        namespace: str | None = None,
    ) -> bool:
        """Delete a single item by key.

        Args:
            key: Key to delete
            namespace: Namespace to delete from (None = default)

        Returns:
            True if the item was deleted, False if it didn't exist
        """
        ...

    async def clear(
        self,
        namespace: str | None = None,
    ) -> int:
        """Clear all items from a namespace.

        Args:
            namespace: Namespace to clear (None = default, "*" = all namespaces)

        Returns:
            Number of items deleted
        """
        ...


@runtime_checkable
class SessionStore(Protocol):
    """Protocol for session persistence backends."""

    async def store_session(self, session: "Session") -> "Session":
        """Store or update a session with optimistic locking."""
        ...

    async def retrieve_session(self, session_id: str) -> "Session | None":
        """Retrieve a session by ID."""
        ...

    async def update_session(self, session: "Session") -> "Session":
        """Update a session with optimistic locking."""
        ...
