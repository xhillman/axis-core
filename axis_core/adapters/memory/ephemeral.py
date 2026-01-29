"""Ephemeral (in-memory) memory adapter.

This adapter provides a simple in-memory dictionary-based storage implementation.
All data is lost when the process terminates.
"""

from typing import Any

from axis_core.protocols.memory import MemoryCapability, MemoryItem


class EphemeralMemory:
    """In-memory dictionary-based memory adapter.

    This adapter stores all data in a Python dictionary. Data persists only
    for the lifetime of the process and is not shared across instances.

    Capabilities:
        - KEYWORD_SEARCH: Simple pattern matching on keys (case-insensitive)

    Not supported:
        - TTL (time-to-live expiration)
        - Namespaces (multi-tenancy)
        - Semantic search

    Example:
        >>> memory = EphemeralMemory()
        >>> await memory.store("user:123", {"name": "Alice"})
        >>> result = await memory.retrieve("user:123")
        >>> print(result)
        {"name": "Alice"}
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory store."""
        self._store: dict[str, dict[str, Any]] = {}

    @property
    def capabilities(self) -> set[MemoryCapability]:
        """Return the set of capabilities supported by this adapter.

        Returns:
            Set containing only KEYWORD_SEARCH capability
        """
        return {MemoryCapability.KEYWORD_SEARCH}

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
            value: Value to store (can be any Python object)
            metadata: Additional metadata for search/filtering
            ttl: Time-to-live in seconds (not supported)
            namespace: Namespace to store in (not supported)

        Raises:
            ValueError: If TTL or namespace is specified (not supported)
        """
        if ttl is not None:
            raise ValueError(
                "TTL is not supported by EphemeralMemory. "
                "Use a different memory adapter (e.g., RedisMemory) for TTL support."
            )

        if namespace is not None:
            raise ValueError(
                "Namespaces are not supported by EphemeralMemory. "
                "Use a different memory adapter (e.g., RedisMemory) for namespace support."
            )

        self._store[key] = {
            "value": value,
            "metadata": metadata or {},
        }

    async def retrieve(
        self,
        key: str,
        namespace: str | None = None,
    ) -> Any | None:
        """Retrieve a value by key.

        Args:
            key: Key to retrieve
            namespace: Namespace to retrieve from (not supported, must be None)

        Returns:
            The stored value, or None if not found
        """
        item = self._store.get(key)
        if item is None:
            return None
        return item["value"]

    async def search(
        self,
        query: str,
        limit: int = 10,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Search for items by keyword matching on keys.

        Performs case-insensitive substring matching on keys.

        Args:
            query: Search query string (matched against keys)
            limit: Maximum number of results to return
            namespace: Namespace to search in (not supported, must be None)
            filters: Metadata filters to apply (currently ignored)

        Returns:
            List of matching MemoryItems (no particular order, no scores)
        """
        query_lower = query.lower()
        results: list[MemoryItem] = []

        for key, item in self._store.items():
            if query_lower in key.lower():
                results.append(
                    MemoryItem(
                        key=key,
                        value=item["value"],
                        metadata=item["metadata"],
                        score=None,  # Keyword search doesn't provide relevance scores
                        namespace=None,
                        created_at=None,
                        expires_at=None,
                    )
                )

                if len(results) >= limit:
                    break

        return results

    async def delete(
        self,
        key: str,
        namespace: str | None = None,
    ) -> bool:
        """Delete a single item by key.

        Args:
            key: Key to delete
            namespace: Namespace to delete from (not supported, must be None)

        Returns:
            True if the item was deleted, False if it didn't exist
        """
        if key in self._store:
            del self._store[key]
            return True
        return False

    async def clear(
        self,
        namespace: str | None = None,
    ) -> int:
        """Clear all items from memory.

        Args:
            namespace: Namespace to clear (not supported, must be None)

        Returns:
            Number of items deleted
        """
        count = len(self._store)
        self._store.clear()
        return count
