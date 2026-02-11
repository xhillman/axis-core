"""Redis-based memory adapter with TTL and namespace support.

This adapter provides persistent storage using Redis with support for
TTL expiration, namespaces, and keyword search via SCAN.

Requires: pip install axis-core[redis]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from redis.asyncio import Redis

from axis_core.errors import ConcurrencyError
from axis_core.protocols.memory import MemoryCapability, MemoryItem
from axis_core.session import SESSION_PREFIX, Session

logger = logging.getLogger("axis_core.adapters.memory.redis")

_DEFAULT_PREFIX = "axis:"
_META_SUFFIX = ":__meta__"


class RedisMemory:
    """Redis-based memory adapter.

    Uses Redis for durable key-value storage with native TTL support,
    namespace isolation via key prefixes, and keyword search via SCAN.

    Capabilities:
        - KEYWORD_SEARCH: Pattern matching via Redis SCAN
        - TTL: Native Redis key expiration
        - NAMESPACES: Key-prefix-based logical partitioning

    Example:
        >>> from redis.asyncio import Redis
        >>> client = Redis.from_url("redis://localhost:6379")
        >>> memory = RedisMemory(client=client)
        >>> await memory.store("user:123", {"name": "Alice"}, ttl=3600)
    """

    def __init__(
        self,
        client: Redis,
        key_prefix: str = _DEFAULT_PREFIX,
    ) -> None:
        """Initialize RedisMemory.

        Args:
            client: An async Redis client instance.
            key_prefix: Global prefix for all keys to avoid collisions.
        """
        self._client = client
        self._prefix = key_prefix

    @property
    def capabilities(self) -> set[MemoryCapability]:
        """Return the set of capabilities supported by this adapter."""
        return {
            MemoryCapability.KEYWORD_SEARCH,
            MemoryCapability.TTL,
            MemoryCapability.NAMESPACES,
        }

    def _full_key(self, key: str, namespace: str | None = None) -> str:
        """Build the full Redis key with prefix and optional namespace."""
        if namespace:
            return f"{self._prefix}{namespace}:{key}"
        return f"{self._prefix}{key}"

    def _meta_key(self, full_key: str) -> str:
        """Build the metadata key for a given full key."""
        return f"{full_key}{_META_SUFFIX}"

    def _strip_prefix(self, full_key: str, namespace: str | None = None) -> str:
        """Strip prefix and namespace from a full Redis key to get the user key."""
        prefix = self._prefix
        if namespace:
            prefix = f"{self._prefix}{namespace}:"
        if full_key.startswith(prefix):
            return full_key[len(prefix):]
        return full_key

    @staticmethod
    def _normalize_scan_cursor(cursor: int | bytes) -> int:
        """Normalize Redis SCAN cursor values to int.

        Redis clients may return cursor values as either int or ASCII bytes.
        We normalize here so subsequent SCAN calls always pass an int cursor.
        """
        if isinstance(cursor, bytes):
            return int(cursor.decode("ascii"))
        return cursor

    async def store(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
        namespace: str | None = None,
    ) -> None:
        """Store a value in Redis.

        Args:
            key: Unique identifier for this value
            value: Value to store (must be JSON-serializable)
            metadata: Additional metadata for search/filtering
            ttl: Time-to-live in seconds (None = no expiration)
            namespace: Namespace to store in (None = default)
        """
        full_key = self._full_key(key, namespace)
        value_json = json.dumps(value)

        if ttl is not None:
            await self._client.setex(full_key, ttl, value_json)
        else:
            await self._client.set(full_key, value_json)

        # Store metadata separately if provided
        if metadata:
            meta_key = self._meta_key(full_key)
            meta_json = json.dumps(metadata)
            if ttl is not None:
                await self._client.setex(meta_key, ttl, meta_json)
            else:
                await self._client.set(meta_key, meta_json)
        else:
            # Clear old metadata if overwriting without metadata
            meta_key = self._meta_key(full_key)
            await self._client.delete(meta_key)

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
        full_key = self._full_key(key, namespace)
        raw = await self._client.get(full_key)
        if raw is None:
            return None
        return json.loads(raw)

    async def search(
        self,
        query: str,
        limit: int = 10,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Search for items using Redis SCAN with pattern matching.

        Args:
            query: Search query (used as a glob-style substring pattern)
            limit: Maximum number of results
            namespace: Namespace to search in (None = default)
            filters: Metadata filters (currently ignored)

        Returns:
            List of matching MemoryItems
        """
        if namespace:
            pattern = f"{self._prefix}{namespace}:*{query}*"
        else:
            pattern = f"{self._prefix}*{query}*"

        results: list[MemoryItem] = []
        cursor = 0

        while True:
            raw_cursor, keys = await self._client.scan(
                cursor=cursor, match=pattern, count=100
            )
            cursor = self._normalize_scan_cursor(raw_cursor)
            for full_key in keys:
                key_str = full_key if isinstance(full_key, str) else full_key.decode()

                # Skip metadata keys
                if key_str.endswith(_META_SUFFIX):
                    continue

                raw_value = await self._client.get(key_str)
                if raw_value is None:
                    continue

                # Get metadata
                meta_raw = await self._client.get(self._meta_key(key_str))
                metadata: dict[str, Any] = {}
                if meta_raw is not None:
                    metadata = json.loads(meta_raw)

                user_key = self._strip_prefix(key_str, namespace)
                results.append(
                    MemoryItem(
                        key=user_key,
                        value=json.loads(raw_value),
                        metadata=metadata,
                        score=None,
                        namespace=namespace,
                        created_at=None,
                        expires_at=None,
                    )
                )

                if len(results) >= limit:
                    return results

            if cursor == 0:
                break

        return results

    async def delete(
        self,
        key: str,
        namespace: str | None = None,
    ) -> bool:
        """Delete a single item by key.

        Returns:
            True if the item was deleted, False if it didn't exist
        """
        full_key = self._full_key(key, namespace)
        count = await self._client.delete(full_key)
        # Also clean up metadata
        await self._client.delete(self._meta_key(full_key))
        return bool(count > 0)

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
        if namespace == "*":
            pattern = f"{self._prefix}*"
        elif namespace:
            pattern = f"{self._prefix}{namespace}:*"
        else:
            pattern = f"{self._prefix}*"

        total = 0
        cursor = 0

        while True:
            raw_cursor, keys = await self._client.scan(
                cursor=cursor, match=pattern, count=100
            )
            cursor = self._normalize_scan_cursor(raw_cursor)
            if keys:
                # Filter out metadata keys for counting (each item has value + meta)
                data_keys = [
                    k for k in keys
                    if not (k if isinstance(k, str) else k.decode()).endswith(_META_SUFFIX)
                ]
                total += len(data_keys)
                await self._client.delete(*keys)
            if cursor == 0:
                break

        return total

    # --- Session support ---

    async def store_session(self, session: Session) -> Session:
        """Store or update a session with optimistic locking."""
        existing = await self.retrieve_session(session.id)
        if existing and existing.version != session.version:
            raise ConcurrencyError(
                message=(
                    f"Session {session.id} was modified. "
                    f"Expected version {session.version}, got {existing.version}"
                ),
                expected_version=session.version,
                actual_version=existing.version,
            )
        session.version += 1
        session.updated_at = datetime.utcnow()
        key = f"{SESSION_PREFIX}{session.id}"
        await self.store(key, session.serialize(), metadata={"type": "session"})
        return session

    async def retrieve_session(self, session_id: str) -> Session | None:
        """Retrieve a session by ID."""
        key = f"{SESSION_PREFIX}{session_id}"
        value = await self.retrieve(key)
        if value is None:
            return None
        if isinstance(value, dict):
            return Session.deserialize(value)
        return None

    async def update_session(self, session: Session) -> Session:
        """Update a session with optimistic locking."""
        return await self.store_session(session)
