"""SQLite-based memory adapter with FTS5 keyword search.

This adapter provides persistent storage using SQLite with full-text search
capabilities via FTS5. All data survives process restarts.

Requires: pip install axis-core[sqlite]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import aiosqlite

from axis_core.errors import ConcurrencyError
from axis_core.protocols.memory import MemoryCapability, MemoryItem
from axis_core.session import SESSION_PREFIX, Session

logger = logging.getLogger("axis_core.adapters.memory.sqlite")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memory (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
USING fts5(key, content=memory, content_rowid=rowid)
"""

_CREATE_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
    INSERT INTO memory_fts(rowid, key) VALUES (new.rowid, new.key);
END;
CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, key) VALUES('delete', old.rowid, old.key);
END;
CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, key) VALUES('delete', old.rowid, old.key);
    INSERT INTO memory_fts(rowid, key) VALUES (new.rowid, new.key);
END;
"""


class SQLiteMemory:
    """SQLite-based persistent memory adapter.

    Uses SQLite for durable key-value storage with FTS5 for keyword search.
    Supports session persistence with optimistic concurrency.

    Capabilities:
        - KEYWORD_SEARCH: Full-text search via FTS5 on keys

    Not supported:
        - TTL (time-to-live expiration)
        - Namespaces (multi-tenancy)
        - Semantic search

    Example:
        >>> async with SQLiteMemory(db_path="agent.db") as memory:
        ...     await memory.store("user:123", {"name": "Alice"})
        ...     result = await memory.retrieve("user:123")
    """

    def __init__(self, db_path: str = "axis_memory.db") -> None:
        """Initialize SQLiteMemory.

        Args:
            db_path: Path to the SQLite database file. Use ":memory:" for
                     in-memory storage (useful for testing).
        """
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the database connection and create tables.

        Must be called before any other operations, or use the async context
        manager which calls this automatically.
        """
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute(_CREATE_TABLE)
        await self._db.execute(_CREATE_FTS)
        await self._db.executescript(_CREATE_FTS_TRIGGERS)
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> SQLiteMemory:
        await self.initialize()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError(
                "SQLiteMemory is not initialized. "
                "Call await memory.initialize() or use 'async with SQLiteMemory(...) as memory:'"
            )
        return self._db

    @property
    def capabilities(self) -> set[MemoryCapability]:
        """Return the set of capabilities supported by this adapter."""
        return {MemoryCapability.KEYWORD_SEARCH}

    async def store(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
        namespace: str | None = None,
    ) -> None:
        """Store a value in SQLite.

        Args:
            key: Unique identifier for this value
            value: Value to store (must be JSON-serializable)
            metadata: Additional metadata for search/filtering
            ttl: Not supported — raises ValueError
            namespace: Not supported — raises ValueError
        """
        if ttl is not None:
            raise ValueError(
                "TTL is not supported by SQLiteMemory. "
                "Use RedisMemory for TTL support."
            )
        if namespace is not None:
            raise ValueError(
                "Namespaces are not supported by SQLiteMemory. "
                "Use RedisMemory for namespace support."
            )

        db = self._conn()
        now = datetime.utcnow().isoformat()
        value_json = json.dumps(value)
        metadata_json = json.dumps(metadata or {})

        await db.execute(
            """
            INSERT INTO memory (key, value, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at
            """,
            (key, value_json, metadata_json, now, now),
        )
        await db.commit()

    async def retrieve(
        self,
        key: str,
        namespace: str | None = None,
    ) -> Any | None:
        """Retrieve a value by key.

        Args:
            key: Key to retrieve
            namespace: Not supported (must be None)

        Returns:
            The stored value, or None if not found
        """
        db = self._conn()
        async with db.execute(
            "SELECT value FROM memory WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return json.loads(row[0])

    async def search(
        self,
        query: str,
        limit: int = 10,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Search for items using FTS5 keyword matching on keys.

        Uses SQLite FTS5 for efficient full-text search on keys.
        Falls back to LIKE-based search for simple substring queries.

        Args:
            query: Search query string
            limit: Maximum number of results
            namespace: Not supported (must be None)
            filters: Metadata filters (currently ignored)

        Returns:
            List of matching MemoryItems
        """
        db = self._conn()

        # Use LIKE for simple substring matching (consistent with ephemeral)
        pattern = f"%{query}%"
        async with db.execute(
            """
            SELECT key, value, metadata, created_at
            FROM memory
            WHERE key LIKE ? COLLATE NOCASE
            LIMIT ?
            """,
            (pattern, limit),
        ) as cursor:
            rows = await cursor.fetchall()

        results: list[MemoryItem] = []
        for row in rows:
            results.append(
                MemoryItem(
                    key=row[0],
                    value=json.loads(row[1]),
                    metadata=json.loads(row[2]),
                    score=None,
                    namespace=None,
                    created_at=datetime.fromisoformat(row[3]),
                    expires_at=None,
                )
            )
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
        db = self._conn()
        cursor = await db.execute("DELETE FROM memory WHERE key = ?", (key,))
        await db.commit()
        return bool(cursor.rowcount > 0)

    async def clear(
        self,
        namespace: str | None = None,
    ) -> int:
        """Clear all items from the database.

        Returns:
            Number of items deleted
        """
        db = self._conn()
        async with db.execute("SELECT COUNT(*) FROM memory") as cursor:
            row = await cursor.fetchone()
            count = row[0] if row else 0
        await db.execute("DELETE FROM memory")
        await db.commit()
        return count

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
