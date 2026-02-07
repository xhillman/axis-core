"""Tests for SQLiteMemory adapter."""

import pytest

aiosqlite = pytest.importorskip("aiosqlite", reason="aiosqlite not installed")

from axis_core.adapters.memory.sqlite import SQLiteMemory
from axis_core.errors import ConcurrencyError
from axis_core.protocols.memory import MemoryCapability, MemoryItem
from axis_core.session import Session


@pytest.mark.unit
class TestSQLiteMemory:
    """Test suite for SQLiteMemory adapter."""

    @pytest.mark.asyncio
    async def test_capabilities(self) -> None:
        """Test that SQLiteMemory declares correct capabilities."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()
        assert memory.capabilities == {MemoryCapability.KEYWORD_SEARCH}
        await memory.close()

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self) -> None:
        """Test basic store and retrieve operations."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        await memory.store("key1", "value1")
        result = await memory.retrieve("key1")

        assert result == "value1"
        await memory.close()

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent(self) -> None:
        """Test retrieving a key that doesn't exist returns None."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        result = await memory.retrieve("nonexistent")

        assert result is None
        await memory.close()

    @pytest.mark.asyncio
    async def test_store_with_metadata(self) -> None:
        """Test storing values with metadata."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        metadata = {"type": "observation", "cycle": 1}
        await memory.store("key1", "value1", metadata=metadata)

        result = await memory.retrieve("key1")
        assert result == "value1"
        await memory.close()

    @pytest.mark.asyncio
    async def test_store_overwrites(self) -> None:
        """Test that storing to the same key overwrites the value."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        await memory.store("key1", "value1")
        await memory.store("key1", "value2")

        result = await memory.retrieve("key1")
        assert result == "value2"
        await memory.close()

    @pytest.mark.asyncio
    async def test_delete_existing(self) -> None:
        """Test deleting an existing key."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        await memory.store("key1", "value1")
        result = await memory.delete("key1")

        assert result is True
        assert await memory.retrieve("key1") is None
        await memory.close()

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self) -> None:
        """Test deleting a nonexistent key returns False."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        result = await memory.delete("nonexistent")

        assert result is False
        await memory.close()

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        """Test clearing all items."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        await memory.store("key1", "value1")
        await memory.store("key2", "value2")
        await memory.store("key3", "value3")

        count = await memory.clear()

        assert count == 3
        assert await memory.retrieve("key1") is None
        assert await memory.retrieve("key2") is None
        assert await memory.retrieve("key3") is None
        await memory.close()

    @pytest.mark.asyncio
    async def test_keyword_search_basic(self) -> None:
        """Test keyword search using FTS5."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        await memory.store("user:123", {"name": "Alice"})
        await memory.store("user:456", {"name": "Bob"})
        await memory.store("post:789", {"title": "Hello"})

        results = await memory.search("user")

        assert len(results) == 2
        assert all(isinstance(item, MemoryItem) for item in results)
        assert {item.key for item in results} == {"user:123", "user:456"}
        await memory.close()

    @pytest.mark.asyncio
    async def test_keyword_search_limit(self) -> None:
        """Test that search respects limit parameter."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        await memory.store("item:1", "value1")
        await memory.store("item:2", "value2")
        await memory.store("item:3", "value3")

        results = await memory.search("item", limit=2)

        assert len(results) <= 2
        await memory.close()

    @pytest.mark.asyncio
    async def test_keyword_search_no_matches(self) -> None:
        """Test search returns empty list when no matches found."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        await memory.store("key1", "value1")

        results = await memory.search("nonexistent")

        assert results == []
        await memory.close()

    @pytest.mark.asyncio
    async def test_keyword_search_case_insensitive(self) -> None:
        """Test that keyword search is case-insensitive."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        await memory.store("User:123", "value1")
        await memory.store("USER:456", "value2")

        results = await memory.search("user")

        assert len(results) == 2
        await memory.close()

    @pytest.mark.asyncio
    async def test_ttl_not_supported(self) -> None:
        """Test that TTL parameter raises ValueError."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        with pytest.raises(ValueError, match="TTL is not supported"):
            await memory.store("key1", "value1", ttl=60)
        await memory.close()

    @pytest.mark.asyncio
    async def test_namespace_not_supported(self) -> None:
        """Test that namespace parameter raises ValueError."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        with pytest.raises(ValueError, match="Namespaces are not supported"):
            await memory.store("key1", "value1", namespace="custom")
        await memory.close()

    @pytest.mark.asyncio
    async def test_store_various_types(self) -> None:
        """Test storing different data types (JSON-serializable)."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        await memory.store("str", "hello")
        assert await memory.retrieve("str") == "hello"

        await memory.store("int", 42)
        assert await memory.retrieve("int") == 42

        await memory.store("dict", {"key": "value"})
        assert await memory.retrieve("dict") == {"key": "value"}

        await memory.store("list", [1, 2, 3])
        assert await memory.retrieve("list") == [1, 2, 3]

        await memory.store("bool", True)
        assert await memory.retrieve("bool") is True

        await memory.store("none", None)
        assert await memory.retrieve("none") is None
        await memory.close()

    @pytest.mark.asyncio
    async def test_search_returns_memory_items_with_metadata(self) -> None:
        """Test that search returns MemoryItem objects with stored metadata."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        metadata1 = {"type": "observation", "cycle": 1}
        metadata2 = {"type": "plan", "cycle": 2}

        await memory.store("obs:1", "data1", metadata=metadata1)
        await memory.store("obs:2", "data2", metadata=metadata2)

        results = await memory.search("obs")

        assert len(results) == 2
        for item in results:
            assert isinstance(item, MemoryItem)
            assert item.value in ["data1", "data2"]
            assert item.metadata in [metadata1, metadata2]
            assert item.created_at is not None  # SQLite tracks timestamps
            assert item.namespace is None
        await memory.close()

    @pytest.mark.asyncio
    async def test_session_store_and_retrieve(self) -> None:
        """Test storing and retrieving sessions."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        session = Session(id="test-session-1")
        stored = await memory.store_session(session)

        assert stored.version == 1

        retrieved = await memory.retrieve_session("test-session-1")
        assert retrieved is not None
        assert retrieved.id == "test-session-1"
        assert retrieved.version == 1
        await memory.close()

    @pytest.mark.asyncio
    async def test_session_retrieve_nonexistent(self) -> None:
        """Test retrieving nonexistent session returns None."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        result = await memory.retrieve_session("nonexistent")
        assert result is None
        await memory.close()

    @pytest.mark.asyncio
    async def test_session_version_conflict(self) -> None:
        """Test optimistic concurrency check on sessions."""
        memory = SQLiteMemory(db_path=":memory:")
        await memory.initialize()

        session = Session(id="test-session-1")
        await memory.store_session(session)

        # Simulate stale version
        stale = Session(id="test-session-1", version=0)
        with pytest.raises(ConcurrencyError):
            await memory.store_session(stale)
        await memory.close()

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager for resource cleanup."""
        async with SQLiteMemory(db_path=":memory:") as memory:
            await memory.store("key1", "value1")
            assert await memory.retrieve("key1") == "value1"
