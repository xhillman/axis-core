"""Tests for RedisMemory adapter."""

import pytest

redis_pkg = pytest.importorskip("redis", reason="redis not installed")
fakeredis = pytest.importorskip("fakeredis", reason="fakeredis not installed")

from axis_core.adapters.memory.redis import RedisMemory  # noqa: E402
from axis_core.errors import ConcurrencyError  # noqa: E402
from axis_core.protocols.memory import MemoryCapability, MemoryItem  # noqa: E402
from axis_core.session import Session  # noqa: E402


def _make_memory(**kwargs: object) -> RedisMemory:
    """Create a RedisMemory backed by fakeredis."""
    import fakeredis.aioredis

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisMemory(client=client, **kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
class TestRedisMemory:
    """Test suite for RedisMemory adapter."""

    @pytest.mark.asyncio
    async def test_capabilities(self) -> None:
        """Test that RedisMemory declares correct capabilities."""
        memory = _make_memory()
        assert memory.capabilities == {
            MemoryCapability.KEYWORD_SEARCH,
            MemoryCapability.TTL,
            MemoryCapability.NAMESPACES,
        }

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self) -> None:
        """Test basic store and retrieve operations."""
        memory = _make_memory()

        await memory.store("key1", "value1")
        result = await memory.retrieve("key1")

        assert result == "value1"

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent(self) -> None:
        """Test retrieving a key that doesn't exist returns None."""
        memory = _make_memory()

        result = await memory.retrieve("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_store_with_metadata(self) -> None:
        """Test storing values with metadata."""
        memory = _make_memory()

        metadata = {"type": "observation", "cycle": 1}
        await memory.store("key1", "value1", metadata=metadata)

        result = await memory.retrieve("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_store_overwrites(self) -> None:
        """Test that storing to the same key overwrites the value."""
        memory = _make_memory()

        await memory.store("key1", "value1")
        await memory.store("key1", "value2")

        result = await memory.retrieve("key1")
        assert result == "value2"

    @pytest.mark.asyncio
    async def test_delete_existing(self) -> None:
        """Test deleting an existing key."""
        memory = _make_memory()

        await memory.store("key1", "value1")
        result = await memory.delete("key1")

        assert result is True
        assert await memory.retrieve("key1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self) -> None:
        """Test deleting a nonexistent key returns False."""
        memory = _make_memory()

        result = await memory.delete("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_clear_default_namespace(self) -> None:
        """Test clearing all items in default namespace."""
        memory = _make_memory()

        await memory.store("key1", "value1")
        await memory.store("key2", "value2")
        await memory.store("key3", "value3")

        count = await memory.clear()

        assert count == 3
        assert await memory.retrieve("key1") is None

    @pytest.mark.asyncio
    async def test_keyword_search_basic(self) -> None:
        """Test keyword search using SCAN pattern matching."""
        memory = _make_memory()

        await memory.store("user:123", {"name": "Alice"})
        await memory.store("user:456", {"name": "Bob"})
        await memory.store("post:789", {"title": "Hello"})

        results = await memory.search("user")

        assert len(results) == 2
        assert all(isinstance(item, MemoryItem) for item in results)
        assert {item.key for item in results} == {"user:123", "user:456"}

    @pytest.mark.asyncio
    async def test_keyword_search_limit(self) -> None:
        """Test that search respects limit parameter."""
        memory = _make_memory()

        await memory.store("item:1", "value1")
        await memory.store("item:2", "value2")
        await memory.store("item:3", "value3")

        results = await memory.search("item", limit=2)

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_keyword_search_no_matches(self) -> None:
        """Test search returns empty list when no matches found."""
        memory = _make_memory()

        await memory.store("key1", "value1")

        results = await memory.search("nonexistent")

        assert results == []

    @pytest.mark.asyncio
    async def test_ttl_support(self) -> None:
        """Test that TTL is accepted (does not raise)."""
        memory = _make_memory()

        # Should not raise
        await memory.store("key1", "value1", ttl=60)

        result = await memory.retrieve("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_namespace_store_and_retrieve(self) -> None:
        """Test storing and retrieving with namespaces."""
        memory = _make_memory()

        await memory.store("key1", "value_default")
        await memory.store("key1", "value_ns1", namespace="ns1")
        await memory.store("key1", "value_ns2", namespace="ns2")

        assert await memory.retrieve("key1") == "value_default"
        assert await memory.retrieve("key1", namespace="ns1") == "value_ns1"
        assert await memory.retrieve("key1", namespace="ns2") == "value_ns2"

    @pytest.mark.asyncio
    async def test_namespace_delete(self) -> None:
        """Test deleting from a specific namespace."""
        memory = _make_memory()

        await memory.store("key1", "default")
        await memory.store("key1", "ns_value", namespace="ns1")

        deleted = await memory.delete("key1", namespace="ns1")
        assert deleted is True
        assert await memory.retrieve("key1", namespace="ns1") is None
        # Default namespace unaffected
        assert await memory.retrieve("key1") == "default"

    @pytest.mark.asyncio
    async def test_namespace_clear(self) -> None:
        """Test clearing a specific namespace."""
        memory = _make_memory()

        await memory.store("a", "1", namespace="ns1")
        await memory.store("b", "2", namespace="ns1")
        await memory.store("c", "3", namespace="ns2")

        count = await memory.clear(namespace="ns1")
        assert count == 2
        assert await memory.retrieve("a", namespace="ns1") is None
        # Other namespace unaffected
        assert await memory.retrieve("c", namespace="ns2") == "3"

    @pytest.mark.asyncio
    async def test_clear_all_namespaces(self) -> None:
        """Test clearing all namespaces with wildcard."""
        memory = _make_memory()

        await memory.store("a", "1")
        await memory.store("b", "2", namespace="ns1")
        await memory.store("c", "3", namespace="ns2")

        count = await memory.clear(namespace="*")
        assert count == 3

    @pytest.mark.asyncio
    async def test_namespace_search(self) -> None:
        """Test search within a specific namespace."""
        memory = _make_memory()

        await memory.store("user:1", "alice", namespace="ns1")
        await memory.store("user:2", "bob", namespace="ns2")

        results = await memory.search("user", namespace="ns1")
        assert len(results) == 1
        assert results[0].key == "user:1"

    @pytest.mark.asyncio
    async def test_store_various_types(self) -> None:
        """Test storing different data types (JSON-serializable)."""
        memory = _make_memory()

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

    @pytest.mark.asyncio
    async def test_session_store_and_retrieve(self) -> None:
        """Test storing and retrieving sessions."""
        memory = _make_memory()

        session = Session(id="test-session-1")
        stored = await memory.store_session(session)

        assert stored.version == 1

        retrieved = await memory.retrieve_session("test-session-1")
        assert retrieved is not None
        assert retrieved.id == "test-session-1"
        assert retrieved.version == 1

    @pytest.mark.asyncio
    async def test_session_retrieve_nonexistent(self) -> None:
        """Test retrieving nonexistent session returns None."""
        memory = _make_memory()

        result = await memory.retrieve_session("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_session_version_conflict(self) -> None:
        """Test optimistic concurrency check on sessions."""
        memory = _make_memory()

        session = Session(id="test-session-1")
        await memory.store_session(session)

        stale = Session(id="test-session-1", version=0)
        with pytest.raises(ConcurrencyError):
            await memory.store_session(stale)

    @pytest.mark.asyncio
    async def test_search_returns_memory_items(self) -> None:
        """Test that search returns MemoryItem objects with metadata."""
        memory = _make_memory()

        metadata = {"type": "observation"}
        await memory.store("obs:1", "data1", metadata=metadata)

        results = await memory.search("obs")

        assert len(results) == 1
        item = results[0]
        assert isinstance(item, MemoryItem)
        assert item.key == "obs:1"
        assert item.value == "data1"
        assert item.metadata == metadata
