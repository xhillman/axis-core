"""Tests for EphemeralMemory adapter."""


import pytest

from axis_core.adapters.memory.ephemeral import EphemeralMemory
from axis_core.protocols.memory import MemoryCapability, MemoryItem


@pytest.mark.unit
class TestEphemeralMemory:
    """Test suite for EphemeralMemory adapter."""

    @pytest.mark.asyncio
    async def test_capabilities(self) -> None:
        """Test that EphemeralMemory declares correct capabilities."""
        memory = EphemeralMemory()
        assert memory.capabilities == {MemoryCapability.KEYWORD_SEARCH}

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self) -> None:
        """Test basic store and retrieve operations."""
        memory = EphemeralMemory()

        await memory.store("key1", "value1")
        result = await memory.retrieve("key1")

        assert result == "value1"

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent(self) -> None:
        """Test retrieving a key that doesn't exist returns None."""
        memory = EphemeralMemory()

        result = await memory.retrieve("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_store_with_metadata(self) -> None:
        """Test storing values with metadata."""
        memory = EphemeralMemory()

        metadata = {"type": "observation", "cycle": 1}
        await memory.store("key1", "value1", metadata=metadata)

        # Retrieve should just return the value
        result = await memory.retrieve("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_store_overwrites(self) -> None:
        """Test that storing to the same key overwrites the value."""
        memory = EphemeralMemory()

        await memory.store("key1", "value1")
        await memory.store("key1", "value2")

        result = await memory.retrieve("key1")
        assert result == "value2"

    @pytest.mark.asyncio
    async def test_delete_existing(self) -> None:
        """Test deleting an existing key."""
        memory = EphemeralMemory()

        await memory.store("key1", "value1")
        result = await memory.delete("key1")

        assert result is True
        assert await memory.retrieve("key1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self) -> None:
        """Test deleting a nonexistent key returns False."""
        memory = EphemeralMemory()

        result = await memory.delete("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_clear_default_namespace(self) -> None:
        """Test clearing all items in default namespace."""
        memory = EphemeralMemory()

        await memory.store("key1", "value1")
        await memory.store("key2", "value2")
        await memory.store("key3", "value3")

        count = await memory.clear()

        assert count == 3
        assert await memory.retrieve("key1") is None
        assert await memory.retrieve("key2") is None
        assert await memory.retrieve("key3") is None

    @pytest.mark.asyncio
    async def test_keyword_search_basic(self) -> None:
        """Test basic keyword search by matching keys."""
        memory = EphemeralMemory()

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
        memory = EphemeralMemory()

        await memory.store("item:1", "value1")
        await memory.store("item:2", "value2")
        await memory.store("item:3", "value3")

        results = await memory.search("item", limit=2)

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_keyword_search_no_matches(self) -> None:
        """Test search returns empty list when no matches found."""
        memory = EphemeralMemory()

        await memory.store("key1", "value1")

        results = await memory.search("nonexistent")

        assert results == []

    @pytest.mark.asyncio
    async def test_keyword_search_case_insensitive(self) -> None:
        """Test that keyword search is case-insensitive."""
        memory = EphemeralMemory()

        await memory.store("User:123", "value1")
        await memory.store("USER:456", "value2")

        results = await memory.search("user")

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_ttl_not_supported(self) -> None:
        """Test that TTL parameter raises ValueError (not supported)."""
        memory = EphemeralMemory()

        with pytest.raises(ValueError, match="TTL is not supported"):
            await memory.store("key1", "value1", ttl=60)

    @pytest.mark.asyncio
    async def test_namespace_not_supported(self) -> None:
        """Test that namespace parameter raises ValueError (not supported)."""
        memory = EphemeralMemory()

        with pytest.raises(ValueError, match="Namespaces are not supported"):
            await memory.store("key1", "value1", namespace="custom")

    @pytest.mark.asyncio
    async def test_store_various_types(self) -> None:
        """Test storing different data types."""
        memory = EphemeralMemory()

        # String
        await memory.store("str", "hello")
        assert await memory.retrieve("str") == "hello"

        # Integer
        await memory.store("int", 42)
        assert await memory.retrieve("int") == 42

        # Dict
        await memory.store("dict", {"key": "value"})
        assert await memory.retrieve("dict") == {"key": "value"}

        # List
        await memory.store("list", [1, 2, 3])
        assert await memory.retrieve("list") == [1, 2, 3]

        # Boolean
        await memory.store("bool", True)
        assert await memory.retrieve("bool") is True

        # None
        await memory.store("none", None)
        assert await memory.retrieve("none") is None

    @pytest.mark.asyncio
    async def test_search_returns_memory_items_with_metadata(self) -> None:
        """Test that search returns MemoryItem objects with stored metadata."""
        memory = EphemeralMemory()

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
            assert item.created_at is None  # Ephemeral doesn't track timestamps
            assert item.expires_at is None
            assert item.namespace is None
            assert item.score is None  # Keyword search doesn't provide scores
