"""Tests for memory protocol and dataclasses."""

from datetime import datetime, timezone

import pytest

from axis_core.protocols.memory import MemoryAdapter, MemoryCapability, MemoryItem


class TestMemoryCapability:
    """Tests for MemoryCapability enum."""

    def test_enum_values(self):
        """Test that enum values match their lowercase names."""
        assert MemoryCapability.SEMANTIC_SEARCH.value == "semantic_search"
        assert MemoryCapability.KEYWORD_SEARCH.value == "keyword_search"
        assert MemoryCapability.TTL.value == "ttl"
        assert MemoryCapability.NAMESPACES.value == "namespaces"

    def test_enum_count(self):
        """Test that we have exactly 4 capabilities."""
        assert len(MemoryCapability) == 4

    def test_enum_membership(self):
        """Test enum membership checks."""
        assert MemoryCapability.SEMANTIC_SEARCH in MemoryCapability
        assert "invalid" not in [c.value for c in MemoryCapability]


class TestMemoryItem:
    """Tests for MemoryItem dataclass."""

    def test_minimal(self):
        """Test MemoryItem with only required fields."""
        item = MemoryItem(key="key1", value="value1")
        assert item.key == "key1"
        assert item.value == "value1"
        assert item.metadata == {}
        assert item.score is None
        assert item.namespace is None
        assert item.created_at is None
        assert item.expires_at is None

    def test_with_metadata(self):
        """Test MemoryItem with metadata."""
        item = MemoryItem(key="key1", value="value1", metadata={"type": "observation"})
        assert item.metadata == {"type": "observation"}

    def test_with_score(self):
        """Test MemoryItem with relevance score."""
        item = MemoryItem(key="key1", value="value1", score=0.95)
        assert item.score == 0.95

    def test_with_namespace(self):
        """Test MemoryItem with namespace."""
        item = MemoryItem(key="key1", value="value1", namespace="session_123")
        assert item.namespace == "session_123"

    def test_with_timestamps(self):
        """Test MemoryItem with timestamps."""
        now = datetime.now(timezone.utc)
        later = datetime.now(timezone.utc)
        item = MemoryItem(key="key1", value="value1", created_at=now, expires_at=later)
        assert item.created_at == now
        assert item.expires_at == later

    def test_with_all_fields(self):
        """Test MemoryItem with all fields populated."""
        now = datetime.now(timezone.utc)
        later = datetime.now(timezone.utc)
        item = MemoryItem(
            key="key1",
            value={"data": "test"},
            metadata={"type": "observation"},
            score=0.95,
            namespace="session_123",
            created_at=now,
            expires_at=later,
        )
        assert item.key == "key1"
        assert item.value == {"data": "test"}
        assert item.metadata == {"type": "observation"}
        assert item.score == 0.95
        assert item.namespace == "session_123"
        assert item.created_at == now
        assert item.expires_at == later

    def test_immutability(self):
        """Test that MemoryItem is immutable."""
        item = MemoryItem(key="key1", value="value1")
        with pytest.raises(AttributeError):
            item.key = "key2"  # type: ignore


class TestMemoryAdapter:
    """Tests for MemoryAdapter protocol."""

    @pytest.mark.asyncio
    async def test_protocol_implementation(self):
        """Test that a class implementing MemoryAdapter conforms to the protocol."""

        class FakeMemoryAdapter:
            @property
            def capabilities(self) -> set[MemoryCapability]:
                return {MemoryCapability.TTL, MemoryCapability.NAMESPACES}

            async def store(self, key, value, **kwargs):
                pass

            async def retrieve(self, key, namespace=None):
                return "stored_value" if key == "key1" else None

            async def search(self, query, limit=10, namespace=None, filters=None):
                return [
                    MemoryItem(key="key1", value="value1", score=0.9),
                    MemoryItem(key="key2", value="value2", score=0.8),
                ]

            async def delete(self, key, namespace=None) -> bool:
                return key == "key1"

            async def clear(self, namespace=None) -> int:
                return 5

        adapter = FakeMemoryAdapter()
        assert isinstance(adapter, MemoryAdapter)

        # Test capabilities
        caps = adapter.capabilities
        assert MemoryCapability.TTL in caps
        assert MemoryCapability.NAMESPACES in caps
        assert MemoryCapability.SEMANTIC_SEARCH not in caps

        # Test store
        await adapter.store("key1", "value1")

        # Test retrieve
        value = await adapter.retrieve("key1")
        assert value == "stored_value"

        # Test search
        results = await adapter.search("query")
        assert len(results) == 2
        assert results[0].score == 0.9

        # Test delete
        deleted = await adapter.delete("key1")
        assert deleted is True

        # Test clear
        count = await adapter.clear()
        assert count == 5

    def test_protocol_missing_methods(self):
        """Test that a class missing methods doesn't conform to protocol."""

        class IncompleteAdapter:
            @property
            def capabilities(self) -> set[MemoryCapability]:
                return set()

            async def store(self, key, value, **kwargs):
                pass

        adapter = IncompleteAdapter()
        assert not isinstance(adapter, MemoryAdapter)
