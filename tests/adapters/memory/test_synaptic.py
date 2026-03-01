"""Tests for SynapticMemory adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("synaptic_core", reason="synaptic-core not installed")

from axis_core.adapters.memory.synaptic import SynapticMemory  # noqa: E402
from axis_core.protocols.memory import MemoryCapability, MemoryItem  # noqa: E402
from axis_core.session import Session  # noqa: E402


def _make_memory(tmp_path: Path, db_name: str = "synaptic.db") -> SynapticMemory:
    db_path = tmp_path / db_name
    return SynapticMemory(db_path=str(db_path))


@pytest.mark.unit
class TestSynapticMemory:
    """Test suite for SynapticMemory adapter."""

    @pytest.mark.asyncio
    async def test_capabilities(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)

        assert memory.capabilities == {
            MemoryCapability.KEYWORD_SEARCH,
            MemoryCapability.TTL,
            MemoryCapability.NAMESPACES,
        }

    @pytest.mark.asyncio
    async def test_store_retrieve_and_search(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)

        await memory.store(
            "obs:1",
            {"value": "alpha"},
            metadata={"type": "observation", "cycle": 1},
        )
        await memory.store(
            "obs:2",
            {"value": "beta"},
            metadata={"type": "observation", "cycle": 2},
        )
        await memory.store(
            "plan:1",
            {"value": "gamma"},
            metadata={"type": "plan"},
        )

        assert await memory.retrieve("obs:1") == {"value": "alpha"}

        results = await memory.search("obs", limit=10, filters={"type": "observation"})

        assert len(results) == 2
        assert all(isinstance(item, MemoryItem) for item in results)
        assert {item.key for item in results} == {"obs:1", "obs:2"}
        for item in results:
            assert item.metadata["type"] == "observation"

    @pytest.mark.asyncio
    async def test_namespace_delete_and_clear(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)

        await memory.store("key", "default")
        await memory.store("key", "ns-value", namespace="team-a")

        assert await memory.retrieve("key") == "default"
        assert await memory.retrieve("key", namespace="team-a") == "ns-value"

        deleted = await memory.delete("key", namespace="team-a")
        assert deleted is True
        assert await memory.retrieve("key", namespace="team-a") is None

        count = await memory.clear()
        assert count == 1
        assert await memory.retrieve("key") is None

    @pytest.mark.asyncio
    async def test_session_round_trip(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)

        session = Session(id="synaptic-session")
        stored = await memory.store_session(session)

        assert stored.version == 1

        retrieved = await memory.retrieve_session("synaptic-session")
        assert retrieved is not None
        assert retrieved.id == "synaptic-session"
        assert retrieved.version == 1
