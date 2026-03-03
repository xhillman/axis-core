"""Tests for SynapticMemory adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("synaptic_core", reason="synaptic-core not installed")

from axis_core.adapters.memory import synaptic as synaptic_adapter  # noqa: E402
from axis_core.adapters.memory.synaptic import SynapticMemory  # noqa: E402
from axis_core.errors import ConfigError  # noqa: E402
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
    async def test_search_returns_empty_for_non_positive_limit(self, tmp_path: Path) -> None:
        memory = _make_memory(tmp_path)
        await memory.store("obs:1", {"value": "alpha"})

        assert await memory.search("obs", limit=0) == []
        assert await memory.search("obs", limit=-5) == []

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

    def test_rejects_unsupported_provider_version(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            synaptic_adapter,
            "_load_synaptic_core_version",
            lambda: "0.1.1",
        )

        with pytest.raises(ConfigError, match="Unsupported synaptic-core version"):
            _make_memory(tmp_path)

    def test_rejects_provider_missing_required_methods(self, tmp_path: Path) -> None:
        class _IncompleteProvider:
            async def kv_set(self, *args: Any, **kwargs: Any) -> None: ...

            async def kv_get(self, *args: Any, **kwargs: Any) -> Any | None:
                return None

            async def kv_delete(self, *args: Any, **kwargs: Any) -> bool:
                return False

            async def kv_clear(self, *args: Any, **kwargs: Any) -> int:
                return 0

            async def store_session(self, session: Any) -> Any:
                return session

            async def retrieve_session(self, session_id: str) -> Any | None:
                return None

            async def update_session(self, session: Any) -> Any:
                return session

        with pytest.raises(
            ConfigError,
            match="missing required methods",
        ):
            SynapticMemory(
                db_path=str(tmp_path / "bad.db"),
                synaptic_memory=cast(Any, _IncompleteProvider()),
            )

    def test_initializes_without_legacy_axis_module(self, tmp_path: Path) -> None:
        assert importlib.util.find_spec("synaptic_core.axis") is None
        memory = _make_memory(tmp_path)
        assert isinstance(memory, SynapticMemory)
