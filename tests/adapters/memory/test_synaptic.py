"""Tests for SynapticMemory adapter."""

from __future__ import annotations

import importlib.util
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any, cast

import pytest

try:
    synaptic_api = import_module("synaptic_core.api")
except Exception as exc:  # pragma: no cover - exercised only in missing/partial installs
    pytest.skip(
        f"synaptic-core API module not available: {exc}",
        allow_module_level=True,
    )

_REQUIRED_PUBLIC_CLIENT_METHODS = (
    "set",
    "get",
    "find",
    "delete",
    "clear",
    "store_session",
    "retrieve_session",
    "update_session",
)
_PUBLIC_CLIENT = getattr(synaptic_api, "Synaptic", None)
if _PUBLIC_CLIENT is None or any(
    not callable(getattr(_PUBLIC_CLIENT, method_name, None))
    for method_name in _REQUIRED_PUBLIC_CLIENT_METHODS
):
    pytest.skip(
        "synaptic-core public Synaptic client contract is not available in the ambient install",
        allow_module_level=True,
    )

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
    async def test_uses_canonical_client_methods(self, tmp_path: Path) -> None:
        class _RecordingClient:
            def __init__(self) -> None:
                self.calls: list[str] = []

            async def set(self, *args: Any, **kwargs: Any) -> None:
                del args, kwargs
                self.calls.append("set")

            async def get(self, *args: Any, **kwargs: Any) -> Any | None:
                del args, kwargs
                self.calls.append("get")
                return {"value": "alpha"}

            async def find(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
                del args, kwargs
                self.calls.append("find")
                return [
                    {
                        "key": "obs:1",
                        "value": {"value": "alpha"},
                        "metadata": {"type": "observation"},
                        "score": 0.7,
                        "namespace": None,
                    }
                ]

            async def delete(self, *args: Any, **kwargs: Any) -> bool:
                del args, kwargs
                self.calls.append("delete")
                return True

            async def clear(self, *args: Any, **kwargs: Any) -> int:
                del args, kwargs
                self.calls.append("clear")
                return 1

            async def store_session(self, session: Any) -> Any:
                self.calls.append("store_session")
                return session

            async def retrieve_session(self, session_id: str) -> Any | None:
                self.calls.append("retrieve_session")
                return {"id": session_id, "version": 1, "metadata": {}, "state": {}}

            async def update_session(self, session: Any) -> Any:
                self.calls.append("update_session")
                return session

        memory = SynapticMemory(
            db_path=str(tmp_path / "canon.db"),
            synaptic_client=cast(Any, _RecordingClient()),
        )

        await memory.store("obs:1", {"value": "alpha"})
        assert await memory.retrieve("obs:1") == {"value": "alpha"}
        results = await memory.search("obs", limit=5)
        assert await memory.delete("obs:1") is True
        assert await memory.clear() == 1
        stored = await memory.store_session(Session(id="canonical-session"))
        retrieved = await memory.retrieve_session("canonical-session")
        updated = await memory.update_session(stored)

        assert [item.key for item in results] == ["obs:1"]
        assert stored.id == "canonical-session"
        assert retrieved is not None
        assert retrieved.id == "canonical-session"
        assert updated.id == "canonical-session"
        assert cast(Any, memory)._client.calls == [
            "set",
            "get",
            "find",
            "delete",
            "clear",
            "store_session",
            "retrieve_session",
            "update_session",
        ]

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
            lambda: "0.2.9",
        )

        with pytest.raises(ConfigError, match="Unsupported synaptic-core version"):
            _make_memory(tmp_path)

    def test_uses_imported_package_version_when_metadata_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise_package_not_found(_: str) -> str:
            raise PackageNotFoundError

        monkeypatch.setattr(synaptic_adapter, "package_version", _raise_package_not_found)

        memory = _make_memory(tmp_path)

        assert isinstance(memory, SynapticMemory)

    def test_rejects_provider_missing_required_methods(self, tmp_path: Path) -> None:
        class _IncompleteProvider:
            async def set(self, *args: Any, **kwargs: Any) -> None: ...

            async def get(self, *args: Any, **kwargs: Any) -> Any | None:
                return None

            async def find(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
                return []

        with pytest.raises(
            ConfigError,
            match="missing required methods",
        ):
            SynapticMemory(
                db_path=str(tmp_path / "bad.db"),
                synaptic_client=cast(Any, _IncompleteProvider()),
            )

    def test_initializes_without_legacy_axis_module(self, tmp_path: Path) -> None:
        assert importlib.util.find_spec("synaptic_core.axis") is None
        memory = _make_memory(tmp_path)
        assert isinstance(memory, SynapticMemory)
