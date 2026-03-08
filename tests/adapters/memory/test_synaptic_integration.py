"""Integration tests for the Synaptic adapter against the public Synaptic client."""

from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path

import pytest

try:
    synaptic_core = import_module("synaptic_core")
except Exception as exc:  # pragma: no cover - exercised only in missing/partial installs
    pytest.skip(
        f"synaptic-core package not available: {exc}",
        allow_module_level=True,
    )

from synaptic_core import Synaptic  # noqa: E402

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
if any(
    not callable(getattr(Synaptic, method_name, None))
    for method_name in _REQUIRED_PUBLIC_CLIENT_METHODS
):
    pytest.skip(
        "synaptic-core public Synaptic client contract is not available in the ambient install",
        allow_module_level=True,
    )

from axis_core.adapters.memory.synaptic import SynapticMemory  # noqa: E402
from axis_core.session import Session  # noqa: E402


def _imported_synaptic_version() -> tuple[int, int, int]:
    raw = getattr(synaptic_core, "__version__", None)
    if not isinstance(raw, str):
        pytest.skip("synaptic-core package version not exposed", allow_module_level=True)
    match = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", raw)
    if match is None:
        pytest.skip(f"Unparseable synaptic-core version: {raw}", allow_module_level=True)
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch))


_IMPORTED_VERSION = _imported_synaptic_version()
if _IMPORTED_VERSION[0:2] != (0, 3):
    pytest.skip(
        f"These integration tests run only against synaptic-core 0.3.x; got {_IMPORTED_VERSION}.",
        allow_module_level=True,
    )


def _make_memory(tmp_path: Path, db_name: str = "synaptic_integration.db") -> SynapticMemory:
    db_path = tmp_path / db_name
    return SynapticMemory(db_path=str(db_path))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_synaptic_03_adapter_round_trip(tmp_path: Path) -> None:
    client = Synaptic(db_path=str(tmp_path / "client.db"))
    assert callable(getattr(client, "set", None))
    assert callable(getattr(client, "get", None))
    assert callable(getattr(client, "find", None))
    assert callable(getattr(client, "delete", None))
    assert callable(getattr(client, "clear", None))
    assert callable(getattr(client, "store_session", None))
    assert callable(getattr(client, "retrieve_session", None))
    assert callable(getattr(client, "update_session", None))

    memory = _make_memory(tmp_path)

    await memory.store(
        "task:1",
        {"status": "open"},
        metadata={"kind": "ticket"},
        namespace="ops",
    )
    assert await memory.retrieve("task:1", namespace="ops") == {"status": "open"}

    results = await memory.search("task", namespace="ops")
    assert len(results) == 1
    assert results[0].key == "task:1"
    assert results[0].metadata.get("kind") == "ticket"

    assert await memory.delete("task:1", namespace="ops") is True
    assert await memory.retrieve("task:1", namespace="ops") is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_synaptic_03_adapter_session_round_trip(tmp_path: Path) -> None:
    memory = _make_memory(tmp_path)

    session = Session(id="syn-03-session")
    stored = await memory.store_session(session)
    assert stored.version == 1

    retrieved = await memory.retrieve_session("syn-03-session")
    assert retrieved is not None
    assert retrieved.id == "syn-03-session"
    assert retrieved.version == 1

    retrieved.metadata["phase"] = "updated"
    updated = await memory.update_session(retrieved)
    assert updated.version == 2
    assert updated.metadata["phase"] == "updated"
