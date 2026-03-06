"""Tests for session management."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from axis_core.adapters.memory.ephemeral import EphemeralMemory
from axis_core.agent import Agent
from axis_core.errors import ConcurrencyError
from axis_core.protocols.model import ModelResponse, UsageStats
from axis_core.protocols.planner import Plan, PlanStep, StepType
from axis_core.session import Message, Session, prepare_session_for_persistence, save_session


class _FallbackMemory:
    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    async def retrieve(self, key: str) -> Any | None:
        return self._store.get(key)

    async def store(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._store[key] = value


def test_session_add_message_truncates_history() -> None:
    session = Session(id="session-1", max_history=2)

    session.add_message(Message(role="user", content="first"))
    session.add_message(Message(role="assistant", content="second"))
    session.add_message(Message(role="user", content="third"))

    assert len(session.history) == 2
    assert session.history[0].content == "second"
    assert session.history[1].content == "third"


def test_session_serialization_roundtrip() -> None:
    session = Session(id="session-2", version=3, metadata={"env": "test"})
    session.add_message(Message(role="user", content="hello"))

    data = session.serialize()
    restored = Session.deserialize(data)

    assert restored.id == session.id
    assert restored.version == session.version
    assert restored.metadata == session.metadata
    assert len(restored.history) == 1
    assert restored.history[0].role == "user"
    assert restored.history[0].content == "hello"


def test_prepare_session_for_persistence_increments_version_and_timestamp() -> None:
    original_updated_at = datetime.utcnow() - timedelta(minutes=5)
    session = Session(id="session-prepare", updated_at=original_updated_at)

    stored = prepare_session_for_persistence(session, existing=None)

    assert stored is session
    assert stored.version == 1
    assert stored.updated_at > original_updated_at


def test_prepare_session_for_persistence_conflict_raises() -> None:
    session = Session(id="session-prepare", version=1)
    existing = Session(id="session-prepare", version=2)

    with pytest.raises(ConcurrencyError) as exc_info:
        prepare_session_for_persistence(session, existing=existing)

    assert exc_info.value.expected_version == 1
    assert exc_info.value.actual_version == 2


@pytest.mark.asyncio
async def test_ephemeral_memory_session_store_and_retrieve() -> None:
    memory = EphemeralMemory()
    session = Session(id="session-3")
    session.add_message(Message(role="user", content="hello"))

    stored = await memory.store_session(session)
    restored = await memory.retrieve_session("session-3")

    assert stored.version == 1
    assert restored is not None
    assert restored.id == session.id
    assert restored.version == stored.version
    assert restored.history[0].content == "hello"


class _MockModel:
    @property
    def model_id(self) -> str:
        return "mock-model"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        return ModelResponse(
            content="ok",
            tool_calls=None,
            usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
            cost_usd=0.0,
        )


class _MockPlanner:
    async def plan(self, observation: Any, ctx: Any) -> Plan:
        return Plan(
            id="plan-1",
            goal="respond",
            steps=[
                PlanStep(
                    id="terminal",
                    type=StepType.TERMINAL,
                    payload={"output": "ok"},
                ),
            ],
        )


@pytest.mark.asyncio
async def test_agent_session_resumes_from_memory() -> None:
    memory = EphemeralMemory()
    session = Session(id="session-4")
    session.add_message(Message(role="user", content="hello"))
    await memory.store_session(session)

    agent = Agent(model=_MockModel(), planner=_MockPlanner(), memory=memory)
    loaded = await agent.session_async(id="session-4")

    assert loaded.id == "session-4"
    assert len(loaded.history) == 1
    assert loaded.history[0].content == "hello"


@pytest.mark.asyncio
async def test_session_version_conflict_raises() -> None:
    memory = EphemeralMemory()
    session = Session(id="session-5")
    await memory.store_session(session)

    stale = Session(id="session-5", version=0)
    with pytest.raises(ConcurrencyError):
        await memory.store_session(stale)


@pytest.mark.asyncio
async def test_save_session_fallback_preserves_optimistic_locking_fields() -> None:
    memory = _FallbackMemory()
    original_updated_at = datetime.utcnow() - timedelta(minutes=5)
    session = Session(id="session-fallback", updated_at=original_updated_at)

    stored = await save_session(memory, session)

    assert stored.version == 1
    assert stored.updated_at > original_updated_at

    stale = Session(id="session-fallback", version=0)
    with pytest.raises(ConcurrencyError) as exc_info:
        await save_session(memory, stale)

    assert exc_info.value.expected_version == 0
    assert exc_info.value.actual_version == 1


@pytest.mark.asyncio
async def test_session_run_persists_history() -> None:
    memory = EphemeralMemory()
    agent = Agent(model=_MockModel(), planner=_MockPlanner(), memory=memory)
    session = await agent.session_async(id="session-6")

    result = await session.run_async("Hello")

    assert result.success is True
    assert len(session.history) == 2

    stored = await memory.retrieve_session("session-6")
    assert stored is not None
    assert len(stored.history) == 2


@pytest.mark.asyncio
async def test_session_run_output_schema_enforces_validation() -> None:
    memory = EphemeralMemory()
    agent = Agent(model=_MockModel(), planner=_MockPlanner(), memory=memory)
    session = await agent.session_async(id="session-7")

    result = await session.run_async("Hello", output_schema=dict)
    assert result.success is False
    assert result.error is not None
    assert "output_schema validation failed" in result.error.message
