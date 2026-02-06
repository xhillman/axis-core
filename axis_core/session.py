"""Session and message types for multi-turn conversations.

Provides session history tracking, optimistic versioning, and
serialization helpers for persistence.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from axis_core.attachments import AttachmentLike
from axis_core.cancel import CancelToken
from axis_core.errors import ConcurrencyError

if TYPE_CHECKING:
    from axis_core.agent import Agent

logger = logging.getLogger("axis_core.session")

SESSION_PREFIX = "session:"
SESSION_CONTEXT_KEY = "__session_history__"


def generate_session_id() -> str:
    """Generate a new session identifier."""
    return str(uuid.uuid4())


@dataclass(frozen=True)
class ContentPart:
    """Represents a multimodal content part in a message."""

    type: str
    data: str | bytes
    mime_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize ContentPart to a dict."""
        if isinstance(self.data, bytes):
            encoded = base64.b64encode(self.data).decode("ascii")
            return {
                "type": self.type,
                "data": encoded,
                "mime_type": self.mime_type,
                "encoding": "base64",
            }
        return {
            "type": self.type,
            "data": self.data,
            "mime_type": self.mime_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContentPart:
        """Deserialize ContentPart from a dict."""
        payload = data.get("data")
        if data.get("encoding") == "base64" and isinstance(payload, str):
            payload = base64.b64decode(payload.encode("ascii"))
        if payload is None:
            payload = ""
        if not isinstance(payload, (str, bytes)):
            payload = str(payload)
        return cls(
            type=data.get("type", "text"),
            data=payload,
            mime_type=data.get("mime_type"),
        )


@dataclass(frozen=True)
class Message:
    """A single message in a session history."""

    role: str
    content: str | list[ContentPart]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    run_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize Message to a dict."""
        if isinstance(self.content, list):
            content: Any = [part.to_dict() for part in self.content]
        else:
            content = self.content

        return {
            "role": self.role,
            "content": content,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "tool_calls": self.tool_calls,
        }

    def to_model_dict(self) -> dict[str, Any]:
        """Convert message to model-facing dict (role/content/tool_calls)."""
        if isinstance(self.content, list):
            content: Any = [part.to_dict() for part in self.content]
        else:
            content = self.content
        data: dict[str, Any] = {
            "role": self.role,
            "content": content,
        }
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Deserialize Message from a dict."""
        content = data.get("content", "")
        if isinstance(content, list):
            content = [ContentPart.from_dict(part) for part in content]
        return cls(
            role=data.get("role", "user"),
            content=content,
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.utcnow(),
            run_id=data.get("run_id"),
            tool_calls=data.get("tool_calls"),
        )


@dataclass
class Session:
    """Represents a multi-turn session with history and versioning."""

    id: str
    version: int = 0
    history: list[Message] = field(default_factory=list)
    max_history: int = 100
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    agent_id: str | None = None
    config_fingerprint: str | None = None
    _agent: Agent | None = field(default=None, init=False, repr=False, compare=False)
    _memory: Any | None = field(default=None, init=False, repr=False, compare=False)

    def add_message(self, message: Message) -> None:
        """Add a message to the history with truncation (AD-026)."""
        self.history.append(message)
        self.updated_at = datetime.utcnow()

        if len(self.history) > self.max_history:
            removed = len(self.history) - self.max_history
            self.history = self.history[removed:]
            logger.info(
                "Session %s history truncated: removed %d oldest messages",
                self.id,
                removed,
            )

    def serialize(self) -> dict[str, Any]:
        """Serialize Session to a dict."""
        return {
            "id": self.id,
            "version": self.version,
            "history": [msg.to_dict() for msg in self.history],
            "max_history": self.max_history,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "agent_id": self.agent_id,
            "config_fingerprint": self.config_fingerprint,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> Session:
        """Deserialize Session from a dict."""
        history = [Message.from_dict(item) for item in data.get("history", [])]
        return cls(
            id=data["id"],
            version=data.get("version", 0),
            history=history,
            max_history=data.get("max_history", 100),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else datetime.utcnow(),
            agent_id=data.get("agent_id"),
            config_fingerprint=data.get("config_fingerprint"),
        )

    def history_as_messages(self) -> list[dict[str, Any]]:
        """Return history as message dicts for model adapters."""
        return [msg.to_model_dict() for msg in self.history]

    def attach(self, agent: Agent | None, memory: Any | None = None) -> None:
        """Attach agent and memory handles for run helpers."""
        self._agent = agent
        self._memory = memory

    def _prepare_context(self, context: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(context or {})
        history = self.history_as_messages()
        if payload.get(SESSION_CONTEXT_KEY) and isinstance(
            payload[SESSION_CONTEXT_KEY], list
        ):
            payload[SESSION_CONTEXT_KEY] = history + list(payload[SESSION_CONTEXT_KEY])
        else:
            payload[SESSION_CONTEXT_KEY] = history
        return payload

    def _append_run_messages(
        self,
        input_text: str,
        output_text: str | None,
        run_id: str | None,
    ) -> None:
        self.add_message(Message(role="user", content=input_text, run_id=run_id))
        if output_text is not None:
            self.add_message(
                Message(role="assistant", content=output_text, run_id=run_id)
            )

    async def run_async(
        self,
        input: str | list[Any],
        *,
        context: dict[str, Any] | None = None,
        attachments: list[AttachmentLike] | None = None,
        output_schema: type | None = None,
        timeout: float | None = None,
        cancel_token: CancelToken | None = None,
    ) -> Any:
        """Run the agent within this session asynchronously."""
        if self._agent is None:
            raise RuntimeError("Session is not attached to an Agent")

        input_text = input if isinstance(input, str) else str(input)
        run_context = self._prepare_context(context)

        result = await self._agent.run_async(
            input,
            context=run_context,
            attachments=attachments,
            output_schema=output_schema,
            timeout=timeout,
            cancel_token=cancel_token,
        )

        output_text = result.output_raw or (
            str(result.output) if result.output is not None else None
        )
        self._append_run_messages(input_text, output_text, result.run_id)

        if self._memory is not None:
            await save_session(self._memory, self)

        return result

    def run(
        self,
        input: str | list[Any],
        *,
        context: dict[str, Any] | None = None,
        attachments: list[AttachmentLike] | None = None,
        output_schema: type | None = None,
        timeout: float | None = None,
        cancel_token: CancelToken | None = None,
    ) -> Any:
        """Run the agent within this session synchronously."""
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "session.run() cannot be called from async context. "
                "Use await session.run_async() instead."
            )
        except RuntimeError as e:
            if "cannot be called from async context" in str(e):
                raise

        return asyncio.run(
            self.run_async(
                input,
                context=context,
                attachments=attachments,
                output_schema=output_schema,
                timeout=timeout,
                cancel_token=cancel_token,
            )
        )


async def load_session(memory: Any, session_id: str) -> Session | None:
    """Load a session from a memory adapter."""
    if hasattr(memory, "retrieve_session"):
        return cast(Session | None, await memory.retrieve_session(session_id))

    value = await memory.retrieve(f"{SESSION_PREFIX}{session_id}")
    if value is None:
        return None
    if isinstance(value, Session):
        return value
    if isinstance(value, dict):
        return Session.deserialize(value)
    return None


async def save_session(memory: Any, session: Session) -> Session:
    """Persist a session using the memory adapter."""
    if hasattr(memory, "store_session"):
        return cast(Session, await memory.store_session(session))

    existing = await load_session(memory, session.id)
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
    await memory.store(
        key=f"{SESSION_PREFIX}{session.id}",
        value=session.serialize(),
        metadata={"type": "session"},
    )
    return session


__all__ = [
    "ContentPart",
    "Message",
    "Session",
    "SESSION_PREFIX",
    "generate_session_id",
]
