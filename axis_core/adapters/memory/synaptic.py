"""Synaptic-backed memory adapter.

This adapter wraps ``synaptic_core.axis.SynapticAxisMemory`` and normalizes
results to axis-core memory/session protocol types.

Requires: pip install synaptic-core>=0.1.1
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import datetime
from typing import Any, cast

from synaptic_core.axis import SynapticAxisMemory as _SynapticAxisMemory
from synaptic_core.core import SynapticMemory as _SynapticMemory

from axis_core.protocols.memory import MemoryCapability, MemoryItem
from axis_core.session import Session


class SynapticMemory:
    """Memory adapter backed by ``synaptic-core`` persistence."""

    def __init__(
        self,
        *,
        db_path: str = "synaptic.db",
        embedding_fn: Callable[[str], Sequence[float] | Any] | None = None,
        synaptic_memory: _SynapticMemory | None = None,
        session_deserializer: Callable[[dict[str, Any]], Any] | None = None,
        **synaptic_kwargs: Any,
    ) -> None:
        self._delegate = _SynapticAxisMemory(
            db_path=db_path,
            embedding_fn=embedding_fn,
            synaptic_memory=synaptic_memory,
            session_deserializer=session_deserializer,
            **synaptic_kwargs,
        )

    @property
    def capabilities(self) -> set[MemoryCapability]:
        """Return supported capabilities, normalized to axis-core enum values."""
        normalized: set[MemoryCapability] = set()
        for capability in self._delegate.capabilities:
            value = capability.value if hasattr(capability, "value") else capability
            if not isinstance(value, str):
                continue
            with suppress(ValueError):
                normalized.add(MemoryCapability(value))
        if normalized:
            return normalized
        return {
            MemoryCapability.KEYWORD_SEARCH,
            MemoryCapability.TTL,
            MemoryCapability.NAMESPACES,
        }

    async def store(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
        namespace: str | None = None,
    ) -> None:
        await self._delegate.store(
            key=key,
            value=value,
            metadata=metadata,
            ttl=ttl,
            namespace=namespace,
        )

    async def retrieve(
        self,
        key: str,
        namespace: str | None = None,
    ) -> Any | None:
        return await self._delegate.retrieve(key=key, namespace=namespace)

    async def search(
        self,
        query: str,
        limit: int = 10,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        items = await self._delegate.search(
            query=query,
            limit=limit,
            namespace=namespace,
            filters=filters,
        )
        return [self._normalize_item(item) for item in items]

    async def delete(
        self,
        key: str,
        namespace: str | None = None,
    ) -> bool:
        deleted = await self._delegate.delete(key=key, namespace=namespace)
        return cast(bool, deleted)

    async def clear(
        self,
        namespace: str | None = None,
    ) -> int:
        count = await self._delegate.clear(namespace=namespace)
        return cast(int, count)

    async def store_session(self, session: Session) -> Session:
        stored = await self._delegate.store_session(session)
        return self._normalize_session(stored)

    async def retrieve_session(self, session_id: str) -> Session | None:
        retrieved = await self._delegate.retrieve_session(session_id)
        if retrieved is None:
            return None
        return self._normalize_session(retrieved)

    async def update_session(self, session: Session) -> Session:
        updated = await self._delegate.update_session(session)
        return self._normalize_session(updated)

    @staticmethod
    def _normalize_item(item: Any) -> MemoryItem:
        namespace = item.namespace if hasattr(item, "namespace") else None
        if namespace is not None:
            namespace = str(namespace)
            if not namespace:
                namespace = None

        return MemoryItem(
            key=str(item.key),
            value=item.value,
            metadata=SynapticMemory._normalize_metadata(
                item.metadata if hasattr(item, "metadata") else None
            ),
            score=SynapticMemory._normalize_score(
                item.score if hasattr(item, "score") else None
            ),
            namespace=namespace,
            created_at=SynapticMemory._normalize_datetime(
                item.created_at if hasattr(item, "created_at") else None
            ),
            expires_at=SynapticMemory._normalize_datetime(
                item.expires_at if hasattr(item, "expires_at") else None
            ),
        )

    @staticmethod
    def _normalize_session(value: Any) -> Session:
        if isinstance(value, Session):
            return value
        if isinstance(value, dict):
            return Session.deserialize(value)
        raise TypeError(
            "Synaptic session operations must return axis_core.session.Session "
            "or a serialized session dict."
        )

    @staticmethod
    def _normalize_metadata(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def _normalize_score(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _normalize_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if raw.endswith("Z"):
                raw = f"{raw[:-1]}+00:00"
            with suppress(ValueError):
                return datetime.fromisoformat(raw)
        return None


# Backward compatibility for earlier naming in axis-core.
SynapticAxisMemory = SynapticMemory
