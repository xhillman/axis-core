"""Synaptic-backed memory adapter.

This axis-owned adapter wraps ``synaptic_core.api.AsyncSynaptic`` canonical
APIs and normalizes results to axis-core memory/session protocol types.

Requires: pip install axis-core[synaptic]
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any, cast

from synaptic_core.api import AsyncSynaptic as _AsyncSynaptic

from axis_core.errors import ConfigError
from axis_core.protocols.memory import MemoryCapability, MemoryItem
from axis_core.session import Session

_SUPPORTED_SYNAPTIC_VERSION = ">=0.3.0,<0.4.0"
_MIN_SUPPORTED_VERSION = (0, 3, 0)
_MAX_SUPPORTED_VERSION_EXCLUSIVE = (0, 4, 0)
_REQUIRED_CLIENT_ASYNC_METHODS = (
    "set",
    "get",
    "find",
    "remember",
    "recall",
)
_REQUIRED_CLIENT_SYNC_METHODS = ("session",)
_REQUIRED_ENGINE_ASYNC_METHODS = (
    "kv_delete",
    "kv_clear",
    "store_session",
    "retrieve_session",
    "update_session",
)


class SynapticMemory:
    """Memory adapter backed by ``synaptic-core`` persistence."""

    def __init__(
        self,
        *,
        db_path: str = "synaptic.db",
        embedding_fn: Callable[[str], Sequence[float] | Any] | None = None,
        synaptic_client: _AsyncSynaptic | None = None,
        session_deserializer: Callable[[dict[str, Any]], Any] | None = None,
        **synaptic_kwargs: Any,
    ) -> None:
        provider_version = _load_synaptic_core_version()
        _validate_provider_version(provider_version)

        if synaptic_client is None:
            self._client = _AsyncSynaptic(
                db_path=db_path,
                embedding_fn=embedding_fn,
                session_deserializer=session_deserializer,
                **synaptic_kwargs,
            )
        else:
            self._client = synaptic_client

        _validate_provider_api(self._client)
        self._engine = self._client.engine

    @property
    def capabilities(self) -> set[MemoryCapability]:
        """Return supported capabilities, normalized to axis-core enum values."""
        raw_capabilities = getattr(self._engine, "capabilities", None)
        if raw_capabilities is None:
            return {
                MemoryCapability.KEYWORD_SEARCH,
                MemoryCapability.TTL,
                MemoryCapability.NAMESPACES,
            }

        normalized: set[MemoryCapability] = set()
        for capability in raw_capabilities:
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
        await self._client.set(
            key,
            value,
            namespace=namespace,
            metadata=metadata,
            ttl=ttl,
        )

    async def retrieve(
        self,
        key: str,
        namespace: str | None = None,
    ) -> Any | None:
        return await self._client.get(key, namespace=namespace)

    async def search(
        self,
        query: str,
        limit: int = 10,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        if limit <= 0:
            return []

        items = await self._client.find(
            query,
            namespace=namespace,
            limit=limit,
            filters=filters,
        )
        return [self._normalize_item(item) for item in items]

    async def delete(
        self,
        key: str,
        namespace: str | None = None,
    ) -> bool:
        deleted = await self._engine.kv_delete(key=key, namespace=namespace)
        return cast(bool, deleted)

    async def clear(
        self,
        namespace: str | None = None,
    ) -> int:
        count = await self._engine.kv_clear(namespace=namespace)
        return cast(int, count)

    async def store_session(self, session: Session) -> Session:
        stored = await self._engine.store_session(session)
        return self._normalize_session(stored)

    async def retrieve_session(self, session_id: str) -> Session | None:
        retrieved = await self._engine.retrieve_session(session_id)
        if retrieved is None:
            return None
        return self._normalize_session(retrieved)

    async def update_session(self, session: Session) -> Session:
        updated = await self._engine.update_session(session)
        return self._normalize_session(updated)

    @staticmethod
    def _normalize_item(item: Any) -> MemoryItem:
        get_value: Callable[[str], Any]
        if isinstance(item, Mapping):
            get_value = item.get
        else:
            def _attribute_value(name: str) -> Any:
                return getattr(item, name, None)

            get_value = _attribute_value

        namespace = get_value("namespace")
        if namespace is not None:
            namespace = str(namespace)
            if not namespace:
                namespace = None

        return MemoryItem(
            key=str(get_value("key")),
            value=get_value("value"),
            metadata=SynapticMemory._normalize_metadata(get_value("metadata")),
            score=SynapticMemory._normalize_score(get_value("score")),
            namespace=namespace,
            created_at=SynapticMemory._normalize_datetime(get_value("created_at")),
            expires_at=SynapticMemory._normalize_datetime(get_value("expires_at")),
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
        if isinstance(value, Mapping):
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

def _load_synaptic_core_version() -> str:
    try:
        return package_version("synaptic-core")
    except PackageNotFoundError as exc:
        raise ConfigError(
            message=(
                "Memory adapter 'synaptic' requires the synaptic-core package. "
                "Install with: pip install 'axis-core[synaptic]'"
            )
        ) from exc


def _validate_provider_version(provider_version: str) -> None:
    parsed = _parse_semver(provider_version)
    if parsed is None:
        raise ConfigError(
            message=(
                "Could not parse installed synaptic-core version "
                f"'{provider_version}'. Supported range is {_SUPPORTED_SYNAPTIC_VERSION}."
            )
        )

    if parsed < _MIN_SUPPORTED_VERSION or parsed >= _MAX_SUPPORTED_VERSION_EXCLUSIVE:
        raise ConfigError(
            message=(
                "Unsupported synaptic-core version "
                f"'{provider_version}'. Supported range is {_SUPPORTED_SYNAPTIC_VERSION}. "
                "Upgrade with: pip install --upgrade 'synaptic-core>=0.3.0,<0.4.0'"
            )
        )


def _validate_provider_api(client: _AsyncSynaptic) -> None:
    missing_methods = [
        method_name
        for method_name in _REQUIRED_CLIENT_ASYNC_METHODS
        if not callable(getattr(client, method_name, None))
    ]
    missing_methods.extend(
        method_name
        for method_name in _REQUIRED_CLIENT_SYNC_METHODS
        if not callable(getattr(client, method_name, None))
    )

    engine = getattr(client, "engine", None)
    if engine is None:
        missing_methods.append("engine")

    if engine is not None:
        missing_methods.extend(
            method_name
            for method_name in _REQUIRED_ENGINE_ASYNC_METHODS
            if not callable(getattr(engine, method_name, None))
        )

    if missing_methods:
        missing = ", ".join(sorted(set(missing_methods)))
        raise ConfigError(
            message=(
                "synaptic-core provider is missing required methods for axis interop: "
                f"{missing}. Supported synaptic-core range is {_SUPPORTED_SYNAPTIC_VERSION}."
            )
        )

    non_async_methods = [
        method_name
        for method_name in _REQUIRED_CLIENT_ASYNC_METHODS
        if not inspect.iscoroutinefunction(getattr(client, method_name))
    ]
    non_async_methods.extend(
        method_name
        for method_name in _REQUIRED_ENGINE_ASYNC_METHODS
        if not inspect.iscoroutinefunction(getattr(engine, method_name))
    )
    if non_async_methods:
        non_async = ", ".join(sorted(set(non_async_methods)))
        raise ConfigError(
            message=(
                "synaptic-core provider methods must be async for axis interop. "
                f"Non-async methods: {non_async}. "
                f"Supported synaptic-core range is {_SUPPORTED_SYNAPTIC_VERSION}."
            )
        )


def _parse_semver(raw_version: str) -> tuple[int, int, int] | None:
    match = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", raw_version)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch))
