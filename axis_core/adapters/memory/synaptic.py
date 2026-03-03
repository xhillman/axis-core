"""Synaptic-backed memory adapter.

This axis-owned adapter wraps ``synaptic_core.core.SynapticMemory`` native
KV/session APIs and normalizes results to axis-core memory/session protocol
types.

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

from synaptic_core.core import SynapticMemory as _SynapticMemory

from axis_core.errors import ConfigError
from axis_core.protocols.memory import MemoryCapability, MemoryItem
from axis_core.session import Session

_SUPPORTED_SYNAPTIC_VERSION = ">=0.2.0,<0.3.0"
_MIN_SUPPORTED_VERSION = (0, 2, 0)
_MAX_SUPPORTED_VERSION_EXCLUSIVE = (0, 3, 0)
_REQUIRED_PROVIDER_METHODS = (
    "kv_set",
    "kv_get",
    "kv_search",
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
        synaptic_memory: _SynapticMemory | None = None,
        session_deserializer: Callable[[dict[str, Any]], Any] | None = None,
        **synaptic_kwargs: Any,
    ) -> None:
        provider_version = _load_synaptic_core_version()
        _validate_provider_version(provider_version)

        if synaptic_memory is None:
            self._delegate = _SynapticMemory(
                db_path=db_path,
                embedding_fn=embedding_fn,
                session_deserializer=session_deserializer,
                **synaptic_kwargs,
            )
        else:
            self._delegate = synaptic_memory

        _validate_provider_api(self._delegate)

    @property
    def capabilities(self) -> set[MemoryCapability]:
        """Return supported capabilities, normalized to axis-core enum values."""
        raw_capabilities = getattr(self._delegate, "capabilities", None)
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
        await self._delegate.kv_set(
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
        return await self._delegate.kv_get(key=key, namespace=namespace)

    async def search(
        self,
        query: str,
        limit: int = 10,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        if limit <= 0:
            return []

        items = await self._delegate.kv_search(
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
        deleted = await self._delegate.kv_delete(key=key, namespace=namespace)
        return cast(bool, deleted)

    async def clear(
        self,
        namespace: str | None = None,
    ) -> int:
        count = await self._delegate.kv_clear(namespace=namespace)
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


# Backward compatibility for earlier naming in axis-core.
SynapticAxisMemory = SynapticMemory


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
                "Upgrade with: pip install --upgrade 'synaptic-core>=0.2.0,<0.3.0'"
            )
        )


def _validate_provider_api(delegate: _SynapticMemory) -> None:
    missing_methods = [
        method_name
        for method_name in _REQUIRED_PROVIDER_METHODS
        if not callable(getattr(delegate, method_name, None))
    ]
    if missing_methods:
        missing = ", ".join(sorted(missing_methods))
        raise ConfigError(
            message=(
                "synaptic-core provider is missing required methods for axis interop: "
                f"{missing}. Supported synaptic-core range is {_SUPPORTED_SYNAPTIC_VERSION}."
            )
        )

    non_async_methods = [
        method_name
        for method_name in _REQUIRED_PROVIDER_METHODS
        if not inspect.iscoroutinefunction(getattr(delegate, method_name))
    ]
    if non_async_methods:
        non_async = ", ".join(sorted(non_async_methods))
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
