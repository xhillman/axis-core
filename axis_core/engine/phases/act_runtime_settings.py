"""Internal act-phase runtime-setting resolution helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from axis_core._scalar_parsing import (
    coerce_bool,
    coerce_non_negative_int,
    coerce_positive_int,
)
from axis_core.context import RunContext
from axis_core.protocols.planner import PlanStep

logger = logging.getLogger("axis_core.engine")

_DEFAULT_CONTEXT_STRATEGY = "smart"
_DEFAULT_MAX_CYCLE_CONTEXT = 5
_DEFAULT_CONTEXT_WARN_TOKENS = 32_000
_DEFAULT_CONTEXT_BLOCK_TOKENS = 16_000


@dataclass(frozen=True)
class TranscriptRuntimeSettings:
    """Resolved transcript-normalization settings for a model step."""

    strict: bool
    max_tool_result_chars: int | None


@dataclass(frozen=True)
class MessageContextRuntimeSettings:
    """Resolved message-building settings for a model step."""

    strategy: str
    max_cycle_context: int


@dataclass(frozen=True)
class ContextWindowRuntimeSettings:
    """Resolved context-window guard settings for a model step."""

    guard_enabled: bool
    tokens: int | None
    warn_tokens: int
    block_tokens: int
    pruning_enabled: bool


class ActRuntimeSettingsResolver:
    """Resolve act-phase settings with step -> config -> default precedence."""

    def __init__(self, ctx: RunContext, step: PlanStep) -> None:
        self._ctx = ctx
        self._step = step

    def transcript(self) -> TranscriptRuntimeSettings:
        return TranscriptRuntimeSettings(
            strict=self._resolve_bool("transcript_strict", default=False),
            max_tool_result_chars=self._resolve_positive_int("max_tool_result_chars"),
        )

    def message_context(self) -> MessageContextRuntimeSettings:
        return MessageContextRuntimeSettings(
            strategy=self._resolve_context_strategy(),
            max_cycle_context=self._resolve_max_cycle_context(),
        )

    def context_window(self) -> ContextWindowRuntimeSettings:
        warn_tokens = (
            self._resolve_positive_int(
                "context_window_warn_tokens",
                default=_DEFAULT_CONTEXT_WARN_TOKENS,
            )
            or _DEFAULT_CONTEXT_WARN_TOKENS
        )
        block_tokens = (
            self._resolve_positive_int(
                "context_window_block_tokens",
                default=_DEFAULT_CONTEXT_BLOCK_TOKENS,
            )
            or _DEFAULT_CONTEXT_BLOCK_TOKENS
        )
        return ContextWindowRuntimeSettings(
            guard_enabled=self._resolve_bool("context_window_guard_enabled", default=False),
            tokens=self._resolve_positive_int("context_window_tokens"),
            warn_tokens=warn_tokens,
            block_tokens=block_tokens,
            pruning_enabled=self._resolve_bool("context_pruning_enabled", default=False),
        )

    def _resolve_context_strategy(self) -> str:
        source, raw_value = self._lookup("context_strategy")
        if isinstance(raw_value, str):
            strategy = raw_value.strip().lower()
            if strategy in {"smart", "full", "minimal"}:
                return strategy

        if raw_value is not None:
            logger.warning(
                "Invalid %s value for context strategy '%s'; falling back to 'smart'",
                source,
                raw_value,
            )

        return _DEFAULT_CONTEXT_STRATEGY

    def _resolve_max_cycle_context(self) -> int:
        source, raw_value = self._lookup("max_cycle_context")
        parsed = coerce_non_negative_int(raw_value)
        if parsed is not None:
            return parsed

        if raw_value is not None:
            logger.warning(
                "Invalid %s value for max cycle context '%s'; falling back to 5",
                source,
                raw_value,
            )

        return _DEFAULT_MAX_CYCLE_CONTEXT

    def _resolve_bool(self, key: str, *, default: bool) -> bool:
        _, raw_value = self._lookup(key)
        if raw_value is None:
            return default
        return coerce_bool(raw_value, default=default)

    def _resolve_positive_int(self, key: str, *, default: int | None = None) -> int | None:
        _, raw_value = self._lookup(key)
        parsed = coerce_positive_int(raw_value)
        if parsed is not None:
            return parsed
        return default

    def _lookup(self, key: str) -> tuple[str, Any]:
        if key in self._step.payload:
            return "step.payload", self._step.payload.get(key)

        config = getattr(self._ctx, "config", None)
        config_value = getattr(config, key, None)
        if config_value is not None:
            return f"config.{key}", config_value

        return "default", None

__all__ = [
    "ActRuntimeSettingsResolver",
    "ContextWindowRuntimeSettings",
    "MessageContextRuntimeSettings",
    "TranscriptRuntimeSettings",
]
