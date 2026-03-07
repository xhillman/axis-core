from __future__ import annotations

import importlib
import logging
import os
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from axis_core._scalar_parsing import coerce_env_flag
from axis_core.budget import Budget
from axis_core.config import (
    CacheConfig,
    RateLimits,
    RetryPolicy,
    Timeouts,
    ToolPolicy,
    config,
)
from axis_core.protocols.telemetry import BufferMode

logger = logging.getLogger("axis_core.agent")

T = TypeVar("T")
ConfirmationHandler = Callable[[str, dict[str, Any]], bool | Awaitable[bool]]


@dataclass(frozen=True)
class AgentConstruction:
    system: str | None
    persona: str | None
    model: Any
    fallback: list[Any]
    memory: Any
    planner: Any
    budget: Budget
    timeouts: Timeouts
    rate_limits: RateLimits | None
    retry: RetryPolicy | None
    cache: CacheConfig | None
    tool_policy: ToolPolicy | None
    verbose: bool
    confirmation_handler: ConfirmationHandler | None
    checkpoint_enabled: bool
    checkpoint_dir: str
    telemetry_enabled: bool
    telemetry_sinks: list[Any]
    tools: dict[str, Any]


def coerce_dataclass(
    value: dict[str, Any] | T | None,
    cls: type[T],
    arg_name: str,
    *,
    default: T | None = None,
) -> T | None:
    """Coerce a dict, instance, or None into the target type."""
    if value is None:
        return default
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls(**value)
    raise TypeError(
        f"Argument '{arg_name}' must be {cls.__name__} or dict, "
        f"got {type(value).__name__}"
    )


def validate_agent_init_arguments(
    *,
    tools: list[Callable[..., Any]] | None,
    system: str | None,
    telemetry: bool | list[Any],
    verbose: bool,
    confirmation_handler: ConfirmationHandler | None,
    checkpoint: bool,
    checkpoint_dir: str,
) -> None:
    """Validate the runtime types accepted by Agent.__init__."""
    if tools is not None and not isinstance(tools, list):
        raise TypeError(
            f"Argument 'tools' must be a list of callables, "
            f"got {type(tools).__name__}"
        )
    if system is not None and not isinstance(system, str):
        raise TypeError(
            f"Argument 'system' must be str, got {type(system).__name__}"
        )
    if not isinstance(telemetry, (bool, list)):
        raise TypeError(
            f"Argument 'telemetry' must be bool or list, "
            f"got {type(telemetry).__name__}"
        )
    if not isinstance(verbose, bool):
        raise TypeError(
            f"Argument 'verbose' must be bool, got {type(verbose).__name__}"
        )
    if confirmation_handler is not None and not callable(confirmation_handler):
        raise TypeError(
            f"Argument 'confirmation_handler' must be callable or None, "
            f"got {type(confirmation_handler).__name__}"
        )
    if not isinstance(checkpoint, bool):
        raise TypeError(
            f"Argument 'checkpoint' must be bool, got {type(checkpoint).__name__}"
        )
    if not isinstance(checkpoint_dir, str):
        raise TypeError(
            f"Argument 'checkpoint_dir' must be str, got {type(checkpoint_dir).__name__}"
        )


def build_agent_construction(
    *,
    tools: list[Callable[..., Any]] | None,
    system: str | None,
    persona: str | None,
    model: Any,
    fallback: list[Any] | None,
    memory: Any,
    planner: Any,
    budget: dict[str, Any] | Budget | None,
    timeouts: dict[str, Any] | Timeouts | None,
    rate_limits: dict[str, Any] | RateLimits | None,
    retry: dict[str, Any] | RetryPolicy | None,
    cache: dict[str, Any] | CacheConfig | None,
    tool_policy: dict[str, Any] | ToolPolicy | None,
    telemetry: bool | list[Any],
    verbose: bool,
    auth: dict[str, dict[str, Any]] | None,
    confirmation_handler: ConfirmationHandler | None,
    checkpoint: bool,
    checkpoint_dir: str,
    unset: object,
) -> AgentConstruction:
    """Resolve Agent.__init__ inputs into concrete runtime state."""
    validate_agent_init_arguments(
        tools=tools,
        system=system,
        telemetry=telemetry,
        verbose=verbose,
        confirmation_handler=confirmation_handler,
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
    )

    if auth is not None:
        warnings.warn(
            "Argument 'auth' is deprecated and ignored. "
            "Manage credentials inside tools (for example via environment variables).",
            DeprecationWarning,
            stacklevel=3,
        )

    telemetry_enabled, telemetry_sinks = resolve_telemetry(telemetry)

    return AgentConstruction(
        system=system,
        persona=persona,
        model=config.default_model if model is unset else model,
        fallback=fallback or [],
        memory=config.default_memory if memory is unset else memory,
        planner=config.default_planner if planner is unset else planner,
        budget=coerce_dataclass(budget, Budget, "budget", default=Budget()) or Budget(),
        timeouts=(
            coerce_dataclass(timeouts, Timeouts, "timeouts", default=Timeouts())
            or Timeouts()
        ),
        rate_limits=coerce_dataclass(rate_limits, RateLimits, "rate_limits"),
        retry=coerce_dataclass(retry, RetryPolicy, "retry"),
        cache=coerce_dataclass(cache, CacheConfig, "cache"),
        tool_policy=coerce_dataclass(tool_policy, ToolPolicy, "tool_policy"),
        verbose=verbose,
        confirmation_handler=confirmation_handler,
        checkpoint_enabled=checkpoint,
        checkpoint_dir=checkpoint_dir,
        telemetry_enabled=telemetry_enabled,
        telemetry_sinks=telemetry_sinks,
        tools=build_tool_registry(tools),
    )


def build_tool_registry(tools: list[Callable[..., Any]] | None) -> dict[str, Any]:
    """Index tool callables by manifest name or function name."""
    registry: dict[str, Any] = {}
    if not tools:
        return registry

    for tool_callable in tools:
        if hasattr(tool_callable, "_axis_manifest"):
            registry[tool_callable._axis_manifest.name] = tool_callable
        elif hasattr(tool_callable, "__name__"):
            registry[tool_callable.__name__] = tool_callable
        else:
            registry[str(tool_callable)] = tool_callable
    return registry


def resolve_telemetry(telemetry: bool | list[Any]) -> tuple[bool, list[Any]]:
    """Resolve constructor telemetry settings into enabled state and sinks."""
    if isinstance(telemetry, bool):
        return telemetry, resolve_telemetry_sinks() if telemetry else []
    return True, telemetry


def resolve_telemetry_sinks() -> list[Any]:
    """Resolve telemetry sinks from environment variables."""
    sink_type = os.getenv("AXIS_TELEMETRY_SINK", "none").lower()
    redact = coerce_env_flag(os.getenv("AXIS_TELEMETRY_REDACT"), default=True)

    def parse_buffer_mode(raw: str) -> BufferMode:
        normalized = raw.strip().lower()
        for mode in BufferMode:
            if mode.value == normalized:
                return mode
        logger.warning(
            "Unknown AXIS_TELEMETRY_BUFFER_MODE value '%s'. "
            "Using 'batched'.",
            raw,
        )
        return BufferMode.BATCHED

    if sink_type == "none":
        return []

    if sink_type == "console":
        from axis_core.adapters.telemetry.console import ConsoleSink

        compact = coerce_env_flag(os.getenv("AXIS_TELEMETRY_COMPACT"), default=False)
        return [ConsoleSink(compact=compact, redact=redact)]

    if sink_type == "file":
        from axis_core.adapters.telemetry.file import FileSink

        file_path = os.getenv("AXIS_TELEMETRY_FILE", "./axis_trace.jsonl")
        raw_batch_size = os.getenv("AXIS_TELEMETRY_BATCH_SIZE", "100")
        buffer_mode = parse_buffer_mode(
            os.getenv("AXIS_TELEMETRY_BUFFER_MODE", "batched")
        )
        try:
            batch_size = int(raw_batch_size)
        except ValueError:
            logger.warning(
                "Invalid AXIS_TELEMETRY_BATCH_SIZE '%s'. Using 100.",
                raw_batch_size,
            )
            batch_size = 100

        return [
            FileSink(
                path=file_path,
                batch_size=batch_size,
                buffering=buffer_mode,
                redact=redact,
            )
        ]

    if sink_type == "callback":
        from axis_core.adapters.telemetry.callback import CallbackSink

        callback_ref = os.getenv("AXIS_TELEMETRY_CALLBACK", "").strip()
        if not callback_ref:
            logger.warning(
                "AXIS_TELEMETRY_SINK=callback requires AXIS_TELEMETRY_CALLBACK "
                "formatted as 'module:function'. Using no telemetry."
            )
            return []

        if ":" not in callback_ref:
            logger.warning(
                "Invalid AXIS_TELEMETRY_CALLBACK '%s'. Expected 'module:function'. "
                "Using no telemetry.",
                callback_ref,
            )
            return []

        module_path, attr_name = callback_ref.split(":", 1)
        if not module_path or not attr_name:
            logger.warning(
                "Invalid AXIS_TELEMETRY_CALLBACK '%s'. Expected 'module:function'. "
                "Using no telemetry.",
                callback_ref,
            )
            return []

        try:
            module = importlib.import_module(module_path)
            callback = getattr(module, attr_name)
        except (ImportError, AttributeError):
            logger.warning(
                "Unable to load AXIS_TELEMETRY_CALLBACK '%s'. Using no telemetry.",
                callback_ref,
                exc_info=True,
            )
            return []

        if not callable(callback):
            logger.warning(
                "AXIS_TELEMETRY_CALLBACK '%s' is not callable. Using no telemetry.",
                callback_ref,
            )
            return []

        return [CallbackSink(handler=callback, redact=redact)]

    logger.warning(
        "Unknown AXIS_TELEMETRY_SINK value: '%s'. "
        "Supported values: console, file, callback, none. Using no telemetry.",
        sink_type,
    )
    return []
