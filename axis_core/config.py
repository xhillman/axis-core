"""Configuration dataclasses for axis-core agents.

This module provides immutable configuration objects for timeouts, retries, rate limiting,
and caching behavior, plus a Config singleton for global defaults.

Architecture Decisions:
- AD-015: deep_merge() for recursive config dictionary merging
- Config resolution order: defaults → env → constructor → runtime
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("axis_core.config")

_DEFAULT_CONTEXT_STRATEGY = "smart"
_DEFAULT_MAX_CYCLE_CONTEXT = 5
_DEFAULT_CONTEXT_WARN_TOKENS = 32_000
_DEFAULT_CONTEXT_BLOCK_TOKENS = 16_000


@dataclass(frozen=True)
class Timeouts:
    """Timeout configuration for each execution phase.

    All timeouts are in seconds. The total timeout is enforced globally across all phases.

    Attributes:
        observe: Timeout for observation phase (gathering context)
        plan: Timeout for planning phase (strategy selection)
        act: Timeout for action phase (tool execution)
        evaluate: Timeout for evaluation phase (result assessment)
        finalize: Timeout for finalization phase (cleanup and results)
        total: Total wall-clock timeout for entire run
    """

    observe: float = 10.0
    plan: float = 30.0
    act: float = 60.0
    evaluate: float = 5.0
    finalize: float = 30.0
    total: float = 300.0


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry behavior on failures.

    Attributes:
        max_attempts: Maximum number of retry attempts (including initial try)
        backoff: Backoff strategy - "exponential", "linear", or "fixed"
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
        jitter: Whether to add random jitter to delays (reduces thundering herd)
        retry_on: List of error types to retry on (None = retry all retriable errors)
    """

    max_attempts: int = 3
    backoff: str = "exponential"
    initial_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    retry_on: list[str] | None = None


@dataclass(frozen=True)
class RateLimits:
    """Rate limiting configuration.

    Rate limits are specified as strings in the format "count/period" where period is
    "second", "minute", or "hour". For example: "60/minute" means 60 requests per minute.

    All fields default to None (no rate limiting). Use parse_rate() to convert rate
    strings to (count, period_seconds) tuples.

    Attributes:
        model_calls: Rate limit for LLM API calls (e.g., "60/minute")
        tool_calls: Rate limit for tool invocations (e.g., "10/second")
        requests: Rate limit for total requests (e.g., "1000/hour")
    """

    model_calls: str | None = None
    tool_calls: str | None = None
    requests: str | None = None

    def parse_rate(self, field_name: str) -> tuple[int, float] | None:
        """Parse a rate limit string into (count, period_seconds) tuple.

        Args:
            field_name: Name of the field to parse ("model_calls", "tool_calls", etc.)

        Returns:
            Tuple of (count, period_seconds) or None if field is None

        Raises:
            ValueError: If rate format is invalid

        Examples:
            >>> limits = RateLimits(model_calls="60/minute")
            >>> limits.parse_rate("model_calls")
            (60, 60.0)

            >>> limits = RateLimits(tool_calls="10/second")
            >>> limits.parse_rate("tool_calls")
            (10, 1.0)
        """
        rate_str = getattr(self, field_name)
        if rate_str is None:
            return None

        # Parse "count/period" format
        if "/" not in rate_str:
            raise ValueError(
                f"Invalid rate format for {field_name}: '{rate_str}'. "
                "Expected format: 'count/period' (e.g., '60/minute')"
            )

        try:
            count_str, period_str = rate_str.split("/", 1)
            count = int(count_str)
        except ValueError:
            raise ValueError(
                f"Invalid rate format for {field_name}: '{rate_str}'. "
                "Count must be an integer."
            )

        # Convert period string to seconds
        period_map = {
            "second": 1.0,
            "minute": 60.0,
            "hour": 3600.0,
        }

        period_seconds = period_map.get(period_str)
        if period_seconds is None:
            raise ValueError(
                f"Invalid period for {field_name}: '{period_str}'. "
                "Must be 'second', 'minute', or 'hour'."
            )

        return (count, period_seconds)


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for caching behavior.

    Caching can significantly reduce latency and cost by reusing previous results.

    Attributes:
        enabled: Whether caching is enabled globally
        model_responses: Whether to cache LLM responses
        tool_results: Whether to cache tool execution results
        ttl: Time-to-live in seconds for cached entries
        backend: Cache backend - "memory", "redis://...", or "sqlite:///..."
        max_size_mb: Maximum cache size in megabytes (for memory backend)
    """

    enabled: bool = True
    model_responses: bool = True
    tool_results: bool = True
    ttl: int = 3600
    backend: str = "memory"
    max_size_mb: int = 100


@dataclass(frozen=True)
class ToolPolicy:
    """Allow/deny tool execution policy using glob patterns.

    Patterns use shell-style wildcards (for example ``db_*`` or ``*_delete``).
    Deny patterns always take precedence over allow patterns.
    """

    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()
    _allow_compiled: tuple[tuple[str, re.Pattern[str]], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _deny_compiled: tuple[tuple[str, re.Pattern[str]], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    @staticmethod
    def _normalize_patterns(
        patterns: tuple[str, ...] | list[str] | None,
        *,
        field_name: str,
    ) -> tuple[str, ...]:
        if patterns is None:
            return ()

        if isinstance(patterns, str):
            raw_items: list[Any] = [patterns]
        else:
            raw_items = list(patterns)

        normalized: list[str] = []
        for item in raw_items:
            if not isinstance(item, str):
                raise TypeError(
                    f"ToolPolicy.{field_name} entries must be strings, "
                    f"got {type(item).__name__}"
                )
            candidate = item.strip()
            if candidate:
                normalized.append(candidate)
        return tuple(normalized)

    @staticmethod
    def _compile_patterns(patterns: tuple[str, ...]) -> tuple[tuple[str, re.Pattern[str]], ...]:
        return tuple((pattern, re.compile(fnmatch.translate(pattern))) for pattern in patterns)

    def __post_init__(self) -> None:
        allow = self._normalize_patterns(self.allow, field_name="allow")
        deny = self._normalize_patterns(self.deny, field_name="deny")
        object.__setattr__(self, "allow", allow)
        object.__setattr__(self, "deny", deny)
        object.__setattr__(self, "_allow_compiled", self._compile_patterns(allow))
        object.__setattr__(self, "_deny_compiled", self._compile_patterns(deny))

    def evaluate(self, tool_name: str) -> tuple[bool, str | None]:
        """Evaluate whether a tool is allowed and return optional denial reason."""
        normalized_name = tool_name.strip()

        for pattern, regex in self._deny_compiled:
            if regex.fullmatch(normalized_name):
                return False, f"matched deny pattern '{pattern}'"

        if not self._allow_compiled:
            return True, None

        for pattern, regex in self._allow_compiled:
            if regex.fullmatch(normalized_name):
                return True, f"matched allow pattern '{pattern}'"

        allow_repr = ", ".join(repr(pattern) for pattern in self.allow)
        return False, f"not matched by allow patterns [{allow_repr}]"


# ===========================================================================
# deep_merge utility (AD-015)
# ===========================================================================


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts, override wins on conflicts (AD-015).

    Recursively merges nested dictionaries. If both base and override contain
    the same key pointing to dicts, those dicts are merged. Otherwise, override
    value replaces base value.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        New merged dictionary (does not mutate inputs)

    Examples:
        >>> deep_merge({"a": 1}, {"b": 2})
        {'a': 1, 'b': 2}

        >>> deep_merge({"a": {"x": 1}}, {"a": {"y": 2}})
        {'a': {'x': 1, 'y': 2}}

        >>> deep_merge({"a": {"x": 1}}, {"a": "replaced"})
        {'a': 'replaced'}
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


# ===========================================================================
# ResolvedConfig dataclass (9.5)
# ===========================================================================


@dataclass(frozen=True)
class ResolvedConfig:
    """Fully resolved configuration for a single agent run.

    Contains all resolved configuration values after applying the resolution order:
    defaults → env → constructor → runtime. This is passed to RunContext.

    Attributes:
        model: Resolved model identifier or adapter
        planner: Resolved planner identifier or adapter
        memory: Resolved memory adapter (optional)
        budget: Budget limits
        timeouts: Phase timeouts
        rate_limits: Rate limiting config (optional)
        retry: Retry policy (optional)
        cache: Cache config (optional)
        context_strategy: Transcript context strategy for model calls
        max_cycle_context: Max historical cycles used for smart context strategy
        transcript_strict: Whether transcript normalization should be strict
        max_tool_result_chars: Optional max chars for tool-result transcript items
        context_window_guard_enabled: Whether pre-model context guard is enabled
        context_window_tokens: Model context window budget in tokens (optional)
        context_window_warn_tokens: Remaining-token warning threshold
        context_window_block_tokens: Remaining-token hard-block threshold
        context_pruning_enabled: Whether tool-result-first pruning is enabled
        tool_policy: Optional allow/deny tool policy
        telemetry_enabled: Whether telemetry is enabled
        verbose: Whether to print events
    """

    model: Any
    planner: Any
    memory: Any | None = None
    budget: Any = None  # Budget instance (imported at runtime to avoid circular imports)
    timeouts: Timeouts | None = None
    rate_limits: RateLimits | None = None
    retry: RetryPolicy | None = None
    cache: CacheConfig | None = None
    context_strategy: str = "smart"
    max_cycle_context: int = 5
    transcript_strict: bool = False
    max_tool_result_chars: int | None = None
    context_window_guard_enabled: bool = False
    context_window_tokens: int | None = None
    context_window_warn_tokens: int = 32000
    context_window_block_tokens: int = 16000
    context_pruning_enabled: bool = False
    tool_policy: ToolPolicy | None = None
    confirmation_handler: (
        Callable[[str, dict[str, Any]], bool | Awaitable[bool]] | None
    ) = None
    telemetry_enabled: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class RuntimeSettings:
    """Runtime-owned settings resolved at run start before execution begins."""

    context_strategy: str = _DEFAULT_CONTEXT_STRATEGY
    max_cycle_context: int = _DEFAULT_MAX_CYCLE_CONTEXT
    transcript_strict: bool = False
    max_tool_result_chars: int | None = None
    context_window_guard_enabled: bool = False
    context_window_tokens: int | None = None
    context_window_warn_tokens: int = _DEFAULT_CONTEXT_WARN_TOKENS
    context_window_block_tokens: int = _DEFAULT_CONTEXT_BLOCK_TOKENS
    context_pruning_enabled: bool = False


def resolve_runtime_settings(
    environ: Mapping[str, str] | None = None,
) -> RuntimeSettings:
    """Resolve runtime-owned env vars into a single run-start settings object."""
    env = os.environ if environ is None else environ

    raw_context_strategy = env.get("AXIS_CONTEXT_STRATEGY")
    context_strategy = coerce_context_strategy(raw_context_strategy)
    if raw_context_strategy is not None and context_strategy is None:
        logger.warning(
            "Invalid AXIS_CONTEXT_STRATEGY='%s'; falling back to '%s'",
            raw_context_strategy,
            _DEFAULT_CONTEXT_STRATEGY,
        )
        context_strategy = _DEFAULT_CONTEXT_STRATEGY
    if context_strategy is None:
        context_strategy = _DEFAULT_CONTEXT_STRATEGY

    raw_max_cycle_context = env.get("AXIS_MAX_CYCLE_CONTEXT")
    max_cycle_context = coerce_non_negative_int(raw_max_cycle_context)
    if raw_max_cycle_context is not None and max_cycle_context is None:
        logger.warning(
            "Invalid AXIS_MAX_CYCLE_CONTEXT='%s'; falling back to %s",
            raw_max_cycle_context,
            _DEFAULT_MAX_CYCLE_CONTEXT,
        )
        max_cycle_context = _DEFAULT_MAX_CYCLE_CONTEXT
    if max_cycle_context is None:
        max_cycle_context = _DEFAULT_MAX_CYCLE_CONTEXT

    return RuntimeSettings(
        context_strategy=context_strategy,
        max_cycle_context=max_cycle_context,
        transcript_strict=coerce_bool(
            env.get("AXIS_TRANSCRIPT_STRICT"),
            default=False,
        ),
        max_tool_result_chars=coerce_positive_int(
            env.get("AXIS_MAX_TOOL_RESULT_CHARS")
        ),
        context_window_guard_enabled=coerce_bool(
            env.get("AXIS_CONTEXT_GUARD_ENABLED"),
            default=False,
        ),
        context_window_tokens=coerce_positive_int(
            env.get("AXIS_CONTEXT_WINDOW_TOKENS")
        ),
        context_window_warn_tokens=(
            coerce_positive_int(env.get("AXIS_CONTEXT_GUARD_WARN_TOKENS"))
            or _DEFAULT_CONTEXT_WARN_TOKENS
        ),
        context_window_block_tokens=(
            coerce_positive_int(env.get("AXIS_CONTEXT_GUARD_BLOCK_TOKENS"))
            or _DEFAULT_CONTEXT_BLOCK_TOKENS
        ),
        context_pruning_enabled=coerce_bool(
            env.get("AXIS_CONTEXT_PRUNE_ENABLED"),
            default=False,
        ),
    )


def resolve_runtime_config(
    *,
    model: Any,
    planner: Any,
    memory: Any,
    budget: Any,
    timeouts: Timeouts,
    rate_limits: RateLimits | None,
    retry: RetryPolicy | None,
    cache: CacheConfig | None,
    tool_policy: ToolPolicy | None,
    confirmation_handler: Callable[[str, dict[str, Any]], bool | Awaitable[bool]] | None,
    telemetry_enabled: bool,
    verbose: bool,
    runtime_settings: RuntimeSettings | None = None,
) -> ResolvedConfig:
    """Build the run config using the config module as the env-resolution boundary."""
    settings = runtime_settings or resolve_runtime_settings()
    return ResolvedConfig(
        model=model,
        planner=planner,
        memory=memory,
        budget=budget,
        timeouts=timeouts,
        rate_limits=rate_limits,
        retry=retry,
        cache=cache,
        context_strategy=settings.context_strategy,
        max_cycle_context=settings.max_cycle_context,
        transcript_strict=settings.transcript_strict,
        max_tool_result_chars=settings.max_tool_result_chars,
        context_window_guard_enabled=settings.context_window_guard_enabled,
        context_window_tokens=settings.context_window_tokens,
        context_window_warn_tokens=settings.context_window_warn_tokens,
        context_window_block_tokens=settings.context_window_block_tokens,
        context_pruning_enabled=settings.context_pruning_enabled,
        tool_policy=tool_policy,
        confirmation_handler=confirmation_handler,
        telemetry_enabled=telemetry_enabled,
        verbose=verbose,
    )


# ===========================================================================
# Config singleton (9.1-9.2, 9.4, 9.6)
# ===========================================================================


class Config:
    """Global configuration singleton.

    Loads defaults from environment variables (via python-dotenv) and provides
    programmatic override with reset() to restore env values.

    Resolution order: defaults → env → constructor → runtime

    Config-owned environment variables:
        - AXIS_DEFAULT_MODEL
        - AXIS_DEFAULT_PLANNER
        - AXIS_DEFAULT_MEMORY
        - ANTHROPIC_API_KEY
        - OPENAI_API_KEY
        - AXIS_TELEMETRY
        - AXIS_VERBOSE
        - AXIS_DEBUG

    Usage:
        from axis_core.config import config

        print(config.default_model)  # "claude-sonnet-4-20250514"

        config.default_model = "gpt-4"  # Override for testing
        config.reset()  # Restore from environment
    """

    def __init__(self) -> None:
        """Initialize config from environment variables."""
        # Try to load .env file (non-fatal if missing)
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass  # python-dotenv not installed

        # Load from environment with defaults
        self._env_default_model = os.getenv(
            "AXIS_DEFAULT_MODEL", "claude-sonnet-4-20250514"
        )
        self._env_default_planner = os.getenv("AXIS_DEFAULT_PLANNER", "auto")
        self._env_default_memory = os.getenv("AXIS_DEFAULT_MEMORY", "ephemeral")
        self._env_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._env_openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self._env_telemetry = os.getenv("AXIS_TELEMETRY", "true").lower() == "true"
        self._env_verbose = os.getenv("AXIS_VERBOSE", "false").lower() == "true"
        self._env_debug = os.getenv("AXIS_DEBUG", "false").lower() == "true"

        # Current values (can be overridden programmatically)
        self.default_model = self._env_default_model
        self.default_planner = self._env_default_planner
        self.default_memory = self._env_default_memory
        self.anthropic_api_key = self._env_anthropic_api_key
        self.openai_api_key = self._env_openai_api_key
        self.telemetry = self._env_telemetry
        self.verbose = self._env_verbose
        self.debug = self._env_debug

    def reset(self) -> None:
        """Reset all values to environment defaults."""
        self.default_model = self._env_default_model
        self.default_planner = self._env_default_planner
        self.default_memory = self._env_default_memory
        self.anthropic_api_key = self._env_anthropic_api_key
        self.openai_api_key = self._env_openai_api_key
        self.telemetry = self._env_telemetry
        self.verbose = self._env_verbose
        self.debug = self._env_debug


def coerce_bool(value: str | bool | None, *, default: bool = False) -> bool:
    """Coerce boolean config values from env-style strings or bools."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def coerce_positive_int(value: str | int | None) -> int | None:
    """Coerce positive integer config values."""
    if isinstance(value, int):
        return value if value > 0 else None
    if value is None:
        return None
    try:
        parsed = int(value.strip())
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def coerce_non_negative_int(value: str | int | None) -> int | None:
    """Coerce non-negative integer config values."""
    if isinstance(value, int):
        return value if value >= 0 else None
    if value is None:
        return None
    try:
        parsed = int(value.strip())
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def coerce_context_strategy(value: str | None) -> str | None:
    """Validate supported transcript context strategies."""
    if value is None:
        return None
    candidate = value.strip().lower()
    if candidate in {"smart", "full", "minimal"}:
        return candidate
    return None


# Global config singleton instance
config = Config()


__all__ = [
    "Timeouts",
    "RetryPolicy",
    "RateLimits",
    "CacheConfig",
    "ToolPolicy",
    "deep_merge",
    "ResolvedConfig",
    "RuntimeSettings",
    "resolve_runtime_settings",
    "resolve_runtime_config",
    "coerce_bool",
    "coerce_positive_int",
    "coerce_non_negative_int",
    "coerce_context_strategy",
    "Config",
    "config",
]
