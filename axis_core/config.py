"""Configuration dataclasses for axis-core agents.

This module provides immutable configuration objects for timeouts, retries, rate limiting,
and caching behavior, plus a Config singleton for global defaults.

Architecture Decisions:
- AD-015: deep_merge() for recursive config dictionary merging
- Config resolution order: defaults → env → constructor → runtime
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


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
    telemetry_enabled: bool = True
    verbose: bool = False


# ===========================================================================
# Config singleton (9.1-9.2, 9.4, 9.6)
# ===========================================================================


class Config:
    """Global configuration singleton.

    Loads defaults from environment variables (via python-dotenv) and provides
    programmatic override with reset() to restore env values.

    Resolution order: defaults → env → constructor → runtime

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


# Global config singleton instance
config = Config()


__all__ = [
    "Timeouts",
    "RetryPolicy",
    "RateLimits",
    "CacheConfig",
    "deep_merge",
    "ResolvedConfig",
    "Config",
    "config",
]
