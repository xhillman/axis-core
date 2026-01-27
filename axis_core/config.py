"""Configuration dataclasses for axis-core agents.

This module provides immutable configuration objects for timeouts, retries, rate limiting,
and caching behavior.
"""

from dataclasses import dataclass


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
