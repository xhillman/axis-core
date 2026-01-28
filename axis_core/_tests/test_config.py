"""Tests for axis_core.config module."""

import pytest

from axis_core.config import CacheConfig, RateLimits, RetryPolicy, Timeouts


class TestTimeouts:
    """Tests for Timeouts dataclass."""

    def test_default_values(self):
        """Timeouts should have sensible defaults for each phase."""
        timeouts = Timeouts()
        assert timeouts.observe == 10.0
        assert timeouts.plan == 30.0
        assert timeouts.act == 60.0
        assert timeouts.evaluate == 5.0
        assert timeouts.finalize == 30.0
        assert timeouts.total == 300.0

    def test_custom_values(self):
        """Timeouts should accept custom values."""
        timeouts = Timeouts(
            observe=5.0,
            plan=15.0,
            act=30.0,
            evaluate=2.5,
            finalize=10.0,
            total=120.0,
        )
        assert timeouts.observe == 5.0
        assert timeouts.plan == 15.0
        assert timeouts.act == 30.0
        assert timeouts.evaluate == 2.5
        assert timeouts.finalize == 10.0
        assert timeouts.total == 120.0

    def test_frozen(self):
        """Timeouts should be immutable (frozen)."""
        timeouts = Timeouts()
        with pytest.raises(AttributeError):
            timeouts.observe = 20.0  # type: ignore


class TestRetryPolicy:
    """Tests for RetryPolicy dataclass."""

    def test_default_values(self):
        """RetryPolicy should have sensible defaults."""
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.backoff == "exponential"
        assert policy.initial_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.jitter is True
        assert policy.retry_on is None

    def test_custom_values(self):
        """RetryPolicy should accept custom values."""
        policy = RetryPolicy(
            max_attempts=5,
            backoff="linear",
            initial_delay=0.5,
            max_delay=30.0,
            jitter=False,
            retry_on=["timeout", "rate_limit"],
        )
        assert policy.max_attempts == 5
        assert policy.backoff == "linear"
        assert policy.initial_delay == 0.5
        assert policy.max_delay == 30.0
        assert policy.jitter is False
        assert policy.retry_on == ["timeout", "rate_limit"]

    def test_frozen(self):
        """RetryPolicy should be immutable (frozen)."""
        policy = RetryPolicy()
        with pytest.raises(AttributeError):
            policy.max_attempts = 10  # type: ignore

    def test_backoff_strategies(self):
        """RetryPolicy should accept different backoff strategies."""
        exponential = RetryPolicy(backoff="exponential")
        assert exponential.backoff == "exponential"

        linear = RetryPolicy(backoff="linear")
        assert linear.backoff == "linear"

        fixed = RetryPolicy(backoff="fixed")
        assert fixed.backoff == "fixed"


class TestRateLimits:
    """Tests for RateLimits dataclass."""

    def test_default_values(self):
        """RateLimits should default to None (no limits)."""
        limits = RateLimits()
        assert limits.model_calls is None
        assert limits.tool_calls is None
        assert limits.requests is None

    def test_custom_values(self):
        """RateLimits should accept custom rate strings."""
        limits = RateLimits(
            model_calls="60/minute",
            tool_calls="10/second",
            requests="1000/hour",
        )
        assert limits.model_calls == "60/minute"
        assert limits.tool_calls == "10/second"
        assert limits.requests == "1000/hour"

    def test_frozen(self):
        """RateLimits should be immutable (frozen)."""
        limits = RateLimits()
        with pytest.raises(AttributeError):
            limits.model_calls = "100/minute"  # type: ignore

    def test_parse_rate_per_second(self):
        """parse_rate should parse per-second rates."""
        limits = RateLimits(model_calls="10/second")
        count, period = limits.parse_rate("model_calls")
        assert count == 10
        assert period == 1.0

    def test_parse_rate_per_minute(self):
        """parse_rate should parse per-minute rates."""
        limits = RateLimits(model_calls="60/minute")
        count, period = limits.parse_rate("model_calls")
        assert count == 60
        assert period == 60.0

    def test_parse_rate_per_hour(self):
        """parse_rate should parse per-hour rates."""
        limits = RateLimits(requests="1000/hour")
        count, period = limits.parse_rate("requests")
        assert count == 1000
        assert period == 3600.0

    def test_parse_rate_none_field(self):
        """parse_rate should return None if field is None."""
        limits = RateLimits()
        result = limits.parse_rate("model_calls")
        assert result is None

    def test_parse_rate_invalid_format_no_slash(self):
        """parse_rate should raise ValueError for invalid format (no slash)."""
        limits = RateLimits(model_calls="100")
        with pytest.raises(ValueError, match="Invalid rate format"):
            limits.parse_rate("model_calls")

    def test_parse_rate_invalid_format_bad_count(self):
        """parse_rate should raise ValueError for non-integer count."""
        limits = RateLimits(model_calls="abc/second")
        with pytest.raises(ValueError, match="Invalid rate format"):
            limits.parse_rate("model_calls")

    def test_parse_rate_invalid_period(self):
        """parse_rate should raise ValueError for invalid period."""
        limits = RateLimits(model_calls="100/century")
        with pytest.raises(ValueError, match="Invalid period"):
            limits.parse_rate("model_calls")

    def test_parse_rate_all_fields(self):
        """parse_rate should work for all rate limit fields."""
        limits = RateLimits(
            model_calls="60/minute",
            tool_calls="10/second",
            requests="1000/hour",
        )

        model_count, model_period = limits.parse_rate("model_calls")
        assert model_count == 60
        assert model_period == 60.0

        tool_count, tool_period = limits.parse_rate("tool_calls")
        assert tool_count == 10
        assert tool_period == 1.0

        req_count, req_period = limits.parse_rate("requests")
        assert req_count == 1000
        assert req_period == 3600.0


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self):
        """CacheConfig should have sensible defaults."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.model_responses is True
        assert config.tool_results is True
        assert config.ttl == 3600
        assert config.backend == "memory"
        assert config.max_size_mb == 100

    def test_custom_values(self):
        """CacheConfig should accept custom values."""
        config = CacheConfig(
            enabled=False,
            model_responses=False,
            tool_results=True,
            ttl=7200,
            backend="redis://localhost:6379",
            max_size_mb=256,
        )
        assert config.enabled is False
        assert config.model_responses is False
        assert config.tool_results is True
        assert config.ttl == 7200
        assert config.backend == "redis://localhost:6379"
        assert config.max_size_mb == 256

    def test_frozen(self):
        """CacheConfig should be immutable (frozen)."""
        config = CacheConfig()
        with pytest.raises(AttributeError):
            config.enabled = False  # type: ignore

    def test_backend_memory(self):
        """CacheConfig should accept memory backend."""
        config = CacheConfig(backend="memory")
        assert config.backend == "memory"

    def test_backend_redis(self):
        """CacheConfig should accept redis:// URLs."""
        config = CacheConfig(backend="redis://localhost:6379/0")
        assert config.backend == "redis://localhost:6379/0"

    def test_backend_sqlite(self):
        """CacheConfig should accept sqlite:/// URLs."""
        config = CacheConfig(backend="sqlite:///path/to/cache.db")
        assert config.backend == "sqlite:///path/to/cache.db"
