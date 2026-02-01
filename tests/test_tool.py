"""Tests for axis_core.tool module.

This module tests all components of the tool system including:
- Capability enum
- ToolManifest dataclass
- generate_tool_schema() function
- @tool decorator with dual syntax
- ToolContext with read-only budget access
- ToolCallRecord dataclass
- RateLimiter with token bucket algorithm
"""

import asyncio
import inspect
import time

import pytest

from axis_core.budget import Budget, BudgetState
from axis_core.config import RetryPolicy
from axis_core.tool import (
    Capability,
    RateLimiter,
    ToolCallRecord,
    ToolContext,
    ToolManifest,
    generate_tool_schema,
    tool,
)


class TestCapability:
    """Tests for Capability enum."""

    def test_has_all_capabilities(self):
        """Capability enum should have all 8 required values."""
        assert hasattr(Capability, "NETWORK")
        assert hasattr(Capability, "FILESYSTEM")
        assert hasattr(Capability, "DATABASE")
        assert hasattr(Capability, "EMAIL")
        assert hasattr(Capability, "PAYMENT")
        assert hasattr(Capability, "DESTRUCTIVE")
        assert hasattr(Capability, "SUBPROCESS")
        assert hasattr(Capability, "SECRETS")

    def test_values_are_lowercase_strings(self):
        """Capability values should be lowercase strings matching their names."""
        assert Capability.NETWORK.value == "network"
        assert Capability.FILESYSTEM.value == "filesystem"
        assert Capability.DATABASE.value == "database"
        assert Capability.EMAIL.value == "email"
        assert Capability.PAYMENT.value == "payment"
        assert Capability.DESTRUCTIVE.value == "destructive"
        assert Capability.SUBPROCESS.value == "subprocess"
        assert Capability.SECRETS.value == "secrets"

    def test_enum_count(self):
        """Capability should have exactly 8 members."""
        assert len(list(Capability)) == 8


class TestToolManifest:
    """Tests for ToolManifest dataclass."""

    def test_frozen(self):
        """ToolManifest should be immutable (frozen)."""
        manifest = ToolManifest(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "string"},
            capabilities=(),
        )
        with pytest.raises(AttributeError):
            manifest.name = "new_name"  # type: ignore

    def test_required_fields(self):
        """ToolManifest should have all required fields."""
        manifest = ToolManifest(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "string"},
            capabilities=(),
        )
        assert manifest.name == "test_tool"
        assert manifest.description == "A test tool"
        assert manifest.input_schema == {"type": "object", "properties": {}}
        assert manifest.output_schema == {"type": "string"}
        assert manifest.capabilities == ()

    def test_optional_fields_default_to_none(self):
        """ToolManifest optional fields should default to None."""
        manifest = ToolManifest(
            name="test_tool",
            description="A test tool",
            input_schema={},
            output_schema={},
            capabilities=(),
        )
        assert manifest.cache_ttl is None
        assert manifest.rate_limit is None
        assert manifest.timeout is None
        assert manifest.retry is None

    def test_optional_fields_can_be_set(self):
        """ToolManifest optional fields can be provided."""
        retry_policy = RetryPolicy(max_attempts=3)
        manifest = ToolManifest(
            name="test_tool",
            description="A test tool",
            input_schema={},
            output_schema={},
            capabilities=(Capability.NETWORK,),
            cache_ttl=300,
            rate_limit="10/minute",
            timeout=30.0,
            retry=retry_policy,
        )
        assert manifest.cache_ttl == 300
        assert manifest.rate_limit == "10/minute"
        assert manifest.timeout == 30.0
        assert manifest.retry == retry_policy

    def test_capabilities_is_tuple(self):
        """ToolManifest capabilities should be a tuple for immutability."""
        capabilities = (Capability.NETWORK, Capability.FILESYSTEM)
        manifest = ToolManifest(
            name="test_tool",
            description="A test tool",
            input_schema={},
            output_schema={},
            capabilities=capabilities,
        )
        assert isinstance(manifest.capabilities, tuple)
        assert manifest.capabilities == capabilities


class TestGenerateToolSchema:
    """Tests for generate_tool_schema() function."""

    def test_basic_string_parameter(self):
        """Schema generation should handle str type."""

        def func(name: str) -> str:
            return name

        schema = generate_tool_schema(func)
        assert schema["properties"]["name"]["type"] == "string"
        assert "name" in schema["required"]

    def test_basic_int_parameter(self):
        """Schema generation should handle int type."""

        def func(count: int) -> int:
            return count

        schema = generate_tool_schema(func)
        assert schema["properties"]["count"]["type"] == "integer"
        assert "count" in schema["required"]

    def test_basic_float_parameter(self):
        """Schema generation should handle float type."""

        def func(amount: float) -> float:
            return amount

        schema = generate_tool_schema(func)
        assert schema["properties"]["amount"]["type"] == "number"
        assert "amount" in schema["required"]

    def test_basic_bool_parameter(self):
        """Schema generation should handle bool type."""

        def func(flag: bool) -> bool:
            return flag

        schema = generate_tool_schema(func)
        assert schema["properties"]["flag"]["type"] == "boolean"
        assert "flag" in schema["required"]

    def test_list_parameter(self):
        """Schema generation should handle list type."""

        def func(items: list) -> list:
            return items

        schema = generate_tool_schema(func)
        assert schema["properties"]["items"]["type"] == "array"
        assert "items" in schema["required"]

    def test_dict_parameter(self):
        """Schema generation should handle dict type."""

        def func(data: dict) -> dict:
            return data

        schema = generate_tool_schema(func)
        assert schema["properties"]["data"]["type"] == "object"
        assert "data" in schema["required"]

    def test_optional_parameter_not_required(self):
        """Optional parameters (T | None) should not be in required list."""

        def func(name: str | None = None) -> str:
            return name or "default"

        schema = generate_tool_schema(func)
        assert schema["properties"]["name"]["type"] == "string"
        assert "name" not in schema["required"]

    def test_parameter_with_default_not_required(self):
        """Parameters with defaults should not be required."""

        def func(name: str = "default") -> str:
            return name

        schema = generate_tool_schema(func)
        assert schema["properties"]["name"]["type"] == "string"
        assert "name" not in schema["required"]

    def test_ctx_parameter_skipped(self):
        """ctx parameter should be skipped in schema."""

        def func(ctx: ToolContext, name: str) -> str:
            return name

        schema = generate_tool_schema(func)
        assert "ctx" not in schema["properties"]
        assert "name" in schema["properties"]
        assert "name" in schema["required"]

    def test_no_parameters(self):
        """Function with no parameters should have empty properties."""

        def func() -> str:
            return "result"

        schema = generate_tool_schema(func)
        assert schema["properties"] == {}
        assert schema["required"] == []

    def test_only_ctx_parameter(self):
        """Function with only ctx parameter should have empty properties."""

        def func(ctx: ToolContext) -> str:
            return "result"

        schema = generate_tool_schema(func)
        assert schema["properties"] == {}
        assert schema["required"] == []

    def test_multiple_parameters(self):
        """Schema generation should handle multiple parameters."""

        def func(name: str, count: int, enabled: bool = False) -> str:
            return f"{name}: {count}"

        schema = generate_tool_schema(func)
        assert len(schema["properties"]) == 3
        assert "name" in schema["required"]
        assert "count" in schema["required"]
        assert "enabled" not in schema["required"]

    def test_pydantic_model_calls_model_json_schema(self):
        """Pydantic models should call model_json_schema()."""
        try:
            from pydantic import BaseModel

            class TestModel(BaseModel):
                field: str

            def func(data: TestModel) -> str:
                return data.field

            schema = generate_tool_schema(func)
            # Pydantic models return their own schema
            assert "properties" in schema["properties"]["data"]
        except ImportError:
            pytest.skip("Pydantic not installed")

    def test_unsupported_union_raises_type_error(self):
        """Union types with multiple non-None types should raise TypeError."""

        def func(value: str | int) -> str:
            return str(value)

        with pytest.raises(TypeError, match="Unsupported Union type"):
            generate_tool_schema(func)


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_decorator_without_parentheses(self):
        """@tool should work without parentheses."""

        @tool
        def simple_tool(name: str) -> str:
            """A simple tool."""
            return f"Hello, {name}!"

        assert hasattr(simple_tool, "_axis_tool")
        assert simple_tool._axis_tool is True
        assert hasattr(simple_tool, "_axis_manifest")
        assert hasattr(simple_tool, "__wrapped__")

    def test_decorator_with_parentheses(self):
        """@tool() should work with parentheses."""

        @tool()
        def simple_tool(name: str) -> str:
            """A simple tool."""
            return f"Hello, {name}!"

        assert hasattr(simple_tool, "_axis_tool")
        assert simple_tool._axis_tool is True

    def test_manifest_generated_from_function(self):
        """Decorator should generate manifest from function metadata."""

        @tool
        def greet(name: str) -> str:
            """Greet a person by name."""
            return f"Hello, {name}!"

        manifest = greet._axis_manifest
        assert manifest.name == "greet"
        assert manifest.description == "Greet a person by name."
        assert "name" in manifest.input_schema["properties"]

    def test_override_name(self):
        """Decorator should accept name override."""

        @tool(name="custom_name")
        def original_name(x: int) -> int:
            """A tool."""
            return x

        assert original_name._axis_manifest.name == "custom_name"

    def test_override_description(self):
        """Decorator should accept description override."""

        @tool(description="Custom description")
        def some_tool(x: int) -> int:
            """Original docstring."""
            return x

        assert some_tool._axis_manifest.description == "Custom description"

    def test_override_capabilities(self):
        """Decorator should accept capabilities override."""

        @tool(capabilities=[Capability.NETWORK, Capability.FILESYSTEM])
        def network_tool(url: str) -> str:
            """Fetch URL."""
            return url

        manifest = network_tool._axis_manifest
        assert len(manifest.capabilities) == 2
        assert Capability.NETWORK in manifest.capabilities
        assert Capability.FILESYSTEM in manifest.capabilities

    def test_override_cache_ttl(self):
        """Decorator should accept cache_ttl override."""

        @tool(cache_ttl=600)
        def cached_tool(x: int) -> int:
            """A cached tool."""
            return x

        assert cached_tool._axis_manifest.cache_ttl == 600

    def test_override_rate_limit(self):
        """Decorator should accept rate_limit override."""

        @tool(rate_limit="10/second")
        def limited_tool(x: int) -> int:
            """A rate-limited tool."""
            return x

        assert limited_tool._axis_manifest.rate_limit == "10/second"

    def test_override_timeout(self):
        """Decorator should accept timeout override."""

        @tool(timeout=30.0)
        def slow_tool(x: int) -> int:
            """A slow tool."""
            return x

        assert slow_tool._axis_manifest.timeout == 30.0

    def test_override_retry(self):
        """Decorator should accept retry policy override."""
        retry_policy = RetryPolicy(max_attempts=5)

        @tool(retry=retry_policy)
        def flaky_tool(x: int) -> int:
            """A flaky tool."""
            return x

        assert flaky_tool._axis_manifest.retry == retry_policy

    @pytest.mark.asyncio
    async def test_async_wrapper_for_sync_function(self):
        """Decorator should create async wrapper for sync functions."""

        @tool
        def sync_tool(x: int) -> int:
            """A sync tool."""
            return x * 2

        # Wrapper should be async
        assert inspect.iscoroutinefunction(sync_tool)

        # Should work when awaited
        result = await sync_tool(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_async_wrapper_for_async_function(self):
        """Decorator should preserve async functions."""

        @tool
        async def async_tool(x: int) -> int:
            """An async tool."""
            await asyncio.sleep(0.001)
            return x * 2

        # Should remain async
        assert inspect.iscoroutinefunction(async_tool)

        # Should work when awaited
        result = await async_tool(5)
        assert result == 10

    def test_preserves_function_metadata(self):
        """Decorator should preserve function metadata via functools.wraps."""

        @tool
        def original_func(x: int) -> int:
            """Original docstring."""
            return x

        # __wrapped__ should point to original
        assert hasattr(original_func, "__wrapped__")
        assert original_func.__wrapped__.__doc__ == "Original docstring."

    def test_empty_docstring(self):
        """Empty docstring should result in empty description."""

        @tool
        def no_doc(x: int) -> int:
            return x

        assert no_doc._axis_manifest.description == ""

    @pytest.mark.asyncio
    async def test_tool_with_ctx_parameter(self):
        """Tool with ctx parameter should receive context."""

        @tool
        def tool_with_ctx(ctx: ToolContext, name: str) -> str:
            """A tool that uses context."""
            return f"{ctx.run_id}: {name}"

        # Should work (implementation will be tested when kernel is built)
        assert inspect.iscoroutinefunction(tool_with_ctx)

    def test_multiple_overrides(self):
        """Decorator should handle multiple overrides at once."""
        retry_policy = RetryPolicy(max_attempts=3)

        @tool(
            name="custom",
            description="Custom desc",
            capabilities=[Capability.NETWORK],
            cache_ttl=300,
            rate_limit="5/second",
            timeout=10.0,
            retry=retry_policy,
        )
        def multi_tool(x: int) -> int:
            """Original."""
            return x

        manifest = multi_tool._axis_manifest
        assert manifest.name == "custom"
        assert manifest.description == "Custom desc"
        assert manifest.capabilities == (Capability.NETWORK,)
        assert manifest.cache_ttl == 300
        assert manifest.rate_limit == "5/second"
        assert manifest.timeout == 10.0
        assert manifest.retry == retry_policy


class TestToolContext:
    """Tests for ToolContext with read-only budget access."""

    def test_creation_with_all_fields(self):
        """ToolContext should be created with all fields."""
        budget = Budget()
        budget_state = BudgetState()
        context_dict = {"key": "value"}

        ctx = ToolContext(
            run_id="run_123",
            agent_id="agent_456",
            cycle=5,
            context=context_dict,
            budget=budget,
            budget_state=budget_state,
        )

        assert ctx.run_id == "run_123"
        assert ctx.agent_id == "agent_456"
        assert ctx.cycle == 5
        assert ctx.context == context_dict
        assert ctx.budget == budget
        assert ctx.budget_state == budget_state

    def test_run_id_is_read_only(self):
        """run_id should be read-only after initialization."""
        ctx = ToolContext(
            run_id="run_123",
            agent_id="agent_456",
            cycle=1,
            context={},
            budget=Budget(),
            budget_state=BudgetState(),
        )

        with pytest.raises(AttributeError, match="run_id is read-only"):
            ctx.run_id = "new_id"

    def test_agent_id_is_read_only(self):
        """agent_id should be read-only after initialization."""
        ctx = ToolContext(
            run_id="run_123",
            agent_id="agent_456",
            cycle=1,
            context={},
            budget=Budget(),
            budget_state=BudgetState(),
        )

        with pytest.raises(AttributeError, match="agent_id is read-only"):
            ctx.agent_id = "new_id"

    def test_cycle_is_read_only(self):
        """cycle should be read-only after initialization."""
        ctx = ToolContext(
            run_id="run_123",
            agent_id="agent_456",
            cycle=1,
            context={},
            budget=Budget(),
            budget_state=BudgetState(),
        )

        with pytest.raises(AttributeError, match="cycle is read-only"):
            ctx.cycle = 5

    def test_budget_is_read_only(self):
        """budget should be read-only after initialization."""
        ctx = ToolContext(
            run_id="run_123",
            agent_id="agent_456",
            cycle=1,
            context={},
            budget=Budget(),
            budget_state=BudgetState(),
        )

        with pytest.raises(AttributeError, match="budget is read-only"):
            ctx.budget = Budget(max_cycles=99)

    def test_budget_state_is_read_only(self):
        """budget_state should be read-only after initialization."""
        ctx = ToolContext(
            run_id="run_123",
            agent_id="agent_456",
            cycle=1,
            context={},
            budget=Budget(),
            budget_state=BudgetState(),
        )

        with pytest.raises(AttributeError, match="budget_state is read-only"):
            ctx.budget_state = BudgetState()

    def test_context_dict_is_mutable(self):
        """context dict contents should be mutable."""
        ctx = ToolContext(
            run_id="run_123",
            agent_id="agent_456",
            cycle=1,
            context={"key": "value"},
            budget=Budget(),
            budget_state=BudgetState(),
        )

        # Should be able to modify dict contents
        ctx.context["new_key"] = "new_value"
        assert ctx.context["new_key"] == "new_value"

        ctx.context["key"] = "updated"
        assert ctx.context["key"] == "updated"

    def test_context_field_can_be_reassigned(self):
        """context field itself can be reassigned."""
        ctx = ToolContext(
            run_id="run_123",
            agent_id="agent_456",
            cycle=1,
            context={"old": "data"},
            budget=Budget(),
            budget_state=BudgetState(),
        )

        # Should be able to reassign context field
        new_dict = {"new": "context"}
        ctx.context = new_dict
        assert ctx.context == new_dict


class TestToolCallRecord:
    """Tests for ToolCallRecord dataclass."""

    def test_frozen(self):
        """ToolCallRecord should be immutable (frozen)."""
        record = ToolCallRecord(
            tool_name="test_tool",
            call_id="call_123",
            args={"x": 1},
            result="success",
            error=None,
            cached=False,
            duration_ms=10.5,
            timestamp=123456.0,
        )
        with pytest.raises(AttributeError):
            record.tool_name = "new_name"  # type: ignore

    def test_all_fields(self):
        """ToolCallRecord should have all required fields."""
        record = ToolCallRecord(
            tool_name="test_tool",
            call_id="call_123",
            args={"x": 1, "y": 2},
            result={"status": "ok"},
            error=None,
            cached=False,
            duration_ms=10.5,
            timestamp=123456.789,
        )

        assert record.tool_name == "test_tool"
        assert record.call_id == "call_123"
        assert record.args == {"x": 1, "y": 2}
        assert record.result == {"status": "ok"}
        assert record.error is None
        assert record.cached is False
        assert record.duration_ms == 10.5
        assert record.timestamp == 123456.789

    def test_captures_errors(self):
        """ToolCallRecord should capture errors."""
        record = ToolCallRecord(
            tool_name="failing_tool",
            call_id="call_456",
            args={},
            result=None,
            error="ValueError: Invalid input",
            cached=False,
            duration_ms=5.0,
            timestamp=123456.0,
        )

        assert record.error == "ValueError: Invalid input"
        assert record.result is None

    def test_tracks_cache_hits(self):
        """ToolCallRecord should track cache hits."""
        record = ToolCallRecord(
            tool_name="cached_tool",
            call_id="call_789",
            args={"x": 1},
            result="cached_result",
            error=None,
            cached=True,
            duration_ms=0.1,
            timestamp=123456.0,
        )

        assert record.cached is True


class TestRateLimiter:
    """Tests for RateLimiter with token bucket algorithm."""

    def test_initialization(self):
        """RateLimiter should initialize with count and period."""
        limiter = RateLimiter(count=10, period_seconds=60.0)
        # Internal state is implementation detail, but it should work
        assert limiter is not None

    def test_try_acquire_returns_true_when_tokens_available(self):
        """try_acquire should return True when tokens are available."""
        limiter = RateLimiter(count=5, period_seconds=60.0)

        # Should succeed for first 5 attempts
        for _ in range(5):
            assert limiter.try_acquire() is True

    def test_try_acquire_returns_false_when_exhausted(self):
        """try_acquire should return False when tokens are exhausted."""
        limiter = RateLimiter(count=2, period_seconds=60.0)

        # Exhaust tokens
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True

        # Should fail now
        assert limiter.try_acquire() is False

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_acquire_waits_when_no_tokens(self):
        """async acquire() should wait when no tokens are available."""
        limiter = RateLimiter(count=2, period_seconds=1.0)

        # Exhaust tokens
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is True

        # This should wait for refill
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start

        # Should have waited approximately period_seconds / count for one token
        # With count=2 and period=1.0, tokens refill at 2 per second (0.5s per token)
        assert elapsed >= 0.4  # Allow some tolerance

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_tokens_refill_over_time(self):
        """Tokens should refill based on period."""
        limiter = RateLimiter(count=3, period_seconds=1.0)

        # Exhaust tokens
        for _ in range(3):
            assert limiter.try_acquire() is True

        # No tokens left
        assert limiter.try_acquire() is False

        # Wait for some tokens to refill
        await asyncio.sleep(0.6)

        # Should have at least one token now
        assert limiter.try_acquire() is True

    def test_try_acquire_with_zero_tokens(self):
        """RateLimiter with 0 tokens should always fail try_acquire."""
        limiter = RateLimiter(count=0, period_seconds=1.0)
        assert limiter.try_acquire() is False

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_acquire_with_zero_tokens_waits_for_refill(self):
        """RateLimiter with 0 initial tokens should wait for refill."""
        limiter = RateLimiter(count=1, period_seconds=0.5)

        # Exhaust the one token
        assert limiter.try_acquire() is True

        # Now wait for refill
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start

        # Should have waited for refill
        assert elapsed >= 0.4

    @pytest.mark.asyncio
    async def test_multiple_concurrent_acquires(self):
        """Multiple concurrent acquire() calls should be handled correctly."""
        limiter = RateLimiter(count=2, period_seconds=1.0)

        # Exhaust tokens
        limiter.try_acquire()
        limiter.try_acquire()

        # Multiple concurrent waits
        start = time.time()
        results = await asyncio.gather(
            limiter.acquire(),
            limiter.acquire(),
        )
        elapsed = time.time() - start

        # All should eventually succeed
        assert all(r is None for r in results)
        # Should have taken time to refill
        assert elapsed >= 0.4
