"""Tests for model protocol and dataclasses."""

from collections.abc import AsyncIterator

import pytest

from axis_core.protocols.model import (
    ModelAdapter,
    ModelChunk,
    ModelResponse,
    ToolCall,
    UsageStats,
)


class TestUsageStats:
    """Tests for UsageStats dataclass."""

    def test_defaults(self):
        """Test UsageStats with explicit values."""
        stats = UsageStats(input_tokens=100, output_tokens=50, total_tokens=150)
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        assert stats.total_tokens == 150

    def test_immutability(self):
        """Test that UsageStats is immutable."""
        stats = UsageStats(input_tokens=100, output_tokens=50, total_tokens=150)
        with pytest.raises(AttributeError):
            stats.input_tokens = 200  # type: ignore

    def test_from_anthropic(self):
        """Test creating UsageStats from Anthropic usage dict."""
        usage = {"input_tokens": 100, "output_tokens": 50}
        stats = UsageStats.from_anthropic(usage)
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        assert stats.total_tokens == 150

    def test_from_anthropic_missing_keys(self):
        """Test from_anthropic with missing keys defaults to 0."""
        usage = {}
        stats = UsageStats.from_anthropic(usage)
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.total_tokens == 0

    def test_from_openai(self):
        """Test creating UsageStats from OpenAI usage dict."""
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        stats = UsageStats.from_openai(usage)
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        assert stats.total_tokens == 150

    def test_from_openai_missing_keys(self):
        """Test from_openai with missing keys defaults to 0."""
        usage = {}
        stats = UsageStats.from_openai(usage)
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.total_tokens == 0


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_with_arguments(self):
        """Test ToolCall with arguments."""
        tool_call = ToolCall(id="tc_123", name="search", arguments={"query": "test"})
        assert tool_call.id == "tc_123"
        assert tool_call.name == "search"
        assert tool_call.arguments == {"query": "test"}

    def test_default_arguments(self):
        """Test ToolCall with default empty arguments."""
        tool_call = ToolCall(id="tc_123", name="search")
        assert tool_call.arguments == {}

    def test_immutability(self):
        """Test that ToolCall is immutable."""
        tool_call = ToolCall(id="tc_123", name="search")
        with pytest.raises(AttributeError):
            tool_call.id = "tc_456"  # type: ignore


class TestModelResponse:
    """Tests for ModelResponse dataclass."""

    def test_with_tool_calls(self):
        """Test ModelResponse with tool calls."""
        usage = UsageStats(input_tokens=100, output_tokens=50, total_tokens=150)
        tool_calls = (ToolCall(id="tc_1", name="search", arguments={"q": "test"}),)
        response = ModelResponse(
            content="Let me search for that",
            tool_calls=tool_calls,
            usage=usage,
            cost_usd=0.01,
        )
        assert response.content == "Let me search for that"
        assert response.tool_calls == tool_calls
        assert response.usage == usage
        assert response.cost_usd == 0.01

    def test_without_tool_calls(self):
        """Test ModelResponse without tool calls."""
        usage = UsageStats(input_tokens=100, output_tokens=50, total_tokens=150)
        response = ModelResponse(
            content="Here's the answer",
            tool_calls=None,
            usage=usage,
            cost_usd=0.01,
        )
        assert response.tool_calls is None

    def test_immutability(self):
        """Test that ModelResponse is immutable."""
        usage = UsageStats(input_tokens=100, output_tokens=50, total_tokens=150)
        response = ModelResponse(
            content="test", tool_calls=None, usage=usage, cost_usd=0.01
        )
        with pytest.raises(AttributeError):
            response.content = "new content"  # type: ignore


class TestModelChunk:
    """Tests for ModelChunk dataclass."""

    def test_defaults(self):
        """Test ModelChunk with default values."""
        chunk = ModelChunk()
        assert chunk.content == ""
        assert chunk.tool_call_delta is None
        assert chunk.is_final is False

    def test_with_content(self):
        """Test ModelChunk with content."""
        chunk = ModelChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.tool_call_delta is None
        assert chunk.is_final is False

    def test_with_tool_call_delta(self):
        """Test ModelChunk with tool call delta."""
        chunk = ModelChunk(tool_call_delta={"id": "tc_1", "name": "search"})
        assert chunk.content == ""
        assert chunk.tool_call_delta == {"id": "tc_1", "name": "search"}

    def test_final_chunk(self):
        """Test ModelChunk marked as final."""
        chunk = ModelChunk(is_final=True)
        assert chunk.is_final is True

    def test_immutability(self):
        """Test that ModelChunk is immutable."""
        chunk = ModelChunk(content="test")
        with pytest.raises(AttributeError):
            chunk.content = "new"  # type: ignore


class TestModelAdapter:
    """Tests for ModelAdapter protocol."""

    @pytest.mark.asyncio
    async def test_protocol_implementation(self):
        """Test that a class implementing ModelAdapter conforms to the protocol."""

        class FakeModelAdapter:
            @property
            def model_id(self) -> str:
                return "fake-model"

            async def complete(self, messages, **kwargs):
                usage = UsageStats(input_tokens=10, output_tokens=5, total_tokens=15)
                return ModelResponse(
                    content="test", tool_calls=None, usage=usage, cost_usd=0.001
                )

            async def stream(self, messages, **kwargs) -> AsyncIterator[ModelChunk]:
                yield ModelChunk(content="test", is_final=True)

            def estimate_tokens(self, text: str) -> int:
                return len(text.split())

            def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
                return (input_tokens * 0.0001) + (output_tokens * 0.0002)

        adapter = FakeModelAdapter()
        assert isinstance(adapter, ModelAdapter)
        assert adapter.model_id == "fake-model"

        # Test complete
        response = await adapter.complete([])
        assert response.content == "test"

        # Test stream
        chunks = [chunk async for chunk in adapter.stream([])]
        assert len(chunks) == 1
        assert chunks[0].content == "test"

        # Test estimate_tokens
        assert adapter.estimate_tokens("hello world") == 2

        # Test estimate_cost
        assert adapter.estimate_cost(100, 50) == pytest.approx(0.02)

    def test_protocol_missing_methods(self):
        """Test that a class missing methods doesn't conform to protocol."""

        class IncompleteAdapter:
            @property
            def model_id(self) -> str:
                return "incomplete"

        adapter = IncompleteAdapter()
        assert not isinstance(adapter, ModelAdapter)
