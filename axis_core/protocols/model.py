"""Model adapter protocol and associated dataclasses.

This module defines the ModelAdapter protocol interface for LLM providers, along with
dataclasses for representing model responses, tool calls, and usage statistics.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class UsageStats:
    """Token usage statistics from a model call.

    Attributes:
        input_tokens: Number of tokens in the input/prompt
        output_tokens: Number of tokens in the output/completion
        total_tokens: Total tokens used (input + output)
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int

    @staticmethod
    def from_anthropic(usage: dict[str, Any]) -> "UsageStats":
        """Create UsageStats from Anthropic API usage object.

        Args:
            usage: Usage dict from Anthropic response
                   (e.g., {"input_tokens": 10, "output_tokens": 20})

        Returns:
            UsageStats instance

        Examples:
            >>> usage = {"input_tokens": 100, "output_tokens": 50}
            >>> stats = UsageStats.from_anthropic(usage)
            >>> stats.total_tokens
            150
        """
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return UsageStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    @staticmethod
    def from_openai(usage: dict[str, Any]) -> "UsageStats":
        """Create UsageStats from OpenAI API usage object.

        Args:
            usage: Usage dict from OpenAI response
                   (e.g., {"prompt_tokens": 10, "completion_tokens": 20,
                   "total_tokens": 30})

        Returns:
            UsageStats instance

        Examples:
            >>> usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            >>> stats = UsageStats.from_openai(usage)
            >>> stats.input_tokens
            100
        """
        return UsageStats(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )


@dataclass(frozen=True)
class ToolCall:
    """Represents a tool call from the model.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool to invoke
        arguments: Tool arguments as a dict
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    """Complete response from a model completion.

    Attributes:
        content: Text content of the response
        tool_calls: Tuple of tool calls requested by the model (None if no tools)
        usage: Token usage statistics
        cost_usd: Estimated cost in USD for this completion
    """

    content: str
    tool_calls: tuple[ToolCall, ...] | None
    usage: UsageStats
    cost_usd: float


@dataclass(frozen=True)
class ModelChunk:
    """A streaming chunk from a model stream.

    Attributes:
        content: Incremental text content (empty string if no content)
        tool_call_delta: Incremental tool call data (None if no tool call)
        is_final: Whether this is the final chunk in the stream
    """

    content: str = ""
    tool_call_delta: dict[str, Any] | None = None
    is_final: bool = False


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for LLM model adapters.

    Model adapters provide a uniform interface to different LLM providers (Anthropic, OpenAI,
    Ollama, etc.). They handle API calls, token counting, and cost estimation.

    Implementations must provide:
    - model_id property returning the model identifier
    - complete() for non-streaming completions
    - stream() for streaming completions
    - estimate_tokens() for token counting
    - estimate_cost() for cost calculation
    """

    @property
    def model_id(self) -> str:
        """Model identifier (e.g., 'claude-sonnet-4-20250514', 'gpt-4')."""
        ...

    async def complete(
        self,
        messages: Any,  # list[Message]
        system: str | None = None,
        tools: Any | None = None,  # list[Tool] | None
        temperature: float = 1.0,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Complete a prompt with the model (non-streaming).

        Args:
            messages: List of messages in the conversation
            system: System prompt/instructions
            tools: Available tools for the model to use
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
            max_tokens: Maximum tokens to generate (None = model default)
            stop_sequences: Sequences that stop generation
            metadata: Additional provider-specific metadata

        Returns:
            ModelResponse with content, tool calls, usage, and cost
        """
        ...

    async def stream(
        self,
        messages: Any,  # list[Message]
        system: str | None = None,
        tools: Any | None = None,  # list[Tool] | None
        temperature: float = 1.0,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[ModelChunk]:
        """Stream a completion from the model.

        Args:
            messages: List of messages in the conversation
            system: System prompt/instructions
            tools: Available tools for the model to use
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
            max_tokens: Maximum tokens to generate (None = model default)
            stop_sequences: Sequences that stop generation
            metadata: Additional provider-specific metadata

        Yields:
            ModelChunk instances with incremental content/tool calls
        """
        ...

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        ...

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        ...
